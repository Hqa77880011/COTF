"""Variable-set reconstruction and frozen counterfactual support scoring."""

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from .backbones import build


class Model(nn.Module):
    """Pool available sensors and retain context in every counterfactual."""

    def __init__(self, config):
        super().__init__()
        self.c = config
        self.kind = config.get("backbone", "senseiver")
        if self.kind == "ensemble":
            member = dict(config, backbone=config.get("ensemble_backbone", "senseiver"))
            self.members = nn.ModuleList(
                [build(member) for _ in range(config.get("ensemble_members", 5))]
            )
        else:
            self.backbone = build(config)

    def prepare(self, data):
        """Fit the robust temperature scale using training fields only."""
        q25, q75 = np.quantile(data["y"], [0.25, 0.75])
        if self.c.get("temperature_scale_mode", "training_iqr") != "fixed":
            self.c["temperature_scale"] = max(float(q75 - q25), 1e-6)
        self.c["training_mean"] = float(np.mean(data["y"]))

    def reconstruct(self, x, available, query, context):
        if self.kind == "ensemble":
            return torch.stack(
                [member(x, available, query, context) for member in self.members], 1
            ).mean(1)
        return self.backbone(x, available, query, context)

    def forward(self, batch):
        x, query = batch["x"].float(), batch["query"].float()
        available = batch["available"].float()
        context = batch.get("context", x.new_empty((len(x), 0))).float()
        variant = self.c.get("variant", "full")
        if self.training and variant != "no_subset" and self.kind != "kriging":
            keep = torch.rand_like(available) > self.c.get("subset_dropout", 0.25)
            if "removals" in batch:
                family = batch["removals"].float()
                selected = family[
                    torch.arange(len(x), device=x.device),
                    torch.randint(family.shape[1], (len(x),), device=x.device),
                ]
                use_structured = torch.rand(len(x), device=x.device) < 0.5
                keep[use_structured] = selected[use_structured] <= 0
            keep[torch.rand(len(x), device=x.device) < 0.1] = False
            available = available * keep
        if self.kind == "ensemble":
            members = torch.stack(
                [member(x, available, query, context) for member in self.members], 1
            )
            mean = members.mean(1)
            radius = members.std(1, unbiased=False).clamp_min(1e-8)
            return {
                "prediction": mean,
                "member_prediction": members,
                "lower": mean - radius,
                "upper": mean + radius,
                "trust_risk": radius / self.c.get("temperature_scale", 1.0),
                "gain": torch.zeros_like(mean),
                "fragility": torch.zeros_like(mean),
            }
        if self.kind == "kriging":
            mean, variance = self.backbone.predict(x, available, query, context)
            radius = variance.sqrt()
            return {
                "prediction": mean,
                "lower": mean - radius,
                "upper": mean + radius,
                "trust_risk": radius / self.c.get("temperature_scale", 1.0),
                "gain": torch.zeros_like(mean),
                "fragility": torch.zeros_like(mean),
            }
        mean = self.reconstruct(x, available, query, context)
        if self.training:
            return {"prediction": mean}
        prior = self.reconstruct(x, torch.zeros_like(available), query, context)
        removals = batch.get("removals")
        if removals is None:
            removals = torch.eye(x.shape[1], device=x.device)[None].expand(
                len(x), -1, -1
            )
        if variant == "random_masks":
            generator = torch.Generator(device=x.device).manual_seed(
                self.c.get("mask_seed", 13)
            )
            removals = (
                torch.rand(removals.shape, generator=generator, device=x.device) < 0.25
            ).float()
        delta = torch.stack(
            [
                abs(
                    self.reconstruct(x, available * (1 - mask.float()), query, context)
                    - mean
                )
                for mask in removals.unbind(1)
            ],
            1,
        )
        scale = self.c.get("temperature_scale", 1.0)
        gain, fragility = (
            abs(mean - prior) / scale,
            delta.square().mean(1).sqrt() / scale,
        )
        floor = self.c.get("tau", 0.01)
        risk = (fragility + floor) / (gain + fragility + floor)
        if variant in {"gain_only", "no_removal"}:
            risk = 1 / (1 + gain)
        if variant in {"fragility_only", "no_gain"}:
            risk = fragility
        if variant == "baseline":
            distance = torch.cdist(query, x[..., :2]).masked_fill(
                available[:, None] == 0, float("inf")
            )
            risk = distance.amin(-1).nan_to_num(posinf=1e6)
        if variant == "oracle":
            if "y" not in batch:
                raise ValueError(
                    "Residual oracle is evaluation-only and requires reference temperatures"
                )
            risk = abs(mean - batch["y"])
        if variant == "shuffle":
            generator = torch.Generator(device=x.device).manual_seed(
                self.c.get("shuffle_seed", 37)
            )
            risk = risk[
                :, torch.randperm(risk.shape[1], generator=generator, device=x.device)
            ]
        radius = scale * (risk + 1e-8)
        return {
            "prediction": mean,
            "lower": mean - radius,
            "upper": mean + radius,
            "trust_risk": risk,
            "gain": gain,
            "fragility": fragility,
            "influence": delta.square() / (delta.square().sum(1, keepdim=True) + 1e-8),
        }

    def objective(self, batch, prediction):
        if "member_prediction" in prediction:
            losses = []
            for member in prediction["member_prediction"].unbind(1):
                indices = (
                    torch.randint(len(member), (len(member),), device=member.device)
                    if self.training
                    else torch.arange(len(member), device=member.device)
                )
                losses.append(F.l1_loss(member[indices], batch["y"][indices].float()))
            return torch.stack(losses).mean()
        return F.l1_loss(prediction["prediction"], batch["y"].float())


def demo(seed, n, config):
    """Analytic fields for software checks, not battery-study observations."""
    rng = np.random.default_rng(seed)
    sensor = rng.uniform(0, 1, (n, 8, 2)).astype("float32")
    query = rng.uniform(0, 1, (n, 64, 2)).astype("float32")
    temperature = lambda z: 20 + 5 * z[..., 0] + 3 * z[..., 1] ** 2
    return {
        "x": np.concatenate([sensor, temperature(sensor)[..., None]], -1).astype(
            "float32"
        ),
        "available": np.ones((n, 8), np.float32),
        "query": query,
        "y": temperature(query).astype("float32"),
        "group": np.arange(n),
        "removals": np.repeat(
            np.repeat(np.eye(4, dtype=np.float32), 2, axis=1)[None], n, axis=0
        ),
    }


def measure(prediction, data, config):
    """Compute each field endpoint, average within parents, then across parents."""
    truth = data["y"]
    error = abs(prediction["prediction"] - truth)
    risk = prediction["trust_risk"]
    rows = []
    for i in range(len(truth)):
        ranked = error[i, np.argsort(risk[i], kind="stable")]
        correlation = spearmanr(risk[i], error[i]).statistic
        actual = int(truth[i].argmax())
        estimated = int(prediction["prediction"][i].argmax())
        rows.append(
            dict(
                aurc=float((np.cumsum(ranked) / np.arange(1, len(ranked) + 1)).mean()),
                rmse=float(np.sqrt(np.mean(error[i] ** 2))),
                mae=float(error[i].mean()),
                hotspot_distance=float(
                    np.linalg.norm(
                        data["query"][i, actual] - data["query"][i, estimated]
                    )
                ),
                error_score_spearman=float(correlation)
                if np.isfinite(correlation)
                else np.nan,
                risk_at_80pct=float(ranked[: max(1, int(0.8 * len(ranked)))].mean()),
            )
        )
    group = data.get("group", np.arange(len(truth)))
    result = {"parent_count": len(np.unique(group)), "field_count": len(truth)}
    for key in rows[0]:
        values = np.array([row[key] for row in rows])
        parents = np.array(
            [
                np.mean(values[(group == g) & np.isfinite(values)])
                if np.any((group == g) & np.isfinite(values))
                else np.nan
                for g in np.unique(group)
            ]
        )
        parents = parents[np.isfinite(parents)]
        result[key] = float(parents.mean()) if len(parents) else None
        result[key + "_parent_sd"] = (
            float(parents.std(ddof=1)) if len(parents) > 1 else None
        )
    result.update(
        mean_evidence_gain=float(prediction["gain"].mean()),
        mean_removal_fragility=float(prediction["fragility"].mean()),
    )
    covered = (truth >= prediction["lower"]) & (truth <= prediction["upper"])
    result["whole_object_coverage"] = float(
        np.mean([covered[group == g].all() for g in np.unique(group)])
    )
    return result
