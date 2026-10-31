"""Variable-set reconstruction and frozen counterfactual support scoring."""
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    """Pool available sensors and retain context in every counterfactual."""
    def __init__(self, config):
        super().__init__()
        self.c = config
        width, context = config.get("width", 32), config.get("context_dimensions", 0)
        self.encoder = nn.Sequential(nn.Linear(4 + context, width), nn.SiLU(), nn.Linear(width, width))
        self.decoder = nn.Sequential(nn.Linear(width + 2 + context, width), nn.SiLU(), nn.Linear(width, 1))

    def prepare(self, data):
        """Fit the robust temperature scale using training fields only."""
        q25, q75 = np.quantile(data["y"], [.25, .75])
        self.c["temperature_scale"] = max(float(q75 - q25), 1e-6)

    def reconstruct(self, x, available, query, context):
        repeated = context[:, None].expand(-1, x.shape[1], -1)
        token = self.encoder(torch.cat([x, available[..., None], repeated], -1))
        summary = (token * available[..., None]).sum(1) / available.sum(1, keepdim=True).clamp_min(1)
        features = torch.cat([query, summary[:, None].expand(-1, query.shape[1], -1),
                              context[:, None].expand(-1, query.shape[1], -1)], -1)
        return self.decoder(features).squeeze(-1)

    def forward(self, batch):
        x, query = batch["x"].float(), batch["query"].float()
        available = batch["available"].float()
        context = batch.get("context", x.new_empty((len(x), 0))).float()
        variant = self.c.get("variant", "full")
        if self.training and variant != "no_subset":
            keep = torch.rand_like(available) > self.c.get("subset_dropout", .25)
            keep[torch.rand(len(x), device=x.device) < .1] = False
            available = available * keep
        mean = self.reconstruct(x, available, query, context)
        prior = self.reconstruct(x, torch.zeros_like(available), query, context)
        removals = batch.get("removals")
        if removals is None:
            removals = torch.eye(x.shape[1], device=x.device)[None].expand(len(x), -1, -1)
        if variant == "random_masks":
            generator = torch.Generator(device=x.device).manual_seed(self.c.get("mask_seed", 13))
            removals = (torch.rand(removals.shape, generator=generator, device=x.device) < .25).float()
        delta = torch.stack([abs(self.reconstruct(x, available * (1 - mask.float()), query, context) - mean)
                             for mask in removals.unbind(1)], 1)
        scale = self.c.get("temperature_scale", 1.)
        gain, fragility = abs(mean - prior) / scale, delta.square().mean(1).sqrt() / scale
        if variant == "no_gain":
            gain = torch.zeros_like(gain)
        if variant == "no_removal":
            fragility = torch.zeros_like(fragility)
        floor = self.c.get("tau", .01)
        risk = (fragility + floor) / (gain + fragility + floor)
        if variant == "baseline":
            distance = torch.cdist(query, x[..., :2]).masked_fill(available[:, None] == 0, float("inf"))
            risk = distance.amin(-1).nan_to_num(posinf=1e6)
        if variant == "shuffle":
            risk = risk.roll(1, dims=1)
        radius = scale * (risk + 1e-8)
        return {"prediction": mean, "lower": mean - radius, "upper": mean + radius,
                "trust_risk": risk, "gain": gain, "fragility": fragility,
                "influence": delta.square() / (delta.square().sum(1, keepdim=True) + 1e-8)}

    def objective(self, batch, prediction):
        return F.l1_loss(prediction["prediction"], batch["y"].float())


def demo(seed, n, config):
    """Analytic fields for software checks, not battery-study observations."""
    rng = np.random.default_rng(seed)
    sensor = rng.uniform(0, 1, (n, 8, 2)).astype("float32")
    query = rng.uniform(0, 1, (n, 64, 2)).astype("float32")
    temperature = lambda z: 20 + 5 * z[..., 0] + 3 * z[..., 1] ** 2
    return {"x": np.concatenate([sensor, temperature(sensor)[..., None]], -1).astype("float32"),
            "available": np.ones((n, 8), np.float32), "query": query,
            "y": temperature(query).astype("float32"), "group": np.arange(n)}


def measure(prediction, data, config):
    """Pool query errors for selective risk and physical groups for coverage."""
    error, score = abs(prediction["prediction"] - data["y"]), prediction["trust_risk"]
    ranked = error.ravel()[np.argsort(score.ravel(), kind="stable")]
    correlation = spearmanr(score.ravel(), error.ravel()).statistic
    estimate_hot, actual_hot = prediction["prediction"].argmax(1), data["y"].argmax(1)
    coordinate = data["query"]
    hotspot = np.linalg.norm(coordinate[np.arange(len(error)), estimate_hot] -
                             coordinate[np.arange(len(error)), actual_hot], axis=-1)
    result = {"aurc": float((np.cumsum(ranked) / np.arange(1, len(ranked) + 1)).mean()),
              "risk_at_80pct": float(ranked[:max(1, int(.8 * len(ranked)))].mean()),
              "error_score_spearman": float(correlation) if np.isfinite(correlation) else None,
              "hotspot_distance": float(hotspot.mean()),
              "mean_evidence_gain": float(prediction["gain"].mean()),
              "mean_removal_fragility": float(prediction["fragility"].mean())}
    covered = (data["y"] >= prediction["lower"]) & (data["y"] <= prediction["upper"])
    if "group" in data:
        result["whole_object_coverage"] = float(np.mean([covered[data["group"] == g].all()
                                                        for g in np.unique(data["group"])]))
    return result
