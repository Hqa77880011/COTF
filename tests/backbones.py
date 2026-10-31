"""Mechanism checks for the distinct reconstruction controls."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotf.method import Model


def main():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = dict(
        width=16,
        context_dimensions=2,
        input_shape=[5, 3],
        heads=4,
        latents=4,
        latent_layers=1,
        grid_size=16,
        modes=4,
        fourier_layers=2,
        ensemble_members=3,
        variant="no_subset",
    )
    x = torch.rand(4, 5, 3)
    query = torch.rand(4, 17, 2)
    context = torch.rand(4, 2)
    batch = dict(
        x=x,
        query=query,
        context=context,
        available=torch.ones(4, 5),
        y=1 + query[..., 0] + context[:, 0, None],
    )
    records = []
    for name in [
        "senseiver",
        "implicit",
        "deeponet",
        "recfno",
        "voronoi_cnn",
        "kriging",
        "ensemble",
    ]:
        network = Model(dict(config, backbone=name))
        network.prepare({"y": batch["y"].numpy()})
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.003)
        network.train()
        losses = []
        old = {key: value.detach().clone() for key, value in network.named_parameters()}
        for _ in range(3):
            prediction = network(batch)
            loss = network.objective(batch, prediction)
            assert torch.isfinite(loss), name
            optimizer.zero_grad()
            loss.backward()
            gradients = {key: value.grad for key, value in network.named_parameters()}
            assert any(
                g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                for g in gradients.values()
            ), name
            if name == "recfno":
                assert network.backbone.fourier[0].weight.grad.abs().sum() > 0
            if name == "ensemble":
                assert all(
                    member.token.weight.grad.abs().sum() > 0
                    for member in network.members
                )
            optimizer.step()
            losses.append(float(loss.detach()))
        assert any(
            not torch.equal(old[key], value)
            for key, value in network.named_parameters()
        ), name
        network.eval()
        predicted = network(batch)["prediction"].detach()
        output = Path("runs/backbones") / name
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state": network.state_dict(), "config": network.c}, output / "model.pt"
        )
        saved = torch.load(output / "model.pt", weights_only=True)
        clone = Model(saved["config"])
        clone.load_state_dict(saved["state"])
        clone.eval()
        assert torch.allclose(predicted, clone(batch)["prediction"], atol=1e-6), name
        if name == "senseiver":
            order = torch.tensor([3, 0, 4, 1, 2])
            permuted = dict(
                batch, x=x[:, order], available=batch["available"][:, order]
            )
            assert torch.allclose(predicted, network(permuted)["prediction"], atol=2e-6)
            missing = dict(batch, available=torch.zeros_like(batch["available"]))
            changed = dict(missing, x=x + 1000)
            assert torch.allclose(
                network(missing)["prediction"],
                network(changed)["prediction"],
                atol=1e-6,
            )
            contextual = dict(missing, context=context + 1)
            assert not torch.allclose(
                network(missing)["prediction"], network(contextual)["prediction"]
            )
        if name == "recfno":
            network.c["grid_size"] = 24
            assert network(batch)["prediction"].shape == batch["y"].shape
        if name == "kriging":
            constant = dict(batch, x=x.clone())
            constant["x"][..., 2] = 7
            assert torch.allclose(
                network(constant)["prediction"],
                torch.full_like(batch["y"], 7),
                atol=1e-5,
            )
        records.append(
            dict(
                backbone=name,
                losses=losses,
                parameters=sum(p.numel() for p in network.parameters()),
                reload_max_error=float(
                    (predicted - clone(batch)["prediction"]).abs().max()
                ),
            )
        )
    # The same file protocol, seed and optimization budget serve every estimator.
    from cotf import cli

    data_path = Path("runs/backbones/data")
    data_path.mkdir(parents=True, exist_ok=True)
    numeric = {k: v.numpy() for k, v in batch.items()}
    for number, split in enumerate(["train", "validation", "calibration", "test"]):
        np.savez(
            data_path / (split + ".npz"), **numeric, group=np.arange(4) + number * 4
        )
    config_path = Path("runs/backbones/config.json")
    config_path.write_text(json.dumps(config))
    for name in [
        "senseiver",
        "implicit",
        "deeponet",
        "recfno",
        "voronoi_cnn",
        "kriging",
        "ensemble",
    ]:
        sys.argv = [
            "cotf",
            "train",
            "--input",
            str(data_path),
            "--output",
            "runs/backbones/cli_" + name,
            "--config",
            str(config_path),
            "--backbone",
            name,
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--variant",
            "no_subset",
        ]
        cli.main()
        sys.argv = [
            "cotf",
            "evaluate",
            "--input",
            str(data_path / "test.npz"),
            "--model",
            "runs/backbones/cli_" + name + "/model.pt",
            "--output",
            "runs/backbones/cli_" + name + "/evaluation",
        ]
        cli.main()
    from cotf import protocol

    sys.argv = [
        "cotf.protocol",
        "select-floor",
        "--input",
        str(data_path / "validation.npz"),
        "--model",
        "runs/backbones/cli_senseiver/model.pt",
        "--output",
        "runs/backbones/selected.pt",
    ]
    protocol.main()
    sys.argv = [
        "cotf.protocol",
        "scores",
        "--input",
        str(data_path / "test.npz"),
        "--model",
        "runs/backbones/selected.pt",
        "--output",
        "runs/backbones/scores",
    ]
    protocol.main()
    print(json.dumps(records))


if __name__ == "__main__":
    main()
