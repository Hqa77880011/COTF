"""Actual HDF5 input/gradient/reload check, without a performance claim."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotf.method import Model
from cotf.public_pouch import read_pouch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    data = read_pouch(args.input, out / "measured.npz", 3, 8)
    assert len(np.unique(data["group"])) == 1 and (data["split"] == 3).all()
    assert np.isfinite(data["x"]).all() and np.isfinite(data["y"]).all()
    # Directly identify selected temperatures by physical x/y, independently of the reader.
    with h5py.File(args.input, "r") as raw:
        for row in range(len(data["x"])):
            group = raw[str(data["condition"][row])]
            xs, ys = np.asarray(group["x"]).ravel(), np.asarray(group["y"]).ravel()
            for point, target in zip(data["query"][row], data["y"][row]):
                xi, yi = abs(xs - point[0]).argmin(), abs(ys - point[1]).argmin()
                assert np.isclose(target, group["temperature"][xi, yi, data["source_frame"][row]])
    torch.set_num_threads(1)
    torch.manual_seed(13)
    config = dict(width=16, heads=4, latents=4, latent_layers=1, input_shape=[8, 3],
                  backbone="senseiver", variant="no_subset", temperature_scale_mode="fixed",
                  temperature_scale=1.0,
                  coordinate_min=data["query"].min((0, 1)).tolist(),
                  coordinate_max=data["query"].max((0, 1)).tolist())
    model = Model(config)
    # Gradient exercise only: these test-labelled records are not a fitted benchmark.
    batch = {k: torch.tensor(data[k][:3]) for k in ["x", "available", "query", "y", "removals"]}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = next(model.parameters()).detach().clone()
    loss = model.objective(batch, model(batch))
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()
    assert not torch.equal(initial, next(model.parameters()))
    model.eval()
    expected = model(batch)["prediction"].detach()
    torch.save(dict(state=model.state_dict(), config=config), out / "functional_model.pt")
    state = torch.load(out / "functional_model.pt", weights_only=True)
    clone = Model(state["config"])
    clone.load_state_dict(state["state"])
    clone.eval()
    error = float((expected - clone(batch)["prediction"]).abs().max().detach())
    assert error < 1e-6
    report = dict(functional_check_only=True, measured_fields=len(data["x"]),
                  queries_per_field=data["y"].shape[1], physical_parents=1,
                  exact_raw_pixel_match=True, finite_gradient=True,
                  objective=float(loss.detach()), checkpoint_max_error=error,
                  grouped_training_or_calibration_claim=False)
    (out / "checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
