"""Validation floor selection and fixed-prediction score controls."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .cli import read, tensor_batch
from .method import Model, measure
from .metrics import measure as field_metrics


def predict(network, data, batch_size=16):
    """Evaluate without retaining gradients and preserve sample order."""
    output = {}
    with torch.no_grad():
        for begin in range(0, len(data["x"]), batch_size):
            batch = tensor_batch(
                data, np.arange(begin, min(begin + batch_size, len(data["x"]))), "cpu"
            )
            for key, value in network(batch).items():
                output.setdefault(key, []).append(value.cpu().numpy())
    return {key: np.concatenate(values) for key, values in output.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["select-floor", "scores"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    network = Model(copy.deepcopy(state["config"]))
    network.load_state_dict(state["state"])
    network.eval()
    if network.kind in {'kriging', 'ensemble'}:
        raise ValueError('COTF floor/score controls require a single frozen neural reconstructor')
    data = read(args.input)
    if args.command == "select-floor":
        if (
            "group" not in data
            or np.intersect1d(data["group"], state["training_groups"]).size
        ):
            raise ValueError(
                "Floor selection requires validation objects disjoint from training"
            )
        expected = state.get("validation_groups", [])
        if not expected or not np.isin(np.unique(data["group"]), expected).all():
            raise ValueError(
                "Use only the recorded checkpoint-validation groups, never calibration/test groups"
            )
        records = []
        for floor in [0.005, 0.01, 0.02, 0.05]:
            network.c.update(variant="full", tau=floor)
            prediction = predict(network, data, args.batch_size)
            records.append(
                dict(tau=floor, aurc=measure(prediction, data, network.c)["aurc"])
            )
        selected = min(records, key=lambda r: (r["aurc"], r["tau"]))
        state["config"]["tau"] = selected["tau"]
        state["floor_validation"] = records
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, output)
        print(json.dumps(selected))
    else:
        output = Path(args.output)
        base = None
        for variant in [
            "full",
            "baseline",
            "gain_only",
            "fragility_only",
            "random_masks",
            "shuffle",
            "oracle",
        ]:
            network.c["variant"] = variant
            prediction = predict(network, data, args.batch_size)
            if base is None:
                base = prediction["prediction"]
            elif not np.allclose(base, prediction["prediction"], atol=1e-6):
                raise AssertionError(
                    "A frozen-field score control changed the reconstructed field"
                )
            metrics = field_metrics(prediction, data)
            metrics.update(measure(prediction, data, network.c))
            target = output / variant
            target.mkdir(parents=True, exist_ok=True)
            (target / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False)
            )
            np.savez_compressed(target / "predictions.npz", **prediction)


if __name__ == "__main__":
    main()
