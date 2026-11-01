"""Read the independent measured pouch-cell fields from Zenodo 17938718.

This is one physical cell. All conditions share one test-only parent; the reader
does not invent independent training/calibration modules from its frames.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def read_pouch(path, output, frames_per_condition=None, query_stride=8):
    """Import measured Celsius fields with millimetre coordinates and provenance."""
    import h5py

    if query_stride < 1 or (frames_per_condition is not None and frames_per_condition < 1):
        raise ValueError("Frame count and query stride must be positive")
    rows, sources = [], []
    with h5py.File(path, "r") as handle:
        for name in sorted(handle):
            group = handle[name]
            x = np.asarray(group["x"]).ravel()
            y = np.asarray(group["y"]).ravel()
            time = np.asarray(group["time"]).ravel()
            raw = np.asarray(group["temperature"])
            # The release's axes attribute conflicts with the stored dimensions.
            # Resolve from actual coordinate/time lengths, never transpose by guess.
            orders = [order for order in itertools.permutations(range(3))
                      if tuple(raw.shape[i] for i in order) == (len(time), len(y), len(x))]
            if len(orders) != 1:
                raise ValueError(f"{name}: cannot uniquely resolve temperature axes")
            temperature = raw.transpose(orders[0])
            if (np.diff(x) <= 0).any() or (np.diff(y) <= 0).any():
                raise ValueError(f"{name}: require increasing physical coordinates")
            if str(group["temperature"].attrs.get("units", "")) != "degC":
                raise ValueError(f"{name}: require physical Celsius temperatures")
            indices = np.arange(len(time))
            if frames_per_condition is not None:
                indices = np.unique(np.linspace(0, len(time) - 1,
                                                min(len(time), frames_per_condition)).astype(int))
            xx, yy = np.meshgrid(x, y)
            coordinates = np.column_stack([xx.ravel(), yy.ravel()])
            finite = np.isfinite(temperature[indices]).all(0).ravel()
            finite_ids = np.flatnonzero(finite)
            if len(finite_ids) < 8:
                raise ValueError(f"{name}: fewer than eight common measured pixels")
            # Eight actual measured pixels spread over the available image region.
            sensors = finite_ids[np.linspace(0, len(finite_ids) - 1, 8).astype(int)]
            query = np.arange(len(coordinates)).reshape(len(y), len(x))[::query_stride, ::query_stride].ravel()
            query = query[finite[query]]
            if not len(query):
                raise ValueError(f"{name}: no finite queries at requested stride")
            for index in indices:
                field = temperature[index].ravel()
                rows.append(dict(
                    x=np.column_stack([coordinates[sensors], field[sensors]]),
                    query=coordinates[query], y=field[query],
                    condition=name, source_frame=int(index), time_s=float(time[index]),
                ))
            sources.append(dict(condition=name, raw_shape=list(raw.shape),
                                stored_to_tyx_axes=list(orders[0]),
                                original_frames=len(time), selected_frames=len(indices),
                                finite_pixels=int(finite.sum()), queries=len(query)))
    # Coordinate extents differ among conditions; keep common fixed query count
    # by deterministic subsampling of measured pixels, without synthesizing values.
    count = min(len(row["y"]) for row in rows)
    for row in rows:
        ids = np.linspace(0, len(row["y"]) - 1, count).astype(int)
        row["query"], row["y"] = row["query"][ids], row["y"][ids]
    removals = np.zeros((4, 8), np.float32)
    for index in range(4):
        removals[index, 2 * index:2 * index + 2] = 1
    result = dict(
        x=np.stack([r["x"] for r in rows]).astype(np.float32),
        available=np.ones((len(rows), 8), np.float32),
        query=np.stack([r["query"] for r in rows]).astype(np.float32),
        y=np.stack([r["y"] for r in rows]).astype(np.float32),
        group=np.repeat("zenodo17938718_physical_cell1", len(rows)),
        split=np.full(len(rows), 3, np.int64),
        removals=np.repeat(removals[None], len(rows), axis=0),
        condition=np.array([r["condition"] for r in rows]),
        source_frame=np.array([r["source_frame"] for r in rows]),
        time_s=np.array([r["time_s"] for r in rows]),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    output.with_suffix(".json").write_text(json.dumps(dict(
        source="https://zenodo.org/records/17938718", physical_cells=1,
        external_measured_input_only=True, coordinate_units="mm",
        temperature_units="degC", group_convention="all six conditions are one physical parent",
        role="test only; not enough independent cells to fit/calibrate a grouped benchmark",
        sensor_convention="eight pixels of the same dense luminescence measurement, not independent thermocouples",
        context="electrical records are not retained as reconstructor context",
        conditions=sources, fields=len(rows), queries_per_field=count,
    ), indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-per-condition", type=int)
    parser.add_argument("--query-stride", type=int, default=8)
    args = parser.parse_args()
    data = read_pouch(args.input, args.output, args.frames_per_condition, args.query_stride)
    print(json.dumps(dict(fields=len(data["x"]), queries=data["y"].shape[1], physical_cells=1)))


if __name__ == "__main__":
    main()
