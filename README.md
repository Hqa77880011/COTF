# COTF

Implementation of **COTF: Assessing Local Measurement Support in Sparse-Sensor Battery Thermal Field Reconstruction**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Validation: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, CPU with one torch thread. `python tests/smoke.py` ran analytic NPZ reading, grouped splitting, two training epochs, finite loss, encoder/decoder gradients and parameter update, grouped calibration, evaluation, plotting, exact checkpoint reload (maximum difference 0), and a hand-calculated AURC example. Analytic data only checks software; it is not evidence for the paper results.

## Data and preprocessing

Generate sensor/query coordinates and the analytic temperature field `20 + 5*x + 3*y**2`.

```bash
cotf demo --output data/generated.npz --count 96 --config config.json --seed 13
cotf prepare --input data/generated.npz --output data/prepared
```

`x[N,S,3]` contains sensor x/y coordinates and measured temperature, `available[N,S]` sensor availability, `query[N,P,2]` field query locations, and `y[N,P]` reference temperature. Split complete thermal runs/battery units, including related charge cycles and sensor layouts. Temperature units and the fixed `temperature_scale` must agree.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
cotf train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
cotf evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline` (geometry score), `no_gain`, `no_removal`, `no_subset`, `random_masks`, `shuffle`. Reuse the prepared data and training seed:

```bash
cotf train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
cotf evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
cotf plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.

## Method correspondence and implementation choices

| Paper component | Entry | Command/configuration |
|---|---|---|
| Variable-set reconstructor and L1 subset training | `cotf/method.py:Model` | `cotf train`; `subset_dropout`, 10% null inputs |
| Retained context and registered failure family | `Model.forward` | optional `context[N,C]`, `removals[N,J,S]` (1 means removed); default leave-one-out |
| Evidence gain / fragility / support deficit | `Model.forward` | training-only IQR temperature scale, fixed `tau` |
| Group-maximum multiplicative intervals | `cotf/cli.py` calibration | commands below; independent calibration groups |
| RMSE, MAE, hotspot distance, AURC, Spearman, point/object coverage, latency | `metrics.py`, `method.measure` | `cotf evaluate` |
| Geometry/random/shuffle/subset controls | `Model.forward` | `--variant` above; plotting uses saved metrics |

```bash
cotf calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
cotf evaluate --input data/prepared/test.npz --model runs/full/calibrated.pt --output runs/full/evaluation
python tests/smoke.py
```

Calibration uses alpha=0.1 and the ceil((G+1)*0.9)-th group-maximum normalized residual. At least nine independent calibration groups are needed for a finite quantile. AURC pools query locations with stable tie ordering; hotspot distance is in the supplied coordinate unit. `tau` must be selected on validation objects before calibration/test. Context is retained for full, null and removal calls. The included backbone is a pooled variable-set MLP; it does not reproduce the paper's Senseiver-style backbone or the kriging, RecFNO, DeepONet and ensemble comparison suite. Those named estimators and real electro-thermal/independent surface-reference data remain outside this repository's verified coverage. Export actual reference fields and group IDs to the stated NPZ schema; `demo` generates an analytic field rather than downloading NASA/CALCE or simulating the paper's electro-thermal model. Current code changes are an implementation revision, separate from the original study's experimental evidence.