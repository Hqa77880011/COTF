# COTF

**Assessing Local Measurement Support in Sparse-Sensor Battery Thermal Field Reconstruction**. COTF provides support scoring, sparse-field reconstruction, grouped interval calibration and comparison methods.

## Installation

Python >=3.10, PyTorch, NumPy, SciPy, scikit-learn and Matplotlib:

```bash
python -m pip install -e .
```

## Data

NPZ arrays: sensor `x[N,S,3]` (coordinates and temperature), binary `available[N,S]`, query coordinates `query[N,P,2]`, reference `y[N,P]`, physical-parent `group[N]`, optional retained `context[N,C]`, and registered `removals[N,J,S]` (1 removes a sensor). Related layouts, trajectories and fields must share a globally unique parent ID. An explicit `split[N]` uses train/validation/calibration/test codes 0/1/2/3. Otherwise complete ordered parents are split 60/15/10/15. Declare chronological/physical holdouts in the input; random query splitting is not a substitute.

```bash
cotf prepare --input data/reference_fields.npz --output data/prepared
# Analytic fields for functional checks only:
cotf demo --output data/analytic.npz --count 128 --seed 13
```

`demo` generates analytic fields with eight sensors and four two-sensor removal groups. For measured data, supply registered dense temperature fields. Temperature is Celsius; hotspot distance uses supplied coordinate units. Set `coordinate_min/max` for raster baselines. Training-only IQR sets the temperature scale; `temperature_scale_mode="fixed", temperature_scale=1` selects a fixed one-degree scale. Context is retained in full/null/removal calls.

## Train, choose the floor, calibrate and evaluate

```bash
cotf train --input data/prepared --output runs/cotf --config config.json --backbone senseiver --variant full --epochs 100 --batch-size 16 --seed 13
python -m cotf.protocol select-floor --input data/prepared/validation.npz --model runs/cotf/model.pt --output runs/cotf/selected.pt
cotf calibrate --input data/prepared/calibration.npz --model runs/cotf/selected.pt --output runs/cotf/calibrated.pt
cotf evaluate --input data/prepared/test.npz --model runs/cotf/calibrated.pt --output runs/cotf/evaluation
cotf plot --input runs --output runs/figures
```

Only recorded validation groups select the floor from {0.005,0.01,0.02,0.05}, with smaller-value ties. Calibration rejects training/validation parent overlap. Alpha=0.1 uses the ceil((G+1)*0.9)-th group-maximum normalized residual as a multiplicative radius; a finite quantile needs at least nine independent calibration parents. Reconstruction is frozen during calibration/inference. `--device cuda` is optional.

## Reconstruction baselines

These are separate mechanisms, implemented in `cotf/backbones.py`:

| Backbone | Mechanism | Core configuration |
|---|---|---|
| `senseiver` | Fourier coordinate features, sensor-to-latent cross-attention, latent self-attention, query-to-latent decoding | width, heads, latents, latent_layers |
| `implicit` | Shared sensor encoder, permutation-invariant pooling, coordinate MLP | width |
| `deeponet` | Fixed-slot branch and coordinate trunk with learned inner product | width; fixed sensor-slot count/order |
| `recfno` | Voronoi/mask/coordinate embedding, learned complex positive/negative low modes and pointwise paths | grid_size, modes, fourier_layers |
| `voronoi_cnn` | Nearest-sensor image and observation mask with encoder/decoder CNN skips | grid_size, width |
| `kriging` | Ordinary Gaussian kriging with unknown-constant mean correction and predictive variance | kriging_length, kriging_noise; fitted on training fields |
| `ensemble` | Independently initialized/bootstrap-optimized members; mean and variance | ensemble_members=5, ensemble_backbone=senseiver |

```bash
cotf train --input data/prepared --output runs/recfno --config config.json --backbone recfno --variant baseline --epochs 100 --batch-size 16 --seed 13
cotf evaluate --input data/prepared/test.npz --model runs/recfno/model.pt --output runs/recfno/evaluation
cotf train --input data/prepared --output runs/ensemble --config config.json --backbone ensemble --epochs 100 --batch-size 16 --seed 13
cotf evaluate --input data/prepared/test.npz --model runs/ensemble/model.pt --output runs/ensemble/evaluation
```

Use other backbone names with the same prepared partitions and budget. Kriging covariance minimizes training L1; its mean solve and predictive variance are analytic. Deterministic baselines use the geometry score (`--variant baseline`); kriging/ensemble use predictive standard deviations. COTF and geometry scores can share a Senseiver checkpoint. Evaluation records parameter count and total inference time.

## Fixed-prediction mechanism controls

```bash
python -m cotf.protocol scores --input data/prepared/test.npz --model runs/cotf/selected.pt --output runs/frozen_scores
cotf train --input data/prepared --output runs/no_subset --config config.json --backbone senseiver --variant no_subset --epochs 100 --batch-size 16 --seed 13
```

The score command asserts identical reconstructed fields across full, geometry, gain-only `1/(1+g)`, fragility-only `v`, random-mask, shuffled-score (seed 37) and evaluation-only residual-oracle controls. It does not retrain the reconstructor. `no_gain/no_removal` are aliases for nondegenerate fragility/gain controls. Training uses independent subsets, registered structured removals and 10% null inputs; `no_subset` removes that exposure. An equally sized deterministic random mask family uses `mask_seed=13`. The residual oracle requires reference temperatures and is not a deployable method. Intervals for a different score require independent recalibration; raw score-control widths are not nominal coverage claims.

## Metrics and outputs

`model.pt` includes configuration, weights and fitting/validation parent IDs; `history.json` records objectives. Evaluation writes prediction/mechanism NPZ arrays, metrics and timing. RMSE, hotspot distance, AURC, retained-80% risk and error-score Spearman are computed per field, averaged within each physical parent, then equally across parents; parent SDs are included. Group-maximum interval coverage requires every query/field of a parent to be covered. Latency includes full/null/removal passes; parameter count includes all ensemble members. Plotting compares the actual saved evaluations.

```bash
python tests/smoke.py
python tests/backbones.py
```

Sources: [Senseiver](https://github.com/OrchardLANL/Senseiver), [RecFNO](https://arxiv.org/abs/2302.09808), [DeepONet](https://doi.org/10.1038/s42256-021-00302-5), [Voronoi reconstruction](https://doi.org/10.1038/s42256-021-00402-2).


## Independent measured thermal-field input

The [López Bonilla et al. pouch-cell release](https://zenodo.org/records/17938718) provides Celsius surface-temperature fields and millimetre coordinates for one 75 Ah NMC/graphite cell under six charge/discharge conditions. Download `surface_temperature_pouchcell_data_allC.h5` to `data/public_pouch/`, retaining the release README and attribution.

```bash
python -m pip install -e ".[public-data]"
python -c "from pathlib import Path; import urllib.request; p=Path('data/public_pouch'); p.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve('https://zenodo.org/records/17938718/files/surface_temperature_pouchcell_data_allC.h5?download=1', p/'surface_temperature_pouchcell_data_allC.h5')"
python -m cotf.public_pouch --input data/public_pouch/surface_temperature_pouchcell_data_allC.h5 --output data/public_pouch/all_fields.npz --query-stride 8
python tests/real_pouch.py --input data/public_pouch/surface_temperature_pouchcell_data_allC.h5 --output runs/real_pouch
```

The native reader resolves the actual temperature axes from x/y/time lengths: this release stores `[Nx,Ny,Nt]` despite an inconsistent axes attribute. It rejects ambiguous dimensions or non-Celsius units, keeps finite measured pixels without image color conversion or invented temperature values, and records exact frame/condition/axis provenance in the adjacent JSON. Eight input sensors are selected measured pixels of that same dense field, not independently measured thermocouples. Four two-pixel removal groups are registered. Electrical records are not retained as model context. All six conditions keep the **same physical cell parent and test-only split**, so this reader cannot fabricate independent training/calibration cells from frames.
