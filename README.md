# COTF

Implementation of **Assessing Local Measurement Support in Sparse-Sensor Battery Thermal Field Reconstruction**. It provides the COTF probe, a Senseiver-style reconstructor, distinct reconstruction baselines, frozen-field score controls and grouped interval calibration. No performance table is prefilled.

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

`demo` uses eight sensors and four two-sensor removal groups. Real NASA/CALCE operating profiles do not themselves contain the dense electro-thermal or independent surface-reference fields used by the paper: supply those registered fields through the same interface. Temperature is in degrees Celsius; hotspot distance uses the supplied coordinate units. Set `coordinate_min/max` to the physical rectangular domain for raster baselines. Training-only IQR fixes the common temperature scale; the C-E02 fixed 1-degree protocol is available through `temperature_scale_mode="fixed", temperature_scale=1`. Context is retained in full/null/removal calls.

## Train, choose the floor, calibrate and evaluate

```bash
cotf train --input data/prepared --output runs/cotf --config config.json --backbone senseiver --variant full --epochs 100 --batch-size 16 --seed 13
python -m cotf.protocol select-floor --input data/prepared/validation.npz --model runs/cotf/model.pt --output runs/cotf/selected.pt
cotf calibrate --input data/prepared/calibration.npz --model runs/cotf/selected.pt --output runs/cotf/calibrated.pt
cotf evaluate --input data/prepared/test.npz --model runs/cotf/calibrated.pt --output runs/cotf/evaluation
cotf plot --input runs --output runs/figures
```

Only the recorded validation groups can select the floor from {0.005,0.01,0.02,0.05}; ties favor the smaller value. Calibration rejects train/validation parent overlap. Alpha=0.1 uses the ceil((G+1)*0.9)-th **group-maximum normalized residual**, giving a multiplicative interval radius. At least nine independent calibration parents are needed for a finite quantile. The reconstructor is frozen at calibration/inference. `--device cuda` is optional. Old MLP checkpoints require retraining after this architecture revision.

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

Substitute the other backbone names to run them on the same prepared partitions, seed and optimization budget. Kriging covariance parameters minimize training reconstruction L1; its ordinary-mean solve and predictive variance remain analytic. The main paper does not give neural-baseline uncertainty heads, so deterministic baseline scores use the explicitly selected geometry convention (`--variant baseline`), while kriging and ensemble use their own predictive standard deviations. These choices do not claim to regenerate the historical baseline AURC numbers. COTF and geometry controls can share the same Senseiver checkpoint. Report the saved parameter count and total measured inference time; five members do not imply an equal FLOP budget for arbitrary backbones or removal families.

## Fixed-prediction mechanism controls

```bash
python -m cotf.protocol scores --input data/prepared/test.npz --model runs/cotf/selected.pt --output runs/frozen_scores
cotf train --input data/prepared --output runs/no_subset --config config.json --backbone senseiver --variant no_subset --epochs 100 --batch-size 16 --seed 13
```

The score command asserts identical reconstructed fields across full, geometry, gain-only `1/(1+g)`, fragility-only `v`, random-mask, shuffled-score (seed 37) and evaluation-only residual-oracle controls. It does not retrain the reconstructor. `no_gain/no_removal` are aliases for nondegenerate fragility/gain controls. Training uses independent subsets, registered structured removals and 10% null inputs; `no_subset` removes that exposure. An equally sized deterministic random mask family uses `mask_seed=13`. The residual oracle requires reference temperatures and is not a deployable method. Intervals for a different score require independent recalibration; raw score-control widths are not nominal coverage claims.

## Metrics, outputs and verification

`model.pt` includes configuration, weights and fitting/validation parent IDs; `history.json` records objectives. Evaluation writes prediction/mechanism NPZ arrays, metrics and timing. RMSE, hotspot distance, AURC, retained-80% risk and error-score Spearman are computed per field, averaged within each physical parent, then equally across parents; parent SDs are included. Group-maximum interval coverage requires every query/field of a parent to be covered. Latency includes full/null/removal passes; parameter count includes all ensemble members. Plotting compares the actual saved evaluations.

```bash
python tests/smoke.py
python tests/backbones.py
```

Actual check environment: Python 3.13.5, PyTorch 2.7.1+cu118, CPU/one torch and BLAS thread, NumPy 2.1.3, seed 13/19. The first check runs analytic NPZ preparation, two training epochs, grouped calibration, evaluation, plotting and hand-calculated AURC. The second trains all seven distinct estimators, checks finite loss/parameter updates, learned Fourier-mode gradients, every ensemble member's gradients, exact checkpoint reloads, sensor permutation and missing-input invariance, context dependence, ordinary-kriging constant reproduction and RecFNO mesh changes. It also executes each estimator through the actual file/CLI train/evaluate path and validation-only floor/frozen-score protocols. These are software checks; real reference-field experiments and full training were not rerun.

The architectural mechanisms follow the primary [Senseiver implementation](https://github.com/OrchardLANL/Senseiver), [RecFNO paper](https://arxiv.org/abs/2302.09808), [DeepONet paper](https://doi.org/10.1038/s42256-021-00302-5) and [Voronoi reconstruction paper](https://doi.org/10.1038/s42256-021-00402-2). This repository reimplements those mechanisms and exposes unspecified widths, grid sizes and covariance choices explicitly; it does not vendor or relabel upstream code.


## Independent measured thermal-field input

The [López Bonilla et al. pouch-cell surface-temperature release](https://zenodo.org/records/17938718) provides measured luminescence-thermometry fields in Celsius and physical coordinates in millimetres for one 75 Ah NMC/graphite cell, six charge/discharge conditions at 1/2/3 C. It is an independent real-input source, rather than the paper's 12-module IR reference or 300-parent electro-thermal corpus.

```bash
python -m pip install -e ".[public-data]"
# Download surface_temperature_pouchcell_data_allC.h5 from the linked release.
python -m cotf.public_pouch --input data/public_pouch/surface_temperature_pouchcell_data_allC.h5 --output data/public_pouch/all_fields.npz --query-stride 8
python tests/real_pouch.py --input data/public_pouch/surface_temperature_pouchcell_data_allC.h5 --output runs/real_pouch
```

The native reader resolves the actual temperature axes from x/y/time lengths: this release stores `[Nx,Ny,Nt]` despite an inconsistent axes attribute. It rejects ambiguous dimensions or non-Celsius units, keeps finite measured pixels without image color conversion or invented temperature values, and records exact frame/condition/axis provenance in the adjacent JSON. Eight input sensors are selected measured pixels of that same dense field, not independently measured thermocouples. Four two-pixel removal groups are registered. Electrical records are not retained as model context. All six conditions keep the **same physical cell parent and test-only split**, so this reader cannot fabricate independent training/calibration cells from frames.

Actual download on 2026-11-01: 18,621,296-byte HDF5, 268 original measured fields. The full import produced 268 fields with 271 selected queries each. The bounded check uses three frames per condition (18 fields), independently matches every target to the raw HDF5 pixel, exercises finite Senseiver gradients and a parameter update, and reloads its locally created functional checkpoint exactly (max error 0). This gradient exercise is not a fitted or calibrated benchmark; no grouped coverage is claimed from a single cell.

The original 180 IR/dense-sensor module fields, associated registration/calibration, 300 electro-thermal parent fields and original model configurations remain absent from the local paper/source/history archive, which contains result summaries but no field arrays. NASA/CALCE single-cell operating traces cannot recover those spatial assets. The public measured-input loader closes an actual data-access check without replacing the missing original experimental corpus.
