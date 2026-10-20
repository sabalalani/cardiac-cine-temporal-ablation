# Four-model reproducibility record

This document describes the four completed configurations reported in the
manuscript. It intentionally excludes incomplete exploratory jobs and the
currently running shuffle-matched E2-Static sensitivity experiment.

## Reported configurations

| Name | Input and architecture | Parameters | Training implementation | Saved result root |
|---|---|---:|---|---|
| E1 | ED or ES volume; three-level Wolny 3D U-Net, 32/64/128 channels | 983,254 | Wolny `pytorch-3dunet` native trainer | `exp1_wolny_native/` |
| E2-Full | ED-aligned 30-frame cine; 3D encoder/decoder with bottleneck bidirectional 3D ConvLSTM | 4,027,332 | `train_exp2_3dconvlstm.py` | `exp2_3dconvlstm/` |
| E2-Static | Same trainable E2 architecture; separate 30-copy ED and 30-copy ES sequences | 4,027,332 | `train_exp2_3dconvlstm.py --static_input` | `exp2_static_convlstm/` |
| E3 | ED-aligned 30-frame cine; factorized spatial 3D plus temporal 1D convolutions | 1,517,124 | `train_exp3_factorized.py` | `exp3_factorized/` |

Patient data, checkpoints, and probability arrays are not distributed through
Git. Their expected paths are documented so authorized dataset users can run
the same code locally.

## Cohorts and fixed partitioning

- ACDC development: patients 001--100.
- Fixed diagnosis-stratified five-fold split:
  `configs/acdc_5fold_splits_seed42.json`.
- Every fold contains 80 training and 20 validation patients; all frames,
  phases, masks, and patches from one patient remain in one partition.
- ACDC test cohort: patients 101--150, never included in model fitting or
  validation-checkpoint selection.
- M&M external validation: 320 labeled subjects in the locally obtained open
  distribution (150 Training/Labeled, 34 Validation, and 136 Testing). The 25
  Training/Unlabeled subjects were excluded.

All four models use the same patient folds and the same spatially standardized
ACDC source volumes. M&M is evaluation-only; no M&M image or annotation is used
for model fitting, fine-tuning, or checkpoint selection.

## Preprocessing and supervision

- Target spacing: 1.5 x 1.5 x 10 mm.
- Target spatial shape: 192 x 192 x 16.
- Temporal length: 30 frames, circularly ED-aligned.
- Exact annotated ED and ES arrays replace their corresponding interpolated
  temporal frames; segmentation masks are never temporally interpolated.
- Labels: 0 background, 1 RV, 2 MYO, 3 LV. M&M labels are remapped from
  1 LV, 2 MYO, 3 RV to this order.
- Training patch: 8 x 128 x 128; stride: 4 x 64 x 64.
- Supervision is restricted to ED and ES.
- Basic augmentation: independent 0.5-probability flips along each spatial
  axis and a uniformly selected in-plane rotation of 0, 90, 180, or 270
  degrees. Temporal transformations are synchronized across all frames and
  the ED/ES masks.

Canonical code:

- `preprocess_acdc_spatial.py`
- `prepare_acdc_temporal.py`
- `prepare_exp1_wolny_native.py`
- `preprocess_mms_external.py`
- `qa_acdc_temporal.py`
- `audit_mms_provenance_alignment.py`

## Shared optimization policy

- Loss: four-class weighted cross entropy, weights `[0.05, 1.3, 2.0, 1.2]`.
- Optimizer: Adam, learning rate 2e-4, weight decay 1e-5.
- Scheduler: `ReduceLROnPlateau`, mode `max`, factor 0.2, patience 8,
  monitoring validation foreground MeanIoU.
- Maximum: 150 epochs; temporal runs stop after the learning rate becomes
  lower than 1e-6.
- Selected checkpoint: strict highest validation foreground MeanIoU.
- Seed: 42 for the split, Python, NumPy, PyTorch CPU, and CUDA.
- E1 batch size: 2. E2-Full, E2-Static, and E3 batch size: 1.
- E2/E3 use automatic mixed precision and gradient clipping at norm 5.

## E2 ConvLSTM specification

Every encoder/decoder block contains two bias-free 3 x 3 x 3 convolutions,
each followed by eight-group normalization and ReLU. Feature widths are 32,
64, and 128. Two 2 x 2 x 2 max-pooling operations produce the bottleneck.

The bidirectional ConvLSTM receives 128-channel bottleneck features. Each
direction has 64 hidden channels and uses a 3 x 3 x 3 convolution to compute
the four LSTM gates. Forward and backward 64-channel states are concatenated
and fused to 128 channels by a 1 x 1 x 1 convolution. Only the ED and ES
bottleneck states are decoded. Their corresponding spatial skip maps are used
by two 2 x 2 x 2 transposed-convolution stages. The output contains four-class
ED and ES logits.

E2-Static retains these trainable layers and parameters. For each source patch,
it builds one 30-frame sequence by repeating ED and another by repeating ES.
The two outputs are included in one cross-entropy calculation and one optimizer
update. It therefore removes both intermediate-frame motion and cross-phase
image access during feature extraction.

## Main-ablation implementation audit

The completed E2-Full and E2-Static runs match in architecture, parameter
count, folds, seed value, source patches, batch size, augmentation distribution,
loss, optimizer, scheduler, stopping rule, supervision, and checkpoint rule.
They do **not** have identical stochastic sample trajectories:

- completed E2-Full used `DataLoader(shuffle=True)` and workers recreated each
  epoch;
- completed E2-Static used `PatientGroupedSampler(seed=42)` and persistent
  workers to reduce repeated compressed-NIfTI I/O.

Both visit every source patch once per epoch and perform the same number of
optimizer updates, but ordering and augmentation RNG trajectories differ. This
must be disclosed as a training-order sensitivity limitation; the completed
ablation should not be described as differing *only* in temporal input.

The historical E2-Full loader behavior is selected explicitly in the current
trainer with:

```bash
python train_exp2_3dconvlstm.py \
  --fold 1 \
  --train_order shuffled \
  --worker_lifecycle per_epoch \
  --augmentation_policy basic
```

The completed E2-Static behavior is selected with:

```bash
python train_exp2_3dconvlstm.py \
  --fold 1 \
  --static_input \
  --train_order patient_grouped \
  --worker_lifecycle persistent \
  --augmentation_policy basic
```

## Actual training and selected epochs

`Final` is the last completed epoch. `Best` is the checkpoint selected by
validation foreground MeanIoU.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| E1 final/best | 93/65 | 76/48 | 109/81 | 101/82 | 88/76 |
| E2-Full final/best | 110/83 | 88/61 | 126/99 | 150/139 | 113/86 |
| E2-Static final/best | 91/64 | 91/64 | 88/70 | 67/40 | 82/55 |
| E3 final/best | 108/99 | 101/83 | 102/75 | 85/58 | 62/44 |

These values are read directly from the saved checkpoints and `history.json`
files by `verify_four_model_release.py`.

## Evaluation

- OOF: each development patient is predicted only by its validation-fold best
  checkpoint.
- ACDC test and M&M: the five frozen fold checkpoints produce probabilities;
  probabilities are averaged before `argmax`.
- Metrics: RV/MYO/LV Dice, foreground Mean Dice, HD95, and ASD on reconstructed
  complete volumes. Patient-paired inference averages ED and ES within patient.
- Primary architecture statistics: Friedman test followed by paired Wilcoxon
  tests with Holm adjustment.
- E2 ablation: paired Wilcoxon and patient-level paired bootstrap estimates.

The release verifier recomputes the following foreground Mean Dice values from
the saved per-volume tables (100 ACDC OOF patients, 50 ACDC test patients, and
320 M&M subjects):

| Model | ACDC OOF | ACDC test | M&M external |
|---|---:|---:|---:|
| E1 | 0.7478 | 0.8030 | 0.6803 |
| E2-Full | 0.8264 | 0.8338 | 0.6972 |
| E2-Static | 0.8073 | 0.8059 | 0.6655 |
| E3 | 0.8107 | 0.8252 | 0.6855 |

These are ensemble results for the two evaluation cohorts: probabilities from
the five frozen fold checkpoints are averaged before label assignment. They
are not results from a separately retrained single model.

Main evaluation code:

- `finalize_exp1_wolny_cv.py`
- `finalize_exp2_cv.py`
- `finalize_exp3_cv.py`
- `evaluate_heldout_ensemble.py`
- `evaluate_mms_external.py`
- `compare_experiments.py`
- `compare_exp2_temporal_static.py`

## Software and hardware

Recorded environment:

- Python 3.13.7
- PyTorch 2.11.0+cu128; CUDA runtime 12.8
- NumPy 2.3.5
- SciPy 1.15.2
- NiBabel 5.3.2
- h5py 3.13.0
- pandas 2.2.3
- PyYAML 6.0.2
- Wolny `pytorch-3dunet` source revision
  `a33e2c7038a0b51296ca4f99b0f0ed4b61740705`

Training and evaluation used shared NVIDIA A100 80 GB PCIe GPUs. Wall-clock
times and utilization are load-sensitive because other users shared the GPUs;
they are not used to rank model efficiency.

## Verification

With local checkpoints/results available:

```bash
python verify_four_model_release.py
```

The verifier checks fixed split membership, parameter counts, actual/best
epochs, checkpoint-selection consistency, E2-Static configuration fields, and
the expected OOF, ACDC-test, and M&M result cardinalities.
