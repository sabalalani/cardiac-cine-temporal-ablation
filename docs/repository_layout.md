# Repository layout and experiment coverage

This page describes every source/configuration group distributed in the
publication-code repository. Data, labels, checkpoints, prediction arrays, and
generated result tables are intentionally excluded.

## Experiment coverage

| Experiment | Configuration | Model interface | Training | OOF evaluation | ACDC test and M&M |
|---|---|---|---|---|---|
| E1 | `configs/e1.yaml` | `models/unet3d.py` | `training/prepare_exp1_wolny_native.py` plus pinned Wolny trainer | `evaluation/finalize_exp1_wolny_cv.py` | Shared ensemble evaluators |
| E2-Full | `configs/e2_full.yaml` | `models/convlstm3d.py` | `training/train_exp2_3dconvlstm.py` | `evaluation/finalize_exp2_cv.py` | Shared ensemble evaluators |
| E2-Static | `configs/e2_static.yaml` | `models/convlstm3d.py` | Same E2 trainer with `--static_input` | Same E2 finalizer with `--static_input` | Shared ensemble evaluators |
| E3 | `configs/e3.yaml` | `models/factorized_unet.py` | `training/train_exp3_factorized.py` | `evaluation/finalize_exp3_cv.py` | Shared ensemble evaluators |

The shared ensemble evaluators are
`evaluation/evaluate_heldout_ensemble.py` and
`evaluation/evaluate_mms_external.py`. They recognize exactly the four
reported configurations above.

## Root files

- `README.md`: project overview, setup, principal commands, results, and the
  E2 loader-order qualification.
- `requirements.txt`: recorded Python-package versions.
- `environment.yml`: Conda environment definition.
- `.gitignore`: prevents patient data, checkpoints, predictions, and generated
  outputs from being committed.
- `verify_four_model_release.py`: optional read-only integrity audit when the
  private/local checkpoints and result tables are placed at their documented
  paths. It is not required for training.

## `configs/`

- `acdc_5fold_splits_seed42.json`: fixed diagnosis-stratified patient split;
  each of the 100 development patients occurs in exactly one validation fold.
- `e1.yaml`, `e2_full.yaml`, `e2_static.yaml`, and `e3.yaml`: concise audited
  records of the reported configurations. The executable CLI defaults remain
  in the training scripts.

## `models/`

- `unet3d.py`: factory for the pinned Wolny 3D U-Net used by E1.
- `convlstm3d.py`: public imports for E2-Full and E2-Static. Both configurations
  use the same trainable architecture and parameter count.
- `factorized_unet.py`: public import for E3.
- `__init__.py`: exposes the four reported model interfaces.

The E2 and E3 definitions are imported directly from their exact training
implementations to prevent a second, potentially divergent architecture copy.

## `training/`

- `prepare_exp1_wolny_native.py`: converts preprocessed ED/ES volumes to HDF5
  and generates fold-specific Wolny training and prediction configurations.
  E1 optimization is performed by the pinned external Wolny trainer.
- `train_exp2_3dconvlstm.py`: dataset, augmentation, sampler, E2 architecture,
  E2-Static construction, optimization, scheduler, checkpointing, and resume
  logic.
- `train_exp3_factorized.py`: E3 factorized architecture and training loop,
  reusing the shared E2 dataset and optimization helpers.
- `__init__.py`: package marker.

There is no separate E2-Static trainer because a second copy could silently
diverge. The `--static_input` flag changes only input construction/model
forward behavior while retaining the E2 training implementation.

## `preprocessing/`

- `preprocess_acdc_spatial.py`: ACDC resampling, crop/pad, label handling, and
  spatial volume generation.
- `prepare_acdc_temporal.py`: periodic resampling to 30 ED-aligned frames and
  exact ED/ES frame preservation.
- `preprocess_mms_external.py`: matching M&M spatial and temporal preparation,
  label remapping, HDF5 export, and audit-table creation.
- `qa_acdc_temporal.py`: ACDC temporal-preprocessing quality checks.
- `qa_mms_external.py`: M&M preprocessing and evaluation quality checks.
- `audit_mms_provenance_alignment.py`: annotation provenance, subject,
  ED/ES, and prediction-to-ground-truth alignment audit.
- `__init__.py`: package marker.

## `evaluation/`

- `evaluate_exp1_wolny_native.py`: E1 reconstructed-volume Dice, HD95, and ASD.
- `evaluate_exp2_3dconvlstm.py`: full-volume E2/E2-Static patch inference and
  metrics.
- `evaluate_exp3_factorized.py`: full-volume E3 inference using the same
  reconstruction and metric implementation as E2.
- `finalize_exp1_wolny_cv.py`, `finalize_exp2_cv.py`, and
  `finalize_exp3_cv.py`: fold evaluation and OOF pooling.
- `evaluate_heldout_ensemble.py`: locked five-checkpoint probability-ensemble
  evaluation on the 50 ACDC test patients.
- `evaluate_mms_external.py`: frozen five-checkpoint probability-ensemble
  evaluation on 320 labeled M&M subjects.
- `compare_experiments.py`: patient-paired comparison of the principal models.
- `compare_exp2_temporal_static.py`: E2-Full versus E2-Static paired Wilcoxon,
  effect-size, and patient-level bootstrap analysis.
- `analyze_mms_e2_vendor_paired.py`: vendor-stratified paired Dice and HD95
  analysis for E2-Full versus E2-Static.
- `__init__.py`: package marker.

## `docs/`

- `implementation_details.md`: short index to mandatory implementation facts.
- `training_runs.md`: audited final and selected epochs for all folds.
- `reproducibility.md`: complete cohorts, preprocessing, architectures,
  optimization, seeds, epochs, evaluation, software, hardware, results, and
  ablation-qualification record.
- `repository_layout.md`: this file.

