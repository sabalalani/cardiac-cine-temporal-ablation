# Cardiac cine temporal ablation

Publication-code package for a controlled comparison of four cardiac MRI
segmentation configurations:

| ID | Input | Model |
|---|---|---|
| E1 | Separate ED and ES volumes | Wolny 3D U-Net |
| E2-Full | Real 30-frame cine | Bidirectional 3D ConvLSTM U-Net |
| E2-Static | 30 repeated ED frames and 30 repeated ES frames | Same E2 network |
| E3 | Real 30-frame cine | Factorized spatial-temporal U-Net |

No patient images, labels, checkpoints, or prediction arrays are distributed.
Obtain ACDC and M&M under their respective terms and supply the paths described
in [docs/reproducibility.md](docs/reproducibility.md).

## Layout

- `configs/`: fixed split and audited experiment records.
- `models/`: importable public model interfaces.
- `training/`: exact scripts used for the four experiments.
- `evaluation/`: OOF, frozen-ensemble, external-validation, and paired analyses.
- `preprocessing/`: ACDC and M&M spatial/temporal preparation and QA.
- `docs/`: implementation, run, and reproducibility records.

See [docs/repository_layout.md](docs/repository_layout.md) for the purpose of
every distributed file and a model-by-model coverage table.

## Environment

```bash
conda env create -f environment.yml
conda activate cardiac-cine-ablation
```

E1 additionally uses Wolny `pytorch-3dunet` at commit
`a33e2c7038a0b51296ca4f99b0f0ed4b61740705`; see the reproducibility record.

The scripts retain their historical filenames and CLI arguments. From this
repository root, expose the executable directories before invoking them:

```bash
export PYTHONPATH="$PWD/training:$PWD/evaluation:$PYTHONPATH"
python training/train_exp2_3dconvlstm.py --fold 1 --augmentation_policy basic
python training/train_exp2_3dconvlstm.py --fold 1 --static_input \
  --train_order patient_grouped --worker_lifecycle persistent \
  --augmentation_policy basic --output_root exp2_static_convlstm
python training/train_exp3_factorized.py --fold 1 --augmentation_policy basic
```

E1 configuration generation is run with:

```bash
python training/prepare_exp1_wolny_native.py --augmentation_policy basic
```

## Important ablation qualification

The completed E2-Full and E2-Static runs shared architecture, folds, seed,
patches, augmentation distribution, loss, optimizer, scheduler, supervision,
and checkpoint criterion. Their historical loader ordering differed: E2-Full
used shuffled loading with per-epoch workers, whereas E2-Static used a
patient-grouped sampler with persistent workers. This is documented as a
training-order sensitivity limitation and must not be hidden.

## Verified aggregate results

| Model | ACDC OOF Dice | ACDC test Dice | M&M Dice |
|---|---:|---:|---:|
| E1 | 0.7478 | 0.8030 | 0.6803 |
| E2-Full | 0.8264 | 0.8338 | 0.6972 |
| E2-Static | 0.8073 | 0.8059 | 0.6655 |
| E3 | 0.8107 | 0.8252 | 0.6855 |

These correspond to OOF predictions and frozen five-checkpoint probability
ensembles, as defined in the documentation. They are not single-model results.

