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
The updated paired E2 ablation is summarized in
[docs/e2_ablation_results.md](docs/e2_ablation_results.md).

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
python training/train_exp2_3dconvlstm.py --fold 1 \
  --train_order shuffled --worker_lifecycle per_epoch \
  --augmentation_policy basic
python training/train_exp2_3dconvlstm.py --fold 1 --static_input \
  --train_order shuffled --worker_lifecycle persistent \
  --augmentation_policy basic --output_root exp2_static_shuffle_matched
python training/train_exp3_factorized.py --fold 1 --augmentation_policy basic
```

E1 configuration generation is run with:

```bash
python training/prepare_exp1_wolny_native.py --augmentation_policy basic
```

## Important ablation qualification

The reported E2-Full and rerun E2-Static models shared shuffled patch loading,
architecture, folds, seed, patches, augmentation distribution, joint ED/ES
loss, optimizer, scheduler, supervision, and checkpoint criterion. E2-Full
recreated workers each epoch, whereas E2-Static retained persistent workers;
the resulting augmentation random-number trajectories were therefore not
guaranteed to be identical. This residual limitation must be disclosed.

E2-Static received repeated ED or ES images, but patch standardization was
computed from the original 30-frame patch before repetition. Intermediate
frames therefore affected two scalar normalization statistics, although their
images were not passed to the encoder or ConvLSTM.

## Verified aggregate results

| Model | ACDC OOF Dice | ACDC test Dice | M&M Dice |
|---|---:|---:|---:|
| E1 | 0.7478 | 0.8030 | 0.6803 |
| E2-Full | 0.8264 | 0.8338 | 0.6972 |
| E2-Static | 0.8258 | 0.8257 | 0.6881 |
| E3 | 0.8107 | 0.8252 | 0.6855 |

These correspond to OOF predictions and frozen five-checkpoint probability
ensembles, as defined in the documentation. They are not single-model results.

