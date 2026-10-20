# Audited training runs

`Final` is the last completed epoch. `Best` is the strict highest validation
foreground MeanIoU checkpoint.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| E1 final/best | 93/65 | 76/48 | 109/81 | 101/82 | 88/76 |
| E2-Full final/best | 110/83 | 88/61 | 126/99 | 150/139 | 113/86 |
| E2-Static final/best | 91/64 | 91/64 | 88/70 | 67/40 | 82/55 |
| E3 final/best | 108/99 | 101/83 | 102/75 | 85/58 | 62/44 |

The values were read from the saved checkpoints and histories, not inferred
from filenames or terminal logs. The currently running shuffle-matched
E2-Static sensitivity experiment is not included.

