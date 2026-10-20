# E2-Full versus shuffled-order E2-Static

This analysis uses the completed E2-Static rerun with shuffled patch ordering.
ED and ES Dice values are averaged within patient before paired inference.
Confidence intervals use 100,000 patient-level paired percentile-bootstrap
replicates with seed 42. Wilcoxon tests are two-sided with Pratt zero handling.

## Primary patient-level Mean Dice contrast

Differences are E2-Full minus E2-Static; positive values favor E2-Full.

| Cohort | Patients | Full | Static | Mean paired difference (95% CI) | Full better | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| ACDC OOF | 100 | 0.8264 | 0.8258 | 0.00065 (-0.00896, 0.00962) | 61% | 0.0252 |
| ACDC held-out | 50 | 0.8338 | 0.8257 | 0.00812 (0.00403, 0.01182) | 82% | 0.000003 |
| M&M external | 320 | 0.6972 | 0.6881 | 0.00910 (0.00476, 0.01312) | 78.1% | 1.07e-17 |

The OOF bootstrap interval for the mean difference includes zero even though
the Wilcoxon test is significant. These procedures estimate different aspects
of the paired distribution and should both be reported.

## M&M vendor-stratified Mean Dice

| Vendor | N | Full | Static | Difference (95% CI) | Full better |
|---|---:|---:|---:|---:|---:|
| Canon | 50 | 0.6738 | 0.6586 | 0.0153 (0.0041, 0.0252) | 82.0% |
| GE | 50 | 0.6527 | 0.6622 | -0.0095 (-0.0259, 0.0039) | 54.0% |
| Philips | 125 | 0.7426 | 0.7260 | 0.0167 (0.0119, 0.0218) | 88.8% |
| Siemens | 95 | 0.6732 | 0.6675 | 0.0057 (-0.0019, 0.0124) | 74.7% |

HD95 results use complete paired outcomes and may have smaller subgroup sample
sizes. Siemens has 93 complete paired aggregate-HD95 patients, although 95
patients have complete Dice results.

Aggregate machine-readable outputs are under `results/e2_full_vs_static/`.
Patient-level rows are intentionally not committed.

## Interpretation qualifications

- Both runs used shuffled patch ordering, but E2-Full recreated workers each
  epoch whereas E2-Static used persistent workers.
- E2-Static receives repeated ED or ES images, but standardization statistics
  are computed from the original 30-frame patch before repetition.
- Both models use one class-weighted mean over jointly flattened ED and ES
  voxels, followed by one backward pass and optimizer update.

