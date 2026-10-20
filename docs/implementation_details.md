# Implementation details

The exhaustive implementation specification is maintained in
[`reproducibility.md`](reproducibility.md). In particular, see:

- **Preprocessing and supervision** for spacing, shape, temporal alignment,
  labels, patches, and augmentation.
- **Shared optimization policy** for the loss, optimizer, scheduler, seeds,
  stopping behavior, and checkpoint rule.
- **E2 ConvLSTM specification** for the complete temporal architecture.
- **Main-ablation implementation audit** for the historical loader-order
  difference that qualifies the E2-Full versus E2-Static comparison.

