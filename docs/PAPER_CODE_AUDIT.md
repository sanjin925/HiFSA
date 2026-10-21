# HiFSA paper-code consistency notes

Updated: 2026-10-15.

## Current implementation contract

- Code, configuration resolution, logs, result paths, optimizer groups, and checkpoint state use AFPI, BFCI, and LCFSA consistently.
- `configs/*_MODULAR.yaml` are the primary minimal configurations. Dataset-specific module hyperparameters are centralized in `utils/hifsa_defaults.py`.
- All modules default to enabled. Ablations are selected only by training CLI overrides.
- `TEST.USE_LATEST` is `True`, so the default evaluation target is the final-epoch checkpoint. Testing rejects an incomplete latest checkpoint.
- Every new checkpoint includes Native HiFSA metadata. Test-time construction restores the module layout from that metadata.
- Historical checkpoints and historical configuration spellings are intentionally unsupported.

## Paper details that must match the experiments

1. The dynamic AFPI router is enabled for BTMRI, BUSI, Covid19, and Kvasir.
2. The default evaluation protocol uses the checkpoint from the final training epoch; no best-validation checkpoint is selected or written.
3. The total loss includes the configured boundary regularizer and the LCFSA auxiliary localization term in addition to Dice, BCE, and CLIP terms.
4. All datasets use batch size 24 and initial learning rate `3e-4`; all optimizer parameter groups inherit that learning rate. Injection layers, warm-up schedules, and auxiliary weights remain dataset-specific and are recorded in the resolved YAML artifact saved beside each checkpoint.
5. BTMRI uses grayscale Haar-energy decomposition without per-image min-max normalization.
6. Predictions are thresholded at 0.5. The evaluation code computes macro-averaged DSC and NSD, with NSD tolerance 2 under unit spacing.

Numerical claims still depend on the exact data splits, checkpoints, prompts, and raw evaluation outputs used for the paper. The naming/configuration migration does not itself validate or reproduce those results.
