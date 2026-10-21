# HiFSA Native naming

The executable code uses one naming system only:

| Paper module | Switch | Class or registered state | Active method |
|---|---|---|---|
| AFPI | `MODEL.USE_AFPI` | `afpi_modules`, `afpi_gate`, `afpi_mix_logits` | `inject_afpi_residual` |
| BFCI | `MODEL.USE_BFCI` | `BFCIResidualBranch`, `bfci_branch` | `inject_bfci_residual` |
| LCFSA | `MODEL.USE_LCFSA` | `LCFSA`, `lcfsa`, `lcfsa_mix_logits` | `inject_lcfsa_residual` |

Shared decomposition settings use the neutral `FREQUENCY_*` prefix. AFPI-, BFCI-, and LCFSA-owned model and optimizer settings use their corresponding module prefix.

There is no alias layer and no dual-spelling compatibility path. Old configuration keys, old registered parameter names, and old checkpoints are rejected rather than silently translated. This keeps runtime names, logs, optimizer groups, result names, and checkpoint keys consistent with the paper.

## Configuration interface

Primary configs are `configs/*_MODULAR.yaml`. They intentionally contain only:

- dataset paths, dataset name, and image size;
- public training and testing values;
- stable CLIP/backbone and baseline model values.

Dataset-specific HiFSA hyperparameters live in [`utils/hifsa_defaults.py`](../utils/hifsa_defaults.py). AFPI, BFCI, and LCFSA default to enabled. Training-time CLI overrides are the ablation interface, for example:

```bash
python train.py --config-file configs/BTMRI_MODULAR.yaml MODEL.USE_BFCI False
```

Result names are generated from the enabled Native modules, such as `HiFSA_AFPI_BFCI_LCFSA_unimedclip_ViT-B-16`.
