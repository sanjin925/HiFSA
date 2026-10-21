# HiFSA architecture trace

## End-to-end path

1. `utils/hifsa_defaults.py` resolves dataset-specific internal settings. The public YAML files contain only stable dataset, training, test, and backbone inputs.
2. `AdaptiveFrequencyDecompose` creates high- and low-frequency images. The frozen CLIP patch embedding, positional embedding, and pre-normalization create one immutable shared HF/LF token bank.
3. The frozen visual and text encoders run with the baseline PVL adapters.
4. AFPI reads the shared bank at its shallow injection layer, weights the HF/LF tokens, applies its adapter/projection and bounded gate, then injects a visual residual.
5. BFCI independently reads the original shared bank at an intermediate layer. `BFCIResidualBranch` performs bidirectional HF/LF cross-attention and injects its own gated residual.
6. LCFSA reads the same shared bank plus the current visual and text states at a deep layer. `LCFSA` performs language-conditioned spatial/channel selection and provides the optional training-only localization loss.
7. The existing MedCLIPSeg head consumes the resulting visual states. No HiFSA branch rewrites the shared token bank used by another branch.

The active integration is in [`trainers/medclipseg_unimedclip.py`](../trainers/medclipseg_unimedclip.py), and the frequency modules are implemented in [`trainers/freq_modules.py`](../trainers/freq_modules.py).

## Dataset-resolved structure

| Dataset | Frequency mode | AFPI / BFCI / LCFSA layer | Dynamic AFPI router | LCFSA context detach |
|---|---|---|---|---|
| BTMRI | `haar_energy` | 3 / 5 / 8 | true | true |
| BUSI | `gaussian` | 3 / 6 / 8 | false | true |
| Covid19 | `gray_normalized` | 4 / 6 / 8 | true | true |
| Kvasir | `haar_energy` | 3 / 5 / 8 | true | true |

The `*_MODULAR.yaml` files are the primary entry points. The matching `*_HiFSA.yaml` files expose the same minimal values. All three modules default to enabled in code; a training command may disable any module with `MODEL.USE_AFPI False`, `MODEL.USE_BFCI False`, or `MODEL.USE_LCFSA False`.

## Gradient and checkpoint boundaries

The vision and text encoder parameters are frozen. PVL adapters, the segmentation head, and enabled HiFSA modules remain trainable. LCFSA uses detached visual, text, and frequency context under the dataset defaults, so its branch cannot back-propagate into those producers.

New checkpoints store the dataset, backbone, CLIP model, and three module switches. Testing restores the module layout from this metadata before model construction. Checkpoints produced before the Native naming migration are intentionally unsupported.
