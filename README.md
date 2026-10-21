# HiFSA: Hierarchical Frequency-Semantic Adaptation for Data-Efficient Medical Vision-Language Segmentation

HiFSA is a frequency-semantic adaptation framework for medical image
segmentation under limited pixel-level annotation. It constructs
dataset-adaptive high- and low-frequency representations and uses three
independent residual modules---AFPI, BFCI, and LCFSA---to enhance visual
features with frequency priors, cross-frequency interaction, and
language-conditioned frequency selection.

## Method overview

### 1. Dataset-adaptive frequency representation

Given an input image, `AdaptiveFrequencyDecompose` generates a
high-frequency representation and a low-frequency representation. The
decomposition strategy is selected according to the dataset's image
characteristics:

-   **Haar energy decomposition (`haar_energy`)**: uses the LL sub-band
    for low-frequency structure and aggregates LH/HL/HH responses by
    energy for high-frequency detail.
-   **Grayscale-normalized Haar decomposition (`gray_normalized`)**:
    converts the input to grayscale, optionally performs per-image
    normalization, and derives complementary frequency components.
-   **Gaussian residual decomposition (`gaussian`)**: obtains the
    low-frequency component through Gaussian smoothing and the
    high-frequency component through an image-space residual.
-   **Laplacian decomposition (`laplacian`)**: combines Gaussian
    low-pass structure with Laplacian edge responses; the implementation
    supports this mode even though it is not selected by the four
    default dataset configurations.

The high- and low-frequency images are converted into patch tokens using
the same pretrained patch-embedding weights. The resulting
frequency-token bank is shared as an immutable input to the three
modules. Each module processes its own copy, so one module does not
overwrite the frequency information used by another.

### 2. AFPI --- Adaptive Frequency Prior Injection

AFPI injects dataset-adaptive frequency information at an early
visual-transformer layer.

1.  It combines high- and low-frequency tokens using learnable mixture
    weights initialized from dataset-specific priors.
2.  An optional image-level dynamic router predicts corrections to the
    frequency mixture and injection gate from pooled visual and
    frequency features.
3.  An optional lightweight frequency adapter refines the fused tokens.
4.  A bottleneck residual projection generates a feature correction,
    which is injected through a bounded learnable gate and a warm-up
    multiplier.

This design lets each image adjust the contribution of high- and
low-frequency information while preserving a controlled residual
pathway. The router's final layer is zero-initialized, so the initial
routing behavior follows the configured prior.

### 3. BFCI --- Bidirectional Frequency Complementary Interaction

BFCI models complementary relationships between high-frequency and
low-frequency tokens through two cross-attention operations:

-   **High from low:** high-frequency queries attend to low-frequency
    keys and values.
-   **Low from high:** low-frequency queries attend to high-frequency
    keys and values.

The two outputs are projected back to the visual feature dimension and
combined with learnable high-/low-frequency mixture weights. A signed,
bounded residual gate and an independent warm-up schedule control the
contribution of BFCI. Its parameters, optimizer group, injection layer,
and gate are independent of AFPI and LCFSA.

### 4. LCFSA --- Language-Conditioned Frequency Selection Adapter

LCFSA selects frequency information according to language and visual
context at a deeper layer.

1.  It projects visual, high-frequency, low-frequency, and text tokens
    into a compact interaction space.
2.  High- and low-frequency tokens are aligned with the current visual
    context.
3.  Text tokens first aggregate visual context; frequency tokens then
    query this contextualized language representation.
4.  Language-conditioned channel and spatial responses select frequency
    information for the residual update.
5.  The module retains spatial logits that can be supervised by an
    auxiliary patch-mask localization loss during training.

LCFSA has its own learnable high-/low-frequency mixture, signed residual
gate, injection layer, and warm-up schedule. Its auxiliary localization
loss is enabled through the dataset-specific training defaults.

### 5. Hierarchical integration and training objective

AFPI, BFCI, and LCFSA are independent residual branches inserted at
progressively deeper visual-transformer layers. They share the raw
frequency-token bank, but have separate parameters, gates, warm-up
schedules, optimizer groups, and configuration switches. The default
configuration enables all three branches.

The training objective is assembled from the following terms, with
weights resolved from the selected dataset configuration:

\[ `\mathcal{L}`{=tex} =
`\lambda`{=tex}*{`\mathrm{dice}`{=tex}}`\mathcal{L}`{=tex}*{`\mathrm{dice}`{=tex}}+
`\lambda`{=tex}*{`\mathrm{bce}`{=tex}}`\mathcal{L}`{=tex}*{`\mathrm{bce}`{=tex}}+
`\lambda`{=tex}*{`\mathrm{clip}`{=tex}}`\mathcal{L}`{=tex}*{`\mathrm{clip}`{=tex}}+
`\lambda`{=tex}*{`\mathrm{bd}`{=tex}}`\mathcal{L}`{=tex}*{`\mathrm{boundary}`{=tex}}+
`\lambda`{=tex}*{`\mathrm{loc}`{=tex}},r*{`\mathrm{LCFSA}`{=tex}}`\mathcal{L}`{=tex}\_{`\mathrm{loc}`{=tex}}.
\]

The boundary and localization terms are used when their configured
weights are positive. The LCFSA localization term is additionally scaled
by its warm-up progress.

## Dataset-specific defaults

The current implementation supports four datasets through the supplied
configuration files and `utils/hifsa_defaults.py`.

  -----------------------------------------------------------------------------------
  Dataset     Frequency            AFPI / BFCI /   Initial AFPI AFPI       AFPI
              decomposition          LCFSA layer       high/low dynamic    adapter
                                         indices        weights router     
  ----------- ------------------- -------------- -------------- ---------- ----------
  `BTMRI`     `haar_energy`            3 / 5 / 8    0.60 / 0.40 Enabled    Disabled

  `BUSI`      `gaussian`               3 / 6 / 8    0.05 / 0.95 Enabled    Enabled

  `Covid19`   `gray_normalized`        4 / 6 / 8    0.20 / 0.80 Enabled    Disabled

  `Kvasir`    `haar_energy`            3 / 5 / 8    0.50 / 0.50 Enabled    Disabled
  -----------------------------------------------------------------------------------

These are initialization settings, not guaranteed final learned weights.
BFCI and LCFSA also have their own dataset-specific frequency mixtures
and gates. Their detailed defaults, warm-up schedules, and loss weights
are centralized in `utils/hifsa_defaults.py`.

## Repository structure

``` text
HiFSA/
├── configs/
│   ├── BTMRI_MODULAR.yaml
│   ├── BUSI_MODULAR.yaml
│   ├── Covid19_MODULAR.yaml
│   └── Kvasir_MODULAR.yaml
├── datasets/
│   └── dataloader.py
├── trainers/
│   ├── freq_modules.py
│   └── medclipseg_unimedclip.py
├── utils/
│   ├── hifsa_defaults.py
│   ├── checkpoint_utils.py
│   ├── experiment_utils.py
│   ├── main_utils.py
│   ├── SurfaceDice.py
│   └── eval.py
├── scripts/
│   ├── efficiency.sh
│   ├── reproduce.sh
│   └── run_hifsa_ablation.sh
├── train.py
├── test.py
├── install.md
└── README.md
```

## Installation

Follow **[install.md](install.md)** for environment setup, Python
dependencies, pretrained checkpoint downloads, and dataset preparation.

The code archive does not contain a pinned `requirements.txt` or a
complete dependency lock file. For reproducible research, record the
package versions from the environment used for the final experiments.

## Dataset preparation

The datasets are not bundled with the code. The default configurations
expect the following layout for each dataset:

``` text
data/<DATASET>/
├── Prompts_Folder/
│   ├── Train_text.xlsx
│   ├── Val_text.xlsx
│   ├── Test_text_original.xlsx
│   └── subset prompt files for low-annotation experiments
├── Train_Folder/
│   ├── img/
│   └── label/
├── Val_Folder/
│   ├── img/
│   └── label/
└── Test_Folder/
    ├── img/
    └── label/
```

The prompt spreadsheets must contain these exact column names:

-   `Image`: image filename relative to the corresponding `img/`
    directory.
-   `Ground Truth`: mask filename relative to the corresponding `label/`
    directory.
-   `Description`: text prompt for the image.

The image and mask filenames must match files on disk. Ground-truth
masks are read as grayscale and binarized using a threshold of 127.

Dataset archives available from the upstream dataset collection:

-   [BTMRI](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BTMRI.zip)
-   [BUSI](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BUSI.zip)
-   [Covid19 /
    QaTa-COV19](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/Covid19.zip)
-   [Kvasir-SEG](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/Kvasir.zip)

Please follow the applicable dataset terms and cite the relevant dataset
publications in academic work.

### Low-annotation experiments

The supported command-line values are `--data_percentage 10`, `25`,
`50`, and `100`. For 10%, 25%, or 50% experiments, ensure the
appropriate subset spreadsheets exist, such as `Train_text_10.xlsx` and
`Val_text_10.xlsx`.

The loader may fall back to the full `Train_text.xlsx` or
`Val_text.xlsx` if the requested subset file is missing. Therefore,
inspect the training log messages `Loading train text from:` and
`Loading val text from:` before treating a run as a low-annotation
experiment. The percentage argument does not generate subset
spreadsheets automatically.

## Training

Run commands from the repository root. The example below trains the full
HiFSA model on BTMRI using all available training annotations.

``` bash
python train.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir outputs \
  --data_percentage 100 \
  --seed 666
```

All three modules are enabled by default through the resolved HiFSA
configuration. Dataset-specific module settings are centralized in
`utils/hifsa_defaults.py`; the public YAML files intentionally expose a
minimal set of stable experiment inputs.

### Resume training

Reuse the same configuration, output directory, seed, annotation ratio,
and module switches, then add `--resume`:

``` bash
python train.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir outputs \
  --data_percentage 100 \
  --seed 666 \
  --resume
```

The training script saves the latest checkpoint after each epoch and
writes the resolved configuration beside the checkpoint.

## Testing and evaluation

### Generate predictions

``` bash
python test.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir outputs \
  --source_dataset BTMRI \
  --data_percentage 100 \
  --seed 666 \
  --prompt_design original
```

The test script restores the AFPI/BFCI/LCFSA layout from checkpoint
metadata. For the default protocol, the latest checkpoint must
correspond to the configured final epoch. For an ablation run, pass the
exact checkpoint path using `--checkpoint-file`.

### Calculate metrics

``` bash
python utils/eval.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir outputs \
  --data_percentage 100 \
  --seed 666 \
  --prompt_design original
```

The evaluator reports mean **Dice Similarity Coefficient (DSC)** and
**Normalized Surface Dice (NSD)**. Ground-truth and predicted masks are
binarized at 127; NSD uses a tolerance of 2 under unit pixel spacing.
Keep the dataset, annotation ratio, seed, prompt design, output
directory, and result name consistent across testing and evaluation.

## Module ablations

The three module switches are controlled at training time:

-   `MODEL.USE_AFPI`
-   `MODEL.USE_BFCI`
-   `MODEL.USE_LCFSA`

For example, train an AFPI-only configuration:

``` bash
python train.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir outputs/ablation_afpi \
  --data_percentage 10 \
  --seed 666 \
  MODEL.USE_AFPI True \
  MODEL.USE_BFCI False \
  MODEL.USE_LCFSA False
```

Run all eight combinations at 10% annotation using seed 666:

``` bash
PERCENTAGE=10 SEED=666 bash scripts/run_hifsa_ablation.sh outputs_hifsa_ablation
```

This script performs training, testing, and evaluation for all eight
combinations. To test a particular ablation checkpoint manually, pass
its exact path to `test.py --checkpoint-file ...`; the script restores
the module layout from the checkpoint metadata. Do not change module
switches manually during evaluation.

## Multi-dataset reproduction

Run the standard train-test-evaluate pipeline for one dataset and
annotation ratio:

``` bash
bash scripts/efficiency.sh outputs BTMRI 100 666
```

Run the supplied reproduction protocol across BTMRI, BUSI, Covid19, and
Kvasir at 10%, 25%, 50%, and 100% annotation:

``` bash
bash scripts/reproduce.sh outputs_hifsa_reproduce
```

This protocol launches multiple independent experiments and may require
substantial GPU time.

## Output organization

A typical full-model run creates files in a structure similar to:

``` text
outputs/
└── BTMRI/
    ├── logs/seed666/
    ├── trained_models/seed666/
    │   ├── HiFSA_AFPI_BFCI_LCFSA_unimedclip_ViT-B-16_latest.pth
    │   └── HiFSA_AFPI_BFCI_LCFSA_unimedclip_ViT-B-16_resolved.yaml
    └── seg_results/seed666/
        ├── HiFSA_AFPI_BFCI_LCFSA_unimedclip_ViT-B-16_Prompt-original/
        └── HiFSA_AFPI_BFCI_LCFSA_unimedclip_ViT-B-16_Prompt-original.csv
```

When `data_percentage` is below 100, the dataset output directory
includes the annotation-ratio suffix. Keep logs, resolved YAML files,
checkpoints, predictions, and evaluation CSV files for every reported
result.

## Implementation notes

-   The supplied modular configurations target `unimedclip` with
    `ViT-B/16`. The current implementation does not support the
    `ViT-L/14` branch in the same trainer.
-   Frequency decomposition modes implemented in code are `haar_energy`,
    `haar_signed`, `gray_normalized`, `gaussian`, and `laplacian`.
-   AFPI, BFCI, and LCFSA use separate parameters, gates, warm-up
    schedules, optimizer groups, and injection layers. Their ablations
    therefore can be run independently.
-   Checkpoints created before the current module-naming and metadata
    format may not pass strict compatibility checks.
-   The repository archive does not include a root `LICENSE` file. Add
    the intended license before redistributing the code.

## Citation

Add the final HiFSA manuscript citation here once the author list and
bibliographic details are finalized. Academic use should also cite the
datasets and pretrained model assets according to their official
publications and terms. Do not list placeholder bibliographic details as
published references.
