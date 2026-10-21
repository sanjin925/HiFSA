# HiFSA Installation Guide

This document explains how to prepare an environment for the current
HiFSA implementation, obtain its pretrained model assets, and arrange
the required datasets. Run all project commands from the repository
root, where `train.py`, `test.py`, `configs/`, and `trainers/` are
located.

> **Dependency note:** The supplied archive does not include a pinned
> `requirements.txt`, `environment.yml`, or a complete dependency lock
> file. The commands below provide a recommended starting environment
> based on the project's imports. Validate the package versions on the
> target machine and record the exact versions used for final
> experiments.

## 1. Prerequisites

-   Linux is recommended.
-   Python 3.10.
-   Conda or Miniconda.
-   An NVIDIA GPU and a compatible driver for GPU training.
-   Internet access on the first run to retrieve pretrained model assets
    from Hugging Face.

The commands below install PyTorch with the CUDA 12.6 runtime wheel. A
separate CUDA toolkit installation is usually unnecessary for running
this wheel, but the NVIDIA driver must be compatible. If your platform
needs another CUDA build, select a compatible PyTorch/torchvision pair
using the [official PyTorch installation
selector](https://pytorch.org/get-started/locally/).

## 2. Create the environment

``` bash
conda create -n hifsa python=3.10 -y
conda activate hifsa
python -m pip install --upgrade pip
```

## 3. Install PyTorch

``` bash
python -m pip install torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Keep PyTorch and torchvision on a compatible version pair. Do not
install CPU-only wheels if you intend to train on an NVIDIA GPU.

## 4. Install Python dependencies

``` bash
python -m pip install \
  "numpy<2" \
  monai \
  pandas openpyxl \
  pillow scipy \
  opencv-python-headless \
  matplotlib tqdm pyyaml \
  "huggingface_hub<1.0" \
  "transformers<5" \
  open_clip_torch timm \
  ftfy regex safetensors packaging typing_extensions
```

The main dependency groups are:

-   `monai`: segmentation loss utilities.
-   `pandas`, `openpyxl`: reading text-prompt spreadsheets.
-   `pillow`, `opencv-python-headless`, `scipy`: image/mask processing.
-   `matplotlib`, `tqdm`: visualization and progress display.
-   `pyyaml`: configuration loading.
-   `huggingface_hub`, `transformers`, `open_clip_torch`, `timm`:
    pretrained model and text-processing dependencies.
-   `ftfy`, `regex`, `safetensors`, `packaging`, `typing_extensions`:
    model/tokenizer support.

If an import error identifies another missing dependency, install it in
the active `hifsa` environment and record the version in your experiment
environment notes.

## 5. Verify the environment

From the repository root, run:

``` bash
python - <<'PY'
import torch
import torchvision
import monai
import cv2
import pandas
import open_clip
import transformers

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("MONAI:", monai.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Core imports: OK")
PY
```

If CUDA is unavailable on a machine with an NVIDIA GPU, check the active
Conda environment, driver compatibility, and installed PyTorch wheel
before training.

## 6. Prepare the datasets

The dataset files are not included in the code archive. The current
experiment configurations support `BTMRI`, `BUSI`, `Covid19`, and
`Kvasir`. Dataset archives are available from the dataset collection:

-   [BTMRI](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BTMRI.zip)
-   [BUSI](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BUSI.zip)
-   [Covid19 /
    QaTa-COV19](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/Covid19.zip)
-   [Kvasir-SEG](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/Kvasir.zip)

Place each dataset under the corresponding `data/<DATASET>/` directory:

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

Each prompt spreadsheet must contain the exact columns `Image`,
`Ground Truth`, and `Description`. Image and mask filenames must match
files under `img/` and `label/`, respectively.

## 7. Prepare low-annotation splits

The supported annotation ratios are `10`, `25`, `50`, and `100`, passed
through `--data_percentage`. For ratios below 100, prepare corresponding
files such as:

``` text
Train_text_10.xlsx
Val_text_10.xlsx
Train_text_25.xlsx
Val_text_25.xlsx
Train_text_50.xlsx
Val_text_50.xlsx
```

The loader may fall back to the full `Train_text.xlsx` and
`Val_text.xlsx` files if a requested subset file is absent. This means
that specifying `--data_percentage 10` alone does not guarantee a 10%
run. Inspect the training log entries `Loading train text from:` and
`Loading val text from:` and verify that the expected subset
spreadsheets were selected.

## 8. Pretrained model assets

The default experiment configurations use `unimedclip` with the
`ViT-B/16` backbone. During model initialization, the code downloads the
required UniMedCLIP checkpoint from Hugging Face if it is not present
locally. The expected checkpoint path is:

``` text
checkpoints/unimed_clip_vit_b16.pt
```

Run the project from its repository root so that the relative checkpoint
path resolves correctly. The first run may also retrieve
tokenizer/text-model assets. If the machine has no network access, cache
or transfer the required model assets using your institution's approved
workflow before launching experiments.

Do not commit large model checkpoints or private datasets to a public
repository unless redistribution is permitted.

## 9. Check the entry points

Before starting a full experiment, verify that the scripts load and
expose their command-line arguments:

``` bash
python train.py --help
python test.py --help
python utils/eval.py --help
```

Then start a full-model run after the dataset is prepared:

``` bash
python train.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir outputs \
  --data_percentage 100 \
  --seed 666
```

For the full training, testing, evaluation, and ablation commands, see
[README.md](README.md).

## Troubleshooting

### `ModuleNotFoundError`

Activate the `hifsa` environment and install the missing package with
`python -m pip install <package>`. For example, the import name
`open_clip` is provided by the package `open_clip_torch`.

### CUDA is unavailable

Confirm that the active environment contains the CUDA-enabled PyTorch
wheel and that the installed NVIDIA driver supports it.

### Hugging Face download errors

Check network connectivity, cache permissions, and write access to the
repository's `checkpoints/` directory. The pretrained checkpoint is not
bundled with the code archive.

### Missing images, masks, or prompt spreadsheets

Verify the configured dataset paths and the exact prompt columns:
`Image`, `Ground Truth`, and `Description`. Check that every filename
resolves to an existing file.

### Low-annotation experiment uses the wrong split

Inspect the log messages that print the selected train and validation
prompt paths. Create the required subset spreadsheets if the loader has
selected the full split files.
