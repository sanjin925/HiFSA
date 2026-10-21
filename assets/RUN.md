# HiFSA training and evaluation

Run commands from the repository root. The supported datasets are `BTMRI`, `BUSI`, `Covid19`, and `Kvasir`; their primary entry points are `configs/*_MODULAR.yaml`.

All three HiFSA modules are enabled by default. A module can be disabled only through a training command-line override:

```bash
python train.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir output \
  --seed 666 \
  MODEL.USE_BFCI False
```

Training writes a resolved YAML beside its checkpoints. New checkpoints also contain the dataset, backbone, CLIP model, and AFPI/BFCI/LCFSA switch state. The checkpoint overwritten after every epoch is named `*_latest.pth`; testing requires it to be the configured final epoch (100 by default).

For the default full model, test and evaluate with:

```bash
python test.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir output \
  --source_dataset BTMRI \
  --seed 666

python utils/eval.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir output \
  --seed 666
```

For an ablation checkpoint, pass its exact path to `test.py`; the module layout is restored from checkpoint metadata:

```bash
python test.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir output \
  --source_dataset BTMRI \
  --checkpoint-file output/BTMRI/trained_models/seed666/HiFSA_AFPI_LCFSA_unimedclip_ViT-B-16_latest.pth \
  --seed 666

python utils/eval.py \
  --config-file configs/BTMRI_MODULAR.yaml \
  --output-dir output \
  --results-name HiFSA_AFPI_LCFSA_unimedclip_ViT-B-16 \
  --seed 666
```

Use `scripts/efficiency.sh` for one dataset/percentage and `scripts/reproduce.sh` for all supported datasets and percentages. `scripts/run_hifsa_ablation.sh` runs all eight AFPI/BFCI/LCFSA combinations.

To run the complete paper protocol for all four datasets and all four annotation ratios with seed 666:

```bash
bash scripts/reproduce.sh outputs_hifsa_reproduce
```

Checkpoints created before the Native naming migration are not compatible with this version.
