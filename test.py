import argparse
import logging
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import *
from utils.checkpoint_utils import (
    apply_checkpoint_module_state,
    validate_hifsa_metadata,
)
from utils.experiment_utils import build_results_name
from utils.main_utils import load_cfg_from_cfg_file, normalize, read_text


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompt_design", type=str, default="original")
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="",
        help="Dataset name used to locate the trained checkpoint.",
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="",
        help="Explicit checkpoint path, required to auto-restore an ablation layout.",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    module_switches = {
        "MODEL.USE_AFPI",
        "MODEL.USE_BFCI",
        "MODEL.USE_LCFSA",
    }
    requested_switches = module_switches.intersection((args.opts or [])[0::2])
    if requested_switches:
        raise ValueError(
            "Test-time module switches come from checkpoint metadata; "
            "do not override: " + ", ".join(sorted(requested_switches))
        )
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="UTF-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def build_model(cfg):
    if cfg.MODEL.CLIP_MODEL == "unimedclip":
        return build_hifsa_unimedclip(cfg)
    if cfg.MODEL.CLIP_MODEL == "biomedclip":
        return build_medclipseg_biomedclip(cfg)
    if cfg.MODEL.CLIP_MODEL == "clip":
        return build_medclipseg_clip(cfg)
    if cfg.MODEL.CLIP_MODEL == "pubmedclip":
        return build_medclipseg_pubmedclip(cfg)
    raise ValueError(f"Unsupported CLIP model: {cfg.MODEL.CLIP_MODEL}")


def resolve_test_text(text_prompt_path, prompt_design):
    candidates = [
        os.path.join(text_prompt_path, f"Test_text_{prompt_design}.xlsx"),
        os.path.join(text_prompt_path, "Test_text.xlsx"),
        os.path.join(text_prompt_path, "Test_text_original.xlsx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Cannot find test prompt file. Tried:\n" + "\n".join(candidates))


def main():
    cfg = get_arguments()
    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)

    base_target_dataset = cfg.DATASET.NAME
    target_output_name = (
        f"{base_target_dataset}_{cfg.data_percentage}"
        if int(cfg.data_percentage) != 100
        else base_target_dataset
    )
    cfg.DATASET.NAME = target_output_name

    source_dataset = cfg.source_dataset or base_target_dataset
    source_output_name = (
        f"{source_dataset}_{cfg.data_percentage}"
        if int(cfg.data_percentage) != 100
        else source_dataset
    )

    if not bool(cfg.TEST.USE_LATEST):
        raise ValueError(
            "TEST.USE_LATEST must be True for the paper protocol; "
            "best-validation checkpoint selection is not supported."
        )
    checkpoint_type = "latest"
    if cfg.checkpoint_file:
        checkpoint_path = os.path.abspath(cfg.checkpoint_file)
    else:
        default_results_name = build_results_name(cfg)
        checkpoint_path = os.path.join(
            cfg.output_dir,
            source_output_name,
            "trained_models",
            f"seed{cfg.seed}",
            f"{default_results_name}_{checkpoint_type}.pth",
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Check the checkpoint path, seed, data percentage, source dataset "
            "and output directory. Use --checkpoint-file for an ablation."
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg.MODEL.DEVICE,
        weights_only=False,
    )
    validate_hifsa_metadata(
        checkpoint.get("hifsa_config"),
        cfg,
        source_dataset,
        require_module_match=False,
    )
    apply_checkpoint_module_state(cfg, checkpoint["hifsa_config"])

    results_name = build_results_name(cfg)
    results_root = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "seg_results",
        f"seed{cfg.seed}",
    )
    os.makedirs(results_root, exist_ok=True)
    logger = logger_config(os.path.join(results_root, f"{results_name}.log"))

    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)
    logger.info(f"Results name: {results_name}")
    logger.info(f"Checkpoint dataset directory: {source_output_name}")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    model = build_model(cfg)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    logger.info(f"Loaded checkpoint epoch: {checkpoint_epoch}")

    require_final = bool(getattr(cfg.TEST, "REQUIRE_FINAL_EPOCH", True))
    if require_final:
        expected_epoch = int(cfg.TRAIN.NUM_EPOCHS)
        if checkpoint_epoch != expected_epoch:
            raise RuntimeError(
                f"The latest checkpoint is epoch {checkpoint_epoch}, but the "
                f"configured final epoch is {expected_epoch}. Complete training "
                "before the paper-protocol test, or explicitly set "
                "TEST.REQUIRE_FINAL_EPOCH: False for debugging only."
            )

    model.load_state_dict(checkpoint["model"], strict=True)
    if hasattr(model, "set_afpi_progress"):
        model.set_afpi_progress(1.0)
    if hasattr(model, "set_bfci_progress"):
        model.set_bfci_progress(1.0)
    if hasattr(model, "set_lcfsa_progress"):
        model.set_lcfsa_progress(1.0)
    model.eval().to(cfg.MODEL.DEVICE)

    if hasattr(model, "get_hifsa_state"):
        logger.info(f"Frequency state: {model.get_hifsa_state()}")

    test_text_path = resolve_test_text(
        cfg.DATASET.TEXT_PROMPT_PATH,
        cfg.prompt_design,
    )
    logger.info(f"Loading test text from: {test_text_path}")

    test_dataset = DatasetSegmentation(
        cfg.DATASET.TEST_PATH,
        base_target_dataset,
        read_text(test_text_path),
        ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE]),
        image_size=cfg.DATASET.SIZE,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    save_dir = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "seg_results",
        f"seed{cfg.seed}",
        results_name + f"_Prompt-{cfg.prompt_design}",
    )
    save_unc_dir = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "unc_results",
        f"seed{cfg.seed}",
        results_name + f"_Prompt-{cfg.prompt_design}",
    )
    clear_predictions = bool(
        getattr(cfg.TEST, "CLEAR_PREDICTIONS", True)
    )
    for path in (save_dir, save_unc_dir):
        if clear_predictions and os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            seg_samples = model(
                image=batch["image"].to(
                    cfg.MODEL.DEVICE,
                    non_blocking=True,
                ),
                text=batch["text_prompt"],
                num_samples=cfg.TEST.NUM_SAMPLES,
            )
            seg_samples = torch.sigmoid(seg_samples)
            predictive_mean = seg_samples.mean(dim=0)
            predictive_entropy = -(
                predictive_mean * torch.log(predictive_mean + 1e-8)
                + (1.0 - predictive_mean)
                * torch.log(1.0 - predictive_mean + 1e-8)
            )
            mask_preds = predictive_mean > 0.5

            for i, mask_name in enumerate(batch["mask_name"]):
                binary_pred = mask_preds[i].cpu().numpy().astype(np.uint8)
                cv2.imwrite(
                    os.path.join(save_dir, mask_name),
                    binary_pred * 255,
                )

                uncertainty = normalize(
                    predictive_entropy[i].cpu().numpy()
                )
                colormap = plt.get_cmap("nipy_spectral")
                color = (colormap(uncertainty)[:, :, :3] * 255).astype(np.uint8)
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_unc_dir, mask_name), color)

    logger.info(f"Predictions saved to: {save_dir}")
    logger.info(f"Uncertainty maps saved to: {save_unc_dir}")


if __name__ == "__main__":
    main()
