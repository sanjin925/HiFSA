import argparse
import logging
import os
import random
from statistics import mean

import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from trainers import *
from utils.checkpoint_utils import (
    build_hifsa_metadata,
    validate_hifsa_metadata,
)
from utils.experiment_utils import build_results_name
from utils.main_utils import dump_resolved_config, load_cfg_from_cfg_file, read_text


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--prompt_design", type=str, default="original")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
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


def resolve_text_path(text_prompt_path, split, prompt_design="original", data_percentage=100):
    split = split.strip()
    data_percentage = int(data_percentage)
    candidates = []

    if split in {"Train", "Val"}:
        if data_percentage != 100:
            candidates.extend(
                [
                    os.path.join(text_prompt_path, f"{split}_text_{data_percentage}.xlsx"),
                    os.path.join(text_prompt_path, f"{split}_text_{data_percentage}%.xlsx"),
                    os.path.join(
                        text_prompt_path,
                        f"{split}_text_{data_percentage}_percent.xlsx",
                    ),
                ]
            )
        candidates.extend(
            [
                os.path.join(text_prompt_path, f"{split}_text.xlsx"),
                os.path.join(text_prompt_path, f"{split}_text_original.xlsx"),
            ]
        )
    elif split == "Test":
        candidates.extend(
            [
                os.path.join(text_prompt_path, f"Test_text_{prompt_design}.xlsx"),
                os.path.join(text_prompt_path, "Test_text.xlsx"),
                os.path.join(text_prompt_path, "Test_text_original.xlsx"),
            ]
        )
    else:
        raise ValueError(f"Unknown split: {split}")

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Cannot find text file.\n"
        f"split={split}, prompt_design={prompt_design}, "
        f"data_percentage={data_percentage}\nTried:\n"
        + "\n".join(candidates)
    )


def get_mask_from_batch(batch):
    if "ground_truth_mask" in batch:
        return batch["ground_truth_mask"]
    if "mask" in batch:
        return batch["mask"]
    raise KeyError(f"Cannot find mask. Available keys: {list(batch.keys())}")


def prepare_mask(mask, device):
    mask = mask.to(device)
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    return mask.float()


def dice_values_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)

    both_empty = (preds.sum(dim=1) == 0) & (targets.sum(dim=1) == 0)
    dice[both_empty] = 1.0
    return dice


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_dice,
    metadata,
    path,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_dice": val_dice,
            "hifsa_config": dict(metadata),
        },
        path,
    )


def log_hifsa_state(model, logger, epoch=None):
    if not hasattr(model, "get_hifsa_state"):
        return

    state = model.get_hifsa_state()
    if not state.get("enabled", False):
        return

    prefix = "[Modules]" if epoch is None else f"[Modules][Epoch {epoch}]"
    parts = [
        f"{prefix} mode={state['mode']}",
        f"AFPI={state.get('use_afpi', False)}",
        f"BFCI={state.get('use_bfci', False)}",
        f"LCFSA={state.get('use_lcfsa', False)}",
    ]

    if state.get("use_afpi", False):
        parts.extend(
            [
                f"afpi_layer={state.get('afpi_inject_layer', -1)}",
                f"afpi_progress={state.get('afpi_progress', 0.0):.3f}",
                f"afpi_mix={state.get('afpi_mix_high', 0.0):.4f}/"
                f"{state.get('afpi_mix_low', 0.0):.4f}",
                f"afpi_gate={state.get('afpi_gate_effective', 0.0):.6f}",
            ]
        )
        if "afpi_dynamic_gate" in state:
            parts.extend(
                [
                    f"dyn_mix={state['afpi_dynamic_mix_high']:.4f}/"
                    f"{state['afpi_dynamic_mix_low']:.4f}",
                    f"dyn_gate={state['afpi_dynamic_gate']:.6f}",
                ]
            )

    if state.get("use_bfci", False):
        parts.extend(
            [
                f"bfci_layer={state.get('bfci_inject_layer', -1)}",
                f"bfci_progress={state.get('bfci_progress', 0.0):.3f}",
                f"bfci_gate={state.get('bfci_gate', 0.0):.6f}",
                f"bfci_mix={state.get('bfci_mix_high', 0.0):.4f}/"
                f"{state.get('bfci_mix_low', 0.0):.4f}",
                f"bfci_rms={state.get('bfci_residual_rms', 0.0):.6f}",
            ]
        )

    if state.get("use_lcfsa", False):
        parts.extend(
            [
                f"lcfsa_layer={state.get('lcfsa_inject_layer', -1)}",
                f"lcfsa_progress={state.get('lcfsa_progress', 0.0):.3f}",
                f"lcfsa_gate={state.get('lcfsa_gate', 0.0):.6f}",
                f"lcfsa_mix={state.get('lcfsa_base_mix_high', 0.0):.4f}/"
                f"{state.get('lcfsa_base_mix_low', 0.0):.4f}",
                f"lcfsa_rms={state.get('lcfsa_residual_rms', 0.0):.6f}",
            ]
        )
        if "lcfsa_high_spatial_mean" in state:
            parts.extend(
                [
                    f"lcfsa_h_sp={state['lcfsa_high_spatial_mean']:.4f}",
                    f"lcfsa_l_sp={state['lcfsa_low_spatial_mean']:.4f}",
                ]
            )

    logger.info(" | ".join(parts))


class SoftBoundaryLoss(nn.Module):
    """Differentiable Sobel-edge loss for improving boundary alignment/NSD."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _edge(self, x):
        gx = F.conv2d(x, self.sobel_x.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(dtype=x.dtype), padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        return edge.clamp(0.0, 4.0) / 4.0

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if probs.dim() == 3:
            probs = probs.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        return F.smooth_l1_loss(self._edge(probs), self._edge(targets))


def get_clip_weight(cfg, epoch):
    start = float(getattr(cfg.TRAIN, "CLIP_WEIGHT", 0.1))
    end = float(getattr(cfg.TRAIN, "CLIP_WEIGHT_END", start))
    decay_start = int(getattr(cfg.TRAIN, "CLIP_DECAY_START_EPOCH", 1))
    decay_end = int(
        getattr(cfg.TRAIN, "CLIP_DECAY_END_EPOCH", cfg.TRAIN.NUM_EPOCHS)
    )
    if decay_end <= decay_start or epoch <= decay_start:
        return start
    if epoch >= decay_end:
        return end
    ratio = (epoch - decay_start) / float(decay_end - decay_start)
    return start + ratio * (end - start)


def set_afpi_warmup(model, cfg, epoch):
    if not hasattr(model, "set_afpi_progress"):
        return 0.0
    if not bool(getattr(cfg.MODEL, "USE_AFPI", False)):
        return 0.0
    warmup_epochs = int(getattr(cfg.MODEL, "AFPI_WARMUP_EPOCHS", 0))
    progress = 1.0 if warmup_epochs <= 0 else min(epoch / warmup_epochs, 1.0)
    model.set_afpi_progress(progress)
    return progress


def set_bfci_warmup(model, cfg, epoch):
    """Independent BFCI delayed linear warm-up."""
    if not hasattr(model, "set_bfci_progress"):
        return 0.0
    if not bool(getattr(cfg.MODEL, "USE_BFCI", False)):
        return 0.0

    start_epoch = int(getattr(cfg.MODEL, "BFCI_START_EPOCH", 1))
    warmup_epochs = int(getattr(cfg.MODEL, "BFCI_WARMUP_EPOCHS", 10))
    if epoch < start_epoch:
        progress = 0.0
    elif warmup_epochs <= 0:
        progress = 1.0
    else:
        progress = min(
            (epoch - start_epoch + 1) / float(warmup_epochs),
            1.0,
        )
    model.set_bfci_progress(progress)
    return progress


def set_lcfsa_warmup(model, cfg, epoch):
    """
    Delay LCFSA until the AFPI/BFCI and PVL streams have reached a stable state.

    LCFSA_START_EPOCH is inclusive. LCFSA_WARMUP_EPOCHS controls the linear
    ramp after the start epoch.
    """
    if not hasattr(model, "set_lcfsa_progress"):
        return 0.0
    if not bool(getattr(cfg.MODEL, "USE_LCFSA", False)):
        return 0.0

    start_epoch = int(getattr(cfg.MODEL, "LCFSA_START_EPOCH", 1))
    warmup_epochs = int(getattr(cfg.MODEL, "LCFSA_WARMUP_EPOCHS", 10))
    if epoch < start_epoch:
        progress = 0.0
    elif warmup_epochs <= 0:
        progress = 1.0
    else:
        progress = min(
            (epoch - start_epoch + 1) / float(warmup_epochs),
            1.0,
        )
    model.set_lcfsa_progress(progress)
    return progress


def build_optimizer(model, cfg, logger):
    """Build disjoint optimizer groups for baseline, AFPI, BFCI and LCFSA."""
    groups_by_name = {
        "base": [],
        "afpi": [],
        "afpi_scalar": [],
        "bfci": [],
        "bfci_scalar": [],
        "lcfsa": [],
        "lcfsa_scalar": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in {"afpi_gate", "afpi_mix_logits"}:
            groups_by_name["afpi_scalar"].append(param)
        elif name.startswith("afpi_modules."):
            groups_by_name["afpi"].append(param)
        elif name in {
            "bfci_branch.gate_logit",
            "bfci_branch.mix_logits",
        }:
            groups_by_name["bfci_scalar"].append(param)
        elif name.startswith("bfci_branch."):
            groups_by_name["bfci"].append(param)
        elif name in {"lcfsa.gate_logit", "lcfsa_mix_logits"}:
            groups_by_name["lcfsa_scalar"].append(param)
        elif name.startswith("lcfsa."):
            groups_by_name["lcfsa"].append(param)
        else:
            groups_by_name["base"].append(param)

    base_lr = float(cfg.TRAIN.LEARNING_RATE)
    base_wd = float(getattr(cfg.TRAIN, "WEIGHT_DECAY", 1e-4))

    afpi_wd = float(getattr(cfg.TRAIN, "AFPI_WEIGHT_DECAY", base_wd))

    bfci_wd = float(getattr(cfg.TRAIN, "BFCI_WEIGHT_DECAY", afpi_wd))

    lcfsa_wd = float(getattr(cfg.TRAIN, "LCFSA_WEIGHT_DECAY", afpi_wd))

    specs = {
        "base": (base_lr, base_wd),
        "afpi": (base_lr, afpi_wd),
        "afpi_scalar": (base_lr, 0.0),
        "bfci": (base_lr, bfci_wd),
        "bfci_scalar": (base_lr, 0.0),
        "lcfsa": (base_lr, lcfsa_wd),
        "lcfsa_scalar": (base_lr, 0.0),
    }

    optimizer_groups = []
    for group_name, params in groups_by_name.items():
        if not params:
            continue
        lr, wd = specs[group_name]
        optimizer_groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": wd,
                "group_name": group_name,
            }
        )

    optimizer = torch.optim.AdamW(optimizer_groups)
    for group in optimizer.param_groups:
        logger.info(
            f"Optimizer group {group['group_name']}: "
            f"lr={group['lr']:.6g}, wd={group['weight_decay']:.6g}, "
            f"tensors={len(group['params'])}"
        )
    return optimizer


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    dice_loss_fn,
    bce_loss_fn,
    boundary_loss_fn,
    cfg,
    epoch,
    logger,
):
    model.train()
    afpi_progress = set_afpi_warmup(model, cfg, epoch)
    bfci_progress = set_bfci_warmup(model, cfg, epoch)
    lcfsa_progress = set_lcfsa_warmup(model, cfg, epoch)
    clip_weight = get_clip_weight(cfg, epoch)
    boundary_weight = float(getattr(cfg.TRAIN, "BOUNDARY_WEIGHT", 0.0))

    total_losses = []
    dice_losses = []
    bce_losses = []
    clip_losses = []
    boundary_losses = []
    lcfsa_aux_losses = []
    dice_values = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch["image"].to(cfg.MODEL.DEVICE, non_blocking=True)
        masks = prepare_mask(
            get_mask_from_batch(batch),
            cfg.MODEL.DEVICE,
        )
        texts = batch["text_prompt"]

        optimizer.zero_grad(set_to_none=True)
        seg_logits, clip_loss = model(image=images, text=texts)

        lcfsa_aux_weight = float(
            getattr(cfg.TRAIN, "LCFSA_AUX_WEIGHT", 0.0)
        )
        if lcfsa_aux_weight > 0.0 and hasattr(model, "get_lcfsa_aux_loss"):
            loss_lcfsa_aux = model.get_lcfsa_aux_loss(masks)
        else:
            loss_lcfsa_aux = torch.zeros((), device=cfg.MODEL.DEVICE)

        seg_logits_for_loss = (
            seg_logits if seg_logits.dim() == 4 else seg_logits.unsqueeze(1)
        )
        masks_for_loss = masks if masks.dim() == 4 else masks.unsqueeze(1)

        loss_dice = dice_loss_fn(seg_logits_for_loss, masks_for_loss)
        loss_bce = bce_loss_fn(seg_logits, masks)

        if clip_loss is None:
            clip_loss = torch.zeros((), device=cfg.MODEL.DEVICE)

        if boundary_weight > 0.0:
            loss_boundary = boundary_loss_fn(seg_logits, masks)
        else:
            loss_boundary = torch.zeros((), device=cfg.MODEL.DEVICE)

        total_loss = (
            cfg.TRAIN.DICE_WEIGHT * loss_dice
            + cfg.TRAIN.CE_WEIGHT * loss_bce
            + clip_weight * clip_loss
            + boundary_weight * loss_boundary
            + lcfsa_aux_weight * lcfsa_progress * loss_lcfsa_aux
        )

        total_loss.backward()
        grad_clip = float(getattr(cfg.TRAIN, "GRAD_CLIP_NORM", 0.0))
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )
        optimizer.step()

        with torch.no_grad():
            batch_dice_values = dice_values_from_logits(seg_logits, masks)

        total_losses.append(total_loss.item())
        dice_losses.append(loss_dice.item())
        bce_losses.append(loss_bce.item())
        clip_losses.append(clip_loss.item())
        boundary_losses.append(loss_boundary.item())
        lcfsa_aux_losses.append(loss_lcfsa_aux.item())
        dice_values.extend(batch_dice_values.detach().cpu().tolist())

        pbar.set_postfix(
            loss=f"{mean(total_losses):.4f}",
            dice=f"{mean(dice_values):.4f}",
        )

    logger.info(
        f"[Train] Epoch {epoch} | Loss: {mean(total_losses):.4f} | "
        f"DiceLoss: {mean(dice_losses):.4f} | "
        f"BCELoss: {mean(bce_losses):.4f} | "
        f"CLIPLoss: {mean(clip_losses):.4f} | "
        f"BoundaryLoss: {mean(boundary_losses):.4f} | "
        f"LCFSAAuxLoss: {mean(lcfsa_aux_losses):.4f} | "
        f"LCFSAAuxWeight: {float(getattr(cfg.TRAIN, 'LCFSA_AUX_WEIGHT', 0.0)):.4f} | "
        f"CLIPWeight: {clip_weight:.4f} | "
        f"AFPIProgress: {afpi_progress:.3f} | "
        f"BFCIProgress: {bfci_progress:.3f} | "
        f"LCFSAProgress: {lcfsa_progress:.3f} | "
        f"Dice: {mean(dice_values):.4f}"
    )
    return mean(total_losses), mean(dice_values)


@torch.no_grad()
def validate(model, val_loader, cfg, epoch, logger):
    model.eval()
    dice_values = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    for batch in pbar:
        images = batch["image"].to(cfg.MODEL.DEVICE, non_blocking=True)
        masks = prepare_mask(
            get_mask_from_batch(batch),
            cfg.MODEL.DEVICE,
        )
        texts = batch["text_prompt"]

        seg_samples = model(image=images, text=texts, num_samples=1)
        seg_logits = seg_samples.mean(dim=0)

        batch_values = dice_values_from_logits(seg_logits, masks)
        dice_values.extend(batch_values.detach().cpu().tolist())
        pbar.set_postfix(dice=f"{mean(dice_values):.4f}")

    val_dice = mean(dice_values)
    logger.info(f"[Val] Epoch {epoch} | Dice: {val_dice:.4f}")
    return val_dice


def main():
    cfg = get_arguments()

    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)

    base_dataset_name = cfg.DATASET.NAME
    output_dataset_name = (
        f"{base_dataset_name}_{cfg.data_percentage}"
        if int(cfg.data_percentage) != 100
        else base_dataset_name
    )
    cfg.DATASET.NAME = output_dataset_name

    results_name = build_results_name(cfg)
    output_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME)
    model_save_dir = os.path.join(
        output_root,
        "trained_models",
        f"seed{cfg.seed}",
    )
    log_dir = os.path.join(output_root, "logs", f"seed{cfg.seed}")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_metadata = build_hifsa_metadata(cfg, base_dataset_name)
    dump_resolved_config(
        cfg,
        os.path.join(model_save_dir, f"{results_name}_resolved.yaml"),
    )

    logger = logger_config(os.path.join(log_dir, f"{results_name}.txt"))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)
    logger.info(f"Base dataset name: {base_dataset_name}")
    logger.info(f"Output dataset name: {cfg.DATASET.NAME}")
    logger.info(f"Results name: {results_name}")

    model = build_model(cfg).to(cfg.MODEL.DEVICE)
    log_hifsa_state(model, logger)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = 0
    logger.info("**************")
    logger.info("Trainable parameters")
    logger.info("**************")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.numel()}")
            total_trainable += param.numel()
    logger.info(f"Total trainable params: {total_trainable}")

    optimizer = build_optimizer(model, cfg, logger)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.TRAIN.NUM_EPOCHS,
        eta_min=1e-6,
    )

    start_epoch = 1
    latest_path = os.path.join(
        model_save_dir,
        f"{results_name}_latest.pth",
    )

    if cfg.resume and os.path.exists(latest_path):
        logger.info(f"Resuming from: {latest_path}")
        checkpoint = torch.load(
            latest_path,
            map_location=cfg.MODEL.DEVICE,
            weights_only=False,
        )
        validate_hifsa_metadata(
            checkpoint.get("hifsa_config"),
            cfg,
            base_dataset_name,
            require_module_match=True,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resume epoch: {start_epoch}")
    elif cfg.resume:
        logger.info(f"No checkpoint found at {latest_path}; training from scratch.")

    train_tf = RandomGenerator(
        output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE]
    )
    val_tf = ValGenerator(
        output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE]
    )

    train_text_path = resolve_text_path(
        cfg.DATASET.TEXT_PROMPT_PATH,
        "Train",
        cfg.prompt_design,
        cfg.data_percentage,
    )
    val_text_path = resolve_text_path(
        cfg.DATASET.TEXT_PROMPT_PATH,
        "Val",
        cfg.prompt_design,
        cfg.data_percentage,
    )
    logger.info(f"Loading train text from: {train_text_path}")
    logger.info(f"Loading val text from: {val_text_path}")

    train_dataset = DatasetSegmentation(
        cfg.DATASET.TRAIN_PATH,
        base_dataset_name,
        read_text(train_text_path),
        train_tf,
        image_size=cfg.DATASET.SIZE,
    )
    val_dataset = DatasetSegmentation(
        cfg.DATASET.VAL_PATH,
        base_dataset_name,
        read_text(val_text_path),
        val_tf,
        image_size=cfg.DATASET.SIZE,
    )

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=bool(getattr(cfg.TRAIN, "DROP_LAST", False)),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    dice_loss_fn = monai.losses.DiceLoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
    )
    bce_loss_fn = BCEWithLogitsLoss()
    boundary_loss_fn = SoftBoundaryLoss().to(cfg.MODEL.DEVICE)

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            dice_loss_fn,
            bce_loss_fn,
            boundary_loss_fn,
            cfg,
            epoch,
            logger,
        )
        val_dice = validate(model, val_loader, cfg, epoch, logger)
        scheduler.step()
        log_hifsa_state(model, logger, epoch)

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_dice,
            checkpoint_metadata,
            latest_path,
        )
        logger.info(
            f"Epoch {epoch} finished. Latest saved to {latest_path}. "
            f"Validation dice: {val_dice:.4f}"
        )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
