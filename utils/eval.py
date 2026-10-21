import argparse
import os
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from SurfaceDice import (
    compute_dice_coefficient,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)
from tqdm import tqdm

try:
    from utils.experiment_utils import build_results_name
    from utils.main_utils import load_cfg_from_cfg_file
except ImportError:
    # Supports: python utils/eval.py
    from experiment_utils import build_results_name
    from main_utils import load_cfg_from_cfg_file


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompt_design", type=str, default="original")
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--results-name",
        type=str,
        default="",
        help="Explicit Native result basename, useful for checkpoint-driven ablations.",
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
            "Evaluation does not accept module-switch overrides; use "
            "--results-name for an ablation result: "
            + ", ".join(sorted(requested_switches))
        )
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def strip_ext(filename):
    return os.path.splitext(filename)[0]


def main():
    cfg = get_arguments()

    gt_path = os.path.join(cfg.DATASET.TEST_PATH, "label")
    results_name = cfg.results_name or build_results_name(cfg)

    cfg.DATASET.NAME = (
        f"{cfg.DATASET.NAME}_{cfg.data_percentage}"
        if int(cfg.data_percentage) != 100
        else cfg.DATASET.NAME
    )

    seg_path = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "seg_results",
        f"seed{cfg.seed}",
        results_name + f"_Prompt-{cfg.prompt_design}",
    )

    save_path = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "seg_results",
        f"seed{cfg.seed}",
        f"{results_name}_Prompt-{cfg.prompt_design}.csv",
    )

    if not os.path.isdir(gt_path):
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_path}")

    if not os.path.isdir(seg_path):
        raise FileNotFoundError(
            f"Prediction directory not found: {seg_path}\n"
            "Make sure test.py and eval.py use the same config, tag, seed, "
            "data percentage and output directory."
        )

    valid_exts = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}

    gt_files = {
        strip_ext(f): f
        for f in os.listdir(gt_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    }

    seg_files = {
        strip_ext(f): f
        for f in os.listdir(seg_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    }

    common_names = sorted(set(gt_files) & set(seg_files))

    print(f"Found {len(common_names)} matching files")

    if not common_names:
        raise RuntimeError(
            f"No matching masks between {gt_path} and {seg_path}."
        )

    metrics = OrderedDict(Name=[], DSC=[], NSD=[])

    with tqdm(common_names) as pbar:
        for name in pbar:
            gt_name = gt_files[name]
            seg_name = seg_files[name]

            gt_mask = cv2.imread(
                os.path.join(gt_path, gt_name),
                cv2.IMREAD_GRAYSCALE,
            )

            seg_mask = cv2.imread(
                os.path.join(seg_path, seg_name),
                cv2.IMREAD_GRAYSCALE,
            )

            if gt_mask is None or seg_mask is None:
                raise RuntimeError(f"Failed to read pair: {gt_name}, {seg_name}")

            seg_mask = cv2.resize(
                seg_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            gt_data = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)[1]
            seg_data = cv2.threshold(seg_mask, 127, 255, cv2.THRESH_BINARY)[1]

            gt_data = np.uint8(gt_data)
            seg_data = np.uint8(seg_data)

            if np.max(gt_data) == 0 and np.max(seg_data) == 0:
                dsc, nsd = 1.0, 1.0

            elif np.max(gt_data) == 0 or np.max(seg_data) == 0:
                dsc, nsd = 0.0, 0.0

            else:
                gt_binary = gt_data > 0
                seg_binary = seg_data > 0

                dsc = compute_dice_coefficient(gt_binary, seg_binary)

                surface_distances = compute_surface_distances(
                    gt_binary[..., None],
                    seg_binary[..., None],
                    [1, 1, 1],
                )

                nsd = compute_surface_dice_at_tolerance(
                    surface_distances,
                    2,
                )

            metrics["Name"].append(gt_name)
            metrics["DSC"].append(round(float(dsc), 4))
            metrics["NSD"].append(round(float(nsd), 4))

            pbar.set_postfix(
                Mean_DSC=f"{np.mean(metrics['DSC']):.4f}",
                Mean_NSD=f"{np.mean(metrics['NSD']):.4f}",
            )

    dataframe = pd.DataFrame(metrics)

    # 先计算平均值，避免 Average 行参与平均
    avg_dsc = float(dataframe["DSC"].mean())
    avg_nsd = float(dataframe["NSD"].mean())

    # 在 CSV 最后一行添加平均结果
    average_row = pd.DataFrame(
        [
            {
                "Name": "Average",
                "DSC": round(avg_dsc, 4),
                "NSD": round(avg_nsd, 4),
            }
        ]
    )

    dataframe = pd.concat([dataframe, average_row], ignore_index=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataframe.to_csv(save_path, index=False)

    print(">" * 20)
    print(f"Experiment: {results_name}")
    print(f"Dataset: {cfg.DATASET.NAME}")
    print(f"Average DSC: {avg_dsc * 100:.2f}%")
    print(f"Average NSD: {avg_nsd * 100:.2f}%")
    print(f"Saved CSV: {save_path}")
    print("<" * 20)


if __name__ == "__main__":
    main()
