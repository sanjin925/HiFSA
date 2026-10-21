"""Native HiFSA defaults and validation.

The public YAML files intentionally expose only stable experiment inputs.
Dataset-specific module settings live here so all entry points resolve to one
authoritative configuration.
"""

from copy import deepcopy
from typing import Mapping


PUBLIC_CONFIG_FIELDS = {
    "DATASET": {
        "NAME",
        "TRAIN_PATH",
        "VAL_PATH",
        "TEST_PATH",
        "TEXT_PROMPT_PATH",
        "SIZE",
    },
    "TRAIN": {
        "BATCH_SIZE",
        "NUM_EPOCHS",
        "LEARNING_RATE",
        "DICE_WEIGHT",
        "CE_WEIGHT",
        "CLIP_WEIGHT",
    },
    "TEST": {"NUM_SAMPLES", "USE_LATEST"},
    "MODEL": {
        "CLIP_MODEL",
        "BACKBONE",
        "ADAPTER_DIM",
        "NUM_UPSCALE",
        "BETA",
        "GATE_INIT",
        "LAYERS",
        "TEMPERATURE",
        "DEVICE",
    },
}

MODULE_SWITCHES = ("USE_AFPI", "USE_BFCI", "USE_LCFSA")

_COMMON_DEFAULTS = {
    "TRAIN": {
        "WEIGHT_DECAY": 0.0001,
        "DROP_LAST": False,
        "GRAD_CLIP_NORM": 1.0,
    },
    "TEST": {
        "REQUIRE_FINAL_EPOCH": True,
        "CLEAR_PREDICTIONS": True,
    },
    "MODEL": {
        "USE_AFPI": True,
        "USE_BFCI": True,
        "USE_LCFSA": True,
    },
}

_DATASET_DEFAULTS = {
    "BTMRI": {
        "TRAIN": {
            "AFPI_WEIGHT_DECAY": 0.0005,
            "BFCI_WEIGHT_DECAY": 0.0005,
            "LCFSA_WEIGHT_DECAY": 0.0005,
            "LCFSA_AUX_WEIGHT": 0.01,
            "BOUNDARY_WEIGHT": 0.05,
            "CLIP_WEIGHT_END": 0.02,
            "CLIP_DECAY_START_EPOCH": 10,
            "CLIP_DECAY_END_EPOCH": 40,
        },
        "MODEL": {
            "FREQUENCY_MODE": "haar_energy",
            "FREQUENCY_GRAY": True,
            "FREQUENCY_PER_IMAGE_NORM": False,
            "FREQUENCY_INPUT_DENORM": True,
            "FREQUENCY_HIGH_STANDARDIZE": False,
            "FREQUENCY_HIGH_CLIP": 5.0,
            "AFPI_INJECT_LAYER": 3,
            "USE_AFPI_ADAPTER": False,
            "AFPI_ADAPTER_REDUCTION": 4,
            "AFPI_RESIDUAL_REDUCTION": 4,
            "AFPI_DYNAMIC_ROUTER": True,
            "AFPI_ROUTER_HIDDEN": 64,
            "AFPI_ROUTER_DROPOUT": 0.05,
            "AFPI_ROUTER_SCALE": 0.50,
            "AFPI_HIGH_WEIGHT": 0.60,
            "AFPI_LOW_WEIGHT": 0.40,
            "AFPI_GATE_INIT": 0.0,
            "AFPI_GATE_MAX": 0.80,
            "AFPI_WARMUP_EPOCHS": 10,
            "AFPI_DROPOUT": 0.05,
            "AFPI_INIT_SEED": 3101,
            "BFCI_INJECT_LAYER": 5,
            "BFCI_ATTN_DIM": 192,
            "BFCI_NUM_HEADS": 4,
            "BFCI_DROPOUT": 0.0,
            "BFCI_GATE_INIT": 0.0,
            "BFCI_GATE_MAX": 0.04,
            "BFCI_HIGH_WEIGHT": 0.55,
            "BFCI_LOW_WEIGHT": 0.45,
            "BFCI_OUTPUT_GAIN": 0.10,
            "BFCI_DETACH_CONTEXT": True,
            "BFCI_START_EPOCH": 6,
            "BFCI_WARMUP_EPOCHS": 10,
            "BFCI_INIT_SEED": 3203,
            "LCFSA_INJECT_LAYER": 8,
            "LCFSA_ATTN_DIM": 192,
            "LCFSA_NUM_HEADS": 4,
            "LCFSA_REDUCTION": 8,
            "LCFSA_DROPOUT": 0.0,
            "LCFSA_GATE_INIT": 0.0,
            "LCFSA_GATE_MAX": 0.04,
            "LCFSA_TAU": 0.25,
            "LCFSA_ZERO_INIT": True,
            "LCFSA_HIGH_WEIGHT": 0.60,
            "LCFSA_LOW_WEIGHT": 0.40,
            "LCFSA_DETACH_CONTEXT": True,
            "LCFSA_START_EPOCH": 16,
            "LCFSA_WARMUP_EPOCHS": 10,
            "LCFSA_INIT_SEED": 3407,
        },
    },
    "BUSI": {
        "TRAIN": {
            "AFPI_WEIGHT_DECAY": 0.001,
            "BFCI_WEIGHT_DECAY": 0.001,
            "LCFSA_WEIGHT_DECAY": 0.001,
            "LCFSA_AUX_WEIGHT": 0.005,
            "BOUNDARY_WEIGHT": 0.0,
            "CLIP_WEIGHT_END": 0.10,
            "CLIP_DECAY_START_EPOCH": 100,
            "CLIP_DECAY_END_EPOCH": 100,
        },
        "MODEL": {
            "FREQUENCY_MODE": "gaussian",
            "FREQUENCY_GRAY": True,
            "FREQUENCY_PER_IMAGE_NORM": False,
            "FREQUENCY_INPUT_DENORM": True,
            "FREQUENCY_HIGH_STANDARDIZE": False,
            "FREQUENCY_HIGH_CLIP": 3.0,
            "FREQUENCY_GAUSSIAN_KERNEL": 9,
            "FREQUENCY_GAUSSIAN_SIGMA": 2.5,
            "AFPI_INJECT_LAYER": 3,
            "USE_AFPI_ADAPTER": True,
            "AFPI_ADAPTER_REDUCTION": 8,
            "AFPI_RESIDUAL_REDUCTION": 8,
            "AFPI_DYNAMIC_ROUTER": True,
            "AFPI_ROUTER_HIDDEN": 48,
            "AFPI_ROUTER_DROPOUT": 0.10,
            "AFPI_ROUTER_SCALE": 0.50,
            "AFPI_HIGH_WEIGHT": 0.05,
            "AFPI_LOW_WEIGHT": 0.95,
            "AFPI_GATE_INIT": -3.0,
            "AFPI_GATE_MAX": 0.08,
            "AFPI_WARMUP_EPOCHS": 20,
            "AFPI_DROPOUT": 0.20,
            "AFPI_INIT_SEED": 3101,
            "BFCI_INJECT_LAYER": 6,
            "BFCI_ATTN_DIM": 128,
            "BFCI_NUM_HEADS": 4,
            "BFCI_DROPOUT": 0.05,
            "BFCI_GATE_INIT": 0.0,
            "BFCI_GATE_MAX": 0.015,
            "BFCI_HIGH_WEIGHT": 0.10,
            "BFCI_LOW_WEIGHT": 0.90,
            "BFCI_OUTPUT_GAIN": 0.05,
            "BFCI_DETACH_CONTEXT": True,
            "BFCI_START_EPOCH": 11,
            "BFCI_WARMUP_EPOCHS": 20,
            "BFCI_INIT_SEED": 3203,
            "LCFSA_INJECT_LAYER": 8,
            "LCFSA_ATTN_DIM": 128,
            "LCFSA_NUM_HEADS": 4,
            "LCFSA_REDUCTION": 16,
            "LCFSA_DROPOUT": 0.05,
            "LCFSA_GATE_INIT": 0.0,
            "LCFSA_GATE_MAX": 0.02,
            "LCFSA_TAU": 0.35,
            "LCFSA_ZERO_INIT": True,
            "LCFSA_HIGH_WEIGHT": 0.10,
            "LCFSA_LOW_WEIGHT": 0.90,
            "LCFSA_DETACH_CONTEXT": True,
            "LCFSA_START_EPOCH": 21,
            "LCFSA_WARMUP_EPOCHS": 20,
            "LCFSA_INIT_SEED": 3407,
        },
    },
    "Covid19": {
        "TRAIN": {
            "AFPI_WEIGHT_DECAY": 0.0005,
            "BFCI_WEIGHT_DECAY": 0.0005,
            "LCFSA_WEIGHT_DECAY": 0.0005,
            "LCFSA_AUX_WEIGHT": 0.01,
            "BOUNDARY_WEIGHT": 0.03,
            "CLIP_WEIGHT_END": 0.02,
            "CLIP_DECAY_START_EPOCH": 10,
            "CLIP_DECAY_END_EPOCH": 40,
        },
        "MODEL": {
            "FREQUENCY_MODE": "gray_normalized",
            "FREQUENCY_GRAY": True,
            "FREQUENCY_PER_IMAGE_NORM": True,
            "FREQUENCY_INPUT_DENORM": True,
            "FREQUENCY_HIGH_TYPE": "energy",
            "FREQUENCY_HIGH_STANDARDIZE": False,
            "FREQUENCY_HIGH_CLIP": 5.0,
            "AFPI_INJECT_LAYER": 4,
            "USE_AFPI_ADAPTER": False,
            "AFPI_ADAPTER_REDUCTION": 4,
            "AFPI_RESIDUAL_REDUCTION": 4,
            "AFPI_DYNAMIC_ROUTER": True,
            "AFPI_ROUTER_HIDDEN": 64,
            "AFPI_ROUTER_DROPOUT": 0.05,
            "AFPI_ROUTER_SCALE": 0.40,
            "AFPI_HIGH_WEIGHT": 0.20,
            "AFPI_LOW_WEIGHT": 0.80,
            "AFPI_GATE_INIT": -0.5,
            "AFPI_GATE_MAX": 0.50,
            "AFPI_WARMUP_EPOCHS": 10,
            "AFPI_DROPOUT": 0.05,
            "AFPI_INIT_SEED": 3101,
            "BFCI_INJECT_LAYER": 6,
            "BFCI_ATTN_DIM": 192,
            "BFCI_NUM_HEADS": 4,
            "BFCI_DROPOUT": 0.0,
            "BFCI_GATE_INIT": 0.0,
            "BFCI_GATE_MAX": 0.03,
            "BFCI_HIGH_WEIGHT": 0.35,
            "BFCI_LOW_WEIGHT": 0.65,
            "BFCI_OUTPUT_GAIN": 0.08,
            "BFCI_DETACH_CONTEXT": True,
            "BFCI_START_EPOCH": 6,
            "BFCI_WARMUP_EPOCHS": 10,
            "BFCI_INIT_SEED": 3203,
            "LCFSA_INJECT_LAYER": 8,
            "LCFSA_ATTN_DIM": 192,
            "LCFSA_NUM_HEADS": 4,
            "LCFSA_REDUCTION": 8,
            "LCFSA_DROPOUT": 0.0,
            "LCFSA_GATE_INIT": 0.0,
            "LCFSA_GATE_MAX": 0.03,
            "LCFSA_TAU": 0.30,
            "LCFSA_ZERO_INIT": True,
            "LCFSA_HIGH_WEIGHT": 0.25,
            "LCFSA_LOW_WEIGHT": 0.75,
            "LCFSA_DETACH_CONTEXT": True,
            "LCFSA_START_EPOCH": 16,
            "LCFSA_WARMUP_EPOCHS": 10,
            "LCFSA_INIT_SEED": 3407,
        },
    },
    "Kvasir": {
        "TRAIN": {
            "AFPI_WEIGHT_DECAY": 0.0005,
            "BFCI_WEIGHT_DECAY": 0.0005,
            "LCFSA_WEIGHT_DECAY": 0.0005,
            "LCFSA_AUX_WEIGHT": 0.015,
            "BOUNDARY_WEIGHT": 0.05,
            "CLIP_WEIGHT_END": 0.02,
            "CLIP_DECAY_START_EPOCH": 10,
            "CLIP_DECAY_END_EPOCH": 40,
        },
        "MODEL": {
            "FREQUENCY_MODE": "haar_energy",
            "FREQUENCY_GRAY": False,
            "FREQUENCY_PER_IMAGE_NORM": False,
            "FREQUENCY_INPUT_DENORM": True,
            "FREQUENCY_HIGH_STANDARDIZE": False,
            "FREQUENCY_HIGH_CLIP": 5.0,
            "AFPI_INJECT_LAYER": 3,
            "USE_AFPI_ADAPTER": False,
            "AFPI_ADAPTER_REDUCTION": 4,
            "AFPI_RESIDUAL_REDUCTION": 4,
            "AFPI_DYNAMIC_ROUTER": True,
            "AFPI_ROUTER_HIDDEN": 64,
            "AFPI_ROUTER_DROPOUT": 0.05,
            "AFPI_ROUTER_SCALE": 0.40,
            "AFPI_HIGH_WEIGHT": 0.50,
            "AFPI_LOW_WEIGHT": 0.50,
            "AFPI_GATE_INIT": 0.0,
            "AFPI_GATE_MAX": 0.80,
            "AFPI_WARMUP_EPOCHS": 10,
            "AFPI_DROPOUT": 0.05,
            "AFPI_INIT_SEED": 3101,
            "BFCI_INJECT_LAYER": 5,
            "BFCI_ATTN_DIM": 256,
            "BFCI_NUM_HEADS": 8,
            "BFCI_DROPOUT": 0.0,
            "BFCI_GATE_INIT": 0.0,
            "BFCI_GATE_MAX": 0.05,
            "BFCI_HIGH_WEIGHT": 0.55,
            "BFCI_LOW_WEIGHT": 0.45,
            "BFCI_OUTPUT_GAIN": 0.10,
            "BFCI_DETACH_CONTEXT": True,
            "BFCI_START_EPOCH": 6,
            "BFCI_WARMUP_EPOCHS": 10,
            "BFCI_INIT_SEED": 3203,
            "LCFSA_INJECT_LAYER": 8,
            "LCFSA_ATTN_DIM": 192,
            "LCFSA_NUM_HEADS": 4,
            "LCFSA_REDUCTION": 8,
            "LCFSA_DROPOUT": 0.0,
            "LCFSA_GATE_INIT": 0.0,
            "LCFSA_GATE_MAX": 0.05,
            "LCFSA_TAU": 0.20,
            "LCFSA_ZERO_INIT": True,
            "LCFSA_HIGH_WEIGHT": 0.50,
            "LCFSA_LOW_WEIGHT": 0.50,
            "LCFSA_DETACH_CONTEXT": True,
            "LCFSA_START_EPOCH": 11,
            "LCFSA_WARMUP_EPOCHS": 10,
            "LCFSA_INIT_SEED": 3407,
        },
    },
}


def _deep_merge(base: dict, override: Mapping[str, object]) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def build_hifsa_defaults(dataset_name: str) -> dict:
    if dataset_name not in _DATASET_DEFAULTS:
        supported = ", ".join(_DATASET_DEFAULTS)
        raise ValueError(
            f"Unsupported DATASET.NAME={dataset_name!r}; expected one of: {supported}"
        )
    return _deep_merge(_COMMON_DEFAULTS, _DATASET_DEFAULTS[dataset_name])


def merge_config(base: Mapping[str, object], override: Mapping[str, object]) -> dict:
    return _deep_merge(dict(base), override)


def validate_public_config(raw_cfg: Mapping[str, object]) -> None:
    if not isinstance(raw_cfg, Mapping):
        raise TypeError("The YAML root must be a mapping")
    unknown_sections = set(raw_cfg) - set(PUBLIC_CONFIG_FIELDS)
    if unknown_sections:
        raise KeyError(f"Unknown configuration section(s): {sorted(unknown_sections)}")
    for section, allowed_fields in PUBLIC_CONFIG_FIELDS.items():
        values = raw_cfg.get(section)
        if not isinstance(values, Mapping):
            raise TypeError(f"{section} must be a mapping, got {type(values).__name__}")
        unknown_fields = set(values) - allowed_fields
        if unknown_fields:
            paths = ", ".join(f"{section}.{name}" for name in sorted(unknown_fields))
            raise KeyError(
                f"Fields are not part of the minimal YAML interface: {paths}. "
                "Module switches must be supplied on the training command line."
            )


def validate_resolved_config(cfg: Mapping[str, object]) -> None:
    model = cfg["MODEL"]
    for name in MODULE_SWITCHES:
        value = model[name]
        if type(value) is not bool:
            raise TypeError(f"MODEL.{name} must be bool, got {value!r}")

    layers = [
        int(model["AFPI_INJECT_LAYER"]),
        int(model["BFCI_INJECT_LAYER"]),
        int(model["LCFSA_INJECT_LAYER"]),
    ]
    if not layers[0] < layers[1] < layers[2]:
        raise ValueError(
            "MODEL injection layers must satisfy "
            f"AFPI < BFCI < LCFSA, got {layers}"
        )

