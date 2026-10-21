"""Checkpoint metadata for the Native HiFSA configuration contract."""

from collections.abc import Mapping


HIFSA_CONFIG_SCHEMA_VERSION = 1
_MODULE_KEYS = ("USE_AFPI", "USE_BFCI", "USE_LCFSA")


def build_hifsa_metadata(cfg, dataset_name):
    """Capture architecture-defining values alongside a checkpoint."""
    return {
        "schema_version": HIFSA_CONFIG_SCHEMA_VERSION,
        "dataset": str(dataset_name),
        "clip_model": str(cfg.MODEL.CLIP_MODEL),
        "backbone": str(cfg.MODEL.BACKBONE),
        "modules": {
            key: bool(getattr(cfg.MODEL, key)) for key in _MODULE_KEYS
        },
    }


def validate_hifsa_metadata(
    metadata,
    cfg,
    dataset_name,
    *,
    require_module_match,
):
    """Reject checkpoints that do not satisfy the Native HiFSA contract."""
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            "Checkpoint is missing Native HiFSA metadata and is not compatible "
            "with this code version. Retrain it with the current code."
        )

    schema_version = metadata.get("schema_version")
    if schema_version != HIFSA_CONFIG_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported HiFSA checkpoint schema version: "
            f"{schema_version!r}; expected {HIFSA_CONFIG_SCHEMA_VERSION}."
        )

    expected = {
        "dataset": str(dataset_name),
        "clip_model": str(cfg.MODEL.CLIP_MODEL),
        "backbone": str(cfg.MODEL.BACKBONE),
    }
    mismatches = [
        f"{key}: checkpoint={metadata.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]

    modules = metadata.get("modules")
    if not isinstance(modules, Mapping):
        mismatches.append("modules: missing or invalid")
    else:
        for key in _MODULE_KEYS:
            if type(modules.get(key)) is not bool:
                mismatches.append(f"modules.{key}: missing or not bool")
            elif require_module_match and modules[key] != bool(
                getattr(cfg.MODEL, key)
            ):
                mismatches.append(
                    f"modules.{key}: checkpoint={modules[key]!r}, "
                    f"current={bool(getattr(cfg.MODEL, key))!r}"
                )

    if mismatches:
        raise RuntimeError(
            "Checkpoint metadata does not match the current HiFSA run:\n- "
            + "\n- ".join(mismatches)
        )


def apply_checkpoint_module_state(cfg, metadata):
    """Restore the three module switches before constructing the model."""
    modules = metadata["modules"]
    for key in _MODULE_KEYS:
        cfg.MODEL[key] = bool(modules[key])
