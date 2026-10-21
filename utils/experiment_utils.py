def build_results_name(cfg):
    """Build one canonical Native experiment name."""
    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    enabled = [
        name
        for name, active in (
            ("AFPI", cfg.MODEL.USE_AFPI),
            ("BFCI", cfg.MODEL.USE_BFCI),
            ("LCFSA", cfg.MODEL.USE_LCFSA),
        )
        if bool(active)
    ]
    method_name = "HiFSA_" + "_".join(enabled) if enabled else "MedCLIPSeg"
    return f"{method_name}_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"
