import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download

from open_clip_lib import (
    create_model_and_transforms,
    HFTokenizer,
    get_mean_std,
)

from .layers import PVL_Adapter
from .scale_block import ScaleBlock
from .freq_modules import (
    AdaptiveFrequencyDecompose,
    BFCIResidualBranch,
    FrequencyResidualProjection,
    FrequencyRouter,
    FrequencyTokenAdapter,
    LCFSA,
)


def download_checkpoint(filename: str):
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    local_path = os.path.join(ckpt_dir, filename)
    if os.path.isfile(local_path):
        print(f"Found checkpoint: {local_path}")
        return local_path

    print(f"Checkpoint not found. Downloading {filename} from Hugging Face...")
    hf_hub_download(
        repo_id="TahaKoleilat/MedCLIPSeg",
        repo_type="model",
        filename=f"checkpoints/{filename}",
        local_dir=".",
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded checkpoint to {local_path}")
    return local_path


def load_unimedclip_to_device(cfg):
    if cfg.MODEL.BACKBONE == "ViT-B/16":
        model_name = "ViT-B-16-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_b16.pt")
    elif cfg.MODEL.BACKBONE == "ViT-L/14":
        model_name = "ViT-L-14-336-quickgelu"
        pretrained_weights = download_checkpoint(
            "unimed_clip_vit_l14_base_text_encoder.pt"
        )
    else:
        raise NotImplementedError(
            f"Backbone {cfg.MODEL.BACKBONE} not implemented."
        )

    text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    mean, std = get_mean_std()
    device = cfg.MODEL.DEVICE

    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained_weights,
        precision="amp",
        device=device,
        force_quick_gelu=True,
        mean=mean,
        std=std,
        inmem=True,
        text_encoder_name=text_encoder_name,
    )
    return model.to(device).eval()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, output_hidden_states=False):
        super().__init__()

        self.cfg = cfg
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text_encoder
        self.logit_scale = clip_model.logit_scale

        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = list(cfg.MODEL.LAYERS)

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512
        elif cfg.MODEL.BACKBONE == "ViT-L/14":
            self.embed_dim = 1024
            self.patch_size = 14
            self.text_proj_dim = 768
            raise NotImplementedError("ViT-L/14 not implemented yet.")
        else:
            raise NotImplementedError(
                f"Backbone {cfg.MODEL.BACKBONE} not implemented."
            )

        self.output_hidden_states = output_hidden_states
        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE

        self.tokenizer = HFTokenizer(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            context_length=256,
            **{},
        )

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )
        self.upscale = nn.Sequential(
            *[ScaleBlock(self.text_proj_dim) for _ in range(self.num_upscale)]
        )
        self.pvl_adapters = nn.ModuleList(
            [
                PVL_Adapter(
                    in_channels_vis=self.embed_dim,
                    in_channels_txt=self.embed_dim,
                    adapter_channels=adapter_channels,
                    beta=self.beta,
                    gate_init=self.gate_init,
                )
                for _ in range(len(self.fusion_stages))
            ]
        )

        # --------------------------------------------------
        # Independent frequency modules
        # --------------------------------------------------
        # Independence here means:
        #   1) AFPI, BFCI and LCFSA have separate switches, parameters, gates,
        #      warm-up schedules, optimizer groups and injection layers;
        #   2) all branches consume the same immutable raw HF/LF tokens;
        #   3) no branch overwrites tokens used by another branch;
        #   4) optional branch construction uses an isolated RNG scope, so
        #      enabling one branch cannot change initialization of the others.
        self.use_afpi = bool(getattr(cfg.MODEL, "USE_AFPI", False))
        self.use_bfci = bool(getattr(cfg.MODEL, "USE_BFCI", False))
        self.use_lcfsa = bool(getattr(cfg.MODEL, "USE_LCFSA", False))
        self.use_frequency_tokens = (
            self.use_afpi or self.use_bfci or self.use_lcfsa
        )

        dataset_base_name = str(cfg.DATASET.NAME).split("_")[0]
        supported_frequency_datasets = {
            "BTMRI",
            "Covid19",
            "BUSI",
            "Kvasir",
        }
        if (
            self.use_frequency_tokens
            and dataset_base_name not in supported_frequency_datasets
        ):
            raise ValueError(
                "The modular frequency implementation supports BTMRI, "
                "Covid19, BUSI and Kvasir. "
                f"Got dataset={cfg.DATASET.NAME}"
            )

        self.frequency_mode = "off"
        self.frequency_config = None
        self.afpi_inject_layer = -1
        self.bfci_inject_layer = -1
        self.lcfsa_inject_layer = -1
        self.use_afpi_adapter = False
        self.use_afpi_dynamic_router = False
        self.afpi_gate_max = 0.0
        self._last_afpi_state = None
        self._bfci_progress_value = 0.0
        self._lcfsa_progress_value = 0.0

        def isolated_build(seed, factory):
            rng_devices = (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available()
                else []
            )
            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                return factory()

        if self.use_frequency_tokens:
            freq_cfg = self._resolve_hifsa_config(cfg)
            self.frequency_config = freq_cfg
            self.frequency_mode = freq_cfg["mode"]

            # Shared provider only: decomposition + frozen CLIP patch embedding.
            # It contains no trainable interaction/router/gate parameters.
            self.frequency_decompose = AdaptiveFrequencyDecompose(
                mode=freq_cfg["mode"],
                gray=freq_cfg["gray"],
                per_image_norm=freq_cfg["per_image_norm"],
                gaussian_kernel=freq_cfg["gaussian_kernel"],
                gaussian_sigma=freq_cfg["gaussian_sigma"],
                input_is_normalized=freq_cfg["input_denorm"],
                input_mean=(0.485, 0.456, 0.406),
                input_std=(0.229, 0.224, 0.225),
                high_standardize=freq_cfg["high_standardize"],
                high_clip=freq_cfg["high_clip"],
                laplacian_abs=freq_cfg["laplacian_abs"],
                high_type=freq_cfg["high_type"],
            )

            num_visual_layers = len(
                self.vision_model.transformer.resblocks
            )

            def validate_layer(name, value):
                value = int(value)
                if not 0 <= value < num_visual_layers:
                    raise ValueError(
                        f"{name} must be between 0 and "
                        f"{num_visual_layers - 1}, got {value}"
                    )
                return value

            # ---------------- AFPI residual branch ----------------
            if self.use_afpi:
                self.afpi_inject_layer = validate_layer(
                    "AFPI_INJECT_LAYER",
                    getattr(cfg.MODEL, "AFPI_INJECT_LAYER", 3),
                )
                self.use_afpi_adapter = freq_cfg["use_afpi_adapter"]
                self.use_afpi_dynamic_router = freq_cfg["dynamic_router"]
                self.router_scale = freq_cfg["router_scale"]
                self.afpi_gate_max = freq_cfg["gate_max"]
                afpi_seed = int(getattr(cfg.MODEL, "AFPI_INIT_SEED", 3101))

                def build_afpi_modules():
                    modules = nn.ModuleDict()
                    modules["norm"] = nn.LayerNorm(self.embed_dim)
                    modules["dropout"] = nn.Dropout(freq_cfg["dropout"])
                    if self.use_afpi_dynamic_router:
                        modules["router"] = FrequencyRouter(
                            dim=self.embed_dim,
                            hidden_dim=freq_cfg["router_hidden"],
                            dropout=freq_cfg["router_dropout"],
                        )
                    if self.use_afpi_adapter:
                        modules["adapter"] = FrequencyTokenAdapter(
                            dim=self.embed_dim,
                            reduction=freq_cfg["adapter_reduction"],
                            dropout=freq_cfg["dropout"],
                        )
                    modules["residual"] = FrequencyResidualProjection(
                        dim=self.embed_dim,
                        reduction=freq_cfg["residual_reduction"],
                        dropout=freq_cfg["dropout"],
                    )
                    return modules

                self.afpi_modules = isolated_build(afpi_seed, build_afpi_modules)
                self.afpi_norm = self.afpi_modules["norm"]
                self.afpi_dropout = self.afpi_modules["dropout"]
                if self.use_afpi_dynamic_router:
                    self.afpi_router = self.afpi_modules["router"]
                if self.use_afpi_adapter:
                    self.afpi_adapter = self.afpi_modules["adapter"]
                self.afpi_residual = self.afpi_modules["residual"]

                self.afpi_gate = nn.Parameter(
                    torch.tensor(freq_cfg["gate_init"], dtype=torch.float32)
                )
                init_mix = torch.tensor(
                    [freq_cfg["high_weight"], freq_cfg["low_weight"]],
                    dtype=torch.float32,
                )
                init_mix = init_mix / init_mix.sum()
                self.afpi_mix_logits = nn.Parameter(
                    torch.log(init_mix + 1e-6)
                )
                self.register_buffer(
                    "afpi_progress",
                    torch.tensor(1.0, dtype=torch.float32),
                    persistent=False,
                )

            # ---------------- BFCI residual branch ----------------
            if self.use_bfci:
                self.bfci_inject_layer = validate_layer(
                    "BFCI_INJECT_LAYER",
                    getattr(cfg.MODEL, "BFCI_INJECT_LAYER", 5),
                )
                self.bfci_detach_context = bool(
                    getattr(cfg.MODEL, "BFCI_DETACH_CONTEXT", True)
                )
                bfci_seed = int(
                    getattr(cfg.MODEL, "BFCI_INIT_SEED", 3203)
                )
                self.bfci_branch = isolated_build(
                    bfci_seed,
                    lambda: BFCIResidualBranch(
                        dim=self.embed_dim,
                        attn_dim=int(
                            getattr(
                                cfg.MODEL,
                                "BFCI_ATTN_DIM",
                                freq_cfg["attn_dim"],
                            )
                        ),
                        num_heads=int(
                            getattr(
                                cfg.MODEL,
                                "BFCI_NUM_HEADS",
                                freq_cfg["num_heads"],
                            )
                        ),
                        dropout=float(
                            getattr(cfg.MODEL, "BFCI_DROPOUT", 0.0)
                        ),
                        gate_init=float(
                            getattr(cfg.MODEL, "BFCI_GATE_INIT", 0.0)
                        ),
                        gate_max=float(
                            getattr(cfg.MODEL, "BFCI_GATE_MAX", 0.05)
                        ),
                        high_weight=float(
                            getattr(
                                cfg.MODEL,
                                "BFCI_HIGH_WEIGHT",
                                freq_cfg["high_weight"],
                            )
                        ),
                        low_weight=float(
                            getattr(
                                cfg.MODEL,
                                "BFCI_LOW_WEIGHT",
                                freq_cfg["low_weight"],
                            )
                        ),
                        output_gain=float(
                            getattr(cfg.MODEL, "BFCI_OUTPUT_GAIN", 0.10)
                        ),
                    ),
                )
                self.register_buffer(
                    "bfci_progress",
                    torch.tensor(0.0, dtype=torch.float32),
                    persistent=False,
                )

            # ---------------- LCFSA residual branch ----------------
            if self.use_lcfsa:
                self.lcfsa_inject_layer = validate_layer(
                    "LCFSA_INJECT_LAYER",
                    getattr(cfg.MODEL, "LCFSA_INJECT_LAYER", 8),
                )
                self.lcfsa_init_seed = int(
                    getattr(cfg.MODEL, "LCFSA_INIT_SEED", 3407)
                )
                self.lcfsa_detach_context = bool(
                    getattr(cfg.MODEL, "LCFSA_DETACH_CONTEXT", True)
                )

                def build_lcfsa_modules():
                    module = LCFSA(
                        dim=self.embed_dim,
                        attn_dim=int(
                            getattr(cfg.MODEL, "LCFSA_ATTN_DIM", 192)
                        ),
                        num_heads=int(
                            getattr(cfg.MODEL, "LCFSA_NUM_HEADS", 4)
                        ),
                        reduction=int(
                            getattr(cfg.MODEL, "LCFSA_REDUCTION", 8)
                        ),
                        dropout=float(
                            getattr(cfg.MODEL, "LCFSA_DROPOUT", 0.0)
                        ),
                        gate_init=float(
                            getattr(cfg.MODEL, "LCFSA_GATE_INIT", 0.0)
                        ),
                        gate_max=float(
                            getattr(cfg.MODEL, "LCFSA_GATE_MAX", 0.05)
                        ),
                        tau=float(
                            getattr(cfg.MODEL, "LCFSA_TAU", 0.25)
                        ),
                        zero_init=bool(
                            getattr(cfg.MODEL, "LCFSA_ZERO_INIT", True)
                        ),
                    )
                    high_weight = float(
                        getattr(
                            cfg.MODEL,
                            "LCFSA_HIGH_WEIGHT",
                            freq_cfg["high_weight"],
                        )
                    )
                    low_weight = float(
                        getattr(
                            cfg.MODEL,
                            "LCFSA_LOW_WEIGHT",
                            freq_cfg["low_weight"],
                        )
                    )
                    weight_sum = high_weight + low_weight
                    if weight_sum <= 0.0:
                        high_weight, low_weight = 0.5, 0.5
                    else:
                        high_weight /= weight_sum
                        low_weight /= weight_sum
                    mix = nn.Parameter(
                        torch.log(
                            torch.tensor(
                                [high_weight, low_weight],
                                dtype=torch.float32,
                            )
                            + 1e-6
                        )
                    )
                    return module, mix

                self.lcfsa, self.lcfsa_mix_logits = isolated_build(
                    self.lcfsa_init_seed,
                    build_lcfsa_modules,
                )
                self.register_buffer(
                    "lcfsa_progress",
                    torch.tensor(0.0, dtype=torch.float32),
                    persistent=False,
                )

            print(self._format_hifsa_state(prefix="[Modules:init]"))

    @staticmethod
    def _get(cfg_node, key, default):
        return getattr(cfg_node, key, default)

    def _resolve_hifsa_config(self, cfg) -> Dict[str, object]:
        """
        Resolve dataset defaults and YAML overrides.

        All YAML values take precedence. Explicit values are strongly
        recommended because they make experiment logs reproducible and prevent
        a mode change from being hidden behind ``FREQUENCY_MODE: auto``.
        """
        dataset_name = str(cfg.DATASET.NAME).lower()
        model_cfg = cfg.MODEL
        requested_mode = str(self._get(model_cfg, "FREQUENCY_MODE", "auto")).lower()

        defaults = {
            "mode": "haar_energy",
            "gray": False,
            "per_image_norm": False,
            "high_weight": 0.5,
            "low_weight": 0.5,
        }

        if "covid" in dataset_name:
            defaults.update(
                mode="gray_normalized",
                gray=True,
                per_image_norm=True,
                high_weight=0.30,
                low_weight=0.70,
            )
        elif "btmri" in dataset_name:
            defaults.update(
                mode="haar_energy",
                gray=True,
                per_image_norm=False,
                high_weight=0.60,
                low_weight=0.40,
            )
        elif "busi" in dataset_name or "eus" in dataset_name:
            defaults.update(
                mode="gaussian",
                gray=True,
                per_image_norm=True,
                high_weight=0.30,
                low_weight=0.70,
            )
        elif "isic" in dataset_name:
            # Main correction: do not independently min-max normalize each RGB
            # channel of a dermoscopy image. It distorts clinically relevant
            # color relationships and can amplify hair/specular extrema.
            defaults.update(
                mode="laplacian",
                gray=False,
                per_image_norm=False,
                high_weight=0.45,
                low_weight=0.55,
            )
        elif "kvasir" in dataset_name:
            defaults.update(
                mode="haar_energy",
                gray=False,
                per_image_norm=False,
                high_weight=0.50,
                low_weight=0.50,
            )

        mode = defaults["mode"] if requested_mode == "auto" else requested_mode
        gray = bool(self._get(model_cfg, "FREQUENCY_GRAY", defaults["gray"]))
        per_image_norm = bool(
            self._get(
                model_cfg,
                "FREQUENCY_PER_IMAGE_NORM",
                defaults["per_image_norm"],
            )
        )
        high_weight = float(
            self._get(model_cfg, "AFPI_HIGH_WEIGHT", defaults["high_weight"])
        )
        low_weight = float(
            self._get(model_cfg, "AFPI_LOW_WEIGHT", defaults["low_weight"])
        )

        weight_sum = high_weight + low_weight
        if weight_sum <= 0:
            high_weight, low_weight = 0.5, 0.5
        else:
            high_weight /= weight_sum
            low_weight /= weight_sum

        gate_max = float(self._get(model_cfg, "AFPI_GATE_MAX", 1.0))
        if not 0.0 < gate_max <= 1.0:
            raise ValueError("AFPI_GATE_MAX must be in (0, 1]")

        dropout = float(self._get(model_cfg, "AFPI_DROPOUT", 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("AFPI_DROPOUT must be in [0, 1)")

        return {
            "mode": mode,
            "gray": gray,
            "per_image_norm": per_image_norm,
            "high_weight": high_weight,
            "low_weight": low_weight,
            "gate_init": float(self._get(model_cfg, "AFPI_GATE_INIT", 0.0)),
            "gate_max": gate_max,
            "dropout": dropout,
            "use_bfci": bool(self._get(model_cfg, "USE_BFCI", True)),
            "use_afpi_adapter": bool(
                self._get(model_cfg, "USE_AFPI_ADAPTER", False)
            ),
            "adapter_reduction": int(
                self._get(model_cfg, "AFPI_ADAPTER_REDUCTION", 4)
            ),
            "pre_norm": bool(self._get(model_cfg, "AFPI_PRENORM", False)),
            "attention_dropout": float(
                self._get(model_cfg, "AFPI_ATTN_DROPOUT", 0.0)
            ),
            "num_heads": int(self._get(model_cfg, "AFPI_NUM_HEADS", 8)),
            "attn_dim": int(
                self._get(model_cfg, "AFPI_ATTN_DIM", self.embed_dim)
            ),
            "dynamic_router": bool(
                self._get(model_cfg, "AFPI_DYNAMIC_ROUTER", False)
            ),
            "router_hidden": int(
                self._get(model_cfg, "AFPI_ROUTER_HIDDEN", 64)
            ),
            "router_dropout": float(
                self._get(model_cfg, "AFPI_ROUTER_DROPOUT", 0.0)
            ),
            "router_scale": float(
                self._get(model_cfg, "AFPI_ROUTER_SCALE", 0.5)
            ),
            "residual_reduction": int(
                self._get(model_cfg, "AFPI_RESIDUAL_REDUCTION", 4)
            ),
            "warmup_epochs": int(
                self._get(model_cfg, "AFPI_WARMUP_EPOCHS", 0)
            ),
            "gaussian_kernel": int(
                self._get(model_cfg, "FREQUENCY_GAUSSIAN_KERNEL", 5)
            ),
            "gaussian_sigma": float(
                self._get(model_cfg, "FREQUENCY_GAUSSIAN_SIGMA", 1.5)
            ),
            "input_denorm": bool(
                self._get(model_cfg, "FREQUENCY_INPUT_DENORM", False)
            ),
            "high_standardize": bool(
                self._get(model_cfg, "FREQUENCY_HIGH_STANDARDIZE", False)
            ),
            "high_clip": float(self._get(model_cfg, "FREQUENCY_HIGH_CLIP", 5.0)),
            "laplacian_abs": bool(
                self._get(model_cfg, "FREQUENCY_LAPLACIAN_ABS", False)
            ),
            "high_type": str(
                self._get(
                    model_cfg,
                    "FREQUENCY_HIGH_TYPE",
                    "signed_sum_normalized",
                )
            ),
        }

    def _format_hifsa_state(self, prefix="[Modules]") -> str:
        if not self.use_frequency_tokens:
            return f"{prefix} disabled"

        parts = [
            f"{prefix} mode={self.frequency_mode}",
            f"AFPI={self.use_afpi}",
            f"BFCI={self.use_bfci}",
            f"LCFSA={self.use_lcfsa}",
        ]
        if self.use_afpi:
            with torch.no_grad():
                mix = torch.softmax(self.afpi_mix_logits.detach(), dim=0)
                gate = self.afpi_gate_max * torch.sigmoid(
                    self.afpi_gate.detach()
                )
            parts.extend(
                [
                    f"afpi_layer={self.afpi_inject_layer}",
                    f"afpi_mix={mix[0].item():.3f}/{mix[1].item():.3f}",
                    f"afpi_gate={gate.item():.4f}",
                ]
            )
        if self.use_bfci:
            parts.extend(
                [
                    f"bfci_layer={self.bfci_inject_layer}",
                    f"bfci_detach={self.bfci_detach_context}",
                ]
            )
        if self.use_lcfsa:
            with torch.no_grad():
                mix = torch.softmax(self.lcfsa_mix_logits.detach(), dim=0)
            parts.extend(
                [
                    f"lcfsa_layer={self.lcfsa_inject_layer}",
                    f"lcfsa_mix={mix[0].item():.3f}/{mix[1].item():.3f}",
                    f"lcfsa_detach={self.lcfsa_detach_context}",
                ]
            )
        return " | ".join(parts)

    def get_hifsa_state(self) -> Dict[str, object]:
        """Return a flat serializable state for all independent branches."""
        if not self.use_frequency_tokens:
            return {"enabled": False}

        state = {
            "enabled": True,
            "mode": self.frequency_mode,
            "use_afpi": bool(self.use_afpi),
            "use_bfci": bool(self.use_bfci),
            "use_lcfsa": bool(self.use_lcfsa),
            "afpi_inject_layer": int(self.afpi_inject_layer),
            "bfci_inject_layer": int(self.bfci_inject_layer),
            "lcfsa_inject_layer": int(self.lcfsa_inject_layer),
            "afpi_progress": 0.0,
            "bfci_progress": 0.0,
            "lcfsa_progress": 0.0,
            "use_afpi_adapter": bool(self.use_afpi_adapter),
            "use_afpi_dynamic_router": bool(self.use_afpi_dynamic_router),
        }

        if self.use_afpi:
            with torch.no_grad():
                mix = torch.softmax(self.afpi_mix_logits.detach(), dim=0)
                gate = self.afpi_gate_max * torch.sigmoid(
                    self.afpi_gate.detach()
                )
            state.update(
                {
                    "afpi_mix_high": float(mix[0].cpu()),
                    "afpi_mix_low": float(mix[1].cpu()),
                    "afpi_gate_raw": float(self.afpi_gate.detach().cpu()),
                    "afpi_gate_effective": float(gate.cpu()),
                    "afpi_gate_max": float(self.afpi_gate_max),
                    "afpi_progress": float(
                        self.afpi_progress.detach().cpu()
                    ),
                }
            )
            if self._last_afpi_state is not None:
                state.update(self._last_afpi_state)

        if self.use_bfci:
            state["bfci_progress"] = float(
                self.bfci_progress.detach().cpu()
            )
            if self.bfci_branch._last_state is not None:
                state.update(self.bfci_branch._last_state)

        if self.use_lcfsa:
            state["lcfsa_progress"] = float(
                self.lcfsa_progress.detach().cpu()
            )
            with torch.no_grad():
                lcfsa_mix = torch.softmax(
                    self.lcfsa_mix_logits.detach(), dim=0
                )
            state["lcfsa_base_mix_high"] = float(lcfsa_mix[0].cpu())
            state["lcfsa_base_mix_low"] = float(lcfsa_mix[1].cpu())
            if self.lcfsa._last_lcfsa_state is not None:
                state.update(self.lcfsa._last_lcfsa_state)

        return state

    def set_afpi_progress(self, progress: float):
        """Set only the AFPI warm-up multiplier."""
        if not self.use_afpi:
            return
        progress = max(0.0, min(float(progress), 1.0))
        self.afpi_progress.fill_(progress)

    def set_bfci_progress(self, progress: float):
        """Set only the independent BFCI warm-up multiplier."""
        if not self.use_bfci:
            return
        progress = max(0.0, min(float(progress), 1.0))
        self.bfci_progress.fill_(progress)
        self._bfci_progress_value = progress

    def set_lcfsa_progress(self, progress: float):
        """Set only the independent LCFSA warm-up multiplier."""
        if not self.use_lcfsa:
            return
        progress = max(0.0, min(float(progress), 1.0))
        self.lcfsa_progress.fill_(progress)
        self._lcfsa_progress_value = progress

    def patch_embed_only(self, image):
        x = self.vision_model.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x.permute(0, 2, 1)

    def build_shared_frequency_token_bank(self, image: torch.Tensor):
        high_img, low_img = self.frequency_decompose(image)

        high_tokens = self.patch_embed_only(high_img)
        low_tokens = self.patch_embed_only(low_img)

        patch_position = (
            self.vision_model.positional_embedding[1:]
            .to(
                device=high_tokens.device,
                dtype=high_tokens.dtype,
            )
            .unsqueeze(0)
        )
        if high_tokens.shape[1] != patch_position.shape[1]:
            raise RuntimeError(
                "Frequency token count does not match "
                "CLIP patch positional embedding."
            )

        high_tokens = high_tokens + patch_position
        low_tokens = low_tokens + patch_position

        high_tokens = self.vision_model.ln_pre(high_tokens)
        low_tokens = self.vision_model.ln_pre(low_tokens)

        # Return immutable raw frequency tokens. AFPI, BFCI and LCFSA each
        # normalize/process their own copy and never overwrite this provider
        # output, which is the key to clean module ablations.
        return high_tokens, low_tokens

    def inject_afpi_residual(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        high_tokens: torch.Tensor,
        low_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ):
        """
        Inject the frequency residual after an early visual/text layer.

        x_img: [1+N, B, C]
        x_txt: [B, L, C]
        high_tokens: [B, N, C]
        low_tokens: [B, N, C]
        """
        image_tokens = x_img.transpose(0, 1)
        cls_token = image_tokens[:, :1, :]
        visual_patches = image_tokens[:, 1:, :]

        if visual_patches.shape != high_tokens.shape:
            raise RuntimeError(
                "Current visual patch tokens and frequency tokens must have "
                "the same shape. "
                f"visual={tuple(visual_patches.shape)}, "
                f"high={tuple(high_tokens.shape)}"
            )

        if self.use_afpi_dynamic_router:
            mix_delta, gate_delta = self.afpi_router(
                visual_patches,
                high_tokens,
                low_tokens,
            )
            mix_logits = (
                self.afpi_mix_logits.unsqueeze(0)
                + self.router_scale * mix_delta
            )
            mix_weights = torch.softmax(mix_logits, dim=-1)
            gate = self.afpi_gate_max * torch.sigmoid(
                self.afpi_gate + self.router_scale * gate_delta
            )
            freq_tokens = (
                mix_weights[:, 0, None, None] * high_tokens
                + mix_weights[:, 1, None, None] * low_tokens
            )
            self._last_afpi_state = {
                "afpi_dynamic_mix_high": float(
                    mix_weights[:, 0].detach().mean().cpu()
                ),
                "afpi_dynamic_mix_low": float(
                    mix_weights[:, 1].detach().mean().cpu()
                ),
                "afpi_dynamic_gate": float(gate.detach().mean().cpu()),
            }
        else:
            mix_weights = torch.softmax(self.afpi_mix_logits, dim=0)
            gate = self.afpi_gate_max * torch.sigmoid(self.afpi_gate)
            freq_tokens = (
                mix_weights[0] * high_tokens
                + mix_weights[1] * low_tokens
            )

        if self.use_afpi_adapter:
            freq_tokens = self.afpi_adapter(freq_tokens)

        freq_tokens = self.afpi_norm(freq_tokens)
        freq_tokens = self.afpi_dropout(freq_tokens)
        afpi_residual = self.afpi_residual(freq_tokens)

        if gate.ndim == 0:
            gate = gate.view(1, 1, 1)
        else:
            gate = gate[:, None, None]

        progress = self.afpi_progress.to(
            device=visual_patches.device,
            dtype=visual_patches.dtype,
        )
        enhanced_patches = (
            visual_patches
            + progress
            * gate.to(visual_patches.dtype)
            * afpi_residual
        )
        enhanced_tokens = torch.cat(
            [cls_token, enhanced_patches],
            dim=1,
        )
        return enhanced_tokens.transpose(0, 1)

    def inject_bfci_residual(
        self,
        x_img: torch.Tensor,
        high_tokens: torch.Tensor,
        low_tokens: torch.Tensor,
    ):
        """Inject the standalone BFCI residual without rewriting HF/LF."""
        if self._bfci_progress_value <= 0.0:
            return x_img

        image_tokens = x_img.transpose(0, 1)
        cls_token = image_tokens[:, :1, :]
        visual_patches = image_tokens[:, 1:, :]
        if visual_patches.shape != high_tokens.shape:
            raise RuntimeError(
                "Current visual patch tokens and BFCI frequency tokens must "
                "have the same shape. "
                f"visual={tuple(visual_patches.shape)}, "
                f"high={tuple(high_tokens.shape)}"
            )

        if self.bfci_detach_context:
            bfci_high = high_tokens.detach()
            bfci_low = low_tokens.detach()
        else:
            bfci_high = high_tokens
            bfci_low = low_tokens

        residual = self.bfci_branch(bfci_high, bfci_low)
        progress = self.bfci_progress.to(
            device=visual_patches.device,
            dtype=visual_patches.dtype,
        )
        enhanced_patches = (
            visual_patches
            + progress * residual.to(visual_patches.dtype)
        )
        return torch.cat(
            [cls_token, enhanced_patches], dim=1
        ).transpose(0, 1)

    def inject_lcfsa_residual(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        high_tokens: torch.Tensor,
        low_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ):
        """
        Add an independent PVL-aligned language-frequency residual.

        The existing AFPI/BFCI injection is not altered. This branch operates
        at its own deeper layer, where current visual and text tokens have
        comparable semantic depth.
        """
        # Do not even execute the branch while its warm-up progress is zero.
        # This prevents LCFSA dropout/attention from consuming RNG and changing
        # the AFPI+BFCI trajectory before LCFSA is supposed to start.
        if self._lcfsa_progress_value <= 0.0:
            return x_img

        image_tokens = x_img.transpose(0, 1)
        cls_token = image_tokens[:, :1, :]
        visual_patches = image_tokens[:, 1:, :]

        if visual_patches.shape != high_tokens.shape:
            raise RuntimeError(
                "Current visual patch tokens and LCFSA frequency tokens must "
                "have the same shape. "
                f"visual={tuple(visual_patches.shape)}, "
                f"high={tuple(high_tokens.shape)}"
            )

        # LCFSA owns its mixture prior. It never reads or rewrites AFPI/BFCI
        # gates, routers, mixture logits or transformed frequency tokens.
        mix_weights = torch.softmax(self.lcfsa_mix_logits, dim=0)

        # LCFSA is an independent adapter. By default its context inputs are
        # detached so its auxiliary/segmentation gradients cannot rewrite the
        # already effective PVL, AFPI, BFCI or router parameters.
        if self.lcfsa_detach_context:
            lcfsa_visual = visual_patches.detach()
            lcfsa_high = high_tokens.detach()
            lcfsa_low = low_tokens.detach()
            lcfsa_text = x_txt.detach()
        else:
            lcfsa_visual = visual_patches
            lcfsa_high = high_tokens
            lcfsa_low = low_tokens
            lcfsa_text = x_txt

        lcfsa_residual = self.lcfsa(
            visual_tokens=lcfsa_visual,
            high_tokens=lcfsa_high,
            low_tokens=lcfsa_low,
            text_tokens=lcfsa_text,
            text_attention_mask=attention_mask,
            mix_weights=mix_weights,
        )
        progress = self.lcfsa_progress.to(
            device=visual_patches.device,
            dtype=visual_patches.dtype,
        )
        enhanced_patches = (
            visual_patches
            + progress * lcfsa_residual.to(visual_patches.dtype)
        )
        return torch.cat(
            [cls_token, enhanced_patches], dim=1
        ).transpose(0, 1)

    def get_lcfsa_aux_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Patch-level supervision for the text-conditioned spatial filter.

        The target is obtained only from training masks and is never used at
        validation/test time. A balanced BCE term handles small lesions, while
        a soft Dice term encourages coherent foreground localization.
        """
        zero = masks.sum() * 0.0
        if not self.use_lcfsa or self._lcfsa_progress_value <= 0.0:
            return zero

        spatial_logits = self.lcfsa.get_last_spatial_logits()
        if spatial_logits is None:
            return zero

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()

        num_patches = spatial_logits.shape[1]
        patch_h = int(round(num_patches ** 0.5))
        if patch_h * patch_h != num_patches:
            raise RuntimeError(
                f"LCFSA auxiliary loss expects a square patch grid, got N={num_patches}"
            )

        target = F.interpolate(
            masks,
            size=(patch_h, patch_h),
            mode="area",
        )
        target = target.flatten(2).transpose(1, 2).clamp(0.0, 1.0)

        # Per-sample balancing avoids the background dominating tiny lesions.
        pos_fraction = target.mean(dim=1, keepdim=True).clamp(0.02, 0.98)
        weights = (
            target * (0.5 / pos_fraction)
            + (1.0 - target) * (0.5 / (1.0 - pos_fraction))
        )
        bce = F.binary_cross_entropy_with_logits(
            spatial_logits,
            target,
            reduction="none",
        )
        bce = (weights * bce).mean()

        probs = torch.sigmoid(spatial_logits)
        intersection = (probs * target).sum(dim=(1, 2))
        denominator = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
        return 0.5 * (bce + dice.mean())

    def encode_text_image(
        self,
        tokenized_prompts,
        text_prompts,
        image,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if attention_mask is None:
            attention_mask = (
                tokenized_prompts != self.text_model.config.pad_token_id
            ).long()

        x_txt = self.text_model.transformer.embeddings(
            inputs_embeds=text_prompts
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            self.dtype
        ).min

        # Original image patch tokens.
        x_img = self.patch_embed_only(image)

        frequency_high = None
        frequency_low = None
        if self.use_frequency_tokens:
            frequency_high, frequency_low = self.build_shared_frequency_token_bank(
                image
            )

        cls_token = (
            self.vision_model.class_embedding.to(x_img.dtype)
            + torch.zeros(
                x_img.shape[0],
                1,
                x_img.shape[-1],
                dtype=x_img.dtype,
                device=x_img.device,
            )
        )

        x_img = torch.cat([cls_token, x_img], dim=1)
        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)
        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)  # NLD -> LND

        hidden_states = []
        for i, (block, layer) in enumerate(
            zip(
                self.vision_model.transformer.resblocks,
                self.text_model.transformer.encoder.layer,
            )
        ):
            if i in self.fusion_stages:
                adapter_idx = self.fusion_stages.index(i)
                vis_pvl, txt_pvl = self.pvl_adapters[adapter_idx](
                    x_img.transpose(1, 0),
                    x_txt,
                )
                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl.transpose(1, 0)

            x_img = block(x_img)
            txt_out = layer(
                x_txt,
                attention_mask=extended_attention_mask,
            )
            x_txt = txt_out[0]

            if self.use_afpi and i == self.afpi_inject_layer:
                x_img = self.inject_afpi_residual(
                    x_img=x_img,
                    x_txt=x_txt,
                    high_tokens=frequency_high,
                    low_tokens=frequency_low,
                    attention_mask=attention_mask,
                )

            if self.use_bfci and i == self.bfci_inject_layer:
                x_img = self.inject_bfci_residual(
                    x_img=x_img,
                    high_tokens=frequency_high,
                    low_tokens=frequency_low,
                )

            if self.use_lcfsa and i == self.lcfsa_inject_layer:
                x_img = self.inject_lcfsa_residual(
                    x_img=x_img,
                    x_txt=x_txt,
                    high_tokens=frequency_high,
                    low_tokens=frequency_low,
                    attention_mask=attention_mask,
                )

            hidden_states.append(x_img)

        x_img = x_img.permute(1, 0, 2)  # LND -> NLD
        x_img = self.vision_model.ln_post(x_img)

        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        pooled_out = x_txt[:, 0, :]
        projected = self.text_model.proj(pooled_out)

        if self.output_hidden_states:
            return x_img, hidden_states, projected
        return x_img, projected

    def compute_seg_logits(
        self,
        image_features,
        text_features,
        batch_size,
        height,
        width,
    ):
        text_features = F.normalize(text_features, dim=-1)

        cls_token = F.normalize(image_features[:, 0, :], dim=-1)
        seg_feats = image_features[:, 1:, :]
        seg_feats = F.normalize(seg_feats, dim=-1)

        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        expected_patches = h_patch * w_patch
        if seg_feats.shape[1] != expected_patches:
            raise RuntimeError(
                f"Patch-token mismatch: got {seg_feats.shape[1]}, "
                f"expected {expected_patches} for input {height}x{width} "
                f"and patch size {self.patch_size}."
            )

        seg_feats = seg_feats.reshape(
            batch_size,
            h_patch,
            w_patch,
            -1,
        ).permute(0, 3, 1, 2)

        text_mask_feature = self.mask_head(text_features).unsqueeze(1)
        seg_logits = torch.einsum(
            "bqc,bchw->bqhw",
            text_mask_feature,
            self.upscale(seg_feats),
        )
        seg_logits = F.interpolate(
            seg_logits,
            size=(self.im_size, self.im_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return seg_logits, cls_token

    @staticmethod
    def soft_cross_entropy(pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def _single_forward(self, image, tokenized_prompts, prompts, attention_mask=None):
        batch_size, _, height, width = image.shape
        image_features, text_features = self.encode_text_image(
            tokenized_prompts,
            prompts,
            image,
            attention_mask=attention_mask,
        )
        seg_logits, cls_token = self.compute_seg_logits(
            image_features,
            text_features,
            batch_size,
            height,
            width,
        )
        return seg_logits, cls_token, image_features, text_features

    def forward(self, image, text, num_samples=30):
        tokenized_prompts = self.tokenizer(text).to(self.device)
        attention_mask = (
            tokenized_prompts != self.text_model.config.pad_token_id
        ).long()

        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(
                tokenized_prompts
            ).type(self.dtype)

        if self.training:
            seg_logits, _, image_features, text_features = self._single_forward(
                image,
                tokenized_prompts,
                prompts,
                attention_mask=attention_mask,
            )

            patch_logits = F.normalize(image_features[:, 1:, :], dim=-1)
            patch_mean = patch_logits.mean(dim=1)

            logits_per_image = (patch_mean @ text_features.T) / self.temperature
            logits_per_text = (text_features @ patch_mean.T) / self.temperature

            with torch.no_grad():
                text_sim = (text_features @ text_features.T) / self.temperature
                text_sim = F.normalize(text_sim, dim=-1)
                soft_targets = F.softmax(text_sim, dim=-1)

            loss_i2t = self.soft_cross_entropy(
                logits_per_image,
                soft_targets,
            )
            loss_t2i = self.soft_cross_entropy(
                logits_per_text,
                soft_targets.T,
            )
            clip_loss = (loss_i2t + loss_t2i) / 2
            return seg_logits, clip_loss

        seg_samples = []
        for _ in range(num_samples):
            seg_logits, _, _, _ = self._single_forward(
                image,
                tokenized_prompts,
                prompts,
                attention_mask=attention_mask,
            )
            seg_samples.append(seg_logits)

        return torch.stack(seg_samples, dim=0)


def build_hifsa_unimedclip(cfg):
    print(f"Loading UniMedCLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_unimedclip_to_device(cfg)
    clip_model.float()

    print("Building custom UniMedCLIP")
    model = CustomCLIP(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")

    for name, param in model.named_parameters():
        trainable = False
        if "pvl_adapters" in name:
            trainable = True
        elif "mask_head" in name or "upscale" in name:
            trainable = True
        elif getattr(model, "use_afpi", False) and (
            name.startswith("afpi_modules.")
            or name in {"afpi_gate", "afpi_mix_logits"}
        ):
            trainable = True
        elif getattr(model, "use_bfci", False) and name.startswith(
            "bfci_branch."
        ):
            trainable = True
        elif getattr(model, "use_lcfsa", False) and (
            name.startswith("lcfsa.") or name == "lcfsa_mix_logits"
        ):
            trainable = True
        param.requires_grad_(trainable)

    return model
