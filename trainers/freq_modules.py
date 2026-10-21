import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFrequencyDecompose(nn.Module):
    """
    Dataset-adaptive frequency decomposition.

    Supported modes
    ----------------
    haar_energy:
        Haar LL is used as low-frequency representation. LH/HL/HH are
        aggregated by energy to avoid signed cancellation.

    gray_normalized:
        Grayscale + per-image min-max normalization + Haar decomposition.
        The high-frequency branch can use either stable energy aggregation
        or the normalized signed-sum implementation.

    gaussian:
        Gaussian low-pass + residual high-pass.

    laplacian:
        Gaussian low-pass + Laplacian high-pass.

    Important
    ---------
    The MedCLIPSeg dataloader supplies ImageNet-normalized images. Frequency
    decomposition is more interpretable in image space, so ``input_is_normalized``
    can be enabled to de-normalize the image first. The low-frequency output is
    then converted back to model input space. The high-frequency response is a
    residual/edge signal and is centered before entering the frozen CLIP patch
    embedding.
    """

    SUPPORTED_MODES = {
        "haar_energy",
        "haar_signed",
        "gray_normalized",
        "gaussian",
        "laplacian",
    }

    def __init__(
        self,
        mode: str = "haar_energy",
        gray: bool = False,
        per_image_norm: bool = False,
        gaussian_kernel: int = 5,
        gaussian_sigma: float = 1.5,
        input_is_normalized: bool = False,
        input_mean: Sequence[float] = (0.485, 0.456, 0.406),
        input_std: Sequence[float] = (0.229, 0.224, 0.225),
        high_standardize: bool = False,
        high_clip: float = 5.0,
        laplacian_abs: bool = False,
        high_type: str = "signed_sum_normalized",
    ):
        super().__init__()

        mode = str(mode).lower().strip()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported frequency mode: {mode}. "
                f"Supported modes: {sorted(self.SUPPORTED_MODES)}"
            )

        if gaussian_kernel < 3 or gaussian_kernel % 2 == 0:
            raise ValueError("gaussian_kernel must be an odd integer >= 3")
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be > 0")
        if high_clip <= 0:
            raise ValueError("high_clip must be > 0")
        if high_type not in {"signed_sum_normalized", "energy"}:
            raise ValueError(
                "high_type must be 'signed_sum_normalized' or 'energy'"
            )

        self.mode = mode
        self.gray = bool(gray)
        self.per_image_norm = bool(per_image_norm)
        self.gaussian_kernel = int(gaussian_kernel)
        self.gaussian_sigma = float(gaussian_sigma)
        self.input_is_normalized = bool(input_is_normalized)
        self.high_standardize = bool(high_standardize)
        self.high_clip = float(high_clip)
        self.laplacian_abs = bool(laplacian_abs)
        self.high_type = high_type

        mean = torch.tensor(input_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(input_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)

    @staticmethod
    def _minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-image, per-channel min-max normalization."""
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    @staticmethod
    def _standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-image, per-channel zero-mean unit-variance normalization."""
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (x - mean) / (std + eps)

    def _stats_for_channels(self, x: torch.Tensor):
        c = x.shape[1]
        mean = self.input_mean.to(device=x.device, dtype=x.dtype)
        std = self.input_std.to(device=x.device, dtype=x.dtype)

        if mean.shape[1] == c:
            return mean, std
        if c == 1:
            return mean.mean(dim=1, keepdim=True), std.mean(dim=1, keepdim=True)
        if mean.shape[1] == 1:
            return mean.repeat(1, c, 1, 1), std.repeat(1, c, 1, 1)

        raise ValueError(
            f"Cannot match normalization stats with {c} input channels."
        )

    def _to_image_space(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_normalized:
            return x

        mean, std = self._stats_for_channels(x)
        return (x * std + mean).clamp(0.0, 1.0)

    def _low_to_model_space(self, low: torch.Tensor) -> torch.Tensor:
        if not self.input_is_normalized:
            return low

        mean, std = self._stats_for_channels(low)
        return (low - mean) / std

    def _high_to_model_space(self, high: torch.Tensor) -> torch.Tensor:
        # Normalized-input path: when image-space de-normalization and
        # high standardization are both disabled, preserve the former branch
        # values exactly. This prevents the corrected ISIC settings from
        # silently changing previously completed BTMRI/BUSI/EUS/Kvasir runs.
        if not self.input_is_normalized and not self.high_standardize:
            return high

        # A high-pass response is not an ordinary image. Remove its DC component
        # before mapping it to the frozen CLIP patch-embedding scale.
        high = high - high.mean(dim=(-2, -1), keepdim=True)

        if self.high_standardize:
            high = self._standardize(high)
        elif self.input_is_normalized:
            _, std = self._stats_for_channels(high)
            high = high / std

        return high.clamp(-self.high_clip, self.high_clip)

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB-like input to luminance while preserving the channel count.
        For repeated grayscale medical images this is exactly equivalent to the
        original input up to numerical precision.
        """
        if x.shape[1] == 1:
            return x

        if x.shape[1] < 3:
            gray = x.mean(dim=1, keepdim=True)
        else:
            weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            gray = (x[:, :3] * weights).sum(dim=1, keepdim=True)

        return gray.repeat(1, x.shape[1], 1, 1)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        k = self.gaussian_kernel
        sigma = self.gaussian_sigma
        c = x.shape[1]

        coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)

        pad = k // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, kernel, groups=c)

    @staticmethod
    def _laplacian_high(x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        kernel = x.new_tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ]
        ).view(1, 1, 3, 3)
        kernel = kernel.repeat(c, 1, 1, 1)

        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x_pad, kernel, groups=c)

    def _haar(self, x: torch.Tensor, high_type: str = "energy"):
        """
        Differentiable 2D Haar decomposition.

        Four subbands are first computed: LL, LH, HL and HH. The downstream
        model remains a two-branch model: LL is the low-frequency branch and
        the three directional high-frequency subbands are aggregated.
        """
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        hp, wp = x.shape[-2:]
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # Orthonormal 2-D Haar scaling factor is 1/2.
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        if high_type == "signed_sum_normalized":
            # Normalized signed-sum implementation.
            lh_n = self._minmax_norm(lh)
            hl_n = self._minmax_norm(hl)
            hh_n = self._minmax_norm(hh)
            high = self._minmax_norm(lh_n + hl_n + hh_n)
            low = self._minmax_norm(ll)

        elif high_type == "frequency_energy":
            # Stable alternative: preserve edge strength and avoid turning
            # zero signed response into an artificial value of about 0.5.
            high = torch.sqrt(lh.square() + hl.square() + hh.square() + 1e-6)
            high = self._minmax_norm(high)
            low = self._minmax_norm(ll)

        elif high_type == "signed":
            high = lh + hl + hh
            low = ll

        elif high_type == "energy":
            high = torch.sqrt(lh.square() + hl.square() + hh.square() + 1e-6)
            low = ll

        else:
            raise ValueError(f"Unsupported Haar high_type: {high_type}")

        high = F.interpolate(
            high, size=(hp, wp), mode="bilinear", align_corners=False
        )
        low = F.interpolate(
            low, size=(hp, wp), mode="bilinear", align_corners=False
        )

        return high[:, :, :h, :w], low[:, :, :h, :w]

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

        # Mode-local preprocessing prevents one dataset's settings from
        # silently leaking into a different decomposition strategy.
        work = self._to_image_space(x)

        if self.mode == "gray_normalized":
            work = self._to_gray(work)
            work = self._minmax_norm(work)
            haar_type = (
                "frequency_energy"
                if self.high_type == "energy"
                else "signed_sum_normalized"
            )
            high, low = self._haar(work, high_type=haar_type)

        else:
            if self.gray:
                work = self._to_gray(work)
            if self.per_image_norm:
                work = self._minmax_norm(work)

            if self.mode == "haar_energy":
                high, low = self._haar(work, high_type="energy")
            elif self.mode == "haar_signed":
                high, low = self._haar(work, high_type="signed")
            elif self.mode == "gaussian":
                low = self._gaussian_blur(work)
                high = work - low
            elif self.mode == "laplacian":
                low = self._gaussian_blur(work)
                high = self._laplacian_high(work)
                if self.laplacian_abs:
                    high = high.abs()
            else:  # guarded in __init__
                raise RuntimeError(f"Unexpected frequency mode: {self.mode}")

        high = self._high_to_model_space(high)
        low = self._low_to_model_space(low)
        return high, low


class BFCIResidualBranch(nn.Module):
    """Independent high/low-frequency bidirectional interaction adapter.

    This branch never overwrites the raw frequency tokens used by AFPI or LCFSA.
    It computes a standalone residual from the same frozen frequency-token
    provider and injects that residual into the visual stream with its own
    signed ReZero gate, mixture weights, warm-up schedule and optimizer group.

    ``gate_init=0`` guarantees an exact identity at initialization, while the
    small non-zero output projections preserve a useful gate gradient.
    """

    def __init__(
        self,
        dim: int,
        attn_dim: int = 192,
        num_heads: int = 4,
        dropout: float = 0.0,
        gate_init: float = 0.0,
        gate_max: float = 0.05,
        high_weight: float = 0.5,
        low_weight: float = 0.5,
        output_gain: float = 0.10,
    ):
        super().__init__()
        dim = int(dim)
        attn_dim = int(attn_dim)
        num_heads = int(num_heads)

        if dim <= 0:
            raise ValueError("dim must be positive")
        if attn_dim <= 0 or attn_dim > dim:
            raise ValueError("BFCI_ATTN_DIM must be in (0, dim]")
        if num_heads <= 0 or attn_dim % num_heads != 0:
            raise ValueError(
                "BFCI_ATTN_DIM must be divisible by BFCI_NUM_HEADS"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("BFCI_DROPOUT must be in [0, 1)")
        if not 0.0 < gate_max <= 1.0:
            raise ValueError("BFCI_GATE_MAX must be in (0, 1]")
        if output_gain <= 0.0:
            raise ValueError("BFCI_OUTPUT_GAIN must be positive")

        weight_sum = float(high_weight) + float(low_weight)
        if weight_sum <= 0.0:
            high_weight, low_weight = 0.5, 0.5
        else:
            high_weight = float(high_weight) / weight_sum
            low_weight = float(low_weight) / weight_sum

        self.dim = dim
        self.attn_dim = attn_dim
        self.gate_max = float(gate_max)

        self.high_norm = nn.LayerNorm(dim)
        self.low_norm = nn.LayerNorm(dim)
        self.high_reduce = nn.Linear(dim, attn_dim, bias=False)
        self.low_reduce = nn.Linear(dim, attn_dim, bias=False)

        self.high_from_low = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.low_from_high = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.high_expand = nn.Linear(attn_dim, dim, bias=False)
        self.low_expand = nn.Linear(attn_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.high_expand.weight, gain=output_gain)
        nn.init.xavier_uniform_(self.low_expand.weight, gain=output_gain)

        self.delta_norm = nn.LayerNorm(dim)
        self.output_dropout = nn.Dropout(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        init_mix = torch.tensor([high_weight, low_weight], dtype=torch.float32)
        self.mix_logits = nn.Parameter(torch.log(init_mix + 1e-6))

        self._last_state = None

    def forward(
        self,
        high_tokens: torch.Tensor,
        low_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if high_tokens.shape != low_tokens.shape:
            raise ValueError("high_tokens and low_tokens must have same shape")
        if high_tokens.ndim != 3 or high_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected frequency tokens [B,N,{self.dim}], "
                f"got {tuple(high_tokens.shape)}"
            )

        high = self.high_reduce(self.high_norm(high_tokens))
        low = self.low_reduce(self.low_norm(low_tokens))

        high_cross, _ = self.high_from_low(
            query=high,
            key=low,
            value=low,
            need_weights=False,
        )
        low_cross, _ = self.low_from_high(
            query=low,
            key=high,
            value=high,
            need_weights=False,
        )

        delta_high = self.high_expand(high_cross)
        delta_low = self.low_expand(low_cross)
        mix = torch.softmax(self.mix_logits, dim=0)
        delta = mix[0] * delta_high + mix[1] * delta_low
        delta = self.output_dropout(self.delta_norm(delta))

        gate = self.gate_max * torch.tanh(self.gate_logit)
        residual = gate.to(delta.dtype) * delta

        if not torch.jit.is_scripting():
            with torch.no_grad():
                self._last_state = {
                    "bfci_gate": float(gate.detach().cpu()),
                    "bfci_gate_abs": float(gate.abs().detach().cpu()),
                    "bfci_mix_high": float(mix[0].detach().cpu()),
                    "bfci_mix_low": float(mix[1].detach().cpu()),
                    "bfci_delta_rms": float(
                        delta.float().pow(2).mean().sqrt().detach().cpu()
                    ),
                    "bfci_residual_rms": float(
                        residual.float().pow(2).mean().sqrt().detach().cpu()
                    ),
                }
        return residual


class FrequencyRouter(nn.Module):
    """
    Image-level dynamic router for high/low mixture and injection gate.

    The final layer is zero-initialized.  Therefore each experiment starts from
    the explicit dataset prior in the YAML file, then learns a small per-image
    correction rather than replacing the prior immediately.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        input_dim = dim * 3
        hidden_dim = max(int(hidden_dim), 16)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        raw_tokens: torch.Tensor,
        high_tokens: torch.Tensor,
        low_tokens: torch.Tensor,
    ):
        pooled = torch.cat(
            [
                raw_tokens.mean(dim=1),
                high_tokens.mean(dim=1),
                low_tokens.mean(dim=1),
            ],
            dim=-1,
        )
        delta = self.net(pooled)
        return delta[:, :2], delta[:, 2]


class FrequencyResidualProjection(nn.Module):
    """
    Zero-initialized bottleneck projection that emits only a residual.

    At initialization the complete frequency branch contributes exactly zero,
    so the network starts from the original MedCLIPSeg behavior.  This is
    especially important for 10% BUSI/ISIC/EUS experiments.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = max(dim // max(int(reduction), 1), 32)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.act(self.down(self.norm(x)))))


class FrequencyTokenAdapter(nn.Module):
    """
    Optional lightweight residual adapter applied after high/low fusion.

    The final projection is zero-initialized, so enabling this module starts
    from an identity mapping rather than perturbing the pretrained CLIP tokens.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = max(dim // max(int(reduction), 1), 32)

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + residual


class LCFSA(nn.Module):
    """
    Stable PVL-aligned language-guided frequency filtering for MedCLIPSeg.

    Compared with the previous implementation, this version addresses three
    practical failure modes observed in low-data experiments:

    1. The text branch is used to *filter existing frequency content* instead
       of primarily injecting attended text values into the visual stream.
    2. A signed ReZero-style gate starts from an exact identity while retaining
       a non-zero gradient, avoiding the ``zero projection + tiny sigmoid gate``
       dead branch.
    3. Spatial logits are retained for an optional patch-mask auxiliary loss,
       giving the prompt gate direct localization supervision during training.

    The module itself remains an independent residual branch. Callers may pass
    detached visual/text/frequency contexts to prevent LCFSA gradients from
    changing the established MedCLIPSeg + AFPI + BFCI path.
    """

    def __init__(
        self,
        dim: int,
        attn_dim: int = 192,
        num_heads: int = 4,
        reduction: int = 8,
        dropout: float = 0.0,
        gate_init: float = 0.0,
        gate_max: float = 0.05,
        tau: float = 0.25,
        zero_init: bool = True,
    ):
        super().__init__()

        dim = int(dim)
        attn_dim = int(attn_dim)
        num_heads = int(num_heads)
        reduction = int(reduction)

        if dim <= 0:
            raise ValueError("dim must be positive")
        if attn_dim <= 0 or attn_dim > dim:
            raise ValueError("LCFSA_ATTN_DIM must be in (0, dim]")
        if num_heads <= 0 or attn_dim % num_heads != 0:
            raise ValueError(
                "LCFSA_ATTN_DIM must be divisible by LCFSA_NUM_HEADS"
            )
        if reduction <= 0:
            raise ValueError("LCFSA_REDUCTION must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LCFSA_DROPOUT must be in [0, 1)")
        if not 0.0 < gate_max <= 1.0:
            raise ValueError("LCFSA_GATE_MAX must be in (0, 1]")
        if tau <= 0.0:
            raise ValueError("LCFSA_TAU must be positive")

        self.dim = dim
        self.attn_dim = attn_dim
        self.gate_max = float(gate_max)
        self.tau = float(tau)

        hidden = max(attn_dim // reduction, 32)

        self.visual_norm = nn.LayerNorm(dim)
        self.high_norm = nn.LayerNorm(dim)
        self.low_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)

        self.visual_reduce = nn.Linear(dim, attn_dim, bias=False)
        self.high_reduce = nn.Linear(dim, attn_dim, bias=False)
        self.low_reduce = nn.Linear(dim, attn_dim, bias=False)
        self.text_reduce = nn.Linear(dim, attn_dim, bias=False)

        # Same-position alignment transfers the current visual semantic depth
        # to the shallow HF/LF tokens without replacing their frequency content.
        self.high_visual_fuse = nn.Sequential(
            nn.Linear(attn_dim * 2, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, attn_dim),
        )
        self.low_visual_fuse = nn.Sequential(
            nn.Linear(attn_dim * 2, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, attn_dim),
        )
        # Start the alignment correction at zero; the frequency token itself
        # remains non-zero and can drive the ReZero gate gradient immediately.
        nn.init.zeros_(self.high_visual_fuse[-1].weight)
        nn.init.zeros_(self.high_visual_fuse[-1].bias)
        nn.init.zeros_(self.low_visual_fuse[-1].weight)
        nn.init.zeros_(self.low_visual_fuse[-1].bias)

        # Vision-first context, consistent with MedCLIPSeg's strongest two-way
        # interaction setting: text first absorbs current visual context and is
        # then queried by HF/LF tokens.
        self.text_to_visual = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.high_to_text = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.low_to_text = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Produce channel *logits* rather than already-squashed gates. They are
        # combined with spatial logits and centered by tanh, allowing both
        # enhancement and suppression around zero.
        self.high_channel_logits = nn.Sequential(
            nn.Linear(attn_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, attn_dim),
        )
        self.low_channel_logits = nn.Sequential(
            nn.Linear(attn_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, attn_dim),
        )

        self.fused_norm = nn.LayerNorm(attn_dim)
        self.output_proj = nn.Linear(attn_dim, dim, bias=False)
        # Identity is guaranteed by the zero ReZero gate, not by a dead output
        # projection. A small-gain projection provides a meaningful direction
        # for the gate gradient at the first active step.
        gain = 0.10 if zero_init else 1.0
        nn.init.xavier_uniform_(self.output_proj.weight, gain=gain)
        self.delta_norm = nn.LayerNorm(dim)
        self.output_dropout = nn.Dropout(dropout)

        # Signed ReZero gate: gate=0 is an exact identity and d(gate)/d(logit)
        # is non-zero. This avoids the near-frozen 0.014 sigmoid gate observed
        # in the previous implementation.
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

        self._last_lcfsa_state = None
        self._last_spatial_logits = None

    @staticmethod
    def _masked_mean(
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1, keepdim=True)
        valid = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        return (x * valid).sum(dim=1, keepdim=True) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    def _semantic_spatial_logits(
        self,
        frequency_tokens: torch.Tensor,
        attended_text: torch.Tensor,
    ) -> torch.Tensor:
        frequency_tokens = F.normalize(frequency_tokens, dim=-1)
        attended_text = F.normalize(attended_text, dim=-1)
        similarity = (frequency_tokens * attended_text).sum(
            dim=-1, keepdim=True
        )
        return similarity / self.tau

    def get_last_spatial_logits(self) -> torch.Tensor | None:
        return self._last_spatial_logits

    def forward(
        self,
        visual_tokens: torch.Tensor,
        high_tokens: torch.Tensor,
        low_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
        mix_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_spatial_logits = None

        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must be [B,N,C]")
        if high_tokens.shape != visual_tokens.shape:
            raise ValueError("high_tokens must match visual_tokens")
        if low_tokens.shape != visual_tokens.shape:
            raise ValueError("low_tokens must match visual_tokens")
        if text_tokens.ndim != 3:
            raise ValueError("text_tokens must be [B,L,C]")
        if text_tokens.shape[0] != visual_tokens.shape[0]:
            raise ValueError("visual/text batch sizes must match")
        if visual_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected visual dim={self.dim}, got {visual_tokens.shape[-1]}"
            )
        if text_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected text dim={self.dim}, got {text_tokens.shape[-1]}"
            )

        visual = self.visual_reduce(self.visual_norm(visual_tokens))
        high = self.high_reduce(self.high_norm(high_tokens))
        low = self.low_reduce(self.low_norm(low_tokens))
        text = self.text_reduce(self.text_norm(text_tokens))

        high = high + self.high_visual_fuse(torch.cat([high, visual], dim=-1))
        low = low + self.low_visual_fuse(torch.cat([low, visual], dim=-1))

        text_visual, _ = self.text_to_visual(
            query=text,
            key=visual,
            value=visual,
            need_weights=False,
        )
        contextual_text = text + text_visual

        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = text_attention_mask == 0

        high_text, _ = self.high_to_text(
            query=high,
            key=contextual_text,
            value=contextual_text,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        low_text, _ = self.low_to_text(
            query=low,
            key=contextual_text,
            value=contextual_text,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        text_context = self._masked_mean(contextual_text, text_attention_mask)
        visual_context = visual.mean(dim=1, keepdim=True)
        global_context = torch.cat([text_context, visual_context], dim=-1)

        high_channel_logits = self.high_channel_logits(global_context)
        low_channel_logits = self.low_channel_logits(global_context)
        high_spatial_logits = self._semantic_spatial_logits(high, high_text)
        low_spatial_logits = self._semantic_spatial_logits(low, low_text)

        # Centered filters in [-1, 1]. Text determines where and which channels
        # of the *existing HF/LF features* should be enhanced or suppressed.
        high_filter = torch.tanh(high_spatial_logits + high_channel_logits)
        low_filter = torch.tanh(low_spatial_logits + low_channel_logits)
        high_update = high * high_filter
        low_update = low * low_filter

        if mix_weights is None:
            mix_weights = visual_tokens.new_tensor([0.5, 0.5])

        if mix_weights.ndim == 1:
            if mix_weights.numel() != 2:
                raise ValueError("mix_weights must have two elements")
            fused_update = (
                mix_weights[0] * high_update
                + mix_weights[1] * low_update
            )
            mixed_spatial_logits = (
                mix_weights[0] * high_spatial_logits
                + mix_weights[1] * low_spatial_logits
            )
            mix_high_mean = mix_weights[0]
            mix_low_mean = mix_weights[1]
        elif mix_weights.ndim == 2:
            if mix_weights.shape != (visual_tokens.shape[0], 2):
                raise ValueError("batched mix_weights must be [B,2]")
            high_weight = mix_weights[:, 0, None, None]
            low_weight = mix_weights[:, 1, None, None]
            fused_update = high_weight * high_update + low_weight * low_update
            mixed_spatial_logits = (
                high_weight * high_spatial_logits
                + low_weight * low_spatial_logits
            )
            mix_high_mean = mix_weights[:, 0].mean()
            mix_low_mean = mix_weights[:, 1].mean()
        else:
            raise ValueError("mix_weights must be [2] or [B,2]")

        # Keep the graph for the optional patch-level auxiliary loss.
        self._last_spatial_logits = mixed_spatial_logits

        delta = self.output_proj(self.fused_norm(fused_update))
        delta = self.delta_norm(delta)
        delta = self.output_dropout(delta)

        gate = self.gate_max * torch.tanh(self.gate_logit)
        residual = gate.to(delta.dtype) * delta

        if not torch.jit.is_scripting():
            with torch.no_grad():
                high_spatial = torch.sigmoid(high_spatial_logits)
                low_spatial = torch.sigmoid(low_spatial_logits)
                self._last_lcfsa_state = {
                    "lcfsa_gate": float(gate.detach().cpu()),
                    "lcfsa_gate_abs": float(gate.abs().detach().cpu()),
                    "lcfsa_mix_high": float(mix_high_mean.detach().cpu()),
                    "lcfsa_mix_low": float(mix_low_mean.detach().cpu()),
                    "lcfsa_high_spatial_mean": float(
                        high_spatial.mean().detach().cpu()
                    ),
                    "lcfsa_high_spatial_std": float(
                        high_spatial.std(unbiased=False).detach().cpu()
                    ),
                    "lcfsa_low_spatial_mean": float(
                        low_spatial.mean().detach().cpu()
                    ),
                    "lcfsa_low_spatial_std": float(
                        low_spatial.std(unbiased=False).detach().cpu()
                    ),
                    "lcfsa_delta_rms": float(
                        delta.square().mean().sqrt().detach().cpu()
                    ),
                    "lcfsa_residual_rms": float(
                        residual.square().mean().sqrt().detach().cpu()
                    ),
                }

        return residual


