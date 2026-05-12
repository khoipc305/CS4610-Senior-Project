"""
Inference Engine for Enhanced 3D Lung Cancer Nodule Detection.

Wraps:
  - Checkpoint loading (Attention U-Net or baseline U-Net)
  - CT volume preprocessing (HU clip + normalization)
  - Sliding-window 3D inference (MONAI)
  - Post-processing (threshold + connected-components filter)
  - Confidence scoring + Grad-CAM heatmap

Used both by the Streamlit deployment app (`app.py`) and by CLI scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from monai.inferers import sliding_window_inference
    _HAS_MONAI = True
except Exception:  # pragma: no cover
    _HAS_MONAI = False

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

from src.models.attention_unet import create_model


# ----------------------------- Data classes -----------------------------

@dataclass
class InferenceResult:
    """Container for a single-volume inference run."""
    probability: np.ndarray            # [D, H, W] in [0, 1]
    mask: np.ndarray                   # [D, H, W] uint8 {0, 1}
    confidence: float                  # mean prob over predicted-positive voxels
    n_nodules: int                     # # connected components after filtering
    nodule_volumes_mm3: list           # volumes (in voxel-count units)
    ct_normalized: np.ndarray          # [D, H, W] float32 in [0, 1] (post-preproc)
    gradcam: Optional[np.ndarray] = None  # [D, H, W] in [0, 1]


# ----------------------------- Preprocessing -----------------------------

def preprocess_ct(
    volume: np.ndarray,
    hu_clip: Tuple[float, float] = (-1000.0, 400.0),
) -> np.ndarray:
    """
    Clip Hounsfield Units to lung window, then normalize to [0, 1].
    Accepts arbitrary [D, H, W] ndarray.
    """
    vol = volume.astype(np.float32)
    lo, hi = hu_clip
    # If volume already looks normalized (max < 5), assume preprocessed.
    if vol.max() <= 5.0 and vol.min() >= -5.0:
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax - vmin < 1e-6:
            return np.zeros_like(vol, dtype=np.float32)
        return ((vol - vmin) / (vmax - vmin)).astype(np.float32)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


# ----------------------------- Post-processing -----------------------------

def postprocess_mask(
    prob: np.ndarray,
    threshold: float = 0.5,
    min_voxels: int = 10,
) -> Tuple[np.ndarray, int, list]:
    """
    Threshold + remove tiny connected components.
    Returns (mask uint8, n_components, list-of-volumes).
    """
    raw = (prob >= threshold).astype(np.uint8)
    if not _HAS_SCIPY or raw.sum() == 0:
        n = int(raw.sum() > 0)
        return raw, n, [int(raw.sum())] if n else []
    labeled, n_lbl = ndi.label(raw)
    keep = np.zeros_like(raw, dtype=np.uint8)
    sizes = []
    for i in range(1, n_lbl + 1):
        comp = labeled == i
        s = int(comp.sum())
        if s >= min_voxels:
            keep[comp] = 1
            sizes.append(s)
    return keep, len(sizes), sizes


# ----------------------------- Engine -----------------------------

class InferenceEngine:
    """
    Lightweight inference engine. Loads a checkpoint once and is reusable.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_config: Optional[dict] = None,
        device: Optional[str] = None,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 2,
        overlap: float = 0.25,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap

        cfg = model_config or {
            "name": "AttentionUNet3D",
            "in_channels": 1,
            "out_channels": 2,
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "norm": "BATCH",
            "dropout": 0.1,
            "use_attention": True,
            "attention_type": "spatial",
        }
        self.model = create_model(cfg).to(self.device)
        self.checkpoint_path = checkpoint_path
        self.has_weights = False
        if checkpoint_path and os.path.isfile(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            self.has_weights = True
        self.model.eval()

    # --------- internals ---------
    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                f"[InferenceEngine] Loaded with {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys (non-strict)."
            )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run sliding-window 3D inference on a [1,1,D,H,W] tensor."""
        if _HAS_MONAI:
            with torch.no_grad():
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=self.overlap,
                    mode="gaussian",
                )
        else:  # fallback: single forward pass (may OOM on full volumes)
            with torch.no_grad():
                logits = self.model(x)
        return logits

    # --------- public API ---------
    def predict(
        self,
        volume: np.ndarray,
        threshold: float = 0.5,
        min_voxels: int = 10,
        with_gradcam: bool = False,
    ) -> InferenceResult:
        """
        Run end-to-end inference on a single CT volume [D, H, W].
        """
        ct = preprocess_ct(volume)
        x = torch.from_numpy(ct).float().unsqueeze(0).unsqueeze(0).to(self.device)

        logits = self._forward(x)                 # [1, C, D, H, W]
        prob = F.softmax(logits, dim=1)[0, 1]     # [D, H, W]
        prob_np = prob.detach().cpu().numpy().astype(np.float32)

        mask, n_comp, sizes = postprocess_mask(prob_np, threshold, min_voxels)
        conf = float(prob_np[mask == 1].mean()) if mask.sum() > 0 else 0.0

        cam = None
        if with_gradcam:
            cam = self.gradcam(x)

        return InferenceResult(
            probability=prob_np,
            mask=mask,
            confidence=conf,
            n_nodules=n_comp,
            nodule_volumes_mm3=sizes,
            ct_normalized=ct,
            gradcam=cam,
        )

    # --------- Grad-CAM ---------
    def gradcam(
        self,
        x: torch.Tensor,
        target_class: int = 1,
        size_multiple: int = 16,
    ) -> np.ndarray:
        """
        Compute a 3D Grad-CAM heatmap on the deepest available conv layer.

        Pads the input so every spatial dim is divisible by ``size_multiple``
        (MONAI's 5-level UNet needs 16) and crops the heatmap back to the
        original shape, so this works on arbitrary input sizes.
        """
        # Find the deepest Conv3d module to hook.
        # Restrict search to the actual U-Net backbone if present; otherwise
        # some Conv3d layers (e.g. in unused attention blocks) would be hit but
        # never executed during forward, so the hook would never fire.
        search_root = getattr(self.model, "unet", self.model)
        target_layer = None
        for module in search_root.modules():
            if isinstance(module, torch.nn.Conv3d):
                target_layer = module

        orig_shape = x.shape[2:]   # (D, H, W)
        if target_layer is None:
            return np.zeros(orig_shape, dtype=np.float32)

        # --- pad to size_multiple ---
        pad_d = (-orig_shape[0]) % size_multiple
        pad_h = (-orig_shape[1]) % size_multiple
        pad_w = (-orig_shape[2]) % size_multiple
        # F.pad order for 5D: (W_left, W_right, H_left, H_right, D_left, D_right)
        x_p = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")

        activations: list = []
        gradients: list = []

        def fwd_hook(_m, _i, o):
            activations.append(o)

        def bwd_hook(_m, _gi, go):
            gradients.append(go[0])

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x_p)
            score = logits[:, target_class].sum()
            score.backward(retain_graph=False)

            act = activations[-1].detach()
            grad = gradients[-1].detach()
            weights = grad.mean(dim=(2, 3, 4), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            # Upsample to padded size, then crop back to original shape
            cam = F.interpolate(
                cam, size=x_p.shape[2:], mode="trilinear", align_corners=False
            )
            cam = cam[..., : orig_shape[0], : orig_shape[1], : orig_shape[2]]
            cam = cam.squeeze().cpu().numpy()
            cam -= cam.min()
            cam = cam / (cam.max() + 1e-8)
            return cam.astype(np.float32)
        finally:
            h1.remove()
            h2.remove()


# ----------------------------- Demo helpers -----------------------------

def make_demo_volume(shape: Tuple[int, int, int] = (96, 128, 128)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic CT-like volume with two spherical 'nodules' for demo mode.
    Returns (volume_in_HU_range, ground_truth_mask).
    """
    rng = np.random.default_rng(0)
    D, H, W = shape
    vol = rng.normal(loc=-700.0, scale=120.0, size=shape).astype(np.float32)
    # Add lung-shaped low-density region (HU ~ -800)
    zz, yy, xx = np.ogrid[:D, :H, :W]
    cx, cy, cz = W // 2, H // 2, D // 2
    lung = ((xx - cx) ** 2 / (W * 0.35) ** 2
            + (yy - cy) ** 2 / (H * 0.35) ** 2
            + (zz - cz) ** 2 / (D * 0.45) ** 2) <= 1.0
    vol[lung] = rng.normal(loc=-820.0, scale=60.0, size=int(lung.sum())).astype(np.float32)

    mask = np.zeros(shape, dtype=np.uint8)
    # Two synthetic nodules
    for (cz0, cy0, cx0, r) in [(D // 2 - 6, H // 2 - 10, W // 2 + 12, 5),
                                (D // 2 + 8, H // 2 + 14, W // 2 - 16, 4)]:
        nodule = ((xx - cx0) ** 2 + (yy - cy0) ** 2 + (zz - cz0) ** 2) <= r ** 2
        vol[nodule] = rng.normal(loc=50.0, scale=20.0, size=int(nodule.sum())).astype(np.float32)
        mask[nodule] = 1
    return vol, mask
