"""
Interactive Deployment App - Enhanced Lung Cancer Nodule Detection
==================================================================

A lightweight Streamlit web app that fulfills the Spring CS4620 proposal's
Phase 5 expected outcome:

    "Develop an interactive interface for CT image upload and inference.
     Integrate trained models to output predictions, confidence scores,
     and visual explanations."

Run:
    streamlit run app.py

Features:
    - Upload a CT volume (.nii, .nii.gz, or .npy), or use the built-in demo.
    - Sliding-window 3D inference with the Attention U-Net.
    - Per-slice axial viewer with mask + Grad-CAM overlay.
    - Multi-plane (axial / sagittal / coronal) view.
    - Per-volume confidence score + nodule count.
    - Download predicted mask as .nii.gz.

Note:
    If no trained checkpoint is provided, the app launches in DEMO MODE
    using a synthetic CT-like volume (so reviewers can still see the UI).
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference import (
    InferenceEngine,
    InferenceResult,
    make_demo_volume,
)

# Optional NIfTI support
try:
    import nibabel as nib
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False


# =====================================================================
# Page config
# =====================================================================
st.set_page_config(
    page_title="Lung Nodule Detection (Spring 2026)",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 Enhanced 3D Lung Cancer Nodule Detection")
st.caption(
    "CS4620 Spring 2026 — Khoi Nguyen Pham — Advisor: Prof. Hao Ji"
)


# =====================================================================
# Sidebar: model + thresholds
# =====================================================================
st.sidebar.header("⚙️ Model & Inference")

default_ckpt = "checkpoints/lung_cancer_attention_unet/best_model.pth"
ckpt_path = st.sidebar.text_input(
    "Checkpoint path (.pth)",
    value=default_ckpt,
    help="Path to a trained model. If missing, app runs in DEMO mode.",
)

device_choice = st.sidebar.selectbox(
    "Device",
    ["auto", "cuda", "cpu"],
    index=0,
)

threshold = st.sidebar.slider(
    "Probability threshold", 0.05, 0.95, 0.50, step=0.05
)
min_voxels = st.sidebar.slider(
    "Min nodule size (voxels)", 1, 200, 10, step=1
)
with_gradcam = st.sidebar.checkbox("Compute Grad-CAM (slower)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Performance")
max_dim = st.sidebar.slider(
    "Max volume dim (downsample if larger)", 64, 512, 192, step=32,
    help=(
        "Real LIDC scans are typically 512x512xN. On CPU, full-res "
        "sliding-window inference takes 10+ min. Downsampling to "
        "~192 lets a demo finish in under a minute."
    ),
)


# =====================================================================
# Cache the engine across reruns
# =====================================================================
@st.cache_resource(show_spinner="Loading model...")
def get_engine(ckpt: str, device: str) -> InferenceEngine:
    dev = None if device == "auto" else device
    return InferenceEngine(checkpoint_path=ckpt if os.path.isfile(ckpt) else None,
                           device=dev)


engine = get_engine(ckpt_path, device_choice)
if engine.has_weights:
    st.sidebar.success(f"✅ Loaded checkpoint: {os.path.basename(ckpt_path)}")
else:
    st.sidebar.warning(
        "⚠️ No checkpoint found at the path above — running with **untrained** "
        "weights. Predictions will be meaningless; use this only to verify the UI."
    )
st.sidebar.caption(f"Device: `{engine.device}`")


# =====================================================================
# Input: upload or demo
# =====================================================================
st.subheader("1. Input CT volume")

col_a, col_b = st.columns([2, 1])
with col_a:
    upload = st.file_uploader(
        "Upload a CT volume (.nii / .nii.gz / .npy) "
        "**OR** one-or-more 2D slices (.png / .jpg / .jpeg)",
        type=["nii", "gz", "npy", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help=(
            "For PNG/JPEG: upload a single slice (it will be replicated into "
            "a thin 3D volume) or upload multiple slices -- they will be "
            "stacked in alphabetical filename order to form the depth axis."
        ),
    )
with col_b:
    use_demo = st.button("🧪 Use synthetic demo volume", type="secondary")


_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _load_single(file) -> np.ndarray:
    """Load one uploaded file into a [D, H, W] ndarray (float32)."""
    name = file.name.lower()

    # ---- NumPy ----
    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(file.read())).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr

    # ---- 2D image (PNG / JPEG) ----
    if name.endswith(_IMG_EXTS):
        from PIL import Image
        img = Image.open(io.BytesIO(file.read())).convert("L")  # grayscale
        slice_2d = np.asarray(img, dtype=np.float32) / 255.0     # -> [0, 1]
        return slice_2d[None, ...]                               # [1, H, W]

    # ---- NIfTI ----
    if not _HAS_NIB:
        raise RuntimeError(
            "nibabel is not installed; cannot read NIfTI. "
            "Install it via `pip install nibabel`."
        )
    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))   # [H,W,D] -> [D,H,W]
        return arr
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_uploaded(files) -> tuple[np.ndarray, str]:
    """
    Dispatch one-or-many uploads into a single [D, H, W] volume.

    Rules:
      * Multiple PNG/JPG files       -> stacked alphabetically along depth.
      * Multiple .nii / .npy files   -> only the first is used.
      * One 2D image / 2D .npy slice -> tiled to depth 16 so the 3D net
        has enough context for one sliding-window step.
    """
    if not isinstance(files, list):
        files = [files]

    # Sort PNG/JPG stacks alphabetically so slice order is deterministic.
    imgs = [f for f in files if f.name.lower().endswith(_IMG_EXTS)]
    others = [f for f in files if not f.name.lower().endswith(_IMG_EXTS)]

    if imgs and not others:
        imgs_sorted = sorted(imgs, key=lambda f: f.name.lower())
        slices = [_load_single(f) for f in imgs_sorted]   # each [1, H, W]
        # Verify consistent shape; resize would be nicer but stays simple here
        ref_h, ref_w = slices[0].shape[1:]
        ok = all(s.shape[1:] == (ref_h, ref_w) for s in slices)
        if not ok:
            raise ValueError(
                "Uploaded images have inconsistent shapes; please resize them "
                "to a common size before uploading."
            )
        vol = np.concatenate(slices, axis=0)   # [N, H, W]
        if vol.shape[0] == 1:
            # Single slice -> tile to a thin 3D stack so the U-Net has depth
            vol = np.repeat(vol, 16, axis=0)
            label = f"1 image (tiled to depth 16): {imgs_sorted[0].name}"
        else:
            label = f"{vol.shape[0]} stacked images: {imgs_sorted[0].name} … {imgs_sorted[-1].name}"
        return vol.astype(np.float32), label

    # Non-image path: use the first NIfTI / .npy.
    f = others[0]
    arr = _load_single(f)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 16, axis=0)
    return arr.astype(np.float32), f"uploaded: {f.name}"


# Decide input source
volume: np.ndarray | None = None
gt_mask: np.ndarray | None = None
source_label: str = ""

if upload:   # truthy when at least one file is selected
    try:
        volume, source_label = _load_uploaded(upload)
    except Exception as e:
        st.error(f"Failed to read upload: {e}")

if volume is None and use_demo:
    volume, gt_mask = make_demo_volume((96, 128, 128))
    source_label = "synthetic demo volume"

if volume is None:
    st.info(
        "Upload a `.nii.gz` / `.nii` / `.npy` CT volume above, "
        "or click **🧪 Use synthetic demo volume** to try the UI."
    )
    st.stop()

st.success(f"Loaded volume from {source_label} — shape `{volume.shape}` (D, H, W).")


# Auto-downsample large volumes so CPU inference stays interactive.
def _downsample(vol: np.ndarray, target_max: int) -> np.ndarray:
    """Stride-downsample so the largest axis <= target_max. Cheap & lossy."""
    longest = max(vol.shape)
    if longest <= target_max:
        return vol
    step = int(np.ceil(longest / target_max))
    return vol[::step, ::step, ::step]


original_shape = volume.shape
volume = _downsample(volume, max_dim)
if volume.shape != original_shape:
    st.info(
        f"⚡ Downsampled from `{original_shape}` -> `{volume.shape}` "
        f"to keep inference fast on CPU. "
        f"Increase the 'Max volume dim' slider in the sidebar for higher fidelity."
    )

# Grad-CAM forces a full-volume forward+backward (memory-heavy). Disable on
# anything bigger than the synthetic demo unless the user is on GPU.
effective_gradcam = with_gradcam
if with_gradcam and np.prod(volume.shape) > 96 * 128 * 128 and str(engine.device) == "cpu":
    st.warning(
        "Grad-CAM disabled for this run: volume is too large for full-volume "
        "backprop on CPU. Lower the 'Max volume dim' slider (e.g. 128) and "
        "re-run, or use the synthetic demo, to see Grad-CAM."
    )
    effective_gradcam = False


# =====================================================================
# Run inference
# =====================================================================
st.subheader("2. Inference")

with st.spinner("Running sliding-window 3D inference..."):
    import time
    t0 = time.time()
    result: InferenceResult = engine.predict(
        volume,
        threshold=threshold,
        min_voxels=min_voxels,
        with_gradcam=effective_gradcam,
    )
    elapsed = time.time() - t0
st.caption(f"Inference took {elapsed:.1f} s on `{engine.device}`.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodules found", f"{result.n_nodules}")
c2.metric("Mean confidence", f"{result.confidence:.3f}")
c3.metric("Predicted voxels", f"{int(result.mask.sum()):,}")
c4.metric("Volume shape", f"{result.mask.shape}")


# =====================================================================
# Axial slice viewer with overlays
# =====================================================================
st.subheader("3. Axial slice viewer")

D = result.ct_normalized.shape[0]
# Default to a slice that contains a prediction, if any
default_slice = D // 2
pos_slices = np.where(result.mask.sum(axis=(1, 2)) > 0)[0]
if pos_slices.size:
    default_slice = int(pos_slices[len(pos_slices) // 2])

slice_idx = st.slider("Axial slice index", 0, D - 1, default_slice)

import matplotlib.pyplot as plt  # local import to avoid slow cold-start

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ct_slice = result.ct_normalized[slice_idx]
mask_slice = result.mask[slice_idx]
prob_slice = result.probability[slice_idx]

axes[0].imshow(ct_slice, cmap="gray")
axes[0].set_title(f"CT (slice {slice_idx})")
axes[0].axis("off")

axes[1].imshow(ct_slice, cmap="gray")
axes[1].imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
               cmap="autumn", alpha=0.55)
axes[1].set_title("Predicted mask")
axes[1].axis("off")

axes[2].imshow(ct_slice, cmap="gray")
axes[2].imshow(prob_slice, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
axes[2].set_title("Probability heatmap")
axes[2].axis("off")

if result.gradcam is not None:
    axes[3].imshow(ct_slice, cmap="gray")
    axes[3].imshow(result.gradcam[slice_idx], cmap="jet", alpha=0.5)
    axes[3].set_title("Grad-CAM")
else:
    axes[3].imshow(ct_slice, cmap="gray")
    axes[3].set_title("Grad-CAM (disabled)")
axes[3].axis("off")

st.pyplot(fig, clear_figure=True)


# =====================================================================
# Multi-plane view
# =====================================================================
st.subheader("4. Multi-plane view (axial / sagittal / coronal)")

D, H, W = result.ct_normalized.shape
mid = (D // 2, H // 2, W // 2)
fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
for i, (label, sl) in enumerate(zip(
    ["Axial", "Sagittal", "Coronal"],
    [result.ct_normalized[mid[0], :, :],
     result.ct_normalized[:, mid[1], :],
     result.ct_normalized[:, :, mid[2]]],
)):
    overlay = [result.mask[mid[0], :, :],
               result.mask[:, mid[1], :],
               result.mask[:, :, mid[2]]][i]
    ax2[i].imshow(sl, cmap="gray")
    ax2[i].imshow(np.ma.masked_where(overlay == 0, overlay),
                  cmap="autumn", alpha=0.55)
    ax2[i].set_title(label)
    ax2[i].axis("off")
st.pyplot(fig2, clear_figure=True)


# =====================================================================
# Per-nodule report
# =====================================================================
st.subheader("5. Detected nodules")
if result.n_nodules == 0:
    st.info("No nodules detected above the current threshold.")
else:
    import pandas as pd

    df = pd.DataFrame({
        "Nodule #": np.arange(1, result.n_nodules + 1),
        "Volume (voxels)": result.nodule_volumes_mm3,
    })
    df["Confidence"] = [f"{result.confidence:.3f}"] * result.n_nodules
    st.dataframe(df, use_container_width=True)


# =====================================================================
# Download mask
# =====================================================================
st.subheader("6. Download")

mask_bytes = io.BytesIO()
np.save(mask_bytes, result.mask.astype(np.uint8))
st.download_button(
    "⬇️ Download predicted mask (.npy)",
    data=mask_bytes.getvalue(),
    file_name="prediction_mask.npy",
    mime="application/octet-stream",
)

if _HAS_NIB:
    # Save as NIfTI too
    nii = nib.Nifti1Image(
        np.transpose(result.mask.astype(np.uint8), (1, 2, 0)),
        affine=np.eye(4),
    )
    nii_path = os.path.join(tempfile.gettempdir(), "prediction_mask.nii.gz")
    nib.save(nii, nii_path)
    with open(nii_path, "rb") as fh:
        st.download_button(
            "⬇️ Download predicted mask (.nii.gz)",
            data=fh.read(),
            file_name="prediction_mask.nii.gz",
            mime="application/gzip",
        )

st.caption(
    "Research prototype only — **not** a medical device. "
    "Predictions must not be used for clinical decision-making."
)
