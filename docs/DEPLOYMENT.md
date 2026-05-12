# Deployment App — How to Run

The Streamlit deployment app fulfills **Phase 5 / Expected Outcome 4** of the
Spring 2026 project proposal: *"an interactive application demonstrating
end-to-end CT inference and interpretability"*.

## 1. Install dependencies

```powershell
cd "d:\Spring project"
pip install -r requirements.txt
```

The relevant new dependency is `streamlit`. PyTorch, MONAI, and nibabel
are already in `requirements.txt`.

## 2. (Optional) Place a trained checkpoint

The app expects a checkpoint at:

```
checkpoints/lung_cancer_attention_unet/best_model.pth
```

If you have one (produced by `scripts/train.py`), the app will load it
automatically. **If no checkpoint is found, the app still runs** but with
untrained weights — useful for a UI walkthrough but the predictions will be
meaningless. A "Use synthetic demo volume" button is provided so you can
demo the full pipeline offline.

The checkpoint path can also be edited live from the sidebar.

## 3. Launch the app

```powershell
streamlit run app.py
```

This opens a browser tab at `http://localhost:8501`. The app has six
sections:

1. **Input CT volume** — upload `.nii`, `.nii.gz`, or `.npy`, **or** click
   *Use synthetic demo volume* for a built-in test case.
2. **Inference** — runs MONAI sliding-window 3D inference; reports nodule
   count, mean confidence, and predicted-voxel total.
3. **Axial slice viewer** — slice slider with four panels: CT, predicted
   mask overlay, probability heatmap, Grad-CAM overlay.
4. **Multi-plane view** — axial / sagittal / coronal mid-slices with mask
   overlays.
5. **Detected nodules** — table of per-component voxel counts.
6. **Download** — predicted mask as `.npy` and `.nii.gz`.

Sidebar controls let you change the checkpoint path, choose
`auto / cuda / cpu`, adjust the probability threshold, and the minimum
nodule size (in voxels) for the connected-components filter.

## 4. Recording the demo

For the final video presentation:

1. Start the app: `streamlit run app.py`.
2. Open OBS Studio (or the PowerPoint screen recorder).
3. Click *Use synthetic demo volume*, walk through the four panels.
4. Adjust the threshold slider live to show sensitivity/specificity
   trade-off.
5. Show the download buttons.

## 5. Troubleshooting

- **`ModuleNotFoundError: monai`** — run `pip install -r requirements.txt`.
- **CUDA OOM** — switch the sidebar device to `cpu`, or lower the
  ROI/sliding-window batch size in `src/inference/engine.py`.
- **NIfTI orientation looks wrong** — `app.py` transposes from `[H, W, D]`
  (typical NIfTI) to `[D, H, W]`. If your file is already `[D, H, W]`,
  remove the `np.transpose` call in `_load_uploaded`.
- **No nodules shown but ground-truth has them (untrained model)** —
  expected. Train via `python scripts/train.py --config config/config_advanced.yaml`
  first, then point the sidebar at the resulting `best_model.pth`.

## 6. Files involved

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `src/inference/engine.py` | `InferenceEngine` (load model, sliding-window inference, post-process, Grad-CAM) |
| `src/inference/__init__.py` | Public exports |
| `requirements.txt` | Adds `streamlit>=1.30.0` |
