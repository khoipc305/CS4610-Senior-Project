# Enhanced 3D Lung Cancer Nodule Detection

CS4620 Senior Project, Spring 2026 — Khoi Nguyen Pham, advised by Prof. Hao Ji.
This repository extends the Fall 2025 3D U-Net baseline (validation Dice 0.5415)
into an Attention U-Net pipeline with composite Focal+Dice+CE loss, plus a
Streamlit deployment app, statistical baselines, transfer-learning baselines,
and Grad-CAM interpretability.

---

## What's new vs. the Fall 2025 baseline

| Component | Fall 2025 | Spring 2026 |
|---|---|---|
| Architecture | 3D U-Net, 4.8M params | Attention U-Net + SE channel gates, ~19M params |
| Loss | Dice + CE | 0.5 Focal + 0.3 Dice + 0.2 CE (alpha 0.75, gamma 2.0) |
| Patch size | 96^3 | 128^3 with 2:1 positive/negative sampling |
| Augmentation | Flips + small rotations | + elastic, intensity, Gaussian noise, scaling |
| Hardware | CPU only | GPU + FP16 mixed precision |
| Training time | 4.3 h (CPU) | ~1.5 h (GPU) |
| Validation Dice | 0.5415 | ~0.70 (target, +30% relative) |
| Deliverables | training pipeline only | + statistical & transfer baselines, Grad-CAM, Streamlit app |

---

## Repository layout

```
.
|-- app.py                     <- Streamlit deployment app
|-- requirements.txt
|-- config/
|   |-- config_baseline.yaml   <- Fall-style baseline configuration
|   `-- config_advanced.yaml   <- Attention U-Net + composite loss
|-- dataset/
|   `-- sample/                <- bundled 71 MB smoke-test data (see below)
|       |-- ct/                <- 2 NIfTI CT volumes (with nodules)
|       |-- masks/             <- matching ground-truth masks
|       `-- png_slices/        <- 13 PNG axial slices around the nodule
|-- docs/
|   |-- DEPLOYMENT.md          <- how to deploy the Streamlit app
|   |-- METHODOLOGY.md         <- technical write-up of methods
|   `-- QUICK_START.md         <- terse "first-run" recipe
|-- scripts/
|   |-- train.py                          <- main training entry point
|   |-- evaluate.py                       <- segmentation evaluation
|   `-- train_baseline_statistical.py     <- LR / RF / SVM baselines
|-- src/
|   |-- data/{dataset.py, transforms.py}
|   |-- inference/engine.py               <- 3D inference + Grad-CAM (used by app.py)
|   |-- models/
|   |   |-- attention_unet.py             <- AttentionUNet3D
|   |   |-- baseline_statistical.py       <- LR / RF / SVM
|   |   `-- transfer_learning.py          <- 3D ResNet / DenseNet
|   |-- training/{losses.py, metrics.py}
|   `-- utils/visualization.py
|-- SETUP_INSTRUCTIONS.md      <- end-to-end environment setup
`-- PROJECT_SUMMARY.md         <- proposal-vs-deliverables alignment
```

---

## Installation

```powershell
# Recommended: a clean conda env
conda create -n lung_enhanced python=3.10 -y
conda activate lung_enhanced

# Install PyTorch -- pick ONE
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # GPU (CUDA 11.8)
pip install torch torchvision                                                       # CPU only

# Project dependencies
pip install -r requirements.txt
```

See `SETUP_INSTRUCTIONS.md` for a longer walkthrough including verification
snippets.

---

## Dataset

### Bundled smoke-test sample (~71 MB, ships with the repo)

`dataset/sample/` contains everything a reviewer needs to run the deployment
app end-to-end without downloading anything else:

```
dataset/sample/
|-- ct/                                            (2 .nii.gz, ~70 MB)
|   |-- LIDC-IDRI-0006_3000563.nii.gz              (nodule, 389 voxels)
|   `-- LIDC-IDRI-0006_3000567.nii.gz              (nodule, 191 voxels)
|-- masks/                                         (matching segmentations)
`-- png_slices/                                    (13 PNGs, ~640 KB)
    `-- slice_01.png ... slice_13.png              (z = 64..76, nodule on 05-08)
```

Both bundled NIfTI scans contain confirmed nodules, and the 13 PNGs are a
lung-windowed export ([-1000, +400] HU rescaled to 0-255) of the nodule region
from the first scan. All files default in every config and CLI script, so a
fresh clone runs immediately on any machine. **The sample is for demoing the
app, not for training the model.**

### Demoing the app with the sample

```powershell
py -m streamlit run app.py
```

Then drag-and-drop into the file uploader, fastest to most realistic:

| Demo | What to upload | What happens |
|---|---|---|
| **PNG single slice** (~10 s) | `dataset/sample/png_slices/slice_07.png` | Tiled along depth axis, runs inference on a thin pseudo-3D volume. |
| **PNG multi-slice stack** (~25 s) | All 13 PNGs in `png_slices/` at once | Stacked alphabetically into a 13-slice 3D volume; Grad-CAM peaks near slice 07. |
| **Full 3D NIfTI** (~60 s on CPU) | `dataset/sample/ct/LIDC-IDRI-0006_3000563.nii.gz` | Sliding-window inference over the full 512x512x133 volume. |

> Without a trained checkpoint at
> `checkpoints/lung_cancer_attention_unet/best_model.pth`, predictions are
> random — the app still demonstrates the full UI and inference loop, but the
> mask is meaningful only after training.

### Full LIDC-IDRI dataset (not redistributed)

The 61-scan preprocessed subset used for training/evaluation is held by the
author (~2.7 GB) and is **not** in this repo for size reasons.

**Option A — re-create from TCIA.** Register at
<https://www.cancerimagingarchive.net/>, download the LIDC-IDRI collection
(DICOM, ~125 GB raw), and use [pylidc](https://pylidc.github.io/) or this
repo's preprocessing in `src/data/` to convert to the `.nii.gz` layout below.

**Option B — request the prepared NIfTI subset.** Email **khoipc305@cpp.edu**
for grading / reproduction access.

The expected layout is:

```
<your_data_root>/
|-- ct/      *.nii.gz
`-- masks/   *_segmask.nii.gz   (matched to CT by removing _segmask)
```

Point the codebase at it in any of three equivalent ways:

```powershell
# (a) PowerShell environment variable (one shell)
$env:LIDC_DATA_ROOT = "D:/path/to/your/LIDC"

# (b) per-command CLI override
python scripts/train.py --config config/config_advanced.yaml --data_root D:/path/to/your/LIDC

# (c) edit the four `data:` entries in config/config_advanced.yaml
```

LIDC-IDRI is distributed under the **TCIA Limited Access Data Use Agreement**;
the bundled sample inherits the same license and is for research and
educational evaluation only. Predictions produced by this codebase are **not**
clinical decisions; the app shows a research-prototype disclaimer on every
page.

---

## Quick-start commands

```powershell
# 1. Statistical baselines (LR / RF / SVM on hand-crafted radiomic features)
python scripts/train_baseline_statistical.py --model all

# 2. Train the Attention U-Net on the full dataset
python scripts/train.py --config config/config_advanced.yaml --gpu 0
python scripts/train.py --config config/config_advanced.yaml --resume checkpoints/last.pth

# 3. Evaluate a trained checkpoint
python scripts/evaluate.py --model checkpoints/best_model.pth                      # bundled sample
python scripts/evaluate.py --model checkpoints/best_model.pth --data_dir <PATH>    # full LIDC

# 4. Run the deployment app
py -m streamlit run app.py
```

---

## Performance targets

| Metric | Fall baseline | Spring target | Improvement |
|---|---|---|---|
| Dice | 0.541 | 0.70 - 0.75 | +30 - 40% |
| IoU | 0.371 | 0.55 - 0.60 | +50 - 60% |
| Sensitivity | 0.62 | 0.75 - 0.80 | new |
| Specificity | 0.96 | 0.95 - 0.98 | maintained |
| Hausdorff95 | 14.2 mm | < 9 mm | tighter boundary |
| Wall-clock training | 4.3 h CPU | ~1.5 h GPU | ~3x faster |

---

## Technical stack

PyTorch 2.x, MONAI 1.3+, nibabel / SimpleITK, scikit-learn (statistical
baselines and metrics), Streamlit (deployment app), Pillow (PNG support),
TensorBoard (training curves), NumPy / pandas / matplotlib.

---

## References

1. Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical
   Image Segmentation.* MICCAI 2015.
2. Oktay et al. *Attention U-Net: Learning Where to Look for the Pancreas.*
   MIDL 2018, arXiv:1804.03999.
3. Lin, Goyal, Girshick, He, Dollar. *Focal Loss for Dense Object Detection.*
   ICCV 2017.
4. Armato III et al. *The Lung Image Database Consortium (LIDC) and Image
   Database Resource Initiative (IDRI).* Medical Physics 38(2), 2011.
5. MONAI Consortium. *Project MONAI: Medical Open Network for AI.* 2020,
   <https://monai.io>.

---

## Acknowledgments

Thanks to Prof. Hao Ji for continuous supervision across CS4610 and CS4620,
the National Cancer Institute's Cancer Imaging Archive for releasing
LIDC-IDRI, and the open-source MONAI and PyTorch communities. The Fall 2025
3D U-Net baseline (CS4610) is the starting point for this work.
