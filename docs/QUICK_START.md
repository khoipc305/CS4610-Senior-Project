# Quick Start Guide - Enhanced Lung Cancer Detection

## Quick Setup (5 minutes)

### 1. Install Dependencies

```powershell
# Create conda environment
conda create -n lung_enhanced python=3.10
conda activate lung_enhanced

# Navigate to project
cd "d:\Spring project"

# Install PyTorch (GPU version - recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify Data Access

A 2-scan **smoke-test sample** ships with the repo at
`dataset/sample/` (~70 MB) and is the default for every config and
script -- so a fresh clone runs immediately on any machine.

```
dataset/sample/
├── ct/      # 2 .nii.gz CT volumes
└── masks/   # 2 matching segmentation masks
```

For full training/evaluation, point the code at your own copy of
LIDC-IDRI by setting `LIDC_DATA_ROOT`, passing `--data_root`, or editing
`config/config_advanced.yaml`. The expected layout is:

```
<your_data_root>/
├── ct/      *.nii.gz
└── masks/   *_segmask.nii.gz
```

See `dataset/README.md` for download instructions.

### 3. Train the Model

#### Option A: Advanced Model (Recommended)
```powershell
python scripts/train.py --config config/config_advanced.yaml
```

This will:
- Use attention mechanisms
- Apply focal loss for class imbalance
- Use 128×128×128 patches
- Train for 100 epochs
- Save to `checkpoints/lung_cancer_attention_unet/`

#### Option B: Baseline Model (for comparison)
```powershell
python scripts/train.py --config config/config_baseline.yaml
```

This replicates your Fall project setup for comparison.

## Monitor Training

### TensorBoard
```powershell
tensorboard --logdir=checkpoints/lung_cancer_attention_unet/tensorboard
```

Then open http://localhost:6006 in your browser.

### Check Progress
Training outputs are saved in:
- `checkpoints/lung_cancer_attention_unet/best_model.pth` - Best model
- `checkpoints/lung_cancer_attention_unet/training_history.csv` - Metrics
- `checkpoints/lung_cancer_attention_unet/tensorboard/` - TensorBoard logs

## Expected Results

### Training Time
- **GPU (RTX 3060 or better):** 1-2 hours for 100 epochs
- **CPU:** 8-12 hours (not recommended)

### Performance Targets
| Metric | Fall (Baseline) | Spring (Target) |
|--------|----------------|-----------------|
| Dice Score | 0.54 | **0.70-0.75** |
| IoU | 0.37 | **0.55-0.60** |
| Sensitivity | - | **0.75-0.80** |

## Troubleshooting

### Out of Memory (GPU)
Reduce batch size in config:
```yaml
training:
 batch_size: 2 # Instead of 4
```

### Slow Training
- Use GPU if available
- Reduce patch size to 96×96×96
- Reduce `samples_per_volume` to 2

### CUDA Not Available
Check your PyTorch installation:
```python
import torch
print(torch.cuda.is_available()) # Should be True
print(torch.cuda.get_device_name(0))
```

## Key Improvements Over Fall Project

1. **Attention Mechanisms** - Model focuses on nodule regions
2. **Focal Loss** - Handles class imbalance (nodules are small)
3. **Larger Patches** - 128³ vs 96³ for more context
4. **Better Augmentation** - Elastic deformations, noise, etc.
5. **Deep Supervision** - Multi-scale learning
6. **Advanced Metrics** - Hausdorff distance, sensitivity, etc.

## Next Steps

After training completes:
1. Review results in `training_history.csv`
2. Check TensorBoard for training curves
3. Run evaluation script for detailed metrics
4. Generate visualizations for your report

## Need Help?

- Check `README.md` for full documentation
- See `config/config_advanced.yaml` for all options
- Review the Fall 2025 baseline write-up in `PROJECT_SUMMARY.md` and `docs/METHODOLOGY.md`

---

**Estimated time to first results:** 1-2 hours (GPU) | 8-12 hours (CPU)
