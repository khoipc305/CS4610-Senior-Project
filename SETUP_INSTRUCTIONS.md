# Setup Instructions - Enhanced Lung Cancer Detection

## Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA 11.8+ (recommended)
- 16GB RAM minimum
- Access to Fall project data

## Step-by-Step Setup

### 1. Environment Setup

```powershell
# Open PowerShell
# Navigate to project directory
cd "d:\Spring project"

# Create conda environment
conda create -n lung_enhanced python=3.10 -y
conda activate lung_enhanced
```

### 2. Install PyTorch

**For GPU (Recommended):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```powershell
pip install torch torchvision
```

**Verify Installation:**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True # or False for CPU
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- MONAI (medical imaging framework)
- nibabel (NIfTI file handling)
- scikit-learn (metrics)
- matplotlib, seaborn (visualization)
- tensorboard (monitoring)
- pandas, numpy (data processing)

### 4. Verify Data Access

Check that Fall project data is accessible:

```powershell
# Check CT scans
dir "D:\Fall Senior Project\LIDC-exact\ct"

# Check masks
dir "D:\Fall Senior Project\LIDC-exact\masks"

# Check manifests
dir "D:\Fall Senior Project\LIDC-clean\*.csv"
```

You should see:
- CT directory: ~61 .nii.gz files
- Masks directory: ~61 .nii.gz files
- Manifest files: train_manifest.csv, val_manifest.csv

### 5. Test Installation

```powershell
# Test model creation
python -c "from src.models.attention_unet import create_model; print(' Models working')"

# Test loss functions
python -c "from src.training.losses import FocalLoss; print(' Loss functions working')"

# Test dataset
python src/data/dataset.py
```

### 6. Optional: Setup TensorBoard

```powershell
# In a separate terminal
conda activate lung_enhanced
cd "d:\Spring project"
tensorboard --logdir=checkpoints
```

Then open http://localhost:6006 in your browser.

## Quick Test Run

Run a short training test to verify everything works:

```powershell
# This will train for just 2 epochs as a test
python scripts/train.py --config config/config_baseline.yaml
```

Press Ctrl+C after 1-2 minutes to stop. If no errors, you're ready!

## Common Issues

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in config file:
```yaml
training:
 batch_size: 2 # Instead of 4
```

### Issue: "No module named 'src'"
**Solution:** Run scripts from project root:
```powershell
cd "d:\Spring project"
python scripts/train.py ...
```

### Issue: "FileNotFoundError" for data
**Solution:** Check paths in config files match your data location.

### Issue: Import errors
**Solution:** Reinstall dependencies:
```powershell
pip install --upgrade -r requirements.txt
```

### Issue: Slow training on CPU
**Solution:**
- Use GPU if available
- Reduce patch size to 96³
- Reduce `samples_per_volume` to 2

## Next Steps

Once setup is complete:
1. Read `docs/QUICK_START.md` for training guide
2. Review `config/config_advanced.yaml` for options
3. Start training with `python scripts/train.py --config config/config_advanced.yaml`

## Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1060 (6GB) | RTX 3060+ (12GB) |
| RAM | 16GB | 32GB |
| Storage | 50GB free | 100GB free |
| CPU | 4 cores | 8+ cores |

## Support

If you encounter issues:
1. Check error messages carefully
2. Review Fall project setup for comparison
3. Verify all paths in config files
4. Check GPU/CUDA availability

---

**Estimated Setup Time:** 15-30 minutes
**Last Updated:** April 17, 2026
