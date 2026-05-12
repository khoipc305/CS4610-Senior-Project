# Enhanced Lung Cancer Nodule Detection - Spring 2026

## Project Overview
This project builds upon the Fall 2025 Senior Project to significantly improve lung cancer nodule detection and segmentation performance. Key enhancements address the limitations identified in the baseline model (Dice: 0.54).

**Student:** Khoi Nguyen Pham  
**Advisor:** Professor Hao Ji  
**Course:** CS4620 - Spring 2026  
**Base Project:** Fall 2025 Lung Cancer Detection (3D U-Net)

## Improvements Over Fall Project

### Fall Project Results (Baseline)
- **Dice Score:** 0.5415 (moderate performance)
- **Architecture:** Basic 3D U-Net
- **Patch Size:** 96×96×96
- **Training:** CPU-only, 50 epochs
- **Dataset:** 61 samples (48 train, 13 val)
- **Main Issues:** Class imbalance, small patches, no attention mechanism

### Spring Enhancements

#### 1. **Advanced Model Architecture** 🧠
- ✨ **Attention U-Net:** Self-attention and channel attention mechanisms
- ✨ **Deep Supervision:** Multi-scale loss for better feature learning
- ✨ **Residual Connections:** Improved gradient flow
- ✨ **Larger capacity:** More feature channels [32, 64, 128, 256, 512]

#### 2. **Improved Data Processing** 📊
- ✨ **Larger patches:** 128×128×128 (vs 96×96×96) for more context
- ✨ **Advanced augmentation:** Elastic deformations, intensity shifts, noise
- ✨ **Smart sampling:** Balanced positive/negative patches with hard mining
- ✨ **Better normalization:** Adaptive windowing for lung tissue

#### 3. **Advanced Loss Functions** 🎯
- ✨ **Focal Loss:** Addresses class imbalance by focusing on hard examples
- ✨ **Tversky Loss:** Better for handling imbalanced segmentation
- ✨ **Combined Loss:** Focal + Dice + BCE with optimal weights
- ✨ **Deep supervision:** Auxiliary losses at multiple scales

#### 4. **Training Optimizations** ⚡
- ✨ **GPU Support:** CUDA-enabled for 10-20x faster training
- ✨ **Mixed Precision:** FP16 training for memory efficiency
- ✨ **Larger batches:** 4-8 samples (vs 2) with gradient accumulation
- ✨ **Advanced scheduler:** CosineAnnealingWarmRestarts
- ✨ **Early stopping:** Prevents overfitting

#### 5. **Comprehensive Evaluation** 📈
- ✨ **Cross-validation:** 5-fold CV for robust performance estimates
- ✨ **Multiple metrics:** Dice, IoU, Precision, Recall, Hausdorff Distance
- ✨ **Statistical analysis:** Confidence intervals, significance tests
- ✨ **Visualization:** 3D interactive predictions, attention maps

#### 6. **Additional Features** 🚀
- ✨ **TensorBoard integration:** Real-time monitoring
- ✨ **Model ensemble:** Combine multiple models for better performance
- ✨ **Post-processing:** Connected components, morphological operations
- ✨ **Web interface:** Gradio app for easy inference

## Project Structure

```
d:\Spring project\
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config/                            # Configuration files
│   ├── config_baseline.yaml          # Baseline configuration
│   └── config_advanced.yaml          # Advanced configuration
├── src/                               # Source code
│   ├── models/                       # Model architectures
│   │   ├── attention_unet.py         # Attention U-Net
│   │   ├── deep_supervision.py       # Deep supervision module
│   │   └── model_factory.py          # Model builder
│   ├── data/                         # Data processing
│   │   ├── dataset.py                # Enhanced dataset class
│   │   ├── transforms.py             # Advanced augmentations
│   │   └── preprocessing.py          # Data preprocessing
│   ├── training/                     # Training utilities
│   │   ├── trainer.py                # Main training loop
│   │   ├── losses.py                 # Custom loss functions
│   │   └── metrics.py                # Evaluation metrics
│   ├── utils/                        # Utilities
│   │   ├── visualization.py          # Plotting and viz
│   │   └── logger.py                 # Logging utilities
│   └── inference/                    # Inference tools
│       ├── predict.py                # Single prediction
│       └── ensemble.py               # Ensemble inference
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Data analysis
│   ├── 02_model_training.ipynb       # Training experiments
│   └── 03_evaluation.ipynb           # Results analysis
├── scripts/                           # Standalone scripts
│   ├── train.py                      # Main training script
│   ├── evaluate.py                   # Evaluation script
│   └── cross_validate.py             # Cross-validation
├── experiments/                       # Experiment outputs
│   └── runs/                         # TensorBoard logs
├── checkpoints/                       # Saved models
└── docs/                             # Documentation
    ├── METHODOLOGY.md                # Technical details
    └── RESULTS.md                    # Results and analysis
```

## Installation & Setup

### Prerequisites
```bash
# Create conda environment
conda create -n lung_enhanced python=3.10
conda activate lung_enhanced

# Install PyTorch with CUDA (if GPU available)
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup
Link to Fall project data:
```bash
# The dataset is in: D:\Fall Senior Project\LIDC-exact\
# We'll reference it in config files to avoid duplication
```

## Alignment with Progress Report 1

**✅ This implementation includes ALL components mentioned in your Progress Report 1:**

1. ✅ **Statistical Baseline Models** - Logistic regression, Random Forest, SVM
2. ✅ **Deep Learning CNNs** - Attention U-Net, Deep Supervision
3. ✅ **Transfer Learning** - ResNet, DenseNet with fine-tuning strategies
4. ✅ **Evaluation Metrics** - Sensitivity, Specificity, ROC-AUC, Confusion Matrix
5. ✅ **Model Interpretability** - Grad-CAM, attention visualization

See `ALIGNMENT_WITH_PROGRESS_REPORT.md` for detailed mapping.

---

## Quick Start

### 1. Train Statistical Baseline (from Progress Report 1)
```bash
# Train logistic regression baseline
python scripts/train_baseline_statistical.py --model logistic

# Compare all statistical models
python scripts/train_baseline_statistical.py --model all
```

### 2. Train Enhanced Deep Learning Model
```bash
# GPU training with advanced features
python scripts/train.py --config config/config_advanced.yaml --gpu 0

# Resume from checkpoint
python scripts/train.py --config config/config_advanced.yaml --resume checkpoints/last.pth
```

### 2. Cross-Validation
```bash
python scripts/cross_validate.py --config config/config_advanced.yaml --folds 5
```

### 3. Evaluate Model
```bash
python scripts/evaluate.py --model checkpoints/best_model.pth --data_dir "D:/Fall Senior Project/LIDC-exact"
```

### 4. Run Inference
```python
from src.inference.predict import predict_single

result = predict_single(
    ct_path="path/to/ct.nii.gz",
    model_path="checkpoints/best_model.pth",
    output_dir="predictions"
)
```

## Expected Improvements

### Performance Targets
| Metric | Fall (Baseline) | Spring (Target) | Improvement |
|--------|----------------|-----------------|-------------|
| Dice Score | 0.54 | **0.70-0.75** | +30-40% |
| IoU | 0.37 | **0.55-0.60** | +50-60% |
| Sensitivity | Unknown | **0.75-0.80** | New |
| Specificity | Unknown | **0.95-0.98** | New |
| Training Time | 4.33h (CPU) | **1-2h (GPU)** | 2-4x faster |

### Key Innovations
1. **Attention mechanisms** → Better feature focus on nodules
2. **Focal loss** → Handles class imbalance effectively
3. **Larger patches** → More spatial context
4. **Deep supervision** → Better gradient flow
5. **Cross-validation** → Robust performance estimates

## Technical Stack

- **Deep Learning:** PyTorch 2.0+, MONAI 1.3+
- **Medical Imaging:** nibabel, SimpleITK
- **Visualization:** TensorBoard, Matplotlib, Plotly
- **Data Science:** NumPy, Pandas, scikit-learn
- **Utilities:** YAML, tqdm, wandb (optional)

## Evaluation Metrics

### Primary Metrics
- **Dice Similarity Coefficient (DSC)**
- **Intersection over Union (IoU)**
- **Sensitivity (Recall)**
- **Specificity**

### Additional Metrics
- **Precision**
- **Hausdorff Distance (95th percentile)**
- **Average Surface Distance**
- **Volumetric Similarity**

## Timeline (Spring 2026)

- ✅ **Week 1-2:** Setup and baseline reimplementation
- 🔄 **Week 3-4:** Implement attention mechanisms and focal loss
- 📅 **Week 5-6:** Advanced augmentation and data processing
- 📅 **Week 7-8:** GPU training and hyperparameter tuning
- 📅 **Week 9-10:** Cross-validation and ensemble methods
- 📅 **Week 11-12:** Evaluation, visualization, and documentation
- 📅 **Week 13-14:** Final report and presentation preparation

## References

### From Fall Project
- LIDC-IDRI Dataset
- MONAI Framework
- 3D U-Net Architecture

### New References
- Attention U-Net: Oktay et al., 2018
- Focal Loss: Lin et al., 2017
- Deep Supervision: Lee et al., 2015
- Tversky Loss: Salehi et al., 2017

## Acknowledgments

This project builds upon the Fall 2025 Senior Project baseline. Special thanks to Professor Hao Ji for guidance and support.

---

**Last Updated:** April 17, 2026  
**Status:** 🚀 Active Development
