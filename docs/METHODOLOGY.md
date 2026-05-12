# Technical Methodology - Spring 2026 Enhancements

## Overview

This document details the technical improvements made to the Fall 2025 lung cancer detection project.

## 1. Model Architecture Enhancements

### Baseline (Fall 2025)
```
3D U-Net
- Channels: [16, 32, 64, 128, 256]
- Parameters: ~4.8M
- No attention mechanisms
- Single-scale output
```

### Enhanced (Spring 2026)
```
Attention 3D U-Net
- Channels: [32, 64, 128, 256, 512]
- Parameters: ~19M (4x larger)
- Spatial attention at each encoder level
- Optional channel attention
- Deep supervision at multiple scales
```

### Attention Mechanism

**Spatial Attention:** Focuses on relevant spatial locations
```python
attention_map = Sigmoid(Conv3D(features))
output = features * attention_map
```

**Channel Attention:** Emphasizes important feature channels
```python
channel_weights = Sigmoid(AdaptiveAvgPool + Conv3D)
output = features * channel_weights
```

**Benefits:**
- Better focus on small nodule regions
- Improved gradient flow
- More discriminative features

## 2. Loss Function Improvements

### Baseline
- DiceCELoss (50% Dice + 50% Cross-Entropy)

### Enhanced
- **Combined Loss:** Focal (50%) + Dice (30%) + CE (20%)

#### Focal Loss
Addresses class imbalance by down-weighting easy examples:

```
FL(pt) = -α(1-pt)^γ * log(pt)
```

Where:
- `pt`: probability of correct class
- `α`: class balancing weight (0.75 for positive class)
- `γ`: focusing parameter (2.0)

**Impact:** Focuses training on hard-to-classify voxels (nodule boundaries)

#### Tversky Loss (Alternative)
Controls FP/FN trade-off:

```
TL = 1 - (TP) / (TP + α*FN + β*FP)
```

Where:
- `α = 0.7`: penalize false negatives more (avoid missing nodules)
- `β = 0.3`: tolerate some false positives

## 3. Data Processing Pipeline

### Patch Extraction

**Baseline:** 96×96×96 patches
**Enhanced:** 128×128×128 patches

**Rationale:** Larger receptive field captures more context around nodules.

### Smart Sampling

**Positive Patch Sampling:**
1. Find all voxels with label = 1
2. Sample center near positive voxel
3. Add randomness (±32 voxels) to increase diversity

**Negative Patch Sampling:**
1. Random sampling from entire volume
2. Hard negative mining (optional): sample near nodules but not containing them

**Ratio:** 2:1 positive to negative (66% positive patches)

### Augmentation Pipeline

#### Geometric Transforms
- Random flips (50% prob, all axes)
- Random rotations (±15°, 50% prob)
- Random scaling (0.9-1.1x, 50% prob)
- Elastic deformations (30% prob) **NEW**

#### Intensity Transforms
- Gaussian noise (σ=0.1, 20% prob) **NEW**
- Gaussian smoothing (σ=0.5-1.0, 20% prob) **NEW**
- Intensity shift (±10%, 30% prob) **NEW**
- Intensity scaling (±10%, 30% prob) **NEW**

## 4. Training Optimizations

### Baseline
- Optimizer: Adam (lr=1e-4)
- Scheduler: ReduceLROnPlateau
- Batch size: 2
- Device: CPU

### Enhanced
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingWarmRestarts
- Batch size: 4-8 (with gradient accumulation)
- Device: GPU with mixed precision (FP16)
- Early stopping (patience=15)

### Mixed Precision Training

Uses `torch.cuda.amp` for faster training:
- Forward/backward in FP16 (2x faster)
- Master weights in FP32 (stable)
- Automatic loss scaling

**Benefits:**
- 2-3x speedup
- 40% less memory
- Same accuracy

### Learning Rate Schedule

**Warmup Phase (5 epochs):**
- Gradually increase LR from 0 to 1e-4
- Stabilizes training with random initialization

**Cosine Annealing:**
- Smoothly decrease LR to 1e-6
- Periodic restarts for better convergence

## 5. Evaluation Metrics

### Overlap Metrics
- **Dice Coefficient:** 2*TP / (2*TP + FP + FN)
- **IoU (Jaccard):** TP / (TP + FP + FN)

### Classification Metrics
- **Sensitivity (Recall):** TP / (TP + FN) - catch all nodules
- **Specificity:** TN / (TN + FP) - avoid false alarms
- **Precision:** TP / (TP + FP) - prediction accuracy

### Distance Metrics
- **Hausdorff Distance (95%):** Maximum boundary distance
- **Average Surface Distance:** Mean boundary distance

**Usage:** Distance metrics detect boundary errors that overlap metrics miss.

## 6. Deep Supervision (Optional)

Adds auxiliary losses at decoder layers:

```
Total Loss = Main Loss + 0.5*DS_1 + 0.25*DS_2 + 0.125*DS_3
```

**Benefits:**
- Better gradient flow to encoder
- Multi-scale feature learning
- Faster convergence

## 7. Post-Processing

Applied during inference:
1. **Connected Components:** Remove isolated false positives
2. **Size Filtering:** Remove objects < 10 voxels
3. **Morphological Closing:** Fill small holes

## Expected Performance Improvements

| Component | Expected Gain |
|-----------|---------------|
| Attention mechanism | +5-10% Dice |
| Focal loss | +10-15% Dice |
| Larger patches | +3-5% Dice |
| Better augmentation | +2-3% Dice |
| GPU training | 10-20x faster |

**Total Expected:** 0.54 → 0.70-0.75 Dice (+30-40%)

## References

1. Oktay et al. (2018) - Attention U-Net
2. Lin et al. (2017) - Focal Loss
3. Salehi et al. (2017) - Tversky Loss
4. Ronneberger et al. (2015) - U-Net
5. MONAI Framework - Medical Imaging Tools

---

**Document Version:** 1.0
**Last Updated:** April 17, 2026
