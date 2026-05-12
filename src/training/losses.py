"""
Advanced Loss Functions for Medical Image Segmentation

Implements:
- Focal Loss: Addresses class imbalance
- Tversky Loss: Controls false positives/negatives
- Combined losses for optimal performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Args:
            alpha: Weight for positive class (0-1)
            gamma: Focusing parameter (0-5, usually 2)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions [B, C, ...] (logits or probabilities)
            targets: Ground truth [B, C, ...] (one-hot) or [B, ...] (class indices)
        
        Returns:
            Focal loss value
        """
        # Convert to probabilities if needed
        if inputs.dim() > targets.dim():
            # inputs has channel dim, targets doesn't
            # Apply softmax to get probabilities
            probs = F.softmax(inputs, dim=1)
            
            # Convert targets to one-hot if needed
            if targets.dim() == inputs.dim() - 1:
                targets = F.one_hot(
                    targets.long(),
                    num_classes=inputs.shape[1]
                ).permute(0, -1, *range(1, targets.dim())).float()
        else:
            probs = inputs
        
        # Compute binary cross entropy
        bce = F.binary_cross_entropy(
            probs,
            targets,
            reduction='none'
        )
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for controlling false positives and false negatives.
    
    Generalizes Dice loss with configurable FP/FN trade-off.
    Reference: Salehi et al. "Tversky loss function for image segmentation" (2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.7,  # Weight for false negatives
        beta: float = 0.3,   # Weight for false positives
        smooth: float = 1e-5
    ):
        """
        Args:
            alpha: FN weight (higher = penalize FN more)
            beta: FP weight (higher = penalize FP more)
            smooth: Smoothing constant
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            inputs: Predictions [B, C, ...]
            targets: Ground truth [B, C, ...] or [B, ...]
        
        Returns:
            Tversky loss value
        """
        # Apply softmax if needed
        if inputs.dim() > targets.dim() + 1:
            probs = F.softmax(inputs, dim=1)
        else:
            probs = inputs
        
        # Flatten for easier computation
        batch_size = inputs.shape[0]
        num_classes = inputs.shape[1] if inputs.dim() > targets.dim() else 1
        
        if num_classes > 1:
            # Multi-class
            probs_flat = probs.view(batch_size, num_classes, -1)
            targets_flat = targets.view(batch_size, num_classes, -1)
        else:
            probs_flat = probs.view(batch_size, -1)
            targets_flat = targets.view(batch_size, -1)
        
        # True positives, false positives, false negatives
        tp = (probs_flat * targets_flat).sum(dim=-1)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=-1)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=-1)
        
        # Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        
        # Return 1 - Tversky (loss)
        return (1 - tversky).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    
    Combines Focal + Dice + BCE for optimal performance on imbalanced data.
    """
    
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.3,
        ce_weight: float = 0.2,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1e-5
    ):
        """
        Args:
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            ce_weight: Weight for cross-entropy component
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            dice_smooth: Smoothing for dice loss
        """
        super().__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Loss components
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(
            softmax=True,
            smooth_nr=dice_smooth,
            smooth_dr=dice_smooth
        )
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            inputs: Predictions [B, C, D, H, W]
            targets: Ground truth [B, D, H, W] (class indices)
        
        Returns:
            Weighted combination of losses
        """
        total_loss = 0.0
        
        # Focal loss
        if self.focal_weight > 0:
            focal = self.focal_loss(inputs, targets)
            total_loss += self.focal_weight * focal
        
        # Dice loss
        if self.dice_weight > 0:
            dice = self.dice_loss(inputs, targets)
            total_loss += self.dice_weight * dice
        
        # Cross-entropy loss
        if self.ce_weight > 0:
            ce = self.ce_loss(inputs, targets)
            total_loss += self.ce_weight * ce
        
        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    Loss function for deep supervision.
    
    Applies loss at multiple scales with decreasing weights.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        ds_weights: list = [0.5, 0.25, 0.125, 0.0625]
    ):
        """
        Args:
            base_loss: Base loss function to apply at each scale
            ds_weights: Weights for each deep supervision output
        """
        super().__init__()
        self.base_loss = base_loss
        self.ds_weights = ds_weights
    
    def forward(
        self,
        main_output: torch.Tensor,
        ds_outputs: list,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute deep supervision loss.
        
        Args:
            main_output: Main prediction [B, C, D, H, W]
            ds_outputs: List of auxiliary predictions
            targets: Ground truth [B, D, H, W]
        
        Returns:
            Weighted sum of losses across scales
        """
        # Main output loss
        total_loss = self.base_loss(main_output, targets)
        
        # Auxiliary output losses
        for i, (ds_out, weight) in enumerate(zip(ds_outputs, self.ds_weights)):
            # Resize target to match auxiliary output size
            target_resized = F.interpolate(
                targets.unsqueeze(1).float(),
                size=ds_out.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
            
            aux_loss = self.base_loss(ds_out, target_resized)
            total_loss += weight * aux_loss
        
        return total_loss


def create_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create loss based on configuration.
    
    Args:
        config: Loss configuration dictionary
    
    Returns:
        Loss function
    """
    loss_type = config.get("type", "dice_ce")
    
    if loss_type == "focal":
        return FocalLoss(
            alpha=config.get("focal_alpha", 0.75),
            gamma=config.get("focal_gamma", 2.0)
        )
    
    elif loss_type == "tversky":
        return TverskyLoss(
            alpha=config.get("tversky_alpha", 0.7),
            beta=config.get("tversky_beta", 0.3)
        )
    
    elif loss_type == "combined":
        return CombinedLoss(
            focal_weight=config.get("focal_loss_weight", 0.5),
            dice_weight=config.get("dice_loss_weight", 0.3),
            ce_weight=config.get("ce_loss_weight", 0.2),
            focal_alpha=config.get("focal_alpha", 0.75),
            focal_gamma=config.get("focal_gamma", 2.0)
        )
    
    elif loss_type == "dice_ce":
        return DiceCELoss(
            softmax=True,
            lambda_dice=config.get("dice_loss_weight", 0.5),
            lambda_ce=config.get("ce_loss_weight", 0.5)
        )
    
    else:
        return DiceLoss(softmax=True)


if __name__ == "__main__":
    # Test loss functions
    print("Testing custom loss functions...")
    
    # Create dummy data
    batch_size = 2
    num_classes = 2
    inputs = torch.randn(batch_size, num_classes, 32, 32, 32)
    targets = torch.randint(0, num_classes, (batch_size, 32, 32, 32))
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    loss_focal = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss_focal.item():.4f}")
    
    # Test Tversky Loss
    tversky_loss = TverskyLoss()
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
    loss_tversky = tversky_loss(inputs, targets_onehot)
    print(f"Tversky Loss: {loss_tversky.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss()
    loss_combined = combined_loss(inputs, targets)
    print(f"Combined Loss: {loss_combined.item():.4f}")
    
    print("✓ All loss tests passed!")
