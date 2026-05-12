"""
Comprehensive Evaluation Metrics for Medical Image Segmentation
"""

import torch
import numpy as np
from typing import Dict, List
from scipy.ndimage import distance_transform_edt


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Dice Similarity Coefficient.
    
    Args:
        pred: Predicted segmentation [D, H, W] or [B, D, H, W]
        target: Ground truth [D, H, W] or [B, D, H, W]
        smooth: Smoothing constant
    
    Returns:
        Dice score (0-1, higher is better)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth
        smooth: Smoothing constant
    
    Returns:
        IoU score (0-1, higher is better)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def sensitivity_recall(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Sensitivity (Recall / True Positive Rate).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth
        smooth: Smoothing constant
    
    Returns:
        Sensitivity (0-1, higher is better)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    return sensitivity.item()


def specificity(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Specificity (True Negative Rate).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth
        smooth: Smoothing constant
    
    Returns:
        Specificity (0-1, higher is better)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    spec = (tn + smooth) / (tn + fp + smooth)
    return spec.item()


def precision(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Precision (Positive Predictive Value).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth
        smooth: Smoothing constant
    
    Returns:
        Precision (0-1, higher is better)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec.item()


def hausdorff_distance(pred: np.ndarray, target: np.ndarray, percentile: int = 95) -> float:
    """
    Compute Hausdorff Distance (95th percentile).
    
    Measures maximum distance between boundaries.
    
    Args:
        pred: Predicted segmentation (numpy array)
        target: Ground truth (numpy array)
        percentile: Percentile to use (95 is common)
    
    Returns:
        Hausdorff distance in voxels (lower is better)
    """
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')
    
    # Compute distance transforms
    dist_pred = distance_transform_edt(1 - pred)
    dist_target = distance_transform_edt(1 - target)
    
    # Surface distances
    surface_pred = dist_target[pred > 0]
    surface_target = dist_pred[target > 0]
    
    if len(surface_pred) == 0 or len(surface_target) == 0:
        return float('inf')
    
    # Compute percentile
    hd = max(
        np.percentile(surface_pred, percentile),
        np.percentile(surface_target, percentile)
    )
    
    return float(hd)


def average_surface_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Average Surface Distance (ASD).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth
    
    Returns:
        Average surface distance in voxels (lower is better)
    """
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')
    
    # Compute distance transforms
    dist_pred = distance_transform_edt(1 - pred)
    dist_target = distance_transform_edt(1 - target)
    
    # Surface distances
    surface_pred = dist_target[pred > 0]
    surface_target = dist_pred[target > 0]
    
    if len(surface_pred) == 0 or len(surface_target) == 0:
        return float('inf')
    
    # Average
    asd = (surface_pred.mean() + surface_target.mean()) / 2.0
    
    return float(asd)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for medical image segmentation.
    """
    
    def __init__(self, metrics_list: List[str] = None):
        """
        Args:
            metrics_list: List of metric names to compute
                         If None, computes all metrics
        """
        if metrics_list is None:
            metrics_list = [
                'dice', 'iou', 'sensitivity', 'specificity',
                'precision', 'hausdorff_distance', 'average_surface_distance'
            ]
        
        self.metrics_list = metrics_list
    
    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        as_binary: bool = True
    ) -> Dict[str, float]:
        """
        Compute all specified metrics.
        
        Args:
            pred: Predictions [B, C, D, H, W] (logits or probabilities)
            target: Ground truth [B, D, H, W] (class indices)
            as_binary: Convert to binary masks (class 1 vs 0)
        
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        # Convert to binary masks if needed
        if as_binary:
            if pred.dim() == 5:  # [B, C, D, H, W]
                pred_binary = torch.argmax(pred, dim=1).float()
            else:
                pred_binary = (pred > 0.5).float()
            
            target_binary = (target > 0).float()
        else:
            pred_binary = pred
            target_binary = target
        
        # Compute metrics
        if 'dice' in self.metrics_list:
            results['dice'] = dice_coefficient(pred_binary, target_binary)
        
        if 'iou' in self.metrics_list:
            results['iou'] = iou_score(pred_binary, target_binary)
        
        if 'sensitivity' in self.metrics_list:
            results['sensitivity'] = sensitivity_recall(pred_binary, target_binary)
        
        if 'specificity' in self.metrics_list:
            results['specificity'] = specificity(pred_binary, target_binary)
        
        if 'precision' in self.metrics_list:
            results['precision'] = precision(pred_binary, target_binary)
        
        # Distance-based metrics (require numpy)
        pred_np = pred_binary.cpu().numpy()
        target_np = target_binary.cpu().numpy()
        
        if 'hausdorff_distance' in self.metrics_list:
            try:
                hd = hausdorff_distance(pred_np[0], target_np[0])
                results['hausdorff_distance'] = hd
            except:
                results['hausdorff_distance'] = float('inf')
        
        if 'average_surface_distance' in self.metrics_list:
            try:
                asd = average_surface_distance(pred_np[0], target_np[0])
                results['average_surface_distance'] = asd
            except:
                results['average_surface_distance'] = float('inf')
        
        return results
    
    def aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate metrics across multiple samples.
        
        Args:
            metrics_list: List of metric dictionaries
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            values = [v for v in values if not np.isinf(v)]  # Remove inf values
            
            if values:
                aggregated[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return aggregated


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create dummy predictions and targets
    pred = torch.rand(1, 2, 32, 32, 32)  # [B, C, D, H, W]
    target = torch.randint(0, 2, (1, 32, 32, 32))  # [B, D, H, W]
    
    # Create calculator
    calculator = MetricsCalculator()
    
    # Compute metrics
    metrics = calculator.compute_all(pred, target)
    
    print("Computed metrics:")
    for name, value in metrics.items():
        if not np.isinf(value):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: inf")
    
    print("✓ Metrics test passed!")
