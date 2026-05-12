"""
Advanced Data Augmentation for 3D Medical Imaging

Implements elastic deformations, intensity transforms, and more.
"""

import numpy as np
import torch
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, RandGaussianNoised, RandGaussianSmoothd,
    RandAffined, SpatialPadd, ToTensord
)
from typing import Dict


class ElasticDeformation:
    """
    Apply elastic deformation to 3D volumes.
    Useful for augmenting medical images while preserving topology.
    """
    
    def __init__(
        self,
        prob: float = 0.3,
        sigma: float = 5.0,
        alpha: float = 50.0
    ):
        """
        Args:
            prob: Probability of applying transform
            sigma: Std dev of Gaussian filter
            alpha: Scaling factor for displacement
        """
        self.prob = prob
        self.sigma = sigma
        self.alpha = alpha
    
    def __call__(self, data: Dict) -> Dict:
        """Apply elastic deformation if probability check passes."""
        if np.random.random() > self.prob:
            return data
        
        image = data['image']
        label = data['label']
        
        # Generate random displacement fields
        shape = image.shape
        dx = np.random.randn(*shape) * self.sigma
        dy = np.random.randn(*shape) * self.sigma
        dz = np.random.randn(*shape) * self.sigma
        
        # Smooth the displacement fields
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, self.sigma, mode='constant') * self.alpha
        dy = gaussian_filter(dy, self.sigma, mode='constant') * self.alpha
        dz = gaussian_filter(dz, self.sigma, mode='constant') * self.alpha
        
        # Create meshgrid
        d, h, w = shape
        z, y, x = np.meshgrid(
            np.arange(d),
            np.arange(h),
            np.arange(w),
            indexing='ij'
        )
        
        # Apply displacement
        indices = (
            np.clip(z + dz, 0, d - 1).astype(int),
            np.clip(y + dy, 0, h - 1).astype(int),
            np.clip(x + dx, 0, w - 1).astype(int)
        )
        
        data['image'] = image[indices]
        data['label'] = label[indices]
        
        return data


def get_training_transforms(config: Dict) -> Compose:
    """
    Create training augmentation pipeline.
    
    Args:
        config: Configuration dictionary with augmentation parameters
    
    Returns:
        MONAI Compose transform
    """
    aug_config = config.get('augmentation', {})
    
    if not aug_config.get('enabled', True):
        return Compose([])
    
    transforms_list = []
    
    # Random flips
    if aug_config.get('random_flip_prob', 0) > 0:
        transforms_list.append(
            RandFlipd(
                keys=['image', 'label'],
                prob=aug_config['random_flip_prob'],
                spatial_axis=[0, 1, 2]
            )
        )
    
    # Random 90-degree rotations
    if aug_config.get('random_rotate_prob', 0) > 0:
        transforms_list.append(
            RandRotate90d(
                keys=['image', 'label'],
                prob=aug_config['random_rotate_prob'],
                spatial_axes=(0, 1)
            )
        )
    
    # Random affine (rotation + scaling)
    if aug_config.get('random_scale_prob', 0) > 0:
        rotate_range = aug_config.get('random_rotate_range', [-15, 15])
        scale_range = aug_config.get('random_scale_range', [0.9, 1.1])
        
        transforms_list.append(
            RandAffined(
                keys=['image', 'label'],
                prob=aug_config.get('random_scale_prob', 0.5),
                rotate_range=(
                    np.radians(rotate_range),
                    np.radians(rotate_range),
                    np.radians(rotate_range)
                ),
                scale_range=(
                    (scale_range[0] - 1, scale_range[1] - 1),
                    (scale_range[0] - 1, scale_range[1] - 1),
                    (scale_range[0] - 1, scale_range[1] - 1)
                ),
                mode=('bilinear', 'nearest'),
                padding_mode='border'
            )
        )
    
    # Intensity augmentations (image only)
    
    # Gaussian noise
    if aug_config.get('gaussian_noise_prob', 0) > 0:
        transforms_list.append(
            RandGaussianNoised(
                keys=['image'],
                prob=aug_config['gaussian_noise_prob'],
                mean=0.0,
                std=0.1
            )
        )
    
    # Gaussian smoothing
    if aug_config.get('gaussian_smooth_prob', 0) > 0:
        transforms_list.append(
            RandGaussianSmoothd(
                keys=['image'],
                prob=aug_config['gaussian_smooth_prob'],
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0)
            )
        )
    
    # Intensity shift
    if aug_config.get('intensity_shift_prob', 0) > 0:
        transforms_list.append(
            RandShiftIntensityd(
                keys=['image'],
                offsets=0.1,
                prob=aug_config['intensity_shift_prob']
            )
        )
    
    # Intensity scale
    if aug_config.get('intensity_scale_prob', 0) > 0:
        transforms_list.append(
            RandScaleIntensityd(
                keys=['image'],
                factors=0.1,
                prob=aug_config['intensity_scale_prob']
            )
        )
    
    return Compose(transforms_list)


def get_validation_transforms(config: Dict) -> Compose:
    """
    Create validation preprocessing pipeline (no augmentation).
    
    Args:
        config: Configuration dictionary
    
    Returns:
        MONAI Compose transform
    """
    # Validation typically doesn't need augmentation
    # Just ensure consistent preprocessing
    return Compose([])


def get_inference_transforms(config: Dict) -> Compose:
    """
    Create inference preprocessing pipeline.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        MONAI Compose transform
    """
    # Similar to validation but may handle full volumes differently
    return Compose([])


if __name__ == "__main__":
    # Test transforms
    print("Testing augmentation transforms...")
    
    # Create dummy config
    config = {
        'augmentation': {
            'enabled': True,
            'random_flip_prob': 0.5,
            'random_rotate_prob': 0.5,
            'random_rotate_range': [-15, 15],
            'random_scale_prob': 0.5,
            'random_scale_range': [0.9, 1.1],
            'gaussian_noise_prob': 0.2,
            'gaussian_smooth_prob': 0.2,
            'intensity_shift_prob': 0.3,
            'intensity_scale_prob': 0.3
        }
    }
    
    # Get transforms
    train_transforms = get_training_transforms(config)
    
    # Create dummy data
    dummy_data = {
        'image': np.random.randn(96, 96, 96).astype(np.float32),
        'label': np.random.randint(0, 2, (96, 96, 96)).astype(np.int64)
    }
    
    # Apply transforms
    augmented = train_transforms(dummy_data)
    
    print(f"Original image shape: {dummy_data['image'].shape}")
    print(f"Augmented image shape: {augmented['image'].shape}")
    print("✓ Transform test passed!")
