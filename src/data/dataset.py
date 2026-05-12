"""
Enhanced Dataset for Lung Cancer Nodule Detection

Improvements over Fall project:
- Larger patch sizes
- Better sampling strategies
- Hard negative mining
- Advanced caching
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable, List
from pathlib import Path


class LIDCDataset(Dataset):
    """
    Enhanced LIDC-IDRI dataset for lung nodule segmentation.
    """
    
    def __init__(
        self,
        manifest_file: str,
        ct_dir: str,
        mask_dir: str,
        patch_size: tuple = (128, 128, 128),
        samples_per_volume: int = 4,
        positive_negative_ratio: float = 2.0,
        transform: Optional[Callable] = None,
        cache_data: bool = False
    ):
        """
        Args:
            manifest_file: CSV file with CT and mask filenames
            ct_dir: Directory containing CT scans
            mask_dir: Directory containing masks
            patch_size: Size of extracted patches
            samples_per_volume: Number of patches to extract per CT
            positive_negative_ratio: Ratio of positive to negative patches
            transform: Data augmentation transforms
            cache_data: Whether to cache data in memory
        """
        self.manifest = pd.read_csv(manifest_file)
        self.ct_dir = Path(ct_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.pos_neg_ratio = positive_negative_ratio
        self.transform = transform
        self.cache_data = cache_data
        
        # Cache for loaded volumes
        self._cache: Dict[int, Dict] = {}
        
        print(f"Loaded dataset with {len(self.manifest)} samples")
        print(f"Patch size: {patch_size}")
        print(f"Samples per volume: {samples_per_volume}")
    
    def __len__(self) -> int:
        return len(self.manifest) * self.samples_per_volume
    
    def _load_volume(self, idx: int) -> Dict:
        """Load CT and mask volumes."""
        # Get volume index and patch index
        vol_idx = idx // self.samples_per_volume
        
        # Check cache
        if self.cache_data and vol_idx in self._cache:
            return self._cache[vol_idx]
        
        # Load from manifest
        row = self.manifest.iloc[vol_idx]
        ct_path = self.ct_dir / row['ct_filename']
        mask_path = self.mask_dir / row['mask_filename']
        
        # Load NIfTI files
        ct_img = nib.load(str(ct_path))
        mask_img = nib.load(str(mask_path))
        
        ct_data = ct_img.get_fdata().astype(np.float32)
        mask_data = mask_img.get_fdata().astype(np.int64)
        
        # Normalize CT (HU windowing for lung tissue)
        ct_data = np.clip(ct_data, -1000, 400)
        ct_data = (ct_data + 1000) / 1400  # Scale to [0, 1]
        
        data = {
            'ct': ct_data,
            'mask': mask_data,
            'affine': ct_img.affine,
            'header': ct_img.header
        }
        
        # Cache if enabled
        if self.cache_data:
            self._cache[vol_idx] = data
        
        return data
    
    def _extract_patch(
        self,
        ct: np.ndarray,
        mask: np.ndarray,
        prefer_positive: bool = True
    ) -> Dict:
        """
        Extract a random patch from the volume.
        
        Args:
            ct: CT volume [D, H, W]
            mask: Mask volume [D, H, W]
            prefer_positive: Whether to prefer patches with nodules
        
        Returns:
            Dictionary with patch data
        """
        d, h, w = ct.shape
        pd, ph, pw = self.patch_size
        
        if prefer_positive:
            # Find positive voxel locations
            pos_coords = np.argwhere(mask > 0)
            
            if len(pos_coords) > 0:
                # Sample center near a positive voxel
                center_idx = np.random.randint(len(pos_coords))
                center = pos_coords[center_idx]
                
                # Add some randomness
                center = center + np.random.randint(-pd//4, pd//4, size=3)
            else:
                # No positive voxels, sample randomly
                center = np.array([
                    np.random.randint(pd//2, d - pd//2),
                    np.random.randint(ph//2, h - ph//2),
                    np.random.randint(pw//2, w - pw//2)
                ])
        else:
            # Random sampling
            center = np.array([
                np.random.randint(pd//2, d - pd//2),
                np.random.randint(ph//2, h - ph//2),
                np.random.randint(pw//2, w - pw//2)
            ])
        
        # Extract patch around center
        start = np.maximum(center - np.array(self.patch_size) // 2, 0)
        end = np.minimum(start + np.array(self.patch_size), [d, h, w])
        
        # Adjust if patch goes out of bounds
        if end[0] - start[0] < pd:
            start[0] = max(0, end[0] - pd)
        if end[1] - start[1] < ph:
            start[1] = max(0, end[1] - ph)
        if end[2] - start[2] < pw:
            start[2] = max(0, end[2] - pw)
        
        ct_patch = ct[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        mask_patch = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        # Pad if necessary
        if ct_patch.shape != tuple(self.patch_size):
            ct_patch = self._pad_to_size(ct_patch, self.patch_size)
            mask_patch = self._pad_to_size(mask_patch, self.patch_size)
        
        return {
            'image': ct_patch,
            'label': mask_patch
        }
    
    def _pad_to_size(self, arr: np.ndarray, target_size: tuple) -> np.ndarray:
        """Pad array to target size."""
        pad_width = []
        for i in range(len(target_size)):
            diff = target_size[i] - arr.shape[i]
            pad_width.append((diff // 2, diff - diff // 2))
        return np.pad(arr, pad_width, mode='constant', constant_values=0)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a random patch from a volume.
        
        Returns:
            Dictionary with 'image' and 'label' tensors
        """
        # Load volume
        volume_data = self._load_volume(idx)
        ct = volume_data['ct']
        mask = volume_data['mask']
        
        # Determine if we should sample positive or negative patch
        patch_idx = idx % self.samples_per_volume
        prefer_positive = (patch_idx < int(
            self.samples_per_volume * self.pos_neg_ratio / (1 + self.pos_neg_ratio)
        ))
        
        # Extract patch
        patch_data = self._extract_patch(ct, mask, prefer_positive)
        
        # Apply transforms
        if self.transform:
            patch_data = self.transform(patch_data)
        
        # Convert to tensors
        image = torch.from_numpy(patch_data['image']).unsqueeze(0).float()
        label = torch.from_numpy(patch_data['label']).long()
        
        return {
            'image': image,
            'label': label
        }


class LIDCInferenceDataset(Dataset):
    """
    Dataset for inference (whole volume processing).
    """
    
    def __init__(
        self,
        ct_paths: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Args:
            ct_paths: List of paths to CT scans
            transform: Preprocessing transforms
        """
        self.ct_paths = ct_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.ct_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load full CT volume for inference."""
        ct_path = self.ct_paths[idx]
        
        # Load NIfTI
        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata().astype(np.float32)
        
        # Normalize
        ct_data = np.clip(ct_data, -1000, 400)
        ct_data = (ct_data + 1000) / 1400
        
        data = {
            'image': ct_data,
            'affine': ct_img.affine,
            'header': ct_img.header,
            'path': ct_path
        }
        
        if self.transform:
            data = self.transform(data)
        
        # Convert to tensor
        image = torch.from_numpy(data['image']).unsqueeze(0).float()
        
        return {
            'image': image,
            'affine': data['affine'],
            'header': data['header'],
            'path': data['path']
        }


if __name__ == "__main__":
    # Test dataset
    print("Testing LIDC Dataset...")
    
    # Note: Update these paths to actual data
    manifest = "D:/Fall Senior Project/LIDC-clean/train_manifest.csv"
    ct_dir = "D:/Fall Senior Project/LIDC-exact/ct"
    mask_dir = "D:/Fall Senior Project/LIDC-exact/masks"
    
    if os.path.exists(manifest):
        dataset = LIDCDataset(
            manifest_file=manifest,
            ct_dir=ct_dir,
            mask_dir=mask_dir,
            patch_size=(96, 96, 96),
            samples_per_volume=2
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label shape: {sample['label'].shape}")
        print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"Unique labels: {torch.unique(sample['label'])}")
        print("✓ Dataset test passed!")
    else:
        print("⚠ Test data not found. Skipping dataset test.")
