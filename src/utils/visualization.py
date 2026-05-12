"""
Model Interpretability and Visualization Tools

As mentioned in Progress Report 1:
- Visualization methods to highlight image regions influencing predictions
- Attention map visualization
- Grad-CAM for CNNs
- Prediction overlays
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import nibabel as nib


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for 3D models.
    Helps visualize which regions the model focuses on.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Args:
            model: PyTorch model
            target_layer: Name of layer to visualize (e.g., 'unet.model.3')
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor [1, C, D, H, W]
            target_class: Target class for visualization
        
        Returns:
            CAM heatmap [D, H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get score for target class
        if output.dim() == 5:  # [B, C, D, H, W]
            score = output[:, target_class, :, :, :].sum()
        else:
            score = output[:, target_class].sum()
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


class AttentionVisualizer:
    """
    Visualize attention maps from Attention U-Net.
    """
    
    @staticmethod
    def extract_attention_maps(
        model: torch.nn.Module,
        input_tensor: torch.Tensor
    ) -> list:
        """
        Extract attention maps from model.
        
        Args:
            model: Attention U-Net model
            input_tensor: Input tensor [1, C, D, H, W]
        
        Returns:
            List of attention maps
        """
        attention_maps = []
        
        def hook_fn(module, input, output):
            # Store attention weights
            if hasattr(module, 'attention'):
                attention_maps.append(output.detach())
        
        # Register hooks on attention blocks
        handles = []
        for name, module in model.named_modules():
            if 'attention' in name.lower():
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return attention_maps
    
    @staticmethod
    def visualize_attention(
        attention_map: torch.Tensor,
        slice_idx: int = None
    ) -> np.ndarray:
        """
        Visualize attention map.
        
        Args:
            attention_map: Attention tensor [B, C, D, H, W]
            slice_idx: Which slice to visualize (middle if None)
        
        Returns:
            Attention heatmap [H, W]
        """
        # Average over channels
        if attention_map.dim() == 5:
            attention_map = attention_map.mean(dim=1)  # [B, D, H, W]
        
        # Get middle slice if not specified
        if slice_idx is None:
            slice_idx = attention_map.shape[1] // 2
        
        # Extract slice
        slice_2d = attention_map[0, slice_idx].cpu().numpy()
        
        # Normalize
        slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
        
        return slice_2d


def visualize_prediction(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    pred_slice: np.ndarray,
    attention_slice: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize CT, ground truth, prediction, and attention map.
    
    Args:
        ct_slice: CT image slice [H, W]
        mask_slice: Ground truth mask [H, W]
        pred_slice: Predicted mask [H, W]
        attention_slice: Attention map [H, W] (optional)
        save_path: Path to save figure
    """
    n_plots = 4 if attention_slice is not None else 3
    
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))
    
    # CT image
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title('CT Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ct_slice, cmap='gray')
    axes[1].imshow(mask_slice, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(ct_slice, cmap='gray')
    axes[2].imshow(pred_slice, cmap='Blues', alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Attention map
    if attention_slice is not None:
        im = axes[3].imshow(attention_slice, cmap='jet')
        axes[3].set_title('Attention Map')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_3d_visualization(
    volume: np.ndarray,
    slice_indices: Tuple[int, int, int] = None,
    title: str = "3D Volume Visualization",
    save_path: Optional[str] = None
):
    """
    Create 3-plane visualization of 3D volume.
    
    Args:
        volume: 3D volume [D, H, W]
        slice_indices: (axial, sagittal, coronal) slice indices
        title: Figure title
        save_path: Path to save figure
    """
    if slice_indices is None:
        # Use middle slices
        slice_indices = (
            volume.shape[0] // 2,
            volume.shape[1] // 2,
            volume.shape[2] // 2
        )
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Axial slice
    axes[0].imshow(volume[slice_indices[0], :, :], cmap='gray')
    axes[0].set_title(f'Axial (z={slice_indices[0]})')
    axes[0].axis('off')
    
    # Sagittal slice
    axes[1].imshow(volume[:, slice_indices[1], :], cmap='gray')
    axes[1].set_title(f'Sagittal (y={slice_indices[1]})')
    axes[1].axis('off')
    
    # Coronal slice
    axes[2].imshow(volume[:, :, slice_indices[2]], cmap='gray')
    axes[2].set_title(f'Coronal (x={slice_indices[2]})')
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved 3D visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    history_csv: str,
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves from history CSV.
    
    Args:
        history_csv: Path to training_history.csv
        save_path: Path to save figure
    """
    import pandas as pd
    
    df = pd.read_csv(history_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in df.columns:
        axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice curves (if available)
    if 'val_dice' in df.columns:
        axes[1].plot(df['epoch'], df['val_dice'], label='Val Dice', linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Validation Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Dice data available',
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Testing visualization tools...")
    
    # Create dummy data
    ct_slice = np.random.randn(128, 128)
    mask_slice = np.random.randint(0, 2, (128, 128))
    pred_slice = np.random.randint(0, 2, (128, 128))
    attention_slice = np.random.rand(128, 128)
    
    # Test visualization
    visualize_prediction(
        ct_slice,
        mask_slice,
        pred_slice,
        attention_slice,
        save_path="test_visualization.png"
    )
    
    print("✓ Visualization test passed!")
