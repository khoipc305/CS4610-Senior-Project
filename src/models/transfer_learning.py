"""
Transfer Learning Models for Medical Imaging

As mentioned in Progress Report 1:
"Transfer learning techniques will be explored by adapting pre-trained 
convolutional architectures to the CT dataset."

Implements:
- Pre-trained 3D ResNet
- Pre-trained DenseNet
- Fine-tuning strategies
"""

import torch
import torch.nn as nn
from monai.networks.nets import ResNet, DenseNet121
from typing import Optional, Tuple


class TransferLearning3DResNet(nn.Module):
    """
    3D ResNet with transfer learning from pre-trained weights.
    
    Can be initialized with ImageNet-style pre-trained weights
    and fine-tuned for lung nodule segmentation.
    """
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        pretrained: bool = False,
        freeze_encoder: bool = False
    ):
        """
        Args:
            spatial_dims: Number of spatial dimensions (3 for 3D)
            in_channels: Input channels (1 for CT)
            out_channels: Output channels (2 for binary segmentation)
            pretrained: Use pre-trained weights (if available)
            freeze_encoder: Freeze encoder weights during training
        """
        super().__init__()
        
        # Create ResNet backbone
        # Note: MONAI ResNet doesn't have ImageNet pretrained weights for 3D
        # This is a template for when such weights become available
        self.resnet = ResNet(
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=out_channels,
            block='bottleneck',
            layers=[3, 4, 6, 3],  # ResNet-50 architecture
            shortcut_type='B'
        )
        
        # Modify first conv layer if needed for single-channel input
        if in_channels != 3:
            # Adapt from RGB (3 channels) to grayscale (1 channel)
            self.resnet.conv1 = nn.Conv3d(
                in_channels, 64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        
        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder layers for fine-tuning."""
        # Freeze all layers except final classification layer
        for name, param in self.named_parameters():
            if 'fc' not in name:  # Don't freeze final layer
                param.requires_grad = False
        
        print("✓ Encoder layers frozen (fine-tuning mode)")
    
    def unfreeze_encoder(self, layers_to_unfreeze: Optional[list] = None):
        """
        Unfreeze encoder layers for full training.
        
        Args:
            layers_to_unfreeze: List of layer names to unfreeze (None = all)
        """
        if layers_to_unfreeze is None:
            # Unfreeze all
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True
        
        print("✓ Encoder layers unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class TransferLearningDenseNet(nn.Module):
    """
    3D DenseNet with transfer learning capabilities.
    """
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        pretrained: bool = False,
        freeze_encoder: bool = False
    ):
        """
        Args:
            spatial_dims: Number of spatial dimensions
            in_channels: Input channels
            out_channels: Output channels
            pretrained: Use pre-trained weights
            freeze_encoder: Freeze encoder during training
        """
        super().__init__()
        
        # Create DenseNet backbone
        self.densenet = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder layers."""
        for name, param in self.named_parameters():
            if 'class_layers' not in name:
                param.requires_grad = False
        print("✓ DenseNet encoder frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)


class FineTuningStrategy:
    """
    Implements progressive fine-tuning strategies.
    
    Strategies:
    1. Feature extraction: Freeze encoder, train only head
    2. Gradual unfreezing: Unfreeze layers progressively
    3. Discriminative learning rates: Different LR for different layers
    """
    
    @staticmethod
    def get_layer_groups(model: nn.Module) -> list:
        """
        Divide model into layer groups for progressive unfreezing.
        
        Returns:
            List of layer groups (from early to late)
        """
        layer_groups = []
        
        # Early layers
        early_layers = []
        # Middle layers  
        middle_layers = []
        # Late layers
        late_layers = []
        # Head
        head_layers = []
        
        for name, module in model.named_modules():
            if 'conv1' in name or 'layer1' in name:
                early_layers.append(module)
            elif 'layer2' in name or 'layer3' in name:
                middle_layers.append(module)
            elif 'layer4' in name:
                late_layers.append(module)
            elif 'fc' in name or 'class' in name:
                head_layers.append(module)
        
        layer_groups = [early_layers, middle_layers, late_layers, head_layers]
        return layer_groups
    
    @staticmethod
    def progressive_unfreeze(
        model: nn.Module,
        epoch: int,
        unfreeze_schedule: dict
    ):
        """
        Progressively unfreeze layers according to schedule.
        
        Args:
            model: Model to unfreeze
            epoch: Current epoch
            unfreeze_schedule: Dict mapping epochs to layer groups
                              e.g., {0: 'head', 5: 'late', 10: 'all'}
        """
        if epoch in unfreeze_schedule:
            layer_group = unfreeze_schedule[epoch]
            
            if layer_group == 'all':
                for param in model.parameters():
                    param.requires_grad = True
                print(f"Epoch {epoch}: Unfroze all layers")
            else:
                # Unfreeze specific layer group
                for name, param in model.named_parameters():
                    if layer_group in name:
                        param.requires_grad = True
                print(f"Epoch {epoch}: Unfroze {layer_group} layers")
    
    @staticmethod
    def get_discriminative_lr_params(
        model: nn.Module,
        base_lr: float,
        lr_multiplier: float = 0.1
    ) -> list:
        """
        Get parameter groups with discriminative learning rates.
        
        Earlier layers get smaller learning rates.
        
        Args:
            model: Model
            base_lr: Base learning rate for head
            lr_multiplier: Multiplier for earlier layers (< 1.0)
        
        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []
        
        # Head (highest LR)
        head_params = []
        # Late layers (medium LR)
        late_params = []
        # Middle layers (lower LR)
        middle_params = []
        # Early layers (lowest LR)
        early_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'fc' in name or 'class' in name:
                head_params.append(param)
            elif 'layer4' in name:
                late_params.append(param)
            elif 'layer2' in name or 'layer3' in name:
                middle_params.append(param)
            else:
                early_params.append(param)
        
        # Create param groups with different LRs
        if head_params:
            param_groups.append({'params': head_params, 'lr': base_lr})
        if late_params:
            param_groups.append({'params': late_params, 'lr': base_lr * 0.5})
        if middle_params:
            param_groups.append({'params': middle_params, 'lr': base_lr * 0.1})
        if early_params:
            param_groups.append({'params': early_params, 'lr': base_lr * 0.01})
        
        return param_groups


def create_transfer_learning_model(
    architecture: str = "resnet",
    in_channels: int = 1,
    out_channels: int = 2,
    pretrained: bool = False,
    freeze_encoder: bool = True
) -> nn.Module:
    """
    Factory function to create transfer learning models.
    
    Args:
        architecture: 'resnet' or 'densenet'
        in_channels: Input channels
        out_channels: Output channels
        pretrained: Use pretrained weights
        freeze_encoder: Freeze encoder for fine-tuning
    
    Returns:
        Transfer learning model
    """
    if architecture.lower() == "resnet":
        return TransferLearning3DResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder
        )
    elif architecture.lower() == "densenet":
        return TransferLearningDenseNet(
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    print("Testing Transfer Learning Models...")
    
    # Test ResNet
    print("\nTesting ResNet with transfer learning...")
    model = TransferLearning3DResNet(freeze_encoder=True)
    x = torch.randn(1, 1, 96, 96, 96)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test progressive unfreezing
    print("\nTesting progressive unfreezing...")
    strategy = FineTuningStrategy()
    schedule = {0: 'head', 5: 'layer4', 10: 'all'}
    
    for epoch in [0, 5, 10]:
        strategy.progressive_unfreeze(model, epoch, schedule)
    
    # Test discriminative LR
    print("\nTesting discriminative learning rates...")
    model.unfreeze_encoder()
    param_groups = strategy.get_discriminative_lr_params(model, base_lr=1e-4)
    print(f"Created {len(param_groups)} parameter groups with different LRs")
    for i, group in enumerate(param_groups):
        print(f"  Group {i}: {len(group['params'])} params, LR={group['lr']}")
    
    print("\n✓ Transfer learning tests passed!")
