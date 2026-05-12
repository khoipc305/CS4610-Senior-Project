"""
Main Training Script for Enhanced Lung Cancer Detection

Usage:
    python scripts/train.py --config config/config_advanced.yaml
    python scripts/train.py --config config/config_baseline.yaml --gpu 0
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.attention_unet import create_model
from training.losses import create_loss_function
from training.metrics import MetricsCalculator
from data.dataset import LIDCDataset
from data.transforms import get_training_transforms, get_validation_transforms


class Trainer:
    """
    Main training class with all enhancements.
    """
    
    def __init__(self, config: dict, output_dir: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        self.set_seed(config['experiment']['seed'])
        
        # Setup device
        self.setup_device()
        
        # Create model
        self.model = create_model(config['model']).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create loss function
        self.criterion = create_loss_function(config['loss'])
        
        # Create optimizer
        self.optimizer = self.create_optimizer()
        
        # Create scheduler
        self.scheduler = self.create_scheduler()
        
        # Create metrics calculator
        self.metrics_calculator = MetricsCalculator(
            config['metrics']['metrics_list']
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        
        # TensorBoard
        if config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(
                log_dir=str(self.output_dir / 'tensorboard')
            )
        else:
            self.writer = None
        
        # Mixed precision
        self.use_amp = config['experiment'].get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def setup_device(self):
        """Setup compute device (GPU or CPU)."""
        use_gpu = self.config['experiment'].get('use_gpu', False)
        
        if use_gpu and torch.cuda.is_available():
            gpu_id = self.config['experiment'].get('gpu_id', 0)
            self.device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
    
    def create_optimizer(self):
        """Create optimizer based on config."""
        opt_config = self.config['training']
        opt_name = opt_config.get('optimizer', 'Adam')
        lr = opt_config.get('learning_rate', 1e-4)
        weight_decay = opt_config.get('weight_decay', 0.0)
        
        if opt_name == 'Adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'AdamW':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'SGD':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def create_scheduler(self):
        """Create learning rate scheduler."""
        sch_config = self.config['training']
        sch_name = sch_config.get('scheduler', 'reduce_on_plateau')
        
        if sch_name == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sch_config.get('lr_factor', 0.5),
                patience=sch_config.get('lr_patience', 5),
                verbose=True
            )
        elif sch_name == 'cosine_warmup':
            warmup_epochs = sch_config.get('warmup_epochs', 5)
            total_epochs = sch_config.get('epochs', 100)
            min_lr = sch_config.get('min_lr', 1e-6)
            
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr
            )
        elif sch_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        epoch_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                epoch_loss += loss.item()
                
                # Compute metrics
                metrics = self.metrics_calculator.compute_all(outputs, labels)
                all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = self.metrics_calculator.aggregate_metrics(all_metrics)
        
        return epoch_loss / len(val_loader), avg_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'last_checkpoint.pth')
        
        # Save periodic
        if self.current_epoch % self.config['checkpoint']['save_interval'] == 0:
            torch.save(
                checkpoint,
                self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
            )
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            print(f"✓ Saved best model (metric: {self.best_metric:.4f})")
    
    def train(self, train_loader, val_loader, epochs: int):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        early_stop_patience = self.config['training'].get('patience', 15)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            if epoch % self.config['validation']['val_interval'] == 0:
                val_loss, val_metrics = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_metrics_history.append(val_metrics)
                
                # Get primary metric
                primary_metric = self.config['metrics']['primary']
                current_metric = val_metrics[primary_metric]['mean']
                
                # Log results
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val {primary_metric}: {current_metric:.4f}")
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Loss/Train', train_loss, epoch)
                    self.writer.add_scalar('Loss/Val', val_loss, epoch)
                    for metric_name, values in val_metrics.items():
                        self.writer.add_scalar(
                            f'Metrics/{metric_name}',
                            values['mean'],
                            epoch
                        )
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(is_best)
                
                # Early stopping
                if self.config['training'].get('early_stopping', False):
                    if patience_counter >= early_stop_patience:
                        print(f"\n✗ Early stopping triggered after {patience_counter} epochs without improvement")
                        break
        
        # Save final results
        self.save_results()
        
        if self.writer:
            self.writer.close()
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best {primary_metric}: {self.best_metric:.4f}")
        print(f"{'='*60}\n")
    
    def save_results(self):
        """Save training results and history."""
        # Save training history
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses[:len(self.train_losses)]
        })
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        
        print(f"✓ Saved training history to {self.output_dir / 'training_history.csv'}")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict):
    """Create training and validation data loaders."""
    # Get transforms
    train_transforms = get_training_transforms(config)
    val_transforms = get_validation_transforms(config)
    
    # Create datasets
    train_dataset = LIDCDataset(
        manifest_file=config['data']['train_manifest'],
        ct_dir=config['data']['ct_dir'],
        mask_dir=config['data']['mask_dir'],
        patch_size=tuple(config['preprocessing']['patch_size']),
        samples_per_volume=config['sampling']['samples_per_volume'],
        positive_negative_ratio=config['sampling']['positive_negative_ratio'],
        transform=train_transforms,
        cache_data=False
    )
    
    val_dataset = LIDCDataset(
        manifest_file=config['data']['val_manifest'],
        ct_dir=config['data']['ct_dir'],
        mask_dir=config['data']['mask_dir'],
        patch_size=tuple(config['preprocessing']['patch_size']),
        samples_per_volume=2,  # Fewer samples for validation
        positive_negative_ratio=1.0,
        transform=val_transforms,
        cache_data=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['validation']['val_batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train lung cancer detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Experiment: {config['experiment']['name']}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / config['experiment']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create trainer
    trainer = Trainer(config, output_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_metric = checkpoint['best_metric']
        print(f"Resumed from checkpoint: {args.resume} (epoch {trainer.current_epoch})")
    
    # Train
    trainer.train(
        train_loader,
        val_loader,
        epochs=config['training']['epochs']
    )


if __name__ == "__main__":
    main()
