"""
Evaluation Script for Trained Models

Generates comprehensive metrics and visualizations.

Usage:
    python scripts/evaluate.py --model checkpoints/best_model.pth --output results/
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.attention_unet import create_model
from training.metrics import MetricsCalculator
from data.dataset import LIDCDataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint['config']
    
    # Create model
    model = create_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best metric: {checkpoint.get('best_metric', 'N/A')}")
    
    return model, config


def evaluate_model(model, data_loader, device, metrics_calculator):
    """Evaluate model on dataset."""
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute metrics
            metrics = metrics_calculator.compute_all(outputs, labels)
            all_metrics.append(metrics)
    
    return all_metrics


def plot_metrics_distribution(metrics_list, output_dir):
    """Plot distribution of metrics across samples."""
    # Organize metrics by name
    metrics_by_name = {}
    for metrics in metrics_list:
        for name, value in metrics.items():
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            if not np.isinf(value):
                metrics_by_name[name].append(value)
    
    # Plot each metric
    n_metrics = len(metrics_by_name)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (name, values) in enumerate(metrics_by_name.items()):
        ax = axes[idx]
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
        ax.set_title(f'{name.replace("_", " ").title()}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(values), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(values):.3f}')
        ax.legend()
    
    # Hide unused subplots
    for idx in range(len(metrics_by_name), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=150)
    print(f"✓ Saved metrics distribution to {output_dir / 'metrics_distribution.png'}")


def create_summary_table(aggregated_metrics, output_dir):
    """Create summary table of metrics."""
    data = []
    for metric_name, stats in aggregated_metrics.items():
        data.append({
            'Metric': metric_name.replace('_', ' ').title(),
            'Mean': f"{stats['mean']:.4f}",
            'Std': f"{stats['std']:.4f}",
            'Min': f"{stats['min']:.4f}",
            'Max': f"{stats['max']:.4f}",
            'Median': f"{stats['median']:.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_dir / 'evaluation_summary.csv', index=False)
    
    # Print to console
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                       default='D:/Fall Senior Project/LIDC-exact',
                       help='Path to data directory')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config = load_model(args.model, device)
    
    # Create dataset (using validation set)
    from data.transforms import get_validation_transforms
    
    val_dataset = LIDCDataset(
        manifest_file=config['data']['val_manifest'],
        ct_dir=config['data']['ct_dir'],
        mask_dir=config['data']['mask_dir'],
        patch_size=tuple(config['preprocessing']['patch_size']),
        samples_per_volume=2,
        positive_negative_ratio=1.0,
        transform=get_validation_transforms(config),
        cache_data=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Evaluating on {len(val_dataset)} samples...")
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator(config['metrics']['metrics_list'])
    
    # Evaluate
    metrics_list = evaluate_model(model, val_loader, device, metrics_calculator)
    
    # Aggregate metrics
    aggregated_metrics = metrics_calculator.aggregate_metrics(metrics_list)
    
    # Save individual metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(output_dir / 'per_sample_metrics.csv', index=False)
    print(f"✓ Saved per-sample metrics to {output_dir / 'per_sample_metrics.csv'}")
    
    # Create summary table
    summary_df = create_summary_table(aggregated_metrics, output_dir)
    
    # Plot distributions
    plot_metrics_distribution(metrics_list, output_dir)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
