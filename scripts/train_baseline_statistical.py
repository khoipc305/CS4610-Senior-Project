"""
Train Statistical Baseline Models

As mentioned in Progress Report 1, this script trains statistical models
(logistic regression, random forest, SVM) for comparison with deep learning.

Usage:
    python scripts/train_baseline_statistical.py --model logistic
    python scripts/train_baseline_statistical.py --model random_forest
    python scripts/train_baseline_statistical.py --model svm --output results/
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.baseline_statistical import StatisticalBaseline, FeatureExtractor


def load_manifest(manifest_path: str, ct_dir: str, mask_dir: str):
    """Load manifest and prepare file paths."""
    df = pd.read_csv(manifest_path)
    
    ct_paths = [str(Path(ct_dir) / row['ct_filename']) for _, row in df.iterrows()]
    mask_paths = [str(Path(mask_dir) / row['mask_filename']) for _, row in df.iterrows()]
    
    # Create binary labels (has nodule = 1, no nodule = 0)
    # For now, assume all samples have nodules (adjust as needed)
    labels = [1] * len(df)  # Modify based on your data
    
    return ct_paths, mask_paths, labels


def plot_results(results: dict, output_dir: Path):
    """Plot comparison of different models."""
    # Prepare data for plotting
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']
    
    data = []
    for model in models:
        for metric in metrics:
            if metric in results[model]:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': results[model][metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Value', hue='Model')
    plt.ylim(0, 1)
    plt.title('Statistical Baseline Models - Performance Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.legend(title='Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = output_dir / 'statistical_baseline_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved comparison plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train statistical baseline models for comparison'
    )
    parser.add_argument('--model', type=str, default='logistic',
                       choices=['logistic', 'random_forest', 'svm', 'all'],
                       help='Model type to train')
    parser.add_argument('--data_dir', type=str,
                       default='dataset/sample',
                       help=('Path to a data directory containing ct/ and '
                             'masks/ sub-folders. Defaults to the bundled '
                             'smoke-test sample. Override with the full '
                             'LIDC path to actually train baselines.'))
    parser.add_argument('--output', type=str, default='results/statistical_baseline',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Statistical Baseline Model Training")
    print("(As mentioned in Progress Report 1)")
    print("="*60)
    
    # Load data -- everything is resolved relative to --data_dir so this
    # script is portable across machines.
    data_dir = Path(args.data_dir)
    ct_dir = data_dir / 'ct'
    mask_dir = data_dir / 'masks'
    train_manifest = data_dir / 'train_manifest.csv'
    val_manifest = data_dir / 'val_manifest.csv'
    
    print("\nLoading training data...")
    train_ct_paths, train_mask_paths, train_labels = load_manifest(
        train_manifest, ct_dir, mask_dir
    )
    
    print("Loading validation data...")
    val_ct_paths, val_mask_paths, val_labels = load_manifest(
        val_manifest, ct_dir, mask_dir
    )
    
    # Determine which models to train
    if args.model == 'all':
        model_types = ['logistic', 'random_forest', 'svm']
    else:
        model_types = [args.model]
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Create model
        model = StatisticalBaseline(model_type=model_type)
        
        # Extract features and prepare data
        print("\nExtracting training features...")
        X_train, y_train = model.prepare_data(
            train_ct_paths[:10],  # Use subset for faster testing
            train_mask_paths[:10],
            train_labels[:10]
        )
        
        print("Extracting validation features...")
        X_val, y_val = model.prepare_data(
            val_ct_paths[:5],  # Use subset
            val_mask_paths[:5],
            val_labels[:5]
        )
        
        # Train model
        model.train(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        metrics = model.evaluate(X_val, y_val)
        
        # Print results
        print(f"\nResults for {model_type}:")
        print("-" * 40)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric:20s}: {value:.4f}")
            else:
                print(f"  {metric:20s}: {value}")
        
        results[model_type] = metrics
        
        # Save model
        model_path = output_dir / f'{model_type}_model.pkl'
        model.save(str(model_path))
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_dir / f'{model_type}_metrics.csv', index=False)
    
    # Create comparison plot if multiple models
    if len(model_types) > 1:
        print("\nCreating comparison plot...")
        plot_results(results, output_dir)
    
    # Save all results
    all_results_df = pd.DataFrame(results).T
    all_results_df.to_csv(output_dir / 'all_models_comparison.csv')
    print(f"\n✓ Saved all results to {output_dir / 'all_models_comparison.csv'}")
    
    print("\n" + "="*60)
    print("Statistical baseline training complete!")
    print("="*60)
    
    # Print summary
    print("\nSummary - Best performing model:")
    best_model = max(results.items(), key=lambda x: x[1].get('f1_score', 0))
    print(f"  Model: {best_model[0]}")
    print(f"  F1 Score: {best_model[1]['f1_score']:.4f}")
    print(f"  ROC-AUC: {best_model[1]['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
