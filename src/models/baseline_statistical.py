"""
Statistical Baseline Models for Comparison

Implements logistic regression and other statistical methods
as mentioned in Progress Report 1.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import nibabel as nib
from typing import Tuple, Dict
import pickle
from pathlib import Path


class FeatureExtractor:
    """
    Extract hand-crafted features from CT scans for statistical models.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def extract_intensity_features(self, ct_volume: np.ndarray) -> np.ndarray:
        """Extract intensity-based features."""
        features = []
        
        # Basic statistics
        features.append(np.mean(ct_volume))
        features.append(np.std(ct_volume))
        features.append(np.median(ct_volume))
        features.append(np.min(ct_volume))
        features.append(np.max(ct_volume))
        
        # Percentiles
        for p in [10, 25, 75, 90]:
            features.append(np.percentile(ct_volume, p))
        
        # Histogram features
        hist, _ = np.histogram(ct_volume, bins=10, range=(-1000, 400))
        features.extend(hist / hist.sum())  # Normalized histogram
        
        return np.array(features)
    
    def extract_texture_features(self, ct_volume: np.ndarray) -> np.ndarray:
        """Extract texture features (simplified GLCM-like features)."""
        features = []
        
        # Gradient magnitude
        gz = np.gradient(ct_volume, axis=0)
        gy = np.gradient(ct_volume, axis=1)
        gx = np.gradient(ct_volume, axis=2)
        grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
        
        features.append(np.mean(grad_mag))
        features.append(np.std(grad_mag))
        
        # Local variance (3x3x3 neighborhoods)
        from scipy.ndimage import generic_filter
        local_var = generic_filter(ct_volume, np.var, size=3)
        features.append(np.mean(local_var))
        features.append(np.std(local_var))
        
        return np.array(features)
    
    def extract_shape_features(self, mask_volume: np.ndarray) -> np.ndarray:
        """Extract shape features from mask."""
        features = []
        
        # Volume
        volume = np.sum(mask_volume > 0)
        features.append(volume)
        
        # Surface area (approximate)
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask_volume)
        surface = np.sum(mask_volume) - np.sum(eroded)
        features.append(surface)
        
        # Compactness
        if volume > 0:
            compactness = (surface ** 1.5) / volume if volume > 0 else 0
            features.append(compactness)
        else:
            features.append(0)
        
        # Bounding box dimensions
        coords = np.argwhere(mask_volume > 0)
        if len(coords) > 0:
            bbox = coords.max(axis=0) - coords.min(axis=0)
            features.extend(bbox)
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def extract_all_features(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray = None
    ) -> np.ndarray:
        """Extract all features from a CT volume."""
        features = []
        
        # Intensity features
        features.extend(self.extract_intensity_features(ct_volume))
        
        # Texture features
        features.extend(self.extract_texture_features(ct_volume))
        
        # Shape features (if mask available)
        if mask_volume is not None:
            features.extend(self.extract_shape_features(mask_volume))
        
        return np.array(features)


class StatisticalBaseline:
    """
    Statistical baseline models for comparison with deep learning.
    
    Implements:
    - Logistic Regression
    - Random Forest
    - SVM
    """
    
    def __init__(self, model_type: str = "logistic"):
        """
        Args:
            model_type: 'logistic', 'random_forest', or 'svm'
        """
        self.model_type = model_type
        self.feature_extractor = FeatureExtractor()
        
        # Initialize model
        if model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_data(
        self,
        ct_paths: list,
        mask_paths: list,
        labels: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from CT scans and prepare for training.
        
        Args:
            ct_paths: List of paths to CT scans
            mask_paths: List of paths to masks
            labels: List of labels (0 or 1 for classification)
        
        Returns:
            X: Feature matrix [N, D]
            y: Labels [N]
        """
        X = []
        y = []
        
        print(f"Extracting features from {len(ct_paths)} samples...")
        
        for ct_path, mask_path, label in zip(ct_paths, mask_paths, labels):
            # Load volumes
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata()
            
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            
            # Extract features
            features = self.feature_extractor.extract_all_features(
                ct_data, mask_data
            )
            
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features
        X = self.feature_extractor.scaler.fit_transform(X)
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the statistical model."""
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("✓ Training complete")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X_test)[:, 1]
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'sensitivity': recall_score(y_test, y_pred, zero_division=0),  # Same as recall
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.feature_extractor.scaler,
                'model_type': self.model_type
            }, f)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_extractor.scaler = data['scaler']
            self.model_type = data['model_type']
        print(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    print("Testing Statistical Baseline Models...")
    
    # Create dummy data
    X_train = np.random.randn(100, 30)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 30)
    y_test = np.random.randint(0, 2, 20)
    
    # Test each model type
    for model_type in ['logistic', 'random_forest', 'svm']:
        print(f"\nTesting {model_type}...")
        
        model = StatisticalBaseline(model_type=model_type)
        model.model.fit(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        print(f"Results for {model_type}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\n✓ All statistical models tested successfully!")
