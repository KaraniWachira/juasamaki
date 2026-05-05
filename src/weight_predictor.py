"""
Weight prediction module using machine learning.
Supports multiple model types: Linear Regression, Random Forest, Neural Networks.
"""

import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class FishMeasurement:
    """Fish measurement and weight data."""
    length_cm: float
    height_cm: float
    area_cm2: float
    weight_kg: float


class WeightPredictor:
    """Predict fish weight from physical measurements."""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize weight predictor.
        
        Args:
            model_type: "linear", "random_forest", or "neural"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ["length_cm", "height_cm", "area_cm2", 
                             "length_height_ratio", "area_length_ratio"]
        self.is_trained = False
    
    def extract_features(self, measurements: List[FishMeasurement]) -> pd.DataFrame:
        """
        Extract engineered features from raw measurements.
        
        Args:
            measurements: List of FishMeasurement objects
            
        Returns:
            DataFrame with features
        """
        features = []
        for m in measurements:
            f = {
                "length_cm": m.length_cm,
                "height_cm": m.height_cm,
                "area_cm2": m.area_cm2,
                "length_height_ratio": m.length_cm / (m.height_cm + 1e-6),
                "area_length_ratio": m.area_cm2 / (m.length_cm + 1e-6),
            }
            features.append(f)
        
        return pd.DataFrame(features)
    
    def train(self, measurements: List[FishMeasurement], 
             test_size: float = 0.2,
             cv_folds: int = 5) -> Dict[str, float]:
        """
        Train weight prediction model.
        
        Args:
            measurements: Training data
            test_size: Train/test split ratio
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with metrics
        """
        # Extract features and targets
        X = self.extract_features(measurements)
        y = np.array([m.weight_kg for m in measurements])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Select model
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "model_type": self.model_type,
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, 
                                    cv=cv_folds, scoring='r2')
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()
        
        self.is_trained = True
        
        print("\n" + "="*60)
        print(f"🤖 Model Training Complete ({self.model_type})")
        print("="*60)
        print(f"Test R² Score: {metrics['r2_score']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.4f} kg")
        print(f"Test MAE: {metrics['mae']:.4f} kg")
        print(f"Cross-validation R² (mean±std): {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print("="*60 + "\n")
        
        return metrics
    
    def predict(self, length_cm: float, height_cm: float, 
               area_cm2: float) -> Tuple[float, Dict]:
        """
        Predict fish weight.
        
        Args:
            length_cm: Fish length in centimeters
            height_cm: Fish height in centimeters
            area_cm2: Fish area in square centimeters
            
        Returns:
            (predicted_weight_kg, feature_dict)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Create feature vector
        features = {
            "length_cm": length_cm,
            "height_cm": height_cm,
            "area_cm2": area_cm2,
            "length_height_ratio": length_cm / (height_cm + 1e-6),
            "area_length_ratio": area_cm2 / (length_cm + 1e-6),
        }
        
        # Convert to array
        X = np.array([[features[fname] for fname in self.feature_names]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        weight = self.model.predict(X_scaled)[0]
        
        return max(weight, 0.1), features  # Ensure positive weight
    
    def predict_batch(self, measurements: List[Dict]) -> pd.DataFrame:
        """
        Predict weights for multiple measurements.
        
        Args:
            measurements: List of dicts with "length_cm", "height_cm", "area_cm2"
            
        Returns:
            DataFrame with predictions
        """
        results = []
        for m in measurements:
            weight, features = self.predict(
                m["length_cm"],
                m["height_cm"],
                m["area_cm2"]
            )
            result = {**m, "predicted_weight_kg": weight}
            results.append(result)
        
        return pd.DataFrame(results)
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file."""
        try:
            if not self.is_trained:
                print("❌ Model not trained yet")
                return False
            
            data = {
                "model": self.model,
                "scaler": self.scaler,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"✅ Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.model_type = data["model_type"]
            self.feature_names = data["feature_names"]
            self.is_trained = True
            
            print(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (for tree-based models)."""
        if not self.is_trained or self.model_type != "random_forest":
            return None
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, 
                                   self.model.feature_importances_):
            importance_dict[name] = float(importance)
        
        return dict(sorted(importance_dict.items(), 
                          key=lambda x: x[1], reverse=True))
    
    def print_feature_importance(self):
        """Print feature importance summary."""
        importance = self.get_feature_importance()
        if not importance:
            print("Feature importance not available for this model")
            return
        
        print("\n" + "="*60)
        print("📊 Feature Importance (Random Forest)")
        print("="*60)
        for feature, imp in importance.items():
            bar_length = int(imp * 50)
            print(f"{feature:.<25} {'█' * bar_length} {imp:.4f}")
        print("="*60 + "\n")
