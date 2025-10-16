from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class BaselineModel(ABC):
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.hyperparameters = kwargs
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None) -> 'BaselineModel':
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= threshold).astype(int)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None
    
    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'model_type': self.__class__.__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaselineModel':
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance with saved hyperparameters
        instance = cls(**model_data['hyperparameters'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'n_features': len(self.feature_names) if self.feature_names else None
        }
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
        
        if self.is_fitted and self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                # Select only the features used during training
                X = X[self.feature_names]
        
        # Checking for missing values
        if X.isnull().any().any():
            raise ValueError("Input data contains missing values")
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.__class__.__name__}({status})"