import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from .base import BaselineModel

class LogisticRegressionModel(BaselineModel):
    def __init__(self, 
                 random_state: int = 42,
                 max_iter: int = 1000,
                 tune_hyperparameters: bool = True,
                 calibrate: bool = True,
                 cv_folds: int = 5,
                 **kwargs):
        super().__init__(
            random_state=random_state,
            max_iter=max_iter,
            tune_hyperparameters=tune_hyperparameters,
            calibrate=calibrate,
            cv_folds=cv_folds,
            **kwargs
        )
        self.scaler = StandardScaler()
        self.base_model = None
        self.calibrated_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None) -> 'LogisticRegressionModel':
        self._validate_input(X)
        self.feature_names = list(X.columns)
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        if self.hyperparameters.get('tune_hyperparameters', True):
            self.base_model = self._tune_hyperparameters(X_scaled, y, sample_weight)
        else:
            self.base_model = LogisticRegression(
                random_state=self.hyperparameters.get('random_state', 42),
                max_iter=self.hyperparameters.get('max_iter', 1000),
                class_weight='balanced'  # Handle class imbalance
            )
            self.base_model.fit(X_scaled, y, sample_weight=sample_weight)
        
        if self.hyperparameters.get('calibrate', True):
            self.calibrated_model = CalibratedClassifierCV(
                self.base_model, 
                method='sigmoid',
                cv=self.hyperparameters.get('cv_folds', 5)
            )
            self.calibrated_model.fit(X_scaled, y, sample_weight=sample_weight)
            self.model = self.calibrated_model
        else:
            self.model = self.base_model
        
        self.is_fitted = True
        return self
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                            sample_weight: Optional[np.ndarray] = None) -> LogisticRegression:
        param_grid = [
            # L2 penalty with different solvers
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2'],
                'solver': ['liblinear'],
                'class_weight': [None, 'balanced']
            },
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2'],
                'solver': ['saga'],
                'class_weight': [None, 'balanced']
            },
            # L1 penalty with compatible solvers
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1'],
                'solver': ['liblinear'],
                'class_weight': [None, 'balanced']
            },
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1'],
                'solver': ['saga'],
                'class_weight': [None, 'balanced']
            }
        ]
        
        base_estimator = LogisticRegression(
            random_state=self.hyperparameters.get('random_state', 42),
            max_iter=self.hyperparameters.get('max_iter', 1000)
        )
        
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=self.hyperparameters.get('cv_folds', 5),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
            
        grid_search.fit(X, y, **fit_params)
        
        return grid_search.best_estimator_
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self._validate_input(X)
        
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return self.model.predict_proba(X_scaled)
    
    def get_coefficients(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get coefficients")
    
        if hasattr(self.base_model, 'coef_'):
            coef = self.base_model.coef_[0]
            return dict(zip(self.feature_names, coef))
        else:
            raise ValueError("Model does not have coefficients")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        if hasattr(self.base_model, 'coef_'):
            return np.abs(self.base_model.coef_[0])
        else:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'best_params': getattr(self.base_model, 'get_params', lambda: {})(),
                'is_calibrated': self.hyperparameters.get('calibrate', True),
                'n_coefficients': len(self.base_model.coef_[0]) if hasattr(self.base_model, 'coef_') else None
            })
        
        return info