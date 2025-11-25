import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import warnings
from abc import ABC, abstractmethod
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available.")

from ..models.base import BaselineModel
from ..models.logistic_regression import LogisticRegressionModel
from ..models.xgboost_model import XGBoostModel


class SHAPExplainer:

    def __init__(self, background_size: int = 100, random_state: int = 42):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability.")
        
        self.background_size = background_size
        self.random_state = random_state
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, model: BaselineModel, X_background: Union[pd.DataFrame, np.ndarray], 
            feature_names: Optional[List[str]] = None) -> 'SHAPExplainer':
        if hasattr(model, 'is_fitted'):
            if not model.is_fitted:
                raise ValueError("Model must be fitted before creating explainer")
        elif hasattr(model, 'n_features_in_'):
            pass
        else:
            if not (hasattr(model, 'coef_') or hasattr(model, 'estimators_') or hasattr(model, 'tree_')):
                raise ValueError("Model must be fitted before creating explainer")

        if isinstance(X_background, pd.DataFrame):
            self.feature_names = list(X_background.columns)
        elif isinstance(X_background, np.ndarray):
            if feature_names is not None:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(X_background.shape[1])]
            X_background = pd.DataFrame(X_background, columns=self.feature_names)
        else:
            raise ValueError("X_background must be a pandas DataFrame or numpy array")
        
        if len(X_background) > self.background_size:
            np.random.seed(self.random_state)
            background_indices = np.random.choice(
                len(X_background), 
                size=self.background_size, 
                replace=False
            )
            self.background_data = X_background.iloc[background_indices].copy()
        else:
            self.background_data = X_background.copy()
        
        if isinstance(model, XGBoostModel):
            try:
                self.explainer = shap.TreeExplainer(
                    model.base_model,
                    data=self.background_data,
                    feature_perturbation='interventional'
                )
            except Exception:
                self.explainer = shap.TreeExplainer(model.base_model)
        elif isinstance(model, LogisticRegressionModel):
            def predict_fn(X):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=self.feature_names)
                return model.predict_proba(X)[:, 1]
            
            self.explainer = shap.KernelExplainer(
                predict_fn,
                self.background_data,
                link="identity"
            )
        elif hasattr(model, 'estimators_') or hasattr(model, 'tree_'):
            try:
                self.explainer = shap.TreeExplainer(
                    model,
                    data=self.background_data,
                    feature_perturbation='interventional'
                )
            except Exception as e1:
                try:
                    self.explainer = shap.TreeExplainer(model)
                except Exception as e2:
                    warnings.warn(
                        f"TreeExplainer failed ({str(e1)}), falling back to KernelSHAP. "
                        "This may be slower."
                    )
                    def predict_fn(X):
                        if isinstance(X, np.ndarray):
                            X = pd.DataFrame(X, columns=self.feature_names)
                        return model.predict_proba(X)[:, 1]
                    
                    self.explainer = shap.KernelExplainer(
                        predict_fn,
                        self.background_data,
                        link="identity"
                    )
        else:
            def predict_fn(X):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=self.feature_names)
                return model.predict_proba(X)[:, 1]
            
            self.explainer = shap.KernelExplainer(
                predict_fn,
                self.background_data,
                link="identity"
            )
        
        self.model = model
        self.is_fitted = True
        return self
    
    def explain_model(self, X: Union[pd.DataFrame, np.ndarray], max_evals: int = 1000) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Explainer must be fitted before generating explanations")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        elif isinstance(X, pd.DataFrame):
            if list(X.columns) != self.feature_names:
                raise ValueError("Feature names must match those used during fitting")
        else:
            raise ValueError("X must be a pandas DataFrame or numpy array")
        
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_values = self.explainer.shap_values(X)
        else:
            shap_values = self.explainer.shap_values(X, nsamples=max_evals, silent=True)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        return shap_values
    
    def get_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        importance_scores = np.mean(np.abs(shap_values), axis=0)

        if importance_scores.ndim > 1:
            importance_scores = importance_scores.flatten()
        
        return {name: float(score) for name, score in zip(self.feature_names, importance_scores)}
    
    def get_top_features(self, shap_values: np.ndarray, top_k: int = 10) -> List[str]:
        importance_dict = self.get_feature_importance(shap_values)
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in sorted_features[:top_k]]
    
    def explain_instance(self, X: pd.DataFrame, instance_idx: int, 
                        max_evals: int = 1000) -> Dict[str, Any]:
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        instance_data = X.iloc[[instance_idx]]
        shap_values = self.explain_model(instance_data, max_evals=max_evals)
        instance_shap = shap_values[0]
        
        prediction_proba = self.model.predict_proba(instance_data)[0, 1]
        prediction = self.model.predict(instance_data)[0]
        
        feature_contributions = []
        for i, feature in enumerate(self.feature_names):
            feature_contributions.append({
                'feature': feature,
                'value': float(instance_data.iloc[0, i]),
                'shap_value': float(instance_shap[i]),
                'contribution': 'positive' if instance_shap[i] > 0 else 'negative'
            })
        
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'instance_idx': instance_idx,
            'prediction': int(prediction),
            'prediction_proba': float(prediction_proba),
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0,
            'feature_contributions': feature_contributions,
            'shap_sum': float(np.sum(instance_shap))
        }
    
    def group_wise_analysis(self, shap_values: np.ndarray, 
                           protected_attr: np.ndarray, 
                           group_names: Optional[List[str]] = None) -> Dict[str, Any]:

        if len(shap_values) != len(protected_attr):
            raise ValueError("SHAP values and protected attributes must have same length")
        
        unique_groups = np.unique(protected_attr)
        if group_names is None:
            group_names = [f"Group_{group}" for group in unique_groups]
        elif len(group_names) != len(unique_groups):
            raise ValueError("Number of group names must match number of unique groups")
        
        group_analysis = {}
        
        for i, group in enumerate(unique_groups):
            group_mask = protected_attr == group
            group_shap = shap_values[group_mask]
            
            if len(group_shap) == 0:
                continue
            
            group_stats = {
                'n_samples': int(np.sum(group_mask)),
                'mean_shap': np.mean(group_shap, axis=0).tolist(),
                'std_shap': np.std(group_shap, axis=0).tolist(),
                'feature_importance': self.get_feature_importance(group_shap),
                'top_features': self.get_top_features(group_shap, top_k=5)
            }
            
            group_analysis[group_names[i]] = group_stats
        
        if len(group_analysis) >= 2:
            group_comparison = self._compare_groups(group_analysis)
            group_analysis['group_comparison'] = group_comparison
        
        return group_analysis
    
    def _compare_groups(self, group_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        group_names = [name for name in group_analysis.keys() if name != 'group_comparison']
        
        if len(group_names) < 2:
            return {}
        
        group1_name, group2_name = group_names[0], group_names[1]
        group1_importance = group_analysis[group1_name]['feature_importance']
        group2_importance = group_analysis[group2_name]['feature_importance']
        
        importance_diff = {}
        for feature in self.feature_names:
            diff = group1_importance.get(feature, 0) - group2_importance.get(feature, 0)
            importance_diff[feature] = diff
        
        sorted_diff = sorted(importance_diff.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'compared_groups': [group1_name, group2_name],
            'importance_differences': importance_diff,
            'top_differential_features': [feature for feature, _ in sorted_diff[:10]],
            'max_difference': max(abs(diff) for diff in importance_diff.values()),
            'mean_absolute_difference': np.mean([abs(diff) for diff in importance_diff.values()])
        }
    
    def get_background_summary(self) -> Dict[str, Any]:
        if self.background_data is None:
            return {}
        
        return {
            'n_samples': len(self.background_data),
            'n_features': len(self.background_data.columns),
            'feature_names': list(self.background_data.columns),
            'feature_means': self.background_data.mean().to_dict(),
            'feature_stds': self.background_data.std().to_dict()
        }