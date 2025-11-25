import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from .base import BaselineModel

class XGBoostModel(BaselineModel):
    def __init__(self, 
                 random_state: int = 42,
                 n_estimators: int = 100,
                 tune_hyperparameters: bool = True,
                 calibrate: bool = True,
                 cv_folds: int = 5,
                 early_stopping_rounds: int = 10,
                 **kwargs):
        super().__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            tune_hyperparameters=tune_hyperparameters,
            calibrate=calibrate,
            cv_folds=cv_folds,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs
        )
        self.base_model = None
        self.calibrated_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None) -> 'XGBoostModel':
        self._validate_input(X)
        self.feature_names = list(X.columns)
        
        if self.hyperparameters.get('tune_hyperparameters', True):
            self.base_model = self._tune_hyperparameters(X, y, sample_weight)
        else:
            self.base_model = xgb.XGBClassifier(
                random_state=self.hyperparameters.get('random_state', 42),
                n_estimators=self.hyperparameters.get('n_estimators', 100),
                eval_metric='logloss',
                use_label_encoder=False
            )

            if len(X) > 100:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.hyperparameters.get('random_state', 42),
                    stratify=y
                )
                
                train_weight = None
                val_weight = None
                if sample_weight is not None:
                    train_mask = X.index.isin(X_train.index)
                    val_mask = X.index.isin(X_val.index)
                    train_weight = sample_weight[train_mask]
                    val_weight = sample_weight[val_mask]

                self.base_model.set_params(early_stopping_rounds=self.hyperparameters.get('early_stopping_rounds', 10))
                
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'verbose': False
                }
                if train_weight is not None:
                    fit_params['sample_weight'] = train_weight
                if val_weight is not None:
                    fit_params['sample_weight_eval_set'] = [val_weight]
                
                self.base_model.fit(X_train, y_train, **fit_params)
            else:
                self.base_model.fit(X, y, sample_weight=sample_weight)
        
        if self.hyperparameters.get('calibrate', True):
            calibration_base = xgb.XGBClassifier(
                random_state=self.hyperparameters.get('random_state', 42),
                n_estimators=self.base_model.n_estimators,
                max_depth=getattr(self.base_model, 'max_depth', 6),
                learning_rate=getattr(self.base_model, 'learning_rate', 0.3),
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            self.calibrated_model = CalibratedClassifierCV(
                calibration_base, 
                method='sigmoid',
                cv=self.hyperparameters.get('cv_folds', 5)
            )
            self.calibrated_model.fit(X, y, sample_weight=sample_weight)
            self.model = self.calibrated_model
        else:
            self.model = self.base_model
        
        self.is_fitted = True
        return self
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                            sample_weight: Optional[np.ndarray] = None) -> xgb.XGBClassifier:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        base_estimator = xgb.XGBClassifier(
            random_state=self.hyperparameters.get('random_state', 42),
            eval_metric='logloss',
            use_label_encoder=False
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
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        if hasattr(self.base_model, 'feature_importances_'):
            return self.base_model.feature_importances_
        else:
            return None
    
    def get_feature_importance_dict(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance = self.get_feature_importance()
        if importance is not None and self.feature_names is not None:
            importance_values = [float(imp) for imp in importance]
            return dict(zip(self.feature_names, importance_values))
        else:
            return {}
    
    def get_booster_info(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get booster info")
        
        info = {}
        if hasattr(self.base_model, 'get_booster'):
            booster = self.base_model.get_booster()
            info.update({
                'num_boosted_rounds': booster.num_boosted_rounds(),
                'num_features': booster.num_features(),
            })
        
        if hasattr(self.base_model, 'best_iteration'):
            info['best_iteration'] = self.base_model.best_iteration
        
        if hasattr(self.base_model, 'best_score'):
            info['best_score'] = self.base_model.best_score
            
        return info
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'best_params': getattr(self.base_model, 'get_params', lambda: {})(),
                'is_calibrated': self.hyperparameters.get('calibrate', True),
                'booster_info': self.get_booster_info()
            })
        
        return info
    
    def plot_importance(self, max_num_features: int = 20, importance_type: str = 'weight'):
        if not self.is_fitted:
            raise ValueError("Model must be fitted to plot importance")
        
        try:
            import matplotlib.pyplot as plt
            from xgboost import plot_importance
            
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_importance(
                self.base_model,
                ax=ax,
                max_num_features=max_num_features,
                importance_type=importance_type
            )
            plt.title(f'XGBoost Feature Importance ({importance_type})')
            plt.tight_layout()
            return fig
        except ImportError:
            raise ImportError("matplotlib is required for plotting feature importance")
    
    def get_tree_info(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get tree info")
        
        info = {}
        if hasattr(self.base_model, 'get_booster'):
            booster = self.base_model.get_booster()
            
            tree_dump = booster.get_dump(dump_format='json')
            info.update({
                'num_trees': len(tree_dump),
                'tree_depths': [],
                'tree_num_nodes': []
            })
            
            import json
            for i, tree_str in enumerate(tree_dump[:10]):
                try:
                    tree_json = json.loads(tree_str)
                    depth = self._get_tree_depth(tree_json)
                    num_nodes = self._count_tree_nodes(tree_json)
                    info['tree_depths'].append(depth)
                    info['tree_num_nodes'].append(num_nodes)
                except:
                    continue
        
        return info
    
    def _get_tree_depth(self, tree_node: Dict, current_depth: int = 0) -> int:
        if 'children' not in tree_node:
            return current_depth
        
        max_child_depth = current_depth
        for child in tree_node['children']:
            child_depth = self._get_tree_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _count_tree_nodes(self, tree_node: Dict) -> int:
        count = 1
        if 'children' in tree_node:
            for child in tree_node['children']:
                count += self._count_tree_nodes(child)
        return count