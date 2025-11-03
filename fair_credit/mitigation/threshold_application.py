import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from .threshold_optimization import ThresholdOptimizer


class ThresholdApplicationSystem:
    def __init__(self, fairness_constraint: str = 'equal_opportunity',
                 constraint_tolerance: float = 0.01,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 **optimizer_kwargs):
        self.fairness_constraint = fairness_constraint
        self.constraint_tolerance = constraint_tolerance
        self.validation_split = validation_split
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        
        # Initializing optimizer
        self.optimizer = ThresholdOptimizer(
            fairness_constraint=fairness_constraint,
            constraint_tolerance=constraint_tolerance,
            **optimizer_kwargs
        )
        
        self.is_fitted = False
        self.optimal_thresholds_ = None
        self.validation_metrics_ = None
        self.decision_boundaries_ = None
        self.validation_split_info_ = None
        
    def fit_thresholds(self, model, X: pd.DataFrame, y: pd.Series, 
                      protected_attr: pd.Series) -> 'ThresholdApplicationSystem':
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val, attr_train, attr_val = train_test_split(
            X, y, protected_attr,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=pd.concat([y, protected_attr], axis=1)  # Stratify by both label and group
        )
        
        # Storing validation split information
        self.validation_split_info_ = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'train_groups': attr_train.value_counts().to_dict(),
            'val_groups': attr_val.value_counts().to_dict()
        }
        
        # predictions on validation set
        y_val_proba_full = model.predict_proba(X_val)
        if y_val_proba_full.ndim == 2 and y_val_proba_full.shape[1] == 2:
            y_val_proba = y_val_proba_full[:, 1]
        else:
            y_val_proba = y_val_proba_full.flatten()[:len(X_val)]
        
        y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
        attr_val_np = attr_val.values if hasattr(attr_val, 'values') else np.array(attr_val)
        
        self.optimizer.fit(X_val, y_val, attr_val)
        
        self.optimal_thresholds_ = self.optimizer.optimize_thresholds(
            y_val_np, y_val_proba, attr_val_np
        )
        
        self.decision_boundaries_ = self._compute_decision_boundaries(
            y_val_proba, attr_val_np, self.optimal_thresholds_
        )
        
        y_train_proba_full = model.predict_proba(X_train)
        if y_train_proba_full.ndim == 2 and y_train_proba_full.shape[1] == 2:
            y_train_proba = y_train_proba_full[:, 1]
        else:
            y_train_proba = y_train_proba_full.flatten()[:len(X_train)]
        y_train_np = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        attr_train_np = attr_train.values if hasattr(attr_train, 'values') else np.array(attr_train)
        
        self.validation_metrics_ = self._validate_thresholds(
            y_train_np, y_train_proba, attr_train_np,
            y_val_np, y_val_proba, attr_val_np
        )
        
        self.is_fitted = True
        return self
    
    def apply_thresholds(self, model, X: pd.DataFrame, 
                        protected_attr: pd.Series) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("ThresholdApplicationSystem must be fitted before applying thresholds")
        
        y_proba_full = model.predict_proba(X)
        if y_proba_full.ndim == 2 and y_proba_full.shape[1] == 2:
            y_proba = y_proba_full[:, 1]
        else:
            y_proba = y_proba_full.flatten()[:len(X)]
        
        protected_attr_np = protected_attr.values if hasattr(protected_attr, 'values') else np.array(protected_attr)
        
        return self.optimizer.apply_thresholds(y_proba, protected_attr_np, self.optimal_thresholds_)
    
    def _compute_decision_boundaries(self, y_proba: np.ndarray, protected_attr: np.ndarray,
                                   thresholds: Dict[str, float]) -> Dict[str, Any]:
        boundaries = {}
        
        for group_str, threshold in thresholds.items():
            group = int(group_str) if group_str.isdigit() else group_str
            group_mask = protected_attr == group
            
            if np.any(group_mask):
                group_proba = y_proba[group_mask]
                
                # Computing statistics around the decision boundary
                boundaries[group_str] = {
                    'threshold': threshold,
                    'n_samples': len(group_proba),
                    'n_positive_predictions': np.sum(group_proba >= threshold),
                    'n_negative_predictions': np.sum(group_proba < threshold),
                    'positive_rate': np.mean(group_proba >= threshold),
                    'prob_stats': {
                        'min': float(np.min(group_proba)),
                        'max': float(np.max(group_proba)),
                        'mean': float(np.mean(group_proba)),
                        'std': float(np.std(group_proba)),
                        'median': float(np.median(group_proba))
                    },
                    'boundary_region': {
                        'near_threshold_count': np.sum(np.abs(group_proba - threshold) <= 0.1),
                        'near_threshold_fraction': np.mean(np.abs(group_proba - threshold) <= 0.1)
                    }
                }
        
        return boundaries
    
    def _validate_thresholds(self, y_train: np.ndarray, y_train_proba: np.ndarray, 
                           attr_train: np.ndarray, y_val: np.ndarray, 
                           y_val_proba: np.ndarray, attr_val: np.ndarray) -> Dict[str, Any]:

        from ..fairness.metrics import FairnessMetrics
        
        # Applying thresholds to both sets
        train_pred = self.optimizer.apply_thresholds(y_train_proba, attr_train, self.optimal_thresholds_)
        val_pred = self.optimizer.apply_thresholds(y_val_proba, attr_val, self.optimal_thresholds_)
        
        # Computing fairness metrics
        fairness_calculator = FairnessMetrics()
        
        if self.fairness_constraint == 'equal_opportunity':
            train_fairness = fairness_calculator.compute_equal_opportunity(y_train, train_pred, attr_train)
            val_fairness = fairness_calculator.compute_equal_opportunity(y_val, val_pred, attr_val)
        elif self.fairness_constraint == 'equalized_odds':
            train_fairness = fairness_calculator.compute_equalized_odds(y_train, train_pred, attr_train)
            val_fairness = fairness_calculator.compute_equalized_odds(y_val, val_pred, attr_val)
        else:
            train_fairness = {}
            val_fairness = {}
        
        # Computing accuracy
        train_accuracy = np.mean(y_train == train_pred)
        val_accuracy = np.mean(y_val == val_pred)
        
        fairness_gap_key = f'{self.fairness_constraint}_gap'
        train_gap = train_fairness.get(fairness_gap_key, 0)
        val_gap = val_fairness.get(fairness_gap_key, 0)
        
        gap_difference = abs(train_gap - val_gap)
        accuracy_difference = abs(train_accuracy - val_accuracy)

        overfitting_indicators = {
            'fairness_gap_difference': gap_difference,
            'accuracy_difference': accuracy_difference,
            'fairness_overfitting': bool(gap_difference > 0.05),  # More than 5% difference
            'accuracy_overfitting': bool(accuracy_difference > 0.05),  # More than 5% difference
            'overall_overfitting': bool(gap_difference > 0.05 or accuracy_difference > 0.05)
        }
        
        return {
            'train_metrics': {
                'fairness': train_fairness,
                'accuracy': train_accuracy
            },
            'val_metrics': {
                'fairness': val_fairness,
                'accuracy': val_accuracy
            },
            'overfitting_check': overfitting_indicators,
            'constraint_satisfaction': {
                'train_satisfied': bool(train_gap <= self.constraint_tolerance),
                'val_satisfied': bool(val_gap <= self.constraint_tolerance),
                'both_satisfied': bool(train_gap <= self.constraint_tolerance and 
                                     val_gap <= self.constraint_tolerance)
            }
        }
    
    def verify_decision_boundaries(self, model, X: pd.DataFrame, y: pd.Series,
                                 protected_attr: pd.Series) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("ThresholdApplicationSystem must be fitted before verification")
        
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = self.apply_thresholds(model, X, protected_attr)
        
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        protected_attr_np = protected_attr.values if hasattr(protected_attr, 'values') else np.array(protected_attr)
        
        verification_results = {}
        
        for group_str, threshold in self.optimal_thresholds_.items():
            group = int(group_str) if group_str.isdigit() else group_str
            group_mask = protected_attr_np == group
            
            if np.any(group_mask):
                group_proba = y_proba[group_mask]
                group_pred = y_pred[group_mask]
                group_true = y_np[group_mask]
                
                expected_pred = (group_proba >= threshold).astype(int)
                threshold_applied_correctly = np.array_equal(group_pred, expected_pred)
                
                near_threshold_mask = np.abs(group_proba - threshold) <= 0.05  # Within 5% of threshold
                near_threshold_predictions = group_pred[near_threshold_mask]
                near_threshold_probabilities = group_proba[near_threshold_mask]
                
                group_accuracy = np.mean(group_true == group_pred)
                
                if np.any(group_true == 1):
                    group_tpr = np.sum((group_pred == 1) & (group_true == 1)) / np.sum(group_true == 1)
                else:
                    group_tpr = 0.0
                
                if np.any(group_true == 0):
                    group_fpr = np.sum((group_pred == 1) & (group_true == 0)) / np.sum(group_true == 0)
                else:
                    group_fpr = 0.0
                
                verification_results[group_str] = {
                    'threshold_applied_correctly': threshold_applied_correctly,
                    'group_size': len(group_proba),
                    'threshold_value': threshold,
                    'positive_predictions': np.sum(group_pred == 1),
                    'positive_rate': np.mean(group_pred == 1),
                    'accuracy': group_accuracy,
                    'tpr': group_tpr,
                    'fpr': group_fpr,
                    'near_threshold_analysis': {
                        'n_samples': len(near_threshold_predictions),
                        'fraction_of_group': len(near_threshold_predictions) / len(group_proba),
                        'prediction_consistency': np.all(
                            (near_threshold_probabilities >= threshold) == (near_threshold_predictions == 1)
                        ) if len(near_threshold_predictions) > 0 else True
                    },
                    'probability_distribution': {
                        'below_threshold': np.sum(group_proba < threshold),
                        'at_threshold': np.sum(group_proba == threshold),
                        'above_threshold': np.sum(group_proba > threshold),
                        'min_prob': float(np.min(group_proba)),
                        'max_prob': float(np.max(group_proba)),
                        'mean_prob': float(np.mean(group_proba))
                    }
                }
        
        all_thresholds_correct = all(
            result['threshold_applied_correctly'] 
            for result in verification_results.values()
        )
        
        all_near_threshold_consistent = all(
            result['near_threshold_analysis']['prediction_consistency']
            for result in verification_results.values()
        )
        
        verification_summary = {
            'all_thresholds_applied_correctly': all_thresholds_correct,
            'all_near_threshold_consistent': all_near_threshold_consistent,
            'total_groups': len(verification_results),
            'groups_verified': list(verification_results.keys()),
            'overall_verification_passed': all_thresholds_correct and all_near_threshold_consistent
        }
        
        return {
            'group_results': verification_results,
            'summary': verification_summary
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        info = {
            'fairness_constraint': self.fairness_constraint,
            'constraint_tolerance': self.constraint_tolerance,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            info.update({
                'optimal_thresholds': self.optimal_thresholds_.copy() if self.optimal_thresholds_ else {},
                'decision_boundaries': self.decision_boundaries_.copy() if self.decision_boundaries_ else {},
                'validation_split_info': self.validation_split_info_.copy() if self.validation_split_info_ else {},
                'validation_metrics': self.validation_metrics_.copy() if self.validation_metrics_ else {},
                'optimizer_info': self.optimizer.get_optimization_info() if self.optimizer.is_fitted else {}
            })
        
        return info
    
    def evaluate_system_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                  protected_attr_test: pd.Series,
                                  baseline_threshold: float = 0.5) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("ThresholdApplicationSystem must be fitted before evaluation")
        
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        baseline_pred = (y_test_proba >= baseline_threshold).astype(int)
        
        system_pred = self.apply_thresholds(model, X_test, protected_attr_test)
        
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        protected_attr_np = protected_attr_test.values if hasattr(protected_attr_test, 'values') else np.array(protected_attr_test)
        
        from ..fairness.metrics import FairnessMetrics
        fairness_calculator = FairnessMetrics()
        
        if self.fairness_constraint == 'equal_opportunity':
            baseline_fairness = fairness_calculator.compute_equal_opportunity(y_test_np, baseline_pred, protected_attr_np)
            system_fairness = fairness_calculator.compute_equal_opportunity(y_test_np, system_pred, protected_attr_np)
        elif self.fairness_constraint == 'equalized_odds':
            baseline_fairness = fairness_calculator.compute_equalized_odds(y_test_np, baseline_pred, protected_attr_np)
            system_fairness = fairness_calculator.compute_equalized_odds(y_test_np, system_pred, protected_attr_np)
        else:
            baseline_fairness = {}
            system_fairness = {}
        
        baseline_accuracy = np.mean(y_test_np == baseline_pred)
        system_accuracy = np.mean(y_test_np == system_pred)
        
        fairness_gap_key = f'{self.fairness_constraint}_gap'
        baseline_gap = baseline_fairness.get(fairness_gap_key, 0)
        system_gap = system_fairness.get(fairness_gap_key, 0)
        
        fairness_improvement = baseline_gap - system_gap
        fairness_relative_improvement = (fairness_improvement / baseline_gap) if baseline_gap > 0 else 0
        
        accuracy_change = system_accuracy - baseline_accuracy
        accuracy_relative_change = (accuracy_change / baseline_accuracy) if baseline_accuracy > 0 else 0
        
        success_criteria = {
            'fairness_improved': bool(fairness_improvement > 0),
            'fairness_50_percent_improvement': bool(fairness_relative_improvement >= 0.5),
            'accuracy_preserved': bool(accuracy_change >= -0.02),  # Allow up to 2% drop
            'constraint_satisfied': bool(system_gap <= self.constraint_tolerance),
            'overall_success': bool(fairness_relative_improvement >= 0.5 and 
                                  accuracy_change >= -0.02 and 
                                  system_gap <= self.constraint_tolerance)
        }
        
        return {
            'baseline_metrics': {
                'fairness': baseline_fairness,
                'accuracy': baseline_accuracy,
                'threshold': baseline_threshold
            },
            'system_metrics': {
                'fairness': system_fairness,
                'accuracy': system_accuracy,
                'thresholds': self.optimal_thresholds_.copy()
            },
            'improvements': {
                'fairness_absolute': fairness_improvement,
                'fairness_relative': fairness_relative_improvement,
                'fairness_percentage': fairness_relative_improvement * 100,
                'accuracy_absolute': accuracy_change,
                'accuracy_relative': accuracy_relative_change,
                'accuracy_percentage': accuracy_relative_change * 100
            },
            'success_criteria': success_criteria,
            'validation_check': self.validation_metrics_.copy() if self.validation_metrics_ else {}
        }