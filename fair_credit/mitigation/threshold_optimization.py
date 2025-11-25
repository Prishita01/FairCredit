import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Callable
from scipy.optimize import minimize, differential_evolution
from .base import PostProcessingTechnique


class ThresholdOptimizer(PostProcessingTechnique):
    def __init__(self, fairness_constraint: str = 'equal_opportunity',
                 constraint_tolerance: float = 0.01,
                 optimization_method: str = 'differential_evolution',
                 **kwargs):
        super().__init__(**kwargs)
        self.fairness_constraint = fairness_constraint
        self.constraint_tolerance = constraint_tolerance
        self.optimization_method = optimization_method
        self.optimization_params = kwargs
        
        self.optimal_thresholds_ = None
        self.optimization_result_ = None
        self.validation_metrics_ = None
        self.groups_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            protected_attr: pd.Series) -> 'ThresholdOptimizer':

        self.groups_ = np.unique(protected_attr.values if hasattr(protected_attr, 'values') 
                                else np.array(protected_attr))
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                 protected_attr: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

        if not self.is_fitted:
            raise ValueError("ThresholdOptimizer must be fitted before transform")
        
        return X, y
    
    def optimize_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray,
                           protected_attr: np.ndarray) -> Dict[str, float]:

        if len(y_true) != len(y_proba) or len(y_true) != len(protected_attr):
            raise ValueError("All input arrays must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot optimize thresholds for empty dataset")
        

        unique_groups = np.unique(protected_attr)
        n_groups = len(unique_groups)
        
        if n_groups < 2:
            raise ValueError("Need at least 2 groups for fairness optimization")
        
        self.groups_ = unique_groups
        
        def objective_function(thresholds):

            total_errors = 0
            total_samples = 0
            
            for i, group in enumerate(unique_groups):
                group_mask = protected_attr == group
                if not np.any(group_mask):
                    continue
                
                group_y_true = y_true[group_mask]
                group_y_proba = y_proba[group_mask]
                threshold = thresholds[i]
                
                group_y_pred = (group_y_proba >= threshold).astype(int)
                
                errors = np.sum(group_y_true != group_y_pred)
                total_errors += errors
                total_samples += len(group_y_true)
            
            if total_samples == 0:
                return float('inf')
            
            return total_errors / total_samples
        
        def fairness_constraint_function(thresholds):
 
            if self.fairness_constraint == 'equal_opportunity':
                return self._equal_opportunity_constraints(thresholds, y_true, y_proba, protected_attr, unique_groups)
            elif self.fairness_constraint == 'equalized_odds':
                return self._equalized_odds_constraints(thresholds, y_true, y_proba, protected_attr, unique_groups)
            else:
                raise ValueError(f"Unsupported fairness constraint: {self.fairness_constraint}")
        
    
        bounds = [(0.0, 1.0) for _ in range(n_groups)]
        
        initial_guess = np.full(n_groups, 0.5)
        
        constraints = []
        if self.fairness_constraint in ['equal_opportunity', 'equalized_odds']:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: -np.max(np.abs(fairness_constraint_function(x))) + self.constraint_tolerance
            })
        
        if self.optimization_method == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds,
                seed=42,
                maxiter=1000,
                popsize=15,
                atol=1e-6,
                **{k: v for k, v in self.optimization_params.items() 
                   if k in ['strategy', 'mutation', 'recombination', 'disp']}
            )
            
            if result.success:
                constraint_violations = fairness_constraint_function(result.x)
                max_violation = np.max(np.abs(constraint_violations))
                if max_violation > self.constraint_tolerance:

                    try:
                        refined_result = minimize(
                            objective_function,
                            result.x,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9}
                        )
                        if refined_result.success:
                            result = refined_result
                    except:
                        pass
            
        elif self.optimization_method == 'minimize':
            result = minimize(
                objective_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
        else:
            raise ValueError(f"Unsupported optimization method: {self.optimization_method}")
        
        self.optimization_result_ = result
        
        if not result.success:
            print(f"Warning: Optimization failed ({result.message}). Using uniform threshold 0.5.")
            optimal_thresholds_array = np.full(n_groups, 0.5)
        else:
            optimal_thresholds_array = result.x
        
        optimal_thresholds = {}
        for i, group in enumerate(unique_groups):
            optimal_thresholds[str(group)] = float(optimal_thresholds_array[i])
        
        self.optimal_thresholds_ = optimal_thresholds
        
        self._validate_optimization_result(y_true, y_proba, protected_attr, optimal_thresholds)
        
        return optimal_thresholds
    
    def _equal_opportunity_constraints(self, thresholds: np.ndarray, y_true: np.ndarray,
                                     y_proba: np.ndarray, protected_attr: np.ndarray,
                                     unique_groups: np.ndarray) -> np.ndarray:
        tprs = []
        for i, group in enumerate(unique_groups):
            group_mask = protected_attr == group
            if not np.any(group_mask):
                tprs.append(0.0)
                continue
            
            group_y_true = y_true[group_mask]
            group_y_proba = y_proba[group_mask]
            threshold = thresholds[i]
            
            group_y_pred = (group_y_proba >= threshold).astype(int)
            
            positive_mask = group_y_true == 1
            if not np.any(positive_mask):
                tpr = 0.0 
            else:
                true_positives = np.sum((group_y_pred == 1) & (group_y_true == 1))
                total_positives = np.sum(positive_mask)
                tpr = true_positives / total_positives if total_positives > 0 else 0.0
            
            tprs.append(tpr)
        
        tprs = np.array(tprs)
        
        constraint_violations = []
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                tpr_diff = abs(tprs[i] - tprs[j])
                constraint_violations.append(tpr_diff)
        
        return np.array(constraint_violations)
    
    def _equalized_odds_constraints(self, thresholds: np.ndarray, y_true: np.ndarray,
                                  y_proba: np.ndarray, protected_attr: np.ndarray,
                                  unique_groups: np.ndarray) -> np.ndarray:

        tprs = []
        fprs = []
        
        for i, group in enumerate(unique_groups):
            group_mask = protected_attr == group
            if not np.any(group_mask):
                tprs.append(0.0)
                fprs.append(0.0)
                continue
            
            group_y_true = y_true[group_mask]
            group_y_proba = y_proba[group_mask]
            threshold = thresholds[i]
            
            group_y_pred = (group_y_proba >= threshold).astype(int)
            
            positive_mask = group_y_true == 1
            if np.any(positive_mask):
                true_positives = np.sum((group_y_pred == 1) & (group_y_true == 1))
                total_positives = np.sum(positive_mask)
                tpr = true_positives / total_positives
            else:
                tpr = 0.0
            
            negative_mask = group_y_true == 0
            if np.any(negative_mask):
                false_positives = np.sum((group_y_pred == 1) & (group_y_true == 0))
                total_negatives = np.sum(negative_mask)
                fpr = false_positives / total_negatives
            else:
                fpr = 0.0
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        tprs = np.array(tprs)
        fprs = np.array(fprs)
        
        constraint_violations = []
        
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                tpr_diff = abs(tprs[i] - tprs[j])
                constraint_violations.append(tpr_diff)
        
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                fpr_diff = abs(fprs[i] - fprs[j])
                constraint_violations.append(fpr_diff)
        
        return np.array(constraint_violations)
    
    def _validate_optimization_result(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    protected_attr: np.ndarray, thresholds: Dict[str, float]):
   
        y_pred = self.apply_thresholds(y_proba, protected_attr, thresholds)
        
        from ..fairness.metrics import FairnessMetrics
        fairness_calculator = FairnessMetrics()
        
        if self.fairness_constraint == 'equal_opportunity':
            fairness_metrics = fairness_calculator.compute_equal_opportunity(y_true, y_pred, protected_attr)
        elif self.fairness_constraint == 'equalized_odds':
            fairness_metrics = fairness_calculator.compute_equalized_odds(y_true, y_pred, protected_attr)
        else:
            fairness_metrics = {}
        

        accuracy = np.mean(y_true == y_pred)
        
        self.validation_metrics_ = {
            'fairness_metrics': fairness_metrics,
            'accuracy': accuracy,
            'thresholds': thresholds.copy(),
            'constraint_satisfied': self._check_constraint_satisfaction(fairness_metrics)
        }
    
    def _check_constraint_satisfaction(self, fairness_metrics: Dict[str, Any]) -> bool:

        if self.fairness_constraint == 'equal_opportunity':
            gap = fairness_metrics.get('equal_opportunity_gap', float('inf'))
        elif self.fairness_constraint == 'equalized_odds':
            gap = fairness_metrics.get('equalized_odds_gap', float('inf'))
        else:
            return False
        
        return gap <= self.constraint_tolerance
    
    def apply_thresholds(self, y_proba: np.ndarray, protected_attr: np.ndarray,
                        thresholds: Dict[str, float]) -> np.ndarray:

        if len(y_proba) != len(protected_attr):
            raise ValueError("y_proba and protected_attr must have the same length")
        
        if len(y_proba) == 0:
            return np.array([], dtype=int)
        
  
        y_pred = np.zeros(len(y_proba), dtype=int)
        
 
        unique_groups = np.unique(protected_attr)
        
        for group in unique_groups:
            group_mask = protected_attr == group
            if np.any(group_mask):

                threshold = thresholds.get(str(group), 0.5)
                y_pred[group_mask] = (y_proba[group_mask] >= threshold).astype(int)
        
        return y_pred
    
    def get_optimization_info(self) -> Dict[str, Any]:
    
        if not self.is_fitted:
            raise ValueError("ThresholdOptimizer must be fitted before getting optimization info")
        
        info = {
            'fairness_constraint': self.fairness_constraint,
            'constraint_tolerance': self.constraint_tolerance,
            'optimization_method': self.optimization_method,
            'optimal_thresholds': self.optimal_thresholds_.copy() if self.optimal_thresholds_ else {},
            'groups': self.groups_.tolist() if self.groups_ is not None else [],
            'validation_metrics': self.validation_metrics_.copy() if self.validation_metrics_ else {}
        }
        
        if self.optimization_result_ is not None:
            info['optimization_result'] = {
                'success': self.optimization_result_.success,
                'message': self.optimization_result_.message,
                'fun': float(self.optimization_result_.fun),
                'nit': getattr(self.optimization_result_, 'nit', None),
                'nfev': getattr(self.optimization_result_, 'nfev', None)
            }
        
        return info
    
    def predict_with_thresholds(self, model, X: pd.DataFrame, 
                              protected_attr: pd.Series) -> np.ndarray:

        if not self.is_fitted or self.optimal_thresholds_ is None:
            raise ValueError("ThresholdOptimizer must be fitted and thresholds optimized before prediction")
        

        y_proba = model.predict_proba(X)[:, 1]
 
        protected_attr_np = protected_attr.values if hasattr(protected_attr, 'values') else np.array(protected_attr)
        

        return self.apply_thresholds(y_proba, protected_attr_np, self.optimal_thresholds_)
    
    def evaluate_threshold_effectiveness(self, y_true: np.ndarray, y_proba: np.ndarray,
                                       protected_attr: np.ndarray,
                                       baseline_threshold: float = 0.5) -> Dict[str, Any]:

        if self.optimal_thresholds_ is None:
            raise ValueError("Must optimize thresholds before evaluation")
        
     
        baseline_pred = (y_proba >= baseline_threshold).astype(int)
        

        optimized_pred = self.apply_thresholds(y_proba, protected_attr, self.optimal_thresholds_)
        
     
        from ..fairness.metrics import FairnessMetrics
        fairness_calculator = FairnessMetrics()
        
       
        if self.fairness_constraint == 'equal_opportunity':
            baseline_fairness = fairness_calculator.compute_equal_opportunity(y_true, baseline_pred, protected_attr)
            optimized_fairness = fairness_calculator.compute_equal_opportunity(y_true, optimized_pred, protected_attr)
        elif self.fairness_constraint == 'equalized_odds':
            baseline_fairness = fairness_calculator.compute_equalized_odds(y_true, baseline_pred, protected_attr)
            optimized_fairness = fairness_calculator.compute_equalized_odds(y_true, optimized_pred, protected_attr)
        else:
            baseline_fairness = {}
            optimized_fairness = {}
        
     
        baseline_accuracy = np.mean(y_true == baseline_pred)
        optimized_accuracy = np.mean(y_true == optimized_pred)
        
        
        fairness_improvement = {}
        if self.fairness_constraint == 'equal_opportunity':
            baseline_gap = baseline_fairness.get('equal_opportunity_gap', 0)
            optimized_gap = optimized_fairness.get('equal_opportunity_gap', 0)
            fairness_improvement['equal_opportunity_gap_reduction'] = baseline_gap - optimized_gap
            fairness_improvement['equal_opportunity_relative_improvement'] = (
                (baseline_gap - optimized_gap) / baseline_gap if baseline_gap > 0 else 0
            )
        elif self.fairness_constraint == 'equalized_odds':
            baseline_gap = baseline_fairness.get('equalized_odds_gap', 0)
            optimized_gap = optimized_fairness.get('equalized_odds_gap', 0)
            fairness_improvement['equalized_odds_gap_reduction'] = baseline_gap - optimized_gap
            fairness_improvement['equalized_odds_relative_improvement'] = (
                (baseline_gap - optimized_gap) / baseline_gap if baseline_gap > 0 else 0
            )
        
        accuracy_change = optimized_accuracy - baseline_accuracy
        
        return {
            'baseline_metrics': {
                'fairness': baseline_fairness,
                'accuracy': baseline_accuracy,
                'threshold': baseline_threshold
            },
            'optimized_metrics': {
                'fairness': optimized_fairness,
                'accuracy': optimized_accuracy,
                'thresholds': self.optimal_thresholds_.copy()
            },
            'improvements': {
                'fairness': fairness_improvement,
                'accuracy_change': accuracy_change,
                'accuracy_relative_change': accuracy_change / baseline_accuracy if baseline_accuracy > 0 else 0
            },
            'success_criteria': {
                'constraint_satisfied': self._check_constraint_satisfaction(optimized_fairness),
                'accuracy_preserved': accuracy_change >= -0.02 
            }
        }