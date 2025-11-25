import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from .base import PreProcessingTechnique

if TYPE_CHECKING:
    from ..models.base import BaselineModel


class ReweighingMitigator(PreProcessingTechnique):

    def __init__(self, **kwargs):
        """Initialize the reweighing mitigator."""
        super().__init__(**kwargs)
        self.weights_ = None
        self.group_weights_ = None
        self.original_distribution_ = None
        self.reweighed_distribution_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            protected_attr: pd.Series) -> 'ReweighingMitigator':
 
       
        self.weights_ = self.compute_weights(y, protected_attr)
        
     
        self._compute_distributions(y, protected_attr, self.weights_)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                 protected_attr: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

        if not self.is_fitted:
            raise ValueError("ReweighingMitigator must be fitted before transform")
        
        return X, y
    
    def compute_weights(self, y: pd.Series, protected_attr: pd.Series) -> np.ndarray:

        if len(y) != len(protected_attr):
            raise ValueError("y and protected_attr must have the same length")
        
        if len(y) == 0:
            raise ValueError("Cannot compute weights for empty dataset")
        
        y_vals = y.values if hasattr(y, 'values') else np.array(y)
        attr_vals = protected_attr.values if hasattr(protected_attr, 'values') else np.array(protected_attr)
        
      
        unique_attrs = np.unique(attr_vals)
        unique_labels = np.unique(y_vals)
        
       
        n_total = len(y_vals)
        
 
        n_attr = {}  
        n_label = {}  
        n_joint = {}  
        
        for attr in unique_attrs:
            n_attr[attr] = np.sum(attr_vals == attr)
        
        for label in unique_labels:
            n_label[label] = np.sum(y_vals == label)
        
        for attr in unique_attrs:
            for label in unique_labels:
                n_joint[(attr, label)] = np.sum((attr_vals == attr) & (y_vals == label))
        

        weights = np.zeros(len(y_vals))
        
        for i in range(len(y_vals)):
            attr_val = attr_vals[i]
            label_val = y_vals[i]
            
   
            n_a = n_attr[attr_val]
            n_y = n_label[label_val]
            n_ay = n_joint[(attr_val, label_val)]
            
            if n_ay == 0:
                
                weights[i] = 0
            else:
                # Apply reweighing formula: w(a,y) = (n_a * n_y) / (n_total * n_ay)
                # This is equivalent to P(A=a) * P(Y=y) / P(A=a, Y=y)
                weights[i] = (n_a * n_y) / (n_total * n_ay)
        
    
        self.group_weights_ = {}
        for attr in unique_attrs:
            for label in unique_labels:
                n_a = n_attr[attr]
                n_y = n_label[label]
                n_ay = n_joint[(attr, label)]
                
                if n_ay > 0:
                    weight = (n_a * n_y) / (n_total * n_ay)
                    self.group_weights_[(attr, label)] = weight
        
        return weights
    
    def _compute_distributions(self, y: pd.Series, protected_attr: pd.Series, weights: np.ndarray):
  
        y_vals = y.values if hasattr(y, 'values') else np.array(y)
        attr_vals = protected_attr.values if hasattr(protected_attr, 'values') else np.array(protected_attr)
        
        unique_attrs = np.unique(attr_vals)
        unique_labels = np.unique(y_vals)
        
   
        n_total = len(y_vals)
        self.original_distribution_ = {}
        
        for attr in unique_attrs:
            for label in unique_labels:
                count = np.sum((attr_vals == attr) & (y_vals == label))
                self.original_distribution_[(attr, label)] = count / n_total
      
        weighted_total = np.sum(weights)
        self.reweighed_distribution_ = {}
        
        for attr in unique_attrs:
            for label in unique_labels:
                mask = (attr_vals == attr) & (y_vals == label)
                weighted_count = np.sum(weights[mask])
                self.reweighed_distribution_[(attr, label)] = weighted_count / weighted_total
    
    def validate_weights(self, y: pd.Series, protected_attr: pd.Series, 
                        weights: np.ndarray, tolerance: float = 1e-6) -> Dict[str, bool]:
 
        validation_results = {}
        

        validation_results['weights_non_negative'] = np.all(weights >= 0)
        
      
        actual_sum = np.sum(weights)
        validation_results['weights_sum_positive'] = actual_sum > 0
        

        y_vals = y.values if hasattr(y, 'values') else np.array(y)
        attr_vals = protected_attr.values if hasattr(protected_attr, 'values') else np.array(protected_attr)
        
        unique_attrs = np.unique(attr_vals)
        unique_labels = np.unique(y_vals)
        

        weighted_total = np.sum(weights)
        
        if weighted_total == 0:
            validation_results['statistical_independence'] = False
            validation_results['independence_violations'] = [('all_groups', 'zero_weights', float('inf'))]
            return validation_results
        
   
        p_attr_weighted = {}
        for attr in unique_attrs:
            mask = attr_vals == attr
            p_attr_weighted[attr] = np.sum(weights[mask]) / weighted_total
        
        p_label_weighted = {}
        for label in unique_labels:
            mask = y_vals == label
            p_label_weighted[label] = np.sum(weights[mask]) / weighted_total
        
  
        p_joint_weighted = {}
        for attr in unique_attrs:
            for label in unique_labels:
                mask = (attr_vals == attr) & (y_vals == label)
                p_joint_weighted[(attr, label)] = np.sum(weights[mask]) / weighted_total
        
   
        independence_violations = []
        
        independence_tolerance = max(tolerance, 1e-3)
        
        for attr in unique_attrs:
            for label in unique_labels:
                joint_prob = p_joint_weighted[(attr, label)]
                expected_prob = p_attr_weighted[attr] * p_label_weighted[label]
                
                
                if joint_prob > 0 and expected_prob > 0:
                    absolute_error = abs(joint_prob - expected_prob)
                    relative_error = absolute_error / expected_prob
                    if relative_error > independence_tolerance:
                        independence_violations.append((attr, label, relative_error))
        
        validation_results['statistical_independence'] = len(independence_violations) == 0
        validation_results['independence_violations'] = independence_violations
        
       
        group_weights_positive = True
        for attr in unique_attrs:
            for label in unique_labels:
                mask = (attr_vals == attr) & (y_vals == label)
                if np.any(mask):  
                    group_weight = np.sum(weights[mask])
                    if group_weight <= 0:
                        group_weights_positive = False
                        break
            if not group_weights_positive:
                break
        
        validation_results['group_weights_positive'] = group_weights_positive
        
        return validation_results
    
    def get_weight_statistics(self) -> Dict[str, Any]:

        if not self.is_fitted or self.weights_ is None:
            raise ValueError("ReweighingMitigator must be fitted before getting statistics")
        
        stats = {
            'n_instances': len(self.weights_),
            'weight_sum': np.sum(self.weights_),
            'weight_mean': np.mean(self.weights_),
            'weight_std': np.std(self.weights_),
            'weight_min': np.min(self.weights_),
            'weight_max': np.max(self.weights_),
            'weight_median': np.median(self.weights_),
            'group_weights': self.group_weights_.copy() if self.group_weights_ else {},
            'original_distribution': self.original_distribution_.copy() if self.original_distribution_ else {},
            'reweighed_distribution': self.reweighed_distribution_.copy() if self.reweighed_distribution_ else {}
        }
        
        return stats
    
    def fit_reweighed_model(self, model: 'BaselineModel', X: pd.DataFrame, 
                           y: pd.Series, protected_attr: pd.Series) -> 'BaselineModel':

        if not self.is_fitted:
            
            self.fit(X, y, protected_attr)
        
    
        fitted_model = model.fit(X, y, sample_weight=self.weights_)
        
        return fitted_model
    
    def compute_correlation_reduction(self, y: pd.Series, protected_attr: pd.Series, 
                                    weights: Optional[np.ndarray] = None) -> Dict[str, float]:
 
        if weights is None:
            if not self.is_fitted or self.weights_ is None:
                raise ValueError("Must provide weights or fit the mitigator first")
            weights = self.weights_
        
        
        y_vals = y.values if hasattr(y, 'values') else np.array(y)
        attr_vals = protected_attr.values if hasattr(protected_attr, 'values') else np.array(protected_attr)
        
        
        unique_attrs = np.unique(attr_vals)
        attr_numeric = np.zeros(len(attr_vals))
        for i, attr in enumerate(unique_attrs):
            attr_numeric[attr_vals == attr] = i
        
       
        original_corr = np.corrcoef(attr_numeric, y_vals)[0, 1]
        
        
        # Use weighted covariance formula: Cov_w(X,Y) = Σw_i(x_i - μ_x)(y_i - μ_y) / Σw_i
        weighted_total = np.sum(weights)
        
        
        weighted_mean_attr = np.sum(weights * attr_numeric) / weighted_total
        weighted_mean_y = np.sum(weights * y_vals) / weighted_total
        
        
        weighted_cov = np.sum(weights * (attr_numeric - weighted_mean_attr) * (y_vals - weighted_mean_y)) / weighted_total
        
        
        weighted_var_attr = np.sum(weights * (attr_numeric - weighted_mean_attr)**2) / weighted_total
        weighted_var_y = np.sum(weights * (y_vals - weighted_mean_y)**2) / weighted_total
        
        weighted_std_attr = np.sqrt(weighted_var_attr)
        weighted_std_y = np.sqrt(weighted_var_y)
        
        
        if weighted_std_attr > 0 and weighted_std_y > 0:
            weighted_corr = weighted_cov / (weighted_std_attr * weighted_std_y)
        else:
            weighted_corr = 0.0
        
       
        abs_original = abs(original_corr)
        abs_weighted = abs(weighted_corr)
        
        if abs_original > 0:
            reduction_percentage = (abs_original - abs_weighted) / abs_original * 100
        else:
            reduction_percentage = 0.0
        
        return {
            'original_correlation': float(original_corr),
            'weighted_correlation': float(weighted_corr),
            'absolute_reduction': float(abs_original - abs_weighted),
            'percentage_reduction': float(reduction_percentage)
        }
    
    def evaluate_fairness_improvement(self, baseline_model: 'BaselineModel', 
                                    reweighed_model: 'BaselineModel',
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    protected_attr_test: pd.Series) -> Dict[str, Any]:

        from ..fairness.metrics import FairnessMetrics
        
        
        baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
        reweighed_proba = reweighed_model.predict_proba(X_test)[:, 1]
        
        
        baseline_pred = (baseline_proba >= 0.5).astype(int)
        reweighed_pred = (reweighed_proba >= 0.5).astype(int)
        
        
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        protected_attr_np = protected_attr_test.values if hasattr(protected_attr_test, 'values') else np.array(protected_attr_test)
        
        
        fairness_calculator = FairnessMetrics()
        
        
        baseline_eo = fairness_calculator.compute_equal_opportunity(y_test_np, baseline_pred, protected_attr_np)
        baseline_eqodds = fairness_calculator.compute_equalized_odds(y_test_np, baseline_pred, protected_attr_np)
        baseline_dp = fairness_calculator.compute_demographic_parity(baseline_pred, protected_attr_np)
        
        
        reweighed_eo = fairness_calculator.compute_equal_opportunity(y_test_np, reweighed_pred, protected_attr_np)
        reweighed_eqodds = fairness_calculator.compute_equalized_odds(y_test_np, reweighed_pred, protected_attr_np)
        reweighed_dp = fairness_calculator.compute_demographic_parity(reweighed_pred, protected_attr_np)
        
        
        eo_improvement = self._compute_metric_improvement(
            baseline_eo['equal_opportunity_gap'], 
            reweighed_eo['equal_opportunity_gap']
        )
        
        eqodds_improvement = self._compute_metric_improvement(
            baseline_eqodds['equalized_odds_gap'],
            reweighed_eqodds['equalized_odds_gap']
        )
        
        dp_improvement = self._compute_metric_improvement(
            baseline_dp['demographic_parity_gap'],
            reweighed_dp['demographic_parity_gap']
        )
        
        return {
            'baseline_metrics': {
                'equal_opportunity': baseline_eo,
                'equalized_odds': baseline_eqodds,
                'demographic_parity': baseline_dp
            },
            'reweighed_metrics': {
                'equal_opportunity': reweighed_eo,
                'equalized_odds': reweighed_eqodds,
                'demographic_parity': reweighed_dp
            },
            'improvements': {
                'equal_opportunity': eo_improvement,
                'equalized_odds': eqodds_improvement,
                'demographic_parity': dp_improvement
            }
        }
    
    def evaluate_utility_preservation(self, baseline_model: 'BaselineModel',
                                    reweighed_model: 'BaselineModel',
                                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:

        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, log_loss
        
        
        baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
        reweighed_proba = reweighed_model.predict_proba(X_test)[:, 1]
        
        baseline_pred = (baseline_proba >= 0.5).astype(int)
        reweighed_pred = (reweighed_proba >= 0.5).astype(int)
        
        
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        
        
        baseline_metrics = {
            'auc': roc_auc_score(y_test_np, baseline_proba),
            'auprc': average_precision_score(y_test_np, baseline_proba),
            'f1': f1_score(y_test_np, baseline_pred),
            'brier_score': brier_score_loss(y_test_np, baseline_proba),
            'log_loss': log_loss(y_test_np, baseline_proba)
        }
        
        
        reweighed_metrics = {
            'auc': roc_auc_score(y_test_np, reweighed_proba),
            'auprc': average_precision_score(y_test_np, reweighed_proba),
            'f1': f1_score(y_test_np, reweighed_pred),
            'brier_score': brier_score_loss(y_test_np, reweighed_proba),
            'log_loss': log_loss(y_test_np, reweighed_proba)
        }
        
        
        utility_changes = {}
        for metric in baseline_metrics:
            baseline_val = baseline_metrics[metric]
            reweighed_val = reweighed_metrics[metric]
            
            absolute_change = reweighed_val - baseline_val
            relative_change = absolute_change / baseline_val if baseline_val != 0 else 0
            
            utility_changes[f'{metric}_baseline'] = baseline_val
            utility_changes[f'{metric}_reweighed'] = reweighed_val
            utility_changes[f'{metric}_absolute_change'] = absolute_change
            utility_changes[f'{metric}_relative_change'] = relative_change
        
        
        auc_drop = baseline_metrics['auc'] - reweighed_metrics['auc']
        utility_changes['auc_drop'] = auc_drop
        utility_changes['meets_auc_criteria'] = auc_drop <= 0.02
        
        return utility_changes
    
    def _compute_metric_improvement(self, baseline_gap: float, reweighed_gap: float) -> Dict[str, float]:

        absolute_improvement = baseline_gap - reweighed_gap
        
        if baseline_gap > 0:
            relative_improvement = absolute_improvement / baseline_gap
            percentage_improvement = relative_improvement * 100
        else:
            relative_improvement = 0.0
            percentage_improvement = 0.0
        
        return {
            'baseline_gap': baseline_gap,
            'reweighed_gap': reweighed_gap,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'percentage_improvement': percentage_improvement,
            'meets_50_percent_criteria': percentage_improvement >= 50.0
        }
    
    def comprehensive_evaluation(self, baseline_model: 'BaselineModel',
                               reweighed_model: 'BaselineModel',
                               X_test: pd.DataFrame, y_test: pd.Series,
                               protected_attr_test: pd.Series) -> Dict[str, Any]:

        fairness_eval = self.evaluate_fairness_improvement(
            baseline_model, reweighed_model, X_test, y_test, protected_attr_test
        )
        

        utility_eval = self.evaluate_utility_preservation(
            baseline_model, reweighed_model, X_test, y_test
        )
        
    
        eo_success = fairness_eval['improvements']['equal_opportunity']['meets_50_percent_criteria']
        auc_success = utility_eval['meets_auc_criteria']
        overall_success = eo_success and auc_success
        
        return {
            'fairness_evaluation': fairness_eval,
            'utility_evaluation': utility_eval,
            'success_criteria': {
                'equal_opportunity_50_percent': eo_success,
                'auc_preservation_2_points': auc_success,
                'overall_success': overall_success
            },
            'summary': {
                'eo_improvement_percentage': fairness_eval['improvements']['equal_opportunity']['percentage_improvement'],
                'auc_drop': utility_eval['auc_drop'],
                'meets_criteria': overall_success
            }
        }