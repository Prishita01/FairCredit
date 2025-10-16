from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from ..models.base import BaselineModel


class BiasmitigationTechnique(ABC):    
    def __init__(self, **kwargs):

        self.is_fitted = False
        self.mitigation_params = kwargs
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            protected_attr: pd.Series) -> 'BiasmitigationTechnique':
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                 protected_attr: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     protected_attr: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:

        self.fit(X, y, protected_attr)
        return self.transform(X, y, protected_attr)
    
    def get_mitigation_info(self) -> Dict[str, Any]:
        return {
            'technique_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'parameters': self.mitigation_params
        }


class PreProcessingTechnique(BiasmitigationTechnique):
    
    @abstractmethod
    def compute_weights(self, y: pd.Series, protected_attr: pd.Series) -> np.ndarray:
        pass


class PostProcessingTechnique(BiasmitigationTechnique):
    
    @abstractmethod
    def optimize_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray,
                           protected_attr: np.ndarray) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def apply_thresholds(self, y_proba: np.ndarray, protected_attr: np.ndarray,
                        thresholds: Dict[str, float]) -> np.ndarray:
        pass


class MitigationEvaluator:
    
    def __init__(self):
        pass
    
    def evaluate_fairness_improvement(self, baseline_metrics: Dict[str, float],
                                    mitigated_metrics: Dict[str, float]) -> Dict[str, float]:
        improvements = {}
        
        for metric_name in baseline_metrics:
            if metric_name in mitigated_metrics:
                baseline_val = baseline_metrics[metric_name]
                mitigated_val = mitigated_metrics[metric_name]
                
                # Calculate absolute and relative improvement
                abs_improvement = baseline_val - mitigated_val
                rel_improvement = abs_improvement / baseline_val if baseline_val != 0 else 0
                
                improvements[f"{metric_name}_abs_improvement"] = abs_improvement
                improvements[f"{metric_name}_rel_improvement"] = rel_improvement
        
        return improvements
    
    def evaluate_utility_preservation(self, baseline_performance: Dict[str, float],
                                    mitigated_performance: Dict[str, float]) -> Dict[str, float]:
        preservation = {}
        
        for metric_name in baseline_performance:
            if metric_name in mitigated_performance:
                baseline_val = baseline_performance[metric_name]
                mitigated_val = mitigated_performance[metric_name]
                
                # Calculate absolute and relative change
                abs_change = mitigated_val - baseline_val
                rel_change = abs_change / baseline_val if baseline_val != 0 else 0
                
                preservation[f"{metric_name}_abs_change"] = abs_change
                preservation[f"{metric_name}_rel_change"] = rel_change
        
        return preservation
    
    def check_success_criteria(self, fairness_improvement: Dict[str, float],
                             utility_preservation: Dict[str, float],
                             fairness_threshold: float = 0.5,
                             utility_threshold: float = 0.02) -> Dict[str, bool]:
        results = {}
        
        # Check fairness improvement criteria
        for metric, improvement in fairness_improvement.items():
            if 'rel_improvement' in metric:
                results[f"{metric}_meets_threshold"] = improvement >= fairness_threshold
        
        # Check utility preservation criteria
        for metric, change in utility_preservation.items():
            if 'abs_change' in metric and 'auc' in metric.lower():
                results[f"{metric}_meets_threshold"] = abs(change) <= utility_threshold
        
        # Overall success
        fairness_success = any(v for k, v in results.items() if 'improvement' in k and 'meets_threshold' in k)
        utility_success = any(v for k, v in results.items() if 'preservation' in k and 'meets_threshold' in k)
        
        results['overall_success'] = fairness_success and utility_success
        
        return results