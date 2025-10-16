from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
import pandas as pd


class FairnessAuditor(ABC):
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.bootstrap_results = {}
    
    @abstractmethod
    def compute_equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 protected_attr: np.ndarray) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def compute_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray,
                              protected_attr: np.ndarray) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def compute_demographic_parity(self, y_pred: np.ndarray,
                                  protected_attr: np.ndarray) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def bootstrap_confidence_intervals(self, metric_func: Callable,
                                     n_bootstrap: int = 1000,
                                     **kwargs) -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def intersectional_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                               protected_attrs: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        pass
    
    def validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray,
                       protected_attr: np.ndarray) -> None:
        if not (len(y_true) == len(y_pred) == len(protected_attr)):
            raise ValueError("All input arrays must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only binary values (0, 1)")
        
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred must contain only binary values (0, 1)")
        
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Input arrays cannot contain NaN values")
        
        unique_groups = np.unique(protected_attr)
        if len(unique_groups) < 2:
            raise ValueError("Protected attribute must have at least 2 groups")
    
    def check_group_sizes(self, y_true: np.ndarray, protected_attr: np.ndarray,
                         min_group_size: int = 10) -> Dict[str, int]:
        group_sizes = {}
        
        for group in np.unique(protected_attr):
            group_mask = protected_attr == group
            group_size = np.sum(group_mask)
            group_sizes[str(group)] = group_size
            
            if group_size < min_group_size:
                print(f"Warning: Group '{group}' has only {group_size} samples "
                      f"(minimum recommended: {min_group_size})")
            
            group_positives = np.sum(y_true[group_mask] == 1)
            group_negatives = np.sum(y_true[group_mask] == 0)
            
            if group_positives == 0:
                print(f"Warning: Group '{group}' has no positive samples")
            if group_negatives == 0:
                print(f"Warning: Group '{group}' has no negative samples")
        
        return group_sizes
    
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           protected_attr: np.ndarray) -> Dict[str, Dict[str, float]]:
        self.validate_inputs(y_true, y_pred, protected_attr)
        self.check_group_sizes(y_true, protected_attr)
        
        results = {
            'equal_opportunity': self.compute_equal_opportunity(y_true, y_pred, protected_attr),
            'equalized_odds': self.compute_equalized_odds(y_true, y_pred, protected_attr),
            'demographic_parity': self.compute_demographic_parity(y_pred, protected_attr)
        }
        
        return results
    
    def get_fairness_summary(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        summary = {}
        
        for metric_name, metric_values in metrics.items():
            if isinstance(metric_values, dict):
                # Extract gap/difference values
                gaps = [v for k, v in metric_values.items() if 'gap' in k.lower() or 'diff' in k.lower()]
                if gaps:
                    summary[f"{metric_name}_max_gap"] = max(gaps)
                    summary[f"{metric_name}_mean_gap"] = np.mean(gaps)
        
        return summary