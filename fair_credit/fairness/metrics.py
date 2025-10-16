import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from sklearn.metrics import confusion_matrix
from .auditor import FairnessAuditor


class FairnessMetrics(FairnessAuditor):
    def __init__(self, confidence_level: float = 0.95, n_jobs: int = -1):
        super().__init__(confidence_level)
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
    
    def compute_equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 protected_attr: np.ndarray) -> Dict[str, float]:
        self.validate_inputs(y_true, y_pred, protected_attr)
        
        results = {}
        tpr_by_group = {}
        
        for group in np.unique(protected_attr):
            group_mask = protected_attr == group
 
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            actual_positives = np.sum(group_y_true == 1)
            
            if actual_positives == 0:
                tpr = 0.0
                print(f"Warning: No positive samples in group '{group}' for Equal Opportunity")
            else:
                true_positives = np.sum((group_y_true == 1) & (group_y_pred == 1))
                tpr = true_positives / actual_positives
            
            tpr_by_group[str(group)] = tpr
            results[f"tpr_group_{group}"] = tpr
        
        groups = list(tpr_by_group.keys())
        max_gap = 0.0
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                gap = abs(tpr_by_group[groups[i]] - tpr_by_group[groups[j]])
                results[f"eo_gap_{groups[i]}_{groups[j]}"] = gap
                max_gap = max(max_gap, gap)
        
        results["equal_opportunity_gap"] = max_gap
        
        return results
    
    def compute_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray,
                              protected_attr: np.ndarray) -> Dict[str, float]:
        self.validate_inputs(y_true, y_pred, protected_attr)
        
        results = {}
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in np.unique(protected_attr):
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_by_group[str(group)] = tpr
            fpr_by_group[str(group)] = fpr
            
            results[f"tpr_group_{group}"] = tpr
            results[f"fpr_group_{group}"] = fpr
        
        groups = list(tpr_by_group.keys())
        max_tpr_gap = 0.0
        max_fpr_gap = 0.0
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                tpr_gap = abs(tpr_by_group[groups[i]] - tpr_by_group[groups[j]])
                fpr_gap = abs(fpr_by_group[groups[i]] - fpr_by_group[groups[j]])
                
                results[f"tpr_gap_{groups[i]}_{groups[j]}"] = tpr_gap
                results[f"fpr_gap_{groups[i]}_{groups[j]}"] = fpr_gap
                
                max_tpr_gap = max(max_tpr_gap, tpr_gap)
                max_fpr_gap = max(max_fpr_gap, fpr_gap)
        
        results["equalized_odds_tpr_gap"] = max_tpr_gap
        results["equalized_odds_fpr_gap"] = max_fpr_gap
        results["equalized_odds_gap"] = max(max_tpr_gap, max_fpr_gap)
        
        return results
    
    def compute_demographic_parity(self, y_pred: np.ndarray,
                                  protected_attr: np.ndarray) -> Dict[str, float]:
        if len(y_pred) != len(protected_attr):
            raise ValueError("y_pred and protected_attr must have the same length")
        
        if len(y_pred) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred must contain only binary values (0, 1)")
        
        results = {}
        pos_rate_by_group = {}
        
        for group in np.unique(protected_attr):
            group_mask = protected_attr == group
            group_y_pred = y_pred[group_mask]
            
            pos_rate = np.mean(group_y_pred)
            pos_rate_by_group[str(group)] = pos_rate
            results[f"pos_rate_group_{group}"] = pos_rate
        
        groups = list(pos_rate_by_group.keys())
        max_gap = 0.0
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                gap = abs(pos_rate_by_group[groups[i]] - pos_rate_by_group[groups[j]])
                results[f"dp_gap_{groups[i]}_{groups[j]}"] = gap
                max_gap = max(max_gap, gap)
        
        results["demographic_parity_gap"] = max_gap
        
        return results
    
    def bootstrap_confidence_intervals(self, metric_func: Callable,
                                     n_bootstrap: int = 1000,
                                     **kwargs) -> Tuple[float, float]:
        y_true = kwargs.get('y_true')
        y_pred = kwargs.get('y_pred') 
        protected_attr = kwargs.get('protected_attr')
        metric_key = kwargs.get('metric_key', 'gap') 
        
        if y_true is None or y_pred is None or protected_attr is None:
            raise ValueError("Must provide y_true, y_pred, and protected_attr for bootstrap")
        
        n_samples = len(y_true)

        bootstrap_metrics = []

        bootstrap_metrics = []
        seeds = np.random.randint(0, 2**31, n_bootstrap)
        
        for seed in seeds:
            np.random.seed(seed)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            boot_y_true = y_true[indices]
            boot_y_pred = y_pred[indices]
            boot_protected = protected_attr[indices]
            
            try:
                result = metric_func(boot_y_true, boot_y_pred, boot_protected)
                if isinstance(result, dict):
                    for key, value in result.items():
                        if metric_key in key.lower():
                            bootstrap_metrics.append(value)
                            break
                    else:
                        for key, value in result.items():
                            if 'gap' in key.lower():
                                bootstrap_metrics.append(value)
                                break
                else:
                    bootstrap_metrics.append(result)
            except Exception:
                continue
        
        bootstrap_metrics = [m for m in bootstrap_metrics if not np.isnan(m)]
        
        if len(bootstrap_metrics) == 0:
            raise ValueError("All bootstrap samples failed")
        
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
        upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
        
        return lower_bound, upper_bound
    
    def intersectional_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                               protected_attrs: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        if len(protected_attrs) != 2:
            raise ValueError("Intersectional analysis currently supports exactly 2 protected attributes")
        
        sex_attr = protected_attrs[0]
        age_attr = protected_attrs[1]
        
        self.validate_inputs(y_true, y_pred, sex_attr)
        
        if len(age_attr) != len(y_true):
            raise ValueError("All protected attribute arrays must have same length as y_true")
        
        intersectional_groups = []
        group_labels = []
        
        for sex in np.unique(sex_attr):
            for age in np.unique(age_attr):
                mask = (sex_attr == sex) & (age_attr == age)
                if np.sum(mask) > 0:  # Only include non-empty groups
                    intersectional_groups.append(mask)
                    group_labels.append(f"{sex}_{age}")
        
        if len(intersectional_groups) < 2:
            raise ValueError("Need at least 2 intersectional groups for analysis")
        
        intersectional_attr = np.full(len(y_true), "", dtype=object)
        for i, (mask, label) in enumerate(zip(intersectional_groups, group_labels)):
            intersectional_attr[mask] = label
        
        results = {
            'equal_opportunity': self.compute_equal_opportunity(y_true, y_pred, intersectional_attr),
            'equalized_odds': self.compute_equalized_odds(y_true, y_pred, intersectional_attr),
            'demographic_parity': self.compute_demographic_parity(y_pred, intersectional_attr)
        }
        
        results['group_sizes'] = {}
        for label, mask in zip(group_labels, intersectional_groups):
            results['group_sizes'][label] = np.sum(mask)
        
        return results


class BootstrapCI:
    
    def __init__(self, confidence_level: float = 0.95, n_jobs: int = -1):
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
    
    def compute_ci(self, data: np.ndarray, statistic_func: Callable,
                   n_bootstrap: int = 1000) -> Tuple[float, float]:
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
            
        n_samples = len(data)
        bootstrap_stats = []
        
        seeds = np.random.randint(0, 2**31, n_bootstrap)
        
        for seed in seeds:
            np.random.seed(seed)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_data = data[indices]
            try:
                stat = statistic_func(boot_data)
                if not np.isnan(stat):
                    bootstrap_stats.append(stat)
            except Exception:
                continue
        
        if len(bootstrap_stats) == 0:
            raise ValueError("All bootstrap samples failed")
        
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
        return lower_bound, upper_bound