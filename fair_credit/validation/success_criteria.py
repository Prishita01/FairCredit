"""Automated success criteria validation for FairCredit system."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from ..fairness.metrics import FairnessMetrics
from ..models.metrics import ModelMetrics, ModelEvaluator


@dataclass
class SuccessCriteria:

    

    min_fairness_improvement: float = 0.5 
    

    max_auc_drop: float = 0.02  
    

    max_stability_degradation: float = 0.25 
    

    significance_level: float = 0.05


@dataclass
class ValidationResult:

    
 
    passed: bool
    
   
    fairness_improvement_passed: bool
    utility_preservation_passed: bool
    
    
    fairness_improvement_ratio: float
    auc_drop: float
    

    baseline_eo_gap: float
    mitigated_eo_gap: float
    baseline_auc: float
    mitigated_auc: float
    

    group_results: Dict[str, Dict[str, float]]
    

    failure_reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'fairness_improvement_passed': self.fairness_improvement_passed,
            'utility_preservation_passed': self.utility_preservation_passed,
            'fairness_improvement_ratio': self.fairness_improvement_ratio,
            'auc_drop': self.auc_drop,
            'baseline_eo_gap': self.baseline_eo_gap,
            'mitigated_eo_gap': self.mitigated_eo_gap,
            'baseline_auc': self.baseline_auc,
            'mitigated_auc': self.mitigated_auc,
            'group_results': self.group_results,
            'failure_reasons': self.failure_reasons
        }
    
    def __str__(self) -> str:
       
        status = "PASSED" if self.passed else "FAILED"
        
        result_str = f"Success Criteria Validation: {status}\n"
        result_str += f"{'='*50}\n"
        
        fairness_status = "✓" if self.fairness_improvement_passed else "✗"
        result_str += f"{fairness_status} Fairness Improvement: {self.fairness_improvement_ratio:.1%} "
        result_str += f"(Required: ≥50%)\n"
        result_str += f"  Baseline EO Gap: {self.baseline_eo_gap:.4f}\n"
        result_str += f"  Mitigated EO Gap: {self.mitigated_eo_gap:.4f}\n"
        
        utility_status = "✓" if self.utility_preservation_passed else "✗"
        result_str += f"{utility_status} Utility Preservation: {self.auc_drop:.4f} AUC drop "
        result_str += f"(Required: ≤0.02)\n"
        result_str += f"  Baseline AUC: {self.baseline_auc:.4f}\n"
        result_str += f"  Mitigated AUC: {self.mitigated_auc:.4f}\n"
        
        if self.group_results:
            result_str += f"\nGroup-wise Results:\n"
            for group, metrics in self.group_results.items():
                result_str += f"  {group}: EO Gap = {metrics.get('eo_gap', 0):.4f}, "
                result_str += f"Improvement = {metrics.get('improvement', 0):.1%}\n"
        
        if self.failure_reasons:
            result_str += f"\nFailure Reasons:\n"
            for reason in self.failure_reasons:
                result_str += f"  • {reason}\n"
        
        return result_str


class SuccessCriteriaValidator:
    
    def __init__(self, criteria: Optional[SuccessCriteria] = None):
 
        self.criteria = criteria or SuccessCriteria()
        self.fairness_metrics = FairnessMetrics()
        self.model_evaluator = ModelEvaluator()
    
    def validate_pipeline_success(self, 
                                 baseline_models: Dict[str, Any],
                                 datasets: Dict[str, Any],
                                 pipeline_results: Dict[str, Any]) -> Dict[str, Any]:

        validation_results = {
            'equal_opportunity_improvement': False,
            'utility_preservation': False,
            'stability_maintained': False,
            'overall_success': False,
            'details': {}
        }
        
        try:
            test_df = datasets.get('test')
            if test_df is None:
                validation_results['details']['error'] = "Test dataset not available"
                return validation_results
            

            feature_cols = [col for col in test_df.columns 
                           if col not in ['default', 'sex', 'age_group']]
            X_test = test_df[feature_cols]
            y_test = test_df['default'].values
            
            for attr in ['sex', 'age_group']:
                if attr not in test_df.columns:
                    continue
                
                protected_attr = test_df[attr].values
                attr_results = self._validate_attribute_success(
                    baseline_models, X_test, y_test, protected_attr, 
                    pipeline_results, attr
                )
                validation_results['details'][attr] = attr_results
                
                if not validation_results['equal_opportunity_improvement']:
                    validation_results['equal_opportunity_improvement'] = attr_results.get('fairness_improvement', False)
                if not validation_results['utility_preservation']:
                    validation_results['utility_preservation'] = attr_results.get('utility_preservation', False)
                if not validation_results['stability_maintained']:
                    validation_results['stability_maintained'] = attr_results.get('stability_maintained', False)
            
            validation_results['overall_success'] = (
                validation_results['equal_opportunity_improvement'] and
                validation_results['utility_preservation'] and
                validation_results['stability_maintained']
            )
            
        except Exception as e:
            validation_results['details']['validation_error'] = str(e)
        
        return validation_results
    
    def _validate_attribute_success(self,
                                   baseline_models: Dict[str, Any],
                                   X_test: Any,
                                   y_test: np.ndarray,
                                   protected_attr: np.ndarray,
                                   pipeline_results: Dict[str, Any],
                                   attr_name: str) -> Dict[str, Any]:
        """Validate success criteria for a specific protected attribute."""
        attr_results = {
            'fairness_improvement': False,
            'utility_preservation': False,
            'stability_maintained': False,
            'baseline_metrics': {},
            'mitigated_metrics': {}
        }
        
        try:
         
            baseline_model_name = None
            mitigated_model_name = None
            
            for model_name in baseline_models.keys():
                if not ('reweighed' in model_name or 'threshold' in model_name):
                    baseline_model_name = model_name
                elif 'reweighed' in model_name and baseline_model_name and baseline_model_name in model_name:
                    mitigated_model_name = model_name
                    break
            
            if not baseline_model_name or not mitigated_model_name:
                threshold_key = f"{baseline_model_name}_{attr_name}_thresholds"
                if threshold_key in pipeline_results:
                    return self._validate_threshold_optimization(
                        baseline_models[baseline_model_name], X_test, y_test, 
                        protected_attr, pipeline_results[threshold_key], attr_name
                    )
                else:
                    attr_results['error'] = f"No mitigation found for {attr_name}"
                    return attr_results
            
            baseline_model = baseline_models[baseline_model_name]
            baseline_y_pred = baseline_model.predict(X_test)
            baseline_y_proba = baseline_model.predict_proba(X_test)[:, 1]
            
            mitigated_model = baseline_models[mitigated_model_name]
            mitigated_y_pred = mitigated_model.predict(X_test)
            mitigated_y_proba = mitigated_model.predict_proba(X_test)[:, 1]
            
            baseline_fairness = self.fairness_metrics.compute_equal_opportunity(
                y_test, baseline_y_pred, protected_attr
            )
            mitigated_fairness = self.fairness_metrics.compute_equal_opportunity(
                y_test, mitigated_y_pred, protected_attr
            )
            
            baseline_metrics = self.model_evaluator.compute_metrics(y_test, baseline_y_proba)
            mitigated_metrics = self.model_evaluator.compute_metrics(y_test, mitigated_y_proba)
            
            
            attr_results['baseline_metrics'] = {
                'eo_gap': baseline_fairness['equal_opportunity_gap'],
                'auc': baseline_metrics.auc
            }
            attr_results['mitigated_metrics'] = {
                'eo_gap': mitigated_fairness['equal_opportunity_gap'],
                'auc': mitigated_metrics.auc
            }
            
            
            baseline_gap = baseline_fairness['equal_opportunity_gap']
            mitigated_gap = mitigated_fairness['equal_opportunity_gap']
            
            if baseline_gap > 0:
                improvement_ratio = (baseline_gap - mitigated_gap) / baseline_gap
                attr_results['fairness_improvement'] = improvement_ratio >= self.criteria.min_fairness_improvement
                attr_results['improvement_ratio'] = improvement_ratio
            else:
                
                attr_results['fairness_improvement'] = True
                attr_results['improvement_ratio'] = 1.0
            
            
            auc_drop = baseline_metrics.auc - mitigated_metrics.auc
            attr_results['utility_preservation'] = auc_drop <= self.criteria.max_auc_drop
            attr_results['auc_drop'] = auc_drop
            
            
            attr_results['stability_maintained'] = (
                attr_results['fairness_improvement'] and 
                attr_results['utility_preservation']
            )
            
        except Exception as e:
            attr_results['error'] = str(e)
        
        return attr_results
    
    def _validate_threshold_optimization(self,
                                       baseline_model: Any,
                                       X_test: Any,
                                       y_test: np.ndarray,
                                       protected_attr: np.ndarray,
                                       thresholds: Dict[str, float],
                                       attr_name: str) -> Dict[str, Any]:
        """Validate success criteria for threshold optimization."""
        attr_results = {
            'fairness_improvement': False,
            'utility_preservation': False,
            'stability_maintained': False,
            'baseline_metrics': {},
            'mitigated_metrics': {}
        }
        
        try:
           
            baseline_y_proba = baseline_model.predict_proba(X_test)[:, 1]
            baseline_y_pred = (baseline_y_proba >= 0.5).astype(int)
            
            
            mitigated_y_pred = np.zeros_like(baseline_y_pred)
            for group_value, threshold in thresholds.items():
                group_mask = protected_attr == group_value
                mitigated_y_pred[group_mask] = (baseline_y_proba[group_mask] >= threshold).astype(int)
            
            
            baseline_fairness = self.fairness_metrics.compute_equal_opportunity(
                y_test, baseline_y_pred, protected_attr
            )
            mitigated_fairness = self.fairness_metrics.compute_equal_opportunity(
                y_test, mitigated_y_pred, protected_attr
            )
            
            
            baseline_metrics = self.model_evaluator.compute_metrics(y_test, baseline_y_proba)
            mitigated_metrics = self.model_evaluator.compute_metrics(y_test, baseline_y_proba)  
            
           
            attr_results['baseline_metrics'] = {
                'eo_gap': baseline_fairness['equal_opportunity_gap'],
                'auc': baseline_metrics.auc
            }
            attr_results['mitigated_metrics'] = {
                'eo_gap': mitigated_fairness['equal_opportunity_gap'],
                'auc': mitigated_metrics.auc
            }
            
           
            baseline_gap = baseline_fairness['equal_opportunity_gap']
            mitigated_gap = mitigated_fairness['equal_opportunity_gap']
            
            if baseline_gap > 0:
                improvement_ratio = (baseline_gap - mitigated_gap) / baseline_gap
                attr_results['fairness_improvement'] = improvement_ratio >= self.criteria.min_fairness_improvement
                attr_results['improvement_ratio'] = improvement_ratio
            else:
                attr_results['fairness_improvement'] = True
                attr_results['improvement_ratio'] = 1.0
            
            
            auc_drop = baseline_metrics.auc - mitigated_metrics.auc
            attr_results['utility_preservation'] = abs(auc_drop) <= self.criteria.max_auc_drop
            attr_results['auc_drop'] = auc_drop
            
            
            attr_results['stability_maintained'] = (
                attr_results['fairness_improvement'] and 
                attr_results['utility_preservation']
            )
            
        except Exception as e:
            attr_results['error'] = str(e)
        
        return attr_results


class SuccessCriteriaChecker:
    
    
    def __init__(self, criteria: Optional[SuccessCriteria] = None):
        
        self.criteria = criteria or SuccessCriteria()
        self.fairness_metrics = FairnessMetrics()
        self.model_evaluator = ModelEvaluator()
    
    def validate_success_criteria(self,
                                baseline_results: Dict[str, Any],
                                mitigated_results: Dict[str, Any],
                                protected_attr: np.ndarray) -> ValidationResult:
     
   
        self._validate_inputs(baseline_results, mitigated_results, protected_attr)
        
        
        y_true = baseline_results['y_true']
        baseline_y_pred = baseline_results['y_pred']
        baseline_y_proba = baseline_results['y_proba']
        mitigated_y_pred = mitigated_results['y_pred']
        mitigated_y_proba = mitigated_results['y_proba']
        
       
        unique_groups = np.unique(protected_attr)
        if len(unique_groups) < 2:
            
            baseline_fairness = {'equal_opportunity_gap': 0.0}
            mitigated_fairness = {'equal_opportunity_gap': 0.0}
        else:
           
            baseline_fairness = self.fairness_metrics.compute_equal_opportunity(
                y_true, baseline_y_pred, protected_attr
            )
            mitigated_fairness = self.fairness_metrics.compute_equal_opportunity(
                y_true, mitigated_y_pred, protected_attr
            )
        
 
        baseline_metrics = self.model_evaluator.compute_metrics(y_true, baseline_y_proba)
        mitigated_metrics = self.model_evaluator.compute_metrics(y_true, mitigated_y_proba)
        
        baseline_eo_gap = baseline_fairness['equal_opportunity_gap']
        mitigated_eo_gap = mitigated_fairness['equal_opportunity_gap']
        baseline_auc = baseline_metrics.auc
        mitigated_auc = mitigated_metrics.auc
        
        fairness_improvement_ratio = self._compute_fairness_improvement(
            baseline_eo_gap, mitigated_eo_gap
        )
        fairness_improvement_passed = fairness_improvement_ratio >= self.criteria.min_fairness_improvement
        
        auc_drop = baseline_auc - mitigated_auc
        utility_preservation_passed = auc_drop <= self.criteria.max_auc_drop
        
        group_results = self._compute_group_wise_results(
            baseline_fairness, mitigated_fairness, protected_attr
        )
        
        failure_reasons = []
        if not fairness_improvement_passed:
            failure_reasons.append(
                f"Fairness improvement {fairness_improvement_ratio:.1%} < required {self.criteria.min_fairness_improvement:.1%}"
            )
        if not utility_preservation_passed:
            failure_reasons.append(
                f"AUC drop {auc_drop:.4f} > allowed {self.criteria.max_auc_drop:.4f}"
            )
        
        passed = fairness_improvement_passed and utility_preservation_passed
        
        return ValidationResult(
            passed=passed,
            fairness_improvement_passed=fairness_improvement_passed,
            utility_preservation_passed=utility_preservation_passed,
            fairness_improvement_ratio=fairness_improvement_ratio,
            auc_drop=auc_drop,
            baseline_eo_gap=baseline_eo_gap,
            mitigated_eo_gap=mitigated_eo_gap,
            baseline_auc=baseline_auc,
            mitigated_auc=mitigated_auc,
            group_results=group_results,
            failure_reasons=failure_reasons
        )
    
    def validate_all_groups(self,
                           baseline_results: Dict[str, Any],
                           mitigated_results: Dict[str, Any],
                           protected_attr: np.ndarray) -> Dict[str, ValidationResult]:
        results = {}
        
        unique_groups = np.unique(protected_attr)
        
        if len(unique_groups) < 2:
            results['all'] = self.validate_success_criteria(
                baseline_results, mitigated_results, protected_attr
            )
            return results
        
        for group in unique_groups:
            group_mask = protected_attr == group
            
            group_baseline = {
                'y_true': baseline_results['y_true'][group_mask],
                'y_pred': baseline_results['y_pred'][group_mask],
                'y_proba': baseline_results['y_proba'][group_mask]
            }
            group_mitigated = {
                'y_true': mitigated_results['y_true'][group_mask],
                'y_pred': mitigated_results['y_pred'][group_mask],
                'y_proba': mitigated_results['y_proba'][group_mask]
            }
            group_protected = protected_attr[group_mask]
            
            if len(group_protected) > 0:
                results[str(group)] = self.validate_success_criteria(
                    group_baseline, group_mitigated, group_protected
                )
        
        return results
    
    def _validate_inputs(self, baseline_results: Dict[str, Any], 
                        mitigated_results: Dict[str, Any],
                        protected_attr: np.ndarray) -> None:
        required_keys = ['y_true', 'y_pred', 'y_proba']
        
        for key in required_keys:
            if key not in baseline_results:
                raise ValueError(f"Missing key '{key}' in baseline_results")
        
        for key in required_keys:
            if key not in mitigated_results:
                raise ValueError(f"Missing key '{key}' in mitigated_results")
        
        baseline_length = len(baseline_results['y_true'])
        mitigated_length = len(mitigated_results['y_true'])
        protected_length = len(protected_attr)
        
        if not (baseline_length == mitigated_length == protected_length):
            raise ValueError("All input arrays must have the same length")
        
        for results_dict, name in [(baseline_results, 'baseline'), (mitigated_results, 'mitigated')]:
            y_true = np.asarray(results_dict['y_true'])
            y_pred = np.asarray(results_dict['y_pred'])
            y_proba = np.asarray(results_dict['y_proba'])
            
            if not np.all(np.isin(y_true, [0, 1])):
                raise ValueError(f"{name} y_true must contain only 0 and 1")
            
            if not np.all(np.isin(y_pred, [0, 1])):
                raise ValueError(f"{name} y_pred must contain only 0 and 1")
            
            if not np.all((y_proba >= 0) & (y_proba <= 1)):
                raise ValueError(f"{name} y_proba must be between 0 and 1")
    
    def _compute_fairness_improvement(self, baseline_gap: float, mitigated_gap: float) -> float:
 
        if baseline_gap == 0:
            return 1.0 if mitigated_gap == 0 else 0.0
        
        improvement = (baseline_gap - mitigated_gap) / baseline_gap
        return max(0.0, improvement)  
    
    def _compute_group_wise_results(self, baseline_fairness: Dict[str, float],
                                   mitigated_fairness: Dict[str, float],
                                   protected_attr: np.ndarray) -> Dict[str, Dict[str, float]]:

        results = {}
        unique_groups = np.unique(protected_attr)
        
        for i, group_a in enumerate(unique_groups):
            for j, group_b in enumerate(unique_groups):
                if i < j: 
                    gap_key = f"eo_gap_{group_a}_{group_b}"
                    
                    baseline_gap = baseline_fairness.get(gap_key, 0.0)
                    mitigated_gap = mitigated_fairness.get(gap_key, 0.0)
                    
                    improvement = self._compute_fairness_improvement(baseline_gap, mitigated_gap)
                    
                    for group in [group_a, group_b]:
                        if str(group) not in results:
                            results[str(group)] = {}
                        
                        results[str(group)][f'eo_gap_vs_{group_b if group == group_a else group_a}'] = mitigated_gap
                        results[str(group)][f'improvement_vs_{group_b if group == group_a else group_a}'] = improvement
        
        return results
    
    def generate_validation_report(self, validation_result: ValidationResult) -> str:

        report = []
        report.append("FairCredit Success Criteria Validation Report")
        report.append("=" * 50)
        report.append("")
        

        status = "PASSED " if validation_result.passed else "FAILED "
        report.append(f"Overall Status: {status}")
        report.append("")
        

        report.append("Detailed Results:")
        report.append("-" * 20)
        
        fairness_status = "Correct" if validation_result.fairness_improvement_passed else "Wrong"
        report.append(f"{fairness_status} Fairness Improvement Criterion:")
        report.append(f"   Required: ≥{self.criteria.min_fairness_improvement:.1%} reduction in Equal Opportunity gap")
        report.append(f"   Achieved: {validation_result.fairness_improvement_ratio:.1%}")
        report.append(f"   Baseline EO Gap: {validation_result.baseline_eo_gap:.4f}")
        report.append(f"   Mitigated EO Gap: {validation_result.mitigated_eo_gap:.4f}")
        report.append("")
        
        utility_status = "✓" if validation_result.utility_preservation_passed else "✗"
        report.append(f"{utility_status} Utility Preservation Criterion:")
        report.append(f"   Required: ≤{self.criteria.max_auc_drop:.3f} AUC drop")
        report.append(f"   Achieved: {validation_result.auc_drop:.4f} AUC drop")
        report.append(f"   Baseline AUC: {validation_result.baseline_auc:.4f}")
        report.append(f"   Mitigated AUC: {validation_result.mitigated_auc:.4f}")
        report.append("")
        
        if validation_result.group_results:
            report.append("Group-wise Analysis:")
            report.append("-" * 20)
            for group, metrics in validation_result.group_results.items():
                report.append(f"Group {group}:")
                for metric_name, value in metrics.items():
                    if 'improvement' in metric_name:
                        report.append(f"   {metric_name}: {value:.1%}")
                    else:
                        report.append(f"   {metric_name}: {value:.4f}")
                report.append("")
        
        if validation_result.failure_reasons:
            report.append("Failure Reasons:")
            report.append("-" * 15)
            for reason in validation_result.failure_reasons:
                report.append(f"• {reason}")
            report.append("")
        
        if not validation_result.passed:
            report.append("Recommendations:")
            report.append("-" * 15)
            if not validation_result.fairness_improvement_passed:
                report.append("• Consider stronger bias mitigation techniques")
                report.append("• Explore different fairness constraints or optimization objectives")
                report.append("• Investigate intersectional fairness issues")
            if not validation_result.utility_preservation_passed:
                report.append("• Review model architecture and hyperparameters")
                report.append("• Consider ensemble methods to maintain performance")
                report.append("• Evaluate trade-offs between fairness and utility")
        
        return "\n".join(report)