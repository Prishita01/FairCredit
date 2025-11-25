import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings

from .success_criteria import SuccessCriteria, ValidationResult
from ..robustness.stability_tester import StabilityTester


@dataclass
class StabilityResult:
    
    passed: bool
    
    fairness_stability_passed: bool
    utility_stability_passed: bool
    
    max_fairness_degradation: float
    max_utility_degradation: float
    
    fairness_significant_degradation: bool
    utility_significant_degradation: bool
    
    fairness_stability_analysis: Dict[str, Any]
    utility_stability_analysis: Dict[str, Any]
    
    confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    
    hypothesis_tests: Optional[Dict[str, Dict[str, Any]]] = None
    
    failure_reasons: List[str] = None
    
    def __post_init__(self):
        if self.failure_reasons is None:
            self.failure_reasons = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'fairness_stability_passed': self.fairness_stability_passed,
            'utility_stability_passed': self.utility_stability_passed,
            'max_fairness_degradation': self.max_fairness_degradation,
            'max_utility_degradation': self.max_utility_degradation,
            'fairness_significant_degradation': self.fairness_significant_degradation,
            'utility_significant_degradation': self.utility_significant_degradation,
            'fairness_stability_analysis': self.fairness_stability_analysis,
            'utility_stability_analysis': self.utility_stability_analysis,
            'confidence_intervals': self.confidence_intervals,
            'hypothesis_tests': self.hypothesis_tests,
            'failure_reasons': self.failure_reasons
        }
    
    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        
        result_str = f"Stability Validation: {status}\n"
        result_str += f"{'='*40}\n"
        
        fairness_status = "✓" if self.fairness_stability_passed else "✗"
        result_str += f"{fairness_status} Fairness Stability: {self.max_fairness_degradation:.1%} max degradation "
        result_str += f"(Required: ≤25%)\n"
        
        utility_status = "✓" if self.utility_stability_passed else "✗"
        result_str += f"{utility_status} Utility Stability: {self.max_utility_degradation:.1%} max degradation "
        result_str += f"(Required: ≤25%)\n"
        
        if self.fairness_significant_degradation:
            result_str += f"⚠ Statistically significant fairness degradation detected\n"
        if self.utility_significant_degradation:
            result_str += f"⚠ Statistically significant utility degradation detected\n"
        

        if self.failure_reasons:
            result_str += f"\nFailure Reasons:\n"
            for reason in self.failure_reasons:
                result_str += f"  • {reason}\n"
        
        return result_str


class StabilityValidator:
   
    def __init__(self, 
                 criteria: Optional[SuccessCriteria] = None,
                 random_state: Optional[int] = None):
        
        self.criteria = criteria or SuccessCriteria()
        self.stability_tester = StabilityTester(random_state=random_state)
        self.random_state = random_state
    
    def validate_stability(self,
                          original_fairness: Dict[str, float],
                          original_utility: Dict[str, float],
                          perturbed_fairness: List[Dict[str, float]],
                          perturbed_utility: List[Dict[str, float]],
                          compute_confidence_intervals: bool = True,
                          perform_hypothesis_tests: bool = True) -> StabilityResult:
        
        
        self._validate_inputs(original_fairness, original_utility, 
                            perturbed_fairness, perturbed_utility)
        
        
        fairness_stability = self.stability_tester.measure_fairness_stability(
            original_fairness, perturbed_fairness, 
            self.criteria.max_stability_degradation
        )
        
        
        utility_stability = self.stability_tester.measure_utility_stability(
            original_utility, perturbed_utility,
            self.criteria.max_stability_degradation
        )
        
        
        max_fairness_degradation = fairness_stability['worst_case_degradation'].get('degradation', 0.0)
        max_utility_degradation = utility_stability['performance_degradation'].get('max_degradation', 0.0)
        
        
        tolerance = 1e-10
        fairness_stability_passed = max_fairness_degradation <= (self.criteria.max_stability_degradation + tolerance)
        utility_stability_passed = max_utility_degradation <= (self.criteria.max_stability_degradation + tolerance)
        
        
        confidence_intervals = None
        hypothesis_tests = None
        fairness_significant_degradation = False
        utility_significant_degradation = False
        
        
        if compute_confidence_intervals:
            try:
                fairness_ci = self.stability_tester.bootstrap_stability_confidence_intervals(
                    original_fairness, perturbed_fairness
                )
                utility_ci = self.stability_tester.bootstrap_stability_confidence_intervals(
                    original_utility, perturbed_utility
                )
                confidence_intervals = {
                    'fairness': fairness_ci,
                    'utility': utility_ci
                }
            except Exception as e:
                warnings.warn(f"Failed to compute confidence intervals: {e}")
        
        
        if perform_hypothesis_tests:
            try:
                fairness_tests = self.stability_tester.test_stability_hypothesis(
                    original_fairness, perturbed_fairness,
                    self.criteria.max_stability_degradation,
                    self.criteria.significance_level
                )
                utility_tests = self.stability_tester.test_stability_hypothesis(
                    original_utility, perturbed_utility,
                    self.criteria.max_stability_degradation,
                    self.criteria.significance_level
                )
                
                hypothesis_tests = {
                    'fairness': fairness_tests,
                    'utility': utility_tests
                }
                
                # Check for significant degradation
                fairness_significant_degradation = self._check_significant_degradation(fairness_tests)
                utility_significant_degradation = self._check_significant_degradation(utility_tests)
                
            except Exception as e:
                warnings.warn(f"Failed to perform hypothesis tests: {e}")
        
        
        failure_reasons = []
        if not fairness_stability_passed:
            failure_reasons.append(
                f"Fairness degradation {max_fairness_degradation:.1%} > allowed {self.criteria.max_stability_degradation:.1%}"
            )
        if not utility_stability_passed:
            failure_reasons.append(
                f"Utility degradation {max_utility_degradation:.1%} > allowed {self.criteria.max_stability_degradation:.1%}"
            )
        if fairness_significant_degradation:
            failure_reasons.append("Statistically significant fairness degradation detected")
        if utility_significant_degradation:
            failure_reasons.append("Statistically significant utility degradation detected")
        
        
        passed = fairness_stability_passed and utility_stability_passed
        
        return StabilityResult(
            passed=passed,
            fairness_stability_passed=fairness_stability_passed,
            utility_stability_passed=utility_stability_passed,
            max_fairness_degradation=max_fairness_degradation,
            max_utility_degradation=max_utility_degradation,
            fairness_significant_degradation=fairness_significant_degradation,
            utility_significant_degradation=utility_significant_degradation,
            fairness_stability_analysis=fairness_stability,
            utility_stability_analysis=utility_stability,
            confidence_intervals=confidence_intervals,
            hypothesis_tests=hypothesis_tests,
            failure_reasons=failure_reasons
        )
    
    def validate_improvement_significance(self,
                                        baseline_metrics: Dict[str, float],
                                        improved_metrics: Dict[str, float],
                                        bootstrap_samples: Optional[List[Dict[str, float]]] = None,
                                        alpha: float = None) -> Dict[str, Dict[str, Any]]:

        if alpha is None:
            alpha = self.criteria.significance_level
        
        significance_results = {}
        
        for metric_name in baseline_metrics.keys():
            if metric_name not in improved_metrics:
                continue
            
            baseline_value = baseline_metrics[metric_name]
            improved_value = improved_metrics[metric_name]
            
            if baseline_value != 0:
                improvement = (improved_value - baseline_value) / abs(baseline_value)
            else:
                improvement = improved_value - baseline_value
            
            result = {
                'baseline_value': baseline_value,
                'improved_value': improved_value,
                'absolute_improvement': improved_value - baseline_value,
                'relative_improvement': improvement,
                'improvement_significant': False,
                'pvalue': None,
                'confidence_interval': None
            }
            
            if bootstrap_samples:
                bootstrap_values = [sample.get(metric_name) for sample in bootstrap_samples 
                                  if metric_name in sample and sample[metric_name] is not None]
                
                if len(bootstrap_values) >= 3:
                    t_stat, p_value = stats.ttest_1samp(bootstrap_values, baseline_value)
                    result['pvalue'] = p_value
                    result['improvement_significant'] = p_value < alpha
                    
                    ci_lower, ci_upper = np.percentile(
                        bootstrap_values, 
                        [100 * alpha/2, 100 * (1 - alpha/2)]
                    )
                    result['confidence_interval'] = (ci_lower, ci_upper)
            
            significance_results[metric_name] = result
        
        return significance_results
    
    def generate_stability_report(self, stability_result: StabilityResult) -> str:

        report = []
        report.append("Stability Validation Report")
        report.append("=" * 30)
        report.append("")
        
        status = "PASSED ✓" if stability_result.passed else "FAILED ✗"
        report.append(f"Overall Status: {status}")
        report.append("")
        
        report.append("Stability Analysis:")
        report.append("-" * 20)
        
        fairness_status = "✓" if stability_result.fairness_stability_passed else "✗"
        report.append(f"{fairness_status} Fairness Stability:")
        report.append(f"   Required: ≤{self.criteria.max_stability_degradation:.1%} degradation")
        report.append(f"   Achieved: {stability_result.max_fairness_degradation:.1%} max degradation")
        
        fairness_analysis = stability_result.fairness_stability_analysis
        stable_fairness = fairness_analysis.get('stable_metrics_count', 0)
        total_fairness = fairness_analysis.get('total_metrics_count', 0)
        report.append(f"   Stable Metrics: {stable_fairness}/{total_fairness}")
        report.append("")
        
        utility_status = "✓" if stability_result.utility_stability_passed else "✗"
        report.append(f"{utility_status} Utility Stability:")
        report.append(f"   Required: ≤{self.criteria.max_stability_degradation:.1%} degradation")
        report.append(f"   Achieved: {stability_result.max_utility_degradation:.1%} max degradation")
        
        utility_analysis = stability_result.utility_stability_analysis
        stable_utility = utility_analysis.get('stable_metrics_count', 0)
        total_utility = utility_analysis.get('total_metrics_count', 0)
        report.append(f"   Stable Metrics: {stable_utility}/{total_utility}")
        report.append("")
        
        if stability_result.fairness_significant_degradation or stability_result.utility_significant_degradation:
            report.append("Statistical Significance:")
            report.append("-" * 22)
            if stability_result.fairness_significant_degradation:
                report.append("⚠ Statistically significant fairness degradation detected")
            if stability_result.utility_significant_degradation:
                report.append("⚠ Statistically significant utility degradation detected")
            report.append("")
        
        if stability_result.confidence_intervals:
            report.append("Confidence Intervals:")
            report.append("-" * 20)
            
            for category, intervals in stability_result.confidence_intervals.items():
                report.append(f"{category.title()} Metrics:")
                for metric, ci_dict in intervals.items():
                    for ci_type, (lower, upper) in ci_dict.items():
                        report.append(f"   {metric} ({ci_type}): [{lower:.4f}, {upper:.4f}]")
                report.append("")
        
        if stability_result.failure_reasons:
            report.append("Failure Reasons:")
            report.append("-" * 15)
            for reason in stability_result.failure_reasons:
                report.append(f"• {reason}")
            report.append("")
        
        if not stability_result.passed:
            report.append("Recommendations:")
            report.append("-" * 15)
            
            if not stability_result.fairness_stability_passed:
                report.append("• Implement more robust fairness mitigation techniques")
                report.append("• Consider ensemble methods for fairness stability")
                report.append("• Investigate data preprocessing improvements")
            
            if not stability_result.utility_stability_passed:
                report.append("• Apply model regularization techniques")
                report.append("• Consider cross-validation for hyperparameter tuning")
                report.append("• Evaluate ensemble methods for utility stability")
            
            if stability_result.fairness_significant_degradation or stability_result.utility_significant_degradation:
                report.append("• Investigate sources of statistical instability")
                report.append("• Consider larger validation datasets")
                report.append("• Review model architecture and training procedures")
        else:
            report.append("Recommendations:")
            report.append("-" * 15)
            report.append("• Model demonstrates good stability under perturbations")
            report.append("• Ready for deployment consideration")
            report.append("• Continue monitoring in production environment")
        
        return "\n".join(report)
    
    def _validate_inputs(self, 
                        original_fairness: Dict[str, float],
                        original_utility: Dict[str, float],
                        perturbed_fairness: List[Dict[str, float]],
                        perturbed_utility: List[Dict[str, float]]) -> None:

        if not original_fairness:
            raise ValueError("original_fairness cannot be empty")
        if not original_utility:
            raise ValueError("original_utility cannot be empty")
        
       
        if not perturbed_fairness:
            raise ValueError("perturbed_fairness cannot be empty")
        if not perturbed_utility:
            raise ValueError("perturbed_utility cannot be empty")
        
        
        if len(perturbed_fairness) != len(perturbed_utility):
            raise ValueError("perturbed_fairness and perturbed_utility must have same length")
        
       
        for metric_name, value in original_fairness.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                raise ValueError(f"Invalid fairness metric value for {metric_name}: {value}")
        
        for metric_name, value in original_utility.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                raise ValueError(f"Invalid utility metric value for {metric_name}: {value}")
    
    def _check_significant_degradation(self, hypothesis_tests: Dict[str, Dict[str, Any]]) -> bool:

        for metric_name, test_results in hypothesis_tests.items():
            tests = test_results.get('tests', {})
            
           
            threshold_test = tests.get('stability_threshold', {})
            if threshold_test.get('exceeds_threshold', False):
                return True
            
         
            distribution_test = tests.get('distribution_change', {})
            if distribution_test.get('significant', False):
                return True
        
        return False