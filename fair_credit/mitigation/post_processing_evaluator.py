import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from .threshold_application import ThresholdApplicationSystem
from .reweighing import ReweighingMitigator


class PostProcessingEffectivenessEvaluator:
    def __init__(self, fairness_metrics: List[str] = None,
                 utility_metrics: List[str] = None):
        self.fairness_metrics = fairness_metrics or [
            'equal_opportunity', 'equalized_odds', 'demographic_parity'
        ]
        self.utility_metrics = utility_metrics or [
            'auc', 'auprc', 'f1_score', 'accuracy', 'brier_score', 'log_loss'
        ]
        
        # Storing evaluation results
        self.evaluation_results_ = {}
        self.comparison_results_ = {}
    
    def evaluate_fairness_improvement(self, baseline_model, mitigated_model,
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    protected_attr_test: pd.Series,
                                    mitigation_method: str = 'post_processing') -> Dict[str, Any]
        from ..fairness.metrics import FairnessMetrics
        
        # Predictions from both models
        if hasattr(mitigated_model, 'apply_thresholds') and callable(getattr(mitigated_model, 'apply_thresholds', None)):
            # Post-processing 
            baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
            baseline_pred = (baseline_proba >= 0.5).astype(int)
            mitigated_pred = mitigated_model.apply_thresholds(baseline_model, X_test, protected_attr_test)
        else:
            # Pre-processing Regular model
            baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
            baseline_pred = (baseline_proba >= 0.5).astype(int)
            mitigated_proba = mitigated_model.predict_proba(X_test)[:, 1]
            mitigated_pred = (mitigated_proba >= 0.5).astype(int)
        
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        protected_attr_np = protected_attr_test.values if hasattr(protected_attr_test, 'values') else np.array(protected_attr_test)
        
        #calculating fairness metrics
        fairness_calculator = FairnessMetrics()
        
        baseline_fairness = {}
        mitigated_fairness = {}
        
        for metric in self.fairness_metrics:
            if metric == 'equal_opportunity':
                baseline_fairness[metric] = fairness_calculator.compute_equal_opportunity(
                    y_test_np, baseline_pred, protected_attr_np
                )
                mitigated_fairness[metric] = fairness_calculator.compute_equal_opportunity(
                    y_test_np, mitigated_pred, protected_attr_np
                )
            elif metric == 'equalized_odds':
                baseline_fairness[metric] = fairness_calculator.compute_equalized_odds(
                    y_test_np, baseline_pred, protected_attr_np
                )
                mitigated_fairness[metric] = fairness_calculator.compute_equalized_odds(
                    y_test_np, mitigated_pred, protected_attr_np
                )
            elif metric == 'demographic_parity':
                baseline_fairness[metric] = fairness_calculator.compute_demographic_parity(
                    baseline_pred, protected_attr_np
                )
                mitigated_fairness[metric] = fairness_calculator.compute_demographic_parity(
                    mitigated_pred, protected_attr_np
                )
        
        improvements = {}
        for metric in self.fairness_metrics:
            if metric in baseline_fairness and metric in mitigated_fairness:
                gap_key = f'{metric}_gap'
                baseline_gap = baseline_fairness[metric].get(gap_key, 0)
                mitigated_gap = mitigated_fairness[metric].get(gap_key, 0)
                
                absolute_improvement = baseline_gap - mitigated_gap
                relative_improvement = (absolute_improvement / baseline_gap) if baseline_gap > 0 else 0
                
                improvements[metric] = {
                    'baseline_gap': baseline_gap,
                    'mitigated_gap': mitigated_gap,
                    'absolute_improvement': absolute_improvement,
                    'relative_improvement': relative_improvement,
                    'percentage_improvement': relative_improvement * 100,
                    'meets_50_percent_criteria': relative_improvement >= 0.5
                }
        
        return {
            'baseline_fairness': baseline_fairness,
            'mitigated_fairness': mitigated_fairness,
            'improvements': improvements,
            'mitigation_method': mitigation_method
        }
    
    def evaluate_utility_preservation(self, baseline_model, mitigated_model,
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    protected_attr_test: Optional[pd.Series] = None) -> Dict[str, Any]:

        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score, 
            accuracy_score, brier_score_loss, log_loss
        )
        
        baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
        baseline_pred = (baseline_proba >= 0.5).astype(int)
        
        if hasattr(mitigated_model, 'apply_thresholds') and callable(getattr(mitigated_model, 'apply_thresholds', None)):
            # Post-processing
            if protected_attr_test is None:
                raise ValueError("protected_attr_test required for post-processing evaluation")
            mitigated_pred = mitigated_model.apply_thresholds(baseline_model, X_test, protected_attr_test)
            mitigated_proba = baseline_proba  # Same probabilities but different thresholds
        else:
            # preprocessing regular model 
            mitigated_proba = mitigated_model.predict_proba(X_test)[:, 1]
            mitigated_pred = (mitigated_proba >= 0.5).astype(int)
        
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        
        # Computing utility metrics
        baseline_metrics = {}
        mitigated_metrics = {}
        
        # Probability-based metrics (Assuming different probabilities)
        if not np.array_equal(baseline_proba, mitigated_proba):
            if 'auc' in self.utility_metrics:
                baseline_metrics['auc'] = roc_auc_score(y_test_np, baseline_proba)
                mitigated_metrics['auc'] = roc_auc_score(y_test_np, mitigated_proba)
            
            if 'auprc' in self.utility_metrics:
                baseline_metrics['auprc'] = average_precision_score(y_test_np, baseline_proba)
                mitigated_metrics['auprc'] = average_precision_score(y_test_np, mitigated_proba)
            
            if 'brier_score' in self.utility_metrics:
                baseline_metrics['brier_score'] = brier_score_loss(y_test_np, baseline_proba)
                mitigated_metrics['brier_score'] = brier_score_loss(y_test_np, mitigated_proba)
            
            if 'log_loss' in self.utility_metrics:
                baseline_metrics['log_loss'] = log_loss(y_test_np, baseline_proba)
                mitigated_metrics['log_loss'] = log_loss(y_test_np, mitigated_proba)
        else:
            # For post-processing, probabilities are the same
            if 'auc' in self.utility_metrics:
                auc_score = roc_auc_score(y_test_np, baseline_proba)
                baseline_metrics['auc'] = auc_score
                mitigated_metrics['auc'] = auc_score
            
            if 'auprc' in self.utility_metrics:
                auprc_score = average_precision_score(y_test_np, baseline_proba)
                baseline_metrics['auprc'] = auprc_score
                mitigated_metrics['auprc'] = auprc_score
            
            if 'brier_score' in self.utility_metrics:
                brier_score = brier_score_loss(y_test_np, baseline_proba)
                baseline_metrics['brier_score'] = brier_score
                mitigated_metrics['brier_score'] = brier_score
            
            if 'log_loss' in self.utility_metrics:
                logloss_score = log_loss(y_test_np, baseline_proba)
                baseline_metrics['log_loss'] = logloss_score
                mitigated_metrics['log_loss'] = logloss_score
        
        if 'f1_score' in self.utility_metrics:
            baseline_metrics['f1_score'] = f1_score(y_test_np, baseline_pred)
            mitigated_metrics['f1_score'] = f1_score(y_test_np, mitigated_pred)
        
        if 'accuracy' in self.utility_metrics:
            baseline_metrics['accuracy'] = accuracy_score(y_test_np, baseline_pred)
            mitigated_metrics['accuracy'] = accuracy_score(y_test_np, mitigated_pred)
        
        utility_changes = {}
        for metric in baseline_metrics:
            baseline_val = baseline_metrics[metric]
            mitigated_val = mitigated_metrics[metric]
            
            absolute_change = mitigated_val - baseline_val
            relative_change = (absolute_change / baseline_val) if baseline_val != 0 else 0
            
            utility_changes[metric] = {
                'baseline_value': baseline_val,
                'mitigated_value': mitigated_val,
                'absolute_change': absolute_change,
                'relative_change': relative_change,
                'percentage_change': relative_change * 100
            }
        
        success_criteria = {}
        if 'auc' in utility_changes:
            auc_drop = -utility_changes['auc']['absolute_change']  # Negative change means drop
            success_criteria['auc_preserved'] = auc_drop <= 0.02  # ≤2 point drop
            success_criteria['auc_drop'] = auc_drop
        
        if 'accuracy' in utility_changes:
            accuracy_drop = -utility_changes['accuracy']['absolute_change']
            success_criteria['accuracy_preserved'] = accuracy_drop <= 0.02
            success_criteria['accuracy_drop'] = accuracy_drop
        
        return {
            'baseline_metrics': baseline_metrics,
            'mitigated_metrics': mitigated_metrics,
            'utility_changes': utility_changes,
            'success_criteria': success_criteria
        }
    
    def compare_mitigation_approaches(self, baseline_model, 
                                    preprocessing_model, postprocessing_system,
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    protected_attr_test: pd.Series) -> Dict[str, Any]:
        preprocessing_fairness = self.evaluate_fairness_improvement(
            baseline_model, preprocessing_model, X_test, y_test, protected_attr_test,
            mitigation_method='pre_processing'
        )
        
        preprocessing_utility = self.evaluate_utility_preservation(
            baseline_model, preprocessing_model, X_test, y_test
        )
        
        postprocessing_fairness = self.evaluate_fairness_improvement(
            baseline_model, postprocessing_system, X_test, y_test, protected_attr_test,
            mitigation_method='post_processing'
        )
        
        postprocessing_utility = self.evaluate_utility_preservation(
            baseline_model, postprocessing_system, X_test, y_test, protected_attr_test
        )
        
        comparison = {}
        
        fairness_comparison = {}
        for metric in self.fairness_metrics:
            if (metric in preprocessing_fairness['improvements'] and 
                metric in postprocessing_fairness['improvements']):
                
                pre_improvement = preprocessing_fairness['improvements'][metric]['percentage_improvement']
                post_improvement = postprocessing_fairness['improvements'][metric]['percentage_improvement']
                
                fairness_comparison[metric] = {
                    'preprocessing_improvement': pre_improvement,
                    'postprocessing_improvement': post_improvement,
                    'difference': post_improvement - pre_improvement,
                    'better_approach': 'post_processing' if post_improvement > pre_improvement else 'pre_processing',
                    'both_meet_criteria': (pre_improvement >= 50 and post_improvement >= 50)
                }
        
        utility_comparison = {}
        for metric in self.utility_metrics:
            if (metric in preprocessing_utility['utility_changes'] and 
                metric in postprocessing_utility['utility_changes']):
                
                pre_change = preprocessing_utility['utility_changes'][metric]['percentage_change']
                post_change = postprocessing_utility['utility_changes'][metric]['percentage_change']
                
                # For utility metrics, positive change is generally better (except for loss metrics)
                loss_metrics = ['brier_score', 'log_loss']
                if metric in loss_metrics:
                    better_approach = 'post_processing' if post_change < pre_change else 'pre_processing'
                else:
                    better_approach = 'post_processing' if post_change > pre_change else 'pre_processing'
                
                utility_comparison[metric] = {
                    'preprocessing_change': pre_change,
                    'postprocessing_change': post_change,
                    'difference': post_change - pre_change,
                    'better_approach': better_approach
                }
        
        overall_comparison = {
            'fairness_winner': self._determine_fairness_winner(fairness_comparison),
            'utility_winner': self._determine_utility_winner(utility_comparison),
            'recommended_approach': None
        }
        
        fairness_winner = overall_comparison['fairness_winner']
        utility_winner = overall_comparison['utility_winner']
        
        if fairness_winner == utility_winner:
            overall_comparison['recommended_approach'] = fairness_winner
        else:
            # If different winners, prefer the approach that meets fairness criteria while preserving utility
            pre_meets_fairness = any(
                comp.get('both_meet_criteria', False) or comp.get('preprocessing_improvement', 0) >= 50
                for comp in fairness_comparison.values()
            )
            post_meets_fairness = any(
                comp.get('both_meet_criteria', False) or comp.get('postprocessing_improvement', 0) >= 50
                for comp in fairness_comparison.values()
            )
            
            if post_meets_fairness and not pre_meets_fairness:
                overall_comparison['recommended_approach'] = 'post_processing'
            elif pre_meets_fairness and not post_meets_fairness:
                overall_comparison['recommended_approach'] = 'pre_processing'
            else:
                # Both or neither meet fairness criteria, go with utility winner
                overall_comparison['recommended_approach'] = utility_winner
        
        comparison = {
            'preprocessing_results': {
                'fairness': preprocessing_fairness,
                'utility': preprocessing_utility
            },
            'postprocessing_results': {
                'fairness': postprocessing_fairness,
                'utility': postprocessing_utility
            },
            'fairness_comparison': fairness_comparison,
            'utility_comparison': utility_comparison,
            'overall_comparison': overall_comparison
        }
        
        self.comparison_results_ = comparison
        
        return comparison
    
    def _determine_fairness_winner(self, fairness_comparison: Dict[str, Any]) -> str:
        # Determining which approach performs better on fairness metrics.
        post_wins = 0
        pre_wins = 0
        
        for metric_comp in fairness_comparison.values():
            if metric_comp['better_approach'] == 'post_processing':
                post_wins += 1
            else:
                pre_wins += 1
        
        return 'post_processing' if post_wins > pre_wins else 'pre_processing'
    
    def _determine_utility_winner(self, utility_comparison: Dict[str, Any]) -> str:
        # Determining which approach performs better on utility metrics.
        post_wins = 0
        pre_wins = 0
        
        for metric_comp in utility_comparison.values():
            if metric_comp['better_approach'] == 'post_processing':
                post_wins += 1
            else:
                pre_wins += 1
        
        return 'post_processing' if post_wins > pre_wins else 'pre_processing'
    
    def generate_effectiveness_report(self, comparison_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if comparison_results is None:
            comparison_results = self.comparison_results_
        
        if not comparison_results:
            raise ValueError("No comparison results available. Run compare_mitigation_approaches first.")
        
        fairness_comparison = comparison_results['fairness_comparison']
        utility_comparison = comparison_results['utility_comparison']
        overall_comparison = comparison_results['overall_comparison']
        
        fairness_summary = {}
        for metric, comp in fairness_comparison.items():
            fairness_summary[metric] = {
                'best_improvement': max(comp['preprocessing_improvement'], comp['postprocessing_improvement']),
                'best_approach': comp['better_approach'],
                'meets_criteria': comp.get('both_meet_criteria', False) or max(comp['preprocessing_improvement'], comp['postprocessing_improvement']) >= 50
            }
        
        utility_summary = {}
        for metric, comp in utility_comparison.items():
            utility_summary[metric] = {
                'preprocessing_impact': comp['preprocessing_change'],
                'postprocessing_impact': comp['postprocessing_change'],
                'better_approach': comp['better_approach']
            }
        
        recommendations = []
        
        recommended_approach = overall_comparison['recommended_approach']
        if recommended_approach:
            recommendations.append(f"Recommended approach: {recommended_approach.replace('_', '-')}")
        
        fairness_met = any(summary['meets_criteria'] for summary in fairness_summary.values())
        if fairness_met:
            recommendations.append("Fairness improvement criteria (≥50%) achieved")
        else:
            recommendations.append("Warning: Fairness improvement criteria not fully met")
        
        utility_preserved = True
        for metric in ['auc', 'accuracy']:
            if metric in utility_comparison:
                for approach in ['preprocessing_change', 'postprocessing_change']:
                    if utility_comparison[metric][approach] < -2:  # More than 2% drop
                        utility_preserved = False
                        break
        
        if utility_preserved:
            recommendations.append("Utility preservation criteria met")
        else:
            recommendations.append("Warning: Significant utility degradation detected")
        
        report = {
            'executive_summary': {
                'recommended_approach': recommended_approach,
                'fairness_criteria_met': fairness_met,
                'utility_preserved': utility_preserved,
                'overall_success': fairness_met and utility_preserved
            },
            'fairness_analysis': {
                'summary': fairness_summary,
                'winner': overall_comparison['fairness_winner']
            },
            'utility_analysis': {
                'summary': utility_summary,
                'winner': overall_comparison['utility_winner']
            },
            'recommendations': recommendations,
            'detailed_results': comparison_results
        }
        
        return report
    
    def create_comparison_visualizations(self, comparison_results: Optional[Dict[str, Any]] = None,
                                       save_path: Optional[str] = None) -> Dict[str, Any]:
        if comparison_results is None:
            comparison_results = self.comparison_results_
        
        if not comparison_results:
            raise ValueError("No comparison results available. Run compare_mitigation_approaches first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Post-Processing vs Pre-Processing Mitigation Comparison', fontsize=16)
        
        ax1 = axes[0, 0]
        fairness_data = []
        metrics = []
        approaches = []
        improvements = []
        
        for metric, comp in comparison_results['fairness_comparison'].items():
            metrics.extend([metric, metric])
            approaches.extend(['Pre-processing', 'Post-processing'])
            improvements.extend([comp['preprocessing_improvement'], comp['postprocessing_improvement']])
        
        if metrics:
            fairness_df = pd.DataFrame({
                'Metric': metrics,
                'Approach': approaches,
                'Improvement (%)': improvements
            })
            
            sns.barplot(data=fairness_df, x='Metric', y='Improvement (%)', hue='Approach', ax=ax1)
            ax1.set_title('Fairness Improvement Comparison')
            ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% Target')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
        
        ax2 = axes[0, 1]
        utility_data = []
        metrics = []
        approaches = []
        changes = []
        
        for metric, comp in comparison_results['utility_comparison'].items():
            metrics.extend([metric, metric])
            approaches.extend(['Pre-processing', 'Post-processing'])
            changes.extend([comp['preprocessing_change'], comp['postprocessing_change']])
        
        if metrics:
            utility_df = pd.DataFrame({
                'Metric': metrics,
                'Approach': approaches,
                'Change (%)': changes
            })
            
            sns.barplot(data=utility_df, x='Metric', y='Change (%)', hue='Approach', ax=ax2)
            ax2.set_title('Utility Preservation Comparison')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.7, label='-2% Threshold')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
        
        ax3 = axes[1, 0]
        if 'equal_opportunity' in comparison_results['fairness_comparison']:
            eo_comp = comparison_results['fairness_comparison']['equal_opportunity']
            
            pre_fairness = eo_comp['preprocessing_improvement']
            post_fairness = eo_comp['postprocessing_improvement']
            
            if 'auc' in comparison_results['utility_comparison']:
                auc_comp = comparison_results['utility_comparison']['auc']
                pre_utility = auc_comp['preprocessing_change']
                post_utility = auc_comp['postprocessing_change']
                
                ax3.scatter(pre_fairness, pre_utility, s=100, label='Pre-processing', alpha=0.7)
                ax3.scatter(post_fairness, post_utility, s=100, label='Post-processing', alpha=0.7)
                ax3.set_xlabel('Fairness Improvement (%)')
                ax3.set_ylabel('Utility Change (%)')
                ax3.set_title('Fairness-Utility Trade-off')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% Fairness Target')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        
        criteria = ['Fairness\nImprovement', 'Utility\nPreservation', 'Overall\nSuccess']
        pre_success = []
        post_success = []
        # Fairness success (≥50% improvement)
        pre_fairness_success = any(
            comp['preprocessing_improvement'] >= 50 
            for comp in comparison_results['fairness_comparison'].values()
        )
        post_fairness_success = any(
            comp['postprocessing_improvement'] >= 50 
            for comp in comparison_results['fairness_comparison'].values()
        )
        
        pre_success.append(1 if pre_fairness_success else 0)
        post_success.append(1 if post_fairness_success else 0)
        
        # Utility success (≤2% degradation in key metrics)
        pre_utility_success = True
        post_utility_success = True
        
        for metric in ['auc', 'accuracy']:
            if metric in comparison_results['utility_comparison']:
                comp = comparison_results['utility_comparison'][metric]
                if comp['preprocessing_change'] < -2:
                    pre_utility_success = False
                if comp['postprocessing_change'] < -2:
                    post_utility_success = False
        
        pre_success.append(1 if pre_utility_success else 0)
        post_success.append(1 if post_utility_success else 0)

        # Overall success (both fairness and utility criteria met)
        pre_success.append(1 if (pre_fairness_success and pre_utility_success) else 0)
        post_success.append(1 if (post_fairness_success and post_utility_success) else 0)
        
        x = np.arange(len(criteria))
        width = 0.35
        
        ax4.bar(x - width/2, pre_success, width, label='Pre-processing', alpha=0.7)
        ax4.bar(x + width/2, post_success, width, label='Post-processing', alpha=0.7)
        ax4.set_xlabel('Success Criteria')
        ax4.set_ylabel('Success (1=Yes, 0=No)')
        ax4.set_title('Success Criteria Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(criteria)
        ax4.legend()
        ax4.set_ylim(0, 1.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        visualization_info = {
            'figure': fig,
            'axes': axes,
            'fairness_data': fairness_df if 'fairness_df' in locals() else None,
            'utility_data': utility_df if 'utility_df' in locals() else None,
            'success_summary': {
                'pre_processing': {
                    'fairness_success': pre_fairness_success,
                    'utility_success': pre_utility_success,
                    'overall_success': pre_fairness_success and pre_utility_success
                },
                'post_processing': {
                    'fairness_success': post_fairness_success,
                    'utility_success': post_utility_success,
                    'overall_success': post_fairness_success and post_utility_success
                }
            }
        }
        
        return visualization_info