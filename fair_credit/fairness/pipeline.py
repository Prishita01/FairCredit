import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from .metrics import FairnessMetrics, BootstrapCI
from .intersectional import IntersectionalAnalyzer


class FairnessAuditPipeline:
    
    def __init__(self, confidence_level: float = 0.95, n_jobs: int = -1):
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs
        
        self.fairness_metrics = FairnessMetrics(confidence_level=confidence_level, n_jobs=n_jobs)
        self.bootstrap_ci = BootstrapCI(confidence_level=confidence_level, n_jobs=n_jobs)
        self.intersectional_analyzer = IntersectionalAnalyzer(confidence_level=confidence_level)
        
        self.audit_results = {}
    
    def run_comprehensive_audit(self, y_true: np.ndarray, y_pred: np.ndarray,
                               protected_attrs: Dict[str, np.ndarray],
                               n_bootstrap: int = 1000) -> Dict[str, Any]:
        self._validate_audit_inputs(y_true, y_pred, protected_attrs)
        
        audit_results = {
            'metadata': {
                'n_samples': len(y_true),
                'n_positive': np.sum(y_true == 1),
                'n_negative': np.sum(y_true == 0),
                'positive_rate': np.mean(y_true),
                'prediction_positive_rate': np.mean(y_pred),
                'confidence_level': self.confidence_level,
                'n_bootstrap': n_bootstrap
            }
        }
        
        audit_results['single_attribute_analysis'] = {}
        
        for attr_name, attr_values in protected_attrs.items():
            print(f"Analyzing fairness for protected attribute: {attr_name}")
            
            single_attr_results = self.fairness_metrics.compute_all_metrics(
                y_true, y_pred, attr_values
            )
            
            single_attr_results['confidence_intervals'] = self._compute_bootstrap_intervals(
                y_true, y_pred, attr_values, n_bootstrap
            )
            
            single_attr_results['group_analysis'] = self._identify_disadvantaged_groups(
                single_attr_results, attr_values
            )
            
            audit_results['single_attribute_analysis'][attr_name] = single_attr_results
        
        if len(protected_attrs) >= 2:
            print("Performing intersectional fairness analysis...")
            
            if 'sex' in protected_attrs and 'age' in protected_attrs:
                intersectional_results = self.intersectional_analyzer.analyze_intersectional_fairness(
                    y_true, y_pred, protected_attrs['sex'], protected_attrs['age']
                )
                audit_results['intersectional_analysis'] = intersectional_results
            else:
                attr_names = list(protected_attrs.keys())[:2]
                intersectional_results = self.intersectional_analyzer.analyze_intersectional_fairness(
                    y_true, y_pred, protected_attrs[attr_names[0]], protected_attrs[attr_names[1]]
                )
                audit_results['intersectional_analysis'] = intersectional_results
        
        audit_results['overall_summary'] = self._generate_overall_summary(audit_results)
        
        audit_results['recommendations'] = self._generate_recommendations(audit_results)
        
        self.audit_results = audit_results
        
        return audit_results
    
    def _validate_audit_inputs(self, y_true: np.ndarray, y_pred: np.ndarray,
                              protected_attrs: Dict[str, np.ndarray]) -> None:
        self.fairness_metrics.validate_inputs(y_true, y_pred, list(protected_attrs.values())[0])
        
        for attr_name, attr_values in protected_attrs.items():
            if len(attr_values) != len(y_true):
                raise ValueError(f"Protected attribute '{attr_name}' must have same length as y_true")
        
        for attr_name, attr_values in protected_attrs.items():
            group_sizes = self.fairness_metrics.check_group_sizes(y_true, attr_values, min_group_size=10)
            print(f"Group sizes for {attr_name}: {group_sizes}")
    
    def _compute_bootstrap_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   protected_attr: np.ndarray, n_bootstrap: int) -> Dict[str, Tuple[float, float]]:
        intervals = {}
        
        try:
            lower, upper = self.fairness_metrics.bootstrap_confidence_intervals(
                self.fairness_metrics.compute_equal_opportunity,
                n_bootstrap=n_bootstrap,
                y_true=y_true,
                y_pred=y_pred,
                protected_attr=protected_attr,
                metric_key='equal_opportunity_gap'
            )
            intervals['equal_opportunity_gap'] = (lower, upper)
        except Exception as e:
            print(f"Warning: Could not compute Equal Opportunity CI: {e}")
            intervals['equal_opportunity_gap'] = (np.nan, np.nan)
        
        try:
            lower, upper = self.fairness_metrics.bootstrap_confidence_intervals(
                self.fairness_metrics.compute_demographic_parity,
                n_bootstrap=n_bootstrap,
                y_true=y_true,
                y_pred=y_pred,
                protected_attr=protected_attr,
                metric_key='demographic_parity_gap'
            )
            intervals['demographic_parity_gap'] = (lower, upper)
        except Exception as e:
            print(f"Warning: Could not compute Demographic Parity CI: {e}")
            intervals['demographic_parity_gap'] = (np.nan, np.nan)
        
        return intervals
    
    def _identify_disadvantaged_groups(self, fairness_results: Dict, 
                                     protected_attr: np.ndarray) -> Dict[str, Any]:
        group_analysis = {}
        
        eo_results = fairness_results.get('equal_opportunity', {})
        tpr_by_group = {}
        
        for key, value in eo_results.items():
            if key.startswith('tpr_group_'):
                group_name = key.replace('tpr_group_', '')
                tpr_by_group[group_name] = value
        
        if tpr_by_group:
            max_tpr_group = max(tpr_by_group.items(), key=lambda x: x[1])
            min_tpr_group = min(tpr_by_group.items(), key=lambda x: x[1])
            
            group_analysis['tpr_analysis'] = {
                'advantaged_group': {'name': max_tpr_group[0], 'tpr': max_tpr_group[1]},
                'disadvantaged_group': {'name': min_tpr_group[0], 'tpr': min_tpr_group[1]},
                'tpr_gap': max_tpr_group[1] - min_tpr_group[1]
            }
        
        dp_results = fairness_results.get('demographic_parity', {})
        pos_rate_by_group = {}
        
        for key, value in dp_results.items():
            if key.startswith('pos_rate_group_'):
                group_name = key.replace('pos_rate_group_', '')
                pos_rate_by_group[group_name] = value
        
        if pos_rate_by_group:
            max_pos_rate_group = max(pos_rate_by_group.items(), key=lambda x: x[1])
            min_pos_rate_group = min(pos_rate_by_group.items(), key=lambda x: x[1])
            
            group_analysis['demographic_parity_analysis'] = {
                'favored_group': {'name': max_pos_rate_group[0], 'pos_rate': max_pos_rate_group[1]},
                'disfavored_group': {'name': min_pos_rate_group[0], 'pos_rate': min_pos_rate_group[1]},
                'pos_rate_gap': max_pos_rate_group[1] - min_pos_rate_group[1]
            }
        
        return group_analysis
    
    def _generate_overall_summary(self, audit_results: Dict) -> Dict[str, Any]:
        summary = {
            'fairness_violations': [],
            'max_gaps': {},
            'statistical_significance': {},
            'intersectional_concerns': []
        }
        
        for attr_name, attr_results in audit_results.get('single_attribute_analysis', {}).items():
            eo_gap = attr_results.get('equal_opportunity', {}).get('equal_opportunity_gap', 0)
            dp_gap = attr_results.get('demographic_parity', {}).get('demographic_parity_gap', 0)
            eq_gap = attr_results.get('equalized_odds', {}).get('equalized_odds_gap', 0)
            
            summary['max_gaps'][attr_name] = {
                'equal_opportunity': eo_gap,
                'demographic_parity': dp_gap,
                'equalized_odds': eq_gap
            }
            
            if eo_gap > 0.1:
                summary['fairness_violations'].append({
                    'attribute': attr_name,
                    'metric': 'Equal Opportunity',
                    'gap': eo_gap,
                    'severity': 'high' if eo_gap > 0.2 else 'moderate'
                })
            
            if dp_gap > 0.1:
                summary['fairness_violations'].append({
                    'attribute': attr_name,
                    'metric': 'Demographic Parity',
                    'gap': dp_gap,
                    'severity': 'high' if dp_gap > 0.2 else 'moderate'
                })
        
        intersectional_results = audit_results.get('intersectional_analysis', {})
        if intersectional_results:
            intersectional_summary = intersectional_results.get('summary_statistics', {})
            
            max_eo_gap = intersectional_summary.get('max_equal_opportunity_gap', 0)
            max_dp_gap = intersectional_summary.get('max_demographic_parity_gap', 0)
            
            if max_eo_gap > 0.15:
                summary['intersectional_concerns'].append({
                    'metric': 'Equal Opportunity',
                    'max_gap': max_eo_gap,
                    'concern_level': 'high' if max_eo_gap > 0.3 else 'moderate'
                })
        
        return summary
    
    def _generate_recommendations(self, audit_results: Dict) -> List[Dict[str, str]]:
        recommendations = []
        
        overall_summary = audit_results.get('overall_summary', {})
        violations = overall_summary.get('fairness_violations', [])
        
        for violation in violations:
            if violation['metric'] == 'Equal Opportunity':
                recommendations.append({
                    'type': 'mitigation',
                    'priority': violation['severity'],
                    'description': f"Address Equal Opportunity gap of {violation['gap']:.3f} for {violation['attribute']}",
                    'suggested_approach': 'Consider post-processing threshold optimization or pre-processing reweighing'
                })
            
            elif violation['metric'] == 'Demographic Parity':
                recommendations.append({
                    'type': 'mitigation',
                    'priority': violation['severity'],
                    'description': f"Address Demographic Parity gap of {violation['gap']:.3f} for {violation['attribute']}",
                    'suggested_approach': 'Consider pre-processing reweighing or in-processing fairness constraints'
                })
        
        intersectional_concerns = overall_summary.get('intersectional_concerns', [])
        if intersectional_concerns:
            recommendations.append({
                'type': 'analysis',
                'priority': 'high',
                'description': 'Intersectional fairness gaps detected',
                'suggested_approach': 'Conduct detailed intersectional analysis and consider group-specific interventions'
            })
        
        metadata = audit_results.get('metadata', {})
        if metadata.get('n_samples', 0) < 1000:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'moderate',
                'description': 'Small sample size may affect reliability of fairness metrics',
                'suggested_approach': 'Consider collecting more data or using stratified sampling'
            })
        
        return recommendations
    
    def generate_audit_report(self, save_path: Optional[str] = None) -> str:
        if not self.audit_results:
            raise ValueError("No audit results available. Run run_comprehensive_audit() first.")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE FAIRNESS AUDIT REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        metadata = self.audit_results.get('metadata', {})
        report_lines.append("DATASET SUMMARY:")
        report_lines.append(f"  Total samples: {metadata.get('n_samples', 0):,}")
        report_lines.append(f"  Positive samples: {metadata.get('n_positive', 0):,} ({metadata.get('positive_rate', 0):.1%})")
        report_lines.append(f"  Negative samples: {metadata.get('n_negative', 0):,}")
        report_lines.append(f"  Prediction positive rate: {metadata.get('prediction_positive_rate', 0):.1%}")
        report_lines.append("")
        
        overall_summary = self.audit_results.get('overall_summary', {})
        violations = overall_summary.get('fairness_violations', [])
        
        report_lines.append("FAIRNESS ASSESSMENT SUMMARY:")
        if violations:
            report_lines.append(f"  ⚠️  {len(violations)} fairness violation(s) detected")
            for violation in violations:
                report_lines.append(f"    - {violation['metric']} gap: {violation['gap']:.3f} ({violation['severity']} severity)")
        else:
            report_lines.append("  ✅ No significant fairness violations detected")
        report_lines.append("")
        
        single_attr_results = self.audit_results.get('single_attribute_analysis', {})
        for attr_name, attr_results in single_attr_results.items():
            report_lines.append(f"ANALYSIS FOR PROTECTED ATTRIBUTE: {attr_name.upper()}")
            report_lines.append("-" * 50)
            
            eo_results = attr_results.get('equal_opportunity', {})
            eo_gap = eo_results.get('equal_opportunity_gap', 0)
            report_lines.append(f"  Equal Opportunity Gap: {eo_gap:.4f}")
            
            for key, value in eo_results.items():
                if key.startswith('tpr_group_'):
                    group_name = key.replace('tpr_group_', '')
                    report_lines.append(f"    {group_name} TPR: {value:.4f}")
            
            dp_results = attr_results.get('demographic_parity', {})
            dp_gap = dp_results.get('demographic_parity_gap', 0)
            report_lines.append(f"  Demographic Parity Gap: {dp_gap:.4f}")
            
            ci_results = attr_results.get('confidence_intervals', {})
            if 'equal_opportunity_gap' in ci_results:
                ci_lower, ci_upper = ci_results['equal_opportunity_gap']
                if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                    report_lines.append(f"    95% CI for EO gap: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            report_lines.append("")
        
        intersectional_results = self.audit_results.get('intersectional_analysis', {})
        if intersectional_results:
            report_lines.append("INTERSECTIONAL FAIRNESS ANALYSIS:")
            report_lines.append("-" * 50)
            
            summary_stats = intersectional_results.get('summary_statistics', {})
            report_lines.append(f"  Number of intersectional groups: {summary_stats.get('n_intersectional_groups', 0)}")
            report_lines.append(f"  Maximum Equal Opportunity gap: {summary_stats.get('max_equal_opportunity_gap', 0):.4f}")
            report_lines.append(f"  Maximum Demographic Parity gap: {summary_stats.get('max_demographic_parity_gap', 0):.4f}")
            report_lines.append("")
        
        recommendations = self.audit_results.get('recommendations', [])
        if recommendations:
            report_lines.append("RECOMMENDATIONS:")
            report_lines.append("-" * 50)
            
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"  {i}. [{rec['priority'].upper()}] {rec['description']}")
                report_lines.append(f"     Suggested approach: {rec['suggested_approach']}")
                report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text
    
    def create_audit_visualizations(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        if not self.audit_results:
            raise ValueError("No audit results available. Run run_comprehensive_audit() first.")
        
        figures = {}
        
        figures['fairness_gaps'] = self._create_fairness_gaps_plot()
        
        intersectional_results = self.audit_results.get('intersectional_analysis', {})
        if intersectional_results:
            figures['intersectional_gaps'] = self.intersectional_analyzer.visualize_intersectional_gaps(
                {'detailed_analysis': intersectional_results.get('detailed_analysis', {})}
            )
            figures['intersectional_heatmap'] = self.intersectional_analyzer.create_fairness_heatmap(
                {'detailed_analysis': intersectional_results.get('detailed_analysis', {})}
            )
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            for fig_name, fig in figures.items():
                fig_path = os.path.join(save_dir, f"{fig_name}.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Saved {fig_name} to {fig_path}")
        
        return figures
    
    def _create_fairness_gaps_plot(self) -> plt.Figure:
        single_attr_results = self.audit_results.get('single_attribute_analysis', {})
        
        if not single_attr_results:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No fairness analysis results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Fairness Gaps Overview')
            return fig
        
        attributes = list(single_attr_results.keys())
        eo_gaps = []
        dp_gaps = []
        eq_gaps = []
        
        for attr_name in attributes:
            attr_results = single_attr_results[attr_name]
            eo_gaps.append(attr_results.get('equal_opportunity', {}).get('equal_opportunity_gap', 0))
            dp_gaps.append(attr_results.get('demographic_parity', {}).get('demographic_parity_gap', 0))
            eq_gaps.append(attr_results.get('equalized_odds', {}).get('equalized_odds_gap', 0))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(attributes))
        width = 0.25
        
        bars1 = ax.bar(x - width, eo_gaps, width, label='Equal Opportunity', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, dp_gaps, width, label='Demographic Parity', alpha=0.8, color='lightcoral')
        bars3 = ax.bar(x + width, eq_gaps, width, label='Equalized Odds', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Protected Attributes')
        ax.set_ylabel('Fairness Gap')
        ax.set_title('Fairness Gaps by Protected Attribute and Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(attributes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        return fig