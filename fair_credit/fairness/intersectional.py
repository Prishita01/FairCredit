import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from .metrics import FairnessMetrics


class IntersectionalAnalyzer:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.fairness_metrics = FairnessMetrics(confidence_level=confidence_level)
    
    def analyze_intersectional_fairness(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       sex_attr: np.ndarray, age_attr: np.ndarray) -> Dict[str, Dict]:
        if not (len(y_true) == len(y_pred) == len(sex_attr) == len(age_attr)):
            raise ValueError("All input arrays must have the same length")

        intersectional_results = self.fairness_metrics.intersectional_analysis(
            y_true, y_pred, [sex_attr, age_attr]
        )

        detailed_analysis = self._compute_detailed_group_analysis(
            y_true, y_pred, sex_attr, age_attr
        )

        results = {
            'intersectional_metrics': intersectional_results,
            'detailed_analysis': detailed_analysis,
            'summary_statistics': self._compute_summary_statistics(intersectional_results),
            'group_comparisons': self._compute_group_comparisons(detailed_analysis)
        }
        
        return results
    
    def _compute_detailed_group_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        sex_attr: np.ndarray, age_attr: np.ndarray) -> Dict[str, Dict]:
        detailed_analysis = {}
        sex_values = np.unique(sex_attr)
        age_values = np.unique(age_attr)
        
        for sex in sex_values:
            for age in age_values:
                group_name = f"{sex}_{age}"
              
                group_mask = (sex_attr == sex) & (age_attr == age)
                
                if np.sum(group_mask) == 0:
                    continue
                
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]

                group_analysis = self._compute_group_metrics(group_y_true, group_y_pred)
                group_analysis['group_size'] = np.sum(group_mask)
                group_analysis['sex'] = sex
                group_analysis['age'] = age
                
                detailed_analysis[group_name] = group_analysis
        
        return detailed_analysis
    
    def _compute_group_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
        
        metrics = {}
        
        metrics['n_samples'] = len(y_true)
        metrics['n_positive'] = np.sum(y_true == 1)
        metrics['n_negative'] = np.sum(y_true == 0)
        
        metrics['positive_rate'] = np.mean(y_pred)  
        metrics['base_rate'] = np.mean(y_true)
   
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
        
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # 1 - Specificity
        metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # 1 - Sensitivity
        
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)
        
        return metrics
    
    def _compute_summary_statistics(self, intersectional_results: Dict) -> Dict[str, float]:
 
        summary = {}
        
        eo_results = intersectional_results.get('equal_opportunity', {})
        eq_results = intersectional_results.get('equalized_odds', {})
        dp_results = intersectional_results.get('demographic_parity', {})

        if 'equal_opportunity_gap' in eo_results:
            summary['max_equal_opportunity_gap'] = eo_results['equal_opportunity_gap']
        
        if 'equalized_odds_gap' in eq_results:
            summary['max_equalized_odds_gap'] = eq_results['equalized_odds_gap']
        
        if 'demographic_parity_gap' in dp_results:
            summary['max_demographic_parity_gap'] = dp_results['demographic_parity_gap']
        
        group_sizes = intersectional_results.get('group_sizes', {})
        summary['n_intersectional_groups'] = len(group_sizes)
        summary['min_group_size'] = min(group_sizes.values()) if group_sizes else 0
        summary['max_group_size'] = max(group_sizes.values()) if group_sizes else 0
        
        return summary
    
    def _compute_group_comparisons(self, detailed_analysis: Dict[str, Dict]) -> Dict[str, Dict]:
        comparisons = {}
        group_names = list(detailed_analysis.keys())
        
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                comparison_key = f"{group1}_vs_{group2}"
                
                group1_data = detailed_analysis[group1]
                group2_data = detailed_analysis[group2]
                
                comparison = {
                    'tpr_diff': abs(group1_data.get('tpr', 0) - group2_data.get('tpr', 0)),
                    'fpr_diff': abs(group1_data.get('fpr', 0) - group2_data.get('fpr', 0)),
                    'positive_rate_diff': abs(group1_data.get('positive_rate', 0) - group2_data.get('positive_rate', 0)),
                    'precision_diff': abs(group1_data.get('precision', 0) - group2_data.get('precision', 0)),
                    'f1_diff': abs(group1_data.get('f1_score', 0) - group2_data.get('f1_score', 0)),
                    'group1_size': group1_data.get('group_size', 0),
                    'group2_size': group2_data.get('group_size', 0)
                }
                
                comparisons[comparison_key] = comparison
        
        return comparisons
    
    def visualize_intersectional_gaps(self, analysis_results: Dict, 
                                     save_path: Optional[str] = None) -> plt.Figure:
        detailed_analysis = analysis_results['detailed_analysis']
        
        groups = []
        tpr_values = []
        fpr_values = []
        pos_rate_values = []
        group_sizes = []
        
        for group_name, group_data in detailed_analysis.items():
            groups.append(group_name)
            tpr_values.append(group_data.get('tpr', 0))
            fpr_values.append(group_data.get('fpr', 0))
            pos_rate_values.append(group_data.get('positive_rate', 0))
            group_sizes.append(group_data.get('group_size', 0))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Intersectional Fairness Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: True Positive Rates (Equal Opportunity)
        axes[0, 0].bar(groups, tpr_values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('True Positive Rate by Intersectional Group')
        axes[0, 0].set_ylabel('TPR')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: False Positive Rates
        axes[0, 1].bar(groups, fpr_values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('False Positive Rate by Intersectional Group')
        axes[0, 1].set_ylabel('FPR')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Positive Prediction Rates (Demographic Parity)
        axes[1, 0].bar(groups, pos_rate_values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Positive Prediction Rate by Intersectional Group')
        axes[1, 0].set_ylabel('Positive Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(groups, group_sizes, color='gold', alpha=0.7)
        axes[1, 1].set_title('Sample Size by Intersectional Group')
        axes[1, 1].set_ylabel('Sample Size')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_fairness_heatmap(self, analysis_results: Dict,
                               save_path: Optional[str] = None) -> plt.Figure:
        detailed_analysis = analysis_results['detailed_analysis']
        
        groups = list(detailed_analysis.keys())
        metrics = ['tpr', 'fpr', 'positive_rate', 'precision', 'f1_score']
        
        data_matrix = []
        for group in groups:
            group_data = detailed_analysis[group]
            row = [group_data.get(metric, 0) for metric in metrics]
            data_matrix.append(row)
        
        df = pd.DataFrame(data_matrix, index=groups, columns=metrics)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0.5, 
                   square=True, ax=ax, cbar_kws={'label': 'Metric Value'})
        
        ax.set_title('Fairness Metrics Heatmap by Intersectional Group', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Fairness Metrics')
        ax.set_ylabel('Intersectional Groups')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_intersectional_report(self, analysis_results: Dict) -> str:
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("INTERSECTIONAL FAIRNESS ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        summary = analysis_results['summary_statistics']
        report_lines.append("SUMMARY STATISTICS:")
        report_lines.append(f"  Number of intersectional groups: {summary.get('n_intersectional_groups', 0)}")
        report_lines.append(f"  Minimum group size: {summary.get('min_group_size', 0)}")
        report_lines.append(f"  Maximum group size: {summary.get('max_group_size', 0)}")
        report_lines.append(f"  Maximum Equal Opportunity gap: {summary.get('max_equal_opportunity_gap', 0):.4f}")
        report_lines.append(f"  Maximum Equalized Odds gap: {summary.get('max_equalized_odds_gap', 0):.4f}")
        report_lines.append(f"  Maximum Demographic Parity gap: {summary.get('max_demographic_parity_gap', 0):.4f}")
        report_lines.append("")
        
        detailed_analysis = analysis_results['detailed_analysis']
        report_lines.append("DETAILED GROUP ANALYSIS:")
        
        for group_name, group_data in detailed_analysis.items():
            report_lines.append(f"\n  Group: {group_name}")
            report_lines.append(f"    Sample size: {group_data.get('group_size', 0)}")
            report_lines.append(f"    True Positive Rate: {group_data.get('tpr', 0):.4f}")
            report_lines.append(f"    False Positive Rate: {group_data.get('fpr', 0):.4f}")
            report_lines.append(f"    Positive Prediction Rate: {group_data.get('positive_rate', 0):.4f}")
            report_lines.append(f"    Precision: {group_data.get('precision', 0):.4f}")
            report_lines.append(f"    F1 Score: {group_data.get('f1_score', 0):.4f}")
        
        comparisons = analysis_results['group_comparisons']
        report_lines.append("\n\nGROUP COMPARISONS (Top 5 largest gaps):")
        
        sorted_comparisons = sorted(comparisons.items(), 
                                  key=lambda x: x[1]['tpr_diff'], reverse=True)
        
        for i, (comparison_name, comparison_data) in enumerate(sorted_comparisons[:5]):
            report_lines.append(f"\n  {i+1}. {comparison_name}")
            report_lines.append(f"     TPR difference: {comparison_data['tpr_diff']:.4f}")
            report_lines.append(f"     FPR difference: {comparison_data['fpr_diff']:.4f}")
            report_lines.append(f"     Positive rate difference: {comparison_data['positive_rate_diff']:.4f}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
