import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ExplanationVisualizer:

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):

        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib and Seaborn are required for visualization.")
        
        self.figsize = figsize
        self.style = style
        sns.set_style(style)
        
    def plot_shap_summary(self, 
                         shap_values: np.ndarray, 
                         X: pd.DataFrame,
                         feature_names: Optional[List[str]] = None,
                         max_display: int = 20,
                         title: str = "SHAP Summary Plot") -> plt.Figure:
        if not SHAP_AVAILABLE:
            return self._plot_manual_shap_summary(shap_values, X, feature_names, max_display, title)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_manual_shap_summary(self, 
                                 shap_values: np.ndarray, 
                                 X: pd.DataFrame,
                                 feature_names: Optional[List[str]] = None,
                                 max_display: int = 20,
                                 title: str = "SHAP Summary Plot") -> plt.Figure:

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(shap_values.shape[1])]
        
        importance = np.mean(np.abs(shap_values), axis=0)
        
        sorted_indices = np.argsort(importance)[::-1][:max_display]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importance = importance[sorted_indices]
        
        fig, ax = plt.subplots(figsize=self.figsize)

        y_pos = np.arange(len(sorted_features))
        bars = ax.barh(y_pos, sorted_importance, color='skyblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis() 
        
        for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
            ax.text(bar.get_width() + 0.01 * max(sorted_importance), 
                   bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', 
                   va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_group_comparison(self, 
                             group_analysis: Dict[str, Any],
                             attribute_name: str,
                             top_k: int = 10,
                             comparison_metric: str = 'mean_abs_shap') -> plt.Figure:

        if attribute_name not in group_analysis['protected_attributes']:
            raise ValueError(f"Attribute '{attribute_name}' not found in analysis results")
        
        attr_analysis = group_analysis['protected_attributes'][attribute_name]
        group_stats = attr_analysis['group_statistics']
        feature_names = group_analysis['feature_names']
        
        groups = list(group_stats.keys())
        n_features = len(feature_names)
        
        overall_importance = np.zeros(n_features)
        for group_data in group_stats.values():
            overall_importance += np.array(group_data['mean_abs_shap'])
        overall_importance /= len(groups)
        
        top_indices = np.argsort(overall_importance)[::-1][:top_k]
        top_features = [feature_names[i] for i in top_indices]
        

        plot_data = []
        for group in groups:
            group_data = group_stats[group]
            for i, feature_idx in enumerate(top_indices):
                plot_data.append({
                    'Group': f"{attribute_name}_{group}",
                    'Feature': top_features[i],
                    'Importance': group_data[comparison_metric][feature_idx],
                    'Feature_Index': i
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.barplot(
            data=df_plot, 
            x='Feature', 
            y='Importance', 
            hue='Group',
            ax=ax
        )
        
        ax.set_title(f'Feature Importance by {attribute_name.title()} Groups', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel(f'{comparison_metric.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_differential_features(self, 
                                  group_analysis: Dict[str, Any],
                                  attribute_name: str,
                                  top_k: int = 10) -> plt.Figure:

        if attribute_name not in group_analysis['protected_attributes']:
            raise ValueError(f"Attribute '{attribute_name}' not found in analysis results")
        
        attr_analysis = group_analysis['protected_attributes'][attribute_name]
        diff_features = attr_analysis['differential_features']['most_differential_features'][:top_k]
        
        if not diff_features:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No differential features found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Differential Features for {attribute_name.title()}')
            return fig
        
        features = [f['feature'] for f in diff_features]
        scores = [f['differential_score'] for f in diff_features]
        significance_counts = [f['significance_count'] for f in diff_features]
        effect_sizes = [f['max_effect_size'] for f in diff_features]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        bars1 = ax1.barh(range(len(features)), scores, color='coral', alpha=0.7)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Differential Score')
        ax1.set_title('Feature Differential Scores')
        ax1.invert_yaxis()

        for i, (bar, score) in enumerate(zip(bars1, scores)):
            ax1.text(bar.get_width() + 0.01 * max(scores), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', 
                    va='center', fontsize=9)
        
        scatter = ax2.scatter(significance_counts, effect_sizes, 
                            s=100, alpha=0.7, c=scores, cmap='viridis')
        
        for i, feature in enumerate(features):
            ax2.annotate(feature, (significance_counts[i], effect_sizes[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Significance Count')
        ax2.set_ylabel('Max Effect Size')
        ax2.set_title('Effect Size vs Significance')
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Differential Score')
        
        plt.suptitle(f'Differential Features Analysis: {attribute_name.title()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_intersectional_heatmap(self, 
                                   intersectional_analysis: Dict[str, Any],
                                   metric: str = 'mean_abs_shap',
                                   feature_idx: int = 0) -> plt.Figure:
        if 'intersectional_groups' not in intersectional_analysis:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No intersectional analysis available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        groups = intersectional_analysis['intersectional_groups']
        attr_names = intersectional_analysis['attribute_names']
        
        if len(attr_names) != 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'Heatmap only available for two-way intersections', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        group_data = {}
        for group_name, group_stats in groups.items():
            parts = group_name.split('_')
            if len(parts) >= 4:
                attr1_val = parts[1]
                attr2_val = parts[3]
                value = group_stats[metric][feature_idx]
                group_data[(attr1_val, attr2_val)] = value
        
        if not group_data:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No valid intersectional data found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        attr1_vals = sorted(set(key[0] for key in group_data.keys()))
        attr2_vals = sorted(set(key[1] for key in group_data.keys()))
        
        matrix = np.full((len(attr2_vals), len(attr1_vals)), np.nan)
        
        for i, attr2_val in enumerate(attr2_vals):
            for j, attr1_val in enumerate(attr1_vals):
                if (attr1_val, attr2_val) in group_data:
                    matrix[i, j] = group_data[(attr1_val, attr2_val)]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            matrix, 
            xticklabels=[f"{attr_names[0]}_{val}" for val in attr1_vals],
            yticklabels=[f"{attr_names[1]}_{val}" for val in attr2_vals],
            annot=True, 
            fmt='.3f',
            cmap='RdYlBu_r',
            ax=ax
        )
        
        ax.set_title(f'Intersectional Analysis: Feature {feature_idx} ({metric})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_explanation_report(self, 
                                 shap_values: np.ndarray,
                                 X: pd.DataFrame,
                                 group_analysis: Dict[str, Any],
                                 model_predictions: Optional[np.ndarray] = None,
                                 output_path: Optional[str] = None) -> str:
        report_html = []
        
        report_html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explainability Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
                h2 { color: #34495e; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                .summary-box { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .warning { color: #e74c3c; font-weight: bold; }
                .good { color: #27ae60; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: left; }
                th { background-color: #3498db; color: white; }
            </style>
        </head>
        <body>
        """)

        report_html.append(f"""
        <h1>Model Explainability Report</h1>
        <div class="summary-box">
            <h3>Dataset Summary</h3>
            <div class="metric">Samples: {len(X)}</div>
            <div class="metric">Features: {len(X.columns)}</div>
            <div class="metric">Protected Attributes: {len(group_analysis['protected_attributes'])}</div>
        </div>
        """)
        
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        report_html.append("""
        <h2>Overall Feature Importance</h2>
        <table>
            <tr><th>Rank</th><th>Feature</th><th>Mean |SHAP Value|</th></tr>
        """)
        
        for i, idx in enumerate(sorted_indices[:10]):
            feature_name = X.columns[idx]
            importance = feature_importance[idx]
            report_html.append(f"""
            <tr><td>{i+1}</td><td>{feature_name}</td><td>{importance:.4f}</td></tr>
            """)
        
        report_html.append("</table>")

        for attr_name, attr_analysis in group_analysis['protected_attributes'].items():
            report_html.append(f"<h2>Analysis for {attr_name.title()}</h2>")

            group_stats = attr_analysis['group_statistics']
            groups = attr_analysis['unique_groups']
            
            report_html.append(f"""
            <div class="summary-box">
                <h3>Group Statistics</h3>
                <div class="metric">Groups: {', '.join(groups)}</div>
                <div class="metric">Max Disparity: {attr_analysis['overall_disparity']['max_disparity']:.4f}</div>
                <div class="metric">Mean Disparity: {attr_analysis['overall_disparity']['mean_disparity']:.4f}</div>
            </div>
            """)

            report_html.append("""
            <h3>Group Sizes</h3>
            <table>
                <tr><th>Group</th><th>Sample Count</th><th>Percentage</th></tr>
            """)
            
            total_samples = sum(group_stats[group]['n_samples'] for group in groups)
            for group in groups:
                count = group_stats[group]['n_samples']
                percentage = (count / total_samples) * 100
                report_html.append(f"""
                <tr><td>{attr_name}_{group}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>
                """)
            
            report_html.append("</table>")
            
            diff_features = attr_analysis['differential_features']['most_differential_features'][:5]
            if diff_features:
                report_html.append("""
                <h3>Most Differential Features</h3>
                <table>
                    <tr><th>Feature</th><th>Differential Score</th><th>Max Effect Size</th><th>Significance Count</th></tr>
                """)
                
                for feature_info in diff_features:
                    score = feature_info['differential_score']
                    effect_size = feature_info['max_effect_size']
                    sig_count = feature_info['significance_count']
                    
                    if effect_size > 0.8:
                        effect_class = "warning"
                    elif effect_size > 0.5:
                        effect_class = "metric"
                    else:
                        effect_class = "good"
                    
                    report_html.append(f"""
                    <tr>
                        <td>{feature_info['feature']}</td>
                        <td>{score:.4f}</td>
                        <td class="{effect_class}">{effect_size:.4f}</td>
                        <td>{sig_count}</td>
                    </tr>
                    """)
                
                report_html.append("</table>")
            
            comparisons = attr_analysis['pairwise_comparisons']
            if comparisons:
                report_html.append("<h3>Pairwise Group Comparisons</h3>")
                
                for comp_name, comp_data in comparisons.items():
                    n_sig = comp_data['n_significant_features']
                    total_features = len(comp_data['p_values'])
                    
                    if n_sig > 0:
                        status_class = "warning" if n_sig > total_features * 0.2 else "metric"
                    else:
                        status_class = "good"
                    
                    report_html.append(f"""
                    <div class="summary-box">
                        <strong>{comp_name}</strong><br>
                        <span class="{status_class}">Significant features: {n_sig}/{total_features}</span><br>
                        Group sizes: {comp_data['group1_size']} vs {comp_data['group2_size']}
                    </div>
                    """)
        
        if 'intersectional_analysis' in group_analysis:
            intersectional = group_analysis['intersectional_analysis']
            report_html.append(f"""
            <h2>Intersectional Analysis</h2>
            <div class="summary-box">
                <h3>Intersectional Groups</h3>
                <div class="metric">Attributes: {', '.join(intersectional['attribute_names'])}</div>
                <div class="metric">Valid Groups: {intersectional['n_intersectional_groups']}</div>
            """)
            
            if 'intersectional_disparities' in intersectional:
                disparities = intersectional['intersectional_disparities']
                if 'interaction_strength' in disparities:
                    strength = disparities['interaction_strength']
                    strength_class = "warning" if strength > 0.3 else "good"
                    report_html.append(f"""
                    <div class="metric">Interaction Strength: <span class="{strength_class}">{strength:.4f}</span></div>
                    """)
            
            report_html.append("</div>")
        
        report_html.append("""
        <h2>Recommendations</h2>
        <div class="summary-box">
            <h3>Fairness Assessment</h3>
        """)
        
        recommendations = self._generate_recommendations(group_analysis)
        for rec in recommendations:
            report_html.append(f"<p>• {rec}</p>")
        
        report_html.append("</div>")
        
        report_html.append("""
        </body>
        </html>
        """)
        
        html_content = "".join(report_html)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content
    
    def _generate_recommendations(self, group_analysis: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        for attr_name, attr_analysis in group_analysis['protected_attributes'].items():
            max_disparity = attr_analysis['overall_disparity']['max_disparity']
            mean_disparity = attr_analysis['overall_disparity']['mean_disparity']
            
            if max_disparity > 0.5:
                recommendations.append(
                    f"HIGH CONCERN: Large feature importance disparities detected for {attr_name} "
                    f"(max disparity: {max_disparity:.3f}). Consider bias mitigation techniques."
                )
            elif max_disparity > 0.3:
                recommendations.append(
                    f"MODERATE CONCERN: Noticeable feature importance differences for {attr_name} "
                    f"(max disparity: {max_disparity:.3f}). Monitor for potential bias."
                )
            else:
                recommendations.append(
                    f"LOW CONCERN: Relatively consistent feature importance across {attr_name} groups "
                    f"(max disparity: {max_disparity:.3f})."
                )
            
            diff_features = attr_analysis['differential_features']['most_differential_features']
            high_diff_features = [f for f in diff_features if f['max_effect_size'] > 0.8]
            
            if high_diff_features:
                feature_names = [f['feature'] for f in high_diff_features[:3]]
                recommendations.append(
                    f"Features with large group differences for {attr_name}: {', '.join(feature_names)}. "
                    "Investigate if these differences are justified or indicate bias."
                )
            
            group_stats = attr_analysis['group_statistics']
            group_sizes = [group_stats[group]['n_samples'] for group in group_stats.keys()]
            if len(group_sizes) > 1:
                size_ratio = max(group_sizes) / min(group_sizes)
                if size_ratio > 3:
                    recommendations.append(
                        f"Significant group size imbalance for {attr_name} (ratio: {size_ratio:.1f}:1). "
                        "Consider stratified sampling or reweighting techniques."
                    )
        
        if 'intersectional_analysis' in group_analysis:
            intersectional = group_analysis['intersectional_analysis']
            if 'intersectional_disparities' in intersectional:
                disparities = intersectional['intersectional_disparities']
                if 'interaction_strength' in disparities:
                    strength = disparities['interaction_strength']
                    if strength > 0.4:
                        recommendations.append(
                            f"Strong intersectional effects detected (strength: {strength:.3f}). "
                            "Consider intersectional fairness metrics and mitigation strategies."
                        )
        
        recommendations.extend([
            "Regularly monitor model explanations across different demographic groups.",
            "Consider using fairness-aware machine learning techniques if significant disparities are found.",
            "Validate findings with domain experts and stakeholders.",
            "Document and track fairness metrics over time as the model is updated."
        ])
        
        return recommendations


class CounterfactualChecker:

    def __init__(self, protected_features: List[str]):

        self.protected_features = protected_features
    
    def check_protected_attribute_dominance(self, 
                                          shap_values: np.ndarray,
                                          feature_names: List[str],
                                          threshold: float = 0.3) -> Dict[str, Any]:

        feature_importance = np.mean(np.abs(shap_values), axis=0)
        total_importance = np.sum(feature_importance)
        
        protected_indices = []
        protected_importance = []
        
        for i, feature_name in enumerate(feature_names):
            if any(prot_feat in feature_name.lower() for prot_feat in 
                  [pf.lower() for pf in self.protected_features]):
                protected_indices.append(i)
                protected_importance.append(feature_importance[i])
        
        if not protected_indices:
            return {
                'protected_features_found': [],
                'protected_importance_ratio': 0.0,
                'dominance_detected': False,
                'recommendation': 'No protected features detected in feature names.'
            }
        
        protected_total_importance = sum(protected_importance)
        protected_ratio = protected_total_importance / total_importance if total_importance > 0 else 0
        
        dominance_detected = protected_ratio > threshold
        
        protected_analysis = []
        for idx, importance in zip(protected_indices, protected_importance):
            feature_ratio = importance / total_importance if total_importance > 0 else 0
            protected_analysis.append({
                'feature_name': feature_names[idx],
                'importance': float(importance),
                'importance_ratio': float(feature_ratio),
                'rank': int(np.sum(feature_importance > importance) + 1)
            })
        
        protected_analysis.sort(key=lambda x: x['importance'], reverse=True)
        
        if dominance_detected:
            top_protected = protected_analysis[0]['feature_name']
            recommendation = (
                f"WARNING: Protected attributes account for {protected_ratio:.1%} of model importance "
                f"(threshold: {threshold:.1%}). Top protected feature: {top_protected}. "
                "Consider feature engineering or bias mitigation techniques."
            )
        else:
            recommendation = (
                f"Protected attributes account for {protected_ratio:.1%} of model importance "
                f"(below threshold: {threshold:.1%}). No dominance detected."
            )
        
        return {
            'protected_features_found': [pf['feature_name'] for pf in protected_analysis],
            'protected_importance_ratio': float(protected_ratio),
            'dominance_detected': dominance_detected,
            'threshold': threshold,
            'protected_feature_analysis': protected_analysis,
            'recommendation': recommendation
        }
    
    def generate_counterfactuals(self, 
                               X: pd.DataFrame,
                               shap_values: np.ndarray,
                               instance_idx: int,
                               n_counterfactuals: int = 5) -> Dict[str, Any]:
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        instance = X.iloc[instance_idx].copy()
        instance_shap = shap_values[instance_idx]
        
        non_protected_features = []
        for i, feature_name in enumerate(X.columns):
            if not any(prot_feat in feature_name.lower() for prot_feat in 
                      [pf.lower() for pf in self.protected_features]):
                non_protected_features.append((i, feature_name))
        
        if not non_protected_features:
            return {
                'counterfactuals': [],
                'message': 'No non-protected features available for counterfactual generation.'
            }

        non_protected_shap = [(idx, name, abs(instance_shap[idx])) 
                             for idx, name in non_protected_features]
        non_protected_shap.sort(key=lambda x: x[2], reverse=True)
        
        counterfactuals = []
        
        for i in range(min(n_counterfactuals, len(non_protected_shap))):
            feature_idx, feature_name, shap_importance = non_protected_shap[i]
            
            counterfactual = instance.copy()
            
            original_value = instance.iloc[feature_idx]
            if isinstance(original_value, (int, float)):
                if instance_shap[feature_idx] > 0:
                    std_val = X.iloc[:, feature_idx].std()
                    counterfactual.iloc[feature_idx] = original_value - std_val
                else:
                    std_val = X.iloc[:, feature_idx].std()
                    counterfactual.iloc[feature_idx] = original_value + std_val
            else:
                value_counts = X.iloc[:, feature_idx].value_counts()
                alternatives = [val for val in value_counts.index if val != original_value]
                if alternatives:
                    counterfactual.iloc[feature_idx] = alternatives[0]
            
            counterfactuals.append({
                'counterfactual_id': i + 1,
                'modified_feature': feature_name,
                'original_value': original_value,
                'counterfactual_value': counterfactual.iloc[feature_idx],
                'feature_shap_importance': float(shap_importance),
                'counterfactual_instance': counterfactual.to_dict()
            })
        
        return {
            'original_instance': instance.to_dict(),
            'counterfactuals': counterfactuals,
            'n_generated': len(counterfactuals),
            'protected_features_preserved': self.protected_features
        }