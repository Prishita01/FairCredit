import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings


class GroupwiseAnalyzer:

    def __init__(self, alpha: float = 0.05, min_group_size: int = 10):
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.feature_names = None
        self.group_stats = None
        
    def analyze_groups(self, 
                      shap_values: np.ndarray, 
                      protected_attrs: Dict[str, np.ndarray],
                      feature_names: List[str]) -> Dict[str, Any]:

        if len(shap_values) == 0:
            raise ValueError("SHAP values array cannot be empty")
        
        if shap_values.shape[1] != len(feature_names):
            raise ValueError("Number of features in SHAP values must match feature names")
        
        self.feature_names = feature_names
        
        for attr_name, attr_values in protected_attrs.items():
            if len(attr_values) != len(shap_values):
                raise ValueError(f"Protected attribute '{attr_name}' length must match SHAP values")
        
        analysis_results = {
            'feature_names': feature_names,
            'n_samples': len(shap_values),
            'n_features': len(feature_names),
            'protected_attributes': {}
        }

        for attr_name, attr_values in protected_attrs.items():
            attr_analysis = self._analyze_single_attribute(
                shap_values, attr_values, attr_name
            )
            analysis_results['protected_attributes'][attr_name] = attr_analysis

        if len(protected_attrs) > 1:
            intersectional_analysis = self._analyze_intersectional(
                shap_values, protected_attrs
            )
            analysis_results['intersectional_analysis'] = intersectional_analysis
        
        return analysis_results
    
    def _analyze_single_attribute(self, 
                                 shap_values: np.ndarray, 
                                 protected_attr: np.ndarray,
                                 attr_name: str) -> Dict[str, Any]:
        unique_groups = np.unique(protected_attr)
        group_stats = {}

        for group in unique_groups:
            group_mask = protected_attr == group
            group_shap = shap_values[group_mask]
            
            if len(group_shap) < self.min_group_size:
                warnings.warn(f"Group {group} has only {len(group_shap)} samples, "
                            f"minimum {self.min_group_size} recommended for reliable analysis")
            
            group_stats[str(group)] = self._calculate_group_statistics(group_shap)
        
        pairwise_comparisons = self._perform_pairwise_comparisons(
            shap_values, protected_attr, unique_groups
        )
        
        differential_features = self._identify_differential_features(
            group_stats, pairwise_comparisons
        )
        
        return {
            'attribute_name': attr_name,
            'unique_groups': [str(g) for g in unique_groups],
            'group_statistics': group_stats,
            'pairwise_comparisons': pairwise_comparisons,
            'differential_features': differential_features,
            'overall_disparity': self._calculate_overall_disparity(group_stats)
        }
    
    def _calculate_group_statistics(self, group_shap: np.ndarray) -> Dict[str, Any]:
        return {
            'n_samples': len(group_shap),
            'mean_shap': np.mean(group_shap, axis=0).tolist(),
            'std_shap': np.std(group_shap, axis=0).tolist(),
            'median_shap': np.median(group_shap, axis=0).tolist(),
            'mean_abs_shap': np.mean(np.abs(group_shap), axis=0).tolist(),
            'feature_importance_ranking': self._rank_features_by_importance(group_shap),
            'top_positive_features': self._get_top_features(group_shap, positive=True),
            'top_negative_features': self._get_top_features(group_shap, positive=False)
        }
    
    def _rank_features_by_importance(self, group_shap: np.ndarray) -> List[Tuple[str, float]]:
        importance_scores = np.mean(np.abs(group_shap), axis=0)
        
        if importance_scores.ndim > 1:
            importance_scores = importance_scores.flatten()
        
        feature_importance = [(name, float(score)) for name, score in zip(self.feature_names, importance_scores)]
        return sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    def _get_top_features(self, group_shap: np.ndarray, positive: bool = True, top_k: int = 5) -> List[Tuple[str, float]]:
        mean_shap = np.mean(group_shap, axis=0)
        
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.flatten()
        
        feature_values = [(name, float(val)) for name, val in zip(self.feature_names, mean_shap)]
        
        if positive:
            sorted_features = sorted(feature_values, key=lambda x: x[1], reverse=True)
        else:
            sorted_features = sorted(feature_values, key=lambda x: x[1])
        
        return sorted_features[:top_k]
    
    def _perform_pairwise_comparisons(self, 
                                    shap_values: np.ndarray, 
                                    protected_attr: np.ndarray,
                                    unique_groups: np.ndarray) -> Dict[str, Any]:
        comparisons = {}
        
        for i, group1 in enumerate(unique_groups):
            for j, group2 in enumerate(unique_groups):
                if i >= j:
                    continue
                
                group1_mask = protected_attr == group1
                group2_mask = protected_attr == group2
                
                group1_shap = shap_values[group1_mask]
                group2_shap = shap_values[group2_mask]
                
                comparison_key = f"{group1}_vs_{group2}"
                comparisons[comparison_key] = self._compare_two_groups(
                    group1_shap, group2_shap, str(group1), str(group2)
                )
        
        return comparisons
    
    def _compare_two_groups(self, 
                           group1_shap: np.ndarray, 
                           group2_shap: np.ndarray,
                           group1_name: str,
                           group2_name: str) -> Dict[str, Any]:

        n_features = group1_shap.shape[1]
        
        t_statistics = []
        p_values = []
        effect_sizes = []
        mean_differences = []
        
        for feature_idx in range(n_features):
            group1_feature = group1_shap[:, feature_idx]
            group2_feature = group2_shap[:, feature_idx]
            
            t_stat, p_val = stats.ttest_ind(group1_feature, group2_feature, equal_var=False)
            
            pooled_std = np.sqrt(((len(group1_feature) - 1) * np.var(group1_feature, ddof=1) + 
                                 (len(group2_feature) - 1) * np.var(group2_feature, ddof=1)) / 
                                (len(group1_feature) + len(group2_feature) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(group1_feature) - np.mean(group2_feature)) / pooled_std
            else:
                cohens_d = 0.0
            
            mean_diff = np.mean(group1_feature) - np.mean(group2_feature)
            
            t_statistics.append(t_stat)
            p_values.append(p_val)
            effect_sizes.append(cohens_d)
            mean_differences.append(mean_diff)

        p_values_corrected = np.array(p_values) * n_features
        p_values_corrected = np.minimum(p_values_corrected, 1.0)

        significant_features = []
        for i, (p_val, effect_size) in enumerate(zip(p_values_corrected, effect_sizes)):
            if isinstance(p_val, np.ndarray):
                p_val_scalar = float(p_val.flatten()[0]) if p_val.size > 0 else float(p_val)
            else:
                p_val_scalar = float(p_val)
            
            if isinstance(effect_size, np.ndarray):
                effect_size_scalar = float(effect_size.flatten()[0]) if effect_size.size > 0 else float(effect_size)
            else:
                effect_size_scalar = float(effect_size)
            
            if p_val_scalar < self.alpha:
                mean_diff = mean_differences[i]
                if isinstance(mean_diff, np.ndarray):
                    mean_diff = float(mean_diff.flatten()[0]) if mean_diff.size > 0 else float(mean_diff)
                else:
                    mean_diff = float(mean_diff)
                
                significant_features.append({
                    'feature': self.feature_names[i],
                    'p_value': p_val_scalar,
                    'effect_size': effect_size_scalar,
                    'mean_difference': mean_diff,
                    'interpretation': self._interpret_effect_size(effect_size_scalar)
                })

        significant_features.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_size': len(group1_shap),
            'group2_size': len(group2_shap),
            't_statistics': t_statistics,
            'p_values': p_values,
            'p_values_corrected': p_values_corrected.tolist(),
            'effect_sizes': effect_sizes,
            'mean_differences': mean_differences,
            'significant_features': significant_features,
            'n_significant_features': len(significant_features)
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _identify_differential_features(self, 
                                      group_stats: Dict[str, Dict], 
                                      pairwise_comparisons: Dict[str, Dict]) -> Dict[str, Any]:
        feature_significance_counts = {feature: 0 for feature in self.feature_names}
        feature_max_effect_sizes = {feature: 0.0 for feature in self.feature_names}
        
        for comparison in pairwise_comparisons.values():
            for sig_feature in comparison['significant_features']:
                feature_name = sig_feature['feature']
                feature_significance_counts[feature_name] += 1
                feature_max_effect_sizes[feature_name] = max(
                    feature_max_effect_sizes[feature_name],
                    abs(sig_feature['effect_size'])
                )

        differential_ranking = []
        for feature in self.feature_names:
            differential_score = (feature_significance_counts[feature] * 
                                feature_max_effect_sizes[feature])
            differential_ranking.append({
                'feature': feature,
                'significance_count': feature_significance_counts[feature],
                'max_effect_size': feature_max_effect_sizes[feature],
                'differential_score': differential_score
            })
        
        differential_ranking.sort(key=lambda x: x['differential_score'], reverse=True)
        
        return {
            'most_differential_features': differential_ranking[:10],
            'features_with_no_differences': [
                f['feature'] for f in differential_ranking 
                if f['significance_count'] == 0
            ]
        }
    
    def _calculate_overall_disparity(self, group_stats: Dict[str, Dict]) -> Dict[str, float]:
        if len(group_stats) < 2:
            return {'max_disparity': 0.0, 'mean_disparity': 0.0}
        
        group_names = list(group_stats.keys())
        feature_disparities = []
        
        for feature_idx in range(len(self.feature_names)):
            feature_importances = []
            for group_name in group_names:
                importance = group_stats[group_name]['mean_abs_shap'][feature_idx]
                feature_importances.append(importance)

            if np.mean(feature_importances) > 0:
                cv = np.std(feature_importances) / np.mean(feature_importances)
            else:
                cv = 0.0
            
            feature_disparities.append(cv)
        
        return {
            'max_disparity': float(np.max(feature_disparities)),
            'mean_disparity': float(np.mean(feature_disparities)),
            'feature_disparities': feature_disparities
        }
    
    def _analyze_intersectional(self, 
                               shap_values: np.ndarray, 
                               protected_attrs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if len(protected_attrs) < 2:
            return {}
        
        attr_names = list(protected_attrs.keys())
        attr_values = list(protected_attrs.values())
        
        if len(attr_names) == 2:
            return self._analyze_two_way_intersection(
                shap_values, attr_values[0], attr_values[1], attr_names
            )
        else:
            return self._analyze_multi_way_intersection(
                shap_values, protected_attrs
            )
    
    def _analyze_two_way_intersection(self, 
                                    shap_values: np.ndarray,
                                    attr1: np.ndarray,
                                    attr2: np.ndarray,
                                    attr_names: List[str]) -> Dict[str, Any]:

        unique_attr1 = np.unique(attr1)
        unique_attr2 = np.unique(attr2)
        
        intersectional_groups = {}
        
        for val1 in unique_attr1:
            for val2 in unique_attr2:
                group_mask = (attr1 == val1) & (attr2 == val2)
                group_shap = shap_values[group_mask]
                
                if len(group_shap) >= self.min_group_size:
                    group_name = f"{attr_names[0]}_{val1}_{attr_names[1]}_{val2}"
                    intersectional_groups[group_name] = self._calculate_group_statistics(group_shap)
        
        intersectional_disparities = self._calculate_intersectional_disparities(
            intersectional_groups, attr_names, unique_attr1, unique_attr2
        )
        
        return {
            'attribute_names': attr_names,
            'intersectional_groups': intersectional_groups,
            'intersectional_disparities': intersectional_disparities,
            'n_intersectional_groups': len(intersectional_groups)
        }
    
    def _analyze_multi_way_intersection(self, 
                                      shap_values: np.ndarray,
                                      protected_attrs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        combined_groups = []
        for i in range(len(shap_values)):
            group_id = "_".join([f"{attr}_{protected_attrs[attr][i]}" 
                               for attr in protected_attrs.keys()])
            combined_groups.append(group_id)
        
        combined_groups = np.array(combined_groups)
        unique_combined = np.unique(combined_groups)
        
        valid_groups = {}
        for group in unique_combined:
            group_mask = combined_groups == group
            if np.sum(group_mask) >= self.min_group_size:
                group_shap = shap_values[group_mask]
                valid_groups[group] = self._calculate_group_statistics(group_shap)
        
        return {
            'attribute_names': list(protected_attrs.keys()),
            'combined_groups': valid_groups,
            'n_combined_groups': len(valid_groups),
            'total_possible_groups': len(unique_combined)
        }
    
    def _calculate_intersectional_disparities(self, 
                                            intersectional_groups: Dict[str, Dict],
                                            attr_names: List[str],
                                            unique_attr1: np.ndarray,
                                            unique_attr2: np.ndarray) -> Dict[str, Any]:
        main_effects = {}
        
        attr1_effects = {}
        for val1 in unique_attr1:
            groups_with_val1 = [group for group in intersectional_groups.keys() 
                              if f"{attr_names[0]}_{val1}" in group]
            if groups_with_val1:
                combined_importance = np.mean([
                    intersectional_groups[group]['mean_abs_shap'] 
                    for group in groups_with_val1
                ], axis=0)
                attr1_effects[str(val1)] = combined_importance.tolist()
        
        main_effects[attr_names[0]] = attr1_effects
        
        attr2_effects = {}
        for val2 in unique_attr2:
            groups_with_val2 = [group for group in intersectional_groups.keys() 
                              if f"{attr_names[1]}_{val2}" in group]
            if groups_with_val2:
                combined_importance = np.mean([
                    intersectional_groups[group]['mean_abs_shap'] 
                    for group in groups_with_val2
                ], axis=0)
                attr2_effects[str(val2)] = combined_importance.tolist()
        
        main_effects[attr_names[1]] = attr2_effects
        
        return {
            'main_effects': main_effects,
            'interaction_strength': self._calculate_interaction_strength(intersectional_groups)
        }
    
    def _calculate_interaction_strength(self, intersectional_groups: Dict[str, Dict]) -> float:
        if len(intersectional_groups) < 2:
            return 0.0
        
        all_importances = []
        for group_stats in intersectional_groups.values():
            all_importances.append(group_stats['mean_abs_shap'])
        
        if len(all_importances) == 0:
            return 0.0

        all_importances = np.array(all_importances)
        feature_cvs = []
        
        for feature_idx in range(all_importances.shape[1]):
            feature_importances = all_importances[:, feature_idx]
            if np.mean(feature_importances) > 0:
                cv = np.std(feature_importances) / np.mean(feature_importances)
                feature_cvs.append(cv)
        
        return float(np.mean(feature_cvs)) if feature_cvs else 0.0
    
    def generate_summary_report(self, analysis_results: Dict[str, Any]) -> str:
        report = []
        report.append("=" * 60)
        report.append("GROUP-WISE FEATURE ANALYSIS SUMMARY")
        report.append("=" * 60)
        report.append(f"Total samples: {analysis_results['n_samples']}")
        report.append(f"Total features: {analysis_results['n_features']}")
        report.append("")
        
        for attr_name, attr_analysis in analysis_results['protected_attributes'].items():
            report.append(f"PROTECTED ATTRIBUTE: {attr_name}")
            report.append("-" * 40)
            report.append(f"Groups: {', '.join(attr_analysis['unique_groups'])}")
            
            group_sizes = []
            for group in attr_analysis['unique_groups']:
                size = attr_analysis['group_statistics'][group]['n_samples']
                group_sizes.append(f"{group}: {size}")
            report.append(f"Group sizes: {', '.join(group_sizes)}")
            
            disparity = attr_analysis['overall_disparity']
            report.append(f"Max feature disparity: {disparity['max_disparity']:.3f}")
            report.append(f"Mean feature disparity: {disparity['mean_disparity']:.3f}")

            diff_features = attr_analysis['differential_features']['most_differential_features'][:5]
            if diff_features:
                report.append("Top differential features:")
                for i, feature_info in enumerate(diff_features, 1):
                    report.append(f"  {i}. {feature_info['feature']} "
                                f"(score: {feature_info['differential_score']:.3f})")
            
            n_comparisons = len(attr_analysis['pairwise_comparisons'])
            total_significant = sum(
                comp['n_significant_features'] 
                for comp in attr_analysis['pairwise_comparisons'].values()
            )
            report.append(f"Pairwise comparisons: {n_comparisons}")
            report.append(f"Total significant feature differences: {total_significant}")
            report.append("")

        if 'intersectional_analysis' in analysis_results:
            intersectional = analysis_results['intersectional_analysis']
            report.append("INTERSECTIONAL ANALYSIS")
            report.append("-" * 40)
            report.append(f"Intersectional groups: {intersectional['n_intersectional_groups']}")
            if 'interaction_strength' in intersectional.get('intersectional_disparities', {}):
                strength = intersectional['intersectional_disparities']['interaction_strength']
                report.append(f"Interaction strength: {strength:.3f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)