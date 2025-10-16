import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch
from .intersectional import IntersectionalAnalyzer


class TestIntersectionalAnalyzer:
    def setup_method(self):
        self.analyzer = IntersectionalAnalyzer(confidence_level=0.95)
    
    def test_analyze_intersectional_fairness_basic(self):
        n_per_group = 10

        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 4)
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 4)
        
        sex_attr = np.array(['male'] * 20 + ['female'] * 20)
        age_attr = np.array(['young'] * 10 + ['old'] * 10 + ['young'] * 10 + ['old'] * 10)
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
        
        assert 'intersectional_metrics' in results
        assert 'detailed_analysis' in results
        assert 'summary_statistics' in results
        assert 'group_comparisons' in results
        
        detailed_analysis = results['detailed_analysis']
        expected_groups = ['male_young', 'male_old', 'female_young', 'female_old']
        
        for group in expected_groups:
            assert group in detailed_analysis
            assert detailed_analysis[group]['group_size'] == n_per_group
    
    def test_analyze_intersectional_fairness_different_patterns(self):

        y_true = np.array([1, 1, 1, 1, 1,  
                          1, 1, 1, 1, 1,  
                          1, 1, 1, 1, 1,  
                          1, 1, 1, 1, 1])
        
        y_pred = np.array([1, 1, 1, 1, 1,
                          1, 1, 0, 0, 0,  
                          1, 1, 1, 0, 0, 
                          0, 0, 0, 0, 0])
        
        sex_attr = np.array(['male'] * 10 + ['female'] * 10)
        age_attr = np.array(['young'] * 5 + ['old'] * 5 + ['young'] * 5 + ['old'] * 5)
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
        
        detailed_analysis = results['detailed_analysis']
        
        assert abs(detailed_analysis['male_young']['tpr'] - 1.0) < 1e-10
        assert abs(detailed_analysis['male_old']['tpr'] - 0.4) < 1e-10
        assert abs(detailed_analysis['female_young']['tpr'] - 0.6) < 1e-10
        assert abs(detailed_analysis['female_old']['tpr'] - 0.0) < 1e-10
        
        summary = results['summary_statistics']
        assert summary['max_equal_opportunity_gap'] > 0.5
    
    def test_detailed_group_analysis_metrics(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        sex_attr = np.array(['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female'])
        age_attr = np.array(['young', 'young', 'young', 'young', 'young', 'young', 'young', 'young'])
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
        
        group_data = results['detailed_analysis']['male_young']
        
        expected_metrics = [
            'n_samples', 'n_positive', 'n_negative', 'positive_rate', 'base_rate',
            'tpr', 'fpr', 'tnr', 'fnr', 'precision', 'recall', 'f1_score',
            'tp', 'fp', 'tn', 'fn', 'group_size', 'sex', 'age'
        ]
        
        for metric in expected_metrics:
            assert metric in group_data
        
        assert group_data['n_samples'] == 4
        assert group_data['n_positive'] == 2
        assert group_data['n_negative'] == 2
        assert group_data['tpr'] == 0.5
        assert group_data['fpr'] == 0.0 
    
    def test_group_comparisons(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        
        sex_attr = np.array(['male'] * 4 + ['female'] * 4)
        age_attr = np.array(['young'] * 8)
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
        
        comparisons = results['group_comparisons']
        
        comparison_key = list(comparisons.keys())[0]
        assert 'male_young' in comparison_key and 'female_young' in comparison_key
        
        comparison = comparisons[comparison_key]
        
        assert comparison['tpr_diff'] == 1.0
        assert comparison['group1_size'] == 4
        assert comparison['group2_size'] == 4
    
    def test_input_validation(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0]) 
        sex_attr = np.array(['male', 'female', 'male'])
        age_attr = np.array(['young', 'old', 'young'])
        
        with pytest.raises(ValueError, match="same length"):
            self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
    
    def test_empty_groups_handling(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        sex_attr = np.array(['male', 'male', 'female', 'female'])
        age_attr = np.array(['young', 'young', 'old', 'old'])
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
        
        detailed_analysis = results['detailed_analysis']
        
        assert 'male_young' in detailed_analysis
        assert 'female_old' in detailed_analysis

        assert len(detailed_analysis) == 2
    
    def test_visualize_intersectional_gaps(self):
        y_true = np.array([1, 1, 0, 0] * 4)
        y_pred = np.array([1, 0, 0, 0] * 4)
        sex_attr = np.array(['male'] * 8 + ['female'] * 8)
        age_attr = np.array(['young'] * 4 + ['old'] * 4 + ['young'] * 4 + ['old'] * 4)
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
        
        fig = self.analyzer.visualize_intersectional_gaps(results)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4
        plt.close(fig)
    
    def test_create_fairness_heatmap(self):
        y_true = np.array([1, 1, 0, 0] * 2)
        y_pred = np.array([1, 0, 0, 0] * 2)
        sex_attr = np.array(['male'] * 4 + ['female'] * 4)
        age_attr = np.array(['young'] * 8)
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
        fig = self.analyzer.create_fairness_heatmap(results)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_generate_intersectional_report(self):
        y_true = np.array([1, 1, 0, 0] * 4)
        y_pred = np.array([1, 0, 0, 0] * 4)
        sex_attr = np.array(['male'] * 8 + ['female'] * 8)
        age_attr = np.array(['young'] * 4 + ['old'] * 4 + ['young'] * 4 + ['old'] * 4)
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
 
        report = self.analyzer.generate_intersectional_report(results)
        assert isinstance(report, str)
        assert "INTERSECTIONAL FAIRNESS ANALYSIS REPORT" in report
        assert "SUMMARY STATISTICS" in report
        assert "DETAILED GROUP ANALYSIS" in report
        assert "GROUP COMPARISONS" in report

        assert "male_young" in report
        assert "female_old" in report
    
    def test_summary_statistics_computation(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1]) 
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        
        sex_attr = np.array(['male'] * 4 + ['female'] * 4)
        age_attr = np.array(['young'] * 8)
        
        results = self.analyzer.analyze_intersectional_fairness(y_true, y_pred, sex_attr, age_attr)
        
        summary = results['summary_statistics']

        assert summary['n_intersectional_groups'] == 2
        assert summary['min_group_size'] == 4
        assert summary['max_group_size'] == 4
        assert summary['max_equal_opportunity_gap'] == 1.0  # |1.0 - 0.0| = 1.0


if __name__ == "__main__":
    pytest.main([__file__])