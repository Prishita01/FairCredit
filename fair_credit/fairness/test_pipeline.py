import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch
from .pipeline import FairnessAuditPipeline

class TestFairnessAuditPipeline:
    
    def setup_method(self):
        self.pipeline = FairnessAuditPipeline(confidence_level=0.95, n_jobs=1)
    
    def create_test_data(self, n_samples: int = 100) -> tuple:
        np.random.seed(42)
        
        n_per_group = n_samples // 4
        
        y_true = np.array([1] * (n_per_group * 2) + [0] * (n_per_group * 2) +  
                         [1] * (n_per_group * 2) + [0] * (n_per_group * 2))   
        
        y_pred = np.array([1] * int(n_per_group * 1.8) + [0] * int(n_per_group * 0.2) +  
                         [0] * (n_per_group * 2) +                                        
                         [1] * int(n_per_group * 1.2) + [0] * int(n_per_group * 0.8) +  
                         [0] * (n_per_group * 2))                                        
        
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        sex_attr = np.array(['male'] * (min_len // 2) + ['female'] * (min_len // 2))
        age_attr = np.array(['young'] * (min_len // 4) + ['old'] * (min_len // 4) + 
                           ['young'] * (min_len // 4) + ['old'] * (min_len // 4))
        
        protected_attrs = {'sex': sex_attr, 'age': age_attr}
        
        return y_true, y_pred, protected_attrs
    
    def test_comprehensive_audit_basic(self):
        y_true, y_pred, protected_attrs = self.create_test_data(n_samples=80)
        
        results = self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=50
        )
        
        assert 'metadata' in results
        assert 'single_attribute_analysis' in results
        assert 'intersectional_analysis' in results
        assert 'overall_summary' in results
        assert 'recommendations' in results
        
        metadata = results['metadata']
        assert metadata['n_samples'] == len(y_true)
        assert metadata['n_positive'] == np.sum(y_true == 1)
        assert metadata['confidence_level'] == 0.95
        
        single_attr = results['single_attribute_analysis']
        assert 'sex' in single_attr
        assert 'age' in single_attr
        
        for attr_name, attr_results in single_attr.items():
            assert 'equal_opportunity' in attr_results
            assert 'demographic_parity' in attr_results
            assert 'equalized_odds' in attr_results
            assert 'confidence_intervals' in attr_results
            assert 'group_analysis' in attr_results
    
    def test_intersectional_analysis_integration(self):
        y_true, y_pred, protected_attrs = self.create_test_data(n_samples=80)
        
        results = self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=20
        )
        
        intersectional = results['intersectional_analysis']
        assert 'intersectional_metrics' in intersectional
        assert 'detailed_analysis' in intersectional
        assert 'summary_statistics' in intersectional
        
        detailed_analysis = intersectional['detailed_analysis']
        expected_groups = ['male_young', 'male_old', 'female_young', 'female_old']
        
        for group in expected_groups:
            if group in detailed_analysis:  
                assert 'tpr' in detailed_analysis[group]
                assert 'group_size' in detailed_analysis[group]
    
    def test_disadvantaged_group_identification(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])  
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])  
        
        protected_attrs = {
            'sex': np.array(['male'] * 4 + ['female'] * 4)
        }
        
        results = self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=20
        )
        
        sex_analysis = results['single_attribute_analysis']['sex']
        group_analysis = sex_analysis['group_analysis']
        
        tpr_analysis = group_analysis.get('tpr_analysis', {})
        if tpr_analysis:
            assert tpr_analysis['advantaged_group']['name'] == 'male'
            assert tpr_analysis['disadvantaged_group']['name'] == 'female'
            assert tpr_analysis['tpr_gap'] > 0.5  
    
    def test_overall_summary_generation(self):
        y_true = np.array([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10)
        y_pred = np.array([1] * 10 + [0] * 10 + [0] * 10 + [0] * 10)  
        
        protected_attrs = {
            'group': np.array(['A'] * 20 + ['B'] * 20)
        }
        
        results = self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=20
        )
        
        summary = results['overall_summary']
        
        violations = summary['fairness_violations']
        assert len(violations) > 0
        
        max_gaps = summary['max_gaps']
        assert 'group' in max_gaps
        assert max_gaps['group']['equal_opportunity'] > 0.5  
    
    def test_recommendations_generation(self):
        y_true = np.array([1] * 8 + [0] * 8 + [1] * 8 + [0] * 8)
        y_pred = np.array([1] * 7 + [0] * 9 + [1] * 5 + [0] * 11)  
        
        protected_attrs = {
            'attribute': np.array(['X'] * 16 + ['Y'] * 16)
        }
        
        results = self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=20
        )
        
        recommendations = results['recommendations']
        assert isinstance(recommendations, list)
        
        if len(recommendations) > 0:
            rec = recommendations[0]
            assert 'type' in rec
            assert 'priority' in rec
            assert 'description' in rec
            assert 'suggested_approach' in rec
    
    def test_audit_report_generation(self):
        y_true, y_pred, protected_attrs = self.create_test_data(n_samples=60)
        
        self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=20
        )
        
        report = self.pipeline.generate_audit_report()
        
        assert isinstance(report, str)
        assert "COMPREHENSIVE FAIRNESS AUDIT REPORT" in report
        assert "DATASET SUMMARY" in report
        assert "FAIRNESS ASSESSMENT SUMMARY" in report
        
        assert "SEX" in report.upper()
        assert "AGE" in report.upper()
    
    def test_audit_report_no_results_error(self):
        with pytest.raises(ValueError, match="No audit results available"):
            self.pipeline.generate_audit_report()
    
    def test_visualization_creation(self):
        y_true, y_pred, protected_attrs = self.create_test_data(n_samples=60)
        
        self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=20
        )
        
        figures = self.pipeline.create_audit_visualizations()
        
        assert isinstance(figures, dict)
        assert 'fairness_gaps' in figures
        
        for fig_name, fig in figures.items():
            assert isinstance(fig, plt.Figure)
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_visualization_no_results_error(self):
        with pytest.raises(ValueError, match="No audit results available"):
            self.pipeline.create_audit_visualizations()
    
    def test_input_validation(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 0])
        
        protected_attrs = {
            'attr': np.array(['A', 'B'])  
        }
        
        with pytest.raises(ValueError, match="same length"):
            self.pipeline.run_comprehensive_audit(y_true, y_pred, protected_attrs)
        
        protected_attrs = {
            'attr': np.array(['A', 'A', 'A'])  
        }
        
        with pytest.raises(ValueError, match="at least 2 groups"):
            self.pipeline.run_comprehensive_audit(y_true, y_pred, protected_attrs)
    
    def test_bootstrap_confidence_intervals_integration(self):
        y_true, y_pred, protected_attrs = self.create_test_data(n_samples=60)
        
        results = self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=30
        )
        
        for attr_name, attr_results in results['single_attribute_analysis'].items():
            ci_results = attr_results['confidence_intervals']
            
            assert 'equal_opportunity_gap' in ci_results
            assert 'demographic_parity_gap' in ci_results
            
            eo_ci = ci_results['equal_opportunity_gap']
            assert isinstance(eo_ci, tuple)
            assert len(eo_ci) == 2
    
    def test_small_sample_size_warning(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 0])
        protected_attrs = {
            'attr': np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        }
        
        results = self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=10
        )
        
        recommendations = results['recommendations']
        sample_size_recs = [r for r in recommendations if r['type'] == 'data_quality']
        
        assert len(sample_size_recs) > 0
        assert 'sample size' in sample_size_recs[0]['description'].lower()
    
    def test_perfect_fairness_scenario(self):
        y_true = np.array([1, 1, 0, 0] * 4)  
        y_pred = np.array([1, 1, 0, 0] * 4)  
        
        protected_attrs = {
            'group': np.array(['A'] * 8 + ['B'] * 8)
        }
        
        results = self.pipeline.run_comprehensive_audit(
            y_true, y_pred, protected_attrs, n_bootstrap=20
        )
        
        violations = results['overall_summary']['fairness_violations']
        assert len(violations) == 0
        
        max_gaps = results['overall_summary']['max_gaps']
        for attr_name, gaps in max_gaps.items():
            assert gaps['equal_opportunity'] < 0.01
            assert gaps['demographic_parity'] < 0.01


if __name__ == "__main__":
    pytest.main([__file__])