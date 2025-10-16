import numpy as np
import pytest
from unittest.mock import patch
from .metrics import FairnessMetrics, BootstrapCI


class TestFairnessMetrics:
    def setup_method(self):
        self.metrics = FairnessMetrics(confidence_level=0.95, n_jobs=1)
    
    def test_equal_opportunity_perfect_fairness(self):
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        protected_attr = np.array(['A'] * 10 + ['B'] * 10)
        
        result = self.metrics.compute_equal_opportunity(y_true, y_pred, protected_attr)

        assert abs(result['tpr_group_A'] - 0.8) < 1e-10
        assert abs(result['tpr_group_B'] - 0.8) < 1e-10

        assert abs(result['equal_opportunity_gap']) < 1e-10
        assert abs(result['eo_gap_A_B']) < 1e-10
    
    def test_equal_opportunity_maximum_bias(self):
        y_true = np.array([1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1,
                          0, 0, 0, 0, 0])
        
        protected_attr = np.array(['A'] * 5 + ['B'] * 5)
        
        result = self.metrics.compute_equal_opportunity(y_true, y_pred, protected_attr)

        assert abs(result['tpr_group_A'] - 1.0) < 1e-10
        assert abs(result['tpr_group_B'] - 0.0) < 1e-10
        
        assert abs(result['equal_opportunity_gap'] - 1.0) < 1e-10
    
    def test_equal_opportunity_no_positives_group(self):
        y_true = np.array([1, 1, 1, 0, 0,  
                          0, 0, 0, 0, 0])  
        
        y_pred = np.array([1, 1, 0, 0, 0,  
                          0, 0, 0, 0, 0])
        
        protected_attr = np.array(['A'] * 5 + ['B'] * 5)
        
        result = self.metrics.compute_equal_opportunity(y_true, y_pred, protected_attr)
        
        assert abs(result['tpr_group_A'] - 2/3) < 1e-10
        
        assert result['tpr_group_B'] == 0.0
    
    def test_equalized_odds_perfect_fairness(self):
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0,  
                          1, 1, 1, 1, 1, 0, 0, 0, 0, 0]) 
        
        y_pred = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
                          1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        
        protected_attr = np.array(['A'] * 10 + ['B'] * 10)
        
        result = self.metrics.compute_equalized_odds(y_true, y_pred, protected_attr)

        assert abs(result['tpr_group_A'] - 0.8) < 1e-10
        assert abs(result['tpr_group_B'] - 0.8) < 1e-10
        assert abs(result['fpr_group_A'] - 0.2) < 1e-10
        assert abs(result['fpr_group_B'] - 0.2) < 1e-10

        assert abs(result['equalized_odds_gap']) < 1e-10
    
    def test_demographic_parity_perfect_fairness(self):
        y_pred = np.array([1, 1, 1, 0, 0, 0,
                          1, 1, 0, 0])
        
        protected_attr = np.array(['A'] * 6 + ['B'] * 4)
        
        result = self.metrics.compute_demographic_parity(y_pred, protected_attr)

        assert abs(result['pos_rate_group_A'] - 0.5) < 1e-10
        assert abs(result['pos_rate_group_B'] - 0.5) < 1e-10

        assert abs(result['demographic_parity_gap']) < 1e-10
    
    def test_demographic_parity_maximum_bias(self):
        y_pred = np.array([1, 1, 1, 1,
                          0, 0, 0, 0])
        
        protected_attr = np.array(['A'] * 4 + ['B'] * 4)
        
        result = self.metrics.compute_demographic_parity(y_pred, protected_attr)

        assert abs(result['pos_rate_group_A'] - 1.0) < 1e-10
        assert abs(result['pos_rate_group_B'] - 0.0) < 1e-10

        assert abs(result['demographic_parity_gap'] - 1.0) < 1e-10
    
    def test_input_validation_mismatched_lengths(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0])
        protected_attr = np.array(['A', 'B', 'A'])
        
        with pytest.raises(ValueError, match="same length"):
            self.metrics.compute_equal_opportunity(y_true, y_pred, protected_attr)
    
    def test_input_validation_empty_arrays(self):
        y_true = np.array([])
        y_pred = np.array([])
        protected_attr = np.array([])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            self.metrics.compute_equal_opportunity(y_true, y_pred, protected_attr)
    
    def test_input_validation_non_binary_labels(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 1])
        protected_attr = np.array(['A', 'B', 'A'])
        
        with pytest.raises(ValueError, match="binary values"):
            self.metrics.compute_equal_opportunity(y_true, y_pred, protected_attr)
    
    def test_input_validation_single_group(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 0])
        protected_attr = np.array(['A', 'A', 'A'])
        
        with pytest.raises(ValueError, match="at least 2 groups"):
            self.metrics.compute_equal_opportunity(y_true, y_pred, protected_attr)
    
    def test_intersectional_analysis_sex_age(self):
        n_per_group = 5
        y_true = np.array([1, 1, 0, 0, 1] * 4)
        y_pred = np.array([1, 0, 0, 0, 1] * 4)
        
        sex_attr = np.array(['male'] * 10 + ['female'] * 10)
        age_attr = np.array(['young'] * 5 + ['old'] * 5 + ['young'] * 5 + ['old'] * 5)
        
        result = self.metrics.intersectional_analysis(y_true, y_pred, [sex_attr, age_attr])

        assert 'equal_opportunity' in result
        assert 'equalized_odds' in result
        assert 'demographic_parity' in result
        assert 'group_sizes' in result

        expected_groups = ['male_young', 'male_old', 'female_young', 'female_old']
        for group in expected_groups:
            assert group in result['group_sizes']
            assert result['group_sizes'][group] == n_per_group
    
    def test_intersectional_analysis_invalid_attrs(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 0])
        protected_attrs = [np.array(['A', 'B', 'A'])]
        
        with pytest.raises(ValueError, match="exactly 2 protected attributes"):
            self.metrics.intersectional_analysis(y_true, y_pred, protected_attrs)


class TestBootstrapCI:
    def setup_method(self):
        self.bootstrap = BootstrapCI(confidence_level=0.95, n_jobs=1)
        self.metrics = FairnessMetrics(confidence_level=0.95, n_jobs=1)
    
    def test_bootstrap_ci_mean(self):
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        
        lower, upper = self.bootstrap.compute_ci(data, np.mean, n_bootstrap=100)
        assert lower < 10 < upper

        width = upper - lower
        assert 0.1 < width < 2.0
    
    def test_bootstrap_ci_coverage_properties(self):
        np.random.seed(123)

        true_mean = 5.0
        coverage_count = 0
        n_trials = 50
        
        for _ in range(n_trials):
            data = np.random.normal(true_mean, 1, 100)
            lower, upper = self.bootstrap.compute_ci(data, np.mean, n_bootstrap=50)
            
            if lower <= true_mean <= upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_trials

        assert 0.8 <= coverage_rate <= 1.0
    
    def test_bootstrap_ci_empty_data(self):
        data = np.array([])
        
        with pytest.raises(ValueError):
            self.bootstrap.compute_ci(data, np.mean, n_bootstrap=10)
    
    def test_fairness_metrics_bootstrap_integration(self):
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0,
                          1, 1, 0, 0, 0, 0, 0, 0]) 
        
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0,
                          1, 0, 0, 0, 0, 0, 0, 0])
        
        protected_attr = np.array(['A'] * 8 + ['B'] * 8)

        lower, upper = self.metrics.bootstrap_confidence_intervals(
            self.metrics.compute_equal_opportunity,
            n_bootstrap=50,
            y_true=y_true,
            y_pred=y_pred,
            protected_attr=protected_attr,
            metric_key='equal_opportunity_gap'
        )

        expected_gap = 0.25

        assert lower <= expected_gap <= upper

        assert 0 <= lower <= upper <= 1


if __name__ == "__main__":
    pytest.main([__file__])