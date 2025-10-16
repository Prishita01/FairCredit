import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from fair_credit.models.metrics import ModelMetrics, ModelEvaluator


class TestModelMetrics:
    
    def test_model_metrics_creation(self):
        cm = np.array([[50, 10], [5, 35]])
        
        metrics = ModelMetrics(
            auc=0.85,
            auprc=0.80,
            f1_score=0.75,
            brier_score=0.15,
            log_loss=0.45,
            accuracy=0.85,
            precision=0.78,
            recall=0.87,
            specificity=0.83,
            confusion_matrix=cm,
            threshold=0.5,
            n_samples=100
        )
        
        assert metrics.auc == 0.85
        assert metrics.auprc == 0.80
        assert metrics.f1_score == 0.75
        assert metrics.threshold == 0.5
        assert metrics.n_samples == 100
        assert np.array_equal(metrics.confusion_matrix, cm)
    
    def test_model_metrics_to_dict(self):
        cm = np.array([[50, 10], [5, 35]])
        
        metrics = ModelMetrics(
            auc=0.85,
            auprc=0.80,
            f1_score=0.75,
            brier_score=0.15,
            log_loss=0.45,
            accuracy=0.85,
            precision=0.78,
            recall=0.87,
            specificity=0.83,
            confusion_matrix=cm,
            threshold=0.5,
            n_samples=100
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['auc'] == 0.85
        assert metrics_dict['f1_score'] == 0.75
        assert metrics_dict['threshold'] == 0.5
        assert 'confusion_matrix' not in metrics_dict  # Not included in to_dict
    
    def test_model_metrics_str_representation(self):
        cm = np.array([[50, 10], [5, 35]])
        
        metrics = ModelMetrics(
            auc=0.85,
            auprc=0.80,
            f1_score=0.75,
            brier_score=0.15,
            log_loss=0.45,
            accuracy=0.85,
            precision=0.78,
            recall=0.87,
            specificity=0.83,
            confusion_matrix=cm,
            threshold=0.5,
            n_samples=100
        )
        
        str_repr = str(metrics)
        assert "ModelMetrics(" in str_repr
        assert "AUC: 0.8500" in str_repr
        assert "F1: 0.7500" in str_repr


class TestModelEvaluator:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            class_sep=1.0,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train a simple model to get probabilities
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        return y_test, y_proba, y_pred
    
    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator(random_state=42)
    
    def test_evaluator_initialization(self):
        evaluator = ModelEvaluator(random_state=123)
        assert evaluator.random_state == 123
    
    def test_compute_metrics_basic(self, evaluator, sample_data):
        y_true, y_proba, y_pred = sample_data
        
        metrics = evaluator.compute_metrics(y_true, y_proba, y_pred)
        
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.auc <= 1
        assert 0 <= metrics.auprc <= 1
        assert 0 <= metrics.f1_score <= 1
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.specificity <= 1
        assert metrics.brier_score >= 0
        assert metrics.log_loss >= 0
        assert metrics.n_samples == len(y_true)
    
    def test_compute_metrics_without_y_pred(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        metrics = evaluator.compute_metrics(y_true, y_proba, threshold=0.6)
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.threshold == 0.6
        assert metrics.n_samples == len(y_true)
    
    def test_compute_metrics_input_validation(self, evaluator):
        with pytest.raises(ValueError, match="must have the same length"):
            evaluator.compute_metrics(np.array([0, 1]), np.array([0.5]))
        
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            evaluator.compute_metrics(np.array([0, 2]), np.array([0.5, 0.7]))
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            evaluator.compute_metrics(np.array([0, 1]), np.array([0.5, 1.5]))
    
    def test_compute_metrics_edge_cases(self, evaluator):
        y_true = np.array([1, 1, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6])
        
        metrics = evaluator.compute_metrics(y_true, y_proba)
        assert metrics.recall == 1.0
        assert metrics.specificity == 0.0
        
        y_true = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        
        metrics = evaluator.compute_metrics(y_true, y_proba)
        assert metrics.specificity == 1.0
        assert metrics.recall == 0.0
    
    def test_calibration_metrics_computation(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        metrics = evaluator.compute_metrics(y_true, y_proba)
        
        assert metrics.calibration_slope is not None
        assert metrics.calibration_intercept is not None
        assert isinstance(metrics.calibration_slope, float)
        assert isinstance(metrics.calibration_intercept, float)
    
    def test_plot_roc_curve(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        fig = evaluator.plot_roc_curve(y_true, y_proba, title="Test ROC")
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'False Positive Rate'
        assert ax.get_ylabel() == 'True Positive Rate'
        assert "Test ROC" in ax.get_title()
        
        plt.close(fig)
    
    def test_plot_precision_recall_curve(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        fig = evaluator.plot_precision_recall_curve(y_true, y_proba, title="Test PR")
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Recall'
        assert ax.get_ylabel() == 'Precision'
        assert "Test PR" in ax.get_title()
        
        plt.close(fig)
    
    def test_plot_calibration_curve(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        fig = evaluator.plot_calibration_curve(y_true, y_proba, title="Test Calibration")
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Mean Predicted Probability'
        assert ax.get_ylabel() == 'Fraction of Positives'
        assert "Test Calibration" in ax.get_title()
        
        plt.close(fig)
    
    def test_plot_confusion_matrix(self, evaluator, sample_data):
        y_true, y_proba, y_pred = sample_data
        
        fig = evaluator.plot_confusion_matrix(y_true, y_pred, title="Test CM")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig = evaluator.plot_confusion_matrix(y_true, y_pred, normalize=True, title="Test CM Norm")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_threshold_analysis(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        fig = evaluator.plot_threshold_analysis(y_true, y_proba, title="Test Threshold")
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplots
        
        plt.close(fig)
    
    def test_compare_models(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        model_results = {
            'Model A': {'y_true': y_true, 'y_proba': y_proba},
            'Model B': {'y_true': y_true, 'y_proba': y_proba * 0.9 + 0.05},
            'Model C': {'y_true': y_true, 'y_proba': y_proba * 0.8 + 0.1}
        }
        
        fig = evaluator.compare_models(model_results, metrics_to_plot=['auc', 'f1_score'])
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # At least 2 subplots for the metrics
        
        plt.close(fig)
    
    def test_generate_evaluation_report(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        report = evaluator.generate_evaluation_report(
            y_true, y_proba, 
            model_name="Test Model",
            save_plots=False
        )
        
        assert 'metrics' in report
        assert 'plots' in report
        assert 'summary' in report
        
        assert isinstance(report['metrics'], ModelMetrics)
        
        expected_plots = ['roc_curve', 'pr_curve', 'calibration', 'confusion_matrix', 'threshold_analysis']
        for plot_name in expected_plots:
            assert plot_name in report['plots']
            assert isinstance(report['plots'][plot_name], plt.Figure)
        
        summary = report['summary']
        assert summary['model_name'] == "Test Model"
        assert summary['n_samples'] == len(y_true)
        assert 'positive_rate' in summary
        assert 'key_metrics' in summary
        
        for fig in report['plots'].values():
            plt.close(fig)
    
    def test_perfect_classifier(self, evaluator):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.0, 0.1, 0.9, 1.0, 0.0, 0.8])
        
        metrics = evaluator.compute_metrics(y_true, y_proba, threshold=0.5)
        
        assert metrics.auc == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.specificity == 1.0
    
    def test_random_classifier(self, evaluator):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_proba = np.random.random(1000)
        
        metrics = evaluator.compute_metrics(y_true, y_proba)
        
        assert 0.4 <= metrics.auc <= 0.6
        
        assert 0.3 <= metrics.accuracy <= 0.7
    
    def test_metrics_consistency(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data

        metrics1 = evaluator.compute_metrics(y_true, y_proba)
        metrics2 = evaluator.compute_metrics(y_true, y_proba)
        
        assert metrics1.auc == metrics2.auc
        assert metrics1.f1_score == metrics2.f1_score
        assert metrics1.accuracy == metrics2.accuracy
        assert np.array_equal(metrics1.confusion_matrix, metrics2.confusion_matrix)
    
    def test_different_thresholds(self, evaluator, sample_data):
        y_true, y_proba, _ = sample_data
        
        metrics_low = evaluator.compute_metrics(y_true, y_proba, threshold=0.3)
        metrics_high = evaluator.compute_metrics(y_true, y_proba, threshold=0.7)
        
        assert abs(metrics_low.auc - metrics_high.auc) < 1e-10
        
        assert metrics_low.threshold == 0.3
        assert metrics_high.threshold == 0.7

        assert metrics_low.recall >= metrics_high.recall or abs(metrics_low.recall - metrics_high.recall) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])