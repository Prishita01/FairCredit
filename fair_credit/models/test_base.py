import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from fair_credit.models.base import BaselineModel


class MockModel(BaselineModel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mock_model = Mock()
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray = None) -> 'MockModel':
        self._validate_input(X)
        self.feature_names = list(X.columns)
        self.model = self.mock_model
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        self._validate_input(X)
        # Return mock probabilities that sum to 1
        n_samples = len(X)
        np.random.seed(42)
        class_1_probs = np.random.random(n_samples)
        class_0_probs = 1 - class_1_probs
        return np.column_stack([class_0_probs, class_1_probs])


class TestBaselineModel:
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randint(0, 5, 100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y
    
    @pytest.fixture
    def fitted_model(self, sample_data):
        X, y = sample_data
        model = MockModel(param1=1.0, param2='test')
        model.fit(X, y)
        return model
    
    def test_model_initialization(self):
        model = MockModel(param1=1.0, param2='test')
        
        assert model.model is None
        assert not model.is_fitted
        assert model.feature_names is None
        assert model.hyperparameters == {'param1': 1.0, 'param2': 'test'}
    
    def test_fit_method(self, sample_data):
        X, y = sample_data
        model = MockModel()
        
        # Test fitting
        result = model.fit(X, y)
        
        assert result is model  # Should return self
        assert model.is_fitted
        assert model.feature_names == list(X.columns)
    
    def test_fit_with_sample_weights(self, sample_data):
        X, y = sample_data
        model = MockModel()
        weights = np.random.random(len(X))
        
        result = model.fit(X, y, sample_weight=weights)
        
        assert result is model
        assert model.is_fitted
    
    def test_predict_proba_unfitted_model(self, sample_data):
        X, y = sample_data
        model = MockModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)
    
    def test_predict_proba_fitted_model(self, fitted_model, sample_data):
        X, y = sample_data
        
        probas = fitted_model.predict_proba(X)
        
        assert probas.shape == (len(X), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-10)
    
    def test_predict_method(self, fitted_model, sample_data):
        X, y = sample_data
        
        # Test with default threshold
        predictions = fitted_model.predict(X)
        assert predictions.shape == (len(X),)
        assert np.all(np.isin(predictions, [0, 1]))
        
        predictions_low = fitted_model.predict(X, threshold=0.1)
        predictions_high = fitted_model.predict(X, threshold=0.9)

        assert np.sum(predictions_low) >= np.sum(predictions_high)
    
    def test_predict_unfitted_model(self, sample_data):
        X, y = sample_data
        model = MockModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)
    
    def test_get_feature_importance_unfitted(self):
        model = MockModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_feature_importance()
    
    def test_get_feature_importance_with_feature_importances(self, fitted_model):
        # Mock the model to have feature_importances_
        fitted_model.model.feature_importances_ = np.array([0.1, 0.3, 0.6])
        
        importance = fitted_model.get_feature_importance()
        
        assert np.array_equal(importance, np.array([0.1, 0.3, 0.6]))
    
    def test_get_feature_importance_with_coef(self, fitted_model):
        # Mock the model to have coef_ but not feature_importances_
        fitted_model.model.coef_ = np.array([[-0.2, 0.5, -0.8]])
        delattr(fitted_model.model, 'feature_importances_') if hasattr(fitted_model.model, 'feature_importances_') else None
        
        importance = fitted_model.get_feature_importance()
        
        expected = np.abs(np.array([-0.2, 0.5, -0.8]))
        assert np.array_equal(importance, expected)
    
    def test_get_feature_importance_none(self, fitted_model):
        # Ensure model has neither feature_importances_ nor coef_
        if hasattr(fitted_model.model, 'feature_importances_'):
            delattr(fitted_model.model, 'feature_importances_')
        if hasattr(fitted_model.model, 'coef_'):
            delattr(fitted_model.model, 'coef_')
        
        importance = fitted_model.get_feature_importance()
        
        assert importance is None
    
    def test_save_and_load_model(self, sample_data):
        # Create a model with a simple serializable mock
        X, y = sample_data
        model = MockModel(param1=1.0, param2='test')
        
        # Replace the mock with a simple dict for serialization
        model.fit(X, y)
        model.model = {'type': 'mock', 'fitted': True}  # Simple serializable object
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_model.pkl'
            
            # Save model
            model.save_model(str(filepath))
            assert filepath.exists()
            
            # Load model
            loaded_model = MockModel.load_model(str(filepath))
            
            assert loaded_model.is_fitted
            assert loaded_model.feature_names == model.feature_names
            assert loaded_model.hyperparameters == model.hyperparameters
            assert loaded_model.__class__.__name__ == model.__class__.__name__
    
    def test_save_unfitted_model(self):
        model = MockModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_model.pkl'
            
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                model.save_model(str(filepath))
    
    def test_get_model_info(self, fitted_model):
        info = fitted_model.get_model_info()
        
        expected_keys = ['model_type', 'is_fitted', 'feature_names', 'hyperparameters', 'n_features']
        assert all(key in info for key in expected_keys)
        
        assert info['model_type'] == 'MockModel'
        assert info['is_fitted'] is True
        assert info['feature_names'] == fitted_model.feature_names
        assert info['n_features'] == len(fitted_model.feature_names)
    
    def test_validate_input_wrong_type(self, fitted_model):
        with pytest.raises(TypeError, match="Input X must be a pandas DataFrame"):
            fitted_model.predict_proba(np.array([[1, 2, 3]]))
    
    def test_validate_input_missing_features(self, fitted_model):
        X_missing = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing features"):
            fitted_model.predict_proba(X_missing)
    
    def test_validate_input_extra_features(self, fitted_model, sample_data):
        X, _ = sample_data
        X_extra = X.copy()
        X_extra['extra_feature'] = np.random.randn(len(X))
        
        probas = fitted_model.predict_proba(X_extra)
        assert probas.shape == (len(X), 2)
    
    def test_validate_input_missing_values(self, fitted_model, sample_data):
        X, _ = sample_data
        X_missing = X.copy()
        X_missing.iloc[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="Input data contains missing values"):
            fitted_model.predict_proba(X_missing)
    
    def test_repr_method(self):
        model = MockModel()
        assert repr(model) == "MockModel(unfitted)"
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        model.fit(X, y)
        
        assert repr(model) == "MockModel(fitted)"

if __name__ == "__main__":
    pytest.main([__file__])