import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from fair_credit.models.logistic_regression import LogisticRegressionModel


class TestLogisticRegressionModel:
    @pytest.fixture
    def sample_data(self):

        np.random.seed(42)
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            class_sep=0.8,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        
        return X_df, y_series
    
    @pytest.fixture
    def train_test_data(self, sample_data):
        X, y = sample_data
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    def test_model_initialization(self):
        model = LogisticRegressionModel()
        assert model.hyperparameters['random_state'] == 42
        assert model.hyperparameters['max_iter'] == 1000
        assert model.hyperparameters['tune_hyperparameters'] is True
        assert model.hyperparameters['calibrate'] is True
        assert not model.is_fitted
        
        model_custom = LogisticRegressionModel(
            random_state=123,
            max_iter=500,
            tune_hyperparameters=False,
            calibrate=False
        )
        assert model_custom.hyperparameters['random_state'] == 123
        assert model_custom.hyperparameters['max_iter'] == 500
        assert model_custom.hyperparameters['tune_hyperparameters'] is False
        assert model_custom.hyperparameters['calibrate'] is False
    
    def test_fit_with_default_params(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(
            tune_hyperparameters=False,
            calibrate=False
        )
        
        result = model.fit(X_train, y_train)
        
        assert result is model  # Should return self
        assert model.is_fitted
        assert model.feature_names == list(X_train.columns)
        assert model.base_model is not None
        assert model.calibrated_model is None
        assert model.model is model.base_model
    
    def test_fit_with_hyperparameter_tuning(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(
            tune_hyperparameters=True,
            calibrate=False,
            cv_folds=3
        )
        
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        assert model.base_model is not None
        assert hasattr(model.base_model, 'C')
    
    def test_fit_with_calibration(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(
            tune_hyperparameters=False,
            calibrate=True,
            cv_folds=3
        )
        
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        assert model.base_model is not None
        assert model.calibrated_model is not None
        assert model.model is model.calibrated_model
    
    def test_fit_with_sample_weights(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(
            tune_hyperparameters=False,
            calibrate=False
        )
        
        sample_weights = np.where(y_train == 1, 2.0, 1.0)
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        assert model.is_fitted
        assert model.base_model is not None
    
    def test_predict_proba(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False, calibrate=False)
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-10)
    
    def test_predict_proba_unfitted_model(self, sample_data):
        X, y = sample_data
        model = LogisticRegressionModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)
    
    def test_predict_binary(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False, calibrate=False)
        model.fit(X_train, y_train)
        
        predictions_default = model.predict(X_test)
        predictions_low = model.predict(X_test, threshold=0.3)
        predictions_high = model.predict(X_test, threshold=0.7)
        
        assert predictions_default.shape == (len(X_test),)
        assert np.all(np.isin(predictions_default, [0, 1]))
        assert np.sum(predictions_low) >= np.sum(predictions_high)
    
    def test_get_coefficients(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False, calibrate=False)
        model.fit(X_train, y_train)
        
        coefficients = model.get_coefficients()
        
        assert isinstance(coefficients, dict)
        assert len(coefficients) == len(X_train.columns)
        assert all(feature in coefficients for feature in X_train.columns)
        assert all(isinstance(coef, (int, float)) for coef in coefficients.values())
    
    def test_get_coefficients_unfitted_model(self):
        model = LogisticRegressionModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_coefficients()
    
    def test_get_feature_importance(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False, calibrate=False)
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == len(X_train.columns)
        assert np.all(importance >= 0)  # Should be absolute values
    
    def test_feature_scaling(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        X_train_scaled = X_train.copy()
        X_train_scaled['feature_0'] *= 1000  # Scale one feature much larger
        
        model = LogisticRegressionModel(tune_hyperparameters=False, calibrate=False)
        model.fit(X_train_scaled, y_train)
        
        probas = model.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
    
    def test_model_reproducibility(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        # Train two models with same random seed
        model1 = LogisticRegressionModel(
            random_state=42,
            tune_hyperparameters=False,
            calibrate=False
        )
        model2 = LogisticRegressionModel(
            random_state=42,
            tune_hyperparameters=False,
            calibrate=False
        )
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        probas1 = model1.predict_proba(X_test)
        probas2 = model2.predict_proba(X_test)
        
        assert np.allclose(probas1, probas2, atol=1e-10)
    
    def test_model_performance_reasonable(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False, calibrate=False)
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)
        predictions = model.predict(X_test)
        
        accuracy = np.mean(predictions == y_test)
        
        assert accuracy > 0.6, f"Model accuracy {accuracy} is too low"
        
        prob_positive = probas[:, 1]
        assert np.mean((prob_positive > 0.05) & (prob_positive < 0.95)) > 0.1
    
    def test_get_model_info(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False, calibrate=True)
        
        info_before = model.get_model_info()
        assert info_before['is_fitted'] is False
        
        model.fit(X_train, y_train)
        info_after = model.get_model_info()
        
        assert info_after['is_fitted'] is True
        assert info_after['model_type'] == 'LogisticRegressionModel'
        assert info_after['is_calibrated'] is True
        assert info_after['n_features'] == len(X_train.columns)
        assert 'best_params' in info_after
    
    def test_input_validation(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False, calibrate=False)
        model.fit(X_train, y_train)
        
        with pytest.raises(TypeError, match="Input X must be a pandas DataFrame"):
            model.predict_proba(X_test.values)  # numpy array instead of DataFrame
        
        X_missing = X_test.drop(columns=['feature_0'])
        with pytest.raises(ValueError, match="Missing features"):
            model.predict_proba(X_missing)
        
        X_nan = X_test.copy()
        X_nan.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input data contains missing values"):
            model.predict_proba(X_nan)


if __name__ == "__main__":
    pytest.main([__file__])