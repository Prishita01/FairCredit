import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from fair_credit.models.xgboost_model import XGBoostModel

class TestXGBoostModel:
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
    def small_data(self):
        np.random.seed(42)
        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_informative=4,
            n_redundant=1,
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
        model = XGBoostModel()
        assert model.hyperparameters['random_state'] == 42
        assert model.hyperparameters['n_estimators'] == 100
        assert model.hyperparameters['tune_hyperparameters'] is True
        assert model.hyperparameters['calibrate'] is True
        assert model.hyperparameters['early_stopping_rounds'] == 10
        assert not model.is_fitted
        
        model_custom = XGBoostModel(
            random_state=123,
            n_estimators=50,
            tune_hyperparameters=False,
            calibrate=False,
            early_stopping_rounds=5
        )
        assert model_custom.hyperparameters['random_state'] == 123
        assert model_custom.hyperparameters['n_estimators'] == 50
        assert model_custom.hyperparameters['tune_hyperparameters'] is False
        assert model_custom.hyperparameters['calibrate'] is False
        assert model_custom.hyperparameters['early_stopping_rounds'] == 5
    
    def test_fit_with_default_params(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False,
            calibrate=False,
            n_estimators=50
        )
        
        result = model.fit(X_train, y_train)
        
        assert result is model
        assert model.is_fitted
        assert model.feature_names == list(X_train.columns)
        assert model.base_model is not None
        assert model.calibrated_model is None
        assert model.model is model.base_model
    
    def test_fit_with_small_dataset(self, small_data):
        X, y = small_data
        
        model = XGBoostModel(
            tune_hyperparameters=False,
            calibrate=False,
            n_estimators=20
        )
        
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.base_model is not None
    
    def test_fit_with_hyperparameter_tuning(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=True,
            calibrate=False,
            cv_folds=3
        )
        
        original_tune = model._tune_hyperparameters
        
        def fast_tune(X, y, sample_weight=None):
            import xgboost as xgb
            from sklearn.model_selection import GridSearchCV
            
            param_grid = {
                'n_estimators': [20, 50],
                'max_depth': [3, 4],
                'learning_rate': [0.1, 0.2]
            }
            
            base_estimator = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            grid_search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            fit_params = {}
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
                
            grid_search.fit(X, y, **fit_params)
            return grid_search.best_estimator_
        
        model._tune_hyperparameters = fast_tune
        
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        assert model.base_model is not None
        assert hasattr(model.base_model, 'n_estimators')
    
    def test_fit_with_calibration(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False,
            calibrate=True,
            cv_folds=3,
            n_estimators=50
        )
        
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        assert model.base_model is not None
        assert model.calibrated_model is not None
        assert model.model is model.calibrated_model
    
    def test_fit_with_sample_weights(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False,
            calibrate=False,
            n_estimators=50
        )
        
        sample_weights = np.where(y_train == 1, 2.0, 1.0)
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        assert model.is_fitted
        assert model.base_model is not None
    
    def test_predict_proba(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=False,
            n_estimators=50
        )
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-10)
    
    def test_predict_proba_unfitted_model(self, sample_data):
        X, y = sample_data
        model = XGBoostModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)
    
    def test_predict_binary(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=False,
            n_estimators=50
        )
        model.fit(X_train, y_train)
        
        predictions_default = model.predict(X_test)
        predictions_low = model.predict(X_test, threshold=0.3)
        predictions_high = model.predict(X_test, threshold=0.7)
        
        assert predictions_default.shape == (len(X_test),)
        assert np.all(np.isin(predictions_default, [0, 1]))
        
        assert np.sum(predictions_low) >= np.sum(predictions_high)
    
    def test_get_feature_importance(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=False,
            n_estimators=50
        )
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == len(X_train.columns)
        assert np.all(importance >= 0)
    
    def test_get_feature_importance_dict(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=False,
            n_estimators=50
        )
        model.fit(X_train, y_train)
        
        importance_dict = model.get_feature_importance_dict()
        
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == len(X_train.columns)
        assert all(feature in importance_dict for feature in X_train.columns)
        assert all(isinstance(imp, (int, float)) for imp in importance_dict.values())
    
    def test_get_feature_importance_unfitted_model(self):
        model = XGBoostModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_feature_importance()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_feature_importance_dict()
    
    def test_get_booster_info(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=False,
            n_estimators=50
        )
        model.fit(X_train, y_train)
        
        booster_info = model.get_booster_info()
        
        assert isinstance(booster_info, dict)
        assert len(booster_info) > 0
    
    def test_get_booster_info_unfitted_model(self):
        model = XGBoostModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_booster_info()
    
    def test_model_reproducibility(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model1 = XGBoostModel(
            random_state=42,
            tune_hyperparameters=False,
            calibrate=False,
            n_estimators=50
        )
        model2 = XGBoostModel(
            random_state=42,
            tune_hyperparameters=False,
            calibrate=False,
            n_estimators=50
        )
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        probas1 = model1.predict_proba(X_test)
        probas2 = model2.predict_proba(X_test)
        
        assert np.allclose(probas1, probas2, atol=1e-10)
    
    def test_model_performance_reasonable(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=False,
            n_estimators=100
        )
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)
        predictions = model.predict(X_test)
        
        accuracy = np.mean(predictions == y_test)
        
        assert accuracy > 0.7, f"Model accuracy {accuracy} is too low"
        
        prob_positive = probas[:, 1]
        assert np.mean((prob_positive > 0.05) & (prob_positive < 0.95)) > 0.1
    
    def test_get_model_info(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=True,
            n_estimators=50
        )
        
        info_before = model.get_model_info()
        assert info_before['is_fitted'] is False
        
        model.fit(X_train, y_train)
        info_after = model.get_model_info()
        
        assert info_after['is_fitted'] is True
        assert info_after['model_type'] == 'XGBoostModel'
        assert info_after['is_calibrated'] is True
        assert info_after['n_features'] == len(X_train.columns)
        assert 'best_params' in info_after
        assert 'booster_info' in info_after
    
    def test_get_tree_info(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=False,
            n_estimators=20
        )
        model.fit(X_train, y_train)
        
        tree_info = model.get_tree_info()
        
        assert isinstance(tree_info, dict)
        assert 'num_trees' in tree_info
        assert tree_info['num_trees'] > 0
    
    def test_get_tree_info_unfitted_model(self):
        model = XGBoostModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_tree_info()
    
    def test_input_validation(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False, 
            calibrate=False,
            n_estimators=50
        )
        model.fit(X_train, y_train)
        
        with pytest.raises(TypeError, match="Input X must be a pandas DataFrame"):
            model.predict_proba(X_test.values)
        
        X_missing = X_test.drop(columns=['feature_0'])
        with pytest.raises(ValueError, match="Missing features"):
            model.predict_proba(X_missing)
        
        X_nan = X_test.copy()
        X_nan.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input data contains missing values"):
            model.predict_proba(X_nan)
    
    def test_early_stopping_behavior(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        
        model = XGBoostModel(
            tune_hyperparameters=False,
            calibrate=False,
            n_estimators=1000,
            early_stopping_rounds=5
        )
        
        model.fit(X_train, y_train)
        
        booster_info = model.get_booster_info()
        if 'num_boosted_rounds' in booster_info:
            assert booster_info['num_boosted_rounds'] < 1000


if __name__ == "__main__":
    pytest.main([__file__])