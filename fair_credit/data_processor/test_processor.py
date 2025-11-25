"""Unit tests for German Credit dataset loader and validator."""

import unittest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, mock_open
import tempfile
import os

from .processor import GermanCreditLoader, ProtectedAttributeEncoder, StratifiedSplitter


class TestGermanCreditLoader(unittest.TestCase):
    
    def setUp(self):
        self.loader = GermanCreditLoader()
        
    def test_init(self):
        self.assertIsInstance(self.loader, GermanCreditLoader)
        self.assertIsNone(self.loader.dataset_info)
        self.assertFalse(self.loader.is_fitted)
        
    def test_load_sample_dataset(self):
        df = self.loader.load_german_credit()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)  
        
        expected_cols = set(self.loader.EXPECTED_FEATURES.keys())
        actual_cols = set(df.columns)
        self.assertEqual(expected_cols, actual_cols)
        
        self.assertTrue(df['default'].isin([0, 1]).all())
        
        numeric_cols = ['duration', 'credit_amount', 'installment_rate', 
                       'residence_since', 'age', 'existing_credits', 'num_dependents']
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))
            
    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_german_credit("nonexistent_file.csv")
            
    def test_generate_sample_dataset_size(self):
        df = self.loader._generate_sample_dataset(n_samples=500)
        self.assertEqual(len(df), 500)
        
    def test_generate_sample_dataset_reproducibility(self):
        df1 = self.loader._generate_sample_dataset(n_samples=100)
        df2 = self.loader._generate_sample_dataset(n_samples=100)
        
        pd.testing.assert_frame_equal(df1, df2)
        
    def test_preprocess_dataset_target_conversion(self):
        test_data = pd.DataFrame({
            'default': [1, 2, 1, 2, 1],
            'age': [25, 30, 35, 40, 45],
            'duration': [12, 24, 36, 48, 60]
        })
        
        processed = self.loader._preprocess_dataset(test_data)

        expected_default = [1, 0, 1, 0, 1]
        self.assertEqual(processed['default'].tolist(), expected_default)
        
    def test_validate_dataset_success(self):
        df = self.loader.load_german_credit()
        self.assertTrue(self.loader.validate_dataset(df))
        
    def test_validate_dataset_missing_columns(self):
        df = pd.DataFrame({'age': [25, 30], 'duration': [12, 24]})
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_dataset(df)
        self.assertIn("Missing required columns", str(context.exception))
        
    def test_validate_dataset_invalid_target(self):
        df = pd.DataFrame({
            'default': [0, 1, 2],  # Invalid value: 2
            'age': [25, 30, 35]
        })
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_dataset(df)
        self.assertIn("must contain only 0 and 1 values", str(context.exception))
        
    def test_validate_dataset_empty(self):
        df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_dataset(df)
        self.assertIn("Dataset is empty", str(context.exception))
        
    def test_encode_protected_attributes_success(self):
        df = pd.DataFrame({
            'personal_status': ['male single', 'female div/dep/mar', 'male mar/wid', 'female single'],
            'age': [22, 35, 45, 20],
            'default': [0, 1, 0, 1]
        })
        
        encoded_df = self.loader.encode_protected_attributes(df)
        
        
        expected_sex = [1, 0, 1, 0]
        self.assertEqual(encoded_df['sex'].tolist(), expected_sex)
        
        expected_age_group = [1, 0, 0, 1]
        self.assertEqual(encoded_df['age_group'].tolist(), expected_age_group)
        
    def test_encode_protected_attributes_missing_columns(self):
     
        df = pd.DataFrame({'default': [0, 1]})
        
        with self.assertRaises(ValueError) as context:
            self.loader.encode_protected_attributes(df)
        self.assertIn("personal_status", str(context.exception))
        
    def test_create_splits_success(self):

        df = self.loader.load_german_credit()
        df = self.loader.encode_protected_attributes(df)
        
        train_df, val_df, test_df = self.loader.create_splits(df)
        
        
        total_size = len(df)
        self.assertAlmostEqual(len(test_df) / total_size, 0.2, delta=0.05)
        self.assertAlmostEqual(len(val_df) / total_size, 0.2, delta=0.05)
        self.assertAlmostEqual(len(train_df) / total_size, 0.6, delta=0.05)
        
       
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        self.assertEqual(len(train_indices & val_indices), 0)
        self.assertEqual(len(train_indices & test_indices), 0)
        self.assertEqual(len(val_indices & test_indices), 0)
        
      
        total_split_size = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_split_size, total_size)
        
    def test_create_splits_missing_protected_attributes(self):
     
        df = pd.DataFrame({
            'default': [0, 1, 0, 1],
            'age': [25, 30, 35, 40]
        })
        
        with self.assertRaises(ValueError) as context:
            self.loader.create_splits(df)
        self.assertIn("Protected attributes must be encoded", str(context.exception))
        
    def test_get_feature_info(self):
       
        df = self.loader.load_german_credit()
        df = self.loader.encode_protected_attributes(df)
        
        info = self.loader.get_feature_info(df)
        
      
        self.assertIn('n_samples', info)
        self.assertIn('n_features', info)
        self.assertIn('feature_names', info)
        self.assertIn('target_distribution', info)
        

        self.assertEqual(info['n_samples'], 1000)
        self.assertGreater(info['n_features'], 20)  
        self.assertIn('sex_distribution', info)
        self.assertIn('age_group_distribution', info)


class TestProtectedAttributeEncoder(unittest.TestCase):

    
    def setUp(self):

        self.encoder = ProtectedAttributeEncoder()
        self.sample_data = pd.DataFrame({
            'personal_status': ['male single', 'female div/dep/mar', 'male mar/wid', 'female single'],
            'age': [22, 35, 45, 20]
        })
        
    def test_init(self):
        self.assertFalse(self.encoder.is_fitted)
        self.assertIsNone(self.encoder.sex_mapping)
        self.assertEqual(self.encoder.age_threshold, 25)
        
    def test_fit_transform(self):
        result = self.encoder.fit_transform(self.sample_data)
        
        self.assertTrue(self.encoder.is_fitted)
        
        expected_sex = [1, 0, 1, 0]  # male=1, female=0
        expected_age_group = [1, 0, 0, 1]  # young (<25)=1, not_young (>=25)=0
        
        self.assertEqual(result['sex'].tolist(), expected_sex)
        self.assertEqual(result['age_group'].tolist(), expected_age_group)
        
    def test_fit_then_transform(self):


        self.encoder.fit(self.sample_data)
        self.assertTrue(self.encoder.is_fitted)
        

        result = self.encoder.transform(self.sample_data)
        
        expected_sex = [1, 0, 1, 0]
        expected_age_group = [1, 0, 0, 1]
        
        self.assertEqual(result['sex'].tolist(), expected_sex)
        self.assertEqual(result['age_group'].tolist(), expected_age_group)
        
    def test_transform_without_fit_raises_error(self):
        with self.assertRaises(ValueError) as context:
            self.encoder.transform(self.sample_data)
        self.assertIn("must be fitted", str(context.exception))
        
    def test_missing_columns_raises_error(self):
        invalid_data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with self.assertRaises(ValueError) as context:
            self.encoder.fit_transform(invalid_data)
        self.assertIn("Missing required columns", str(context.exception))
        
    def test_null_values_raise_error(self):
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[0, 'age'] = None
        
        with self.assertRaises(ValueError) as context:
            self.encoder.fit_transform(data_with_nulls)
        self.assertIn("null values", str(context.exception))
        
    def test_sex_mapping_creation(self):
        personal_status_values = ['male single', 'female div/dep/mar', 'male mar/wid']
        mapping = self.encoder._create_sex_mapping(personal_status_values)
        
        self.assertEqual(mapping['male single'], 1)
        self.assertEqual(mapping['female div/dep/mar'], 0)
        self.assertEqual(mapping['male mar/wid'], 1)
        
    def test_unknown_personal_status_raises_error(self):
        data_with_unknown = pd.DataFrame({
            'personal_status': ['unknown status'],
            'age': [25]
        })
        
        with self.assertRaises(ValueError) as context:
            self.encoder.fit_transform(data_with_unknown)
        self.assertIn("Cannot determine sex", str(context.exception))


class TestStratifiedSplitter(unittest.TestCase):
    
    def setUp(self):
        self.splitter = StratifiedSplitter()
        
        np.random.seed(42)
        n_samples = 200
        self.sample_data = pd.DataFrame({
            'default': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'sex': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'age_group': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        
    def test_init_default_params(self):
        self.assertEqual(self.splitter.test_size, 0.2)
        self.assertEqual(self.splitter.val_size, 0.2)
        self.assertEqual(self.splitter.random_state, 42)
        
    def test_init_custom_params(self):
        splitter = StratifiedSplitter(test_size=0.3, val_size=0.15, random_state=123)
        self.assertEqual(splitter.test_size, 0.3)
        self.assertEqual(splitter.val_size, 0.15)
        self.assertEqual(splitter.random_state, 123)
        
    def test_init_invalid_params(self):
        with self.assertRaises(ValueError):
            StratifiedSplitter(test_size=1.5)  # > 1
            
        with self.assertRaises(ValueError):
            StratifiedSplitter(test_size=0.0)  # = 0
            
        with self.assertRaises(ValueError):
            StratifiedSplitter(test_size=0.6, val_size=0.5)  # sum >= 1
            
    def test_split_success(self):
        train_df, val_df, test_df = self.splitter.split(self.sample_data)
        
        total_size = len(self.sample_data)
        test_prop = len(test_df) / total_size
        val_prop = len(val_df) / total_size
        train_prop = len(train_df) / total_size
        
        self.assertAlmostEqual(test_prop, 0.2, delta=0.05)
        self.assertAlmostEqual(val_prop, 0.2, delta=0.05)
        self.assertAlmostEqual(train_prop, 0.6, delta=0.05)
        
        all_indices = set(train_df.index) | set(val_df.index) | set(test_df.index)
        self.assertEqual(len(all_indices), total_size)
        
        for split_df in [train_df, val_df, test_df]:
            self.assertEqual(len(split_df['default'].unique()), 2)
            self.assertEqual(len(split_df['sex'].unique()), 2)
            
    def test_split_missing_columns(self):
        invalid_data = pd.DataFrame({
            'default': [0, 1, 0, 1],
            'feature1': [1, 2, 3, 4]
        })
        
        with self.assertRaises(ValueError) as context:
            self.splitter.split(invalid_data)
        self.assertIn("Missing required columns", str(context.exception))
        
    def test_split_invalid_values(self):
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'default'] = 2 
        
        with self.assertRaises(ValueError) as context:
            self.splitter.split(invalid_data)
        self.assertIn("must contain only 0 and 1 values", str(context.exception))
        
    def test_split_reproducibility(self):
        train1, val1, test1 = self.splitter.split(self.sample_data)
        train2, val2, test2 = self.splitter.split(self.sample_data)
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)
        
    def test_split_small_groups_fallback(self):
        small_data = pd.DataFrame({
            'default': [0, 0, 0, 1],  # Very imbalanced
            'sex': [0, 0, 1, 1],
            'age_group': [0, 1, 0, 1],
            'feature1': [1, 2, 3, 4]
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_df, val_df, test_df = self.splitter.split(small_data)

        total_split_size = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_split_size, len(small_data))


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.loader = GermanCreditLoader()
        
    def test_full_pipeline(self):
        df = self.loader.load_german_credit()
        self.assertIsInstance(df, pd.DataFrame)
        
        df_encoded = self.loader.encode_protected_attributes(df)
        self.assertIn('sex', df_encoded.columns)
        self.assertIn('age_group', df_encoded.columns)
        
        train_df, val_df, test_df = self.loader.create_splits(df_encoded)
        
        total_original = len(df)
        total_splits = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_original, total_splits)
        
        required_cols = ['default', 'sex', 'age_group']
        for split_df in [train_df, val_df, test_df]:
            for col in required_cols:
                self.assertIn(col, split_df.columns)
                
    def test_pipeline_with_standalone_components(self):
        df = self.loader.load_german_credit()
        
        encoder = ProtectedAttributeEncoder()
        df_encoded = encoder.fit_transform(df)
        
        splitter = StratifiedSplitter()
        train_df, val_df, test_df = splitter.split(df_encoded)
        
        self.assertIn('sex', df_encoded.columns)
        self.assertIn('age_group', df_encoded.columns)
        
        total_original = len(df)
        total_splits = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_original, total_splits)


if __name__ == '__main__':
    unittest.main(verbosity=2)