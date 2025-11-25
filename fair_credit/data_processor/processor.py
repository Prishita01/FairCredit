"""Main data processor interface for German Credit dataset."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings


class DataProcessor(ABC):
    
    def __init__(self):
        self.is_fitted = False
        self.feature_names = None
        self.protected_attributes = None
    
    @abstractmethod
    def load_german_credit(self, filepath: Optional[str] = None) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def encode_protected_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def create_splits(self, df: pd.DataFrame, 
                     test_size: float = 0.2, 
                     val_size: float = 0.2,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        if df.empty:
            raise ValueError("Dataset is empty")
        required_columns = ['default']  
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if not df['default'].isin([0, 1]).all():
            raise ValueError("Target variable 'default' must contain only 0 and 1 values")
        
        if df.duplicated().any():
            print("Warning: Dataset contains duplicate rows")
        
        return True
    
    def get_feature_info(self, df: pd.DataFrame) -> dict:

        info = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,  # Exclude target
            'feature_names': [col for col in df.columns if col != 'default'],
            'target_distribution': df['default'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        if 'sex' in df.columns:
            info['sex_distribution'] = df['sex'].value_counts().to_dict()
        if 'age_group' in df.columns:
            info['age_group_distribution'] = df['age_group'].value_counts().to_dict()
        
        return info
    
    def get_protected_groups(self, df: pd.DataFrame) -> dict:
        if 'sex' not in df.columns or 'age_group' not in df.columns:
            raise ValueError("Protected attributes 'sex' and 'age_group' must be present")
        
        groups = {
            'sex_groups': df['sex'].unique().tolist(),
            'age_groups': df['age_group'].unique().tolist()
        }
        
        intersectional = df.groupby(['sex', 'age_group']).size()
        groups['intersectional_groups'] = {
            f"{sex}_{age}": count 
            for (sex, age), count in intersectional.items()
        }
        
        total_samples = len(df)
        for group_name, group_dict in groups.items():
            if isinstance(group_dict, dict):
                for subgroup, count in group_dict.items():
                    groups[f"{group_name}_proportions"] = groups.get(f"{group_name}_proportions", {})
                    groups[f"{group_name}_proportions"][subgroup] = count / total_samples
        
        return groups


class GermanCreditLoader(DataProcessor):

    EXPECTED_FEATURES = {
        'checking_status': 'categorical',
        'duration': 'numeric',
        'credit_history': 'categorical', 
        'purpose': 'categorical',
        'credit_amount': 'numeric',
        'savings_status': 'categorical',
        'employment': 'categorical',
        'installment_rate': 'numeric',
        'personal_status': 'categorical', 
        'other_parties': 'categorical',
        'residence_since': 'numeric',
        'property_magnitude': 'categorical',
        'age': 'numeric',
        'other_payment_plans': 'categorical',
        'housing': 'categorical',
        'existing_credits': 'numeric',
        'job': 'categorical',
        'num_dependents': 'numeric',
        'own_telephone': 'categorical',
        'foreign_worker': 'categorical',
        'default': 'target'  
    }
    
    def __init__(self):
        super().__init__()
        self.dataset_info = None
        
    def load_german_credit(self, filepath: Optional[str] = None) -> pd.DataFrame:
        if filepath is not None:
            try:
                df = pd.read_csv(filepath, sep=' ', header=None)
                df = self._apply_column_names(df)
            except FileNotFoundError:
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
            except Exception as e:
                raise ValueError(f"Error loading dataset from {filepath}: {str(e)}")
        else:
            df = self._generate_sample_dataset()
            
        df = self._preprocess_dataset(df)
        self.validate_dataset(df)
        
        self.dataset_info = self.get_feature_info(df)
        
        return df
    
    def _apply_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_columns = list(self.EXPECTED_FEATURES.keys())
        
        if len(df.columns) != len(expected_columns):
            raise ValueError(f"Expected {len(expected_columns)} columns, got {len(df.columns)}")
            
        df.columns = expected_columns
        return df
    
    def _generate_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:

        np.random.seed(42) 
        
        data = {}
        
        data['checking_status'] = np.random.choice(
            ['<0', '0<=X<200', '>=200', 'no checking'], 
            size=n_samples, p=[0.4, 0.3, 0.2, 0.1]
        )
        
        data['credit_history'] = np.random.choice(
            ['no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit'],
            size=n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]
        )
        
        data['purpose'] = np.random.choice(
            ['new car', 'used car', 'furniture/equipment', 'radio/tv', 'domestic appliance', 
             'repairs', 'education', 'vacation', 'retraining', 'business', 'others'],
            size=n_samples, p=[0.25, 0.15, 0.18, 0.12, 0.06, 0.08, 0.05, 0.03, 0.02, 0.04, 0.02]
        )
        
        data['savings_status'] = np.random.choice(
            ['<100', '100<=X<500', '500<=X<1000', '>=1000', 'no known savings'],
            size=n_samples, p=[0.6, 0.15, 0.1, 0.05, 0.1]
        )
        
        data['employment'] = np.random.choice(
            ['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'],
            size=n_samples, p=[0.05, 0.15, 0.35, 0.25, 0.2]
        )
        
  
        data['personal_status'] = np.random.choice(
            ['male div/sep', 'female div/dep/mar', 'male single', 'male mar/wid', 'female single'],
            size=n_samples, p=[0.05, 0.35, 0.35, 0.15, 0.1]
        )
        
        data['other_parties'] = np.random.choice(
            ['none', 'co applicant', 'guarantor'],
            size=n_samples, p=[0.9, 0.05, 0.05]
        )
        
        data['property_magnitude'] = np.random.choice(
            ['real estate', 'life insurance', 'car', 'no known property'],
            size=n_samples, p=[0.25, 0.25, 0.35, 0.15]
        )
        
        data['other_payment_plans'] = np.random.choice(
            ['bank', 'stores', 'none'],
            size=n_samples, p=[0.15, 0.05, 0.8]
        )
        
        data['housing'] = np.random.choice(
            ['rent', 'own', 'for free'],
            size=n_samples, p=[0.4, 0.5, 0.1]
        )
        
        data['job'] = np.random.choice(
            ['unemp/unskilled non res', 'unskilled resident', 'skilled', 'high qualif/self emp/mgmt'],
            size=n_samples, p=[0.02, 0.18, 0.65, 0.15]
        )
        
        data['own_telephone'] = np.random.choice(
            ['none', 'yes'], size=n_samples, p=[0.4, 0.6]
        )
        
        data['foreign_worker'] = np.random.choice(
            ['yes', 'no'], size=n_samples, p=[0.96, 0.04]
        )
        
        
        data['duration'] = np.random.randint(4, 73, size=n_samples)
        data['credit_amount'] = np.random.randint(250, 18425, size=n_samples)
        data['installment_rate'] = np.random.randint(1, 5, size=n_samples)
        data['residence_since'] = np.random.randint(1, 5, size=n_samples)
        data['age'] = np.random.randint(19, 76, size=n_samples)
        data['existing_credits'] = np.random.randint(1, 5, size=n_samples)
        data['num_dependents'] = np.random.randint(1, 3, size=n_samples)
        
        
        data['default'] = np.random.choice([1, 2], size=n_samples, p=[0.3, 0.7])
        
        return pd.DataFrame(data)
    
    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
 
        df = df.copy()
        
        
        if df['default'].dtype == 'object' or df['default'].max() > 1:
            df['default'] = (df['default'] == 1).astype(int)
        
       
        numeric_columns = [col for col, dtype in self.EXPECTED_FEATURES.items() 
                          if dtype == 'numeric']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.isnull().any().any():
            warnings.warn("Some values could not be converted to numeric. Check data quality.")
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        super().validate_dataset(df)
        
        if len(df) < 100:
            warnings.warn(f"Dataset has only {len(df)} samples. German Credit typically has ~1000.")
        
        expected_cols = set(self.EXPECTED_FEATURES.keys())
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        numeric_validations = {
            'age': (18, 100),
            'duration': (1, 100),
            'credit_amount': (0, 50000),
            'installment_rate': (1, 4),
            'residence_since': (1, 4),
            'existing_credits': (1, 4),
            'num_dependents': (1, 2)
        }
        
        for col, (min_val, max_val) in numeric_validations.items():
            if col in df.columns:
                if df[col].min() < min_val or df[col].max() > max_val:
                    warnings.warn(f"Column {col} has values outside expected range [{min_val}, {max_val}]")
        
        if 'personal_status' in df.columns:
            personal_status_values = df['personal_status'].unique()
            has_male = any('male' in str(val).lower() for val in personal_status_values)
            has_female = any('female' in str(val).lower() for val in personal_status_values)
            
            if not (has_male and has_female):
                warnings.warn("personal_status column may not contain proper sex information")
        
        return True
    
    def encode_protected_attributes(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        
        if 'personal_status' not in df.columns:
            raise ValueError("Column 'personal_status' required for sex encoding")
        

        df['sex'] = df['personal_status'].str.contains(r'\bmale\b', case=False, na=False, regex=True).astype(int)
        
        if 'age' not in df.columns:
            raise ValueError("Column 'age' required for age group encoding")
        
        df['age_group'] = (df['age'] < 25).astype(int)

        self._validate_protected_encoding(df)
        
        return df
    
    def _validate_protected_encoding(self, df: pd.DataFrame) -> None:
        if 'sex' not in df.columns:
            raise ValueError("Sex encoding failed - 'sex' column not created")
        
        if not df['sex'].isin([0, 1]).all():
            raise ValueError("Sex encoding failed - values must be 0 or 1")
        
        if 'age_group' not in df.columns:
            raise ValueError("Age group encoding failed - 'age_group' column not created")
        
        if not df['age_group'].isin([0, 1]).all():
            raise ValueError("Age group encoding failed - values must be 0 or 1")
        
        if len(df['sex'].unique()) < 2:
            warnings.warn("Only one sex group found in dataset")
        
        if len(df['age_group'].unique()) < 2:
            warnings.warn("Only one age group found in dataset")
    
    def create_splits(self, df: pd.DataFrame, 
                     test_size: float = 0.2, 
                     val_size: float = 0.2,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if 'sex' not in df.columns or 'age_group' not in df.columns:
            raise ValueError("Protected attributes must be encoded before splitting")
        
        df = df.copy()
        df['stratify_key'] = (
            df['default'].astype(str) + '_' + 
            df['sex'].astype(str) + '_' + 
            df['age_group'].astype(str)
        )
        
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            stratify=df['stratify_key'],
            random_state=random_state
        )

        adjusted_val_size = val_size / (1 - test_size)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            stratify=train_val_df['stratify_key'],
            random_state=random_state
        )
        
        train_df = train_df.drop('stratify_key', axis=1)
        val_df = val_df.drop('stratify_key', axis=1)
        test_df = test_df.drop('stratify_key', axis=1)
        
        self._validate_splits(train_df, val_df, test_df, df)
        
        return train_df, val_df, test_df
    
    def _validate_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame, original_df: pd.DataFrame) -> None:
        total_samples = len(original_df)
        train_prop = len(train_df) / total_samples
        val_prop = len(val_df) / total_samples  
        test_prop = len(test_df) / total_samples
        
        expected_train = 0.6  
        expected_val = 0.2    
        expected_test = 0.2   
        
        if abs(train_prop - expected_train) > 0.05:
            warnings.warn(f"Train split proportion {train_prop:.3f} differs from expected {expected_train}")
        
        if abs(val_prop - expected_val) > 0.05:
            warnings.warn(f"Validation split proportion {val_prop:.3f} differs from expected {expected_val}")
            
        if abs(test_prop - expected_test) > 0.05:
            warnings.warn(f"Test split proportion {test_prop:.3f} differs from expected {expected_test}")
        
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            for attr in ['sex', 'age_group', 'default']:
                if len(split_df[attr].unique()) < len(original_df[attr].unique()):
                    warnings.warn(f"{split_name} split missing some {attr} groups")


class ProtectedAttributeEncoder:

    def __init__(self):
        self.is_fitted = False
        self.sex_mapping = None
        self.age_threshold = 25
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        return self.fit(df).transform(df)
    
    def fit(self, df: pd.DataFrame) -> 'ProtectedAttributeEncoder':

        self._validate_input(df)
        
        personal_status_values = df['personal_status'].value_counts()
        self.sex_mapping = self._create_sex_mapping(personal_status_values.index)
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        self._validate_input(df)
        df = df.copy()
        
        df['sex'] = df['personal_status'].map(self.sex_mapping).fillna(-1)
        
        if (df['sex'] == -1).any():
            unmapped = df[df['sex'] == -1]['personal_status'].unique()
            raise ValueError(f"Unmapped personal_status values: {unmapped}")
        
        df['sex'] = df['sex'].astype(int)
        
        df['age_group'] = (df['age'] < self.age_threshold).astype(int)
        
        self._validate_encoding(df)
        
        return df
    
    def _validate_input(self, df: pd.DataFrame) -> None:

        required_cols = ['personal_status', 'age']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df['age'].isnull().any():
            raise ValueError("Age column contains null values")
        
        if df['personal_status'].isnull().any():
            raise ValueError("Personal status column contains null values")
    
    def _create_sex_mapping(self, personal_status_values) -> Dict[str, int]:
        mapping = {}
        
        for status in personal_status_values:
            status_lower = str(status).lower()
            if 'female' in status_lower:
                mapping[status] = 0  # female
            elif 'male' in status_lower:
                mapping[status] = 1  # male
            else:
                # Handle edge cases - assume based on common patterns
                if any(word in status_lower for word in ['div/sep', 'single', 'mar/wid']):
                    # These typically refer to male in German Credit dataset
                    mapping[status] = 1
                else:
                    raise ValueError(f"Cannot determine sex from personal_status: {status}")
        
        return mapping
    
    def _validate_encoding(self, df: pd.DataFrame) -> None:
        if not df['sex'].isin([0, 1]).all():
            raise ValueError("Sex encoding produced invalid values")
        
        if not df['age_group'].isin([0, 1]).all():
            raise ValueError("Age group encoding produced invalid values")
        
        sex_counts = df['sex'].value_counts()
        age_counts = df['age_group'].value_counts()
        
        if len(sex_counts) < 2:
            warnings.warn("Only one sex group found after encoding")
        
        if len(age_counts) < 2:
            warnings.warn("Only one age group found after encoding")
        

        if sex_counts.min() / len(df) < 0.1:
            warnings.warn("One sex group has very low representation (<10%)")
        
        if age_counts.min() / len(df) < 0.1:
            warnings.warn("One age group has very low representation (<10%)")


class StratifiedSplitter:

    def __init__(self, test_size: float = 0.2, val_size: float = 0.2, 
                 random_state: int = 42):

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        if not (0 < val_size < 1):
            raise ValueError("val_size must be between 0 and 1")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        self._validate_input(df)
        
        df = df.copy()
        stratify_key = self._create_stratify_key(df)
        
        group_counts = stratify_key.value_counts()
        min_group_size = group_counts.min()
        
        if min_group_size < 3:
            warnings.warn(f"Smallest group has only {min_group_size} samples. "
                         "Stratification may not work properly.")
        
        try:
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                stratify=stratify_key,
                random_state=self.random_state
            )
            
            train_val_stratify = self._create_stratify_key(train_val_df)
            
            adjusted_val_size = self.val_size / (1 - self.test_size)
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=adjusted_val_size,
                stratify=train_val_stratify,
                random_state=self.random_state
            )
            
        except ValueError as e:
            if "least populated class" in str(e) or "stratify" in str(e).lower():
                warnings.warn("Stratification failed due to insufficient samples in some groups, falling back to random split")
                return self._random_split(df)
            else:
                raise e

        self._validate_splits(train_df, val_df, test_df, df)
        
        return train_df, val_df, test_df
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        required_cols = ['default', 'sex', 'age_group']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns for stratification: {missing_cols}")
        
        for col in required_cols:
            if not df[col].isin([0, 1]).all():
                raise ValueError(f"Column {col} must contain only 0 and 1 values")
    
    def _create_stratify_key(self, df: pd.DataFrame) -> pd.Series:

        return (
            df['default'].astype(str) + '_' +
            df['sex'].astype(str) + '_' +
            df['age_group'].astype(str)
        )
    
    def _random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        adjusted_val_size = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            random_state=self.random_state
        )
        
        return train_df, val_df, test_df
    
    def _validate_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, original_df: pd.DataFrame) -> None:
     
        total_samples = len(original_df)
        
        split_total = len(train_df) + len(val_df) + len(test_df)
        if split_total != total_samples:
            raise ValueError(f"Sample count mismatch: {split_total} != {total_samples}")
        
        
        actual_test_prop = len(test_df) / total_samples
        actual_val_prop = len(val_df) / total_samples
        actual_train_prop = len(train_df) / total_samples
        
        expected_train_prop = 1 - self.test_size - (self.val_size * (1 - self.test_size))
        
       
        tolerance = 0.05
        
        if abs(actual_test_prop - self.test_size) > tolerance:
            warnings.warn(f"Test proportion {actual_test_prop:.3f} differs from expected {self.test_size}")
        
        if abs(actual_train_prop - expected_train_prop) > tolerance:
            warnings.warn(f"Train proportion {actual_train_prop:.3f} differs from expected {expected_train_prop:.3f}")
        
      
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            for attr in ['default', 'sex', 'age_group']:
                original_groups = set(original_df[attr].unique())
                split_groups = set(split_df[attr].unique())
                
                if split_groups != original_groups:
                    missing_groups = original_groups - split_groups
                    warnings.warn(f"{split_name} split missing {attr} groups: {missing_groups}")