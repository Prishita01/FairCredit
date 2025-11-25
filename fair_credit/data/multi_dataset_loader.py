import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import logging

try:
    from ..config import Config
except ImportError:
    from config import Config


class DatasetConfig:
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.file_path = None
        self.target_column = None
        self.protected_attributes = []
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.preprocessing_params = {}


class BaseDatasetLoader(ABC):
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def encode_protected_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def load_and_preprocess(self) -> pd.DataFrame:
        df = self.load_raw_data()
        df = self.preprocess_data(df)
        df = self.encode_protected_attributes(df)
        return df


class GermanCreditLoader(BaseDatasetLoader):
    
    def __init__(self, data_path: str = None):
        config = DatasetConfig("german_credit")
        # Try multiple possible paths
        if data_path is None:
            possible_paths = [
                "data/german_credit_data.csv",
                "../data/german_credit_data.csv",
                "../../data/german_credit_data.csv"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    data_path = path
                    break
            if data_path is None:
                data_path = "data/german_credit_data.csv"  # Default fallback
        config.file_path = data_path
        config.target_column = "default"
        config.protected_attributes = ["sex", "age"]
        super().__init__(config)
    
    def load_raw_data(self) -> pd.DataFrame:

        try:
            df = pd.read_csv(self.config.file_path)
            self.logger.info(f"Loaded German Credit dataset: {len(df)} samples")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load German Credit dataset: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess German Credit dataset."""

        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
 
        if 'Risk' in df.columns:
            df['default'] = (df['Risk'] == 'bad').astype(int)
            df = df.drop('Risk', axis=1)
        elif 'class' in df.columns:
            df['default'] = (df['class'] == 'bad').astype(int)
            df = df.drop('class', axis=1)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['Sex', 'sex', 'Age', 'age']: 
                df[col] = pd.Categorical(df[col]).codes
        
        return df
    
    def encode_protected_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
 
        sex_col = 'Sex' if 'Sex' in df.columns else 'sex' if 'sex' in df.columns else None
        if sex_col:
            df['sex'] = df[sex_col].map({'male': 1, 'female': 0})
            df['sex'] = df['sex'].fillna(df['sex'].mode()[0] if not df['sex'].empty else 0)
            if sex_col != 'sex':
                df = df.drop(sex_col, axis=1)
        
        age_col = 'Age' if 'Age' in df.columns else 'age' if 'age' in df.columns else None
        if age_col:
            df['age_group'] = (df[age_col] >= 25).astype(int)
        
        return df


class PortugueseBankLoader(BaseDatasetLoader):
    
    def __init__(self, data_path: str = None):
        config = DatasetConfig("portuguese_bank")
        if data_path is None:
            possible_paths = [
                "data/Portugese_bank.csv",
                "../data/Portugese_bank.csv",
                "../../data/Portugese_bank.csv"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    data_path = path
                    break
            if data_path is None:
                data_path = "data/Portugese_bank.csv" 
        config.file_path = data_path
        config.target_column = "y" 
        config.protected_attributes = ["age", "marital"]
        super().__init__(config)
    
    def load_raw_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.config.file_path, sep=';')
            self.logger.info(f"Loaded Portuguese Bank dataset: {len(df)} samples")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load Portuguese Bank dataset: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['default'] = (df['y'] == 'yes').astype(int)
        df = df.drop('y', axis=1)
        
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                           'loan', 'contact', 'month', 'poutcome']
        
        for col in categorical_cols:
            if col in df.columns and col not in ['marital']: 
                df[col] = pd.Categorical(df[col]).codes
        
        numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 
                         'pdays', 'previous']
        
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def encode_protected_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'age' in df.columns:
            df['age_group'] = (df['age'] >= 30).astype(int)
        
        if 'marital' in df.columns:
            df['sex'] = df['marital'].map({'married': 1, 'divorced': 1, 'single': 0})
            df['sex'] = df['sex'].fillna(0)
            df = df.drop('marital', axis=1)
        
        return df


class USCreditLoader(BaseDatasetLoader):

    
    def __init__(self, data_path: str = None):
        config = DatasetConfig("us_credit")
        if data_path is None:
            possible_paths = [
                "data/US_credit_2023_sample.csv",
                "../data/US_credit_2023_sample.csv",
                "../../data/US_credit_2023_sample.csv"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    data_path = path
                    break
            if data_path is None:
                data_path = "data/US_credit_2023_sample.csv"  
        config.file_path = data_path
        config.target_column = "approved"
        config.protected_attributes = ["derived_sex", "applicant_age"]
        super().__init__(config)
    
    def load_raw_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.config.file_path)
            self.logger.info(f"Loaded US Credit dataset: {len(df)} samples")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load US Credit dataset: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['default'] = (df['approved'] == 0).astype(int)
        key_features = [
            'loan_amount', 'income', 'debt_to_income_ratio',
            'property_value', 'loan_term', 'interest_rate',
            'derived_sex', 'applicant_age', 'derived_race', 'derived_ethnicity',
            'approved', 'default'
        ]
        
        available_features = [col for col in key_features if col in df.columns]
        df = df[available_features].copy()
        
        numerical_cols = ['loan_amount', 'income', 'debt_to_income_ratio',
                         'property_value', 'loan_term', 'interest_rate']
        
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = ['derived_race', 'derived_ethnicity']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        return df
    
    def encode_protected_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'derived_sex' in df.columns:
            sex_mapping = {'Male': 1, 'Female': 0, 'Joint': 1}
            df['sex'] = df['derived_sex'].map(sex_mapping)
            df['sex'] = df['sex'].fillna(df['sex'].mode()[0] if not df['sex'].empty else 0)
            df = df.drop('derived_sex', axis=1)
        
        if 'applicant_age' in df.columns:
            def parse_age_group(age_str):
                if pd.isna(age_str) or age_str == '9999':
                    return 1 
                if isinstance(age_str, str):
                    if '-' in age_str:
                        age_start = int(age_str.split('-')[0])
                        return 1 if age_start >= 35 else 0
                    elif '>' in age_str:
                        return 1 
                    elif '<' in age_str:
                        return 0  
                return 1  
            
            df['age_group'] = df['applicant_age'].apply(parse_age_group)
            df = df.drop('applicant_age', axis=1)
        
        return df


class MultiDatasetManager:

    
    def __init__(self):
        self.loaders = {
            'german_credit': GermanCreditLoader,
            'portuguese_bank': PortugueseBankLoader,
            'us_credit': USCreditLoader
        }
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_dataset(self, dataset_name: str, data_path: Optional[str] = None) -> pd.DataFrame:

        if dataset_name not in self.loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.loaders.keys())}")
        
        loader_class = self.loaders[dataset_name]
        
        if data_path:
            loader = loader_class(data_path)
        else:
            loader = loader_class()
        
        return loader.load_and_preprocess()
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:

        datasets = {}
        
        for dataset_name in self.loaders.keys():
            try:
                datasets[dataset_name] = self.load_dataset(dataset_name)
                self.logger.info(f"Successfully loaded {dataset_name}")
            except Exception as e:
                self.logger.error(f"Failed to load {dataset_name}: {str(e)}")
        
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
  
        if dataset_name not in self.loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        df = self.load_dataset(dataset_name)
        
        info = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'target_column': 'default',
            'protected_attributes': ['sex', 'age_group'],
            'missing_values': df.isnull().sum().to_dict(),
            'target_distribution': df['default'].value_counts().to_dict(),
            'protected_group_distribution': {}
        }
        
        for attr in ['sex', 'age_group']:
            if attr in df.columns:
                info['protected_group_distribution'][attr] = df[attr].value_counts().to_dict()
        
        return info
    
    def compare_datasets(self) -> pd.DataFrame:
        comparison_data = []
        
        for dataset_name in self.loaders.keys():
            try:
                info = self.get_dataset_info(dataset_name)
                
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Samples': info['shape'][0],
                    'Features': info['shape'][1],
                    'Positive_Rate': info['target_distribution'].get(1, 0) / info['shape'][0],
                    'Missing_Values': sum(info['missing_values'].values()),
                    'Sex_Groups': len(info['protected_group_distribution'].get('sex', {})),
                    'Age_Groups': len(info['protected_group_distribution'].get('age_group', {}))
                })
            except Exception as e:
                self.logger.error(f"Failed to get info for {dataset_name}: {str(e)}")
        
        return pd.DataFrame(comparison_data)


def load_german_credit() -> pd.DataFrame:
    loader = GermanCreditLoader()
    return loader.load_and_preprocess()


def load_portuguese_bank() -> pd.DataFrame:
    loader = PortugueseBankLoader()
    return loader.load_and_preprocess()


def load_us_credit() -> pd.DataFrame:
    loader = USCreditLoader()
    return loader.load_and_preprocess()


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    manager = MultiDatasetManager()
    return manager.load_all_datasets()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = MultiDatasetManager()
    
    datasets = manager.load_all_datasets()
    
    comparison = manager.compare_datasets()
    print("Dataset Comparison:")
    print(comparison)
    
    for dataset_name in datasets.keys():
        print(f"\n{dataset_name.upper()} Dataset Info:")
        info = manager.get_dataset_info(dataset_name)
        print(f"  Shape: {info['shape']}")
        print(f"  Target distribution: {info['target_distribution']}")
        print(f"  Protected groups: {info['protected_group_distribution']}")