import os
import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class Config:
    
    random_seed: int = 42
    numpy_seed: int = 42
    sklearn_seed: int = 42
    
    test_size: float = 0.2
    val_size: float = 0.2
    stratify_by_protected: bool = True
    
    logistic_regression_params: Dict[str, Any] = None
    xgboost_params: Dict[str, Any] = None
    
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    reweighing_enabled: bool = True
    threshold_optimization_enabled: bool = True
    
    distribution_shift_magnitude: float = 0.25
    missing_data_rate: float = 0.1
    stability_threshold: float = 0.25
    
    output_dir: str = "results"
    save_plots: bool = True
    save_models: bool = True
    
    def __post_init__(self):

        if self.logistic_regression_params is None:
            self.logistic_regression_params = {
                'random_state': self.sklearn_seed,
                'max_iter': 1000,
                'solver': 'liblinear'
            }
            
        if self.xgboost_params is None:
            self.xgboost_params = {
                'random_state': self.sklearn_seed,
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**config_dict)
    
    def to_file(self, config_path: str) -> None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def setup_reproducibility(self) -> None:
        import random
        import numpy as np
        
        random.seed(self.random_seed)
        
        np.random.seed(self.numpy_seed)
        
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
    
    def create_output_directory(self) -> Path:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def get_experiment_id(self) -> str:
        import hashlib
        
        config_str = f"{self.random_seed}_{self.test_size}_{self.val_size}_{self.bootstrap_samples}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]