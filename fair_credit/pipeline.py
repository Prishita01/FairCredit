# Authors: Akash Rana & Prishita Ghanathe Krishna

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from .config import Config
from .data_processor.processor import DataProcessor
from .models.base import BaselineModel
from .fairness.auditor import FairnessAuditor
from .mitigation.base import BiasmitigationTechnique, MitigationEvaluator
from .explainability.shap_explainer import SHAPExplainer
from .robustness.robustness_tester import RobustnessTester


class FairCreditPipeline:
    def __init__(self, config: Config):
        # Initializing the FairCredit pipeline.

        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components (will be set during pipeline execution)
        self.data_processor: Optional[DataProcessor] = None
        self.baseline_models: Dict[str, BaselineModel] = {}
        self.fairness_auditor: Optional[FairnessAuditor] = None
        self.mitigation_techniques: Dict[str, BiasmitigationTechnique] = {}
        self.explainer: Optional[SHAPExplainer] = None
        self.robustness_tester: Optional[RobustnessTester] = None
        self.evaluator = MitigationEvaluator()
        
        # Data storage
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Any] = {}
        
        # Set up reproducibility
        self.config.setup_reproducibility()
        
    def _setup_logging(self) -> logging.Logger:
        # Setting up logging for the pipeline.
        logger = logging.getLogger('FairCredit')
        logger.setLevel(logging.INFO)
        
        # Creating console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def load_and_preprocess_data(self) -> Dict[str, pd.DataFrame]:
        # German Credit Dataset
        self.logger.info("Loading and preprocessing German Credit dataset...")
        
        if self.data_processor is None:
            raise ValueError("Data processor not initialized")
        
        df = self.data_processor.load_german_credit()
        self.logger.info(f"Loaded dataset with {len(df)} samples")
        
        df = self.data_processor.encode_protected_attributes(df)
        
        train_df, val_df, test_df = self.data_processor.create_splits(
            df, 
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_seed
        )
        
        self.datasets = {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'full': df
        }
        
        self.logger.info(f"Created splits - Train: {len(train_df)}, "
                        f"Val: {len(val_df)}, Test: {len(test_df)}")
        
        return self.datasets
    
    def train_baseline_models(self) -> Dict[str, BaselineModel]:

        self.logger.info("Training baseline models...")
        
        if 'train' not in self.datasets:
            raise ValueError("Training data not available. Run load_and_preprocess_data() first.")
        
        train_df = self.datasets['train']
        X_train = train_df.drop(['default'], axis=1)
        y_train = train_df['default']
        
        # Train models (implementations will be added in subsequent tasks)
        # This is just the interface setup
        
        self.logger.info(f"Trained {len(self.baseline_models)} baseline models")
        return self.baseline_models
    
    def audit_fairness(self, model_name: str) -> Dict[str, Any]:
        
        self.logger.info(f"Auditing fairness for model: {model_name}")
        
        if model_name not in self.baseline_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if self.fairness_auditor is None:
            raise ValueError("Fairness auditor not initialized")
        
        # Getting the test data
        test_df = self.datasets['test']
        X_test = test_df.drop(['default', 'sex', 'age_group'], axis=1)
        y_test = test_df['default'].values
        
        # Making predictions
        model = self.baseline_models[model_name]
        y_pred = model.predict(X_test)

        # Auditing fairness for each protected attribute (will be expanded in subsequent tasks)
        fairness_results = {}
        
        for attr in ['sex', 'age_group']:
            protected_attr = test_df[attr].values
            metrics = self.fairness_auditor.compute_all_metrics(y_test, y_pred, protected_attr)
            fairness_results[attr] = metrics
        
        self.results[f'{model_name}_fairness'] = fairness_results
        
        self.logger.info(f"Completed fairness audit for {model_name}")
        return fairness_results
    
    def apply_mitigation(self, technique_name: str, model_name: str) -> BaselineModel:
        # Applying bias mitigation technique to a model.
        self.logger.info(f"Applying {technique_name} mitigation to {model_name}")
        
        if technique_name not in self.mitigation_techniques:
            raise ValueError(f"Mitigation technique '{technique_name}' not found")
        
        if model_name not in self.baseline_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Implementation will be added later
        
        self.logger.info(f"Applied {technique_name} mitigation")
        return self.baseline_models[model_name]
    
    def explain_model(self, model_name: str) -> Dict[str, Any]:

        self.logger.info(f"Generating explanations for model: {model_name}")
        
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        # Implementation will be added later
        explanation_results = {}
        self.results[f'{model_name}_explanations'] = explanation_results
        
        return explanation_results
    
    def test_robustness(self, model_name: str) -> Dict[str, Any]:
        self.logger.info(f"Testing robustness for model: {model_name}")
        
        if self.robustness_tester is None:
            raise ValueError("Robustness tester not initialized")
        
        # Implementation will be added later
        
        robustness_results = {}
        self.results[f'{model_name}_robustness'] = robustness_results
        
        return robustness_results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
       
        self.logger.info("Starting full FairCredit pipeline...")
        
        # 1. Loading and preprocessing the data
        self.load_and_preprocess_data()
        
        # 2. Training baseline models
        self.train_baseline_models()
        
        # 3. Auditing fairness for each model
        for model_name in self.baseline_models:
            self.audit_fairness(model_name)
        
        # 4. Applying mitigation techniques
        for technique_name in self.mitigation_techniques:
            for model_name in self.baseline_models:
                self.apply_mitigation(technique_name, model_name)
        
        # 5. Generating explanations
        for model_name in self.baseline_models:
            self.explain_model(model_name)
        
        # 6. Testing our model's robustness
        for model_name in self.baseline_models:
            self.test_robustness(model_name)
        
        self.logger.info("Completed full FairCredit pipeline")
        return self.results
    
    def save_results(self, output_dir: Optional[str] = None) -> None:
        
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.config.to_file(output_path / "config.yaml")
        
        # Save results (implementation will be expanded in subsequent tasks)
        self.logger.info(f"Results saved to {output_path}")
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        # Summarizing the pipeline execution
        summary = {
            'config': self.config.get_experiment_id(),
            'datasets': {name: len(df) for name, df in self.datasets.items()},
            'models': list(self.baseline_models.keys()),
            'mitigation_techniques': list(self.mitigation_techniques.keys()),
            'results_available': list(self.results.keys())
        }
        
        return summary