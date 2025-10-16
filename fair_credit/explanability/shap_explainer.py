from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class SHAPExplainer(ABC):
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def explain_model(self, model, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def group_wise_analysis(self, shap_values: np.ndarray, 
                           protected_attr: np.ndarray) -> Dict[str, Any]:
        pass