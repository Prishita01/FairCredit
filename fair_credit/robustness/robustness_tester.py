from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np

class RobustnessTester(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def test_distribution_shift(self, data: pd.DataFrame, 
                               shift_magnitude: float = 0.25) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def test_missing_features(self, X: pd.DataFrame, 
                             missing_rate: float = 0.1) -> Dict[str, Any]:
        pass