""
Base class for time series forecasting models.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for time series forecasting models."""
    
    def __init__(self, **kwargs):
        """Initialize the base model with any model-specific parameters."""
        self.model = None
        self.is_fitted = False
        self.model_params = kwargs
        
    @abstractmethod
    def fit(self, train_data: pd.Series) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            train_data: Pandas Series with datetime index containing the training data
            
        Returns:
            self: Returns an instance of self
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted model.
        
        Args:
            steps: Number of time steps to forecast
            **kwargs: Additional arguments specific to the model implementation
            
        Returns:
            Tuple containing:
                - forecast: Array of predicted values
                - lower_bound: Array of lower bound values for confidence interval
                - upper_bound: Array of upper bound values for confidence interval
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Pandas Series with datetime index containing the test data
            
        Returns:
            Dictionary of evaluation metrics (e.g., MAE, RMSE, MAPE)
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model's parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.model_params
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set the model's parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: Returns an instance of self
        """
        self.model_params.update(params)
        return self
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: Returns an instance of self with the loaded model
        """
        import joblib
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def _check_is_fitted(self) -> None:
        """Check if the model is fitted, raise an error if not."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call 'fit' before using this method.")
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate common regression metrics.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            Dictionary of metrics (MAE, RMSE, MAPE, R2)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
