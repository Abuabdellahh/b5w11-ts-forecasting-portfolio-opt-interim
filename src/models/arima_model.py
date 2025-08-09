"""
ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.
"""
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import pmdarima as pm

from .base_model import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = None, seasonal_order: Tuple[int, int, int, int] = None,
                 auto_fit: bool = True, **kwargs):
        """
        Initialize the ARIMA model.
        
        Args:
            order: (p,d,q) order of the ARIMA model
            seasonal_order: (P,D,Q,s) order of the seasonal component
            auto_fit: Whether to automatically determine the best order using auto_arima
            **kwargs: Additional arguments to pass to the ARIMA model
        """
        super().__init__(**kwargs)
        self.order = order or (1, 1, 1)
        self.seasonal_order = seasonal_order
        self.auto_fit = auto_fit
        self.model = None
        self.is_fitted = False
        self.model_params = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'auto_fit': self.auto_fit,
            **kwargs
        }
        
    def fit(self, train_data: pd.Series) -> 'ARIMAModel':
        """
        Fit the ARIMA model to the training data.
        
        Args:
            train_data: Pandas Series with datetime index containing the training data
            
        Returns:
            self: Returns an instance of self
        """
        if self.auto_fit and (self.order is None or self.seasonal_order is None):
            logger.info("Automatically determining ARIMA order using auto_arima...")
            self._auto_fit(train_data)
        else:
            logger.info(f"Fitting ARIMA{self.order} model...")
            self.model = ARIMAModel(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                **{k: v for k, v in self.model_params.items() 
                   if k not in ['order', 'seasonal_order', 'auto_fit']}
            )
            self.model = self.model.fit()
            
        self.is_fitted = True
        logger.info("Model fitting complete.")
        return self
    
    def _auto_fit(self, train_data: pd.Series) -> None:
        """
        Automatically determine the best ARIMA order using auto_arima.
        
        Args:
            train_data: Training data series
        """
        logger.info("Running auto_arima to find optimal parameters...")
        
        # Determine if we should use seasonal ARIMA
        if self.seasonal_order is not None:
            m = self.seasonal_order[3]  # seasonal period
        else:
            m = 1  # non-seasonal
        
        # Run auto_arima
        stepwise_fit = pm.auto_arima(
            train_data,
            start_p=1, start_q=1,
            max_p=3, max_q=3, m=m,
            start_P=0, seasonal=True if m > 1 else False,
            d=None, D=None, trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        # Extract the best parameters
        self.order = stepwise_fit.order
        self.seasonal_order = stepwise_fit.seasonal_order
        
        logger.info(f"Best ARIMA order: {self.order}")
        if self.seasonal_order:
            logger.info(f"Best seasonal order: {self.seasonal_order}")
        
        # Store the fitted model
        self.model = stepwise_fit
    
    def predict(self, steps: int, return_conf_int: bool = True, alpha: float = 0.05, 
               **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Make predictions using the fitted ARIMA model.
        
        Args:
            steps: Number of time steps to forecast
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            **kwargs: Additional arguments to pass to the predict method
            
        Returns:
            Tuple containing:
                - forecast: Array of predicted values
                - lower_bound: Array of lower bound values for confidence interval (or None)
                - upper_bound: Array of upper bound values for confidence interval (or None)
        """
        self._check_is_fitted()
        
        # Make predictions
        if hasattr(self.model, 'predict'):  # For statsmodels ARIMA
            forecast_result = self.model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean.values
            
            if return_conf_int:
                conf_int = forecast_result.conf_int(alpha=alpha)
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            else:
                lower_bound = upper_bound = None
        else:  # For pmdarima ARIMA
            forecast, conf_int = self.model.predict(
                n_periods=steps, 
                return_conf_int=True, 
                alpha=alpha,
                **kwargs
            )
            forecast = forecast.values
            
            if return_conf_int and conf_int is not None:
                lower_bound = conf_int[:, 0]
                upper_bound = conf_int[:, 1]
            else:
                lower_bound = upper_bound = None
        
        return forecast, lower_bound, upper_bound
    
    def evaluate(self, test_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate the ARIMA model on test data.
        
        Args:
            test_data: Pandas Series with datetime index containing the test data
            
        Returns:
            Dictionary of evaluation metrics (MAE, RMSE, MAPE, R2)
        """
        self._check_is_fitted()
        
        # Make predictions for the test period
        steps = len(test_data)
        forecast, _, _ = self.predict(steps=steps, return_conf_int=False)
        
        # Calculate metrics
        metrics = self.calculate_metrics(test_data.values, forecast)
        
        return metrics
    
    def summary(self) -> None:
        """Print the model summary."""
        self._check_is_fitted()
        if hasattr(self.model, 'summary'):
            print(self.model.summary())
        else:
            print("Model summary not available for this model type.")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model's parameters.
        
        Returns:
            Dictionary of model parameters
        """
        params = super().get_params()
        if hasattr(self.model, 'get_params'):
            model_params = self.model.get_params()
            params.update(model_params)
        return params
    
    def save_model(self, filepath: str) -> None:
        """
        Save the ARIMA model to a file.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        
        # For pmdarima models, we can save directly
        if hasattr(self.model, 'save'):
            self.model.save(filepath)
        # For statsmodels models, we need to use joblib
        else:
            joblib.dump(self.model, filepath)
            
        logger.info(f"ARIMA model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'ARIMAModel':
        """
        Load an ARIMA model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: Returns an instance of self with the loaded model
        """
        import joblib
        import os
        
        # Check if it's a pmdarima model
        if os.path.exists(filepath) and os.path.isdir(filepath):
            from pmdarima.utils import load_model
            self.model = load_model(filepath)
        # Otherwise try to load with joblib
        else:
            self.model = joblib.load(filepath)
            
        self.is_fitted = True
        logger.info(f"ARIMA model loaded from {filepath}")
        return self
