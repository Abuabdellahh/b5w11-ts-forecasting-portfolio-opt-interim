""
Test script for the portfolio optimization pipeline.

This script tests the data loading, preprocessing, and EDA components
of the portfolio optimization pipeline.
"""
import sys
import os
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading and preprocessing."""
    from src.data.data_loader import DataLoader
    
    logger.info("Testing data loading and preprocessing...")
    
    # Initialize data loader
    tickers = ['TSLA', 'BND', 'SPY']
    start_date = '2015-07-01'
    end_date = '2025-07-31'
    
    loader = DataLoader(tickers, start_date, end_date)
    
    # Fetch and preprocess data
    raw_data = loader.fetch_data()
    processed_data, returns_data = loader.preprocess_data()
    
    # Basic assertions
    assert len(raw_data) == len(tickers), "Incorrect number of tickers in raw data"
    assert len(processed_data) == len(tickers), "Incorrect number of tickers in processed data"
    assert len(returns_data) == len(tickers), "Incorrect number of tickers in returns data"
    
    for ticker in tickers:
        assert ticker in raw_data, f"{ticker} not found in raw data"
        assert ticker in processed_data, f"{ticker} not found in processed data"
        assert ticker in returns_data, f"{ticker} not found in returns data"
        
        # Check if processed data has expected columns
        expected_columns = [
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker',
            'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 'Quarter',
            'Daily_Return', 'Log_Return', 'Volatility_20D', 'MA_20', 'MA_50'
        ]
        
        for col in expected_columns:
            if col not in processed_data[ticker].columns:
                logger.warning(f"Column '{col}' not found in processed data for {ticker}")
    
    logger.info("Data loading and preprocessing tests passed!")
    return True

def test_eda():
    """Test exploratory data analysis."""
    from src.visualization.eda import FinancialAnalyzer
    
    logger.info("Testing exploratory data analysis...")
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer('data/processed')
    analyzer.load_data()
    
    # Test visualization functions
    if not os.path.exists('reports/figures'):
        os.makedirs('reports/figures')
    
    analyzer.plot_price_series('reports/figures')
    analyzer.plot_daily_returns('reports/figures')
    analyzer.plot_volatility(window=20, save_path='reports/figures')
    
    # Test risk metrics calculation
    risk_metrics = analyzer.calculate_risk_metrics()
    assert not risk_metrics.empty, "Risk metrics calculation failed"
    
    expected_metrics = [
        'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
        '1-day 95% VaR', 'Max Drawdown', 'Skewness', 'Kurtosis'
    ]
    
    for metric in expected_metrics:
        assert metric in risk_metrics.columns, f"{metric} not found in risk metrics"
    
    # Test stationarity tests
    stationarity_results = analyzer.test_stationarity()
    assert stationarity_results, "Stationarity test failed"
    
    logger.info("EDA tests passed!")
    return True

def test_arima_model():
    """Test ARIMA model implementation."""
    import pandas as pd
    import numpy as np
    from src.models import ARIMAModel
    
    logger.info("Testing ARIMA model...")
    
    # Generate synthetic time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    trend = np.linspace(0, 10, len(dates))
    noise = np.random.normal(0, 1, len(dates))
    data = pd.Series(trend + noise, index=dates, name='synthetic')
    
    # Split into train and test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Test auto ARIMA
    model = ARIMAModel(auto_fit=True)
    model.fit(train_data)
    
    # Test prediction
    forecast, lower, upper = model.predict(steps=len(test_data), return_conf_int=True)
    assert len(forecast) == len(test_data), "Forecast length mismatch"
    assert len(lower) == len(test_data), "Lower bound length mismatch"
    assert len(upper) == len(test_data), "Upper bound length mismatch"
    
    # Test evaluation
    metrics = model.evaluate(test_data)
    expected_metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    for metric in expected_metrics:
        assert metric in metrics, f"{metric} not found in evaluation metrics"
    
    # Test model saving and loading
    model_path = 'models/test_arima_model.pkl'
    model.save_model(model_path)
    
    loaded_model = ARIMAModel()
    loaded_model.load_model(model_path)
    
    # Test that loaded model can make predictions
    loaded_forecast, _, _ = loaded_model.predict(steps=5)
    assert len(loaded_forecast) == 5, "Loaded model prediction failed"
    
    logger.info("ARIMA model tests passed!")
    return True

def main():
    """Run all tests."""
    tests = [
        ("Data Loading", test_data_loading),
        ("EDA", test_eda),
        ("ARIMA Model", test_arima_model)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            logger.info(f"Running {test_name} tests...")
            if test_func():
                logger.info(f"{test_name} tests passed!\n")
            else:
                logger.error(f"{test_name} tests failed!\n")
                all_passed = False
        except Exception as e:
            logger.exception(f"{test_name} tests failed with exception: {str(e)}\n")
            all_passed = False
    
    if all_passed:
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Some tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
