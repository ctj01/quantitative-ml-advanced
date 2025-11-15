"""
Financial Data Loaders and Processors
======================================

This module provides functions to download, clean, and process financial data
for quantitative trading analysis.

Author: Cristian Mendoza
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Union, List, Tuple
import warnings


def load_financial_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = '3y',
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Load financial data from Yahoo Finance
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol (e.g., 'BTC-USD', 'AAPL', 'SPY')
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
    period : str
        Period to download ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    interval : str
        Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with OHLCV data and additional columns
    """
    try:
        if start_date and end_date:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        else:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Clean column names
        data.columns = [col.lower() for col in data.columns]
        
        # Add returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Add basic volatility
        data['realized_vol'] = data['returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Remove NaN values
        data = data.dropna()
        
        return data
        
    except Exception as e:
        raise ValueError(f"Error loading data for {symbol}: {str(e)}")


def load_multiple_assets(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = '3y'
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple assets
    
    Parameters:
    -----------
    symbols : list of str
        List of ticker symbols
    start_date, end_date : str, optional
        Date range
    period : str
        Period to download
        
    Returns:
    --------
    data_dict : dict
        Dictionary mapping symbols to DataFrames
    """
    data_dict = {}
    
    for symbol in symbols:
        try:
            data_dict[symbol] = load_financial_data(
                symbol, start_date, end_date, period
            )
            print(f"✓ Loaded {symbol}: {len(data_dict[symbol])} observations")
        except Exception as e:
            print(f"✗ Failed to load {symbol}: {str(e)}")
    
    return data_dict


def create_panel_data(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    column: str = 'close'
) -> pd.DataFrame:
    """
    Create panel data (wide format) with one column per asset
    
    Parameters:
    -----------
    symbols : list of str
        List of ticker symbols
    start_date, end_date : str, optional
        Date range
    column : str
        Column to extract ('close', 'returns', etc.)
        
    Returns:
    --------
    panel : pd.DataFrame
        Wide-format DataFrame with dates as index and symbols as columns
    """
    data_dict = load_multiple_assets(symbols, start_date, end_date)
    
    panel = pd.DataFrame()
    for symbol, data in data_dict.items():
        panel[symbol] = data[column]
    
    # Align dates and forward-fill missing values
    panel = panel.fillna(method='ffill').dropna()
    
    return panel


def clean_data(df: pd.DataFrame, 
               remove_outliers: bool = True,
               outlier_std: float = 5.0,
               fill_method: str = 'ffill') -> pd.DataFrame:
    """
    Clean financial data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data
    remove_outliers : bool
        Whether to remove outliers
    outlier_std : float
        Number of standard deviations for outlier detection
    fill_method : str
        Method to fill missing values ('ffill', 'bfill', 'interpolate')
        
    Returns:
    --------
    df_clean : pd.DataFrame
        Cleaned data
    """
    df_clean = df.copy()
    
    # Handle missing values
    if fill_method == 'ffill':
        df_clean = df_clean.fillna(method='ffill')
    elif fill_method == 'bfill':
        df_clean = df_clean.fillna(method='bfill')
    elif fill_method == 'interpolate':
        df_clean = df_clean.interpolate(method='linear')
    
    # Remove outliers from returns
    if remove_outliers and 'returns' in df_clean.columns:
        mean_return = df_clean['returns'].mean()
        std_return = df_clean['returns'].std()
        
        lower_bound = mean_return - outlier_std * std_return
        upper_bound = mean_return + outlier_std * std_return
        
        outliers = (df_clean['returns'] < lower_bound) | (df_clean['returns'] > upper_bound)
        
        if outliers.sum() > 0:
            print(f"Removed {outliers.sum()} outliers ({outliers.sum()/len(df_clean)*100:.2f}%)")
            df_clean.loc[outliers, 'returns'] = np.nan
            df_clean['returns'] = df_clean['returns'].fillna(method='ffill')
    
    return df_clean


def split_train_test(
    df: pd.DataFrame,
    train_size: float = 0.8,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to split
    train_size : float
        Proportion of data for training
    shuffle : bool
        Whether to shuffle (NOT recommended for time series!)
        
    Returns:
    --------
    train, test : pd.DataFrame, pd.DataFrame
        Training and testing sets
    """
    if shuffle:
        warnings.warn("Shuffling time series data can introduce look-ahead bias!")
        df = df.sample(frac=1).reset_index(drop=True)
    
    split_idx = int(len(df) * train_size)
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    print(f"Train: {len(train)} observations ({train.index[0]} to {train.index[-1]})")
    print(f"Test: {len(test)} observations ({test.index[0]} to {test.index[-1]})")
    
    return train, test


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lagged features for time series analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data
    columns : list of str
        Columns to lag
    lags : list of int
        Number of lags to create
        
    Returns:
    --------
    df_lagged : pd.DataFrame
        Data with lagged features
    """
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    df_lagged = df_lagged.dropna()
    
    return df_lagged


def calculate_rolling_features(
    df: pd.DataFrame,
    column: str = 'close',
    windows: List[int] = [5, 10, 20, 50, 200]
) -> pd.DataFrame:
    """
    Calculate rolling statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data
    column : str
        Column to calculate statistics for
    windows : list of int
        Rolling window sizes
        
    Returns:
    --------
    df_rolling : pd.DataFrame
        Data with rolling features
    """
    df_rolling = df.copy()
    
    for window in windows:
        # Moving averages
        df_rolling[f'ma{window}'] = df[column].rolling(window=window).mean()
        
        # Exponential moving average
        df_rolling[f'ema{window}'] = df[column].ewm(span=window, adjust=False).mean()
        
        # Rolling standard deviation
        df_rolling[f'std{window}'] = df[column].rolling(window=window).std()
        
        # Rolling min/max
        df_rolling[f'min{window}'] = df[column].rolling(window=window).min()
        df_rolling[f'max{window}'] = df[column].rolling(window=window).max()
        
        # Distance from MA
        df_rolling[f'dist_ma{window}'] = (df[column] - df_rolling[f'ma{window}']) / df_rolling[f'ma{window}']
    
    return df_rolling


# Example usage
if __name__ == "__main__":
    print("Testing Data Loaders...")
    print("=" * 60)
    
    # Load single asset
    print("\n1. Loading Bitcoin data...")
    btc_data = load_financial_data('BTC-USD', period='1y')
    print(btc_data.head())
    print(f"\nShape: {btc_data.shape}")
    
    # Load multiple assets
    print("\n2. Loading multiple cryptocurrencies...")
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    crypto_data = load_multiple_assets(symbols, period='6mo')
    
    # Create panel data
    print("\n3. Creating panel data...")
    panel = create_panel_data(symbols, period='6mo', column='close')
    print(panel.head())
    
    # Clean data
    print("\n4. Cleaning data...")
    btc_clean = clean_data(btc_data, remove_outliers=True, outlier_std=4.0)
    
    # Split train/test
    print("\n5. Splitting train/test...")
    train, test = split_train_test(btc_clean, train_size=0.8)
    
    print("\n✓ All tests passed!")
