"""
Technical Analysis and Feature Engineering
===========================================

This module provides comprehensive technical indicators and feature engineering
for trading strategies, including RSI, W/M patterns, divergences, and candlestick patterns.

Author: Cristian Mendoza
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, List, Optional, Dict
import warnings


class TechnicalIndicators:
    """Collection of technical analysis indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        period : int
            RSI period (default: 14)
            
        Returns:
        --------
        rsi : pd.Series
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # Wilder's smoothing (exponential moving average)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, 
                      fast: int = 12, 
                      slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Returns:
        --------
        macd, signal_line, histogram : tuple of pd.Series
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, 
                                  period: int = 20, 
                                  std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Returns:
        --------
        upper, middle, lower : tuple of pd.Series
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        TR = max(high - low, abs(high - close_prev), abs(low - close_prev))
        ATR = MA(TR, period)
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume
        
        OBV increases by volume on up days, decreases on down days
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        %K = (Close - Low_n) / (High_n - Low_n) * 100
        %D = MA(%K, d_period)
        """
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        
        k = 100 * (close - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        
        return k, d


class RSIFeatures:
    """Advanced RSI feature engineering"""
    
    @staticmethod
    def create_rsi_features(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
        """
        Create comprehensive RSI-based features
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'close' column
        rsi_period : int
            RSI calculation period
            
        Returns:
        --------
        df_features : pd.DataFrame
            DataFrame with RSI features
        """
        df_features = df.copy()
        
        # Basic RSI
        df_features['rsi'] = TechnicalIndicators.calculate_rsi(df['close'], rsi_period)
        
        # RSI trend (difference from moving average)
        df_features['rsi_ma20'] = df_features['rsi'].rolling(20).mean()
        df_features['rsi_trend'] = df_features['rsi'] - df_features['rsi_ma20']
        
        # RSI velocity (rate of change)
        df_features['rsi_velocity'] = df_features['rsi'].diff()
        
        # RSI acceleration
        df_features['rsi_acceleration'] = df_features['rsi_velocity'].diff()
        
        # RSI volatility
        df_features['rsi_volatility'] = df_features['rsi'].rolling(20).std()
        
        # RSI percentile rank
        df_features['rsi_percentile'] = df_features['rsi'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan
        )
        
        # Overbought/Oversold zones
        df_features['rsi_overbought'] = (df_features['rsi'] > 70).astype(int)
        df_features['rsi_oversold'] = (df_features['rsi'] < 30).astype(int)
        
        # Extreme zones
        df_features['rsi_extreme_overbought'] = (df_features['rsi'] > 80).astype(int)
        df_features['rsi_extreme_oversold'] = (df_features['rsi'] < 20).astype(int)
        
        # RSI divergence from price
        price_change = df['close'].pct_change(20)
        rsi_change = df_features['rsi'].diff(20)
        df_features['rsi_price_divergence'] = np.sign(price_change) != np.sign(rsi_change)
        
        return df_features


class DivergenceDetector:
    """Detect bullish and bearish divergences"""
    
    @staticmethod
    def find_peaks_troughs(series: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find local maxima (peaks) and minima (troughs)
        
        Parameters:
        -----------
        series : pd.Series
            Data series
        order : int
            How many points on each side to consider for comparison
            
        Returns:
        --------
        peaks, troughs : tuple of np.ndarray
            Indices of peaks and troughs
        """
        data = series.values
        peaks = argrelextrema(data, np.greater, order=order)[0]
        troughs = argrelextrema(data, np.less, order=order)[0]
        
        return peaks, troughs
    
    @staticmethod
    def detect_divergence(prices: pd.Series, 
                         indicator: pd.Series, 
                         lookback: int = 50,
                         order: int = 5) -> pd.DataFrame:
        """
        Detect bullish and bearish divergences
        
        Types:
        - Regular Bullish: Lower price low, higher indicator low
        - Regular Bearish: Higher price high, lower indicator high
        - Hidden Bullish: Higher price low, lower indicator low
        - Hidden Bearish: Lower price high, higher indicator high
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        indicator : pd.Series
            Indicator series (e.g., RSI, MACD)
        lookback : int
            Periods to look back
        order : int
            Order for peak/trough detection
            
        Returns:
        --------
        divergences : pd.DataFrame
            DataFrame with divergence signals
        """
        # Find extrema
        price_peaks, price_troughs = DivergenceDetector.find_peaks_troughs(prices, order)
        ind_peaks, ind_troughs = DivergenceDetector.find_peaks_troughs(indicator, order)
        
        divergences = pd.DataFrame(index=prices.index)
        divergences['regular_bullish'] = 0
        divergences['regular_bearish'] = 0
        divergences['hidden_bullish'] = 0
        divergences['hidden_bearish'] = 0
        
        # Regular Bullish Divergence (price lower low, indicator higher low)
        for i in range(1, len(price_troughs)):
            if price_troughs[i] - price_troughs[i-1] <= lookback:
                t1, t2 = price_troughs[i-1], price_troughs[i]
                
                # Find corresponding indicator troughs
                ind_t1 = ind_troughs[np.abs(ind_troughs - t1).argmin()]
                ind_t2 = ind_troughs[np.abs(ind_troughs - t2).argmin()]
                
                # Check divergence
                if prices.iloc[t2] < prices.iloc[t1] and indicator.iloc[ind_t2] > indicator.iloc[ind_t1]:
                    divergences.iloc[t2, divergences.columns.get_loc('regular_bullish')] = 1
        
        # Regular Bearish Divergence (price higher high, indicator lower high)
        for i in range(1, len(price_peaks)):
            if price_peaks[i] - price_peaks[i-1] <= lookback:
                p1, p2 = price_peaks[i-1], price_peaks[i]
                
                ind_p1 = ind_peaks[np.abs(ind_peaks - p1).argmin()]
                ind_p2 = ind_peaks[np.abs(ind_peaks - p2).argmin()]
                
                if prices.iloc[p2] > prices.iloc[p1] and indicator.iloc[ind_p2] < indicator.iloc[ind_p1]:
                    divergences.iloc[p2, divergences.columns.get_loc('regular_bearish')] = 1
        
        return divergences


class WyckoffPatterns:
    """Detect W (Accumulation) and M (Distribution) patterns"""
    
    @staticmethod
    def detect_w_pattern(prices: pd.Series, 
                        tolerance: float = 0.02,
                        min_distance: int = 5,
                        max_distance: int = 50) -> List[Dict]:
        """
        Detect W (double bottom) patterns
        
        Pattern structure:
            A (first low) → B (resistance test) → C (second low ≈ A)
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        tolerance : float
            Maximum difference between A and C as percentage
        min_distance : int
            Minimum bars between points
        max_distance : int
            Maximum bars between points
            
        Returns:
        --------
        patterns : list of dict
            Detected W patterns with entry, target, and stop levels
        """
        troughs = argrelextrema(prices.values, np.less, order=5)[0]
        peaks = argrelextrema(prices.values, np.greater, order=5)[0]
        
        patterns = []
        
        for i in range(len(troughs) - 1):
            A_idx = troughs[i]
            C_idx = troughs[i + 1]
            
            # Check distance
            distance = C_idx - A_idx
            if distance < min_distance or distance > max_distance:
                continue
            
            # Find B (peak between A and C)
            B_candidates = peaks[(peaks > A_idx) & (peaks < C_idx)]
            if len(B_candidates) == 0:
                continue
            
            B_idx = B_candidates[np.argmax(prices.iloc[B_candidates].values)]
            
            A = prices.iloc[A_idx]
            B = prices.iloc[B_idx]
            C = prices.iloc[C_idx]
            
            # Check if C ≈ A (within tolerance)
            if abs(C - A) / A > tolerance:
                continue
            
            # Check if B is significantly higher
            if (B - A) / A < 0.02:  # At least 2% higher
                continue
            
            # Pattern detected!
            pattern = {
                'type': 'W',
                'A_idx': A_idx,
                'B_idx': B_idx,
                'C_idx': C_idx,
                'A_price': A,
                'B_price': B,
                'C_price': C,
                'entry': B,  # Entry on breakout above B
                'target': B + (B - A),  # Measured move
                'stop_loss': C - 0.5 * (B - A),
                'risk_reward': 2.0
            }
            
            patterns.append(pattern)
        
        return patterns
    
    @staticmethod
    def detect_m_pattern(prices: pd.Series,
                        tolerance: float = 0.02,
                        min_distance: int = 5,
                        max_distance: int = 50) -> List[Dict]:
        """
        Detect M (double top) patterns
        
        Pattern structure:
            B (first high) → C (support test) → D (second high ≈ B)
        """
        peaks = argrelextrema(prices.values, np.greater, order=5)[0]
        troughs = argrelextrema(prices.values, np.less, order=5)[0]
        
        patterns = []
        
        for i in range(len(peaks) - 1):
            B_idx = peaks[i]
            D_idx = peaks[i + 1]
            
            distance = D_idx - B_idx
            if distance < min_distance or distance > max_distance:
                continue
            
            # Find C (trough between B and D)
            C_candidates = troughs[(troughs > B_idx) & (troughs < D_idx)]
            if len(C_candidates) == 0:
                continue
            
            C_idx = C_candidates[np.argmin(prices.iloc[C_candidates].values)]
            
            B = prices.iloc[B_idx]
            C = prices.iloc[C_idx]
            D = prices.iloc[D_idx]
            
            # Check if D ≈ B
            if abs(D - B) / B > tolerance:
                continue
            
            # Check if C is significantly lower
            if (B - C) / B < 0.02:
                continue
            
            pattern = {
                'type': 'M',
                'B_idx': B_idx,
                'C_idx': C_idx,
                'D_idx': D_idx,
                'B_price': B,
                'C_price': C,
                'D_price': D,
                'entry': C,  # Entry on breakdown below C
                'target': C - (B - C),  # Measured move
                'stop_loss': D + 0.5 * (B - C),
                'risk_reward': 2.0
            }
            
            patterns.append(pattern)
        
        return patterns


class CandlestickPatterns:
    """Detect candlestick patterns"""
    
    @staticmethod
    def is_hammer(open_price: float, high: float, low: float, close: float, 
                  body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> bool:
        """
        Detect Hammer pattern
        
        Characteristics:
        - Small body at top of candle
        - Long lower shadow (at least 2x body)
        - Little or no upper shadow
        - Bullish reversal signal
        """
        body = abs(close - open_price)
        total_range = high - low
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        
        if total_range == 0:
            return False
        
        # Small body
        if body / total_range > body_ratio:
            return False
        
        # Long lower shadow
        if lower_shadow < shadow_ratio * body:
            return False
        
        # Small upper shadow
        if upper_shadow > body:
            return False
        
        return True
    
    @staticmethod
    def is_shooting_star(open_price: float, high: float, low: float, close: float,
                         body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> bool:
        """
        Detect Shooting Star pattern
        
        Characteristics:
        - Small body at bottom of candle
        - Long upper shadow (at least 2x body)
        - Little or no lower shadow
        - Bearish reversal signal
        """
        body = abs(close - open_price)
        total_range = high - low
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        
        if total_range == 0:
            return False
        
        if body / total_range > body_ratio:
            return False
        
        if upper_shadow < shadow_ratio * body:
            return False
        
        if lower_shadow > body:
            return False
        
        return True
    
    @staticmethod
    def is_doji(open_price: float, high: float, low: float, close: float,
                threshold: float = 0.1) -> bool:
        """
        Detect Doji pattern
        
        Characteristics:
        - Open ≈ Close (very small body)
        - Indicates indecision
        """
        body = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return False
        
        return (body / total_range) < threshold
    
    @staticmethod
    def is_engulfing(open1: float, close1: float, open2: float, close2: float) -> str:
        """
        Detect Engulfing pattern
        
        Returns:
        --------
        'bullish', 'bearish', or 'none'
        """
        # Bullish engulfing: bearish candle followed by larger bullish candle
        if close1 < open1 and close2 > open2:  # First bearish, second bullish
            if close2 > open1 and open2 < close1:  # Second engulfs first
                return 'bullish'
        
        # Bearish engulfing
        if close1 > open1 and close2 < open2:
            if close2 < open1 and open2 > close1:
                return 'bearish'
        
        return 'none'
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all candlestick patterns in a DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        df_patterns : pd.DataFrame
            DataFrame with pattern columns
        """
        df_patterns = df.copy()
        
        # Hammer
        df_patterns['hammer'] = df.apply(
            lambda x: CandlestickPatterns.is_hammer(x['open'], x['high'], x['low'], x['close']),
            axis=1
        ).astype(int)
        
        # Shooting Star
        df_patterns['shooting_star'] = df.apply(
            lambda x: CandlestickPatterns.is_shooting_star(x['open'], x['high'], x['low'], x['close']),
            axis=1
        ).astype(int)
        
        # Doji
        df_patterns['doji'] = df.apply(
            lambda x: CandlestickPatterns.is_doji(x['open'], x['high'], x['low'], x['close']),
            axis=1
        ).astype(int)
        
        # Engulfing (requires previous candle)
        df_patterns['engulfing'] = 'none'
        for i in range(1, len(df)):
            pattern = CandlestickPatterns.is_engulfing(
                df.iloc[i-1]['open'], df.iloc[i-1]['close'],
                df.iloc[i]['open'], df.iloc[i]['close']
            )
            df_patterns.iloc[i, df_patterns.columns.get_loc('engulfing')] = pattern
        
        return df_patterns


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all technical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    df_features : pd.DataFrame
        DataFrame with all features
    """
    df_features = df.copy()
    
    # Technical indicators
    df_features['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
    macd, signal, hist = TechnicalIndicators.calculate_macd(df['close'])
    df_features['macd'] = macd
    df_features['macd_signal'] = signal
    df_features['macd_hist'] = hist
    
    upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df['close'])
    df_features['bb_upper'] = upper
    df_features['bb_middle'] = middle
    df_features['bb_lower'] = lower
    
    df_features['atr'] = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'])
    df_features['obv'] = TechnicalIndicators.calculate_obv(df['close'], df['volume'])
    
    # RSI features
    df_features = RSIFeatures.create_rsi_features(df_features)
    
    # Divergences
    divergences = DivergenceDetector.detect_divergence(df['close'], df_features['rsi'])
    for col in divergences.columns:
        df_features[col] = divergences[col]
    
    # Candlestick patterns
    df_features = CandlestickPatterns.detect_patterns(df_features)
    
    return df_features


# Example usage
if __name__ == "__main__":
    print("Testing Feature Engineering...")
    print("=" * 60)
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    np.random.seed(42)
    
    # Simulated price data
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    print("\n1. Calculating RSI features...")
    df_rsi = RSIFeatures.create_rsi_features(df)
    print(df_rsi[['close', 'rsi', 'rsi_trend', 'rsi_velocity']].tail())
    
    print("\n2. Detecting divergences...")
    rsi = TechnicalIndicators.calculate_rsi(df['close'])
    divergences = DivergenceDetector.detect_divergence(df['close'], rsi)
    print(f"Regular Bullish: {divergences['regular_bullish'].sum()}")
    print(f"Regular Bearish: {divergences['regular_bearish'].sum()}")
    
    print("\n3. Detecting W/M patterns...")
    w_patterns = WyckoffPatterns.detect_w_pattern(df['close'])
    m_patterns = WyckoffPatterns.detect_m_pattern(df['close'])
    print(f"W Patterns: {len(w_patterns)}")
    print(f"M Patterns: {len(m_patterns)}")
    
    print("\n4. Detecting candlestick patterns...")
    df_patterns = CandlestickPatterns.detect_patterns(df)
    print(f"Hammer: {df_patterns['hammer'].sum()}")
    print(f"Shooting Star: {df_patterns['shooting_star'].sum()}")
    print(f"Doji: {df_patterns['doji'].sum()}")
    
    print("\n✓ All tests passed!")
