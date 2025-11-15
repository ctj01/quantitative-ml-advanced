"""
Complete Volatility-Based Trading System
=========================================

This mini-project integrates:
- GARCH volatility forecasting
- RSI with regime detection
- W/M pattern recognition
- Divergence analysis
- Candlestick patterns
- Risk management
- Backtesting

Author: Cristian Mendoza
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.models.garch import GARCHModel, EGARCHModel
from src.data.loaders import load_financial_data, split_train_test
from src.features.technical_indicators import (
    TechnicalIndicators, RSIFeatures, DivergenceDetector,
    WyckoffPatterns, CandlestickPatterns, create_all_features
)


class VolatilityTradingSystem:
    """
    Complete trading system combining volatility forecasting with technical analysis
    """
    
    def __init__(self, 
                 symbol: str = 'BTC-USD',
                 initial_capital: float = 100000,
                 target_volatility: float = 0.02,
                 max_position: float = 1.0):
        """
        Initialize trading system
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        initial_capital : float
            Initial capital in USD
        target_volatility : float
            Target daily volatility for position sizing
        max_position : float
            Maximum position as fraction of capital
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.target_volatility = target_volatility
        self.max_position = max_position
        
        # Models
        self.garch_model = None
        self.egarch_model = None
        
        # Data
        self.data = None
        self.train_data = None
        self.test_data = None
        
        # Results
        self.trades = []
        self.equity_curve = []
        self.signals = None
        
    def load_data(self, period: str = '2y'):
        """Load and prepare data"""
        print(f"Loading data for {self.symbol}...")
        self.data = load_financial_data(self.symbol, period=period)
        
        # Add all technical features
        self.data = create_all_features(self.data)
        
        # Split train/test
        self.train_data, self.test_data = split_train_test(self.data, train_size=0.7)
        
        print(f"✓ Data loaded: {len(self.train_data)} train, {len(self.test_data)} test observations")
        
    def fit_volatility_models(self):
        """Fit GARCH and EGARCH models"""
        print("\nFitting volatility models...")
        
        # GARCH(1,1)
        self.garch_model = GARCHModel(p=1, q=1, mean_model='constant')
        self.garch_model.fit(self.train_data['returns'] * 100)  # Scale to percentage
        print("\n✓ GARCH model fitted")
        
        # EGARCH(1,1)
        self.egarch_model = EGARCHModel(p=1, q=1)
        self.egarch_model.fit(self.train_data['returns'] * 100)
        print("✓ EGARCH model fitted")
        
    def detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime using multiple indicators
        
        Returns:
        --------
        regime : pd.Series
            'Bull', 'Bear', or 'Neutral'
        """
        regime = pd.Series(index=df.index, data='Neutral')
        
        # Trend: Price vs MA50
        df['ma50'] = df['close'].rolling(50).mean()
        trend_up = df['close'] > df['ma50']
        
        # Momentum: RSI
        strong_momentum_up = df['rsi'] > 60
        strong_momentum_down = df['rsi'] < 40
        
        # Volatility: Compare to average
        avg_vol = df['realized_vol'].rolling(50).mean()
        high_vol = df['realized_vol'] > avg_vol * 1.2
        
        # Bull regime: Uptrend + Strong momentum + Not high vol
        bull = trend_up & strong_momentum_up & ~high_vol
        regime[bull] = 'Bull'
        
        # Bear regime: Downtrend + Weak momentum + High vol
        bear = ~trend_up & strong_momentum_down & high_vol
        regime[bear] = 'Bear'
        
        return regime
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals combining multiple factors
        
        Signal Components:
        1. RSI levels (regime-adjusted)
        2. W/M patterns
        3. Divergences
        4. Candlestick patterns
        5. Volatility forecast
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0
        signals['confidence'] = 0.0
        signals['position_size'] = 0.0
        
        # Detect regime
        regime = self.detect_market_regime(df)
        signals['regime'] = regime
        
        # Component 1: RSI signals (regime-dependent)
        rsi_signal = np.zeros(len(df))
        
        for i in range(len(df)):
            if regime.iloc[i] == 'Bull':
                # In bull market, buy on pullbacks
                if df['rsi'].iloc[i] < 45:
                    rsi_signal[i] = 1.0
                elif df['rsi'].iloc[i] > 75:
                    rsi_signal[i] = -0.5  # Partial exit
            
            elif regime.iloc[i] == 'Bear':
                # In bear market, sell on bounces
                if df['rsi'].iloc[i] > 55:
                    rsi_signal[i] = -1.0
                elif df['rsi'].iloc[i] < 25:
                    rsi_signal[i] = 0.5  # Cautious buy
            
            else:  # Neutral
                # Traditional RSI levels
                if df['rsi'].iloc[i] < 30:
                    rsi_signal[i] = 0.8
                elif df['rsi'].iloc[i] > 70:
                    rsi_signal[i] = -0.8
        
        signals['rsi_signal'] = rsi_signal
        
        # Component 2: Divergence signals
        div_signal = np.zeros(len(df))
        div_signal[df['regular_bullish'] == 1] = 1.0
        div_signal[df['regular_bearish'] == 1] = -1.0
        signals['divergence_signal'] = div_signal
        
        # Component 3: Candlestick signals
        candle_signal = np.zeros(len(df))
        candle_signal[df['hammer'] == 1] = 0.5
        candle_signal[df['shooting_star'] == 1] = -0.5
        candle_signal[df['engulfing'] == 'bullish'] = 0.7
        candle_signal[df['engulfing'] == 'bearish'] = -0.7
        signals['candle_signal'] = candle_signal
        
        # Component 4: W/M patterns
        # (Would require forward-looking validation in real trading)
        w_patterns = WyckoffPatterns.detect_w_pattern(df['close'])
        m_patterns = WyckoffPatterns.detect_m_pattern(df['close'])
        
        pattern_signal = np.zeros(len(df))
        for pattern in w_patterns:
            if pattern['C_idx'] < len(df):
                pattern_signal[pattern['C_idx']] = 1.0
        for pattern in m_patterns:
            if pattern['D_idx'] < len(df):
                pattern_signal[pattern['D_idx']] = -1.0
        signals['pattern_signal'] = pattern_signal
        
        # Aggregate signals with weights
        weights = {
            'rsi': 0.3,
            'divergence': 0.25,
            'candle': 0.20,
            'pattern': 0.25
        }
        
        signals['signal'] = (
            weights['rsi'] * signals['rsi_signal'] +
            weights['divergence'] * signals['divergence_signal'] +
            weights['candle'] * signals['candle_signal'] +
            weights['pattern'] * signals['pattern_signal']
        )
        
        # Calculate confidence (how many components agree)
        agreement = (
            (np.sign(signals['rsi_signal']) == np.sign(signals['signal'])).astype(int) +
            (np.sign(signals['divergence_signal']) == np.sign(signals['signal'])).astype(int) +
            (np.sign(signals['candle_signal']) == np.sign(signals['signal'])).astype(int) +
            (np.sign(signals['pattern_signal']) == np.sign(signals['signal'])).astype(int)
        )
        signals['confidence'] = agreement / 4.0
        
        return signals
    
    def calculate_position_size(self, signal: float, confidence: float, vol_forecast: float) -> float:
        """
        Calculate position size using inverse volatility weighting
        
        Parameters:
        -----------
        signal : float
            Trading signal (-1 to 1)
        confidence : float
            Signal confidence (0 to 1)
        vol_forecast : float
            Forecasted volatility
            
        Returns:
        --------
        position_size : float
            Position size as fraction of capital
        """
        # Base position size (inverse volatility)
        base_size = self.target_volatility / (vol_forecast / 100)
        
        # Adjust by signal strength and confidence
        position_size = base_size * abs(signal) * confidence
        
        # Apply limits
        position_size = min(position_size, self.max_position)
        position_size = max(position_size, 0)
        
        # Apply signal direction
        position_size *= np.sign(signal)
        
        return position_size
    
    def backtest(self):
        """
        Run backtest on test data
        """
        print("\nRunning backtest...")
        
        # Generate signals for test data
        self.signals = self.generate_signals(self.test_data)
        
        # Initialize tracking
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.capital = self.initial_capital
        position = 0
        entry_price = 0
        
        # Forecast volatility for position sizing
        vol_forecast = self.garch_model.forecast(horizon=1)[0]
        
        for i in range(1, len(self.test_data)):
            date = self.test_data.index[i]
            price = self.test_data['close'].iloc[i]
            signal = self.signals['signal'].iloc[i]
            confidence = self.signals['confidence'].iloc[i]
            
            # Update volatility forecast periodically
            if i % 5 == 0:
                recent_returns = self.test_data['returns'].iloc[max(0,i-100):i] * 100
                temp_model = GARCHModel(p=1, q=1)
                try:
                    temp_model.fit(recent_returns)
                    vol_forecast = temp_model.forecast(horizon=1)[0]
                except:
                    pass  # Keep previous forecast
            
            # Trading logic
            if position == 0:  # No position
                if abs(signal) > 0.3 and confidence > 0.4:
                    # Open position
                    position_size = self.calculate_position_size(signal, confidence, vol_forecast)
                    position = position_size
                    entry_price = price
                    
                    trade = {
                        'entry_date': date,
                        'entry_price': entry_price,
                        'position': position,
                        'signal': signal,
                        'confidence': confidence,
                        'vol_forecast': vol_forecast
                    }
                    self.trades.append(trade)
            
            else:  # Have position
                # Exit conditions
                exit_trade = False
                exit_reason = ''
                
                # 1. Opposite signal
                if np.sign(position) != np.sign(signal) and abs(signal) > 0.3:
                    exit_trade = True
                    exit_reason = 'opposite_signal'
                
                # 2. Take profit (3x volatility)
                pnl_pct = (price / entry_price - 1) * np.sign(position)
                if pnl_pct > 3 * (vol_forecast / 100):
                    exit_trade = True
                    exit_reason = 'take_profit'
                
                # 3. Stop loss (1.5x volatility)
                if pnl_pct < -1.5 * (vol_forecast / 100):
                    exit_trade = True
                    exit_reason = 'stop_loss'
                
                # 4. Time-based exit (max 20 days)
                if i - self.test_data.index.get_loc(trade['entry_date']) > 20:
                    exit_trade = True
                    exit_reason = 'time_exit'
                
                if exit_trade:
                    # Close position
                    pnl = (price - entry_price) * position * self.capital
                    self.capital += pnl
                    
                    self.trades[-1].update({
                        'exit_date': date,
                        'exit_price': price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Update equity curve
            if position != 0:
                current_value = self.capital + (price - entry_price) * position * self.capital
            else:
                current_value = self.capital
            
            self.equity_curve.append(current_value)
        
        print(f"✓ Backtest complete: {len([t for t in self.trades if 'exit_date' in t])} trades executed")
        
    def analyze_performance(self):
        """
        Calculate and display performance metrics
        """
        closed_trades = [t for t in self.trades if 'exit_date' in t]
        
        if len(closed_trades) == 0:
            print("No completed trades to analyze")
            return
        
        trades_df = pd.DataFrame(closed_trades)
        
        # Performance metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        # Risk metrics
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Print results
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"\nTotal Trades: {len(trades_df)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Profit Factor: {abs(avg_win / avg_loss):.2f}" if avg_loss != 0 else "N/A")
        print(f"\nSharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        # Exit reasons
        print(f"\nExit Reasons:")
        for reason in trades_df['exit_reason'].value_counts().items():
            print(f"  {reason[0]}: {reason[1]}")
        
        print("="*70)
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_df': trades_df
        }
    
    def plot_results(self):
        """
        Create visualization of results
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 1. Equity curve
        axes[0].plot(self.equity_curve, linewidth=2, color='darkblue')
        axes[0].axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital')
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Capital ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Price with signals
        test_dates = self.test_data.index
        axes[1].plot(test_dates, self.test_data['close'], label='Price', linewidth=1.5, color='black')
        
        # Mark entry points
        for trade in self.trades:
            if 'exit_date' in trade:
                if trade['pnl'] > 0:
                    color = 'green'
                    marker = '^'
                else:
                    color = 'red'
                    marker = 'v'
                
                axes[1].scatter(trade['entry_date'], trade['entry_price'], 
                              color=color, marker=marker, s=100, alpha=0.7, zorder=5)
        
        axes[1].set_title('Price with Trade Signals', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. RSI with regime
        axes[2].plot(test_dates, self.test_data['rsi'], label='RSI', linewidth=1.5)
        axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.5)
        axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        axes[2].fill_between(test_dates, 70, 100, alpha=0.1, color='red')
        axes[2].fill_between(test_dates, 0, 30, alpha=0.1, color='green')
        axes[2].set_title('RSI Indicator', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('RSI')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Signal strength
        axes[3].plot(test_dates, self.signals['signal'], label='Composite Signal', linewidth=1.5)
        axes[3].fill_between(test_dates, 0, self.signals['signal'], alpha=0.3)
        axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[3].set_title('Trading Signal Strength', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Signal')
        axes[3].set_xlabel('Date')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trading_system_results.png', dpi=150, bbox_inches='tight')
        print("\n✓ Results saved to 'trading_system_results.png'")
        plt.show()
    
    def run_full_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("\n" + "="*70)
        print("VOLATILITY-BASED TRADING SYSTEM")
        print("="*70)
        
        # Load data
        self.load_data(period='2y')
        
        # Fit models
        self.fit_volatility_models()
        
        # Backtest
        self.backtest()
        
        # Analyze
        results = self.analyze_performance()
        
        # Visualize
        self.plot_results()
        
        return results


def main():
    """
    Main execution function
    """
    # Create trading system
    system = VolatilityTradingSystem(
        symbol='BTC-USD',
        initial_capital=100000,
        target_volatility=0.02,  # 2% daily target
        max_position=1.0
    )
    
    # Run full analysis
    results = system.run_full_analysis()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Volatility forecasting improves position sizing")
    print("2. Multi-signal approach reduces false signals")
    print("3. Regime detection adapts strategy to market conditions")
    print("4. Risk management is crucial for long-term success")
    print("\nNext Steps:")
    print("- Optimize parameters using walk-forward analysis")
    print("- Add more asset classes for diversification")
    print("- Implement real-time monitoring")
    print("- Paper trade before live deployment")
    print("="*70)


if __name__ == "__main__":
    main()
