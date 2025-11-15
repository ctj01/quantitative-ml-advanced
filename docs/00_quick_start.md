# Quick Start Guide: Quantitative ML Advanced

## 🚀 Get Started in 10 Minutes

This guide will get you up and running with volatility modeling and trading strategies.

### Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/ctj01/quantitative-ml-advanced.git
cd quantitative-ml-advanced

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import arch; import yfinance; print('✓ Installation successful!')"
```

### Step 2: Your First GARCH Model (3 minutes)

```python
# Import libraries
from src.models.garch import GARCHModel
from src.data.loaders import load_financial_data

# Load Bitcoin data
data = load_financial_data('BTC-USD', period='1y')
returns = data['returns'] * 100  # Convert to percentage

# Fit GARCH(1,1) model
model = GARCHModel(p=1, q=1, mean_model='constant')
model.fit(returns)
model.summary()

# Forecast next 10 days of volatility
forecasts = model.forecast(horizon=10)
print("Volatility forecasts:", forecasts)
```

**Output**: You'll see parameter estimates and volatility forecasts!

### Step 3: Generate Trading Signals (3 minutes)

```python
from src.features.technical_indicators import TechnicalIndicators, RSIFeatures

# Calculate RSI
data['rsi'] = TechnicalIndicators.calculate_rsi(data['close'])

# Create advanced RSI features
data = RSIFeatures.create_rsi_features(data)

# Simple strategy: Buy when RSI < 30, Sell when RSI > 70
data['signal'] = 0
data.loc[data['rsi'] < 30, 'signal'] = 1  # Buy
data.loc[data['rsi'] > 70, 'signal'] = -1  # Sell

print(f"Buy signals: {(data['signal'] == 1).sum()}")
print(f"Sell signals: {(data['signal'] == -1).sum()}")
```

### Step 4: Run Complete Trading System (2 minutes)

```python
# Run the integrated trading system
import sys
sys.path.append('notebooks/08_projects')
from volatility_trading_system import VolatilityTradingSystem

# Create and run system
system = VolatilityTradingSystem(
    symbol='BTC-USD',
    initial_capital=100000
)

results = system.run_full_analysis()
```

**Output**: Complete backtest results with visualizations!

---

## 📊 Common Use Cases

### Use Case 1: Volatility Forecasting

```python
from src.models.garch import GARCHModel, EGARCHModel, compare_models

# Compare different volatility models
comparison = compare_models(returns, models=['ARCH', 'GARCH', 'EGARCH'])
print(comparison)

# Use best model for trading
best_model = GARCHModel(p=1, q=1).fit(returns)
vol_forecast = best_model.forecast(horizon=1)[0]

# Position sizing: Inverse volatility
target_vol = 0.02  # 2% daily target
position_size = target_vol / (vol_forecast / 100)
print(f"Recommended position size: {position_size:.2f}x")
```

### Use Case 2: Pattern Detection

```python
from src.features.technical_indicators import WyckoffPatterns, DivergenceDetector

# Detect W/M patterns
w_patterns = WyckoffPatterns.detect_w_pattern(data['close'])
print(f"Found {len(w_patterns)} W patterns")

for pattern in w_patterns:
    print(f"Entry: ${pattern['entry']:.2f}")
    print(f"Target: ${pattern['target']:.2f}")
    print(f"Stop Loss: ${pattern['stop_loss']:.2f}")
    print(f"Risk/Reward: {pattern['risk_reward']:.1f}")
    print("-" * 40)

# Detect divergences
rsi = TechnicalIndicators.calculate_rsi(data['close'])
divergences = DivergenceDetector.detect_divergence(data['close'], rsi)
print(f"Bullish divergences: {divergences['regular_bullish'].sum()}")
print(f"Bearish divergences: {divergences['regular_bearish'].sum()}")
```

### Use Case 3: Multi-Asset Analysis

```python
from src.data.loaders import load_multiple_assets, create_panel_data

# Load multiple cryptocurrencies
symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD']
crypto_data = load_multiple_assets(symbols, period='1y')

# Calculate correlations
returns_panel = create_panel_data(symbols, period='1y', column='returns')
correlation_matrix = returns_panel.corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Fit GARCH for each asset
for symbol, data in crypto_data.items():
    returns = data['returns'] * 100
    model = GARCHModel(p=1, q=1).fit(returns)
    vol_forecast = model.forecast(horizon=1)[0]
    print(f"{symbol}: Forecasted volatility = {vol_forecast:.2f}%")
```

### Use Case 4: Risk Management

```python
# Calculate VaR using GARCH forecast
def calculate_var_garch(returns, confidence=0.95):
    from scipy import stats
    
    # Fit GARCH
    model = GARCHModel(p=1, q=1).fit(returns * 100)
    vol_forecast = model.forecast(horizon=1)[0] / 100
    
    # Parametric VaR
    mean = returns.mean()
    var = -(mean + vol_forecast * stats.norm.ppf(1 - confidence))
    
    return var * 100  # Return as percentage

var_95 = calculate_var_garch(data['returns'])
var_99 = calculate_var_garch(data['returns'], confidence=0.99)

print(f"95% VaR: {var_95:.2f}%")
print(f"99% VaR: {var_99:.2f}%")
print(f"\nInterpretation: 95% confidence that daily loss won't exceed {var_95:.2f}%")
```

---

## 🎯 Learning Path

### Week 1-2: Foundations
- [ ] Read `docs/01_mathematical_foundations.md`
- [ ] Complete exercises 1.1 - 1.2 in `docs/03_exercises_and_challenges.md`
- [ ] Run `notebooks/01_time_series_advanced/01_garch_volatility_models.ipynb`

### Week 3-4: Implementation
- [ ] Complete exercises 2.1 - 2.2
- [ ] Implement your own ARCH model from scratch
- [ ] Compare your implementation with the library version

### Week 5-6: Trading Strategies
- [ ] Read `docs/02_trading_integration_guide.md`
- [ ] Complete exercises 3.1 - 3.2
- [ ] Modify the trading system for a different asset

### Week 7-8: Advanced Topics
- [ ] Study EGARCH and leverage effects
- [ ] Complete exercises 4.1 - 4.2
- [ ] Conduct original research (Exercise 5.1 or 5.2)

---

## 💡 Tips for Success

### 1. Start Simple
Don't try to understand everything at once. Master ARCH before GARCH, GARCH before EGARCH.

### 2. Always Visualize
```python
import matplotlib.pyplot as plt

# Visualize volatility clustering
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data['returns'])
plt.title('Returns (Note the clustering)')
plt.subplot(2, 1, 2)
plt.plot(np.sqrt(model.sigma2))
plt.title('Conditional Volatility')
plt.tight_layout()
plt.show()
```

### 3. Test Everything
```python
# Always validate on out-of-sample data
from src.data.loaders import split_train_test

train, test = split_train_test(data, train_size=0.7)

# Fit on train
model = GARCHModel(p=1, q=1).fit(train['returns'] * 100)

# Evaluate on test
forecasts = []
actual = []
for i in range(len(test) - 1):
    vol_forecast = model.forecast(horizon=1)[0]
    actual_vol = abs(test['returns'].iloc[i]) * 100
    forecasts.append(vol_forecast)
    actual.append(actual_vol)

mse = np.mean((np.array(forecasts) - np.array(actual))**2)
print(f"Out-of-sample MSE: {mse:.4f}")
```

### 4. Document Your Findings
Keep a trading journal:
- What strategies did you test?
- What were the results?
- What did you learn?
- What will you try next?

### 5. Be Patient
Quantitative trading is complex. Take your time to understand each concept deeply.

---

## 🐛 Troubleshooting

### Issue: "Model optimization did not converge"
**Solution**: Try different initial parameters or increase maxiter

```python
# If default doesn't work, try manual initialization
model = GARCHModel(p=1, q=1)
# Modify initial params in the fit method
```

### Issue: "Negative volatility forecasts"
**Solution**: Check your data for errors and ensure returns are properly scaled

```python
# Check data
print(returns.describe())
print(f"Min: {returns.min()}, Max: {returns.max()}")

# Ensure proper scaling
returns = data['returns'] * 100  # Scale to percentage
```

### Issue: "Import errors"
**Solution**: Make sure you're running from the project root

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

---

## 📚 Next Steps

After mastering the basics:

1. **Explore Other Notebooks**: 
   - `02_market_regimes/` for HMM
   - `03_transformers_time_series/` for deep learning
   - `04_feature_engineering/` for alpha factors

2. **Customize the Trading System**:
   - Add your own indicators
   - Implement different position sizing rules
   - Test on various assets

3. **Conduct Research**:
   - Investigate new volatility models
   - Test novel trading strategies
   - Publish your findings

4. **Paper Trade**:
   - Test strategies in real-time (without real money)
   - Track performance
   - Refine based on results

5. **Join the Community**:
   - Star the repo
   - Report issues
   - Contribute improvements

---

## 🎓 Resources

### Books
- Tsay, R.S. "Analysis of Financial Time Series"
- Lopez de Prado, M. "Advances in Financial Machine Learning"
- Chan, E. "Quantitative Trading"

### Papers
- Bollerslev (1986) - Original GARCH paper
- Nelson (1991) - EGARCH paper
- Engle (2001) - GARCH 101

### Online
- [Arch Documentation](https://arch.readthedocs.io/)
- [QuantStart Tutorials](https://www.quantstart.com/)
- [GitHub Discussions](https://github.com/ctj01/quantitative-ml-advanced/discussions)

---

**Ready to dive deeper? Open a notebook and start experimenting!**

*Remember: Past performance does not guarantee future results. Always test thoroughly before risking real capital.*
