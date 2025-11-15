# Exercises: Mastering Volatility Models and Trading

## 🎯 Exercise Set 1: ARCH/GARCH Fundamentals (Beginner)

### Exercise 1.1: Manual ARCH(1) Calculation
**Difficulty**: ⭐⭐

Given the following returns and ARCH(1) parameters:
- Returns: [0.02, -0.01, 0.03, -0.02, 0.01]
- Parameters: ω = 0.0001, α = 0.3

Calculate the conditional variance σ²_t for each time period manually.

**Hints**:
- Start with σ²_0 = unconditional variance
- Use: σ²_t = ω + α * r²_{t-1}
- Check your work: volatility should increase after large returns

<details>
<summary>Solution</summary>

```python
import numpy as np

returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
omega = 0.0001
alpha = 0.3

# Initial variance
sigma2 = np.zeros(len(returns) + 1)
sigma2[0] = np.var(returns)  # Unconditional variance

# Calculate
for t in range(1, len(returns) + 1):
    sigma2[t] = omega + alpha * returns[t-1]**2

print("Conditional Variances:", sigma2[1:])
print("Conditional Volatilities:", np.sqrt(sigma2[1:]))

# Expected results:
# t=1: 0.0001 + 0.3 * (0.02)² = 0.00022
# t=2: 0.0001 + 0.3 * (-0.01)² = 0.00013
# t=3: 0.0001 + 0.3 * (0.03)² = 0.00037
# etc.
```
</details>

---

### Exercise 1.2: GARCH(1,1) Stationarity
**Difficulty**: ⭐⭐

For a GARCH(1,1) model with parameters ω = 0.00001, α = 0.12, β = 0.85:

a) Check if the model is stationary
b) Calculate the unconditional variance
c) Calculate the persistence (half-life of shocks)

**Concepts tested**: Stationarity condition, long-run variance

<details>
<summary>Solution</summary>

```python
omega = 0.00001
alpha = 0.12
beta = 0.85

# a) Stationarity: α + β < 1
persistence = alpha + beta
is_stationary = persistence < 1

print(f"Persistence: {persistence}")
print(f"Stationary: {is_stationary}")

# b) Unconditional variance: ω / (1 - α - β)
if is_stationary:
    unconditional_var = omega / (1 - persistence)
    unconditional_vol = np.sqrt(unconditional_var)
    print(f"Unconditional Variance: {unconditional_var:.6f}")
    print(f"Unconditional Volatility: {unconditional_vol:.4f} ({unconditional_vol*100:.2f}%)")

# c) Half-life: log(0.5) / log(α + β)
half_life = np.log(0.5) / np.log(persistence)
print(f"Half-life of shocks: {half_life:.2f} periods")

# Interpretation:
# Persistence = 0.97 → Very persistent (realistic for daily financial data)
# Half-life ≈ 22 days → Shocks decay slowly
```
</details>

---

## 🎯 Exercise Set 2: Model Estimation and Comparison (Intermediate)

### Exercise 2.1: Implement ARCH Likelihood
**Difficulty**: ⭐⭐⭐

Write a function to compute the log-likelihood for an ARCH(1) model. Then use it with scipy.optimize to estimate parameters.

**Data**: Use 500 days of simulated returns with known parameters (ω=0.0001, α=0.15)

**Requirements**:
- Implement `log_likelihood_arch()` function
- Use `minimize()` to find optimal parameters
- Compare estimated vs true parameters

<details>
<summary>Solution</summary>

```python
from scipy.optimize import minimize
import numpy as np

def simulate_arch(n, omega, alpha, seed=42):
    """Simulate ARCH(1) process"""
    np.random.seed(seed)
    epsilon = np.random.standard_normal(n)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2
        returns[t] = np.sqrt(sigma2[t]) * epsilon[t]
    
    return returns

def log_likelihood_arch(params, returns):
    """Compute negative log-likelihood for ARCH(1)"""
    omega, alpha = params
    
    # Check constraints
    if omega <= 0 or alpha < 0 or alpha >= 1:
        return 1e10
    
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2
    
    sigma2 = np.maximum(sigma2, 1e-8)
    
    # Log-likelihood
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    
    return -ll  # Return negative for minimization

# Generate data
true_omega = 0.0001
true_alpha = 0.15
returns = simulate_arch(500, true_omega, true_alpha)

# Estimate
initial_guess = [0.0005, 0.1]
result = minimize(
    log_likelihood_arch,
    initial_guess,
    args=(returns,),
    method='L-BFGS-B',
    bounds=[(1e-6, None), (0, 0.999)]
)

print("True parameters:", true_omega, true_alpha)
print("Estimated parameters:", result.x)
print("Estimation error:", np.abs(result.x - [true_omega, true_alpha]))
```
</details>

---

### Exercise 2.2: Model Selection
**Difficulty**: ⭐⭐⭐

Given Bitcoin returns, compare ARCH(1), GARCH(1,1), and EGARCH(1,1) using:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Forecast accuracy (out-of-sample)

**Deliverable**: Table showing which model performs best and why.

<details>
<summary>Solution Framework</summary>

```python
from src.models.garch import ARCHModel, GARCHModel, EGARCHModel
from src.data.loaders import load_financial_data

# Load data
data = load_financial_data('BTC-USD', period='2y')
returns = data['returns'] * 100

# Split
train_size = int(len(returns) * 0.8)
train = returns.iloc[:train_size]
test = returns.iloc[train_size:]

# Fit models
models = {
    'ARCH(1)': ARCHModel(q=1).fit(train),
    'GARCH(1,1)': GARCHModel(p=1, q=1).fit(train),
    'EGARCH(1,1)': EGARCHModel(p=1, q=1).fit(train)
}

# Compare
results = []
for name, model in models.items():
    # Information criteria
    T = len(train)
    k = {'ARCH(1)': 2, 'GARCH(1,1)': 3, 'EGARCH(1,1)': 4}[name]
    ll = model.loglikelihood
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(T)
    
    # Forecast accuracy
    forecasts = []
    realized = []
    for i in range(len(test) - 1):
        vol_forecast = model.forecast(horizon=1)[0]
        forecasts.append(vol_forecast)
        realized.append(abs(test.iloc[i]))
    
    mse = np.mean((np.array(forecasts) - np.array(realized))**2)
    
    results.append({
        'Model': name,
        'AIC': aic,
        'BIC': bic,
        'Forecast MSE': mse
    })

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)

# Best model: Lowest AIC/BIC, lowest forecast error
```
</details>

---

## 🎯 Exercise Set 3: Trading Strategy Development (Advanced)

### Exercise 3.1: Volatility Timing Strategy
**Difficulty**: ⭐⭐⭐⭐

Develop a strategy that:
1. Forecasts volatility using GARCH(1,1)
2. Increases position size when volatility is LOW
3. Decreases position size when volatility is HIGH
4. Uses RSI for entry/exit timing

**Performance targets**:
- Sharpe ratio > 1.0
- Max drawdown < 20%
- Win rate > 45%

<details>
<summary>Solution Framework</summary>

```python
class VolatilityTimingStrategy:
    def __init__(self, target_vol=0.02, max_leverage=2.0):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.garch = GARCHModel(p=1, q=1)
        
    def calculate_position_size(self, vol_forecast):
        """Inverse volatility weighting"""
        position = self.target_vol / (vol_forecast / 100)
        return min(position, self.max_leverage)
    
    def generate_signals(self, df):
        """RSI-based signals"""
        signals = pd.Series(0, index=df.index)
        rsi = df['rsi']
        
        # Buy when RSI < 30
        signals[rsi < 30] = 1
        
        # Sell when RSI > 70
        signals[rsi > 70] = -1
        
        return signals
    
    def backtest(self, data):
        # Fit GARCH on training data
        train = data.iloc[:int(len(data)*0.5)]
        self.garch.fit(train['returns'] * 100)
        
        # Trade on test data
        test = data.iloc[int(len(data)*0.5):]
        signals = self.generate_signals(test)
        
        # Calculate positions
        positions = []
        for i in range(len(test)):
            vol_forecast = self.garch.forecast(horizon=1)[0]
            position_size = self.calculate_position_size(vol_forecast)
            position = signals.iloc[i] * position_size
            positions.append(position)
        
        # Calculate returns
        strategy_returns = np.array(positions) * test['returns'].values
        
        # Performance metrics
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        cumulative = (1 + strategy_returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        max_dd = drawdown.min()
        
        win_rate = (strategy_returns > 0).mean()
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_return': (cumulative.iloc[-1] - 1) * 100
        }

# Test strategy
strategy = VolatilityTimingStrategy()
results = strategy.backtest(data)
print(results)
```
</details>

---

### Exercise 3.2: Multi-Timeframe Analysis
**Difficulty**: ⭐⭐⭐⭐

Create a strategy that uses:
- Daily GARCH for volatility regime
- 4-hour RSI for entry timing
- 1-hour candlestick patterns for confirmation

**Challenge**: Handle multiple timeframes correctly (no look-ahead bias!)

---

### Exercise 3.3: Regime-Dependent Strategy
**Difficulty**: ⭐⭐⭐⭐⭐

Build a complete system that:
1. Uses HMM to detect regimes (bull/bear/neutral)
2. Applies different RSI thresholds per regime
3. Adjusts position sizing based on EGARCH forecasts
4. Implements dynamic stop-losses based on ATR

**Bonus**: Add transaction costs and slippage modeling

---

## 🎯 Exercise Set 4: Risk Management (Advanced)

### Exercise 4.1: VaR Calculation
**Difficulty**: ⭐⭐⭐

Calculate 95% and 99% Value-at-Risk using:
a) Historical simulation
b) Parametric method (using GARCH forecast)
c) Monte Carlo simulation

Compare the three methods.

<details>
<summary>Solution</summary>

```python
def calculate_var(returns, confidence=0.95, method='historical'):
    """
    Calculate Value-at-Risk
    
    Returns:
    --------
    var : float
        VaR at specified confidence level (positive number)
    """
    if method == 'historical':
        # Historical VaR
        var = -np.percentile(returns, (1 - confidence) * 100)
        
    elif method == 'parametric':
        # Parametric VaR (assuming normality)
        from scipy import stats
        mean = returns.mean()
        std = returns.std()
        var = -(mean + std * stats.norm.ppf(1 - confidence))
        
    elif method == 'parametric_garch':
        # Parametric VaR with GARCH volatility
        garch = GARCHModel(p=1, q=1).fit(returns * 100)
        vol_forecast = garch.forecast(horizon=1)[0] / 100
        mean = returns.mean()
        var = -(mean + vol_forecast * stats.norm.ppf(1 - confidence))
        
    elif method == 'monte_carlo':
        # Monte Carlo VaR
        garch = GARCHModel(p=1, q=1).fit(returns * 100)
        vol_forecast = garch.forecast(horizon=1)[0] / 100
        
        # Simulate 10,000 scenarios
        simulated_returns = np.random.normal(returns.mean(), vol_forecast, 10000)
        var = -np.percentile(simulated_returns, (1 - confidence) * 100)
    
    return var

# Test
returns = data['returns'].dropna()

var_95_hist = calculate_var(returns, 0.95, 'historical')
var_95_param = calculate_var(returns, 0.95, 'parametric')
var_95_garch = calculate_var(returns, 0.95, 'parametric_garch')
var_95_mc = calculate_var(returns, 0.95, 'monte_carlo')

print(f"95% VaR (Historical): {var_95_hist:.4f}")
print(f"95% VaR (Parametric): {var_95_param:.4f}")
print(f"95% VaR (GARCH): {var_95_garch:.4f}")
print(f"95% VaR (Monte Carlo): {var_95_mc:.4f}")
```
</details>

---

### Exercise 4.2: Kelly Criterion Position Sizing
**Difficulty**: ⭐⭐⭐⭐

Implement Kelly Criterion for position sizing:
- Use historical win rate and profit/loss ratio
- Add fractional Kelly (e.g., 0.5 Kelly) for safety
- Backtest with different Kelly fractions

**Formula**: f* = (bp - q) / b
- where b = profit/loss ratio, p = win probability, q = 1-p

---

## 🎯 Exercise Set 5: Research Projects (Expert)

### Project 5.1: Leverage Effect Investigation
**Difficulty**: ⭐⭐⭐⭐⭐

**Goal**: Quantify the leverage effect in cryptocurrency vs traditional markets.

**Tasks**:
1. Fit EGARCH to BTC, ETH, SPY, and QQQ
2. Compare γ coefficients (leverage effect parameter)
3. Test if negative returns truly increase volatility more
4. Write a 2-page report with findings

**Hypothesis**: Crypto has weaker leverage effect than equities.

---

### Project 5.2: High-Frequency Volatility Patterns
**Difficulty**: ⭐⭐⭐⭐⭐

**Goal**: Analyze intraday volatility patterns.

**Tasks**:
1. Download 1-minute data for BTC-USD
2. Calculate realized volatility per hour
3. Identify time-of-day patterns
4. Develop a trading strategy exploiting these patterns

**Expected Finding**: Volatility spikes at market open/close.

---

## 📊 Self-Assessment Rubric

After completing exercises, rate yourself:

**Beginner** (1.1 - 1.2): ✓ Understand basic formulas  
**Intermediate** (2.1 - 2.2): ✓ Can implement and estimate models  
**Advanced** (3.1 - 3.3): ✓ Build complete trading strategies  
**Expert** (4.1 - 5.2): ✓ Conduct original research  

---

## 🎓 Additional Challenges

1. **Optimization Challenge**: Find optimal GARCH parameters using grid search
2. **Speed Challenge**: Optimize GARCH estimation to run in <1 second
3. **Prediction Challenge**: Build ensemble of volatility models
4. **Visualization Challenge**: Create interactive volatility surface plot

---

## 📚 Further Reading

- Tsay, R.S. (2010). *Analysis of Financial Time Series*. Chapter 3.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New Approach"
- Engle, R.F. (2001). "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics"

---

**Solutions Repository**: Full solutions available in `notebooks/solutions/`

**Discussion Forum**: Ask questions at github.com/ctj01/quantitative-ml-advanced/discussions

---

*Remember: The goal is not just to complete exercises, but to deeply understand the concepts and their applications in real trading!*
