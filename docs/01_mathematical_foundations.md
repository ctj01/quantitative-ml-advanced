# Mathematical Foundations for Quantitative ML

## Table of Contents
1. [Time Series Mathematics](#time-series-mathematics)
2. [Stochastic Processes](#stochastic-processes)
3. [Volatility Modeling](#volatility-modeling)
4. [Hidden Markov Models](#hidden-markov-models)
5. [Optimization Theory](#optimization-theory)
6. [Information Theory](#information-theory)

---

## Time Series Mathematics

### 1.1 Autoregressive Processes

**Intuition**: An AR process uses past values to predict future values, like how yesterday's stock price influences today's price.

**Geometric Explanation**: 
- Imagine a rubber band connecting consecutive points on a time series
- The AR process creates "memory" - pulling future values toward past patterns
- Higher order AR captures more complex oscillations

**Mathematical Definition**:

An AR(p) process is defined as:
$$
X_t = c + \sum_{i=1}^{p} \phi_i X_{t-i} + \varepsilon_t
$$

where:
- $X_t$ is the value at time $t$
- $c$ is a constant
- $\phi_i$ are autoregressive coefficients
- $\varepsilon_t \sim N(0, \sigma^2)$ is white noise

**Stationarity Condition**:
The characteristic polynomial roots must lie outside the unit circle:
$$
1 - \phi_1 z - \phi_2 z^2 - ... - \phi_p z^p = 0 \implies |z| > 1
$$

**Numerical Example - AR(1)**:
```
Given: X_t = 0.7*X_{t-1} + ε_t, X_0 = 10, σ = 1

t=0: X_0 = 10.00
t=1: X_1 = 0.7*10 + ε_1 = 7.00 + 0.5 = 7.50
t=2: X_2 = 0.7*7.5 + ε_2 = 5.25 - 0.3 = 4.95
t=3: X_3 = 0.7*4.95 + ε_3 = 3.47 + 0.8 = 4.27

Mean reversion to 0 with decay rate 0.7
```

**Trading Application**:
- Mean reversion strategies for pairs trading
- Predicting short-term price movements
- Identifying oversold/overbought conditions

**Risks**:
- Assumes linear relationships (markets are often non-linear)
- Stationarity assumption often violated in real markets
- Structural breaks invalidate the model

---

### 1.2 Moving Average Processes

**Intuition**: MA processes model current value as a weighted sum of recent shocks/surprises, capturing temporary impacts.

**Geometric Explanation**:
- Think of waves in water after dropping stones
- Each "shock" creates ripples that fade over time
- MA(q) = q previous shocks still affecting the current value

**Mathematical Definition**:

MA(q) process:
$$
X_t = \mu + \varepsilon_t + \sum_{i=1}^{q} \theta_i \varepsilon_{t-i}
$$

where:
- $\mu$ is the mean
- $\theta_i$ are MA coefficients
- $\varepsilon_t \sim N(0, \sigma^2)$ are independent shocks

**Invertibility Condition**:
$$
1 + \theta_1 z + \theta_2 z^2 + ... + \theta_q z^q = 0 \implies |z| > 1
$$

**Numerical Example - MA(2)**:
```
Given: X_t = 5 + ε_t + 0.6*ε_{t-1} + 0.3*ε_{t-2}

Shocks: ε_0=1, ε_1=-0.5, ε_2=0.8, ε_3=0.2

t=0: X_0 = 5 + 1.0 + 0 + 0 = 6.00
t=1: X_1 = 5 + (-0.5) + 0.6*1.0 + 0 = 5.10
t=2: X_2 = 5 + 0.8 + 0.6*(-0.5) + 0.3*1.0 = 5.80
t=3: X_3 = 5 + 0.2 + 0.6*0.8 + 0.3*(-0.5) = 5.53
```

---

## Stochastic Processes

### 2.1 Brownian Motion (Wiener Process)

**Intuition**: Random walk in continuous time - like a drunk person's path, each step is random.

**Geometric Visualization**:
```
Price path looks like:
    ^
    |     /\  /\/\
    |    /  \/    \
    |   /          \/\
    |  /              \
    +-------------------->
      Time
```

**Mathematical Properties**:

A process $W_t$ is Brownian motion if:
1. $W_0 = 0$
2. $W_t$ has independent increments
3. $W_t - W_s \sim N(0, t-s)$ for $t > s$
4. $W_t$ has continuous paths

**Quadratic Variation**:
$$
[W, W]_t = t
$$

This fundamental property leads to Itô's lemma.

**Numerical Example**:
```python
# Simulating Brownian Motion
dt = 0.01
n_steps = 1000
dW = np.random.normal(0, np.sqrt(dt), n_steps)
W = np.cumsum(dW)

# Properties:
# Var(W_1) ≈ 1.0
# E[W_1] ≈ 0
```

**Trading Application**:
- Foundation of Black-Scholes option pricing
- Modeling random price movements
- Monte Carlo simulations

---

### 2.2 Geometric Brownian Motion (GBM)

**Intuition**: Stock prices follow GBM - they can't go negative, and percentage changes are normally distributed.

**Mathematical Definition**:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

**Solution** (via Itô's lemma):
$$
S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right)
$$

**Numerical Example**:
```
S_0 = 100, μ = 0.10 (10% drift), σ = 0.20 (20% vol), dt = 1/252

Day 1: dW = 0.05
S_1 = 100 * exp((0.10 - 0.20²/2)*1/252 + 0.20*0.05)
    = 100 * exp(0.000357 + 0.01)
    = 100 * 1.0104 = 101.04

Day 2: dW = -0.08
S_2 = 101.04 * exp(0.000357 - 0.016)
    = 101.04 * 0.9844 = 99.46
```

**Trading Reality**:
- Real markets have fat tails (larger moves than GBM predicts)
- Volatility clusters (GARCH effects)
- Jump components (sudden crashes)

---

## Volatility Modeling

### 3.1 ARCH Process

**Intuition**: Today's volatility depends on yesterday's squared returns - big moves cluster together.

**Geometric Explanation**:
- Volatility comes in waves
- Periods of calm → Periods of chaos → Periods of calm
- "Volatility clustering" - Mandelbrot's observation

**Mathematical Definition**:

ARCH(q) model:
$$
r_t = \sigma_t \varepsilon_t
$$
$$
\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i r_{t-i}^2
$$

where:
- $r_t$ is the return at time $t$
- $\varepsilon_t \sim N(0,1)$ is standardized innovation
- $\sigma_t^2$ is conditional variance

**Constraints**:
- $\omega > 0$
- $\alpha_i \geq 0$ for all $i$
- $\sum_{i=1}^{q} \alpha_i < 1$ (stationarity)

**Numerical Example - ARCH(1)**:
```
Parameters: ω = 0.01, α = 0.3

t=0: r_0 = 2%, σ_0² = 0.04%
t=1: σ_1² = 0.01 + 0.3*(2%)² = 0.01 + 0.00012 = 0.01012
     σ_1 = 10.06%, ε_1 = -0.5
     r_1 = 10.06% * (-0.5) = -5.03%
     
t=2: σ_2² = 0.01 + 0.3*(-5.03%)² = 0.01 + 0.000759 = 0.010759
     σ_2 = 10.37%
```

**Trading Application**:
- Volatility forecasting for option pricing
- Risk management (VaR calculations)
- Position sizing based on volatility regimes

**Common Mistakes**:
- Using ARCH for long-term forecasting (it mean-reverts quickly)
- Ignoring leverage effects (use EGARCH instead)
- Not checking residuals for remaining autocorrelation

---

### 3.2 GARCH Process

**Intuition**: GARCH adds "momentum" to volatility - it remembers both past shocks AND past volatility levels.

**Mathematical Definition**:

GARCH(p,q) model:
$$
\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i r_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2
$$

**Most Common**: GARCH(1,1)
$$
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2
$$

**Unconditional Variance**:
$$
\text{Var}(r_t) = \frac{\omega}{1 - \alpha - \beta}
$$

**Persistence**: $\alpha + \beta$ measures volatility persistence
- Close to 1: very persistent (slow decay)
- Far from 1: quick mean reversion

**Numerical Example - GARCH(1,1)**:
```
Parameters: ω = 0.00001, α = 0.08, β = 0.90
(Note: α + β = 0.98 → highly persistent)

Initial: σ_0² = 0.0004 (2% daily vol)

t=1: r_1 = 3%
     σ_1² = 0.00001 + 0.08*(0.03)² + 0.90*0.0004
          = 0.00001 + 0.000072 + 0.00036
          = 0.000442
     σ_1 = 2.10%

t=2: r_2 = -1%
     σ_2² = 0.00001 + 0.08*(-0.01)² + 0.90*0.000442
          = 0.00001 + 0.000008 + 0.000398
          = 0.000416
     σ_2 = 2.04%
```

**Step-by-Step Application to Trading**:

1. **Estimate GARCH on historical data** (e.g., 1 year of daily returns)
2. **Forecast tomorrow's volatility**: Use the GARCH equation
3. **Adjust position size**: 
   - If $\sigma_{t+1}$ high → Reduce position size
   - If $\sigma_{t+1}$ low → Increase position size
4. **Option pricing**: Use forecasted vol in Black-Scholes
5. **Stop-loss placement**: Set stops at $k \times \sigma_{t+1}$

**Combining with RSI**:
```python
# Trading Logic
if RSI < 30 and σ_forecast < σ_avg:
    # Oversold + Low volatility → High conviction buy
    position_size = 2.0
elif RSI < 30 and σ_forecast > σ_avg:
    # Oversold + High volatility → Cautious buy
    position_size = 0.5
```

---

### 3.3 EGARCH (Exponential GARCH)

**Intuition**: Bad news (negative returns) increases volatility MORE than good news of the same magnitude.

**Why EGARCH?**
- GARCH is symmetric: +5% and -5% have same impact on volatility
- Reality: -5% drop causes more fear/volatility than +5% gain
- EGARCH captures "leverage effect"

**Mathematical Definition**:
$$
\log(\sigma_t^2) = \omega + \sum_{i=1}^{q} \left[\alpha_i \left|\frac{\varepsilon_{t-i}}{\sigma_{t-i}}\right| + \gamma_i \frac{\varepsilon_{t-i}}{\sigma_{t-i}}\right] + \sum_{j=1}^{p} \beta_j \log(\sigma_{t-j}^2)
$$

**EGARCH(1,1)**:
$$
\log(\sigma_t^2) = \omega + \alpha \left|\frac{\varepsilon_{t-1}}{\sigma_{t-1}}\right| + \gamma \frac{\varepsilon_{t-1}}{\sigma_{t-1}} + \beta \log(\sigma_{t-1}^2)
$$

**Key Parameter**: $\gamma$ (leverage effect)
- $\gamma < 0$: Negative shocks increase volatility more (typical)
- $\gamma = 0$: Symmetric (like GARCH)
- $\gamma > 0$: Positive shocks increase volatility more (rare)

**Numerical Example**:
```
Parameters: ω = -0.1, α = 0.15, γ = -0.05, β = 0.95

Initial: log(σ_0²) = -6.0 → σ_0 = 5%

Scenario A: Positive shock (ε_1/σ_0 = +1.5)
log(σ_1²) = -0.1 + 0.15*|1.5| + (-0.05)*1.5 + 0.95*(-6.0)
          = -0.1 + 0.225 - 0.075 - 5.7
          = -5.65
σ_1 = 5.82%

Scenario B: Negative shock (ε_1/σ_0 = -1.5)
log(σ_1²) = -0.1 + 0.15*|-1.5| + (-0.05)*(-1.5) + 0.95*(-6.0)
          = -0.1 + 0.225 + 0.075 - 5.7
          = -5.50
σ_1 = 6.74%

→ Negative shock increases volatility more! (6.74% vs 5.82%)
```

**Trading with EGARCH**:

1. **Crash Detection**: If γ < 0 and large negative return → Expect higher vol
2. **Risk Management**: Tighten stops after negative moves
3. **Option Strategies**: 
   - Buy put options after market drops (vol will spike)
   - Sell call options after rallies (vol won't spike as much)

---

## Hidden Markov Models

### 4.1 Core Concepts

**Intuition**: Markets have hidden "states" (bull, bear, sideways) that we can't observe directly, but we can infer from price behavior.

**Geometric Explanation**:
```
Hidden States:    [Bull] → [Bull] → [Bear] → [Bear] → [Bull]
                     ↓        ↓        ↓        ↓        ↓
Observable:       [+2%]    [+1%]    [-3%]    [-2%]    [+1%]
```

**Mathematical Components**:

1. **States**: $S = \{s_1, s_2, ..., s_N\}$
2. **Observations**: $O = \{o_1, o_2, ..., o_T\}$
3. **Transition Matrix**: $A = [a_{ij}]$ where $a_{ij} = P(s_t = j | s_{t-1} = i)$
4. **Emission Matrix**: $B = [b_j(k)]$ where $b_j(k) = P(o_t = k | s_t = j)$
5. **Initial Distribution**: $\pi = [\pi_i]$ where $\pi_i = P(s_1 = i)$

**Example: 2-State Market Model**

States: {Bull, Bear}

Transition Matrix A:
```
         Bull   Bear
Bull  [  0.90   0.10 ]
Bear  [  0.20   0.80 ]
```
Interpretation: 
- If in Bull, 90% chance stay Bull, 10% switch to Bear
- If in Bear, 80% chance stay Bear, 20% switch to Bull

Emission Distributions (Gaussian):
```
Bull: μ = 0.05%, σ = 1.0%  (small positive drift, low vol)
Bear: μ = -0.10%, σ = 2.5% (negative drift, high vol)
```

**Numerical Example**:

Observations: [+0.5%, +0.3%, -2.0%, -1.5%, +0.2%]

Step 1: Initialize (forward algorithm)
```
π(Bull) = 0.6, π(Bear) = 0.4

t=1, o_1 = +0.5%:
P(o_1|Bull) = N(0.5; 0.05, 1.0) = 0.398
P(o_1|Bear) = N(0.5; -0.10, 2.5) = 0.155

α_1(Bull) = π(Bull) * P(o_1|Bull) = 0.6 * 0.398 = 0.239
α_1(Bear) = π(Bear) * P(o_1|Bear) = 0.4 * 0.155 = 0.062
```

Step 2: Recursion
```
t=2, o_2 = +0.3%:
α_2(Bull) = P(o_2|Bull) * [α_1(Bull)*0.90 + α_1(Bear)*0.20]
         = 0.396 * [0.239*0.90 + 0.062*0.20]
         = 0.396 * 0.227 = 0.090
         
α_2(Bear) = P(o_2|Bear) * [α_1(Bull)*0.10 + α_1(Bear)*0.80]
         = 0.161 * [0.239*0.10 + 0.062*0.80]
         = 0.161 * 0.074 = 0.012
```

**Viterbi Algorithm** (Most Likely State Sequence):

Find: $\arg\max_{s_1,...,s_T} P(s_1,...,s_T | o_1,...,o_T)$

```python
# Pseudocode
δ[t][i] = max probability of state i at time t
ψ[t][i] = best previous state

# Initialization
δ[1][i] = π[i] * B[i][o_1]

# Recursion
for t in range(2, T+1):
    for j in range(N):
        δ[t][j] = max_i(δ[t-1][i] * A[i][j]) * B[j][o_t]
        ψ[t][j] = argmax_i(δ[t-1][i] * A[i][j])

# Backtrack
s_T = argmax_i(δ[T][i])
for t in range(T-1, 0, -1):
    s_t = ψ[t+1][s_{t+1}]
```

**Trading Application with HMM**:

```python
# Regime Detection Strategy
if current_state == 'Bull':
    # Trend following
    if close > MA20:
        signal = 'BUY'
    # But prepare for regime change
    if P(transition to Bear) > 0.3:
        reduce_position_size()

elif current_state == 'Bear':
    # Mean reversion or stay out
    if RSI < 20:
        signal = 'BUY'  # Oversold in bear market
    else:
        signal = 'HOLD' or 'SELL'
        
elif current_state == 'Sideways':
    # Range trading
    if close < support + tolerance:
        signal = 'BUY'
    elif close > resistance - tolerance:
        signal = 'SELL'
```

**Common Mistakes**:
1. **Too many states**: Start with 2-3 states, not 10
2. **Overfitting**: HMM can fit noise if too complex
3. **Ignoring uncertainty**: Use state probabilities, not just most likely state
4. **Look-ahead bias**: Don't train on future data

---

## Optimization Theory

### 5.1 Convex Optimization

**Intuition**: Finding the best solution where "better" is unambiguous - there's one global minimum.

**Geometric Explanation**:
```
Non-convex (bad):          Convex (good):
    ^                          ^
    | /\  /\                  |    /\
    |/  \/  \                 |   /  \
    |        \                |  /    \
    +---------->              +----------->
Multiple minima            Single minimum
```

**Mathematical Definition**:

A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if:
$$
f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)
$$
for all $x, y \in \mathbb{R}^n$ and $\lambda \in [0, 1]$.

**Key Property**: Any local minimum is a global minimum!

**Examples in Finance**:

1. **Mean-Variance Optimization** (Markowitz):
$$
\min_w \quad w^T \Sigma w
$$
$$
\text{s.t.} \quad w^T \mu \geq r_{\text{target}}, \quad w^T \mathbf{1} = 1
$$

This is convex because:
- Objective: $w^T \Sigma w$ is quadratic with $\Sigma \succeq 0$
- Constraints: Linear

2. **Portfolio with Transaction Costs**:
$$
\min_w \quad w^T \Sigma w + \kappa ||w - w_{\text{old}}||_1
$$

Still convex! (L1 norm is convex)

**Numerical Example - Gradient Descent**:
```
Objective: f(w) = w² - 4w + 4 = (w-2)²
Gradient: f'(w) = 2w - 4

Starting point: w_0 = 0
Learning rate: α = 0.1

Iteration 1:
  f'(0) = -4
  w_1 = 0 - 0.1*(-4) = 0.4
  
Iteration 2:
  f'(0.4) = 2*0.4 - 4 = -3.2
  w_2 = 0.4 - 0.1*(-3.2) = 0.72
  
Iteration 3:
  f'(0.72) = 2*0.72 - 4 = -2.56
  w_3 = 0.72 - 0.1*(-2.56) = 0.976
  
Converges to w* = 2.0
```

---

### 5.2 Lagrangian and KKT Conditions

**Intuition**: Convert constrained problem into unconstrained by "penalizing" constraint violations.

**The Lagrangian**:
$$
\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x) + \sum_{j=1}^{p} \nu_j h_j(x)
$$

where:
- $f(x)$: objective function
- $g_i(x) \leq 0$: inequality constraints
- $h_j(x) = 0$: equality constraints
- $\lambda_i, \nu_j$: Lagrange multipliers

**KKT Conditions** (necessary for optimality):
1. **Stationarity**: $\nabla_x \mathcal{L} = 0$
2. **Primal feasibility**: $g_i(x) \leq 0$, $h_j(x) = 0$
3. **Dual feasibility**: $\lambda_i \geq 0$
4. **Complementary slackness**: $\lambda_i g_i(x) = 0$

**Example: Portfolio Optimization**

$$
\min_w \quad \frac{1}{2} w^T \Sigma w
$$
$$
\text{s.t.} \quad w^T \mu = r_0, \quad w^T \mathbf{1} = 1
$$

Lagrangian:
$$
\mathcal{L}(w, \lambda_1, \lambda_2) = \frac{1}{2} w^T \Sigma w + \lambda_1(w^T \mu - r_0) + \lambda_2(w^T \mathbf{1} - 1)
$$

KKT Conditions:
$$
\Sigma w + \lambda_1 \mu + \lambda_2 \mathbf{1} = 0
$$
$$
w^T \mu = r_0, \quad w^T \mathbf{1} = 1
$$

**Solution**:
$$
w^* = \Sigma^{-1} \left[\lambda_1 \mu + \lambda_2 \mathbf{1}\right]
$$

Solve for $\lambda_1, \lambda_2$ using the constraints.

---

## Information Theory

### 6.1 Entropy and Information

**Intuition**: Entropy measures surprise/uncertainty. High entropy = hard to predict.

**Mathematical Definition**:

Shannon entropy:
$$
H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)
$$

**Numerical Example**:
```
Coin A: P(Heads) = 0.5, P(Tails) = 0.5
H(A) = -0.5*log₂(0.5) - 0.5*log₂(0.5) = 1 bit

Coin B: P(Heads) = 0.9, P(Tails) = 0.1
H(B) = -0.9*log₂(0.9) - 0.1*log₂(0.1) = 0.47 bits

Coin A has more uncertainty!
```

**Trading Application**:

1. **Market Efficiency**: High entropy → Hard to predict → Efficient market
2. **Alpha Decay**: As entropy decreases, your edge decays
3. **Information Ratio**: Measures signal/noise ratio

**Cross-Entropy Loss** (for ML models):
$$
L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

Used in classification: predicting up/down/sideways

---

### 6.2 Mutual Information

**Definition**:
$$
I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

**Intuition**: How much does knowing X tell us about Y?

**Trading Example**:
```
X = {RSI overbought, RSI normal}
Y = {Price down next day, Price up next day}

High I(X;Y) → RSI is informative
Low I(X;Y) → RSI provides no edge
```

**Calculating MI for Trading Signals**:
```python
def mutual_information(signal, returns):
    """
    Discretize signal and returns
    Calculate joint and marginal distributions
    Compute MI
    """
    # Example:
    # If MI > 0.1 bits → Signal is useful
    # If MI < 0.01 bits → Signal is noise
```

---

## Summary: Applying to Algorithmic Trading

### Integration Framework

**1. Data Pipeline**:
```
Raw prices → Returns → Feature engineering → Regime detection (HMM)
```

**2. Volatility Forecasting**:
```
Returns → GARCH/EGARCH → σ_forecast → Position sizing
```

**3. Signal Generation**:
```
Technical indicators (RSI, Divergences) → ML model → Trade signal
```

**4. Portfolio Construction**:
```
Signals + Volatility forecasts → Optimization (mean-variance) → Weights
```

**5. Risk Management**:
```
Positions → VaR (from GARCH) → Stop-loss levels
```

### Combining Everything

**Example Trading System**:

```python
# Step 1: Detect regime
regime = hmm.predict(recent_returns)

# Step 2: Forecast volatility
vol_forecast = garch.forecast(horizon=1)

# Step 3: Generate signals
rsi = calculate_rsi(prices)
divergence = detect_divergence(prices, rsi)
ml_signal = lstm_model.predict(features)

# Step 4: Combine signals based on regime
if regime == 'Bull' and rsi < 30 and divergence == 'bullish':
    signal = ml_signal * 1.5  # High conviction
elif regime == 'Bear':
    signal = 0  # Stay out
else:
    signal = ml_signal * 0.5  # Low conviction

# Step 5: Position sizing
position = signal / vol_forecast  # Inverse volatility weighting

# Step 6: Risk management
stop_loss = entry_price - 2 * vol_forecast
```

---

## References

1. Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley.
2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
4. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
5. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.

---

**Next Steps**: Proceed to module-specific notebooks for hands-on implementations and exercises.
