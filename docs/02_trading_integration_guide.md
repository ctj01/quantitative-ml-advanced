# Trading Integration Guide: Combining ML with Technical Analysis

## Table of Contents
1. [Integrating RSI with ML Models](#integrating-rsi-with-ml-models)
2. [W/M Patterns and Regime Detection](#wm-patterns-and-regime-detection)
3. [Divergence Analysis](#divergence-analysis)
4. [Candlestick Patterns with Deep Learning](#candlestick-patterns-with-deep-learning)
5. [Complete Trading System Architecture](#complete-trading-system-architecture)

---

## Integrating RSI with ML Models

### Understanding RSI Deeply

**Mathematical Definition**:
$$
RSI = 100 - \frac{100}{1 + RS}
$$
where:
$$
RS = \frac{\text{Average Gain over n periods}}{\text{Average Loss over n periods}}
$$

**Typical calculation** (n=14):
$$
\text{Avg Gain}_t = \frac{13 \times \text{Avg Gain}_{t-1} + \text{Current Gain}}{14}
$$

**Geometric Intuition**:
```
RSI Space:
100 |------------------------| Overbought (>70)
    |        Buying          |
 50 |========================| Neutral
    |        Selling         |
  0 |------------------------| Oversold (<30)
```

**What RSI Really Measures**:
- Momentum of price changes
- Ratio of up-moves to down-moves
- NOT the same as price level!

**Common Misconceptions**:
❌ "RSI > 70 means sell immediately"
✅ "RSI > 70 in uptrend can persist for weeks"

❌ "RSI < 30 means guaranteed bounce"
✅ "RSI < 30 in downtrend can go to 10"

### Numerical Example: RSI Calculation

```
Day  Close  Change  Gain  Loss  Avg Gain  Avg Loss   RS    RSI
1    100     -       -     -       -         -        -     -
2    102    +2      2     0       -         -        -     -
3    101    -1      0     1       -         -        -     -
4    103    +2      2     0       -         -        -     -
5    105    +2      2     0       -         -        -     -
6    104    -1      0     1       -         -        -     -
7    106    +2      2     0       -         -        -     -
8    108    +2      2     0       -         -        -     -
9    107    -1      0     1       -         -        -     -
10   109    +2      2     0       -         -        -     -
11   110    +1      1     0       -         -        -     -
12   109    -1      0     1       -         -        -     -
13   111    +2      2     0       -         -        -     -
14   112    +1      1     0       -         -        -     -
15   113    +1      1     0      1.43      0.36    3.97   79.9

Day 15 calculation:
Avg Gain = (2+0+2+2+0+2+2+0+2+1+0+2+1+1)/14 = 1.43
Avg Loss = (0+1+0+0+1+0+0+1+0+0+1+0+0+0)/14 = 0.36
RS = 1.43/0.36 = 3.97
RSI = 100 - 100/(1+3.97) = 79.9 → Overbought!
```

### Enhancing RSI with Machine Learning

**Problem**: Basic RSI rules (buy at 30, sell at 70) fail in trending markets.

**Solution**: Context-aware RSI using ML.

**Feature Engineering**:

```python
def create_rsi_features(df):
    """
    Create context-aware RSI features
    """
    # Basic RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # RSI in different market regimes
    df['rsi_trend'] = df['rsi'] - df['rsi'].rolling(20).mean()
    
    # RSI velocity
    df['rsi_velocity'] = df['rsi'].diff()
    
    # RSI acceleration
    df['rsi_acceleration'] = df['rsi_velocity'].diff()
    
    # RSI volatility
    df['rsi_vol'] = df['rsi'].rolling(20).std()
    
    # RSI relative to price action
    df['rsi_price_div'] = (df['rsi'] - 50) * np.sign(df['returns'])
    
    # Historical RSI percentile
    df['rsi_percentile'] = df['rsi'].rolling(252).apply(
        lambda x: percentileofscore(x[:-1], x.iloc[-1])
    )
    
    return df
```

**ML Model Integration**:

```python
# Scenario 1: RSI as feature in LSTM
class RSI_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=10,  # RSI + 9 other features
            hidden_size=50,
            num_layers=2
        )
        self.fc = nn.Linear(50, 3)  # Buy/Hold/Sell
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        # features include: rsi, rsi_trend, rsi_velocity, etc.
        lstm_out, _ = self.lstm(x)
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction

# Scenario 2: RSI regime classifier
def classify_rsi_regime(rsi, price_data):
    """
    Use ML to classify RSI effectiveness
    """
    features = [
        rsi,
        price_data['volatility'],
        price_data['trend_strength'],
        price_data['volume_profile']
    ]
    
    # Model predicts: 'trending_up', 'trending_down', 'ranging'
    regime = ml_model.predict(features)
    
    if regime == 'trending_up':
        # In uptrend, buy on RSI pullbacks to 40-50
        if 40 < rsi < 50:
            return 'BUY'
    elif regime == 'trending_down':
        # In downtrend, sell on RSI bounces to 50-60
        if 50 < rsi < 60:
            return 'SELL'
    elif regime == 'ranging':
        # In range, use traditional levels
        if rsi < 30:
            return 'BUY'
        elif rsi > 70:
            return 'SELL'
    
    return 'HOLD'
```

### Trading Rules with Context

**Rule 1: Bull Market**
```
If 50 < MA20 and price > MA50:
    # Strong uptrend
    if RSI < 40:
        signal = 'BUY'  # Pullback buying opportunity
        confidence = HIGH
    elif RSI > 80:
        signal = 'HOLD'  # Can stay overbought
        confidence = MEDIUM
```

**Rule 2: Bear Market**
```
If price < MA20 and MA20 < MA50:
    # Strong downtrend
    if RSI > 60:
        signal = 'SELL'  # Bounce selling opportunity
        confidence = HIGH
    elif RSI < 20:
        signal = 'HOLD'  # Can stay oversold
        confidence = LOW
```

**Rule 3: Range-Bound**
```
If abs(returns_20d) < volatility_threshold:
    # Sideways market
    if RSI < 25:
        signal = 'BUY'
        confidence = MEDIUM
    elif RSI > 75:
        signal = 'SELL'
        confidence = MEDIUM
```

---

## W/M Patterns and Regime Detection

### Understanding W and M Patterns

**W Pattern (Double Bottom)**:
```
Price Chart:
    |           C
    |          / \
    |    B    /   \
    |   / \  /     
    |  /   \/      
    | A           
    +----------------
      Time

A: First low
B: First high (resistance test)
C: Second low (should be ≈ A or slightly higher)
Breakout: Above B confirms pattern
```

**M Pattern (Double Top)**:
```
Price Chart:
    |  B       D
    | / \     / \
    |/   \   /   \
    |     \ /     
    |      C      
    +----------------
      Time

B: First high
C: First low (support test)
D: Second high (should be ≈ B or slightly lower)
Breakdown: Below C confirms pattern
```

### Mathematical Detection

**W Pattern Detection Algorithm**:

```python
def detect_w_pattern(prices, window=20):
    """
    Detect W pattern using local extrema
    """
    # Find local minima and maxima
    minima = argrelextrema(prices, np.less, order=5)[0]
    maxima = argrelextrema(prices, np.greater, order=5)[0]
    
    for i in range(len(minima) - 1):
        A_idx = minima[i]
        C_idx = minima[i + 1]
        
        # Find B (peak between A and C)
        B_candidates = maxima[(maxima > A_idx) & (maxima < C_idx)]
        if len(B_candidates) == 0:
            continue
        B_idx = B_candidates[0]
        
        A = prices[A_idx]
        B = prices[B_idx]
        C = prices[C_idx]
        
        # W pattern criteria:
        # 1. C should be similar to A (within 2%)
        if abs(C - A) / A > 0.02:
            continue
            
        # 2. B should be significantly higher (>3%)
        if (B - A) / A < 0.03:
            continue
            
        # 3. Time symmetry (optional)
        time_AB = B_idx - A_idx
        time_BC = C_idx - B_idx
        if abs(time_AB - time_BC) / time_AB > 0.5:
            continue
            
        # Found W pattern!
        return {
            'type': 'W',
            'A': (A_idx, A),
            'B': (B_idx, B),
            'C': (C_idx, C),
            'target': B + (B - A),  # Measured move
            'stop_loss': C - 0.5 * (B - A)
        }
    
    return None
```

**Numerical Example**:
```
Price sequence: [100, 98, 96, 95, 97, 99, 101, 99, 97, 96, 95, 97, 99, 102]

Detected points:
A = (3, 95)   # First low
B = (6, 101)  # Resistance
C = (10, 95)  # Second low

Criteria check:
1. C ≈ A? |95-95|/95 = 0% ✓
2. B significantly higher? (101-95)/95 = 6.3% ✓
3. Time symmetry? |3-4|/3 = 33% ✓

W Pattern confirmed!
Target = 101 + (101-95) = 107
Stop Loss = 95 - 0.5*(101-95) = 92
```

### Integrating with HMM

**Key Insight**: W/M patterns are MORE reliable in specific market regimes.

```python
class PatternRegimeModel:
    def __init__(self):
        self.hmm = HiddenMarkovModel(n_states=3)
        # States: Bull, Bear, Ranging
        
    def pattern_reliability(self, pattern_type, current_regime):
        """
        Return confidence multiplier based on regime
        """
        reliability_matrix = {
            'W': {
                'Bull': 0.8,    # W patterns work well in bull
                'Bear': 0.9,    # Even better in bear (reversal)
                'Ranging': 0.6  # Moderate in range
            },
            'M': {
                'Bull': 0.9,    # M patterns work well in bull (reversal)
                'Bear': 0.8,    # Also good in bear
                'Ranging': 0.6  # Moderate in range
            }
        }
        return reliability_matrix[pattern_type][current_regime]
    
    def generate_signal(self, pattern, regime):
        """
        Combine pattern and regime for signal
        """
        base_signal = pattern['signal']
        confidence = self.pattern_reliability(pattern['type'], regime)
        
        adjusted_signal = base_signal * confidence
        
        # Additional regime-specific adjustments
        if regime == 'Bear' and pattern['type'] == 'W':
            # More cautious in bear markets
            stop_loss = pattern['stop_loss'] * 1.2
            position_size = 0.5
        elif regime == 'Bull' and pattern['type'] == 'M':
            # More cautious in bull markets
            stop_loss = pattern['stop_loss'] * 1.2
            position_size = 0.5
        else:
            position_size = 1.0
            
        return {
            'signal': adjusted_signal,
            'confidence': confidence,
            'position_size': position_size,
            'stop_loss': stop_loss
        }
```

---

## Divergence Analysis

### Types of Divergences

**1. Regular Bullish Divergence**
```
Price:     Makes lower lows
RSI:       Makes higher lows
Signal:    Potential reversal up
```

**2. Regular Bearish Divergence**
```
Price:     Makes higher highs
RSI:       Makes lower highs
Signal:    Potential reversal down
```

**3. Hidden Bullish Divergence**
```
Price:     Makes higher lows (uptrend continuation)
RSI:       Makes lower lows
Signal:    Uptrend continuation
```

**4. Hidden Bearish Divergence**
```
Price:     Makes lower highs (downtrend continuation)
RSI:       Makes higher highs
Signal:    Downtrend continuation
```

### Mathematical Detection

```python
def detect_divergence(prices, rsi, lookback=14):
    """
    Detect all types of divergences
    """
    # Find price extrema
    price_peaks = argrelextrema(prices, np.greater, order=5)[0]
    price_troughs = argrelextrema(prices, np.less, order=5)[0]
    
    # Find RSI extrema
    rsi_peaks = argrelextrema(rsi, np.greater, order=5)[0]
    rsi_troughs = argrelextrema(rsi, np.less, order=5)[0]
    
    divergences = []
    
    # Regular Bullish Divergence
    for i in range(len(price_troughs) - 1):
        t1, t2 = price_troughs[i], price_troughs[i+1]
        
        # Find corresponding RSI troughs
        rsi_t1 = find_closest(rsi_troughs, t1)
        rsi_t2 = find_closest(rsi_troughs, t2)
        
        if rsi_t1 is None or rsi_t2 is None:
            continue
            
        # Check divergence conditions
        if (prices[t2] < prices[t1] and  # Lower price low
            rsi[rsi_t2] > rsi[rsi_t1]):   # Higher RSI low
            
            divergences.append({
                'type': 'regular_bullish',
                'price_points': (t1, t2),
                'rsi_points': (rsi_t1, rsi_t2),
                'strength': (rsi[rsi_t2] - rsi[rsi_t1]) / rsi[rsi_t1]
            })
    
    # Regular Bearish Divergence
    for i in range(len(price_peaks) - 1):
        p1, p2 = price_peaks[i], price_peaks[i+1]
        
        rsi_p1 = find_closest(rsi_peaks, p1)
        rsi_p2 = find_closest(rsi_peaks, p2)
        
        if rsi_p1 is None or rsi_p2 is None:
            continue
            
        if (prices[p2] > prices[p1] and  # Higher price high
            rsi[rsi_p2] < rsi[rsi_p1]):   # Lower RSI high
            
            divergences.append({
                'type': 'regular_bearish',
                'price_points': (p1, p2),
                'rsi_points': (rsi_p1, rsi_p2),
                'strength': (rsi[rsi_p1] - rsi[rsi_p2]) / rsi[rsi_p1]
            })
    
    return divergences
```

### Numerical Example

```
Time:  1    2    3    4    5    6    7    8    9   10
Price: 100  105  103  110  108  115  112  118  114  120
RSI:   45   55   50   62   58   68   64   65   62   63

Analysis:
Price peaks: [2:105, 4:110, 6:115, 8:118, 10:120]
RSI peaks:   [2:55,  4:62,  6:68,  8:65,  10:63]

Compare peaks 6 and 8:
Price: 118 > 115 (higher high) ✓
RSI:   65 < 68 (lower high) ✓

→ Regular Bearish Divergence detected!
Strength: (68-65)/68 = 4.4%
Signal: Potential reversal down
```

### Combining Divergences with ML

```python
class DivergenceMLModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        
    def create_divergence_features(self, div):
        """
        Extract features from divergence
        """
        return {
            'div_type': encode(div['type']),
            'div_strength': div['strength'],
            'div_duration': div['price_points'][1] - div['price_points'][0],
            'rsi_level': current_rsi,
            'price_volatility': recent_volatility,
            'volume_ratio': current_volume / avg_volume,
            'trend_strength': abs(returns_50d) / volatility_50d,
            'regime': current_regime  # from HMM
        }
    
    def predict_divergence_success(self, divergence):
        """
        Predict if divergence will lead to reversal
        """
        features = self.create_divergence_features(divergence)
        probability = self.model.predict_proba([features])[0][1]
        
        return {
            'probability': probability,
            'recommended_action': 'TRADE' if probability > 0.65 else 'IGNORE',
            'confidence': 'HIGH' if probability > 0.75 else 'MEDIUM'
        }
```

### Trading Strategy

```python
def divergence_strategy(prices, rsi, hmm_regime):
    """
    Complete divergence trading strategy
    """
    # Detect divergences
    divergences = detect_divergence(prices, rsi)
    
    for div in divergences:
        # ML filtering
        prediction = ml_model.predict_divergence_success(div)
        
        if prediction['probability'] < 0.65:
            continue  # Skip low-probability divergences
        
        # Generate signal based on type
        if div['type'] == 'regular_bullish':
            if hmm_regime in ['Bear', 'Ranging']:
                # Divergences work better at trend exhaustion
                signal = 'BUY'
                target = prices[-1] * 1.05  # 5% target
                stop = prices[-1] * 0.97    # 3% stop
                
        elif div['type'] == 'regular_bearish':
            if hmm_regime in ['Bull', 'Ranging']:
                signal = 'SELL'
                target = prices[-1] * 0.95
                stop = prices[-1] * 1.03
                
        # Position sizing based on confidence
        position_size = base_size * prediction['probability']
        
        return {
            'signal': signal,
            'entry': prices[-1],
            'target': target,
            'stop_loss': stop,
            'position_size': position_size,
            'confidence': prediction['confidence']
        }
```

---

## Candlestick Patterns with Deep Learning

### Traditional Candlestick Patterns

**Hammer**:
```
    |
    |
  +---+
  |   |
  +---+
  
Characteristics:
- Small body at top
- Long lower shadow (2x body)
- Little/no upper shadow
- Bullish reversal
```

**Shooting Star**:
```
    |
    |
  +---+
  |   |
  +---+
  
Characteristics:
- Small body at bottom
- Long upper shadow (2x body)
- Little/no lower shadow
- Bearish reversal
```

### Deep Learning Approach

**Why CNN for Candlesticks?**
- CNNs detect visual patterns
- Candlesticks are visual patterns
- CNN can learn complex multi-candle patterns

**Architecture**:

```python
class CandlestickCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: (batch, channels, height, width)
        # We convert candlesticks to image: (batch, 4, 50, 50)
        # 4 channels: Open, High, Low, Close
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)  # Buy/Hold/Sell
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def candles_to_image(ohlc_data, size=50):
    """
    Convert OHLC data to image representation
    """
    image = np.zeros((4, size, size))
    
    # Normalize prices to fit in image
    max_price = ohlc_data['high'].max()
    min_price = ohlc_data['low'].min()
    
    for i, (_, row) in enumerate(ohlc_data.iterrows()):
        if i >= size:
            break
            
        # Map prices to pixel coordinates
        o = int((row['open'] - min_price) / (max_price - min_price) * (size-1))
        h = int((row['high'] - min_price) / (max_price - min_price) * (size-1))
        l = int((row['low'] - min_price) / (max_price - min_price) * (size-1))
        c = int((row['close'] - min_price) / (max_price - min_price) * (size-1))
        
        # Fill channels
        image[0, h:l+1, i] = 1  # High-Low line
        image[1, min(o,c):max(o,c)+1, i] = 1  # Body
        image[2, i, o] = 1  # Open marker
        image[3, i, c] = 1  # Close marker
    
    return image
```

### Hybrid Approach: Combine Traditional + ML

```python
def hybrid_candlestick_strategy(ohlc_data):
    """
    Combine traditional pattern recognition with CNN
    """
    # Traditional patterns
    traditional_signals = {
        'hammer': detect_hammer(ohlc_data),
        'engulfing': detect_engulfing(ohlc_data),
        'doji': detect_doji(ohlc_data),
        'morning_star': detect_morning_star(ohlc_data)
    }
    
    # CNN prediction
    candle_image = candles_to_image(ohlc_data)
    cnn_prediction = cnn_model.predict(candle_image)
    
    # Combine signals
    if traditional_signals['hammer'] and cnn_prediction == 'BUY':
        return {
            'signal': 'BUY',
            'confidence': 'HIGH',
            'reason': 'Hammer pattern + CNN confirmation'
        }
    
    elif traditional_signals['engulfing'] == 'bullish' and cnn_prediction == 'BUY':
        return {
            'signal': 'BUY',
            'confidence': 'HIGH',
            'reason': 'Bullish engulfing + CNN confirmation'
        }
    
    elif cnn_prediction == 'BUY' and not any(traditional_signals.values()):
        return {
            'signal': 'BUY',
            'confidence': 'MEDIUM',
            'reason': 'CNN pattern (novel)'
        }
    
    else:
        return {'signal': 'HOLD', 'confidence': 'LOW'}
```

---

## Complete Trading System Architecture

### System Flowchart

```
┌─────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                        │
├─────────────────────────────────────────────────────────┤
│  Raw Price Data → Cleaning → Feature Engineering        │
│  (OHLCV)          ↓          ↓                          │
│                   Technical  Fundamental                 │
│                   Indicators Features                    │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│                  REGIME DETECTION                        │
├─────────────────────────────────────────────────────────┤
│  HMM Model → Current Regime (Bull/Bear/Ranging)         │
│  ↓                                                       │
│  Regime-Specific Parameters                             │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│               MULTI-SIGNAL GENERATION                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │     RSI     │  │  Divergences │  │   W/M Patterns │ │
│  │   + ML      │  │              │  │                │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬───────┘ │
│         │                 │                    │         │
│  ┌──────┴─────────────────┴────────────────────┴──────┐ │
│  │          Candlestick CNN                            │ │
│  └─────────────────────────┬───────────────────────────┘ │
└────────────────────────────┼─────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────┐
│                  SIGNAL AGGREGATION                      │
├─────────────────────────────────────────────────────────┤
│  Weighted Ensemble → Final Signal                       │
│  (Regime-adjusted weights)                              │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│             RISK MANAGEMENT & SIZING                     │
├─────────────────────────────────────────────────────────┤
│  GARCH Volatility Forecast → Position Size              │
│  ↓                                                       │
│  Kelly Criterion / Fixed Fraction                       │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│              PORTFOLIO OPTIMIZATION                      │
├─────────────────────────────────────────────────────────┤
│  Mean-Variance / Risk Parity → Final Weights            │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│                ORDER EXECUTION                           │
├─────────────────────────────────────────────────────────┤
│  Smart Order Routing → Fill                             │
│  (TWAP/VWAP if large size)                              │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│            MONITORING & FEEDBACK                         │
├─────────────────────────────────────────────────────────┤
│  Performance Analytics → Model Retraining               │
│  Risk Monitoring → Alert System                         │
└─────────────────────────────────────────────────────────┘
```

### Implementation

```python
class AdvancedTradingSystem:
    def __init__(self):
        # Models
        self.hmm = HiddenMarkovModel(n_states=3)
        self.garch = GARCH(p=1, q=1)
        self.rsi_lstm = RSI_LSTM()
        self.candlestick_cnn = CandlestickCNN()
        self.divergence_ml = DivergenceMLModel()
        
        # Parameters
        self.regime_weights = {
            'Bull': {'rsi': 0.2, 'divergence': 0.1, 'pattern': 0.3, 'cnn': 0.4},
            'Bear': {'rsi': 0.3, 'divergence': 0.3, 'pattern': 0.2, 'cnn': 0.2},
            'Ranging': {'rsi': 0.3, 'divergence': 0.2, 'pattern': 0.2, 'cnn': 0.3}
        }
        
    def process_signal(self, data):
        """
        Main signal processing pipeline
        """
        # 1. Regime Detection
        regime = self.hmm.predict(data['returns'])
        
        # 2. Volatility Forecast
        vol_forecast = self.garch.forecast(data['returns'])
        
        # 3. Generate individual signals
        signals = {}
        
        # RSI Signal
        rsi = calculate_rsi(data['close'])
        rsi_features = create_rsi_features(data)
        signals['rsi'] = self.rsi_lstm.predict(rsi_features)
        
        # Divergence Signal
        divergences = detect_divergence(data['close'], rsi)
        if divergences:
            signals['divergence'] = self.divergence_ml.predict_divergence_success(
                divergences[-1]
            )['probability']
        else:
            signals['divergence'] = 0
        
        # W/M Pattern Signal
        w_pattern = detect_w_pattern(data['close'])
        m_pattern = detect_m_pattern(data['close'])
        if w_pattern:
            signals['pattern'] = 1.0
        elif m_pattern:
            signals['pattern'] = -1.0
        else:
            signals['pattern'] = 0
        
        # CNN Signal
        candle_image = candles_to_image(data)
        signals['cnn'] = self.candlestick_cnn.predict(candle_image)
        
        # 4. Aggregate signals (regime-weighted)
        weights = self.regime_weights[regime]
        final_signal = sum(signals[key] * weights[key] for key in weights)
        
        # 5. Position sizing (inverse volatility)
        target_vol = 0.01  # 1% daily vol target
        position_size = (target_vol / vol_forecast) * final_signal
        
        # 6. Risk checks
        if abs(position_size) > 2.0:  # Max 2x leverage
            position_size = np.sign(position_size) * 2.0
        
        return {
            'regime': regime,
            'signal': final_signal,
            'position_size': position_size,
            'vol_forecast': vol_forecast,
            'individual_signals': signals
        }
```

### Backtesting Framework

```python
class Backtester:
    def __init__(self, system, data, initial_capital=100000):
        self.system = system
        self.data = data
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        
    def run(self):
        """
        Execute backtest
        """
        for t in range(100, len(self.data)):  # Start after warmup
            # Get current market data
            current_data = self.data.iloc[t-100:t]
            
            # Generate signal
            result = self.system.process_signal(current_data)
            
            # Execute trade
            if abs(result['signal']) > 0.5:  # Threshold
                trade = {
                    'timestamp': self.data.index[t],
                    'signal': result['signal'],
                    'size': result['position_size'],
                    'entry_price': self.data.iloc[t]['close'],
                    'regime': result['regime'],
                    'vol_forecast': result['vol_forecast']
                }
                
                # Calculate returns
                next_return = (self.data.iloc[t+1]['close'] / 
                              self.data.iloc[t]['close'] - 1)
                trade['return'] = next_return * trade['size']
                trade['exit_price'] = self.data.iloc[t+1]['close']
                
                self.trades.append(trade)
                self.capital *= (1 + trade['return'])
        
        return self.analyze_results()
    
    def analyze_results(self):
        """
        Calculate performance metrics
        """
        trades_df = pd.DataFrame(self.trades)
        
        return {
            'total_return': (self.capital / 100000 - 1) * 100,
            'sharpe_ratio': trades_df['return'].mean() / trades_df['return'].std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(trades_df),
            'win_rate': (trades_df['return'] > 0).mean(),
            'avg_win': trades_df[trades_df['return'] > 0]['return'].mean(),
            'avg_loss': trades_df[trades_df['return'] < 0]['return'].mean(),
            'total_trades': len(trades_df)
        }
```

---

## Risks and Common Mistakes

### 1. Overfitting
**Problem**: Model learns noise instead of signal
**Solution**: 
- Use walk-forward optimization
- Cross-validation with time series splits
- Regularization (L1/L2, dropout)
- Limit model complexity

### 2. Look-Ahead Bias
**Problem**: Using future information in past decisions
**Solution**:
- Strict time-based splitting
- Only use data available at time t for decision at time t
- Be careful with indicators that "peek" forward

### 3. Data Snooping
**Problem**: Testing many strategies, reporting only the best
**Solution**:
- Pre-register hypothesis
- Bonferroni correction for multiple tests
- Out-of-sample testing

### 4. Ignoring Transaction Costs
**Problem**: Theoretical profits disappear with real costs
**Solution**:
- Model slippage (0.1-0.5% per trade)
- Include commissions
- Consider market impact for large orders

### 5. Regime Changes
**Problem**: Models trained on bull market fail in bear
**Solution**:
- Use HMM for regime detection
- Regime-specific models
- Regular retraining

### 6. Correlation Breakdown
**Problem**: Diversification fails in crisis
**Solution**:
- Tail risk hedging
- Stress testing
- Dynamic correlation models (DCC-GARCH)

---

## Next Steps

1. **Study the notebooks** in order (01 → 08)
2. **Implement each component** separately
3. **Combine gradually** - start simple, add complexity
4. **Backtest rigorously** with realistic assumptions
5. **Paper trade** before real money
6. **Monitor and adapt** - markets evolve

Remember: **Past performance ≠ Future results**

---

## References

- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*.
- Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*.
- Pring, M. J. (2002). *Technical Analysis Explained*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
