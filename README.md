# Quantitative Machine Learning: Advanced Topics

A comprehensive repository covering advanced quantitative machine learning techniques for financial markets, with focus on time series analysis, regime detection, deep learning architectures, and portfolio optimization.

## Overview

This repository provides a complete learning path from advanced time series modeling to production-ready quantitative strategies. Each module contains rigorous mathematical theory, practical implementations, solved examples, and exercises.

## Prerequisites

### Mathematics
- Multivariate calculus
- Linear algebra
- Probability theory and statistics
- Stochastic processes
- Optimization theory

### Programming
- Python 3.8+
- Experience with NumPy, Pandas, and Scikit-learn
- Basic understanding of PyTorch
- Familiarity with Jupyter notebooks

### Finance
- Asset pricing fundamentals
- Portfolio theory
- Risk management concepts
- Market microstructure basics

## Installation

```bash
git clone https://github.com/ctj01/quantitative-ml-advanced.git
cd quantitative-ml-advanced
pip install -r requirements.txt
```

## Repository Structure

```
quantitative-ml-advanced/
├── notebooks/
│   ├── 01_time_series_advanced/        # GARCH, EGARCH, volatility modeling
│   ├── 02_market_regimes/              # Hidden Markov Models
│   ├── 03_transformers_time_series/    # Attention mechanisms, TFT
│   ├── 04_feature_engineering/         # Alpha factors, cointegration
│   ├── 05_backtesting/                 # Walk-forward, transaction costs
│   ├── 06_portfolio_optimization/      # HRP, Black-Litterman
│   ├── 07_advanced_models/             # EVT, copulas
│   └── 08_projects/                    # Complete quantitative projects
├── src/                                # Production-ready source code
│   ├── data/                           # Data loaders and processors
│   ├── models/                         # Model implementations
│   ├── features/                       # Feature engineering
│   ├── backtesting/                    # Backtesting engine
│   └── utils/                          # Utility functions
├── tests/                              # Unit tests
├── docs/                               # Detailed documentation
└── data/                               # Data directory
```

## Learning Path

### Module 1: Time Series Advanced (Weeks 1-2)
- ARCH/GARCH volatility models
- EGARCH and asymmetric effects
- Volatility forecasting
- Model diagnostics and selection

### Module 2: Market Regimes (Week 3)
- Hidden Markov Models theory
- Forward-backward algorithm
- Viterbi decoding
- Regime-dependent strategies

### Module 3: Transformers for Time Series (Weeks 4-5)
- Attention mechanisms
- Temporal Fusion Transformer
- Multi-horizon forecasting
- Comparative analysis with LSTM

### Module 4: Feature Engineering (Week 6)
- Alpha factor construction
- Information coefficient analysis
- Cointegration and pairs trading
- Factor orthogonalization

### Module 5: Backtesting (Weeks 7-8)
- Walk-forward optimization
- Transaction costs and slippage
- Market impact modeling
- Statistical significance testing

### Module 6: Portfolio Optimization (Weeks 9-10)
- Hierarchical Risk Parity
- Black-Litterman model
- Risk budgeting
- Dynamic rebalancing

### Module 7: Advanced Models (Weeks 11-12)
- Extreme Value Theory
- Copulas for dependency modeling
- Tail risk management
- Stress testing

### Module 8: Complete Projects (Weeks 13-16)
- Market regime detector system
- Alpha factory pipeline
- LSTM vs TFT comparison study
- Risk parity portfolio implementation
- EVT-based crash prediction model

## Notebook Structure

Each notebook follows a consistent structure:

1. **Mathematical Theory**
   - Formal definitions
   - Mathematical derivations
   - Theoretical properties
   - Literature references

2. **Implementation**
   - Production-quality code
   - Type hints and docstrings
   - Performance considerations
   - Best practices

3. **Solved Examples**
   - Three complete examples per topic
   - Detailed explanations
   - Real financial data
   - Interpretation of results

4. **Exercises**
   - Five problems per topic
   - Varying difficulty levels
   - Hints and guidance
   - Solutions in separate directory

## Key Dependencies

- **Data**: yfinance, pandas-datareader
- **Numerical**: numpy, scipy, pandas
- **Statistics**: statsmodels, arch
- **Machine Learning**: scikit-learn, hmmlearn
- **Deep Learning**: torch, pytorch-lightning, pytorch-forecasting
- **Portfolio**: riskfolio-lib, cvxpy
- **Optimization**: optuna
- **Visualization**: matplotlib, seaborn, plotly

## Usage

### Running Notebooks

```python
jupyter notebook notebooks/01_time_series_advanced/01_garch_models.ipynb
```

### Using Source Code

```python
from src.models.garch import GARCHModel
from src.data.loaders import load_financial_data

data = load_financial_data("BTC-USD", period="3y")
model = GARCHModel(p=1, q=1)
model.fit(data)
forecasts = model.forecast(horizon=10)
```

### Running Tests

```bash
pytest tests/
```

## Projects

### Project 1: Market Regime Detector
Build a complete system for detecting market regimes using HMM and adapting trading strategies accordingly.

### Project 2: Alpha Factory
Construct and evaluate multiple alpha factors, calculate information coefficients, and build a factor-based portfolio.

### Project 3: LSTM vs TFT Comparison
Comprehensive comparison of LSTM and Temporal Fusion Transformer for cryptocurrency price forecasting.

### Project 4: Risk Parity Portfolio
Implement a Hierarchical Risk Parity portfolio with monthly rebalancing across multiple asset classes.

### Project 5: EVT Crash Prediction
Use Extreme Value Theory to model tail risks and predict crash probabilities.

## Contributing

Contributions are welcome. Please ensure:
- Code follows PEP 8 style guide
- All functions have type hints and docstrings
- Tests are included for new functionality
- Documentation is updated accordingly

## References

### Academic Papers
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series
- Lim, B., et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample

### Books
- Tsay, R. S. (2010). Analysis of Financial Time Series
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). Quantitative Risk Management

## License

MIT License

## Author

Cristian Mendoza (ctj01)

## Acknowledgments

Built for quantitative researchers, data scientists, and machine learning engineers working in financial markets.

---

**Note**: This repository is for educational purposes. All models should be thoroughly tested before use in production environments. Past performance does not guarantee future results.