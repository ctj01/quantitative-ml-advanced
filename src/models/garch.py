"""
GARCH Model Implementation from Scratch
========================================

This module provides a complete implementation of ARCH, GARCH, and EGARCH models
for volatility forecasting in financial markets.

Author: Cristian Mendoza
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from typing import Tuple, Optional, Dict
import warnings


class ARCHModel:
    """
    ARCH(q) - Autoregressive Conditional Heteroskedasticity Model
    
    Mathematical Model:
        r_t = σ_t * ε_t
        σ_t² = ω + Σ(α_i * r²_{t-i}) for i=1 to q
        
    Where:
        r_t: return at time t
        σ_t: conditional volatility
        ε_t ~ N(0,1): standardized residuals
        ω > 0: constant term
        α_i ≥ 0: ARCH coefficients
    """
    
    def __init__(self, q: int = 1):
        """
        Initialize ARCH model
        
        Parameters:
        -----------
        q : int
            Order of ARCH model (number of lagged squared returns)
        """
        self.q = q
        self.params = None
        self.sigma2 = None
        self.resid = None
        self.fitted = False
        
    def _initial_params(self, returns: np.ndarray) -> np.ndarray:
        """Generate initial parameter guesses"""
        omega = np.var(returns) * 0.01
        alphas = np.array([0.1 / self.q] * self.q)
        return np.concatenate([[omega], alphas])
    
    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute negative log-likelihood for MLE estimation
        
        Log-likelihood:
            L = -0.5 * Σ[log(2π) + log(σ_t²) + r_t²/σ_t²]
        """
        omega = params[0]
        alphas = params[1:]
        
        T = len(returns)
        sigma2 = np.zeros(T)
        
        # Initial variance (unconditional)
        sigma2[0] = np.var(returns)
        
        # Compute conditional variances
        for t in range(1, T):
            sigma2[t] = omega
            for i in range(self.q):
                if t - i - 1 >= 0:
                    sigma2[t] += alphas[i] * returns[t - i - 1]**2
        
        # Avoid log(0)
        sigma2 = np.maximum(sigma2, 1e-8)
        
        # Negative log-likelihood
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
        
        return -log_likelihood
    
    def fit(self, returns: pd.Series) -> 'ARCHModel':
        """
        Fit ARCH model using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
            
        Returns:
        --------
        self : ARCHModel
            Fitted model
        """
        returns_array = np.asarray(returns)
        
        # Initial parameters
        initial_params = self._initial_params(returns_array)
        
        # Constraints: ω > 0, α_i ≥ 0
        bounds = [(1e-6, None)] + [(0, None)] * self.q
        
        # Stationarity constraint: Σα_i < 1
        constraints = {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[1:])}
        
        # Optimization
        result = minimize(
            self._neg_log_likelihood,
            initial_params,
            args=(returns_array,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn("Optimization did not converge")
        
        self.params = result.x
        self.fitted = True
        
        # Compute fitted values
        self._compute_fitted_values(returns_array)
        
        return self
    
    def _compute_fitted_values(self, returns: np.ndarray):
        """Compute fitted conditional variances"""
        omega = self.params[0]
        alphas = self.params[1:]
        
        T = len(returns)
        self.sigma2 = np.zeros(T)
        self.sigma2[0] = np.var(returns)
        
        for t in range(1, T):
            self.sigma2[t] = omega
            for i in range(self.q):
                if t - i - 1 >= 0:
                    self.sigma2[t] += alphas[i] * returns[t - i - 1]**2
        
        self.resid = returns / np.sqrt(self.sigma2)
    
    def forecast(self, horizon: int = 1, returns: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forecast volatility for specified horizon
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon (number of steps ahead)
        returns : np.ndarray, optional
            Recent returns for forecasting. If None, use fitted values.
            
        Returns:
        --------
        forecasts : np.ndarray
            Volatility forecasts (standard deviations)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        omega = self.params[0]
        alphas = self.params[1:]
        
        if returns is None:
            # Use last q returns from fitted data
            returns = self.returns[-self.q:]
        
        forecasts = np.zeros(horizon)
        
        # One-step ahead
        forecasts[0] = np.sqrt(omega + np.sum(alphas * returns[-self.q:]**2))
        
        # Multi-step ahead (assuming zero mean)
        for h in range(1, horizon):
            # E[r²_{t+h}] = E[σ²_{t+h}]
            forecasts[h] = np.sqrt(omega + np.sum(alphas) * forecasts[h-1]**2)
        
        return forecasts
    
    def summary(self) -> pd.DataFrame:
        """Return model summary statistics"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        param_names = ['omega'] + [f'alpha[{i+1}]' for i in range(self.q)]
        
        return pd.DataFrame({
            'Parameter': param_names,
            'Value': self.params,
            'Constraint': ['> 0'] + ['≥ 0'] * self.q
        })


class GARCHModel:
    """
    GARCH(p,q) - Generalized ARCH Model
    
    Mathematical Model:
        r_t = σ_t * ε_t
        σ_t² = ω + Σ(α_i * r²_{t-i}) + Σ(β_j * σ²_{t-j})
        
    The most common specification is GARCH(1,1):
        σ_t² = ω + α * r²_{t-1} + β * σ²_{t-1}
    """
    
    def __init__(self, p: int = 1, q: int = 1, mean_model: str = 'zero'):
        """
        Initialize GARCH model
        
        Parameters:
        -----------
        p : int
            Order of GARCH component (lagged variances)
        q : int
            Order of ARCH component (lagged squared returns)
        mean_model : str
            'zero' or 'constant'
        """
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.params = None
        self.sigma2 = None
        self.resid = None
        self.fitted = False
        self.returns = None
        
    def _initial_params(self, returns: np.ndarray) -> np.ndarray:
        """Generate initial parameter guesses"""
        var = np.var(returns)
        
        omega = var * 0.01
        alphas = np.array([0.05 / self.q] * self.q)
        betas = np.array([0.90 / self.p] * self.p)
        
        if self.mean_model == 'constant':
            mu = np.mean(returns)
            return np.concatenate([[mu], [omega], alphas, betas])
        else:
            return np.concatenate([[omega], alphas, betas])
    
    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Compute negative log-likelihood"""
        if self.mean_model == 'constant':
            mu = params[0]
            omega = params[1]
            alphas = params[2:2+self.q]
            betas = params[2+self.q:]
            resid = returns - mu
        else:
            omega = params[0]
            alphas = params[1:1+self.q]
            betas = params[1+self.q:]
            resid = returns
        
        T = len(returns)
        sigma2 = np.zeros(T)
        
        # Initial variance
        sigma2[0] = np.var(resid)
        
        # Compute conditional variances
        for t in range(1, T):
            sigma2[t] = omega
            
            # ARCH component
            for i in range(self.q):
                if t - i - 1 >= 0:
                    sigma2[t] += alphas[i] * resid[t - i - 1]**2
            
            # GARCH component
            for j in range(self.p):
                if t - j - 1 >= 0:
                    sigma2[t] += betas[j] * sigma2[t - j - 1]
        
        # Avoid numerical issues
        sigma2 = np.maximum(sigma2, 1e-8)
        
        # Negative log-likelihood
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + resid**2 / sigma2)
        
        return -log_likelihood
    
    def fit(self, returns: pd.Series) -> 'GARCHModel':
        """
        Fit GARCH model using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
            
        Returns:
        --------
        self : GARCHModel
            Fitted model
        """
        self.returns = np.asarray(returns)
        
        # Initial parameters
        initial_params = self._initial_params(self.returns)
        
        # Constraints
        if self.mean_model == 'constant':
            # mu: any, ω > 0, α_i ≥ 0, β_j ≥ 0
            bounds = [(None, None), (1e-6, None)] + [(0, None)] * (self.q + self.p)
            # Stationarity: Σ(α_i + β_j) < 1
            constraints = {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[2:])}
        else:
            bounds = [(1e-6, None)] + [(0, None)] * (self.q + self.p)
            constraints = {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[1:])}
        
        # Optimization
        result = minimize(
            self._neg_log_likelihood,
            initial_params,
            args=(self.returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        self.params = result.x
        self.fitted = True
        
        # Compute fitted values
        self._compute_fitted_values(self.returns)
        
        # Store log-likelihood
        self.loglikelihood = -result.fun
        
        return self
    
    def _compute_fitted_values(self, returns: np.ndarray):
        """Compute fitted conditional variances"""
        if self.mean_model == 'constant':
            mu = self.params[0]
            omega = self.params[1]
            alphas = self.params[2:2+self.q]
            betas = self.params[2+self.q:]
            resid = returns - mu
        else:
            omega = self.params[0]
            alphas = self.params[1:1+self.q]
            betas = self.params[1+self.q:]
            resid = returns
        
        T = len(returns)
        self.sigma2 = np.zeros(T)
        self.sigma2[0] = np.var(resid)
        
        for t in range(1, T):
            self.sigma2[t] = omega
            for i in range(self.q):
                if t - i - 1 >= 0:
                    self.sigma2[t] += alphas[i] * resid[t - i - 1]**2
            for j in range(self.p):
                if t - j - 1 >= 0:
                    self.sigma2[t] += betas[j] * self.sigma2[t - j - 1]
        
        self.resid = resid / np.sqrt(self.sigma2)
    
    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility for specified horizon
        
        For GARCH(1,1):
            σ²_{t+h} = V_L + (α + β)^h * (σ²_{t+1} - V_L)
            
        Where V_L is the long-run variance: ω / (1 - α - β)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.mean_model == 'constant':
            omega = self.params[1]
            alphas = self.params[2:2+self.q]
            betas = self.params[2+self.q:]
            resid = self.returns - self.params[0]
        else:
            omega = self.params[0]
            alphas = self.params[1:1+self.q]
            betas = self.params[1+self.q:]
            resid = self.returns
        
        # Long-run variance
        persistence = np.sum(alphas) + np.sum(betas)
        long_run_var = omega / (1 - persistence)
        
        # One-step ahead forecast
        sigma2_1 = omega + np.sum(alphas * resid[-self.q:]**2) + np.sum(betas * self.sigma2[-self.p:])
        
        forecasts = np.zeros(horizon)
        forecasts[0] = np.sqrt(sigma2_1)
        
        # Multi-step ahead
        for h in range(1, horizon):
            sigma2_h = long_run_var + persistence**h * (sigma2_1 - long_run_var)
            forecasts[h] = np.sqrt(sigma2_h)
        
        return forecasts
    
    def summary(self) -> pd.DataFrame:
        """Return model summary"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if self.mean_model == 'constant':
            param_names = ['mu', 'omega'] + [f'alpha[{i+1}]' for i in range(self.q)] + [f'beta[{j+1}]' for j in range(self.p)]
        else:
            param_names = ['omega'] + [f'alpha[{i+1}]' for i in range(self.q)] + [f'beta[{j+1}]' for j in range(self.p)]
        
        if self.mean_model == 'constant':
            persistence = np.sum(self.params[2:])
            long_run_vol = np.sqrt(self.params[1] / (1 - persistence))
        else:
            persistence = np.sum(self.params[1:])
            long_run_vol = np.sqrt(self.params[0] / (1 - persistence))
        
        summary_df = pd.DataFrame({
            'Parameter': param_names,
            'Value': self.params
        })
        
        stats_df = pd.DataFrame({
            'Statistic': ['Persistence', 'Long-run Vol', 'Log-Likelihood'],
            'Value': [persistence, long_run_vol, self.loglikelihood]
        })
        
        print("="*60)
        print(f"GARCH({self.p},{self.q}) Model Summary")
        print("="*60)
        print("\nParameter Estimates:")
        print(summary_df.to_string(index=False))
        print("\nModel Statistics:")
        print(stats_df.to_string(index=False))
        print("="*60)
        
        return summary_df


class EGARCHModel:
    """
    EGARCH(p,q) - Exponential GARCH Model (Nelson, 1991)
    
    Mathematical Model:
        log(σ_t²) = ω + Σ[α_i * |z_{t-i}| + γ_i * z_{t-i}] + Σ[β_j * log(σ²_{t-j})]
        
    Where z_t = ε_t / σ_t is the standardized residual.
    
    Key Advantages:
    1. No parameter constraints (can't have negative volatility)
    2. Captures leverage effect (γ < 0)
    3. Multiplicative rather than additive
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize EGARCH model
        
        Parameters:
        -----------
        p : int
            Order of EGARCH component
        q : int
            Order of ARCH component
        """
        self.p = p
        self.q = q
        self.params = None
        self.sigma2 = None
        self.resid = None
        self.fitted = False
        self.returns = None
        
    def _initial_params(self, returns: np.ndarray) -> np.ndarray:
        """Generate initial parameter guesses"""
        log_var = np.log(np.var(returns))
        
        omega = log_var * 0.05
        alphas = np.array([0.1 / self.q] * self.q)
        gammas = np.array([-0.05 / self.q] * self.q)  # Negative for leverage effect
        betas = np.array([0.90 / self.p] * self.p)
        
        return np.concatenate([[omega], alphas, gammas, betas])
    
    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Compute negative log-likelihood"""
        omega = params[0]
        alphas = params[1:1+self.q]
        gammas = params[1+self.q:1+2*self.q]
        betas = params[1+2*self.q:]
        
        T = len(returns)
        log_sigma2 = np.zeros(T)
        
        # Initial log-variance
        log_sigma2[0] = np.log(np.var(returns))
        
        # Compute conditional log-variances
        for t in range(1, T):
            log_sigma2[t] = omega
            
            for i in range(self.q):
                if t - i - 1 >= 0:
                    # Standardized residual
                    z = returns[t - i - 1] / np.sqrt(np.exp(log_sigma2[t - i - 1]))
                    log_sigma2[t] += alphas[i] * np.abs(z) + gammas[i] * z
            
            for j in range(self.p):
                if t - j - 1 >= 0:
                    log_sigma2[t] += betas[j] * log_sigma2[t - j - 1]
        
        sigma2 = np.exp(log_sigma2)
        sigma2 = np.maximum(sigma2, 1e-8)
        
        # Negative log-likelihood
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + log_sigma2 + returns**2 / sigma2)
        
        return -log_likelihood
    
    def fit(self, returns: pd.Series) -> 'EGARCHModel':
        """Fit EGARCH model using MLE"""
        self.returns = np.asarray(returns)
        
        initial_params = self._initial_params(self.returns)
        
        # No parameter constraints needed for EGARCH!
        result = minimize(
            self._neg_log_likelihood,
            initial_params,
            args=(self.returns,),
            method='BFGS',
            options={'maxiter': 1000, 'disp': False}
        )
        
        if not result.success:
            warnings.warn(f"Optimization warning: {result.message}")
        
        self.params = result.x
        self.fitted = True
        self.loglikelihood = -result.fun
        
        # Compute fitted values
        self._compute_fitted_values(self.returns)
        
        return self
    
    def _compute_fitted_values(self, returns: np.ndarray):
        """Compute fitted conditional variances"""
        omega = self.params[0]
        alphas = self.params[1:1+self.q]
        gammas = self.params[1+self.q:1+2*self.q]
        betas = self.params[1+2*self.q:]
        
        T = len(returns)
        log_sigma2 = np.zeros(T)
        log_sigma2[0] = np.log(np.var(returns))
        
        for t in range(1, T):
            log_sigma2[t] = omega
            for i in range(self.q):
                if t - i - 1 >= 0:
                    z = returns[t - i - 1] / np.sqrt(np.exp(log_sigma2[t - i - 1]))
                    log_sigma2[t] += alphas[i] * np.abs(z) + gammas[i] * z
            for j in range(self.p):
                if t - j - 1 >= 0:
                    log_sigma2[t] += betas[j] * log_sigma2[t - j - 1]
        
        self.sigma2 = np.exp(log_sigma2)
        self.resid = returns / np.sqrt(self.sigma2)
    
    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast volatility"""
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        omega = self.params[0]
        alphas = self.params[1:1+self.q]
        gammas = self.params[1+self.q:1+2*self.q]
        betas = self.params[1+2*self.q:]
        
        # Get recent standardized residuals
        recent_z = self.resid[-self.q:]
        recent_log_sigma2 = np.log(self.sigma2[-self.p:])
        
        forecasts = np.zeros(horizon)
        
        # One-step ahead
        log_sigma2_1 = omega
        for i in range(self.q):
            if i < len(recent_z):
                z = recent_z[-(i+1)]
                log_sigma2_1 += alphas[i] * np.abs(z) + gammas[i] * z
        for j in range(self.p):
            if j < len(recent_log_sigma2):
                log_sigma2_1 += betas[j] * recent_log_sigma2[-(j+1)]
        
        forecasts[0] = np.sqrt(np.exp(log_sigma2_1))
        
        # Multi-step: E[|z_t|] ≈ sqrt(2/π), E[z_t] = 0
        expected_abs_z = np.sqrt(2 / np.pi)
        for h in range(1, horizon):
            log_sigma2_h = omega + np.sum(alphas) * expected_abs_z + np.sum(betas) * log_sigma2_1
            forecasts[h] = np.sqrt(np.exp(log_sigma2_h))
            log_sigma2_1 = log_sigma2_h
        
        return forecasts
    
    def summary(self) -> pd.DataFrame:
        """Return model summary"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        param_names = (['omega'] + 
                      [f'alpha[{i+1}]' for i in range(self.q)] +
                      [f'gamma[{i+1}]' for i in range(self.q)] +
                      [f'beta[{j+1}]' for j in range(self.p)])
        
        summary_df = pd.DataFrame({
            'Parameter': param_names,
            'Value': self.params
        })
        
        # Leverage effect interpretation
        gamma_avg = np.mean(self.params[1+self.q:1+2*self.q])
        leverage_effect = "Strong" if gamma_avg < -0.1 else "Moderate" if gamma_avg < 0 else "None"
        
        print("="*60)
        print(f"EGARCH({self.p},{self.q}) Model Summary")
        print("="*60)
        print("\nParameter Estimates:")
        print(summary_df.to_string(index=False))
        print(f"\nLeverage Effect: {leverage_effect} (avg γ = {gamma_avg:.4f})")
        print(f"Log-Likelihood: {self.loglikelihood:.2f}")
        print("="*60)
        
        return summary_df


def compare_models(returns: pd.Series, models: list = ['ARCH', 'GARCH', 'EGARCH']) -> pd.DataFrame:
    """
    Compare multiple volatility models using information criteria
    
    Parameters:
    -----------
    returns : pd.Series
        Returns series
    models : list
        List of model names to compare
        
    Returns:
    --------
    comparison : pd.DataFrame
        Comparison table with AIC, BIC, and log-likelihood
    """
    results = []
    
    for model_name in models:
        if model_name == 'ARCH':
            model = ARCHModel(q=1).fit(returns)
            n_params = 2
        elif model_name == 'GARCH':
            model = GARCHModel(p=1, q=1).fit(returns)
            n_params = 3
        elif model_name == 'EGARCH':
            model = EGARCHModel(p=1, q=1).fit(returns)
            n_params = 4
        
        T = len(returns)
        log_lik = model.loglikelihood
        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(T)
        
        results.append({
            'Model': model_name,
            'Log-Likelihood': log_lik,
            'AIC': aic,
            'BIC': bic,
            'Parameters': n_params
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df['Best_AIC'] = comparison_df['AIC'] == comparison_df['AIC'].min()
    comparison_df['Best_BIC'] = comparison_df['BIC'] == comparison_df['BIC'].min()
    
    return comparison_df


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    T = 1000
    
    # Simulate GARCH(1,1) process
    omega, alpha, beta = 0.01, 0.08, 0.90
    epsilon = np.random.standard_normal(T)
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * epsilon[t]
    
    returns_series = pd.Series(returns)
    
    print("Testing GARCH(1,1) Model")
    print("-" * 60)
    
    # Fit model
    garch = GARCHModel(p=1, q=1)
    garch.fit(returns_series)
    garch.summary()
    
    # Forecast
    forecasts = garch.forecast(horizon=10)
    print(f"\nVolatility Forecasts (next 10 periods):")
    print(forecasts)
    
    print("\n✓ Model testing complete!")
