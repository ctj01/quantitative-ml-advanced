"""
Quantitative ML Advanced Package
=================================

A comprehensive library for quantitative trading with advanced ML techniques.

Modules:
--------
- data: Data loading and preprocessing
- models: GARCH, HMM, and other statistical models
- features: Technical indicators and feature engineering
- backtesting: Backtesting framework
- utils: Utility functions

Author: Cristian Mendoza
"""

__version__ = '0.1.0'
__author__ = 'Cristian Mendoza'

from . import data
from . import models
from . import features
from . import utils

__all__ = ['data', 'models', 'features', 'utils']
