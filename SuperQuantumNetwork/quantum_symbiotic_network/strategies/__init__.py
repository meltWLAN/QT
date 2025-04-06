"""
策略模块 - 提供各种交易策略和策略组合功能
"""

from .strategy_ensemble import (
    BaseStrategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
    MACDStrategy,
    VolumeStrategy,
    PriceBreakoutStrategy,
    TrendFollowingStrategy,
    StrategyEnsemble,
    create_default_strategy_ensemble
)

__all__ = [
    'BaseStrategy',
    'MovingAverageCrossStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'VolumeStrategy',
    'PriceBreakoutStrategy',
    'TrendFollowingStrategy',
    'StrategyEnsemble',
    'create_default_strategy_ensemble'
] 