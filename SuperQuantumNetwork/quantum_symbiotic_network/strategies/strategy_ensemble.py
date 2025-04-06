#!/usr/bin/env python3
"""
策略集 - 整合多种交易策略，产生综合信号
"""

import logging
import numpy as np
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StrategyEnsemble")

class BaseStrategy:
    """基础策略类"""
    
    def __init__(self, name, weight=1.0):
        """初始化基础策略
        
        Args:
            name (str): 策略名称
            weight (float): 策略权重
        """
        self.name = name
        self.weight = weight
        
    def generate_signal(self, data):
        """生成交易信号
        
        Args:
            data (dict): 市场数据
            
        Returns:
            dict: 交易信号，包含动作和置信度
        """
        # 基类方法，需要被子类重写
        raise NotImplementedError("子类必须实现generate_signal方法")
        
class MovingAverageCrossStrategy(BaseStrategy):
    """移动平均线交叉策略"""
    
    def __init__(self, short_period=5, long_period=20, weight=1.0):
        """初始化移动平均线交叉策略
        
        Args:
            short_period (int): 短期移动平均线周期
            long_period (int): 长期移动平均线周期
            weight (float): 策略权重
        """
        super().__init__(f"MA{short_period}/{long_period}交叉", weight)
        self.short_period = short_period
        self.long_period = long_period
        
    def generate_signal(self, data):
        """生成交易信号
        
        Args:
            data (dict): 市场数据
            
        Returns:
            dict: 交易信号，包含动作和置信度
        """
        # 检查是否有必要的数据
        short_ma_key = f"ma{self.short_period}"
        long_ma_key = f"ma{self.long_period}"
        
        if short_ma_key not in data or long_ma_key not in data:
            return {"action": "hold", "confidence": 0.0}
            
        short_ma = data[short_ma_key]
        long_ma = data[long_ma_key]
        
        # 检查是否有有效值
        if short_ma is None or long_ma is None:
            return {"action": "hold", "confidence": 0.0}
            
        # 计算信号
        if short_ma > long_ma:
            # 金叉，买入信号
            confidence = min(0.8, (short_ma / long_ma - 1) * 10)
            return {"action": "buy", "confidence": confidence}
        elif short_ma < long_ma:
            # 死叉，卖出信号
            confidence = min(0.8, (1 - short_ma / long_ma) * 10)
            return {"action": "sell", "confidence": confidence}
        else:
            # 平行，持有信号
            return {"action": "hold", "confidence": 0.1}
            
class RSIStrategy(BaseStrategy):
    """RSI超买超卖策略"""
    
    def __init__(self, period=14, oversold=30, overbought=70, weight=1.0):
        """初始化RSI超买超卖策略
        
        Args:
            period (int): RSI周期
            oversold (float): 超卖阈值
            overbought (float): 超买阈值
            weight (float): 策略权重
        """
        super().__init__(f"RSI{period}超买超卖", weight)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def generate_signal(self, data):
        """生成交易信号
        
        Args:
            data (dict): 市场数据
            
        Returns:
            dict: 交易信号，包含动作和置信度
        """
        # 检查是否有必要的数据
        if "rsi" not in data or data["rsi"] is None:
            return {"action": "hold", "confidence": 0.0}
            
        rsi = data["rsi"]
        
        # 计算信号
        if rsi < self.oversold:
            # 超卖，买入信号
            confidence = min(0.9, (self.oversold - rsi) / self.oversold)
            return {"action": "buy", "confidence": confidence}
        elif rsi > self.overbought:
            # 超买，卖出信号
            confidence = min(0.9, (rsi - self.overbought) / (100 - self.overbought))
            return {"action": "sell", "confidence": confidence}
        else:
            # 正常区间，持有信号
            # 根据RSI位置相对中值的偏离程度给出轻微信号
            mid_value = (self.oversold + self.overbought) / 2
            if rsi < mid_value:
                confidence = 0.2 * (mid_value - rsi) / (mid_value - self.oversold)
                return {"action": "buy", "confidence": confidence}
            else:
                confidence = 0.2 * (rsi - mid_value) / (self.overbought - mid_value)
                return {"action": "sell", "confidence": confidence}
                
class MACDStrategy(BaseStrategy):
    """MACD策略"""
    
    def __init__(self, weight=1.0):
        """初始化MACD策略
        
        Args:
            weight (float): 策略权重
        """
        super().__init__("MACD交叉", weight)
        
    def generate_signal(self, data):
        """生成交易信号
        
        Args:
            data (dict): 市场数据
            
        Returns:
            dict: 交易信号，包含动作和置信度
        """
        # 检查是否有必要的数据
        if "macd" not in data or "signal" not in data:
            return {"action": "hold", "confidence": 0.0}
            
        macd = data.get("macd")
        signal = data.get("signal")
        
        if macd is None or signal is None:
            return {"action": "hold", "confidence": 0.0}
            
        # 计算信号
        if macd > signal:
            # MACD在信号线上方，买入信号
            confidence = min(0.8, (macd - signal) * 20)
            return {"action": "buy", "confidence": confidence}
        elif macd < signal:
            # MACD在信号线下方，卖出信号
            confidence = min(0.8, (signal - macd) * 20)
            return {"action": "sell", "confidence": confidence}
        else:
            # 相等，持有信号
            return {"action": "hold", "confidence": 0.1}
            
class VolumeStrategy(BaseStrategy):
    """交易量策略"""
    
    def __init__(self, volume_ratio_threshold=1.5, weight=0.7):
        """初始化交易量策略
        
        Args:
            volume_ratio_threshold (float): 交易量比值阈值
            weight (float): 策略权重
        """
        super().__init__("交易量分析", weight)
        self.volume_ratio_threshold = volume_ratio_threshold
        
    def generate_signal(self, data):
        """生成交易信号
        
        Args:
            data (dict): 市场数据
            
        Returns:
            dict: 交易信号，包含动作和置信度
        """
        # 检查是否有必要的数据
        if "volume" not in data or not data.get("history"):
            return {"action": "hold", "confidence": 0.0}
            
        current_volume = data["volume"]
        
        # 计算过去5天的平均交易量
        history = data["history"]
        if len(history) < 5:
            return {"action": "hold", "confidence": 0.0}
            
        past_volumes = [h["volume"] for h in history[-5:]]
        avg_volume = sum(past_volumes) / len(past_volumes)
        
        # 计算交易量比值
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 计算当日价格变化
        if data.get("close") is None or data.get("open") is None:
            return {"action": "hold", "confidence": 0.0}
            
        price_change = data["close"] - data["open"]
        
        # 生成信号
        if volume_ratio > self.volume_ratio_threshold:
            # 交易量异常放大
            if price_change > 0:
                # 价格上涨，可能是突破开始
                confidence = min(0.7, (volume_ratio - 1) * 0.3)
                return {"action": "buy", "confidence": confidence}
            elif price_change < 0:
                # 价格下跌，可能是恐慌抛售
                confidence = min(0.7, (volume_ratio - 1) * 0.3)
                return {"action": "sell", "confidence": confidence}
                
        # 默认持有
        return {"action": "hold", "confidence": 0.1}
        
class PriceBreakoutStrategy(BaseStrategy):
    """价格突破策略"""
    
    def __init__(self, lookback_days=20, weight=0.8):
        """初始化价格突破策略
        
        Args:
            lookback_days (int): 回顾天数
            weight (float): 策略权重
        """
        super().__init__(f"价格{lookback_days}日突破", weight)
        self.lookback_days = lookback_days
        
    def generate_signal(self, data):
        """生成交易信号
        
        Args:
            data (dict): 市场数据
            
        Returns:
            dict: 交易信号，包含动作和置信度
        """
        # 检查是否有必要的数据
        if "close" not in data or not data.get("history"):
            return {"action": "hold", "confidence": 0.0}
            
        history = data["history"]
        if len(history) < self.lookback_days:
            return {"action": "hold", "confidence": 0.0}
            
        current_price = data["close"]
        
        # 计算历史最高最低价
        highs = [h["high"] for h in history[-self.lookback_days:]]
        lows = [h["low"] for h in history[-self.lookback_days:]]
        
        highest = max(highs)
        lowest = min(lows)
        
        # 计算信号
        if current_price > highest:
            # 突破历史最高价，强烈买入信号
            breakout_ratio = (current_price - highest) / highest
            confidence = min(0.9, breakout_ratio * 10)
            return {"action": "buy", "confidence": confidence}
        elif current_price < lowest:
            # 跌破历史最低价，强烈卖出信号
            breakdown_ratio = (lowest - current_price) / lowest
            confidence = min(0.9, breakdown_ratio * 10)
            return {"action": "sell", "confidence": confidence}
            
        # 计算价格在范围内的位置
        price_range = highest - lowest
        if price_range > 0:
            position = (current_price - lowest) / price_range
            
            if position > 0.7:
                # 接近历史最高价，轻微买入信号
                confidence = 0.3 * (position - 0.7) / 0.3
                return {"action": "buy", "confidence": confidence}
            elif position < 0.3:
                # 接近历史最低价，轻微卖出信号
                confidence = 0.3 * (0.3 - position) / 0.3
                return {"action": "sell", "confidence": confidence}
                
        # 价格在正常范围，持有信号
        return {"action": "hold", "confidence": 0.1}

class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def __init__(self, short_period=5, long_period=20, weight=0.9):
        """初始化趋势跟踪策略
        
        Args:
            short_period (int): 短期趋势周期
            long_period (int): 长期趋势周期
            weight (float): 策略权重
        """
        super().__init__(f"趋势跟踪{short_period}/{long_period}", weight)
        self.short_period = short_period
        self.long_period = long_period
        
    def generate_signal(self, data):
        """生成交易信号
        
        Args:
            data (dict): 市场数据
            
        Returns:
            dict: 交易信号，包含动作和置信度
        """
        # 检查是否有必要的数据
        if "close" not in data or not data.get("history"):
            return {"action": "hold", "confidence": 0.0}
            
        history = data.get("history", [])
        if len(history) < self.long_period:
            return {"action": "hold", "confidence": 0.0}
            
        # 计算短期和长期趋势
        short_prices = [h["close"] for h in history[-self.short_period:]] + [data["close"]]
        long_prices = [h["close"] for h in history[-self.long_period:]] + [data["close"]]
        
        # 计算趋势斜率
        short_slope = (short_prices[-1] - short_prices[0]) / short_prices[0]
        long_slope = (long_prices[-1] - long_prices[0]) / long_prices[0]
        
        # 计算信号
        if short_slope > 0 and long_slope > 0:
            # 短期和长期都是上升趋势，买入信号
            confidence = min(0.8, (short_slope + long_slope) * 5)
            return {"action": "buy", "confidence": confidence}
        elif short_slope < 0 and long_slope < 0:
            # 短期和长期都是下降趋势，卖出信号
            confidence = min(0.8, (-short_slope - long_slope) * 5)
            return {"action": "sell", "confidence": confidence}
        elif short_slope > 0 and long_slope < 0:
            # 短期上升，长期下降，可能是反弹，轻微买入信号
            confidence = min(0.4, short_slope * 5)
            return {"action": "buy", "confidence": confidence}
        elif short_slope < 0 and long_slope > 0:
            # 短期下降，长期上升，可能是回调，轻微卖出信号
            confidence = min(0.4, -short_slope * 5)
            return {"action": "sell", "confidence": confidence}
            
        # 趋势不明显，持有信号
        return {"action": "hold", "confidence": 0.1}
        
class StrategyEnsemble:
    """策略集类，整合多种交易策略"""
    
    def __init__(self, strategies=None):
        """初始化策略集
        
        Args:
            strategies (list): 策略列表
        """
        self.strategies = strategies or []
        self.strategy_weights = {}
        self.performance_metrics = {}
        
        # 计算策略权重
        self._update_weights()
        
    def add_strategy(self, strategy):
        """添加策略
        
        Args:
            strategy (BaseStrategy): 策略对象
        """
        self.strategies.append(strategy)
        self._update_weights()
        
    def _update_weights(self):
        """更新策略权重"""
        total_weight = sum(strategy.weight for strategy in self.strategies)
        
        if total_weight > 0:
            for strategy in self.strategies:
                self.strategy_weights[strategy.name] = strategy.weight / total_weight
                
    def update_performance(self, strategy_name, performance):
        """更新策略性能
        
        Args:
            strategy_name (str): 策略名称
            performance (float): 性能评分
        """
        self.performance_metrics[strategy_name] = performance
        
        # 更新策略权重
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                # 根据性能调整权重
                strategy.weight = max(0.1, strategy.weight * (1 + performance))
                break
                
        self._update_weights()
        
    def generate_signal(self, data):
        """生成综合交易信号
        
        Args:
            data (dict): 市场数据
            
        Returns:
            dict: 交易信号，包含动作和置信度
        """
        if not self.strategies:
            return {"action": "hold", "confidence": 0.0}
            
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        # 收集各策略信号
        for strategy in self.strategies:
            signal = strategy.generate_signal(data)
            weight = self.strategy_weights.get(strategy.name, 0.0)
            
            if signal["action"] == "buy":
                buy_signals.append((signal["confidence"], weight))
            elif signal["action"] == "sell":
                sell_signals.append((signal["confidence"], weight))
            else:
                hold_signals.append((signal["confidence"], weight))
                
        # 计算加权信号
        buy_strength = sum(conf * weight for conf, weight in buy_signals) if buy_signals else 0
        sell_strength = sum(conf * weight for conf, weight in sell_signals) if sell_signals else 0
        hold_strength = sum(conf * weight for conf, weight in hold_signals) if hold_signals else 0
        
        # 添加策略数量因子，避免单一策略误导
        buy_count_factor = min(1.0, len(buy_signals) / 3)
        sell_count_factor = min(1.0, len(sell_signals) / 3)
        
        buy_strength *= buy_count_factor
        sell_strength *= sell_count_factor
        
        # 计算最终信号
        if buy_strength > sell_strength and buy_strength > hold_strength:
            return {"action": "buy", "confidence": buy_strength}
        elif sell_strength > buy_strength and sell_strength > hold_strength:
            return {"action": "sell", "confidence": sell_strength}
        else:
            # 如果持有信号强度几乎与买入/卖出信号相当，选择更有倾向性的信号
            if buy_strength > sell_strength and buy_strength > 0.2:
                return {"action": "buy", "confidence": buy_strength * 0.7}
            elif sell_strength > buy_strength and sell_strength > 0.2:
                return {"action": "sell", "confidence": sell_strength * 0.7}
            else:
                return {"action": "hold", "confidence": max(0.3, hold_strength)}
                
def create_default_strategy_ensemble():
    """创建默认策略集
    
    Returns:
        StrategyEnsemble: 默认策略集
    """
    strategies = [
        MovingAverageCrossStrategy(5, 20, 1.0),
        MovingAverageCrossStrategy(10, 50, 0.8),
        RSIStrategy(14, 30, 70, 1.0),
        MACDStrategy(1.0),
        VolumeStrategy(1.5, 0.7),
        PriceBreakoutStrategy(20, 0.8),
        TrendFollowingStrategy(5, 20, 0.9)
    ]
    
    return StrategyEnsemble(strategies) 