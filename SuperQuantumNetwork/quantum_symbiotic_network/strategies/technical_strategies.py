"""
技术分析策略模块 - 包含各种技术指标和交易策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class StrategyBase:
    """策略基类"""
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据市场数据生成交易信号
        
        Args:
            data: 市场数据
            
        Returns:
            交易信号
        """
        raise NotImplementedError("子类必须实现此方法")


class MovingAverageCrossStrategy(StrategyBase):
    """
    移动平均线交叉策略
    
    当短期均线上穿长期均线时买入，下穿时卖出
    """
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.short_window = self.config.get("short_window", 5)
        self.long_window = self.config.get("long_window", 20)
        self.buy_threshold = self.config.get("buy_threshold", 0.01)  # 上穿阈值
        self.sell_threshold = self.config.get("sell_threshold", -0.01)  # 下穿阈值
        
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 获取历史数据
        symbol = data.get("symbol", "default")
        history = data.get("history", [])
        history.append(data)  # 添加当前数据
        
        if len(history) < self.long_window + 1:
            return {"action": "hold", "symbol": symbol, "confidence": 0.0}
            
        # 计算短期和长期移动平均线
        closes = [bar["close"] for bar in history[-self.long_window-1:]]
        short_ma = np.mean(closes[-self.short_window:])
        long_ma = np.mean(closes)
        
        # 计算前一日的均线差值
        prev_closes = closes[:-1]
        prev_short_ma = np.mean(prev_closes[-self.short_window:])
        prev_long_ma = np.mean(prev_closes)
        
        prev_diff = prev_short_ma - prev_long_ma
        curr_diff = short_ma - long_ma
        
        # 计算均线差距变化
        diff_change = curr_diff - prev_diff
        
        # 生成交易信号
        # 根据上下穿强度计算置信度
        confidence = min(abs(diff_change) / long_ma * 10, 1.0)
        
        if curr_diff > 0 and prev_diff <= 0:
            # 短期均线上穿长期均线
            return {"action": "buy", "symbol": symbol, "confidence": confidence}
        elif curr_diff < 0 and prev_diff >= 0:
            # 短期均线下穿长期均线
            return {"action": "sell", "symbol": symbol, "confidence": confidence}
        elif diff_change > self.buy_threshold * long_ma:
            # 差距扩大（向上），考虑买入
            return {"action": "buy", "symbol": symbol, "confidence": confidence * 0.7}
        elif diff_change < self.sell_threshold * long_ma:
            # 差距扩大（向下），考虑卖出
            return {"action": "sell", "symbol": symbol, "confidence": confidence * 0.7}
        else:
            # 无明显信号
            return {"action": "hold", "symbol": symbol, "confidence": 0.0}


class RSIStrategy(StrategyBase):
    """
    RSI策略
    
    当RSI低于超卖阈值时买入，高于超买阈值时卖出
    """
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.oversold_threshold = self.config.get("oversold_threshold", 30)
        self.overbought_threshold = self.config.get("overbought_threshold", 70)
        
    def calculate_rsi(self, closes: List[float]) -> float:
        """计算RSI值"""
        if len(closes) <= self.rsi_period:
            return 50.0  # 默认中性值
            
        # 计算价格变化
        deltas = np.diff(closes)
        
        # 分离上涨和下跌
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # 计算平均上涨和下跌
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = data.get("symbol", "default")
        
        # 如果数据直接包含RSI，直接使用
        if "rsi" in data and data["rsi"] is not None:
            rsi = data["rsi"]
        else:
            # 否则自行计算
            history = data.get("history", [])
            history.append(data)
            
            if len(history) < self.rsi_period + 1:
                return {"action": "hold", "symbol": symbol, "confidence": 0.0}
                
            closes = [bar["close"] for bar in history]
            rsi = self.calculate_rsi(closes)
            
        # 生成交易信号
        if rsi < self.oversold_threshold:
            # 超卖区域，买入信号
            # 根据距离阈值的远近计算置信度
            confidence = min((self.oversold_threshold - rsi) / self.oversold_threshold, 1.0)
            return {"action": "buy", "symbol": symbol, "confidence": confidence}
        elif rsi > self.overbought_threshold:
            # 超买区域，卖出信号
            confidence = min((rsi - self.overbought_threshold) / (100 - self.overbought_threshold), 1.0)
            return {"action": "sell", "symbol": symbol, "confidence": confidence}
        else:
            # 中性区域，无明显信号
            return {"action": "hold", "symbol": symbol, "confidence": 0.0}


class MACDStrategy(StrategyBase):
    """
    MACD策略
    
    当MACD线上穿信号线时买入，下穿时卖出
    """
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.fast_period = self.config.get("fast_period", 12)
        self.slow_period = self.config.get("slow_period", 26)
        self.signal_period = self.config.get("signal_period", 9)
        
    def calculate_macd(self, closes: List[float]) -> tuple:
        """计算MACD值"""
        if len(closes) < self.slow_period + self.signal_period:
            return 0.0, 0.0, 0.0
            
        # 计算快线和慢线的EMA
        ema_fast = pd.Series(closes).ewm(span=self.fast_period, adjust=False).mean().values[-1]
        ema_slow = pd.Series(closes).ewm(span=self.slow_period, adjust=False).mean().values[-1]
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线 (MACD的EMA)
        macd_series = []
        for i in range(len(closes) - self.slow_period + 1):
            ema_f = pd.Series(closes[i:i+self.slow_period]).ewm(span=self.fast_period, adjust=False).mean().values[-1]
            ema_s = pd.Series(closes[i:i+self.slow_period]).ewm(span=self.slow_period, adjust=False).mean().values[-1]
            macd_series.append(ema_f - ema_s)
            
        signal_line = pd.Series(macd_series).ewm(span=self.signal_period, adjust=False).mean().values[-1]
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = data.get("symbol", "default")
        
        # 如果数据直接包含MACD相关指标，直接使用
        if all(k in data and data[k] is not None for k in ["macd", "signal"]):
            macd_line = data["macd"]
            signal_line = data["signal"]
            histogram = macd_line - signal_line
        else:
            # 否则自行计算
            history = data.get("history", [])
            history.append(data)
            
            if len(history) < self.slow_period + self.signal_period:
                return {"action": "hold", "symbol": symbol, "confidence": 0.0}
                
            closes = [bar["close"] for bar in history]
            macd_line, signal_line, histogram = self.calculate_macd(closes)
            
        # 获取前一日数据计算变化
        prev_data = data.get("history", [])[-1] if data.get("history", []) else None
        
        if prev_data:
            if all(k in prev_data and prev_data[k] is not None for k in ["macd", "signal"]):
                prev_macd = prev_data["macd"]
                prev_signal = prev_data["signal"]
                prev_histogram = prev_macd - prev_signal
            else:
                # 使用计算好的当前MACD假设
                prev_histogram = 0
        else:
            prev_histogram = 0
            
        # 生成交易信号
        # 柱状图由负转正，买入信号
        if histogram > 0 and prev_histogram <= 0:
            confidence = min(abs(histogram) / abs(macd_line) * 2 if macd_line != 0 else 0.5, 1.0)
            return {"action": "buy", "symbol": symbol, "confidence": confidence}
        # 柱状图由正转负，卖出信号
        elif histogram < 0 and prev_histogram >= 0:
            confidence = min(abs(histogram) / abs(macd_line) * 2 if macd_line != 0 else 0.5, 1.0)
            return {"action": "sell", "symbol": symbol, "confidence": confidence}
        # 柱状图持续走高，继续买入
        elif histogram > 0 and histogram > prev_histogram:
            confidence = min((histogram - prev_histogram) / abs(macd_line) * 3 if macd_line != 0 else 0.3, 0.8)
            return {"action": "buy", "symbol": symbol, "confidence": confidence}
        # 柱状图持续走低，继续卖出
        elif histogram < 0 and histogram < prev_histogram:
            confidence = min((prev_histogram - histogram) / abs(macd_line) * 3 if macd_line != 0 else 0.3, 0.8)
            return {"action": "sell", "symbol": symbol, "confidence": confidence}
        else:
            return {"action": "hold", "symbol": symbol, "confidence": 0.0}


class BollingerBandsStrategy(StrategyBase):
    """
    布林带策略
    
    当价格触及下轨时买入，触及上轨时卖出
    """
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.period = self.config.get("period", 20)
        self.std_dev = self.config.get("std_dev", 2.0)
        
    def calculate_bollinger_bands(self, closes: List[float]) -> tuple:
        """计算布林带值"""
        if len(closes) < self.period:
            # 返回默认值
            middle = closes[-1]
            upper = middle * 1.1
            lower = middle * 0.9
            return middle, upper, lower
            
        # 计算中轨 (SMA)
        middle = np.mean(closes[-self.period:])
        
        # 计算标准差
        std = np.std(closes[-self.period:])
        
        # 计算上下轨
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        return middle, upper, lower
        
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = data.get("symbol", "default")
        close = data.get("close", 0)
        
        # 如果数据直接包含布林带值，直接使用
        if all(k in data and data[k] is not None for k in ["boll_mid", "boll_upper", "boll_lower"]):
            middle = data["boll_mid"]
            upper = data["boll_upper"]
            lower = data["boll_lower"]
        else:
            # 否则自行计算
            history = data.get("history", [])
            history.append(data)
            
            if len(history) < self.period:
                return {"action": "hold", "symbol": symbol, "confidence": 0.0}
                
            closes = [bar["close"] for bar in history]
            middle, upper, lower = self.calculate_bollinger_bands(closes)
            
        # 生成交易信号
        # 计算价格位置百分比 (0表示下轨，1表示上轨)
        if upper == lower:  # 防止除零错误
            position = 0.5
        else:
            position = (close - lower) / (upper - lower)
            
        # 计算置信度
        if position < 0.2:
            # 接近下轨，买入信号
            confidence = min((0.2 - position) * 5, 1.0)
            return {"action": "buy", "symbol": symbol, "confidence": confidence}
        elif position > 0.8:
            # 接近上轨，卖出信号
            confidence = min((position - 0.8) * 5, 1.0)
            return {"action": "sell", "symbol": symbol, "confidence": confidence}
        else:
            # 处于中间区域，无明显信号
            return {"action": "hold", "symbol": symbol, "confidence": 0.0}


class TrendFollowingStrategy(StrategyBase):
    """
    趋势跟踪策略
    
    使用多种技术指标确认趋势方向
    """
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.ma_period = self.config.get("ma_period", 20)
        self.ema_period = self.config.get("ema_period", 10)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.strength_threshold = self.config.get("strength_threshold", 0.3)
        
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = data.get("symbol", "default")
        close = data.get("close", 0)
        
        history = data.get("history", [])
        if not history:
            return {"action": "hold", "symbol": symbol, "confidence": 0.0}
            
        history.append(data)
        
        # 收集信号
        signals = []
        
        # 1. 价格与移动平均线的关系
        if len(history) > self.ma_period:
            closes = [bar["close"] for bar in history]
            ma = np.mean(closes[-self.ma_period:])
            
            # 价格相对MA的位置
            if close > ma * 1.03:  # 价格显著高于MA
                signals.append(("ma", "buy", min((close/ma - 1) * 3, 1.0)))
            elif close < ma * 0.97:  # 价格显著低于MA
                signals.append(("ma", "sell", min((1 - close/ma) * 3, 1.0)))
                
        # 2. 价格动量
        if len(history) > 5:
            prev_closes = [bar["close"] for bar in history[-6:-1]]
            momentum = (close / np.mean(prev_closes)) - 1
            
            if momentum > 0.02:  # 正动量
                signals.append(("momentum", "buy", min(momentum * 20, 1.0)))
            elif momentum < -0.02:  # 负动量
                signals.append(("momentum", "sell", min(abs(momentum) * 20, 1.0)))
                
        # 3. RSI指标
        if "rsi" in data and data["rsi"] is not None:
            rsi = data["rsi"]
            
            if rsi < 30:  # 超卖
                signals.append(("rsi", "buy", min((30 - rsi) / 30, 1.0)))
            elif rsi > 70:  # 超买
                signals.append(("rsi", "sell", min((rsi - 70) / 30, 1.0)))
                
        # 4. MACD指标
        if all(k in data and data[k] is not None for k in ["macd", "signal"]):
            macd = data["macd"]
            signal = data["signal"]
            histogram = macd - signal
            
            if histogram > 0 and abs(histogram) > 0.01 * close:
                signals.append(("macd", "buy", min(abs(histogram) / (0.01 * close), 1.0)))
            elif histogram < 0 and abs(histogram) > 0.01 * close:
                signals.append(("macd", "sell", min(abs(histogram) / (0.01 * close), 1.0)))
                
        # 汇总信号
        if not signals:
            return {"action": "hold", "symbol": symbol, "confidence": 0.0}
            
        # 计算买入和卖出信号的总强度
        buy_strength = sum(conf for _, action, conf in signals if action == "buy")
        sell_strength = sum(conf for _, action, conf in signals if action == "sell")
        
        # 归一化强度
        signal_count = len(signals)
        if signal_count > 0:
            buy_strength /= signal_count
            sell_strength /= signal_count
            
        # 决定最终动作
        if buy_strength > sell_strength and buy_strength > self.strength_threshold:
            return {"action": "buy", "symbol": symbol, "confidence": buy_strength}
        elif sell_strength > buy_strength and sell_strength > self.strength_threshold:
            return {"action": "sell", "symbol": symbol, "confidence": sell_strength}
        else:
            return {"action": "hold", "symbol": symbol, "confidence": 0.0}


class StrategyEnsemble:
    """
    策略集成器
    
    整合多个策略的信号生成最终决策
    """
    def __init__(self, strategies: List[StrategyBase] = None, weights: Dict[str, float] = None):
        self.strategies = strategies or []
        self.weights = weights or {}
        
        # 如果没有指定权重，使用相等权重
        if not self.weights:
            equal_weight = 1.0 / len(self.strategies) if self.strategies else 0
            self.weights = {strategy.name: equal_weight for strategy in self.strategies}
            
    def add_strategy(self, strategy: StrategyBase, weight: float = None):
        """添加策略"""
        self.strategies.append(strategy)
        
        if weight is None:
            # 重新计算均等权重
            equal_weight = 1.0 / len(self.strategies)
            self.weights = {strategy.name: equal_weight for strategy in self.strategies}
        else:
            # 添加新策略的权重
            total_weight = sum(self.weights.values()) + weight
            scale = 1.0 / total_weight
            
            # 重新调整所有权重
            for name in self.weights:
                self.weights[name] *= scale
                
            self.weights[strategy.name] = weight * scale
            
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合信号
        
        Args:
            data: 市场数据
            
        Returns:
            综合交易信号
        """
        if not self.strategies:
            return {"action": "hold", "symbol": data.get("symbol", "default"), "confidence": 0.0}
            
        # 收集每个策略的信号
        signals = []
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(data)
                weight = self.weights.get(strategy.name, 1.0 / len(self.strategies))
                signals.append((signal, weight))
                logger.debug(f"策略 {strategy.name} 生成信号: {signal['action']}, 置信度: {signal['confidence']:.2f}")
            except Exception as e:
                logger.error(f"策略 {strategy.name} 生成信号时出错: {e}")
                
        if not signals:
            return {"action": "hold", "symbol": data.get("symbol", "default"), "confidence": 0.0}
            
        # 计算各种动作的加权得分
        action_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        
        for signal, weight in signals:
            action = signal["action"]
            confidence = signal["confidence"]
            
            if action in action_scores:
                action_scores[action] += confidence * weight
                
        # 找出得分最高的动作
        max_action = max(action_scores.items(), key=lambda x: x[1])
        action = max_action[0]
        score = max_action[1]
        
        # 如果最高得分低于阈值，改为hold
        if score < 0.2:
            action = "hold"
            score = 0.0
            
        return {
            "action": action,
            "symbol": data.get("symbol", "default"),
            "confidence": score,
            "strategy_ensemble": True,
            "individual_signals": [(s["action"], s["confidence"]) for s, _ in signals]
        }


# 创建默认策略集
def create_default_strategy_ensemble() -> StrategyEnsemble:
    """
    创建默认策略集
    
    Returns:
        策略集实例
    """
    # 创建各个策略
    ma_strategy = MovingAverageCrossStrategy({
        "short_window": 5,
        "long_window": 20
    })
    
    rsi_strategy = RSIStrategy({
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70
    })
    
    macd_strategy = MACDStrategy()
    
    bb_strategy = BollingerBandsStrategy()
    
    trend_strategy = TrendFollowingStrategy()
    
    # 创建策略集
    ensemble = StrategyEnsemble()
    
    # 添加策略并设置权重
    ensemble.add_strategy(ma_strategy, 0.2)
    ensemble.add_strategy(rsi_strategy, 0.2)
    ensemble.add_strategy(macd_strategy, 0.2)
    ensemble.add_strategy(bb_strategy, 0.2)
    ensemble.add_strategy(trend_strategy, 0.2)
    
    return ensemble 