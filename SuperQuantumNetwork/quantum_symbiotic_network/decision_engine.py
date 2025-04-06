"""
决策引擎 - 集成系统的各个组件，提供最终交易决策
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class DecisionEngine:
    """决策引擎类 - 生成交易决策和预测"""
    
    def __init__(self, config: Dict[str, Any],
                quantum_framework=None, 
                market_analyzer=None,
                neural_network=None,
                fractal_intelligence=None,
                sentiment_analyzer=None,
                risk_manager=None,
                position_manager=None,
                quantum_entanglement_engine=None):
        """
        初始化决策引擎
        
        Args:
            config: 配置参数
            quantum_framework: 量子概率框架
            market_analyzer: 市场分析器
            neural_network: 神经网络
            fractal_intelligence: 分形智能
            sentiment_analyzer: 情感分析器
            risk_manager: 风险管理器
            position_manager: 持仓管理器
            quantum_entanglement_engine: 量子纠缠引擎
        """
        self.config = config
        self.quantum_framework = quantum_framework
        self.market_analyzer = market_analyzer
        self.neural_network = neural_network
        self.fractal_intelligence = fractal_intelligence
        self.sentiment_analyzer = sentiment_analyzer
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.quantum_entanglement_engine = quantum_entanglement_engine
        
        # 子策略权重
        self.strategy_weights = config.get("strategy_weights", {
            "quantum": 0.3,
            "neural": 0.2,
            "fractal": 0.2,
            "market": 0.15,
            "sentiment": 0.05,
            "quantum_entanglement": 0.1
        })
        
        # 正在跟踪的决策
        self.active_decisions = {}
        
        # 决策历史
        self.decision_history = []
        
        # 当前市场情绪
        self.market_sentiment = 0.0
        
        logger.info("决策引擎初始化完成")
    
    def initialize(self) -> bool:
        """初始化决策引擎，连接各个子组件"""
        logger.info("初始化决策引擎")
        return True
    
    def make_decisions(self, market_data: Dict[str, Any], 
                      market_state: Dict[str, Any] = None,
                      resonance_state: Dict[str, float] = None) -> Dict[str, Dict[str, Any]]:
        """
        基于当前市场状态和各个子系统生成交易决策
        
        Args:
            market_data: 市场数据
            market_state: 市场状态分析结果
            resonance_state: 量子共振状态
            
        Returns:
            交易决策字典 {symbol: decision_data}
        """
        logger.info("生成交易决策")
        decisions = {}
        
        # 收集所有交易标的
        symbols = []
        for segment, segment_data in market_data.items():
            for symbol in segment_data.keys():
                if symbol not in symbols:
                    symbols.append(symbol)
        
        # 获取市场整体情绪
        self.market_sentiment = self._estimate_market_sentiment(market_state)
        
        # 获取量子预测
        quantum_predictions = {}
        if self.quantum_framework:
            for segment in market_data:
                # 生成量子决策
                quantum_decision = self.quantum_framework.quantum_decision(
                    segment=segment,
                    classical_signals=market_state
                )
                
                # 提取该分段所有股票
                segment_symbols = list(market_data.get(segment, {}).keys())
                
                # 应用到所有股票
                for symbol in segment_symbols:
                    quantum_predictions[symbol] = {
                        "action": quantum_decision.get("action", "hold"),
                        "confidence": quantum_decision.get("confidence", 0.5),
                        "direction": 0.0 if quantum_decision.get("action") == "hold" else 
                                    (1.0 if quantum_decision.get("action") == "buy" else -1.0)
                    }
        
        # 获取量子纠缠预测
        entanglement_predictions = {}
        if self.quantum_entanglement_engine:
            entanglement_predictions = self.quantum_entanglement_engine.predict_market_movement(symbols)
        
        # 获取神经网络预测
        neural_predictions = {}
        if self.neural_network:
            # 将市场状态转换为神经网络输入
            neural_inputs = self._prepare_neural_inputs(market_state)
            
            # 生成神经网络预测
            neural_outputs = self.neural_network.forward(neural_inputs)
            
            # 转换为决策
            for symbol in symbols:
                # 确定该股票所属分段
                segment = self._find_segment_for_symbol(symbol, market_data)
                
                if segment and segment in neural_outputs:
                    output = neural_outputs[segment]
                    
                    # 找出最大概率的动作
                    actions = ["buy", "sell", "hold"]
                    probs = [output.get(i, 0) for i in range(3)]
                    max_idx = np.argmax(probs)
                    
                    neural_predictions[symbol] = {
                        "action": actions[max_idx],
                        "confidence": probs[max_idx],
                        "probabilities": {actions[i]: probs[i] for i in range(3)}
                    }
        
        # 分形预测
        fractal_predictions = {}
        if self.fractal_intelligence:
            # TODO: 实现分形智能预测
            pass
        
        # 情感分析预测
        sentiment_predictions = {}
        if self.sentiment_analyzer:
            # TODO: 实现情感分析预测
            for symbol in symbols:
                sentiment_predictions[symbol] = {
                    "sentiment_score": 0.0,  # 默认中性
                    "confidence": 0.5
                }
        
        # 生成最终决策
        for symbol in symbols:
            # 收集各个预测结果
            quantum_pred = quantum_predictions.get(symbol, {"action": "hold", "confidence": 0.5, "direction": 0.0})
            entanglement_pred = entanglement_predictions.get(symbol, {"direction": 0.0, "strength": 0.5})
            neural_pred = neural_predictions.get(symbol, {"action": "hold", "confidence": 0.5})
            fractal_pred = fractal_predictions.get(symbol, {"action": "hold", "confidence": 0.5})
            sentiment_pred = sentiment_predictions.get(symbol, {"sentiment_score": 0.0, "confidence": 0.5})
            
            # 市场数据
            symbol_data = None
            symbol_segment = self._find_segment_for_symbol(symbol, market_data)
            if symbol_segment and symbol in market_data.get(symbol_segment, {}):
                symbol_data = market_data[symbol_segment][symbol]
            
            # 生成最终决策
            decision = self._blend_predictions(
                symbol=symbol,
                symbol_data=symbol_data,
                quantum_pred=quantum_pred,
                entanglement_pred=entanglement_pred,
                neural_pred=neural_pred,
                fractal_pred=fractal_pred,
                sentiment_pred=sentiment_pred,
                resonance=resonance_state.get(symbol, 0.5) if resonance_state else 0.5
            )
            
            # 应用风险管理
            if self.risk_manager:
                # 检查是否符合风险标准
                risk_check = self.risk_manager.evaluate_risk(symbol, decision, symbol_data)
                
                # 如果风险过高，将动作改为"hold"
                if not risk_check.get("approved", True):
                    decision["action"] = "hold"
                    decision["override_reason"] = f"Risk: {risk_check.get('reason', 'unknown')}"
            
            # 记录决策
            if decision.get("action") != "hold":
                self.active_decisions[symbol] = {
                    "decision": decision,
                    "timestamp": datetime.now().isoformat(),
                    "status": "active"
                }
                
                # 保存到历史记录
                self.decision_history.append({
                    "symbol": symbol,
                    "decision": decision,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 保持历史记录长度不超过1000
                if len(self.decision_history) > 1000:
                    self.decision_history.pop(0)
                
            # 存储决策
            decisions[symbol] = decision
        
        logger.info(f"生成了{len(decisions)}个交易决策")
        return decisions
    
    def _estimate_market_sentiment(self, market_state: Dict[str, Any]) -> float:
        """
        估计整体市场情绪
        
        Args:
            market_state: 市场状态
            
        Returns:
            市场情绪分数，-1.0到1.0，正值表示乐观，负值表示悲观
        """
        sentiment_scores = []
        
        # 从市场状态中提取信号
        for segment, state in market_state.items():
            if "sentiment" in state:
                sentiment_scores.append(state["sentiment"])
        
        # 计算整体情绪
        if sentiment_scores:
            return np.mean(sentiment_scores)
        else:
            return 0.0  # 默认中性
    
    def _prepare_neural_inputs(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """
        准备神经网络输入数据
        
        Args:
            market_state: 市场状态
            
        Returns:
            神经网络输入字典
        """
        inputs = {}
        
        # 将市场状态转换为神经网络输入
        for segment, state in market_state.items():
            # 可以添加更多特征
            segment_id = segment  # 使用分段作为节点ID
            
            features = {
                "price_momentum": state.get("momentum", 0.0),
                "volume_change": state.get("volume_change", 0.0),
                "volatility": state.get("volatility", 0.0),
                "sentiment": state.get("sentiment", 0.0),
                "rsi": state.get("rsi", 50.0) / 100.0,  # 归一化到0-1
                "ma_trend": state.get("ma_trend", 0.0)
            }
            
            # 创建输入向量
            for feature, value in features.items():
                inputs[f"{segment_id}_{feature}"] = value
        
        return inputs
    
    def _find_segment_for_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Optional[str]:
        """
        查找股票所属的市场分段
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            
        Returns:
            市场分段名称，如果找不到则返回None
        """
        for segment, segment_data in market_data.items():
            if symbol in segment_data:
                return segment
        return None
    
    def _blend_predictions(self, symbol: str, symbol_data: Dict[str, Any],
                         quantum_pred: Dict[str, Any],
                         entanglement_pred: Dict[str, Any],
                         neural_pred: Dict[str, Any],
                         fractal_pred: Dict[str, Any],
                         sentiment_pred: Dict[str, Any],
                         resonance: float) -> Dict[str, Any]:
        """
        混合各个预测源，生成最终决策
        
        Args:
            symbol: 股票代码
            symbol_data: 股票市场数据
            quantum_pred: 量子预测
            entanglement_pred: 量子纠缠预测
            neural_pred: 神经网络预测
            fractal_pred: 分形预测
            sentiment_pred: 情感分析预测
            resonance: 量子共振度
            
        Returns:
            最终决策字典
        """
        # 获取各个策略权重
        w_quantum = self.strategy_weights.get("quantum", 0.3)
        w_neural = self.strategy_weights.get("neural", 0.2)
        w_fractal = self.strategy_weights.get("fractal", 0.2)
        w_market = self.strategy_weights.get("market", 0.15)
        w_sentiment = self.strategy_weights.get("sentiment", 0.05)
        w_entanglement = self.strategy_weights.get("quantum_entanglement", 0.1)
        
        # 计算加权方向分数
        direction_score = 0.0
        
        # 量子方向
        quantum_direction = quantum_pred.get("direction", 0.0)
        quantum_confidence = quantum_pred.get("confidence", 0.5)
        direction_score += quantum_direction * quantum_confidence * w_quantum
        
        # 量子纠缠方向
        entanglement_direction = entanglement_pred.get("direction", 0.0)
        entanglement_strength = entanglement_pred.get("strength", 0.5)
        direction_score += entanglement_direction * entanglement_strength * w_entanglement
        
        # 神经网络方向
        neural_action = neural_pred.get("action", "hold")
        neural_confidence = neural_pred.get("confidence", 0.5)
        neural_direction = 0.0
        if neural_action == "buy":
            neural_direction = 1.0
        elif neural_action == "sell":
            neural_direction = -1.0
        direction_score += neural_direction * neural_confidence * w_neural
        
        # 分形方向
        fractal_action = fractal_pred.get("action", "hold")
        fractal_confidence = fractal_pred.get("confidence", 0.5)
        fractal_direction = 0.0
        if fractal_action == "buy":
            fractal_direction = 1.0
        elif fractal_action == "sell":
            fractal_direction = -1.0
        direction_score += fractal_direction * fractal_confidence * w_fractal
        
        # 情感影响
        sentiment_score = sentiment_pred.get("sentiment_score", 0.0)
        sentiment_confidence = sentiment_pred.get("confidence", 0.5)
        direction_score += sentiment_score * sentiment_confidence * w_sentiment
        
        # 市场数据影响
        market_signal = 0.0
        if symbol_data:
            # 根据价格变动添加市场信号
            price_change = symbol_data.get("price_change_pct", 0.0)
            volume_change = symbol_data.get("volume_change_pct", 0.0)
            
            # 简单动量策略：价格上涨且成交量增加为买入信号，反之为卖出信号
            if price_change > 0 and volume_change > 0:
                market_signal = min(price_change, volume_change) * 5  # 乘以5放大效果
            elif price_change < 0 and volume_change > 0:
                market_signal = max(price_change, -volume_change) * 5
        
        direction_score += market_signal * w_market
        
        # 根据最终方向分数确定行动
        action = "hold"
        # 设置阈值，避免过于频繁交易
        buy_threshold = 0.2
        sell_threshold = -0.2
        
        # 应用共振因子 - 高共振时降低交易阈值，更容易产生交易信号
        resonance_factor = 0.5 + resonance * 0.5  # 0.5-1.0
        buy_threshold = buy_threshold / resonance_factor
        sell_threshold = sell_threshold / resonance_factor
        
        if direction_score > buy_threshold:
            action = "buy"
        elif direction_score < sell_threshold:
            action = "sell"
        
        # 计算置信度 - 基于方向分数的绝对值和共振度
        confidence = min(abs(direction_score) * 2, 1.0)  # 将方向分数映射到0-1
        confidence = confidence * (0.7 + 0.3 * resonance)  # 应用共振因子
        
        # 生成决策字典
        decision = {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "direction_score": direction_score,
            "quantity": self._calculate_position_size(symbol, action, confidence, resonance),
            "timestamp": datetime.now().isoformat(),
            "components": {
                "quantum": {
                    "weight": w_quantum,
                    "prediction": quantum_pred
                },
                "entanglement": {
                    "weight": w_entanglement,
                    "prediction": entanglement_pred
                },
                "neural": {
                    "weight": w_neural,
                    "prediction": neural_pred
                },
                "fractal": {
                    "weight": w_fractal,
                    "prediction": fractal_pred
                },
                "sentiment": {
                    "weight": w_sentiment,
                    "prediction": sentiment_pred
                },
                "market": {
                    "weight": w_market,
                    "signal": market_signal
                }
            },
            "resonance": resonance
        }
        
        return decision
    
    def _calculate_position_size(self, symbol: str, action: str, 
                               confidence: float, resonance: float) -> int:
        """
        计算头寸大小
        
        Args:
            symbol: 股票代码
            action: 交易动作
            confidence: 决策置信度
            resonance: 量子共振度
            
        Returns:
            建议交易数量
        """
        if action == "hold":
            return 0
            
        # 基础头寸
        base_quantity = 100  # 默认手数
        
        # 根据置信度调整
        confidence_factor = 1.0 + confidence  # 1.0-2.0
        
        # 根据共振度调整
        resonance_factor = 1.0 + resonance * 0.5  # 1.0-1.5
        
        # 计算最终数量
        quantity = int(base_quantity * confidence_factor * resonance_factor / 100) * 100
        
        # 如果是卖出，检查当前持仓
        if action == "sell" and self.position_manager:
            position = self.position_manager.get_position(symbol)
            if position:
                current_holding = position.get("quantity", 0)
                quantity = min(quantity, current_holding)
        
        return quantity
    
    def generate_predictions(self, symbols: List[str], 
                           entanglement_predictions: Dict[str, Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
        """
        生成市场预测
        
        Args:
            symbols: 股票代码列表
            entanglement_predictions: 量子纠缠预测结果
            
        Returns:
            预测结果字典 {symbol: prediction_data}
        """
        predictions = {}
        
        for symbol in symbols:
            # 获取量子纠缠预测
            entanglement_pred = entanglement_predictions.get(symbol, {}) if entanglement_predictions else {}
            
            # 获取活跃决策
            active_decision = self.active_decisions.get(symbol, {}).get("decision", {})
            
            # 创建预测
            prediction = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "horizon": "1d",  # 默认预测1天
                "direction": 0.0,
                "confidence": 0.0,
                "price_target": None,
                "price_range": None
            }
            
            # 如果有量子纠缠预测，使用它
            if entanglement_pred:
                prediction["direction"] = entanglement_pred.get("direction", 0.0)
                prediction["confidence"] = entanglement_pred.get("strength", 0.5)
                prediction["entanglement_data"] = {
                    "up_probability": entanglement_pred.get("up_probability", 0.5),
                    "down_probability": entanglement_pred.get("down_probability", 0.5),
                    "resonance": entanglement_pred.get("resonance", 0.5)
                }
            # 否则使用活跃决策
            elif active_decision:
                prediction["direction"] = active_decision.get("direction_score", 0.0)
                prediction["confidence"] = active_decision.get("confidence", 0.0)
            
            # 添加到结果
            predictions[symbol] = prediction
        
        return predictions 