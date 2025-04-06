"""
分形智能结构核心
实现智能体的分形组织架构，从微观到宏观形成递归式决策网络
"""

import uuid
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import time

# 设置日志
logger = logging.getLogger(__name__)

class Agent(ABC):
    """智能体基类"""
    
    def __init__(self, agent_id: Optional[str] = None, specialization: Optional[Dict] = None):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体ID，如果为None则自动生成
            specialization: 智能体专精配置
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.specialization = specialization or {}
        self.performance_history = []
        self.active = True
        self.creation_time = 0  # 将在系统运行时设置
        self.last_updated = 0   # 最后更新时间
        self.connections = []   # 与其他智能体的连接
        self.state = {}         # 当前状态
        self.knowledge = {}     # 累积的知识
        
    @abstractmethod
    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """感知环境"""
        pass
        
    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """基于感知做出决策"""
        pass
        
    @abstractmethod
    def adapt(self, feedback: Dict[str, Any]) -> None:
        """基于反馈调整自身"""
        pass
        
    def connect(self, other_agent: 'Agent', strength: float = 1.0) -> None:
        """与其他智能体建立连接"""
        self.connections.append((other_agent, strength))
        
    def get_fitness(self) -> float:
        """获取智能体的适应度"""
        if not self.performance_history:
            return 0.0
        # 偏向近期表现，使用指数加权
        weights = np.exp(np.linspace(0, 1, len(self.performance_history)))
        weighted_performance = np.average(self.performance_history, weights=weights)
        return weighted_performance


class MicroAgent(Agent):
    """微观智能体 - 处理狭窄的市场领域"""
    
    def __init__(self, market_segment: str, feature_focus: List[str], **kwargs):
        """
        初始化微观智能体
        
        Args:
            market_segment: 关注的市场片段
            feature_focus: 关注的特征列表
        """
        super().__init__(**kwargs)
        self.market_segment = market_segment
        self.feature_focus = feature_focus
        self.prediction_model = None  # 将在初始化后设置
        self.prediction_history = []
        self.confidence_history = []
        self.last_decisions = []  # 存储最近的决策历史
        self.success_rate = 0.5  # 初始成功率
        self.decision_count = 0  # 决策计数
        
    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """感知环境中与自己专注领域相关的信息"""
        # 提取相关市场片段数据
        if self.market_segment not in environment:
            return {"error": f"Market segment {self.market_segment} not in environment"}
            
        segment_data = environment[self.market_segment]
        
        # 提取关注的特征
        perception = {}
        for feature in self.feature_focus:
            if feature in segment_data:
                perception[feature] = segment_data[feature]
                
        # 添加市场上下文信息
        if "global_market" in environment:
            perception["market_context"] = environment["global_market"]
            
        return {
            "relevant_data": perception,
            "timestamp": environment.get("timestamp", 0)
        }
        
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """基于感知做出决策"""
        if "error" in perception:
            return {"decision": "no_action", "confidence": 0.0, "reason": perception["error"]}
            
        if not perception.get("relevant_data"):
            return {"decision": "no_action", "confidence": 0.0, "reason": "No relevant data"}
            
        self.decision_count += 1
        
        # 检查是否有策略集可用
        if hasattr(self, "strategy_ensemble") and self.strategy_ensemble:
            try:
                # 使用策略集生成交易信号
                relevant_data = perception.get("relevant_data", {})
                signal = self.strategy_ensemble.generate_signal(relevant_data)
                
                # 增加交易信号的敏感度，使系统更倾向于产生买卖信号
                action = signal.get("action", "hold")
                confidence = signal.get("confidence", 0.0) * 1.8  # 大幅提高置信度
                
                # 如果是hold但接近决策边界，有时改为轻微买入或卖出以增加交易频率
                if action == "hold" and confidence > 0.3 and np.random.random() < 0.4:
                    # 根据之前的成功率决定倾向
                    if self.success_rate > 0.5 and self.last_decisions:
                        # 倾向于继续上一个成功的动作
                        action = self.last_decisions[-1].get("action", "hold")
                        if action == "hold":
                            action = "buy" if np.random.random() < 0.6 else "sell"
                    else:
                        action = "buy" if np.random.random() < 0.6 else "sell"
                    confidence = 0.3 + np.random.random() * 0.2
                
                decision = {
                    "action": action,
                    "confidence": min(confidence, 1.0),  # 确保置信度不超过1
                    "signal_source": "strategy_ensemble",
                    "timestamp": perception.get("timestamp", 0)
                }
                
                # 保存决策历史
                self.last_decisions.append(decision)
                if len(self.last_decisions) > 10:
                    self.last_decisions.pop(0)
                    
                self.prediction_history.append(decision["action"])
                self.confidence_history.append(decision["confidence"])
                
                return decision
                
            except Exception as e:
                logger.warning(f"使用策略集生成信号失败: {e}，回退到增强型决策模型")
        
        # 增强型决策逻辑
        relevant_data = perception.get("relevant_data", {})
        
        # 尝试从数据中提取有价值的信号
        decision = self._extract_decision_from_data(relevant_data)
        
        # 如果成功提取到决策，直接返回
        if decision and decision.get("action") != "hold" and decision.get("confidence", 0) > 0.3:
            # 保存决策历史
            self.last_decisions.append(decision)
            if len(self.last_decisions) > 10:
                self.last_decisions.pop(0)
                
            self.prediction_history.append(decision["action"])
            self.confidence_history.append(decision["confidence"])
            
            return decision
        
        # 否则使用改进的随机决策模型，但降低hold的概率
        signal = np.random.normal(0, 1.8)  # 增加标准差，使信号更分散
        
        # 根据智能体专注领域调整信号
        if "type" in self.specialization:
            if self.specialization["type"] == "trend" and "trend" in relevant_data:
                # 趋势型智能体 - 增强趋势信号
                trend = relevant_data.get("trend", 0)
                signal += trend * 2
            elif self.specialization["type"] == "reversal" and "rsi" in relevant_data:
                # 反转型智能体 - 过热/过冷反转信号
                rsi = relevant_data.get("rsi", 50)
                # RSI高则增加卖出倾向，RSI低则增加买入倾向
                signal -= (rsi - 50) / 25  # 归一化调整
            elif self.specialization["type"] == "volume" and "volume_change_pct" in relevant_data:
                # 交易量型智能体 - 根据交易量变化增强信号
                vol_change = relevant_data.get("volume_change_pct", 0)
                signal += vol_change * 3
        
        # 适应性偏移 - 基于过去性能调整
        if self.success_rate > 0.6:  # 如果智能体表现良好
            # 增加决策的连贯性
            if self.last_decisions and np.random.random() < 0.4:
                last_action = self.last_decisions[-1].get("action")
                if last_action == "buy":
                    signal += 0.5
                elif last_action == "sell":
                    signal -= 0.5
        
        # 计算信心
        confidence = min(abs(signal) / 2.5, 1.0)  # 优化信心计算
        
        # 降低hold的阈值范围，使系统更容易产生买卖信号
        if signal > 0.25:  # 降低买入阈值
            action = "buy"
        elif signal < -0.25:  # 降低卖出阈值
            action = "sell"
        else:
            action = "hold"
            confidence *= 0.5  # 降低hold决策的信心
            
        decision = {
            "action": action,
            "confidence": confidence,
            "signal_strength": signal,
            "signal_source": "enhanced_model",
            "timestamp": perception.get("timestamp", 0)
        }
        
        # 保存决策历史
        self.last_decisions.append(decision)
        if len(self.last_decisions) > 10:
            self.last_decisions.pop(0)
            
        self.prediction_history.append(decision["action"])
        self.confidence_history.append(decision["confidence"])
        
        return decision
        
    def _extract_decision_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从市场数据中提取决策信号"""
        try:
            # 检查是否有足够的数据进行分析
            if not data:
                return None
                
            # 初始化信号强度和信心
            signal = 0.0
            confidence = 0.0
            reasons = []
            
            # 移动平均线交叉信号
            if all(k in data for k in ["ma5", "ma10"]):
                ma5 = data["ma5"]
                ma10 = data["ma10"]
                
                if ma5 > ma10:
                    # 金叉，买入信号
                    signal += 1.0
                    confidence += 0.3
                    reasons.append("MA金叉")
                elif ma5 < ma10:
                    # 死叉，卖出信号
                    signal -= 1.0
                    confidence += 0.3
                    reasons.append("MA死叉")
                    
            # RSI超买超卖信号
            if "rsi" in data:
                rsi = data["rsi"]
                
                if rsi < 30:
                    # 超卖，买入信号
                    signal += 1.5
                    confidence += 0.4
                    reasons.append("RSI超卖")
                elif rsi > 70:
                    # 超买，卖出信号
                    signal -= 1.5
                    confidence += 0.4
                    reasons.append("RSI超买")
                    
            # 交易量异常信号
            if "volume_change_pct" in data:
                vol_change = data["volume_change_pct"]
                
                if vol_change > 0.5 and "price_change_pct" in data:
                    price_change = data["price_change_pct"]
                    if price_change > 0:
                        # 量价齐升，买入信号
                        signal += 1.2
                        confidence += 0.35
                        reasons.append("量价齐升")
                    elif price_change < 0:
                        # 量增价跌，卖出信号
                        signal -= 1.0
                        confidence += 0.3
                        reasons.append("量增价跌")
                        
            # 波动率信号
            if "volatility" in data:
                volatility = data["volatility"]
                avg_volatility = data.get("avg_volatility", volatility)
                
                if volatility > avg_volatility * 1.5:
                    # 波动率突然增加，可能是趋势变化的信号
                    # 方向由其他指标决定，这里只增加信心
                    confidence += 0.2
                    reasons.append("波动率增加")
                    
            # 趋势强度信号
            if "trend" in data:
                trend = data["trend"]
                trend_strength = data.get("trend_strength", abs(trend))
                
                if trend > 0 and trend_strength > 0.2:
                    # 上升趋势明显，买入信号
                    signal += 1.0 * trend_strength
                    confidence += 0.25 * trend_strength
                    reasons.append("上升趋势")
                elif trend < 0 and trend_strength > 0.2:
                    # 下降趋势明显，卖出信号
                    signal -= 1.0 * trend_strength
                    confidence += 0.25 * trend_strength
                    reasons.append("下降趋势")
                    
            # 合成最终决策
            # 根据信号强度确定动作
            if signal > 0.8:
                action = "buy"
            elif signal < -0.8:
                action = "sell"
            else:
                action = "hold"
                confidence = min(confidence, 0.5)  # hold决策的信心上限较低
                
            # 如果没有足够的信心，返回None
            if confidence < 0.3:
                return None
                
            # 构建决策结果
            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "signal_strength": signal,
                "reasons": reasons,
                "signal_source": "data_analysis"
            }
            
            return decision
        except Exception as e:
            logger.error(f"从数据提取决策失败: {e}")
            return None
        
    def adapt(self, feedback: Dict[str, Any]) -> None:
        """基于反馈调整自身"""
        if "performance" in feedback:
            self.performance_history.append(feedback["performance"])
            
            # 更新成功率
            if feedback["performance"] > 0:
                # 正向反馈
                self.success_rate = (self.success_rate * self.decision_count + 1) / (self.decision_count + 1)
            else:
                # 负向反馈
                self.success_rate = (self.success_rate * self.decision_count) / (self.decision_count + 1)
            
            # 根据反馈调整专注度
            if len(self.performance_history) > 5:
                recent_performance = np.mean(self.performance_history[-5:])
                
                # 如果表现不佳，尝试调整关注的特征
                if recent_performance < 0:
                    if np.random.random() < 0.4:  # 增加调整概率 
                        all_features = feedback.get("available_features", [])
                        if all_features:
                            # 随机替换一个特征
                            if self.feature_focus and np.random.random() < 0.5:
                                feature_to_replace = np.random.choice(self.feature_focus)
                                new_feature = np.random.choice(
                                    [f for f in all_features if f not in self.feature_focus]
                                )
                                self.feature_focus.remove(feature_to_replace)
                                self.feature_focus.append(new_feature)
                                logger.info(f"Agent {self.agent_id} replaced feature {feature_to_replace} with {new_feature}")


class MidAgent(Agent):
    """中层智能体 - 整合相关微观智能体的输出"""
    
    def __init__(self, domain: str, **kwargs):
        """
        初始化中层智能体
        
        Args:
            domain: 关注的领域
        """
        super().__init__(**kwargs)
        self.domain = domain
        self.micro_agents: List[MicroAgent] = []
        self.weights = {}  # 每个微观智能体的权重
        self.integration_strategy = "weighted_average"  # 整合策略
        
    def add_micro_agent(self, agent: MicroAgent, weight: float = 1.0) -> None:
        """添加微观智能体"""
        self.micro_agents.append(agent)
        self.weights[agent.agent_id] = weight
        
    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """获取所有微观智能体的决策"""
        micro_decisions = {}
        
        for agent in self.micro_agents:
            perception = agent.perceive(environment)
            decision = agent.decide(perception)
            micro_decisions[agent.agent_id] = {
                "decision": decision,
                "weight": self.weights[agent.agent_id],
                "specialization": agent.specialization
            }
            
        return {
            "micro_decisions": micro_decisions,
            "timestamp": environment.get("timestamp", 0),
            "domain": self.domain
        }
        
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """整合微观智能体的决策"""
        if not perception.get("micro_decisions"):
            return {"decision": "no_action", "confidence": 0.0, "reason": "No micro agent decisions"}
            
        micro_decisions = perception["micro_decisions"]
        
        # 提取所有行动决策和信心
        actions = []
        confidences = []
        weights = []
        
        for agent_id, data in micro_decisions.items():
            decision = data["decision"]
            weight = data["weight"]
            
            if decision["action"] == "buy":
                action_value = 1.0
            elif decision["action"] == "sell":
                action_value = -1.0
            else:  # hold
                action_value = 0.0
                
            actions.append(action_value)
            confidences.append(decision["confidence"])
            weights.append(weight)
            
        # 按照智能体权重计算加权结果
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化权重
        
        weighted_action = np.sum(np.array(actions) * weights * np.array(confidences))
        avg_confidence = np.average(confidences, weights=weights)
        
        # 转换为决策
        if weighted_action > 0.2:
            action = "buy"
        elif weighted_action < -0.2:
            action = "sell"
        else:
            action = "hold"
            
        # 计算决策的一致性
        action_values = np.array(actions)
        consistency = 1.0 - np.std(action_values) / 2.0  # 标准差越小，一致性越高
        
        return {
            "action": action,
            "confidence": avg_confidence * consistency,  # 信心与一致性相乘
            "weighted_signal": weighted_action,
            "consistency": consistency,
            "timestamp": perception.get("timestamp", 0)
        }
        
    def adapt(self, feedback: Dict[str, Any]) -> None:
        """调整对微观智能体的权重分配"""
        if "performance" in feedback:
            self.performance_history.append(feedback["performance"])
            
            # 传递反馈给各微观智能体
            for agent in self.micro_agents:
                # 为每个微观智能体创建特定的反馈
                agent_feedback = {
                    "performance": feedback.get("micro_performances", {}).get(agent.agent_id, feedback["performance"]),
                    "available_features": feedback.get("available_features", [])
                }
                agent.adapt(agent_feedback)
            
            # 更新微观智能体权重
            if len(self.performance_history) >= 10:
                # 根据最近表现动态调整权重
                for agent in self.micro_agents:
                    agent_fitness = agent.get_fitness()
                    # 动态调整权重，但保持一定的探索性
                    self.weights[agent.agent_id] = 0.2 + 0.8 * agent_fitness


class MetaAgent(Agent):
    """顶层决策智能体 - 整合中层智能体的输出并管理资源分配"""
    
    def __init__(self, name: str = "MainMetaAgent", **kwargs):
        """初始化顶层智能体"""
        super().__init__(agent_id=name, **kwargs)
        self.mid_agents: List[MidAgent] = []
        self.domain_weights = {}  # 各领域权重
        self.resource_allocation = {}  # 资源分配
        self.global_strategy = "balanced"  # 全局策略
        self.risk_tolerance = 0.6  # 提高风险容忍度(0-1)，增强交易信号
        self.performance_metrics = {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0
        }
        self.market_state_history = []  # 保存市场状态历史
        self.decision_history = []  # 保存决策历史
        self.momentum_factor = 0.4  # 动量因子，用于持续趋势交易
        self.contrarian_factor = 0.3  # 反向因子，用于过度行情的反向交易
        self.adaptive_mode = True  # 是否启用自适应模式
        
    def add_mid_agent(self, agent: MidAgent, weight: float = 1.0) -> None:
        """添加中层智能体"""
        self.mid_agents.append(agent)
        self.domain_weights[agent.domain] = weight
        self.resource_allocation[agent.domain] = 1.0 / (len(self.mid_agents) if len(self.mid_agents) > 0 else 1)
        
    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """获取所有中层智能体的决策"""
        mid_decisions = {}
        
        for agent in self.mid_agents:
            perception = agent.perceive(environment)
            decision = agent.decide(perception)
            mid_decisions[agent.domain] = {
                "decision": decision,
                "weight": self.domain_weights[agent.domain],
                "resource_allocation": self.resource_allocation[agent.domain]
            }
            
        # 添加全局市场情况
        market_state = environment.get("global_market", {})
        
        # 保存市场状态历史
        if market_state:
            self.market_state_history.append(market_state)
            if len(self.market_state_history) > 30:  # 保留30天历史
                self.market_state_history.pop(0)
        
        return {
            "mid_decisions": mid_decisions,
            "market_state": market_state,
            "timestamp": environment.get("timestamp", 0)
        }
        
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """整合中层智能体决策，制定全局策略"""
        if not perception.get("mid_decisions"):
            return {"decision": "no_action", "confidence": 0.0, "reason": "No mid agent decisions"}
            
        mid_decisions = perception["mid_decisions"]
        market_state = perception.get("market_state", {})
        
        # 提取各领域决策
        domain_actions = {}
        domain_confidences = {}
        domain_signals = {}
        
        for domain, data in mid_decisions.items():
            decision = data["decision"]
            domain_actions[domain] = decision["action"]
            domain_confidences[domain] = decision["confidence"]
            # 提取信号强度（如果有）
            domain_signals[domain] = decision.get("weighted_signal", 0)
        
        # 市场状态分析
        market_context = self._analyze_market_context(market_state)
        
        # 判断市场状态类型
        market_type = self._determine_market_type(market_context)
        
        # 确定全局风险水平
        market_volatility = market_state.get("volatility", 0.5)
        # 根据市场类型调整风险容忍度
        if market_type == "trending":
            adjusted_risk_tolerance = self.risk_tolerance * 1.2  # 趋势市场加大风险
        elif market_type == "ranging":
            adjusted_risk_tolerance = self.risk_tolerance * 0.9  # 盘整市场降低风险
        elif market_type == "volatile":
            adjusted_risk_tolerance = self.risk_tolerance * 0.8  # 剧烈波动市场降低风险
        else:  # normal
            adjusted_risk_tolerance = self.risk_tolerance
            
        # 再根据波动率调整
        adjusted_risk_tolerance *= (1 - market_volatility * 0.5)
        
        # 计算整体仓位
        position_sizing = {}
        total_allocation = 0.0
        
        for domain, action in domain_actions.items():
            weight = self.domain_weights[domain]
            confidence = domain_confidences[domain]
            allocation = self.resource_allocation[domain]
            signal = domain_signals.get(domain, 0)
            
            # 仓位大小基于动作、信心、风险容忍度和资源分配
            if action == "buy":
                # 增加信号强度系数，使买入更积极
                size = allocation * confidence * adjusted_risk_tolerance * 1.2
                # 如果是趋势市场，进一步增强信号
                if market_type == "trending" and signal > 0:
                    size *= 1.3
            elif action == "sell":
                size = -allocation * confidence * adjusted_risk_tolerance * 1.2
                # 如果是趋势市场，进一步增强信号
                if market_type == "trending" and signal < 0:
                    size *= 1.3
            else:  # hold
                # hold也可能有轻微偏向，基于信号强度
                size = allocation * signal * 0.2 * adjusted_risk_tolerance
                
            position_sizing[domain] = size
            total_allocation += abs(size)
            
        # 归一化分配，确保总分配不超过1
        if total_allocation > 1.0:
            for domain in position_sizing:
                position_sizing[domain] /= total_allocation
        
        # 考虑决策一致性
        action_consistency = self._calculate_action_consistency(domain_actions)
        
        # 确定整体行动方向
        net_position = sum(position_sizing.values())
        
        # 应用动量和反向因子
        net_position = self._apply_momentum_and_contrarian(net_position, action_consistency, market_type)
        
        # 确定最终动作，降低hold阈值使系统更倾向产生交易信号
        if net_position > 0.08:  # 降低买入阈值
            overall_action = "buy"
        elif net_position < -0.08:  # 降低卖出阈值
            overall_action = "sell"
        else:
            overall_action = "hold"
            
        # 计算整体信心
        weighted_confidence = sum(
            domain_confidences[d] * self.domain_weights[d] * (2 if domain_actions[d] == overall_action else 1)
            for d in domain_confidences
        ) / (sum(self.domain_weights.values()) * 1.5)
        
        # 应用一致性增强
        if action_consistency > 0.7:  # 如果各智能体决策高度一致
            weighted_confidence = min(1.0, weighted_confidence * 1.3)
        
        # 构建最终决策
        decision = {
            "action": overall_action,
            "confidence": weighted_confidence,
            "position_sizing": position_sizing,
            "net_position": net_position,
            "risk_level": adjusted_risk_tolerance,
            "market_type": market_type,
            "action_consistency": action_consistency,
            "timestamp": perception.get("timestamp", 0)
        }
        
        # 保存决策历史
        self.decision_history.append(decision)
        if len(self.decision_history) > 20:
            self.decision_history.pop(0)
            
        return decision
        
    def _analyze_market_context(self, market_state):
        """分析市场上下文信息"""
        context = {
            "volatility": market_state.get("volatility", 0.5),
            "trend_strength": market_state.get("trend_strength", 0),
            "is_trending": False,
            "is_volatile": False,
            "is_ranging": False,
            "market_type": "normal"
        }
        
        # 检查是否有趋势
        if abs(market_state.get("trend", 0)) > 0.02:
            context["is_trending"] = True
            context["trend_direction"] = "up" if market_state.get("trend", 0) > 0 else "down"
            
        # 检查波动性
        if market_state.get("volatility", 0) > 0.8:
            context["is_volatile"] = True
            
        # 检查是否是盘整市场
        if (abs(market_state.get("trend", 0)) < 0.01 and 
            market_state.get("volatility", 0) < 0.4):
            context["is_ranging"] = True
            
        return context
        
    def _determine_market_type(self, market_context):
        """确定市场类型"""
        if market_context["is_trending"]:
            return "trending"
        elif market_context["is_volatile"]:
            return "volatile"
        elif market_context["is_ranging"]:
            return "ranging"
        else:
            return "normal"
            
    def _calculate_action_consistency(self, domain_actions):
        """计算中层智能体决策的一致性"""
        if not domain_actions:
            return 0.5
            
        actions_count = {"buy": 0, "sell": 0, "hold": 0}
        
        for action in domain_actions.values():
            if action in actions_count:
                actions_count[action] += 1
                
        total = sum(actions_count.values())
        
        if total == 0:
            return 0.5
            
        # 计算最常见行动的占比
        max_action_ratio = max(actions_count.values()) / total
        
        return max_action_ratio
        
    def _apply_momentum_and_contrarian(self, net_position, consistency, market_type):
        """应用动量和反向因子"""
        if not self.decision_history:
            return net_position
            
        # 计算最近决策的平均方向
        recent_decisions = self.decision_history[-5:] if len(self.decision_history) >= 5 else self.decision_history
        
        recent_positions = [d["net_position"] for d in recent_decisions]
        avg_position = sum(recent_positions) / len(recent_positions)
        
        # 如果当前信号与近期趋势一致，应用动量因子
        if (net_position > 0 and avg_position > 0) or (net_position < 0 and avg_position < 0):
            # 在趋势市场中增强动量效应
            momentum_multiplier = 1.5 if market_type == "trending" else 1.0
            net_position += avg_position * self.momentum_factor * momentum_multiplier
            
        # 如果信号很强但与一致性低，可能是过度行情，应用反向因子
        if abs(net_position) > 0.7 and consistency < 0.5:
            net_position -= (net_position * self.contrarian_factor)
            
        # 确保不超过合理范围
        return max(-1.0, min(1.0, net_position))
        
    def adapt(self, feedback: Dict[str, Any]) -> None:
        """调整策略和资源分配"""
        if "performance" in feedback:
            self.performance_history.append(feedback["performance"])
            
            # 更新全局性能指标
            for metric, value in feedback.get("metrics", {}).items():
                self.performance_metrics[metric] = value
            
            # 自适应调整风险参数
            if self.adaptive_mode and len(self.performance_history) >= 10:
                self._adapt_risk_parameters()
            
            # 为每个中层智能体创建反馈
            domain_performances = feedback.get("domain_performances", {})
            for agent in self.mid_agents:
                domain = agent.domain
                # 传递领域特定表现或整体表现
                domain_feedback = {
                    "performance": domain_performances.get(domain, feedback["performance"]),
                    "micro_performances": feedback.get("micro_performances", {}),
                    "available_features": feedback.get("available_features", [])
                }
                agent.adapt(domain_feedback)
            
            # 动态调整资源分配
            if len(self.performance_history) >= 20:
                self._rebalance_resources(domain_performances)
                
    def _adapt_risk_parameters(self):
        """自适应调整风险参数"""
        try:
            # 获取最近的性能
            recent_performance = self.performance_history[-10:]
            win_count = sum(1 for p in recent_performance if p > 0)
            win_rate = win_count / len(recent_performance)
            
            # 根据胜率调整风险容忍度
            if win_rate > 0.7:
                # 高胜率，可以增加风险
                self.risk_tolerance = min(0.8, self.risk_tolerance + 0.02)
                self.momentum_factor = min(0.6, self.momentum_factor + 0.01)
            elif win_rate < 0.4:
                # 低胜率，降低风险
                self.risk_tolerance = max(0.3, self.risk_tolerance - 0.02)
                self.momentum_factor = max(0.2, self.momentum_factor - 0.01)
                # 低胜率时增加反向因子
                self.contrarian_factor = min(0.5, self.contrarian_factor + 0.02)
            else:
                # 中等胜率，微调参数
                if sum(recent_performance) > 0:
                    # 总体正收益，略微增加风险
                    self.risk_tolerance = min(0.7, self.risk_tolerance + 0.01)
                else:
                    # 总体负收益，略微降低风险
                    self.risk_tolerance = max(0.4, self.risk_tolerance - 0.01)
                    
            logger.info(f"调整风险参数: risk_tolerance={self.risk_tolerance:.2f}, "
                        f"momentum_factor={self.momentum_factor:.2f}, "
                        f"contrarian_factor={self.contrarian_factor:.2f}")
                        
        except Exception as e:
            logger.error(f"自适应调整风险参数失败: {e}")
                
    def _rebalance_resources(self, domain_performances: Dict[str, float]) -> None:
        """重新分配资源"""
        # 计算每个领域的相对表现
        if not domain_performances:
            return
            
        total_performance = sum(max(0.1, perf) for perf in domain_performances.values())
        
        # 基于表现分配75%的资源，25%平均分配以保持探索
        base_allocation = 0.25 / len(self.mid_agents)
        
        for agent in self.mid_agents:
            domain = agent.domain
            if domain in domain_performances:
                perf = max(0.1, domain_performances[domain])  # 确保性能值为正
                performance_based_allocation = 0.75 * (perf / total_performance)
                self.resource_allocation[domain] = base_allocation + performance_based_allocation
                
                # 同时更新领域权重，给表现好的领域更高权重
                self.domain_weights[domain] = 0.2 + 0.8 * (perf / total_performance)
                
                logger.info(f"重新分配资源: {domain}={self.resource_allocation[domain]:.2f}, "
                           f"权重={self.domain_weights[domain]:.2f}")
                           
        # 确保资源分配的总和为1
        total_allocation = sum(self.resource_allocation.values())
        if total_allocation != 1.0:
            scale_factor = 1.0 / total_allocation
            for domain in self.resource_allocation:
                self.resource_allocation[domain] *= scale_factor


class FractalIntelligenceNetwork:
    """分形智能网络 - 管理智能体网络的生成和进化"""
    
    def __init__(self, config=None):
        """初始化分形智能网络
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 设置网络基础参数
        self.current_time = 0
        self.domains = self._define_domains()
        
        # 初始化三层智能体
        self.micro_agents = []
        self.mid_agents = []
        self.macro_agent = None
        
        # 网络进化参数
        self.evolution_generation = 0
        self.evolution_rate = 0.05  # 每步有5%的概率进行网络进化
        self.max_agents = 100  # 最大智能体数量
        
        # 智能体性能统计
        self.agents_performance = {}
        
        # 可用智能体类型
        self.available_agent_types = ['trend', 'reversal', 'volume', 'volatility', 'pattern']
        
        # 初始化基础智能体
        self._create_default_strategies()
        
        # 创建主元智能体
        self.main_meta_agent = self._create_meta_agent()
        
        logger.info("Fractal Intelligence Network initialized")
        
    def _define_domains(self):
        """定义分形网络的领域和市场分段
        
        Returns:
            dict: 领域字典，key为领域名称，value为该领域包含的市场分段列表
        """
        default_domains = {
            'stock': ['000001', '000002', '000003', '000004', '000005'],
            'index': ['000001.SH', '399001.SZ', '399006.SZ'],
            'sector': ['finance', 'technology', 'consumer', 'healthcare', 'energy']
        }
        
        # 如果配置中有自定义领域，使用配置中的领域
        if self.config and 'domains' in self.config:
            return self.config['domains']
        
        return default_domains
        
    def create_initial_network(self, market_segments: List[str] = None, features: Dict[str, List[str]] = None) -> None:
        """
        创建初始智能体网络
        
        Args:
            market_segments: 市场分段列表
            features: 每个分段的特征
        """
        # 如果没有提供市场分段，使用默认的单一分段
        if not market_segments:
            market_segments = ["default"]
            
        # 如果没有提供特征，使用默认的特征集
        default_features = ["price", "volume", "trend"]
        if not features:
            features = {segment: default_features.copy() for segment in market_segments}
            
        # 获取策略集（如果配置中有）
        strategy_ensemble = None
        if self.config.get("use_strategy_ensemble", True):
            try:
                from quantum_symbiotic_network.strategies import create_default_strategy_ensemble
                strategy_ensemble = create_default_strategy_ensemble()
                logger.info("已创建默认策略集")
            except ImportError:
                logger.warning("无法导入策略集，将使用随机决策模型")
                
        # 为每个市场分段创建多个微观智能体
        micro_agents = []
        
        for segment in market_segments:
            segment_features = features.get(segment, default_features)
            
            # 为每个分段创建几个具有不同专注点的智能体
            for _ in range(self.config.get("micro_agents_per_segment", 5)):
                # 随机选择一部分特征作为该智能体的专注点
                focus_count = max(2, int(len(segment_features) * 0.7))
                feature_focus = np.random.choice(
                    segment_features, 
                    size=min(focus_count, len(segment_features)), 
                    replace=False
                ).tolist()
                
                # 创建微观智能体
                agent = MicroAgent(
                    market_segment=segment,
                    feature_focus=feature_focus,
                    specialization={"focus": feature_focus}
                )
                
                # 如果有策略集，添加到智能体
                if strategy_ensemble:
                    agent.strategy_ensemble = strategy_ensemble
                    
                micro_agents.append(agent)
                
        # 将市场分段分组到不同领域
        domains = self._group_segments_into_domains(market_segments)
        
        # 创建中层智能体
        mid_agents = []
        for domain_name, domain_segments in domains.items():
            mid_agent = MidAgent(domain=domain_name)
            
            # 将相关的微观智能体连接到该中层智能体
            for micro_agent in micro_agents:
                if micro_agent.market_segment in domain_segments:
                    # 随机初始化权重
                    weight = np.random.uniform(0.5, 1.5)
                    mid_agent.add_micro_agent(micro_agent, weight)
                    
            if mid_agent.micro_agents:  # 只有连接了微观智能体的中层才添加
                mid_agents.append(mid_agent)
                
        # 创建元智能体
        self.main_meta_agent = MetaAgent()
        
        # 将所有中层智能体连接到元智能体
        for mid_agent in mid_agents:
            # 随机初始化权重
            weight = np.random.uniform(0.7, 1.3)
            self.main_meta_agent.add_mid_agent(mid_agent, weight)
            
        self.micro_agents = micro_agents
        self.mid_agents = mid_agents
        
        logger.info(f"Created initial network with {len(micro_agents)} micro agents, {len(mid_agents)} mid agents")
        
    def _group_segments_into_domains(self, segments: List[str]) -> Dict[str, List[str]]:
        """将市场分段分组为领域"""
        # 简单实现 - 可以替换为更复杂的聚类算法
        domains = {}
        
        # 基于某种规则分组，这里使用前缀作为示例
        for segment in segments:
            prefix = segment[:3]  # 根据前缀分组
            if prefix not in domains:
                domains[prefix] = []
            domains[prefix].append(segment)
            
        return domains
        
    def step(self, market_data, feedback=None):
        """处理单个时间步，生成交易决策
        
        Args:
            market_data: 市场数据
            feedback: 上一步的反馈
            
        Returns:
            dict: 包含交易决策的字典
        """
        try:
            # 默认决策，确保总是返回包含action的决策
            default_decision = {
                "action": "hold",
                "confidence": 1.0,
                "timestamp": int(time.time()),
                "source": "fractal_network_default"
            }
            
            if not market_data:
                logger.warning("没有市场数据提供给分形网络")
                return default_decision
            
            # 提取市场特征
            market_features = self._extract_market_features(market_data)
            
            # 提供反馈给智能体网络
            if feedback:
                self._provide_feedback(feedback)
            
            # 有机会进行网络进化
            if np.random.random() < self.evolution_rate:
                self._evolve_network()
            
            # 聚合智能体决策
            micro_decisions = []
            
            # 收集所有微观智能体的决策
            for agent in self.micro_agents:
                try:
                    agent_decision = agent.decide(market_features)
                    if agent_decision and "action" in agent_decision:
                        micro_decisions.append(agent_decision)
                except Exception as e:
                    logger.error(f"智能体决策生成失败: {e}")
            
            # 如果没有有效决策，返回默认决策
            if not micro_decisions:
                logger.warning("没有微观智能体产生有效决策")
                return default_decision
            
            # 宏观智能体生成汇总决策
            try:
                if self.macro_agent:
                    macro_decision = self.macro_agent.synthesize(micro_decisions, market_features)
                    if macro_decision and "action" in macro_decision:
                        # 添加时间戳和来源
                        macro_decision["timestamp"] = int(time.time())
                        macro_decision["source"] = "fractal_network"
                        return macro_decision
            except Exception as e:
                logger.error(f"宏观智能体决策合成失败: {e}")
            
            # 如果宏观决策失败，使用多数投票
            actions = [d.get("action") for d in micro_decisions if d.get("action")]
            if not actions:
                return default_decision
                
            # 统计各动作的票数
            from collections import Counter
            action_counts = Counter(actions)
            most_common_action = action_counts.most_common(1)[0][0]
            
            # 计算信心度
            confidence = action_counts[most_common_action] / len(actions)
            
            # 构建最终决策
            decision = {
                "action": most_common_action,
                "confidence": confidence,
                "vote_count": dict(action_counts),
                "timestamp": int(time.time()),
                "source": "fractal_network_voting"
            }
            
            return decision
        
        except Exception as e:
            logger.error(f"分形网络决策失败: {e}")
            return {
                "action": "hold",
                "confidence": 1.0,
                "error": str(e),
                "timestamp": int(time.time()),
                "source": "fractal_network_error"
            }
        
    def _preprocess_environment(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """预处理环境数据，确保格式正确
        
        Args:
            environment: 原始环境数据
            
        Returns:
            处理后的环境数据
        """
        processed_env = {"timestamp": environment.get("timestamp", self.current_time)}
        
        # 提取市场数据
        if "stocks" in environment:
            # 直接使用stocks字典作为市场分段
            for stock_code, stock_data in environment["stocks"].items():
                processed_env[stock_code] = stock_data
                
            # 添加全局市场指标
            if "indices" in environment:
                processed_env["global_market"] = environment["indices"]
        else:
            # 环境可能已经是处理过的格式
            processed_env = environment
            
        return processed_env
        
    def provide_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        向网络提供反馈
        
        Args:
            feedback: 包含性能评估的反馈
        """
        # 添加时间戳
        feedback["timestamp"] = self.current_time
        
        # 传递给主元智能体，它会递归传递给下层
        self.main_meta_agent.adapt(feedback)
        
    def _evolve_network(self):
        """进化网络结构和参数，探索更优性能
        网络会根据历史表现自动调整智能体组合和决策权重
        """
        try:
            # 确保micro_agents属性存在且有值
            if not hasattr(self, 'micro_agents') or self.micro_agents is None:
                logger.warning("网络无法进化：micro_agents属性未初始化")
                self.micro_agents = []
            
            if len(self.micro_agents) == 0:
                # 初始化基础智能体
                self._initialize_agents()
                logger.info("已初始化基础智能体网络")
                return
            
            logger.info("开始网络进化过程...")
            
            # 如果存在性能记录，使用它进行进化
            if hasattr(self, 'agents_performance') and self.agents_performance:
                # 按平均回报对智能体排序
                sorted_performance = sorted(
                    self.agents_performance.items(),
                    key=lambda x: x[1]['average_reward'],
                    reverse=True
                )
                
                # 保留表现最好的智能体
                top_agents_ids = [agent_id for agent_id, _ in sorted_performance[:int(len(self.micro_agents) * 0.7)]]
                
                # 识别和删除表现最差的智能体
                bottom_agents = [
                    agent for agent in self.micro_agents 
                    if (hasattr(agent, 'id') and agent.id not in top_agents_ids)
                ]
                
                for agent in bottom_agents:
                    if np.random.random() < 0.3:  # 有30%的概率删除表现不佳的智能体
                        if agent in self.micro_agents:
                            self.micro_agents.remove(agent)
                            logger.info(f"移除了表现不佳的智能体 {agent.id if hasattr(agent, 'id') else id(agent)}")
                
                # 复制表现最好的智能体并做变异
                for agent_id in top_agents_ids[:3]:  # 只复制前3名
                    for agent in self.micro_agents:
                        if hasattr(agent, 'id') and agent.id == agent_id:
                            # 如果总智能体数量过多，则不再添加
                            if len(self.micro_agents) >= self.max_agents:
                                break
                            
                            # 复制并变异
                            try:
                                import copy
                                new_agent = copy.deepcopy(agent)
                                
                                # 确保新智能体有唯一ID
                                if hasattr(new_agent, 'id'):
                                    new_agent.id = f"{new_agent.id}_mutated_{int(time.time())}"
                                    
                                # 变异智能体参数
                                if hasattr(new_agent, 'mutate'):
                                    new_agent.mutate(mutation_rate=0.2)
                                    
                                self.micro_agents.append(new_agent)
                                logger.info(f"创建了变异智能体 {new_agent.id if hasattr(new_agent, 'id') else id(new_agent)}")
                            except Exception as e:
                                logger.error(f"智能体复制变异失败: {e}")
                            
            # 随机生成一些全新的智能体，增加多样性
            new_agents_count = max(1, int(len(self.micro_agents) * 0.1))  # 添加10%的新智能体
            for _ in range(new_agents_count):
                if len(self.micro_agents) < self.max_agents:
                    try:
                        # 随机选择一种智能体类型
                        agent_type = np.random.choice(self.available_agent_types)
                        new_agent = self._create_agent(agent_type)
                        self.micro_agents.append(new_agent)
                        logger.info(f"创建了新智能体 {new_agent.id if hasattr(new_agent, 'id') else id(new_agent)}")
                    except Exception as e:
                        logger.error(f"创建新智能体失败: {e}")
            
            # 更新宏观智能体
            if hasattr(self, 'macro_agent') and self.macro_agent and hasattr(self.macro_agent, 'update'):
                try:
                    self.macro_agent.update(self.micro_agents, self.agents_performance)
                except Exception as e:
                    logger.error(f"更新宏观智能体失败: {e}")
                
            # 记录当前网络状态
            logger.info(f"网络进化完成。当前有 {len(self.micro_agents)} 个微观智能体")
            
            # 重置部分性能统计，让新智能体有公平的机会
            if np.random.random() < 0.2:  # 20%的概率重置统计
                for agent_id in self.agents_performance:
                    self.agents_performance[agent_id]['decisions_count'] = max(1, self.agents_performance[agent_id]['decisions_count'] // 2)
                
        except Exception as e:
            logger.error(f"网络进化失败: {e}")
            # 确保错误不会导致网络崩溃，至少保留一些基础智能体
            if not hasattr(self, 'micro_agents') or not self.micro_agents:
                self._initialize_agents()
                logger.info("进化失败后重新初始化基础智能体网络")

    def _extract_market_features(self, market_data):
        """从市场数据中提取特征
        
        Args:
            market_data: 市场数据
            
        Returns:
            dict: 提取的市场特征
        """
        features = {}
        
        try:
            # 确保市场数据存在
            if not market_data:
                return features
            
            # 提取股票价格和交易量信息
            if isinstance(market_data, dict):
                # 如果是字典格式
                if "prices" in market_data:
                    features["price"] = market_data["prices"]
                    
                    # 计算简单的价格变化
                    if isinstance(features["price"], list) and len(features["price"]) > 1:
                        features["price_change"] = features["price"][-1] - features["price"][-2]
                        features["price_change_pct"] = features["price_change"] / features["price"][-2] if features["price"][-2] != 0 else 0
                
                if "volumes" in market_data:
                    features["volume"] = market_data["volumes"]
                    
                    # 计算交易量变化
                    if isinstance(features["volume"], list) and len(features["volume"]) > 1:
                        features["volume_change"] = features["volume"][-1] - features["volume"][-2]
                        features["volume_change_pct"] = features["volume_change"] / features["volume"][-2] if features["volume"][-2] != 0 else 0
                
                # 提取市场指标
                if "indicators" in market_data:
                    features["indicators"] = market_data["indicators"]
                    
                # 提取时间戳
                if "timestamp" in market_data:
                    features["timestamp"] = market_data["timestamp"]
                    
            # 添加简单的移动平均线计算
            if "price" in features and isinstance(features["price"], list):
                prices = features["price"]
                
                # 计算短期移动平均线 (5周期)
                if len(prices) >= 5:
                    features["ma5"] = sum(prices[-5:]) / 5
                    
                # 计算中期移动平均线 (10周期)
                if len(prices) >= 10:
                    features["ma10"] = sum(prices[-10:]) / 10
                    
                # 计算长期移动平均线 (20周期)
                if len(prices) >= 20:
                    features["ma20"] = sum(prices[-20:]) / 20
                    
                # 计算移动平均线交叉信号
                if "ma5" in features and "ma10" in features:
                    # 当前交叉状态
                    features["ma_cross"] = "above" if features["ma5"] > features["ma10"] else "below"
                    
                    # 如果有足够的历史数据，检测交叉发生
                    if len(prices) > 11:
                        prev_ma5 = sum(prices[-6:-1]) / 5
                        prev_ma10 = sum(prices[-11:-1]) / 10
                        prev_cross = "above" if prev_ma5 > prev_ma10 else "below"
                        
                        if prev_cross != features["ma_cross"]:
                            features["ma_cross_event"] = f"{prev_cross}_to_{features['ma_cross']}"
            
            # 计算简单的波动率 (过去5个周期的标准差)
            if "price" in features and isinstance(features["price"], list) and len(features["price"]) >= 5:
                import numpy as np
                features["volatility"] = np.std(features["price"][-5:])
                
            # 简单的趋势检测
            if "price" in features and isinstance(features["price"], list) and len(features["price"]) >= 10:
                # 计算过去10个周期的简单线性回归斜率
                import numpy as np
                y = np.array(features["price"][-10:])
                x = np.arange(10)
                slope = np.polyfit(x, y, 1)[0]
                features["trend"] = slope
                features["trend_strength"] = abs(slope) / (features["price"][-1] / 100) if features["price"][-1] != 0 else 0
                
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
        
        return features
        
    def _provide_feedback(self, feedback):
        """向智能体网络提供反馈
        
        Args:
            feedback: 反馈信息
        """
        try:
            if not feedback:
                return
            
            # 将反馈提供给所有微观智能体
            for agent in self.micro_agents:
                try:
                    if hasattr(agent, 'receive_feedback'):
                        agent.receive_feedback(feedback)
                except Exception as e:
                    logger.error(f"向智能体提供反馈失败: {e}")
                
            # 将反馈提供给宏观智能体
            if self.macro_agent and hasattr(self.macro_agent, 'receive_feedback'):
                try:
                    self.macro_agent.receive_feedback(feedback)
                except Exception as e:
                    logger.error(f"向宏观智能体提供反馈失败: {e}")
                
            # 更新每个智能体的表现统计
            if "reward" in feedback:
                reward = feedback["reward"]
                
                # 当前决策的智能体ID
                agent_id = feedback.get("agent_id", None)
                
                if agent_id:
                    # 更新特定智能体的表现
                    if agent_id in self.agents_performance:
                        perf = self.agents_performance[agent_id]
                        perf["total_reward"] += reward
                        perf["decisions_count"] += 1
                        perf["average_reward"] = perf["total_reward"] / perf["decisions_count"]
                        
                        # 更新最高和最低奖励
                        if reward > perf["highest_reward"]:
                            perf["highest_reward"] = reward
                        if reward < perf["lowest_reward"]:
                            perf["lowest_reward"] = reward
                            
                        # 记录成功失败次数
                        if reward > 0:
                            perf["success_count"] += 1
                        elif reward < 0:
                            perf["failure_count"] += 1
                    else:
                        # 初始化新的表现记录
                        self.agents_performance[agent_id] = {
                            "total_reward": reward,
                            "decisions_count": 1,
                            "average_reward": reward,
                            "highest_reward": reward,
                            "lowest_reward": reward,
                            "success_count": 1 if reward > 0 else 0,
                            "failure_count": 1 if reward < 0 else 0
                        }
                else:
                    # 如果没有特定的智能体ID，更新所有参与决策的智能体
                    for agent in self.micro_agents:
                        agent_id = agent.id if hasattr(agent, 'id') else str(id(agent))
                        
                        if agent_id in self.agents_performance:
                            perf = self.agents_performance[agent_id]
                            perf["total_reward"] += reward
                            perf["decisions_count"] += 1
                            perf["average_reward"] = perf["total_reward"] / perf["decisions_count"]
                        else:
                            # 初始化新的表现记录
                            self.agents_performance[agent_id] = {
                                "total_reward": reward,
                                "decisions_count": 1,
                                "average_reward": reward,
                                "highest_reward": reward,
                                "lowest_reward": reward,
                                "success_count": 1 if reward > 0 else 0,
                                "failure_count": 1 if reward < 0 else 0
                            }
                            
        except Exception as e:
            logger.error(f"处理反馈失败: {e}") 

    def _initialize_agents(self):
        """初始化基础智能体网络"""
        try:
            # 确保micro_agents属性存在
            if not hasattr(self, 'micro_agents'):
                self.micro_agents = []
            
            # 确保agents_performance存在
            if not hasattr(self, 'agents_performance'):
                self.agents_performance = {}
            
            # 确保定义了最大智能体数量
            if not hasattr(self, 'max_agents'):
                self.max_agents = 100
            
            # 确保定义了可用智能体类型
            if not hasattr(self, 'available_agent_types'):
                self.available_agent_types = ['trend', 'reversal', 'volume', 'volatility', 'pattern']
            
            # 创建初始智能体集合
            initial_count = 10  # 初始智能体数量
            for i in range(initial_count):
                # 随机选择一种智能体类型
                agent_type = self.available_agent_types[i % len(self.available_agent_types)]
                new_agent = self._create_agent(agent_type)
                self.micro_agents.append(new_agent)
            
            # 创建宏观智能体（如果未定义）
            if not hasattr(self, 'macro_agent') or self.macro_agent is None:
                self.macro_agent = self._create_macro_agent()
            
            logger.info(f"初始化了 {len(self.micro_agents)} 个微观智能体和 1 个宏观智能体")
        except Exception as e:
            logger.error(f"初始化智能体失败: {e}")
            # 确保至少有一些基础智能体
            if not hasattr(self, 'micro_agents') or not self.micro_agents:
                self.micro_agents = [
                    MicroAgent(
                        market_segment="default",
                        feature_focus=["price", "trend"], 
                        specialization={"type": "trend"},
                        agent_id="basic_trend_agent"
                    ),
                    MicroAgent(
                        market_segment="default",
                        feature_focus=["price", "momentum"], 
                        specialization={"type": "reversal"},
                        agent_id="basic_reversal_agent"
                    ),
                    MicroAgent(
                        market_segment="default",
                        feature_focus=["price", "volume"], 
                        specialization={"type": "volume"},
                        agent_id="basic_volume_agent"
                    )
                ]
        
    def _create_agent(self, agent_type):
        """创建指定类型的智能体
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            MicroAgent: 创建的微观智能体实例
        """
        try:
            agent_id = f"{agent_type}_{int(time.time() * 1000)}"
            
            # 基于类型选择特征关注点
            if agent_type == 'trend':
                feature_focus = ["price", "ma", "trend"]
                specialization = {"type": "trend", "focus": feature_focus}
                market_segment = np.random.choice(list(self.domains.keys()))
            elif agent_type == 'reversal':
                feature_focus = ["price", "rsi", "momentum"]
                specialization = {"type": "reversal", "focus": feature_focus}
                market_segment = np.random.choice(list(self.domains.keys()))
            elif agent_type == 'volume':
                feature_focus = ["volume", "price", "liquidity"]
                specialization = {"type": "volume", "focus": feature_focus}
                market_segment = np.random.choice(list(self.domains.keys()))
            elif agent_type == 'volatility':
                feature_focus = ["volatility", "atr", "price_range"]
                specialization = {"type": "volatility", "focus": feature_focus}
                market_segment = np.random.choice(list(self.domains.keys()))
            elif agent_type == 'pattern':
                feature_focus = ["pattern", "price", "formation"]
                specialization = {"type": "pattern", "focus": feature_focus}
                market_segment = np.random.choice(list(self.domains.keys()))
            else:
                feature_focus = ["price", "volume"]
                specialization = {"type": "basic", "focus": feature_focus}
                market_segment = np.random.choice(list(self.domains.keys()))
            
            # 创建MicroAgent实例
            agent = MicroAgent(
                market_segment=market_segment,
                feature_focus=feature_focus,
                specialization=specialization,
                agent_id=agent_id
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"创建智能体失败: {e}")
            # 返回基础智能体
            return MicroAgent(
                market_segment="default",
                feature_focus=["price", "volume"],
                specialization={"type": "basic"},
                agent_id=f"basic_{int(time.time() * 1000)}"
            )

    def _create_macro_agent(self):
        """创建宏观决策智能体
        
        Returns:
            dict: 宏观智能体对象
        """
        macro_agent = {
            "id": f"macro_{int(time.time())}",
            "type": "macro",
            "created_at": int(time.time()),
            "weights": {},  # 各微观智能体的权重
            "synthesize": self._synthesize_decisions,
            "update": self._update_macro_agent
        }
        
        return macro_agent
        
    def _mutate_agent(self, agent, mutation_rate=0.2):
        """变异智能体参数
        
        Args:
            agent: 要变异的智能体
            mutation_rate: 变异率
        """
        # 只有参数才会变异
        if "parameters" not in agent:
            return
        
        for param_name, param_value in agent["parameters"].items():
            # 以mutation_rate的概率变异每个参数
            if np.random.random() < mutation_rate:
                if isinstance(param_value, (int, np.integer)):
                    # 整数参数变异
                    change = np.random.randint(-max(1, int(param_value * 0.2)), 
                                              max(1, int(param_value * 0.2)) + 1)
                    agent["parameters"][param_name] = max(1, param_value + change)
                elif isinstance(param_value, (float, np.float)):
                    # 浮点参数变异
                    change = np.random.uniform(-param_value * 0.2, param_value * 0.2)
                    agent["parameters"][param_name] = max(0.001, param_value + change)
                elif isinstance(param_value, str):
                    # 字符串参数（如模式类型）变异
                    if param_name == "pattern_type" and agent["type"] == "pattern":
                        patterns = ["double_top", "head_shoulders", "triangle", "flag"]
                        agent["parameters"][param_name] = np.random.choice(patterns)
                    
        # 记录变异
        logger.debug(f"智能体 {agent['id']} 已变异")
        
    def _agent_receive_feedback(self, agent, feedback):
        """智能体接收反馈
        
        Args:
            agent: 接收反馈的智能体
            feedback: 反馈信息
        """
        # 可以在这里实现各种反馈处理逻辑
        pass
        
    def _synthesize_decisions(self, micro_decisions, market_features):
        """合成多个微观决策为一个宏观决策
        
        Args:
            micro_decisions: 微观决策列表
            market_features: 市场特征
            
        Returns:
            dict: 合成的宏观决策
        """
        if not micro_decisions:
            return {
                "action": "hold",
                "confidence": 1.0,
                "source": "macro_default"
            }
        
        # 统计各类决策的数量和信心总和
        action_stats = {
            "buy": {"count": 0, "confidence_sum": 0},
            "sell": {"count": 0, "confidence_sum": 0},
            "hold": {"count": 0, "confidence_sum": 0}
        }
        
        # 收集每种决策的信息
        for decision in micro_decisions:
            action = decision.get("action", "hold")
            confidence = decision.get("confidence", 0.5)
            
            if action in action_stats:
                action_stats[action]["count"] += 1
                action_stats[action]["confidence_sum"] += confidence
                
        # 选择得票最多的决策
        max_count = 0
        max_action = "hold"
        for action, stats in action_stats.items():
            if stats["count"] > max_count:
                max_count = stats["count"]
                max_action = action
                
        # 计算信心度
        total_decisions = len(micro_decisions)
        vote_ratio = max_count / total_decisions if total_decisions > 0 else 0
        
        # 计算平均信心度
        avg_confidence = (action_stats[max_action]["confidence_sum"] / max_count 
                          if max_count > 0 else 0.5)
        
        # 最终信心度是投票比例和平均信心的加权
        final_confidence = 0.7 * vote_ratio + 0.3 * avg_confidence
        
        # 计算每个动作的百分比
        action_percents = {}
        for action, stats in action_stats.items():
            action_percents[action] = stats["count"] / total_decisions if total_decisions > 0 else 0
        
        # 构建决策
        decision = {
            "action": max_action,
            "confidence": final_confidence,
            "vote_counts": {action: stats["count"] for action, stats in action_stats.items()},
            "vote_percents": action_percents,
            "total_votes": total_decisions,
            "source": "macro_synthesis"
        }
        
        return decision
        
    def _update_macro_agent(self, micro_agents, performance_stats):
        """更新宏观智能体的权重和策略
        
        Args:
            micro_agents: 微观智能体列表
            performance_stats: 性能统计
        """
        # 这里可以实现宏观智能体的学习更新逻辑
        pass
        
    def _create_default_strategies(self):
        """创建默认智能体网络的策略
        
        初始化基础策略集合，用于微观智能体决策
        """
        # 创建基础策略逻辑
        try:
            # 确保domains属性存在
            if not hasattr(self, 'domains') or not self.domains:
                self.domains = self._define_domains()
                
            # 确保micro_agents属性初始化
            if not hasattr(self, 'micro_agents') or self.micro_agents is None:
                self.micro_agents = []
                
            # 预定义几种智能体类型的特征关注点
            specialized_features = {
                'trend': ["price", "ma5", "ma10", "trend", "trend_strength"],
                'reversal': ["price", "rsi", "price_change", "momentum"],
                'volume': ["price", "volume", "volume_change_pct", "price_change_pct"],
                'volatility': ["volatility", "price_range", "price"],
                'pattern': ["price", "pattern", "price_change"]
            }
            
            # 为每个领域创建多个类型的智能体
            for domain_name, segments in self.domains.items():
                logger.info(f"为领域 {domain_name} 创建智能体")
                
                # 对每个市场分段，创建多种类型的智能体
                for segment in segments:
                    # 创建趋势型智能体
                    trend_agent = MicroAgent(
                        market_segment=segment,
                        feature_focus=specialized_features['trend'],
                        specialization={"type": "trend", "domain": domain_name},
                        agent_id=f"trend_{domain_name}_{segment}_{int(time.time())}"
                    )
                    self.micro_agents.append(trend_agent)
                    
                    # 创建反转型智能体
                    reversal_agent = MicroAgent(
                        market_segment=segment,
                        feature_focus=specialized_features['reversal'],
                        specialization={"type": "reversal", "domain": domain_name},
                        agent_id=f"reversal_{domain_name}_{segment}_{int(time.time())}"
                    )
                    self.micro_agents.append(reversal_agent)
                    
                    # 创建交易量型智能体
                    volume_agent = MicroAgent(
                        market_segment=segment,
                        feature_focus=specialized_features['volume'],
                        specialization={"type": "volume", "domain": domain_name},
                        agent_id=f"volume_{domain_name}_{segment}_{int(time.time())}"
                    )
                    self.micro_agents.append(volume_agent)
                    
                    # 创建波动率型智能体
                    volatility_agent = MicroAgent(
                        market_segment=segment,
                        feature_focus=specialized_features['volatility'],
                        specialization={"type": "volatility", "domain": domain_name},
                        agent_id=f"volatility_{domain_name}_{segment}_{int(time.time())}"
                    )
                    self.micro_agents.append(volatility_agent)
                    
                    # 创建形态型智能体
                    pattern_agent = MicroAgent(
                        market_segment=segment,
                        feature_focus=specialized_features['pattern'],
                        specialization={"type": "pattern", "domain": domain_name},
                        agent_id=f"pattern_{domain_name}_{segment}_{int(time.time())}"
                    )
                    self.micro_agents.append(pattern_agent)
            
            # 尝试创建并添加策略集合
            try:
                from quantum_symbiotic_network.strategies import create_default_strategy_ensemble
                strategy_ensemble = create_default_strategy_ensemble()
                logger.info("已创建默认策略集")
                
                # 将策略集添加到所有智能体
                for agent in self.micro_agents:
                    agent.strategy_ensemble = strategy_ensemble
            except ImportError:
                logger.warning("无法导入策略集，智能体将使用增强型随机决策")
            
            # 创建中层智能体
            self._create_mid_agents()
            
            logger.info(f"已创建 {len(self.micro_agents)} 个微观智能体")
        except Exception as e:
            logger.error(f"创建默认策略集失败: {e}")
            
    def _create_mid_agents(self):
        """创建中层智能体,并将微观智能体连接到中层智能体"""
        # 确保mid_agents属性初始化
        if not hasattr(self, 'mid_agents') or self.mid_agents is None:
            self.mid_agents = []
            
        # 按领域分组微观智能体
        domain_agents = {}
        for agent in self.micro_agents:
            domain = agent.specialization.get("domain", "default")
            if domain not in domain_agents:
                domain_agents[domain] = []
            domain_agents[domain].append(agent)
            
        # 为每个领域创建中层智能体
        for domain, agents in domain_agents.items():
            mid_agent = MidAgent(domain=domain)
            
            # 将相关的微观智能体连接到中层智能体
            for micro_agent in agents:
                # 随机初始化权重
                weight = np.random.uniform(0.5, 1.5)
                mid_agent.add_micro_agent(micro_agent, weight)
                
            self.mid_agents.append(mid_agent)
            logger.info(f"创建了领域 '{domain}' 的中层智能体,连接了 {len(agents)} 个微观智能体")
            
    def step(self, market_data, feedback=None):
        """处理单个时间步，生成交易决策
        
        Args:
            market_data: 市场数据
            feedback: 上一步的反馈
            
        Returns:
            dict: 包含交易决策的字典
        """
        try:
            # 默认决策，确保总是返回包含action的决策
            default_decision = {
                "action": "hold",
                "confidence": 1.0,
                "timestamp": int(time.time()),
                "source": "fractal_network_default"
            }
            
            if not market_data:
                logger.warning("没有市场数据提供给分形网络")
                return default_decision
            
            # 预处理环境数据,确保格式符合要求
            processed_env = self._preprocess_environment(market_data)
            
            # 提供反馈给智能体网络
            if feedback:
                self._provide_feedback(feedback)
            
            # 有概率进行网络进化
            if np.random.random() < self.evolution_rate:
                self._evolve_network()
            
            # 如果没有主元智能体,创建一个
            if not hasattr(self, 'main_meta_agent') or self.main_meta_agent is None:
                self.main_meta_agent = self._create_meta_agent()
                
            # 使用主元智能体感知环境并做出决策
            meta_perception = self.main_meta_agent.perceive(processed_env)
            meta_decision = self.main_meta_agent.decide(meta_perception)
            
            # 如果元智能体决策无效，收集所有微观智能体的决策
            if meta_decision.get("action", "no_action") == "no_action":
                logger.warning("元智能体未产生有效决策，回退到微观智能体投票")
                
                # 收集所有微观智能体的决策
                micro_decisions = []
                for agent in self.micro_agents:
                    try:
                        perception = agent.perceive(processed_env)
                        agent_decision = agent.decide(perception)
                        if agent_decision and "action" in agent_decision:
                            micro_decisions.append(agent_decision)
                    except Exception as e:
                        logger.error(f"智能体决策生成失败: {e}")
                
                # 如果没有有效决策，返回默认决策
                if not micro_decisions:
                    logger.warning("没有微观智能体产生有效决策")
                    return default_decision
                
                # 使用多数投票
                actions = [d.get("action") for d in micro_decisions if d.get("action")]
                if not actions:
                    return default_decision
                    
                # 统计各动作的票数
                from collections import Counter
                action_counts = Counter(actions)
                most_common_action = action_counts.most_common(1)[0][0]
                
                # 计算信心度
                confidence = action_counts[most_common_action] / len(actions)
                
                # 构建最终决策
                decision = {
                    "action": most_common_action,
                    "confidence": confidence,
                    "vote_count": dict(action_counts),
                    "timestamp": int(time.time()),
                    "source": "fractal_network_voting"
                }
                
                return decision
            else:
                # 返回元智能体的决策
                meta_decision["timestamp"] = int(time.time())
                meta_decision["source"] = "meta_agent_decision"
                return meta_decision
        
        except Exception as e:
            logger.error(f"分形网络决策失败: {e}")
            return {
                "action": "hold",
                "confidence": 1.0,
                "error": str(e),
                "timestamp": int(time.time()),
                "source": "fractal_network_error"
            }
            
    def _preprocess_environment(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """预处理环境数据，确保格式正确,增强特征提取
        
        Args:
            environment: 原始环境数据
            
        Returns:
            处理后的环境数据
        """
        processed_env = {"timestamp": environment.get("timestamp", self.current_time)}
        
        try:
            # 提取市场数据
            if "stocks" in environment:
                # 直接使用stocks字典作为市场分段
                for stock_code, stock_data in environment["stocks"].items():
                    # 增强每个股票的数据,添加计算的指标
                    enhanced_data = self._enhance_stock_data(stock_data)
                    processed_env[stock_code] = enhanced_data
                    
                # 添加全局市场指标
                if "indices" in environment:
                    processed_env["global_market"] = self._enhance_stock_data(environment["indices"])
            elif isinstance(environment, dict):
                # 环境可能已经是处理过的格式或其他结构
                # 尝试处理每个键值对
                for key, value in environment.items():
                    if key != "timestamp" and isinstance(value, dict):
                        processed_env[key] = self._enhance_stock_data(value)
                    else:
                        processed_env[key] = value
        except Exception as e:
            logger.error(f"环境数据预处理失败: {e}")
            # 如果预处理失败,至少保留原始数据
            for key, value in environment.items():
                if key not in processed_env:
                    processed_env[key] = value
                    
        return processed_env
        
    def _enhance_stock_data(self, stock_data):
        """增强股票数据,计算额外的技术指标"""
        if not isinstance(stock_data, dict):
            return stock_data
            
        enhanced_data = stock_data.copy()
        
        try:
            # 确保关键数据存在
            if "prices" in stock_data and len(stock_data["prices"]) > 0:
                prices = stock_data["prices"]
                
                # 计算简单的价格变化
                if len(prices) > 1:
                    enhanced_data["price_change"] = prices[-1] - prices[-2]
                    enhanced_data["price_change_pct"] = enhanced_data["price_change"] / prices[-2] if prices[-2] != 0 else 0
                
                # 计算移动平均线
                if len(prices) >= 5:
                    enhanced_data["ma5"] = sum(prices[-5:]) / 5
                if len(prices) >= 10:
                    enhanced_data["ma10"] = sum(prices[-10:]) / 10
                if len(prices) >= 20:
                    enhanced_data["ma20"] = sum(prices[-20:]) / 20
                
                # 计算相对强弱指标(RSI)
                if len(prices) >= 14:
                    gains = []
                    losses = []
                    for i in range(1, 14):
                        change = prices[-i] - prices[-i-1]
                        if change >= 0:
                            gains.append(change)
                            losses.append(0)
                        else:
                            gains.append(0)
                            losses.append(abs(change))
                    
                    avg_gain = sum(gains) / 14
                    avg_loss = sum(losses) / 14
                    
                    if avg_loss == 0:
                        enhanced_data["rsi"] = 100
                    else:
                        rs = avg_gain / avg_loss
                        enhanced_data["rsi"] = 100 - (100 / (1 + rs))
                
                # 计算波动率(标准差)
                if len(prices) >= 5:
                    import numpy as np
                    enhanced_data["volatility"] = np.std(prices[-5:])
                
                # 简单趋势检测
                if len(prices) >= 10:
                    import numpy as np
                    y = np.array(prices[-10:])
                    x = np.arange(10)
                    try:
                        slope = np.polyfit(x, y, 1)[0]
                        enhanced_data["trend"] = slope
                        enhanced_data["trend_strength"] = abs(slope) / (prices[-1] / 100) if prices[-1] != 0 else 0
                    except:
                        enhanced_data["trend"] = 0
                        enhanced_data["trend_strength"] = 0
            
            # 处理交易量数据
            if "volumes" in stock_data and len(stock_data["volumes"]) > 0:
                volumes = stock_data["volumes"]
                
                # 计算交易量变化
                if len(volumes) > 1:
                    enhanced_data["volume_change"] = volumes[-1] - volumes[-2]
                    enhanced_data["volume_change_pct"] = enhanced_data["volume_change"] / volumes[-2] if volumes[-2] != 0 else 0
                
                # 计算相对交易量(与平均值比较)
                if len(volumes) >= 5:
                    avg_volume = sum(volumes[-5:]) / 5
                    enhanced_data["relative_volume"] = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        except Exception as e:
            logger.error(f"增强股票数据失败: {e}")
            
        # 确保加工后的数据中包含价格
        if "prices" in stock_data and len(stock_data["prices"]) > 0:
            enhanced_data["price"] = stock_data["prices"][-1]
            
        if "volumes" in stock_data and len(stock_data["volumes"]) > 0:
            enhanced_data["volume"] = stock_data["volumes"][-1]
            
        return enhanced_data

    def _create_meta_agent(self):
        """创建元智能体
        
        Returns:
            MetaAgent: 主元智能体实例
        """
        # 创建主元智能体
        try:
            meta_agent = MetaAgent(name="MainMetaAgent")
            
            # 将所有中层智能体连接到元智能体
            if hasattr(self, 'mid_agents') and self.mid_agents:
                for mid_agent in self.mid_agents:
                    # 随机初始化权重
                    weight = np.random.uniform(0.7, 1.3)
                    meta_agent.add_mid_agent(mid_agent, weight)
            
            return meta_agent
        except Exception as e:
            logger.error(f"创建元智能体失败: {e}")
            # 创建一个基本的元智能体
            return MetaAgent(name="BasicMetaAgent")