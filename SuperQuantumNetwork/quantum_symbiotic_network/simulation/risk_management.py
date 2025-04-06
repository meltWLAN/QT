"""
动态风险管理模块 - 实现高级风险控制和资金管理
提供最优仓位管理、止损策略和投资组合优化
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class RiskProfile:
    """风险配置"""
    max_position_size: float = 0.2  # 单一资产最大仓位
    max_portfolio_risk: float = 0.3  # 整体组合最大风险
    risk_free_rate: float = 0.03  # 无风险利率(年化)
    target_sharpe: float = 1.0  # 目标夏普比率
    max_drawdown_limit: float = 0.15  # 最大回撤限制
    position_sizing_model: str = "kelly"  # 仓位大小模型
    stop_loss_pct: float = 0.05  # 止损百分比
    take_profit_pct: float = 0.15  # 止盈百分比
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "max_position_size": self.max_position_size,
            "max_portfolio_risk": self.max_portfolio_risk,
            "risk_free_rate": self.risk_free_rate,
            "target_sharpe": self.target_sharpe,
            "max_drawdown_limit": self.max_drawdown_limit,
            "position_sizing_model": self.position_sizing_model,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskProfile':
        """从字典创建风险配置"""
        return cls(
            max_position_size=data.get("max_position_size", 0.2),
            max_portfolio_risk=data.get("max_portfolio_risk", 0.3),
            risk_free_rate=data.get("risk_free_rate", 0.03),
            target_sharpe=data.get("target_sharpe", 1.0),
            max_drawdown_limit=data.get("max_drawdown_limit", 0.15),
            position_sizing_model=data.get("position_sizing_model", "kelly"),
            stop_loss_pct=data.get("stop_loss_pct", 0.05),
            take_profit_pct=data.get("take_profit_pct", 0.15)
        )

class RiskManagementEngine:
    """风险管理引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化风险管理引擎
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.risk_profile = RiskProfile.from_dict(self.config.get("risk_profile", {}))
        
        # 历史波动率
        self.volatility_history: Dict[str, List[float]] = {}
        # 资产相关性矩阵
        self.correlation_matrix: pd.DataFrame = None
        # 资产权重
        self.asset_weights: Dict[str, float] = {}
        # 投资组合表现历史
        self.portfolio_history: List[Dict[str, Any]] = []
        # 当前仓位
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        # 止损止盈水平
        self.stop_levels: Dict[str, Dict[str, float]] = {}
        
        logger.info("风险管理引擎初始化完成")
        
    def calculate_volatility(self, symbol: str, price_history: List[float], window: int = 20) -> float:
        """计算资产波动率
        
        Args:
            symbol: 资产代码
            price_history: 价格历史
            window: 窗口大小
            
        Returns:
            float: 波动率(标准差)
        """
        if len(price_history) < window:
            logger.warning(f"价格历史不足，无法计算{symbol}的波动率")
            return 0.15  # 默认波动率
            
        # 计算收益率
        returns = [price_history[i] / price_history[i-1] - 1 for i in range(1, len(price_history))]
        
        # 使用指数加权移动平均计算波动率
        weights = np.exp(np.linspace(0, 1, min(window, len(returns))))
        weights = weights / weights.sum()
        
        # 收益率加权标准差
        volatility = np.sqrt(np.sum(weights * (returns[-min(window, len(returns)):]**2)))
        
        # 记录波动率历史
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []
        self.volatility_history[symbol].append(volatility)
        
        return volatility
        
    def update_correlation_matrix(self, price_data: Dict[str, List[float]]):
        """更新资产相关性矩阵
        
        Args:
            price_data: 各资产价格数据
        """
        if not price_data or len(price_data) < 2:
            logger.warning("数据不足，无法更新相关性矩阵")
            return
            
        # 创建收益率数据
        returns_data = {}
        
        for symbol, prices in price_data.items():
            if len(prices) < 2:
                continue
                
            # 计算日收益率
            returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
            returns_data[symbol] = returns
            
        if len(returns_data) < 2:
            logger.warning("收益率数据不足，无法更新相关性矩阵")
            return
            
        # 创建DataFrame用于计算相关性
        returns_df = pd.DataFrame(returns_data)
        
        # 计算相关性矩阵
        self.correlation_matrix = returns_df.corr()
        
        logger.debug(f"已更新{len(returns_data)}个资产的相关性矩阵")
        
    def calculate_optimal_weights(self, expected_returns: Dict[str, float],
                                volatilities: Dict[str, float]) -> Dict[str, float]:
        """计算最优资产权重(最小方差)
        
        Args:
            expected_returns: 预期收益率
            volatilities: 波动率
            
        Returns:
            dict: 资产权重
        """
        if not self.correlation_matrix is not None:
            logger.warning("相关性矩阵未初始化，无法计算最优权重")
            return {symbol: 1.0 / len(expected_returns) for symbol in expected_returns}
            
        symbols = list(expected_returns.keys())
        
        # 检查所有资产是否都在相关性矩阵中
        valid_symbols = [s for s in symbols if s in self.correlation_matrix.index]
        
        if len(valid_symbols) < len(symbols):
            logger.warning(f"部分资产({len(symbols)-len(valid_symbols)}个)不在相关性矩阵中")
            
        if len(valid_symbols) < 2:
            # 无法优化，使用等权重
            return {symbol: 1.0 / len(symbols) for symbol in symbols}
            
        # 提取相关性子矩阵
        sub_corr = self.correlation_matrix.loc[valid_symbols, valid_symbols]
        
        # 创建协方差矩阵
        cov_matrix = pd.DataFrame(index=valid_symbols, columns=valid_symbols)
        for i in valid_symbols:
            for j in valid_symbols:
                cov_matrix.loc[i, j] = sub_corr.loc[i, j] * volatilities[i] * volatilities[j]
                
        # 最小方差优化
        n = len(valid_symbols)
        
        # 简化版本：反比于总风险分配权重
        weights = {}
        total_inverse_volatility = 0
        
        for symbol in symbols:
            # 使用波动率的倒数作为权重基础
            if volatilities.get(symbol, 0) > 0:
                weights[symbol] = 1.0 / volatilities[symbol]
                total_inverse_volatility += weights[symbol]
            else:
                weights[symbol] = 0
                
        # 归一化权重
        if total_inverse_volatility > 0:
            for symbol in weights:
                weights[symbol] = weights[symbol] / total_inverse_volatility
        else:
            # 等权重
            for symbol in symbols:
                weights[symbol] = 1.0 / len(symbols)
                
        # 应用最大仓位限制
        for symbol in weights:
            weights[symbol] = min(weights[symbol], self.risk_profile.max_position_size)
            
        # 再次归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            for symbol in weights:
                weights[symbol] = weights[symbol] / total_weight
                
        self.asset_weights = weights
        return weights
        
    def calculate_position_size(self, symbol: str, trade_type: str, price: float,
                               confidence: float, volatility: float = None) -> Tuple[float, str]:
        """计算最优仓位大小
        
        Args:
            symbol: 资产代码
            trade_type: 交易类型(buy/sell)
            price: 当前价格
            confidence: 决策置信度
            volatility: 资产波动率
            
        Returns:
            Tuple[float, str]: (仓位比例, 模型)
        """
        # 获取波动率
        if volatility is None:
            if symbol in self.volatility_history and self.volatility_history[symbol]:
                volatility = self.volatility_history[symbol][-1]
            else:
                volatility = 0.15  # 默认波动率
                
        # 获取资产权重
        weight = self.asset_weights.get(symbol, self.risk_profile.max_position_size)
        
        # 根据不同仓位模型计算
        if self.risk_profile.position_sizing_model == "kelly":
            # 使用Kelly准则：f* = edge / odds
            # 其中edge是优势，odds是赔率
            # 计算期望收益率
            if trade_type == "buy":
                expected_return = confidence * 0.1 - (1 - confidence) * volatility
            else:  # sell
                expected_return = confidence * 0.1 - (1 - confidence) * volatility
                
            # 计算Kelly比例
            if expected_return > 0:
                kelly_fraction = expected_return / (volatility * volatility)
                # 通常使用半Kelly或四分之一Kelly来控制风险
                position_size = kelly_fraction * 0.5  # 半Kelly
            else:
                position_size = 0
                
            model = "kelly"
            
        elif self.risk_profile.position_sizing_model == "risk_parity":
            # 风险平价：分配相同的风险
            risk_budget = self.risk_profile.max_portfolio_risk / len(self.asset_weights)
            position_size = risk_budget / volatility
            model = "risk_parity"
            
        else:
            # 默认：固定分数
            position_size = weight * confidence
            model = "fixed_fraction"
            
        # 限制最大仓位
        position_size = min(position_size, self.risk_profile.max_position_size)
        
        # 根据交易类型调整
        if trade_type == "sell":
            position_size = -position_size
            
        return position_size, model
        
    def set_stop_levels(self, symbol: str, entry_price: float, position_type: str) -> Dict[str, float]:
        """设置止损止盈水平
        
        Args:
            symbol: 资产代码
            entry_price: 入场价格
            position_type: 仓位类型(long/short)
            
        Returns:
            dict: 止损止盈水平
        """
        if position_type == "long":
            stop_loss = entry_price * (1 - self.risk_profile.stop_loss_pct)
            take_profit = entry_price * (1 + self.risk_profile.take_profit_pct)
        else:  # short
            stop_loss = entry_price * (1 + self.risk_profile.stop_loss_pct)
            take_profit = entry_price * (1 - self.risk_profile.take_profit_pct)
            
        levels = {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_type": position_type,
            "created_time": datetime.now().isoformat()
        }
        
        self.stop_levels[symbol] = levels
        return levels
        
    def update_dynamic_stops(self, symbol: str, current_price: float) -> Dict[str, float]:
        """更新动态止损水平
        
        Args:
            symbol: 资产代码
            current_price: 当前价格
            
        Returns:
            dict: 更新后的止损水平
        """
        if symbol not in self.stop_levels:
            logger.warning(f"{symbol}没有止损设置")
            return None
            
        levels = self.stop_levels[symbol]
        position_type = levels["position_type"]
        current_stop = levels["stop_loss"]
        
        # 跟踪止损逻辑
        if position_type == "long":
            # 多头使用向上跟踪止损
            if current_price > levels["entry_price"]:
                # 价格上涨，更新止损
                new_stop = current_price * (1 - self.risk_profile.stop_loss_pct * 0.8)
                if new_stop > current_stop:
                    levels["stop_loss"] = new_stop
                    logger.debug(f"{symbol}多头止损上移至{new_stop:.2f}")
        else:  # short
            # 空头使用向下跟踪止损
            if current_price < levels["entry_price"]:
                # 价格下跌，更新止损
                new_stop = current_price * (1 + self.risk_profile.stop_loss_pct * 0.8)
                if new_stop < current_stop:
                    levels["stop_loss"] = new_stop
                    logger.debug(f"{symbol}空头止损下移至{new_stop:.2f}")
                    
        # 更新止损设置
        self.stop_levels[symbol] = levels
        return levels
        
    def check_stop_conditions(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """检查止损止盈条件
        
        Args:
            symbol: 资产代码
            current_price: 当前价格
            
        Returns:
            dict: 检查结果
        """
        if symbol not in self.stop_levels:
            return {"triggered": False, "reason": "no_stop_level"}
            
        levels = self.stop_levels[symbol]
        position_type = levels["position_type"]
        stop_loss = levels["stop_loss"]
        take_profit = levels["take_profit"]
        
        result = {"triggered": False, "reason": "none", "current_price": current_price}
        
        if position_type == "long":
            if current_price <= stop_loss:
                result["triggered"] = True
                result["reason"] = "stop_loss"
                result["action"] = "sell"
                result["level"] = stop_loss
            elif current_price >= take_profit:
                result["triggered"] = True
                result["reason"] = "take_profit"
                result["action"] = "sell"
                result["level"] = take_profit
        else:  # short
            if current_price >= stop_loss:
                result["triggered"] = True
                result["reason"] = "stop_loss"
                result["action"] = "buy"
                result["level"] = stop_loss
            elif current_price <= take_profit:
                result["triggered"] = True
                result["reason"] = "take_profit"
                result["action"] = "buy"
                result["level"] = take_profit
                
        return result
        
    def update_portfolio_risk(self, positions: Dict[str, Dict[str, Any]], 
                             market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """更新投资组合风险状态
        
        Args:
            positions: 当前持仓
            market_data: 市场数据
            
        Returns:
            dict: 风险状态
        """
        if not positions:
            return {
                "portfolio_risk": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "diversification_score": 0.0,
                "risk_contribution": {}
            }
            
        # 更新相关性矩阵
        price_data = {}
        for symbol, data in market_data.items():
            if "price_history" in data:
                price_data[symbol] = data["price_history"]
                
        self.update_correlation_matrix(price_data)
        
        # 计算各资产波动率
        volatilities = {}
        for symbol, data in market_data.items():
            if "price_history" in data and len(data["price_history"]) > 1:
                volatilities[symbol] = self.calculate_volatility(symbol, data["price_history"])
                
        # 更新仓位
        self.current_positions = positions
        
        # 计算组合风险
        position_sizes = {}
        for symbol, position in positions.items():
            # 计算资产权重
            position_sizes[symbol] = position.get("size", 0.0)
            
        # 检查是否有足够的数据计算
        if not position_sizes or not volatilities or not self.correlation_matrix is not None:
            return {
                "portfolio_risk": sum(abs(size) * volatilities.get(symbol, 0.15) 
                                    for symbol, size in position_sizes.items()),
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "diversification_score": 0.0,
                "risk_contribution": {}
            }
            
        # 获取持仓资产的相关性子矩阵
        symbols = list(position_sizes.keys())
        valid_symbols = [s for s in symbols if s in self.correlation_matrix.index]
        
        if len(valid_symbols) < 2:
            # 单资产或数据不足，简化计算
            portfolio_risk = sum(abs(position_sizes.get(s, 0)) * volatilities.get(s, 0.15) for s in symbols)
            return {
                "portfolio_risk": portfolio_risk,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "diversification_score": 0.0,
                "risk_contribution": {s: abs(position_sizes.get(s, 0)) * volatilities.get(s, 0.15) for s in symbols}
            }
            
        # 计算投资组合风险（简化版，忽略相关性）
        # 实际应用中应使用完整的投资组合方差计算
        portfolio_risk = np.sqrt(sum((position_sizes.get(s, 0) * volatilities.get(s, 0.15))**2 for s in symbols))
        
        # 计算风险贡献
        risk_contribution = {}
        for symbol in symbols:
            position_size = position_sizes.get(symbol, 0.0)
            vol = volatilities.get(symbol, 0.15)
            risk_contribution[symbol] = abs(position_size) * vol / portfolio_risk if portfolio_risk > 0 else 0
            
        # 多样化得分 - 基于相关性
        avg_correlation = 0.5  # 默认中等相关性
        if len(valid_symbols) >= 2:
            sub_corr = self.correlation_matrix.loc[valid_symbols, valid_symbols]
            # 计算平均相关系数（排除对角线）
            mask = ~np.eye(len(valid_symbols), dtype=bool)
            avg_correlation = sub_corr.values[mask].mean()
            
        diversification_score = 1.0 - avg_correlation
        
        # 检查总风险是否超过阈值
        risk_status = "low" if portfolio_risk < self.risk_profile.max_portfolio_risk else "high"
        
        result = {
            "portfolio_risk": portfolio_risk,
            "risk_status": risk_status,
            "max_drawdown": 0.0,  # 需要历史数据计算
            "sharpe_ratio": 0.0,  # 需要收益率数据
            "diversification_score": diversification_score,
            "avg_correlation": avg_correlation,
            "risk_contribution": risk_contribution
        }
        
        return result
        
    def adjust_for_risk_limits(self, positions: Dict[str, Dict[str, Any]], 
                              risk_metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """根据风险限制调整仓位
        
        Args:
            positions: 当前持仓
            risk_metrics: 风险指标
            
        Returns:
            dict: 调整后的持仓
        """
        if not positions or risk_metrics.get("portfolio_risk", 0) <= self.risk_profile.max_portfolio_risk:
            return positions  # 风险在可接受范围内，无需调整
            
        # 需要降低风险：按风险贡献比例缩减仓位
        risk_reduction_factor = self.risk_profile.max_portfolio_risk / risk_metrics.get("portfolio_risk", 1.0)
        
        # 调整仓位
        adjusted_positions = {}
        for symbol, position in positions.items():
            adjusted_position = position.copy()
            # 按风险贡献比例缩减
            new_size = position.get("size", 0.0) * risk_reduction_factor
            adjusted_position["size"] = new_size
            adjusted_position["adjusted_for_risk"] = True
            adjusted_position["original_size"] = position.get("size", 0.0)
            adjusted_positions[symbol] = adjusted_position
            
            logger.info(f"风险控制: {symbol}仓位从{position.get('size', 0.0):.4f}调整为{new_size:.4f}")
            
        return adjusted_positions 