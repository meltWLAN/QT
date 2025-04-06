#!/usr/bin/env python3
"""
风险管理模块 - 提供高级风险控制和资金管理功能
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RiskManagement")

class RiskManager:
    """风险管理器，提供风险控制和资金管理功能"""
    
    def __init__(self, config=None):
        """初始化风险管理器
        
        Args:
            config (dict): 配置信息
        """
        self.config = config or {}
        
        # 默认配置参数
        self.max_position_size = self.config.get('max_position_size', 0.2)  # 单一持仓最大比例
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)  # 最大组合风险
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.15)  # 最大回撤限制
        self.position_sizing_method = self.config.get('position_sizing_method', 'kelly')  # 仓位计算方法
        self.kelly_fraction = self.config.get('kelly_fraction', 0.5)  # 凯利系数分数
        self.stop_loss_atr_multiple = self.config.get('stop_loss_atr_multiple', 2.0)  # 止损ATR倍数
        self.take_profit_atr_multiple = self.config.get('take_profit_atr_multiple', 3.0)  # 止盈ATR倍数
        self.trailing_stop_activation = self.config.get('trailing_stop_activation', 0.03)  # 追踪止损激活阈值
        self.trailing_stop_distance = self.config.get('trailing_stop_distance', 0.02)  # 追踪止损距离
        self.var_confidence_level = self.config.get('var_confidence_level', 0.95)  # VaR置信度
        self.max_correlation = self.config.get('max_correlation', 0.7)  # 最大相关性限制
        self.risk_free_rate = self.config.get('risk_free_rate', 0.03)  # 无风险利率(年化)
        
        # 风险监控历史
        self.risk_history = []
        self.position_history = []
        self.drawdown_history = []
        self.var_history = []
        
        # 性能指标
        self.peak_value = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        
        # 全局风险度量
        self.current_var = 0
        self.current_portfolio_volatility = 0
        self.current_sharpe_ratio = 0
        
        # 日志和报告目录
        self.log_dir = os.path.join("quantum_symbiotic_network", "logs", "risk_management")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def calculate_position_size(self, capital, price, volatility, win_probability, expected_return=None, risk_per_trade=None):
        """计算仓位大小
        
        Args:
            capital (float): 可用资金
            price (float): 股票价格
            volatility (float): 波动率(比如ATR值)
            win_probability (float): 胜率(0-1之间)
            expected_return (float, optional): 预期收益率
            risk_per_trade (float, optional): 每笔交易风险敞口
            
        Returns:
            dict: 仓位信息，包含股数和百分比
        """
        # 如果未指定每笔交易风险敞口，默认使用最大组合风险的一部分
        if risk_per_trade is None:
            risk_per_trade = self.max_portfolio_risk / 5
        
        # 如果未指定预期收益率，使用波动率的一半
        if expected_return is None:
            expected_return = volatility / 2
        
        # 根据策略选择不同的仓位计算方法
        if self.position_sizing_method == 'fixed_risk':
            # 固定风险法
            risk_amount = capital * risk_per_trade
            risk_per_share = volatility * price
            shares = max(1, int(risk_amount / risk_per_share))
            position_value = shares * price
            position_percent = position_value / capital
        
        elif self.position_sizing_method == 'kelly':
            # 凯利公式
            loss_probability = 1 - win_probability
            win_loss_ratio = expected_return / volatility if volatility > 0 else 1
            
            # Kelly比例 = win_probability - (loss_probability / win_loss_ratio)
            kelly = win_probability - (loss_probability / win_loss_ratio)
            
            # 使用Kelly分数来降低风险
            position_percent = max(0, kelly * self.kelly_fraction)
            
            # 限制最大仓位
            position_percent = min(position_percent, self.max_position_size)
            
            position_value = capital * position_percent
            shares = max(1, int(position_value / price))
        
        elif self.position_sizing_method == 'volatility_sizing':
            # 波动率调整法
            target_volatility = 0.01  # 目标日波动率
            position_percent = target_volatility / volatility if volatility > 0 else 0
            
            # 限制最大仓位
            position_percent = min(position_percent, self.max_position_size)
            
            position_value = capital * position_percent
            shares = max(1, int(position_value / price))
        
        else:
            # 默认使用固定比例法
            position_percent = 0.05  # 默认5%
            position_value = capital * position_percent
            shares = max(1, int(position_value / price))
        
        # 确保不超过最大持仓限制
        position_value = shares * price
        position_percent = position_value / capital
        
        if position_percent > self.max_position_size:
            position_percent = self.max_position_size
            position_value = capital * position_percent
            shares = max(1, int(position_value / price))
        
        return {
            'shares': shares,
            'position_value': shares * price,
            'position_percent': (shares * price) / capital,
            'risk_amount': risk_per_trade * capital
        }
    
    def calculate_stop_loss(self, entry_price, current_price, volatility, direction='long'):
        """计算止损价格
        
        Args:
            entry_price (float): 入场价格
            current_price (float): 当前价格
            volatility (float): 波动率(如ATR)
            direction (str): 'long'或'short'
            
        Returns:
            dict: 止损信息
        """
        # ATR倍数止损法
        if direction == 'long':
            stop_price = entry_price - (volatility * self.stop_loss_atr_multiple)
            
            # 如果有利润，考虑追踪止损
            if current_price > entry_price:
                profit_percent = (current_price - entry_price) / entry_price
                
                if profit_percent >= self.trailing_stop_activation:
                    # 激活追踪止损
                    trailing_stop = current_price * (1 - self.trailing_stop_distance)
                    
                    # 使用较高的止损价
                    stop_price = max(stop_price, trailing_stop)
        else:
            # 空头情况
            stop_price = entry_price + (volatility * self.stop_loss_atr_multiple)
            
            # 如果有利润，考虑追踪止损
            if current_price < entry_price:
                profit_percent = (entry_price - current_price) / entry_price
                
                if profit_percent >= self.trailing_stop_activation:
                    # 激活追踪止损
                    trailing_stop = current_price * (1 + self.trailing_stop_distance)
                    
                    # 使用较低的止损价
                    stop_price = min(stop_price, trailing_stop)
        
        # 计算止损距离百分比
        if direction == 'long':
            stop_distance_percent = (current_price - stop_price) / current_price
        else:
            stop_distance_percent = (stop_price - current_price) / current_price
        
        return {
            'stop_price': stop_price,
            'stop_type': 'trailing' if (direction == 'long' and current_price > entry_price and 
                                       (current_price - entry_price) / entry_price >= self.trailing_stop_activation) or
                                      (direction == 'short' and current_price < entry_price and
                                       (entry_price - current_price) / entry_price >= self.trailing_stop_activation)
                         else 'fixed',
            'stop_distance_percent': stop_distance_percent
        }
    
    def calculate_take_profit(self, entry_price, volatility, direction='long'):
        """计算止盈价格
        
        Args:
            entry_price (float): 入场价格
            volatility (float): 波动率(如ATR)
            direction (str): 'long'或'short'
            
        Returns:
            float: 止盈价格
        """
        if direction == 'long':
            take_profit = entry_price + (volatility * self.take_profit_atr_multiple)
        else:
            take_profit = entry_price - (volatility * self.take_profit_atr_multiple)
        
        return take_profit
    
    def calculate_value_at_risk(self, portfolio_value, returns, positions=None, holding_period=1):
        """计算在险价值(VaR)
        
        Args:
            portfolio_value (float): 投资组合总价值
            returns (DataFrame/array): 历史收益率数据
            positions (dict, optional): 当前持仓信息，格式: {symbol: weight}
            holding_period (int): 持有期(天)
            
        Returns:
            float: 在险价值(VaR)
        """
        # 如果是Pandas Series/DataFrame，转换为NumPy数组
        if isinstance(returns, (pd.Series, pd.DataFrame)):
            returns_data = returns.values
        else:
            returns_data = returns
            
        # 如果提供了持仓信息，计算加权收益率
        if positions is not None and len(positions) > 0:
            weighted_returns = np.zeros(len(returns_data))
            for symbol, weight in positions.items():
                if symbol in returns.columns:
                    weighted_returns += weight * returns[symbol].values
            returns_data = weighted_returns
        
        # 计算收益率的百分位数
        var_percentile = 1 - self.var_confidence_level
        var_daily = np.percentile(returns_data, var_percentile * 100)
        
        # 根据持有期调整VaR(假设收益率是独立同分布的)
        var = var_daily * np.sqrt(holding_period)
        
        # 转换为货币价值
        var_amount = portfolio_value * abs(var)
        
        # 记录VaR历史
        self.var_history.append({
            'timestamp': datetime.now(),
            'var_percent': var,
            'var_amount': var_amount,
            'portfolio_value': portfolio_value
        })
        
        self.current_var = var
        
        return {
            'var_percent': var,
            'var_amount': var_amount,
            'confidence_level': self.var_confidence_level,
            'holding_period': holding_period
        }
    
    def analyze_portfolio_risk(self, positions, historical_data, current_value):
        """分析投资组合风险
        
        Args:
            positions (dict): 当前持仓信息，格式: {symbol: {shares: int, value: float}}
            historical_data (dict): 历史数据，格式: {symbol: DataFrame}
            current_value (float): 投资组合当前价值
            
        Returns:
            dict: 风险分析结果
        """
        if not positions or len(positions) == 0:
            return {
                'total_risk': 0,
                'diversification_score': 1,
                'concentration_risk': 0,
                'var': 0,
                'sharpe_ratio': 0,
                'max_drawdown': self.max_drawdown
            }
        
        # 计算仓位权重
        weights = {}
        for symbol, pos_info in positions.items():
            weights[symbol] = pos_info['value'] / current_value
        
        # 计算相关性矩阵
        returns = {}
        for symbol, hist_data in historical_data.items():
            if symbol in positions and 'close' in hist_data.columns:
                returns[symbol] = hist_data['close'].pct_change().dropna()
        
        # 如果数据不足，返回默认值
        if len(returns) < 2:
            return {
                'total_risk': 0.1,  # 默认风险
                'diversification_score': 0.5,
                'concentration_risk': sum([w**2 for w in weights.values()]),
                'var': current_value * 0.05,  # 默认VaR
                'sharpe_ratio': 0,
                'max_drawdown': self.max_drawdown
            }
        
        # 创建收益率DataFrame
        returns_df = pd.DataFrame(returns)
        
        # 计算协方差矩阵
        cov_matrix = returns_df.cov() * 252  # 年化
        
        # 计算相关性矩阵
        corr_matrix = returns_df.corr()
        
        # 计算投资组合波动率
        portfolio_variance = 0
        for i, symbol_i in enumerate(weights.keys()):
            for j, symbol_j in enumerate(weights.keys()):
                if symbol_i in cov_matrix.index and symbol_j in cov_matrix.columns:
                    weight_i = weights[symbol_i]
                    weight_j = weights[symbol_j]
                    covariance = cov_matrix.loc[symbol_i, symbol_j]
                    portfolio_variance += weight_i * weight_j * covariance
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        self.current_portfolio_volatility = portfolio_volatility
        
        # 计算分散化得分(平均相关性的倒数)
        avg_correlation = 0
        corr_count = 0
        
        for i, symbol_i in enumerate(weights.keys()):
            for j, symbol_j in enumerate(weights.keys()):
                if i < j and symbol_i in corr_matrix.index and symbol_j in corr_matrix.columns:
                    avg_correlation += abs(corr_matrix.loc[symbol_i, symbol_j])
                    corr_count += 1
        
        if corr_count > 0:
            avg_correlation /= corr_count
            diversification_score = 1 - avg_correlation
        else:
            diversification_score = 0.5  # 默认值
        
        # 计算集中度风险(赫芬达尔-赫希曼指数)
        concentration_risk = sum([w**2 for w in weights.values()])
        
        # 计算VaR
        portfolio_returns = np.zeros(len(returns_df))
        for symbol, weight in weights.items():
            if symbol in returns_df.columns:
                portfolio_returns += weight * returns_df[symbol].values
        
        var_result = self.calculate_value_at_risk(current_value, portfolio_returns)
        
        # 计算夏普比率
        portfolio_daily_returns = np.zeros(len(returns_df))
        for symbol, weight in weights.items():
            if symbol in returns_df.columns:
                portfolio_daily_returns += weight * returns_df[symbol].values
        
        avg_return = np.mean(portfolio_daily_returns) * 252  # 年化
        daily_rf = self.risk_free_rate / 252
        excess_return = avg_return - self.risk_free_rate
        
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        self.current_sharpe_ratio = sharpe_ratio
        
        # 返回风险分析结果
        return {
            'total_risk': portfolio_volatility,
            'diversification_score': diversification_score,
            'concentration_risk': concentration_risk,
            'var': var_result,
            'sharpe_ratio': sharpe_ratio,
            'volatility_breakdown': {symbol: np.std(returns_df[symbol]) * np.sqrt(252) for symbol in weights.keys() if symbol in returns_df.columns},
            'max_drawdown': self.max_drawdown
        }
    
    def update_drawdown(self, current_value):
        """更新回撤计算
        
        Args:
            current_value (float): 当前投资组合价值
            
        Returns:
            dict: 回撤信息
        """
        # 更新峰值
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0
        else:
            # 计算当前回撤
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            
            # 更新最大回撤
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # 记录回撤历史
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'value': current_value,
            'peak': self.peak_value,
            'drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown
        })
        
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'peak_value': self.peak_value
        }
    
    def check_risk_limits(self, portfolio_value, positions, historical_data):
        """检查风险限制
        
        Args:
            portfolio_value (float): 投资组合价值
            positions (dict): 当前持仓
            historical_data (dict): 历史数据
            
        Returns:
            dict: 风险检查结果和建议
        """
        # 更新回撤
        drawdown_info = self.update_drawdown(portfolio_value)
        
        # 分析投资组合风险
        risk_analysis = self.analyze_portfolio_risk(positions, historical_data, portfolio_value)
        
        # 初始化风险检查结果
        risk_status = {
            'exceeded_limits': False,
            'warnings': [],
            'recommendations': []
        }
        
        # 检查回撤限制
        if drawdown_info['current_drawdown'] >= self.max_drawdown_limit:
            risk_status['exceeded_limits'] = True
            risk_status['warnings'].append({
                'type': 'max_drawdown',
                'message': f"当前回撤 ({drawdown_info['current_drawdown']:.2%}) 超过最大回撤限制 ({self.max_drawdown_limit:.2%})",
                'severity': 'high'
            })
            risk_status['recommendations'].append("考虑减少头寸或使用对冲策略")
        elif drawdown_info['current_drawdown'] >= self.max_drawdown_limit * 0.8:
            risk_status['warnings'].append({
                'type': 'max_drawdown',
                'message': f"当前回撤 ({drawdown_info['current_drawdown']:.2%}) 接近最大回撤限制 ({self.max_drawdown_limit:.2%})",
                'severity': 'medium'
            })
            risk_status['recommendations'].append("考虑调整仓位，降低风险暴露")
        
        # 检查VaR限制
        var_limit = portfolio_value * self.max_portfolio_risk
        if risk_analysis['var']['var_amount'] > var_limit:
            risk_status['exceeded_limits'] = True
            risk_status['warnings'].append({
                'type': 'var',
                'message': f"当前VaR ({risk_analysis['var']['var_amount']:.2f}) 超过限制 ({var_limit:.2f})",
                'severity': 'high'
            })
            risk_status['recommendations'].append("减少高风险资产敞口")
        
        # 检查集中度风险
        concentration_limit = 0.3  # 集中度风险限制
        if risk_analysis['concentration_risk'] > concentration_limit:
            risk_status['warnings'].append({
                'type': 'concentration',
                'message': f"集中度风险 ({risk_analysis['concentration_risk']:.2f}) 过高",
                'severity': 'medium'
            })
            risk_status['recommendations'].append("增加资产多样性，降低单一资产集中度")
        
        # 检查分散化得分
        if risk_analysis['diversification_score'] < 0.3:
            risk_status['warnings'].append({
                'type': 'diversification',
                'message': f"分散化得分 ({risk_analysis['diversification_score']:.2f}) 较低",
                'severity': 'medium'
            })
            risk_status['recommendations'].append("增加负相关性资产，提高投资组合分散化")
        
        # 检查单个持仓风险
        for symbol, pos_info in positions.items():
            position_weight = pos_info['value'] / portfolio_value
            
            if position_weight > self.max_position_size:
                risk_status['warnings'].append({
                    'type': 'position_size',
                    'message': f"{symbol} 持仓比例 ({position_weight:.2%}) 超过限制 ({self.max_position_size:.2%})",
                    'severity': 'medium'
                })
                risk_status['recommendations'].append(f"考虑减少 {symbol} 的持仓比例")
        
        # 记录风险状态
        self.risk_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'risk_analysis': risk_analysis,
            'risk_status': risk_status,
            'drawdown': drawdown_info
        })
        
        return {
            'risk_status': risk_status,
            'risk_analysis': risk_analysis,
            'drawdown': drawdown_info
        }
    
    def optimize_position_sizes(self, capital, stocks_data, predictions, max_positions=5):
        """优化投资组合持仓比例
        
        Args:
            capital (float): 可用资金
            stocks_data (dict): 股票数据，格式: {symbol: DataFrame}
            predictions (dict): 预测数据，格式: {symbol: {expected_return: float, volatility: float, confidence: float}}
            max_positions (int): 最大持仓数量
            
        Returns:
            dict: 优化后的持仓分配
        """
        # 提取每只股票的预期收益、波动率和置信度
        stocks_metrics = []
        
        for symbol, prediction in predictions.items():
            if symbol in stocks_data:
                # 提取股票数据
                stock_df = stocks_data[symbol]
                price = stock_df['close'].iloc[-1] if 'close' in stock_df.columns else 0
                
                if price > 0:
                    stocks_metrics.append({
                        'symbol': symbol,
                        'price': price,
                        'expected_return': prediction.get('expected_return', 0),
                        'volatility': prediction.get('volatility', 0.01),
                        'confidence': prediction.get('confidence', 0.5),
                        'sharpe': prediction.get('expected_return', 0) / prediction.get('volatility', 0.01) if prediction.get('volatility', 0.01) > 0 else 0
                    })
        
        # 如果没有足够的数据，返回空结果
        if len(stocks_metrics) == 0:
            return {}
        
        # 按风险调整后收益率(夏普比率*置信度)排序
        stocks_metrics.sort(key=lambda x: x['sharpe'] * x['confidence'], reverse=True)
        
        # 选择前N只股票
        selected_stocks = stocks_metrics[:max_positions]
        
        # 如果没有选择任何股票，返回空结果
        if len(selected_stocks) == 0:
            return {}
        
        # 计算总的置信度调整夏普比率
        total_adjusted_sharpe = sum([s['sharpe'] * s['confidence'] for s in selected_stocks])
        
        # 按调整后的夏普比率分配权重
        allocations = {}
        
        for stock in selected_stocks:
            # 计算理论权重
            if total_adjusted_sharpe > 0:
                weight = (stock['sharpe'] * stock['confidence']) / total_adjusted_sharpe
            else:
                weight = 1.0 / len(selected_stocks)
            
            # 计算凯利比例校正
            win_probability = (stock['confidence'] + 0.5) / 2  # 从置信度转换为胜率
            volatility = stock['volatility']
            expected_return = stock['expected_return']
            
            position_info = self.calculate_position_size(
                capital * weight,
                stock['price'],
                volatility,
                win_probability,
                expected_return
            )
            
            allocations[stock['symbol']] = {
                'weight': weight,
                'shares': position_info['shares'],
                'value': position_info['position_value'],
                'expected_return': stock['expected_return'],
                'volatility': stock['volatility'],
                'confidence': stock['confidence']
            }
        
        # 确保总分配不超过1
        total_allocated = sum([a['value'] for a in allocations.values()])
        
        if total_allocated > capital:
            scale_factor = capital / total_allocated
            for symbol in allocations:
                allocations[symbol]['shares'] = math.floor(allocations[symbol]['shares'] * scale_factor)
                allocations[symbol]['value'] = allocations[symbol]['shares'] * stocks_metrics[0]['price']
                allocations[symbol]['weight'] = allocations[symbol]['value'] / capital
        
        return allocations
    
    def generate_risk_report(self, portfolio_value, positions, historical_data):
        """生成风险报告
        
        Args:
            portfolio_value (float): 投资组合价值
            positions (dict): 当前持仓
            historical_data (dict): 历史数据
            
        Returns:
            dict: 风险报告
        """
        # 检查风险限制
        risk_check = self.check_risk_limits(portfolio_value, positions, historical_data)
        
        # 生成报告
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio_value': portfolio_value,
            'risk_analysis': risk_check['risk_analysis'],
            'risk_status': risk_check['risk_status'],
            'drawdown': risk_check['drawdown'],
            'positions_summary': {
                'count': len(positions),
                'largest_position': max([p['value'] for symbol, p in positions.items()]) if positions else 0,
                'largest_position_pct': max([p['value'] / portfolio_value for symbol, p in positions.items()]) if positions else 0,
            },
            'recommendations': risk_check['risk_status']['recommendations'],
            'performance_metrics': {
                'sharpe_ratio': self.current_sharpe_ratio,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'volatility': self.current_portfolio_volatility
            }
        }
        
        # 保存报告
        report_file = os.path.join(self.log_dir, f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
        
        return report 