"""
超神系统 - 量子共生网络 - 仓位管理器 (v3.0)
增强版: 动态风险平价与凯利准则高阶优化
"""

import numpy as np
import pandas as pd
import logging
import random
from datetime import datetime
from scipy import stats, optimize

logger = logging.getLogger(__name__)

class SuperPositionManager:
    """
    超级仓位管理器 3.0 - 智能仓位大小计算与动态优化
    核心特性:
    1. 动态风险平价算法 - 基于实时市场条件调整资产配置权重
    2. 凯利准则高阶优化 - 多维风险模型精确控制仓位暴露
    3. 情景模拟仓位验证 - 蒙特卡洛模拟验证决策稳健性
    4. 自适应杠杆控制系统 - 根据波动率特征动态调整杠杆水平
    5. 多资产相关性优化 - 考虑资产相关性结构的配置优化
    """
    
    def __init__(self):
        """初始化仓位管理器"""
        self.name = "超级仓位管理器"
        self.version = "3.0.0"
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化{self.name} v{self.version}")
        self.max_position_size = 0.20  # 单一资产最大仓位
        self.max_total_exposure = 0.60  # 总体最大风险敞口
        self.drawdown_limit = 0.15  # 最大可接受回撤
        self.current_positions = {}  # 当前持仓
        self.position_history = []  # 仓位历史
        
        # 风险平价参数
        self.risk_parity_target_contribution = 'equal'  # 平等风险贡献
        self.risk_parity_lookback = 90  # 风险平价回看天数
        self.volatility_scaling_factor = 0.12  # 波动率缩放因子(年化12%)
        self.rebalance_threshold = 0.05  # 再平衡阈值(权重偏离5%)
        
        # 凯利准则高阶参数
        self.kelly_fraction = 0.5  # 半凯利策略
        self.max_kelly_cap = 0.2  # 最大凯利配置上限
        self.drawdown_protection_factor = 0.3  # 回撤保护因子
        self.multi_asset_kelly_correlation = True  # 启用考虑相关性的多资产凯利
        
        # 情景模拟参数
        self.monte_carlo_trials = 1000  # 蒙特卡洛模拟次数
        self.stress_scenarios = {
            'extreme_volatility': {'factor': 2.5},  # 极端波动
            'correlation_breakdown': {'factor': 0.3},  # 相关性崩溃
            'liquidity_crisis': {'factor': 0.5}  # 流动性危机
        }
        
        # 自适应杠杆参数
        self.max_leverage = 1.5  # 最大杠杆倍数
        self.volatility_scaling_enabled = True  # 启用波动率缩放
        self.target_portfolio_volatility = 0.15  # 目标组合年化波动率
        self.leverage_adjustment_speed = 0.2  # 杠杆调整速度
        
        # 初始化风险分配矩阵
        self.risk_contribution_matrix = {}
        
    def calculate_position_size(self, capital, risk_per_trade, stop_loss_percent):
        """
        计算适当的仓位大小
        """
        self.logger.info("计算仓位大小...")
        
        # 防止除零错误
        if stop_loss_percent <= 0:
            self.logger.warning("止损百分比必须大于0，使用默认值2%")
            stop_loss_percent = 0.02
        
        # 风险金额
        risk_amount = capital * risk_per_trade
        
        # 计算仓位大小
        position_size = risk_amount / stop_loss_percent
        
        # 应用最大仓位限制
        max_allowed = capital * self.max_position_size
        if position_size > max_allowed:
            self.logger.info(f"计算的仓位 {position_size:.2f} 超过最大允许值，调整为 {max_allowed:.2f}")
            position_size = max_allowed
            
        return position_size
    
    def apply_kelly_criterion(self, win_rate, win_loss_ratio, volatility=None, correlation_matrix=None, assets=None):
        """
        应用高级凯利公式计算最优仓位比例，支持多维风险模型
        
        参数:
        - win_rate: 交易胜率
        - win_loss_ratio: 盈亏比
        - volatility: 资产波动率
        - correlation_matrix: 多资产相关性矩阵
        - assets: 多资产列表
        
        返回: 最优仓位比例
        """
        # 防止除零错误
        if win_loss_ratio <= 0:
            return 0
            
        # 基础凯利公式: K% = W - [(1-W)/R]
        # K% = 最优仓位百分比, W = 胜率, R = 盈亏比
        simple_kelly_percentage = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # 限制在合理范围内
        simple_kelly_percentage = max(0, simple_kelly_percentage)
        
        # 应用凯利分数(通常使用半凯利或更低)
        fractionalized_kelly = simple_kelly_percentage * self.kelly_fraction
        
        # 如果没有额外的风险信息，返回简单凯利结果
        if volatility is None or volatility <= 0:
            self.logger.info(f"简单凯利计算: 全凯利={simple_kelly_percentage:.4f}, 分数凯利({self.kelly_fraction})={fractionalized_kelly:.4f}")
            return min(fractionalized_kelly, self.max_kelly_cap)
        
        # 高级凯利: 考虑波动率
        volatility_adjustment = 1.0 - (volatility / 0.5)  # 随波动率增加而减少仓位(0.5为参考波动率)
        volatility_adjustment = max(0.2, min(1.2, volatility_adjustment))  # 限制调整范围
        
        # 考虑多资产相关性
        correlation_adjustment = 1.0  # 默认无调整
        
        if self.multi_asset_kelly_correlation and correlation_matrix is not None and assets is not None and len(assets) > 1:
            # 基于相关性矩阵计算多样化效益
            avg_correlation = 0.0
            count = 0
            
            # 计算平均相关性
            for i in range(len(assets)):
                for j in range(i+1, len(assets)):
                    if assets[i] in correlation_matrix and assets[j] in correlation_matrix[assets[i]]:
                        avg_correlation += correlation_matrix[assets[i]][assets[j]]
                        count += 1
            
            if count > 0:
                avg_correlation /= count
                
                # 相关性越低，允许的总风险敞口越大
                correlation_adjustment = 1.0 + (0.5 * (1.0 - avg_correlation))
                
        # 综合调整后的凯利
        adjusted_kelly = fractionalized_kelly * volatility_adjustment * correlation_adjustment
        
        # 应用最大上限
        final_kelly = min(adjusted_kelly, self.max_kelly_cap)
        
        self.logger.info(
            f"高级凯利计算: 基础={simple_kelly_percentage:.4f}, " +
            f"分数凯利={fractionalized_kelly:.4f}, " +
            f"波动率调整={volatility_adjustment:.2f}, " +
            f"相关性调整={correlation_adjustment:.2f}, " +
            f"最终配置={final_kelly:.4f}"
        )
        
        return final_kelly
    
    def calculate_risk_parity_allocation(self, assets_data, risk_free_rate=0.0, target_volatility=None):
        """
        计算风险平价配置权重
        
        参数:
        - assets_data: 包含各资产历史收益率的字典
        - risk_free_rate: 无风险利率
        - target_volatility: 目标波动率(可选)
        
        返回: 风险平价权重
        """
        if not assets_data or not isinstance(assets_data, dict):
            self.logger.warning("无有效资产数据，无法计算风险平价权重")
            return {}
            
        assets = list(assets_data.keys())
        n_assets = len(assets)
        
        if n_assets == 0:
            return {}
            
        # 准备收益率数据和协方差矩阵
        returns_matrix = []
        for asset in assets:
            if 'returns' in assets_data[asset] and isinstance(assets_data[asset]['returns'], list):
                returns_matrix.append(assets_data[asset]['returns'])
            else:
                self.logger.warning(f"资产 {asset} 缺乏有效的收益率数据")
                returns_matrix.append([0.0] * 20)  # 使用默认值
                
        # 转换为numpy数组
        returns_array = np.array(returns_matrix)
        
        # 计算协方差矩阵
        try:
            cov_matrix = np.cov(returns_array)
        except Exception as e:
            self.logger.error(f"计算协方差矩阵失败: {e}")
            # 返回均值分配
            return {asset: 1.0/n_assets for asset in assets}
            
        # 确保协方差矩阵可逆
        if np.linalg.det(cov_matrix) < 1e-10:
            self.logger.warning("协方差矩阵接近奇异，添加微小扰动")
            cov_matrix = cov_matrix + np.eye(n_assets) * 1e-6
            
        # 使用风险平价优化
        # 目标函数：最小化风险贡献偏差
        def risk_parity_objective(weights, cov_matrix):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_vol
            target_risk = portfolio_vol / n_assets  # 平等风险贡献
            return np.sum((risk_contributions - target_risk)**2)
            
        # 约束条件
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})  # 权重和为1
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # 权重在0和1之间
        
        # 初始猜测 - 等权重
        equal_weights = np.ones(n_assets) / n_assets
        
        # 优化求解
        try:
            result = optimize.minimize(
                risk_parity_objective, 
                equal_weights,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            risk_parity_weights = result['x']
            
            # 如果指定了目标波动率，进行缩放
            if target_volatility is not None:
                current_vol = np.sqrt(np.dot(risk_parity_weights.T, np.dot(cov_matrix, risk_parity_weights)))
                scaling_factor = target_volatility / current_vol
                
                # 应用杠杆或减仓
                risk_parity_weights = risk_parity_weights * scaling_factor
                
                # 如果需要杠杆，确保不超过最大杠杆
                if scaling_factor > 1.0:
                    if scaling_factor > self.max_leverage:
                        scaling_factor = self.max_leverage
                        risk_parity_weights = risk_parity_weights * (self.max_leverage / scaling_factor)
                        
                self.logger.info(f"应用波动率目标: 当前={current_vol:.4f}, 目标={target_volatility:.4f}, 缩放因子={scaling_factor:.2f}")
                
        except Exception as e:
            self.logger.error(f"风险平价优化失败: {e}")
            # 返回均值分配
            return {asset: 1.0/n_assets for asset in assets}
            
        # 转换为字典格式
        allocation = {}
        for i, asset in enumerate(assets):
            allocation[asset] = float(risk_parity_weights[i])
            
        # 记录风险贡献
        self._update_risk_contribution(allocation, cov_matrix, assets)
        
        return allocation
    
    def _update_risk_contribution(self, weights, cov_matrix, assets):
        """更新风险贡献矩阵"""
        weights_array = np.array([weights[asset] for asset in assets])
        portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        
        if portfolio_vol > 0:
            risk_contributions = {}
            for i, asset in enumerate(assets):
                asset_contribution = weights_array[i] * np.dot(cov_matrix[i], weights_array) / portfolio_vol
                risk_contributions[asset] = float(asset_contribution)
                
            self.risk_contribution_matrix = {
                'weights': weights.copy(),
                'contributions': risk_contributions,
                'total_risk': float(portfolio_vol),
                'timestamp': datetime.now()
            }
        
    def run_monte_carlo_position_test(self, position_size, expected_return, volatility, 
                                      skewness=0, kurtosis=3, confidence_level=0.95):
        """
        通过蒙特卡洛模拟验证仓位大小
        
        参数:
        - position_size: 初始仓位大小
        - expected_return: 预期收益率
        - volatility: 波动率
        - skewness: 偏度
        - kurtosis: 峰度
        - confidence_level: 置信水平
        
        返回: 模拟结果和调整后的仓位
        """
        self.logger.info(f"执行蒙特卡洛仓位测试: 初始仓位={position_size:.4f}, 目标置信度={confidence_level:.2f}")
        
        # 生成蒙特卡洛模拟的收益率分布
        np.random.seed(int(datetime.now().timestamp()))
        
        # 使用Johnson SU分布来支持偏度和峰度
        sim_returns = []
        try:
            # 如果有偏度或超额峰度，使用更复杂的分布
            if abs(skewness) > 0.1 or abs(kurtosis - 3) > 0.1:
                # 估计Johnson SU分布参数
                gamma, delta, xi, lam = stats.johnsonsu.fit([0] * 10)  # 初始拟合
                
                # 生成模拟收益率
                sim_returns = stats.johnsonsu.rvs(
                    gamma, delta, loc=expected_return, scale=volatility, 
                    size=self.monte_carlo_trials
                )
            else:
                # 使用正态分布
                sim_returns = np.random.normal(
                    expected_return, volatility, self.monte_carlo_trials
                )
        except Exception as e:
            self.logger.warning(f"高级分布模拟失败: {e}，回退到正态分布")
            sim_returns = np.random.normal(
                expected_return, volatility, self.monte_carlo_trials
            )
        
        # 计算累积收益率
        cumulative_returns = position_size * sim_returns
        
        # 计算各种风险指标
        var = np.percentile(cumulative_returns, (1 - confidence_level) * 100)
        expected_shortfall = np.mean(cumulative_returns[cumulative_returns <= var])
        max_loss = np.min(cumulative_returns)
        probability_of_loss = np.mean(cumulative_returns < 0)
        
        # 根据风险指标调整仓位
        adjusted_position = position_size
        
        # 如果最大损失超过可接受范围，调整仓位
        if abs(max_loss) > self.drawdown_limit:
            adjustment_factor = self.drawdown_limit / abs(max_loss)
            adjusted_position = position_size * adjustment_factor
            self.logger.info(f"根据最大损失调整仓位: {position_size:.4f} -> {adjusted_position:.4f} (调整因子: {adjustment_factor:.2f})")
            
        # 返回模拟结果
        result = {
            'original_position': position_size,
            'adjusted_position': adjusted_position,
            'var': var,
            'expected_shortfall': expected_shortfall,
            'max_loss': max_loss,
            'probability_of_loss': probability_of_loss,
            'num_simulations': self.monte_carlo_trials
        }
        
        return result
    
    def calculate_adaptive_leverage(self, portfolio_volatility, market_stress_index=0.5):
        """
        计算自适应杠杆倍数
        
        参数:
        - portfolio_volatility: 当前组合波动率
        - market_stress_index: 市场压力指数(0-1)，1表示高压力
        
        返回: 适应性杠杆倍数
        """
        if not self.volatility_scaling_enabled:
            return 1.0  # 不启用自适应杠杆
            
        if portfolio_volatility <= 0:
            return 1.0  # 防止除零错误
            
        # 基础杠杆 = 目标波动率 / 当前波动率
        base_leverage = self.target_portfolio_volatility / portfolio_volatility
        
        # 市场压力调整
        stress_adjustment = 1.0 - (market_stress_index * 0.7)  # 高压力时最多减少70%杠杆
        
        # 应用调整
        adjusted_leverage = base_leverage * stress_adjustment
        
        # 确保在合理范围内
        adjusted_leverage = max(0.1, min(self.max_leverage, adjusted_leverage))
        
        self.logger.info(
            f"自适应杠杆计算: 当前波动率={portfolio_volatility:.4f}, " +
            f"基础杠杆={base_leverage:.2f}, " +
            f"压力调整={stress_adjustment:.2f}, " +
            f"最终杠杆={adjusted_leverage:.2f}"
        )
        
        return adjusted_leverage
    
    def calculate_optimal_allocation(self, assets_data, risk_tolerance, method='risk_parity'):
        """
        计算多资产最优配置比例，支持多种优化方法
        
        参数:
        - assets_data: 资产数据字典
        - risk_tolerance: 风险容忍度(0-1)
        - method: 优化方法('risk_parity', 'kelly', 'equal_weight')
        
        返回: 最优分配比例
        """
        asset_count = len(assets_data)
        if asset_count == 0:
            return {}
            
        # 根据指定方法计算配置
        if method == 'risk_parity':
            target_vol = risk_tolerance * self.volatility_scaling_factor
            return self.calculate_risk_parity_allocation(assets_data, target_volatility=target_vol)
            
        elif method == 'kelly':
            # 为每个资产计算凯利比例
            kelly_allocations = {}
            
            # 提取相关性矩阵
            correlation_matrix = {}
            for asset1 in assets_data:
                correlation_matrix[asset1] = {}
                for asset2 in assets_data:
                    if asset1 == asset2:
                        correlation_matrix[asset1][asset2] = 1.0
                    else:
                        # 假设我们已经有相关性数据
                        correlation_matrix[asset1][asset2] = assets_data[asset1].get(
                            'correlation', {}).get(asset2, 0.5)  # 默认中等相关性
            
            # 计算每个资产的凯利配置
            for asset in assets_data:
                win_rate = assets_data[asset].get('win_rate', 0.5)
                win_loss_ratio = assets_data[asset].get('win_loss_ratio', 1.0)
                volatility = assets_data[asset].get('volatility', 0.2)
                
                kelly_allocations[asset] = self.apply_kelly_criterion(
                    win_rate, win_loss_ratio, volatility, correlation_matrix, list(assets_data.keys())
                )
                
            # 归一化凯利权重
            total_allocation = sum(kelly_allocations.values())
            if total_allocation > 0:
                kelly_allocations = {k: v/total_allocation for k, v in kelly_allocations.items()}
                
            # 应用风险容忍度
            for asset in kelly_allocations:
                kelly_allocations[asset] *= risk_tolerance
                
            return kelly_allocations
                
        else:  # 默认等权重
            equal_weight = 1.0 / asset_count
            allocations = {asset: equal_weight * risk_tolerance for asset in assets_data}
            return allocations
    
    def adjust_for_correlation(self, positions, correlation_matrix):
        """
        根据资产相关性调整仓位
        """
        adjusted_positions = positions.copy()
        
        # 模拟相关性调整
        # 相关性高的资产，总体仓位应减少
        total_correlation = 0
        for sym1 in positions:
            for sym2 in positions:
                if sym1 != sym2:
                    corr = correlation_matrix.get((sym1, sym2), 0)
                    total_correlation += corr
                    
        # 简单调整因子
        if len(positions) > 1:
            avg_correlation = total_correlation / (len(positions) * (len(positions) - 1))
            adjustment_factor = 1.0 - (avg_correlation * 0.5)  # 高相关降低总体仓位
            
            for symbol in adjusted_positions:
                adjusted_positions[symbol] *= adjustment_factor
                
            self.logger.info(f"相关性调整因子: {adjustment_factor:.4f}")
            
        return adjusted_positions
    
    def track_position(self, symbol, quantity, entry_price, stop_loss, take_profit):
        """
        记录新建仓位
        """
        if symbol in self.current_positions:
            self.logger.warning(f"更新已有仓位 {symbol}")
            
        position = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'unrealized_pnl': 0.0,
            'unrealized_pnl_percent': 0.0
        }
        
        self.current_positions[symbol] = position
        self.position_history.append({
            'action': 'OPEN',
            'timestamp': datetime.now(),
            'position': position.copy()
        })
        
        self.logger.info(f"新建仓位: {symbol}, 数量: {quantity}, 入场价: {entry_price}")
        
        return position
    
    def update_position(self, symbol, current_price):
        """
        更新仓位状态
        """
        if symbol not in self.current_positions:
            self.logger.warning(f"尝试更新不存在的仓位: {symbol}")
            return None
            
        position = self.current_positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        # 计算未实现盈亏
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_percent = (current_price - entry_price) / entry_price
        
        # 更新仓位信息
        position['current_price'] = current_price
        position['unrealized_pnl'] = unrealized_pnl
        position['unrealized_pnl_percent'] = unrealized_pnl_percent
        position['last_update_time'] = datetime.now()
        
        return position
    
    def close_position(self, symbol, exit_price):
        """
        平仓并记录结果
        """
        if symbol not in self.current_positions:
            self.logger.warning(f"尝试平不存在的仓位: {symbol}")
            return None
            
        position = self.current_positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        # 计算已实现盈亏
        realized_pnl = (exit_price - entry_price) * quantity
        realized_pnl_percent = (exit_price - entry_price) / entry_price
        
        # 记录平仓
        closed_position = position.copy()
        closed_position['exit_price'] = exit_price
        closed_position['exit_time'] = datetime.now()
        closed_position['realized_pnl'] = realized_pnl
        closed_position['realized_pnl_percent'] = realized_pnl_percent
        closed_position['holding_period'] = (closed_position['exit_time'] - 
                                           position['entry_time']).total_seconds() / 3600  # 小时
        
        # 添加到历史记录
        self.position_history.append({
            'action': 'CLOSE',
            'timestamp': datetime.now(),
            'position': closed_position
        })
        
        # 从当前仓位移除
        del self.current_positions[symbol]
        
        self.logger.info(f"平仓: {symbol}, 出场价: {exit_price}, 盈亏: {realized_pnl:.2f} ({realized_pnl_percent:.2%})")
        
        return closed_position 