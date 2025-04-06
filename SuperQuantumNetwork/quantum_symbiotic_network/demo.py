#!/usr/bin/env python
"""
量子共生网络 - 演示脚本
这个脚本展示了量子共生网络的基本用法，模拟了简单的市场环境
"""

import os
import sys
import logging
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(BASE_DIR))

from quantum_symbiotic_network import QuantumSymbioticNetwork

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QuantumSymbioticDemo")

class MarketSimulator:
    """简单的市场模拟器"""
    
    def __init__(self, symbols=None, start_date=None, days=30, volatility=0.015):
        """
        初始化市场模拟器
        
        Args:
            symbols: 股票代码列表
            start_date: 起始日期
            days: 模拟天数
            volatility: 波动率
        """
        self.symbols = symbols or ["000001.SZ", "000002.SZ", "000063.SZ"]
        self.start_date = start_date or datetime.now() - timedelta(days=days)
        self.days = days
        self.volatility = volatility
        self.current_day = 0
        self.prices = self._generate_price_series()
        self.volumes = self._generate_volume_series()
        
    def _generate_price_series(self):
        """生成价格序列"""
        prices = {}
        
        for symbol in self.symbols:
            # 初始价格在10-100之间随机
            initial_price = np.random.uniform(10, 100)
            
            # 生成随机价格序列
            daily_returns = np.random.normal(0.0005, self.volatility, self.days)
            price_series = [initial_price]
            
            for ret in daily_returns:
                price_series.append(price_series[-1] * (1 + ret))
                
            prices[symbol] = price_series
            
        return prices
        
    def _generate_volume_series(self):
        """生成成交量序列"""
        volumes = {}
        
        for symbol in self.symbols:
            # 初始成交量在10万-100万之间随机
            base_volume = np.random.uniform(100000, 1000000)
            
            # 生成随机成交量序列
            volume_series = []
            for i in range(self.days + 1):
                # 成交量在基础成交量的50%-150%之间波动
                volume = base_volume * np.random.uniform(0.5, 1.5)
                volume_series.append(int(volume))
                
            volumes[symbol] = volume_series
            
        return volumes
        
    def get_current_data(self):
        """获取当前市场数据"""
        if self.current_day > self.days:
            return None
            
        date = self.start_date + timedelta(days=self.current_day)
        
        market_data = {
            "date": date.strftime("%Y-%m-%d"),
            "global_market": {
                "volatility": np.random.uniform(0, 0.03),
                "trend": np.random.uniform(-0.02, 0.02)
            }
        }
        
        # 添加各股票数据
        for symbol in self.symbols:
            price = self.prices[symbol][self.current_day]
            prev_price = self.prices[symbol][max(0, self.current_day - 1)]
            volume = self.volumes[symbol][self.current_day]
            
            # 计算当日OHLC (简化模拟)
            open_price = price * np.random.uniform(0.99, 1.01)
            high_price = price * np.random.uniform(1.01, 1.03)
            low_price = price * np.random.uniform(0.97, 0.99)
            close_price = price
            
            # 计算一些技术指标
            ma5 = np.mean([self.prices[symbol][max(0, self.current_day - i)] 
                         for i in range(5)]) if self.current_day >= 4 else price
            ma10 = np.mean([self.prices[symbol][max(0, self.current_day - i)] 
                          for i in range(10)]) if self.current_day >= 9 else price
            ma20 = np.mean([self.prices[symbol][max(0, self.current_day - i)] 
                          for i in range(20)]) if self.current_day >= 19 else price
            
            # 简化的RSI计算
            rsi = 50 + 25 * ((price - prev_price) / prev_price) / self.volatility
            rsi = max(0, min(100, rsi))
            
            # 简化的MACD
            macd = (ma5 - ma10) / price * 100
            
            market_data[symbol] = {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "ma5": ma5,
                "ma10": ma10,
                "ma20": ma20,
                "rsi": rsi,
                "macd": macd,
                f"{symbol}_special_0": np.random.normal(0, 1),
                f"{symbol}_special_1": np.random.normal(0, 1),
                f"{symbol}_special_2": np.random.normal(0, 1)
            }
            
        return market_data
        
    def step(self):
        """市场前进一步"""
        self.current_day += 1
        return self.get_current_data()
        
    def calculate_performance(self, actions, symbol=None):
        """
        计算策略表现
        
        Args:
            actions: 操作列表 [(day, action, price), ...]
            symbol: 计算哪个股票的表现，如果为None则计算总表现
            
        Returns:
            表现指标字典
        """
        if not actions:
            return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
            
        # 如果未指定symbol，使用第一个交易的股票
        if symbol is None and actions:
            symbol = actions[0][3]  # 假设action格式为(day, action, price, symbol)
            
        # 筛选指定股票的操作
        symbol_actions = [a for a in actions if a[3] == symbol] if symbol else actions
        
        if not symbol_actions:
            return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
            
        # 计算每日收益
        daily_returns = []
        position = 0
        cash = 10000  # 初始资金
        value_history = [cash]
        
        for day in range(self.days):
            # 查找当天的操作
            day_actions = [a for a in symbol_actions if a[0] == day]
            
            # 计算当天结束时的资产价值
            price = self.prices[symbol][day] if symbol else self.prices[self.symbols[0]][day]
            
            # 执行操作
            for _, action, size, _ in day_actions:
                if action == "buy":
                    cost = price * size
                    if cost <= cash:
                        cash -= cost
                        position += size
                elif action == "sell":
                    if position >= size:
                        cash += price * size
                        position -= size
                        
            # 当前总资产价值
            total_value = cash + position * price
            value_history.append(total_value)
            
            # 计算日收益率
            if day > 0:
                daily_return = (total_value - value_history[-2]) / value_history[-2]
                daily_returns.append(daily_return)
                
        # 计算总收益
        total_return = (value_history[-1] - value_history[0]) / value_history[0]
        
        # 计算夏普比率 (简化版，假设无风险利率为0)
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        
        # 计算最大回撤
        peak = value_history[0]
        max_drawdown = 0
        
        for value in value_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "final_value": value_history[-1],
            "value_history": value_history
        }


def run_simulation():
    """运行模拟"""
    logger.info("启动量子共生网络演示...")
    
    # 创建市场模拟器
    market = MarketSimulator(days=60, volatility=0.02)
    
    # 创建量子共生网络系统
    network = QuantumSymbioticNetwork()
    network.initialize()
    
    # 记录操作历史
    actions = []  # [(day, action, size, symbol), ...]
    
    # 运行模拟
    for day in range(market.days):
        # 获取市场数据
        market_data = market.get_current_data()
        
        # 系统决策
        result = network.step(market_data)
        decision = result["decision"]
        trade_id = result["trade_id"]
        
        # 如果有交易信号，执行交易
        if trade_id and decision["action"] in ["buy", "sell"]:
            # 等待随机时间后执行（模拟量子态保持一段时间）
            delay = np.random.uniform(0.5, 2.0)
            time.sleep(delay)
            
            # 执行交易
            execution = network.execute_trade(trade_id)
            
            if "symbol" in execution and "size" in execution:
                logger.info(f"日期: {market_data['date']}, 执行: {execution['action']}, 标的: {execution['symbol']}, 大小: {execution['size']:.2f}")
                
                # 记录操作
                actions.append((day, execution["action"], execution["size"], execution["symbol"]))
                
        # 计算表现并提供反馈
        performance = {}
        domain_performances = {}
        
        for symbol in market.symbols:
            symbol_perf = market.calculate_performance(actions, symbol)
            domain_performances[symbol[:3]] = symbol_perf["total_return"]  # 使用前缀作为领域
            
        overall_perf = market.calculate_performance(actions)
        
        # 创建反馈
        feedback = {
            "performance": overall_perf["total_return"],
            "metrics": overall_perf,
            "domain_performances": domain_performances,
            "available_features": ["open", "high", "low", "close", "volume", "ma5", "ma10", "ma20", "rsi", "macd"]
        }
        
        # 提供反馈
        network.provide_feedback(feedback)
        
        # 下一天
        market.step()
        
        # 每10天打印一次表现
        if day % 10 == 9:
            logger.info(f"第 {day+1} 天, 总收益: {overall_perf['total_return']:.2%}, 夏普: {overall_perf['sharpe']:.2f}, 最大回撤: {overall_perf['max_drawdown']:.2%}")
            
    # 最终表现
    final_performance = market.calculate_performance(actions)
    logger.info("模拟结束")
    logger.info(f"总收益: {final_performance['total_return']:.2%}")
    logger.info(f"夏普比率: {final_performance['sharpe']:.2f}")
    logger.info(f"最大回撤: {final_performance['max_drawdown']:.2%}")
    
    # 绘制资产价值变化
    plt.figure(figsize=(12, 6))
    
    # 确保value_history存在，如果不存在则创建初始值为起始资金的列表
    if 'value_history' not in final_performance:
        # 创建一个平坦的价值历史（没有交易时价值不变）
        final_performance['value_history'] = [10000] * (market.days + 1)
        final_performance['final_value'] = 10000
    
    plt.plot(final_performance["value_history"])
    plt.title("资产价值变化")
    plt.xlabel("天数")
    plt.ylabel("价值")
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(BASE_DIR, "quantum_symbiotic_network", "data", "performance.png")
    plt.savefig(plot_path)
    logger.info(f"性能图表已保存到 {plot_path}")
    
    return final_performance


if __name__ == "__main__":
    performance = run_simulation()
    # 不显示图表窗口，只保存图片
    plt.close() 