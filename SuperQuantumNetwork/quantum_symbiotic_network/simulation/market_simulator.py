#!/usr/bin/env python3
"""
市场模拟器 - 用于回测交易策略
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketSimulator")

class MarketSimulator:
    """市场模拟器，用于回测交易策略"""
    
    def __init__(self, config=None):
        """初始化市场模拟器
        
        Args:
            config (dict): 配置参数
        """
        # 默认配置
        default_config = {
            "initial_capital": 100000.0,  # 初始资金
            "transaction_fee_rate": 0.0003,  # 交易费率
            "slippage_rate": 0.0005,  # 滑点
            "data_granularity": "daily",  # 数据粒度
            "risk_free_rate": 0.02 / 252,  # 无风险日收益率
        }
        
        # 合并配置
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # 初始化状态
        self.reset()
        
    def reset(self):
        """重置模拟器状态"""
        self.cash = self.config["initial_capital"]
        self.positions = {}  # 持仓 {symbol: quantity}
        self.position_values = {}  # 持仓价值 {symbol: value}
        self.trades = []  # 交易记录
        self.current_idx = 0  # 当前时间索引
        self.current_date = None  # 当前日期
        self.data = None  # 市场数据
        self.symbols = []  # 交易的股票代码
        self.days = 0  # 交易日数
        self.daily_values = []  # 每日总资产价值
        self.daily_returns = []  # 每日收益率
        
    def load_real_data(self, market_data):
        """加载真实市场数据
        
        Args:
            market_data (dict): 市场数据，包含股票和指数
        """
        self.data = market_data
        self.symbols = list(market_data["stocks"].keys())
        
        # 确定交易日数
        first_symbol = self.symbols[0]
        self.days = len(market_data["stocks"][first_symbol])
        
        # 按日期排序
        for symbol in self.symbols:
            self.data["stocks"][symbol] = self.data["stocks"][symbol].sort_values("trade_date")
            
        logger.info(f"加载了 {len(self.symbols)} 只股票的数据，共 {self.days} 个交易日")
        
    def _get_current_price(self, symbol):
        """获取当前价格
        
        Args:
            symbol (str): 股票代码
            
        Returns:
            float: 当前收盘价
        """
        if symbol not in self.data["stocks"]:
            logger.warning(f"股票 {symbol} 不在数据集中")
            return None
            
        df = self.data["stocks"][symbol]
        
        if self.current_idx >= len(df):
            logger.warning(f"当前索引 {self.current_idx} 超出数据范围")
            return None
            
        return float(df.iloc[self.current_idx]["close"])
        
    def _calculate_transaction_cost(self, price, quantity):
        """计算交易成本
        
        Args:
            price (float): 价格
            quantity (float): 数量
            
        Returns:
            float: 交易成本
        """
        fee = price * quantity * self.config["transaction_fee_rate"]
        slippage = price * quantity * self.config["slippage_rate"]
        return fee + slippage
        
    def _update_position_values(self):
        """更新持仓价值"""
        total_value = self.cash
        
        for symbol, quantity in self.positions.items():
            price = self._get_current_price(symbol)
            if price is not None:
                value = price * quantity
                self.position_values[symbol] = value
                total_value += value
                
        self.daily_values.append(total_value)
        
        # 计算日收益率
        if len(self.daily_values) > 1:
            daily_return = self.daily_values[-1] / self.daily_values[-2] - 1
        else:
            daily_return = 0
            
        self.daily_returns.append(daily_return)
        
    def step(self):
        """前进一个交易日
        
        Returns:
            tuple: (market_data, is_done)
                market_data (dict): 当天市场数据
                is_done (bool): 是否结束
        """
        if self.current_idx >= self.days:
            return None, True
            
        # 获取当前日期
        first_symbol = self.symbols[0]
        self.current_date = self.data["stocks"][first_symbol].iloc[self.current_idx]["trade_date"]
        
        # 构建当天市场数据
        daily_data = {}
        
        for symbol in self.symbols:
            df = self.data["stocks"][symbol]
            if self.current_idx < len(df):
                row = df.iloc[self.current_idx]
                
                # 准备单个股票数据
                stock_data = {
                    "symbol": symbol,
                    "date": str(row["trade_date"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["vol"]),
                    "history": []
                }
                
                # 添加技术指标
                for indicator in ["ma5", "ma10", "ma20", "rsi14", "macd", "signal"]:
                    if indicator in row and not pd.isna(row[indicator]):
                        # 移除14/5/10/20等后缀
                        key = indicator.rstrip("1234567890")
                        stock_data[key] = float(row[indicator])
                
                # 添加历史数据
                history_start = max(0, self.current_idx - 20)
                history = df.iloc[history_start:self.current_idx]
                
                for i in range(len(history)):
                    hist_row = history.iloc[i]
                    history_data = {
                        "date": str(hist_row["trade_date"]),
                        "open": float(hist_row["open"]),
                        "high": float(hist_row["high"]),
                        "low": float(hist_row["low"]),
                        "close": float(hist_row["close"]),
                        "volume": float(hist_row["vol"])
                    }
                    stock_data["history"].append(history_data)
                    
                daily_data[symbol] = stock_data
                
        # 更新持仓价值
        self._update_position_values()
        
        # 增加索引
        self.current_idx += 1
        
        return daily_data, self.current_idx >= self.days
        
    def execute_action(self, action):
        """执行交易动作
        
        Args:
            action (dict): 交易动作
                {
                    "action": "buy/sell/hold",
                    "symbol": "股票代码",
                    "size": 仓位大小(0-1)
                }
                
        Returns:
            dict: 执行结果
        """
        action_type = action.get("action")
        symbol = action.get("symbol")
        size = action.get("size", 0.1)  # 默认仓位大小为10%
        
        # 检查动作类型
        if action_type not in ["buy", "sell", "hold"]:
            return {"status": "failed", "message": f"不支持的动作类型: {action_type}"}
            
        # 持仓动作
        if action_type == "hold":
            return {"status": "success", "message": "保持当前持仓"}
            
        # 检查股票是否在可交易列表中
        if symbol not in self.symbols:
            return {"status": "failed", "message": f"股票 {symbol} 不在可交易列表中"}
            
        # 获取当前价格
        price = self._get_current_price(symbol)
        if price is None:
            return {"status": "failed", "message": f"无法获取股票 {symbol} 的价格"}
            
        # 计算当前总资产
        total_assets = self.cash
        for sym, qty in self.positions.items():
            sym_price = self._get_current_price(sym)
            if sym_price is not None:
                total_assets += sym_price * qty
                
        # 计算目标交易金额
        target_amount = total_assets * size
        
        # 买入操作
        if action_type == "buy":
            # 检查资金是否足够
            if target_amount > self.cash:
                target_amount = self.cash  # 限制为可用资金
                
            # 计算买入数量
            quantity = target_amount / price
            quantity = int(quantity * 100) / 100  # 保留两位小数
            
            # 计算交易成本
            cost = self._calculate_transaction_cost(price, quantity)
            
            # 检查资金是否足够支付交易成本
            total_cost = price * quantity + cost
            if total_cost > self.cash:
                quantity = (self.cash - cost) / price
                quantity = int(quantity * 100) / 100  # 保留两位小数
                total_cost = price * quantity + cost
                
            # 执行买入
            if quantity > 0:
                self.cash -= total_cost
                
                # 更新持仓
                if symbol in self.positions:
                    self.positions[symbol] += quantity
                else:
                    self.positions[symbol] = quantity
                    
                # 记录交易
                trade = {
                    "date": self.current_date,
                    "action": "buy",
                    "symbol": symbol,
                    "price": price,
                    "quantity": quantity,
                    "value": price * quantity,
                    "cost": cost,
                    "cash_after": self.cash
                }
                self.trades.append(trade)
                
                return {"status": "success", "message": f"买入 {symbol}: {quantity} 股，价格: {price:.2f}，总额: {total_cost:.2f}"}
            else:
                return {"status": "failed", "message": "资金不足以执行交易"}
                
        # 卖出操作
        elif action_type == "sell":
            # 检查是否持有该股票
            if symbol not in self.positions or self.positions[symbol] <= 0:
                return {"status": "failed", "message": f"未持有股票 {symbol}"}
                
            # 计算卖出数量
            current_value = self.positions[symbol] * price
            sell_ratio = min(1.0, target_amount / current_value)
            quantity = self.positions[symbol] * sell_ratio
            quantity = int(quantity * 100) / 100  # 保留两位小数
            
            # 计算交易成本
            cost = self._calculate_transaction_cost(price, quantity)
            
            # 执行卖出
            self.positions[symbol] -= quantity
            sell_value = price * quantity - cost
            self.cash += sell_value
            
            # 如果持仓为0，则移除
            if self.positions[symbol] <= 0:
                del self.positions[symbol]
                
            # 记录交易
            trade = {
                "date": self.current_date,
                "action": "sell",
                "symbol": symbol,
                "price": price,
                "quantity": quantity,
                "value": price * quantity,
                "cost": cost,
                "cash_after": self.cash
            }
            self.trades.append(trade)
            
            return {"status": "success", "message": f"卖出 {symbol}: {quantity} 股，价格: {price:.2f}，金额: {sell_value:.2f}"}
            
    def calculate_performance(self):
        """计算回测性能
        
        Returns:
            dict: 性能指标
        """
        # 计算最终资产价值
        final_value = self.cash
        for symbol, quantity in self.positions.items():
            price = self._get_current_price(symbol)
            if price is not None:
                final_value += price * quantity
                
        # 计算总收益率
        total_return = final_value / self.config["initial_capital"] - 1
        
        # 计算年化收益率
        annual_return = (1 + total_return) ** (252 / self.days) - 1
        
        # 计算最大回撤
        max_drawdown = 0
        peak_value = 0
        
        for value in self.daily_values:
            if value > peak_value:
                peak_value = value
            else:
                drawdown = (peak_value - value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)
                
        # 计算夏普比率
        daily_returns = np.array(self.daily_returns)
        risk_free_rate = self.config["risk_free_rate"]
        excess_returns = daily_returns - risk_free_rate
        
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            sharpe = 0
            
        # 计算胜率
        win_trades = sum(1 for trade in self.trades if trade["action"] == "sell" and 
                         trade["value"] > self.trades[self.trades.index(trade) - 1]["value"])
        if len(self.trades) > 0:
            win_rate = win_trades / len(self.trades)
        else:
            win_rate = 0
            
        # 返回性能指标
        performance = {
            "initial_capital": self.config["initial_capital"],
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "trade_count": len(self.trades),
            "trades": self.trades,
            "value_history": self.daily_values,
            "return_history": self.daily_returns
        }
        
        return performance
        
    def save_performance_report(self, performance, output_path=None):
        """保存性能报告
        
        Args:
            performance (dict): 性能指标
            output_path (str): 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join("quantum_symbiotic_network", "data", "performance_report.txt")
            
        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write("====== 回测性能报告 ======\n\n")
            f.write(f"初始资金: {performance['initial_capital']:.2f}\n")
            f.write(f"最终价值: {performance['final_value']:.2f}\n")
            f.write(f"总收益率: {performance['total_return']*100:.2f}%\n")
            f.write(f"年化收益率: {performance['annual_return']*100:.2f}%\n")
            f.write(f"最大回撤: {performance['max_drawdown']*100:.2f}%\n")
            f.write(f"夏普比率: {performance['sharpe']:.2f}\n")
            f.write(f"胜率: {performance['win_rate']*100:.2f}%\n")
            f.write(f"交易次数: {performance['trade_count']}\n\n")
            
            f.write("====== 交易记录 ======\n\n")
            for i, trade in enumerate(performance['trades']):
                f.write(f"交易 {i+1}:\n")
                f.write(f"  日期: {trade['date']}\n")
                f.write(f"  动作: {trade['action']}\n")
                f.write(f"  股票: {trade['symbol']}\n")
                f.write(f"  价格: {trade['price']:.2f}\n")
                f.write(f"  数量: {trade['quantity']:.2f}\n")
                f.write(f"  价值: {trade['value']:.2f}\n")
                f.write(f"  成本: {trade['cost']:.2f}\n")
                f.write(f"  交易后现金: {trade['cash_after']:.2f}\n\n")
                
        logger.info(f"性能报告已保存到 {output_path}")
        
    def plot_performance(self, performance, save_path=None):
        """绘制性能图表
        
        Args:
            performance (dict): 性能指标
            save_path (str): 保存路径
        """
        plt.figure(figsize=(12, 10))
        
        # 绘制资产价值曲线
        plt.subplot(2, 1, 1)
        plt.plot(performance["value_history"])
        plt.title("资产价值变化")
        plt.xlabel("交易日")
        plt.ylabel("资产价值")
        plt.grid(True)
        
        # 绘制收益率曲线
        plt.subplot(2, 1, 2)
        plt.plot(np.cumsum(performance["return_history"]), label="累计收益率")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.title("累计收益率")
        plt.xlabel("交易日")
        plt.ylabel("收益率")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            logger.info(f"性能图表已保存到 {save_path}")
        else:
            plt.show() 