#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 数据控制器
管理数据获取和预处理，为GUI提供数据支持
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 导入数据源
try:
    from quantum_symbiotic_network.data_sources import TushareDataSource
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    logging.warning("无法导入TushareDataSource，将使用模拟数据")


class DataController:
    """数据控制器，负责管理系统数据"""
    
    def __init__(self):
        """初始化数据控制器"""
        self.logger = logging.getLogger("DataController")
        self.data_source = None
        self.config = self._load_config()
        self.cache = {}
    
    def _load_config(self):
        """加载配置"""
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def initialize(self):
        """初始化数据源"""
        try:
            if TUSHARE_AVAILABLE:
                token = self.config.get("tushare_token", "")
                self.data_source = TushareDataSource(token=token)
                self.logger.info("数据源初始化成功")
                return True
            else:
                self.logger.warning("使用模拟数据源")
                return self._initialize_mock_data_source()
        except Exception as e:
            self.logger.error(f"初始化数据源失败: {e}")
            return False
    
    def _initialize_mock_data_source(self):
        """初始化模拟数据源"""
        # 创建模拟数据结构
        self.mock_data = {
            "market_status": {
                "status": "开盘",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "stocks": {},
            "indices": {}
        }
        
        # 创建模拟股票数据
        stock_codes = [
            "600000", "600036", "601398", "600519", "600276", 
            "601318", "600887", "601857", "601288", "600030"
        ]
        stock_names = [
            "浦发银行", "招商银行", "工商银行", "贵州茅台", "恒瑞医药",
            "平安保险", "伊利股份", "中国石油", "农业银行", "中信证券"
        ]
        
        # 为每只股票生成随机价格数据
        for code, name in zip(stock_codes, stock_names):
            base_price = np.random.uniform(10, 100)
            days = 100
            prices = [base_price]
            for i in range(1, days):
                change = np.random.normal(0, 0.02)  # 每日涨跌幅服从正态分布
                prices.append(prices[-1] * (1 + change))
            
            current_price = prices[-1]
            prev_close = prices[-2]
            change = (current_price - prev_close) / prev_close
            
            self.mock_data["stocks"][code] = {
                "code": code,
                "name": name,
                "price": current_price,
                "prev_close": prev_close,
                "change": change,
                "volume": np.random.uniform(1000000, 10000000),
                "turnover": current_price * np.random.uniform(1000000, 10000000),
                "history": {
                    "dates": [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)],
                    "prices": prices,
                    "volumes": [np.random.uniform(1000000, 10000000) for _ in range(days)]
                }
            }
        
        # 创建模拟指数数据
        index_codes = ["000001.SH", "399001.SZ", "399006.SZ", "000688.SH"]
        index_names = ["上证指数", "深证成指", "创业板指", "科创50"]
        
        for code, name in zip(index_codes, index_names):
            base_price = np.random.uniform(2000, 4000) if "000001" in code else np.random.uniform(8000, 12000)
            days = 100
            prices = [base_price]
            for i in range(1, days):
                change = np.random.normal(0, 0.01)  # 每日涨跌幅服从正态分布
                prices.append(prices[-1] * (1 + change))
            
            current_price = prices[-1]
            prev_close = prices[-2]
            change = (current_price - prev_close) / prev_close
            
            self.mock_data["indices"][code] = {
                "code": code,
                "name": name,
                "price": current_price,
                "prev_close": prev_close,
                "change": change,
                "history": {
                    "dates": [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)],
                    "prices": prices
                }
            }
        
        self.logger.info("模拟数据源初始化成功")
        return True
    
    def get_market_status(self):
        """获取市场状态"""
        try:
            if self.data_source:
                return self.data_source.get_market_status()
            else:
                return self.mock_data["market_status"]
        except Exception as e:
            self.logger.error(f"获取市场状态失败: {e}")
            return {"status": "未知", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    def get_stock_data(self, code):
        """获取股票数据"""
        try:
            if self.data_source:
                return self.data_source.get_stock_data(code)
            else:
                return self.mock_data["stocks"].get(code, {})
        except Exception as e:
            self.logger.error(f"获取股票数据失败 ({code}): {e}")
            return {}
    
    def get_index_data(self, code=None):
        """获取指数数据"""
        try:
            if self.data_source:
                return self.data_source.get_index_data(code)
            else:
                if code:
                    return self.mock_data["indices"].get(code, {})
                else:
                    return self.mock_data["indices"]
        except Exception as e:
            self.logger.error(f"获取指数数据失败: {e}")
            return {}
    
    def get_recommended_stocks(self):
        """获取推荐股票列表"""
        try:
            if self.data_source:
                return self.data_source.get_recommended_stocks()
            else:
                # 从模拟数据中选择几只股票作为推荐
                stocks = list(self.mock_data["stocks"].values())
                # 添加推荐度
                for stock in stocks:
                    stock["recommendation"] = np.random.uniform(0.6, 0.95)
                return stocks
        except Exception as e:
            self.logger.error(f"获取推荐股票失败: {e}")
            return []
    
    def get_historical_data(self, code, start_date=None, end_date=None):
        """获取历史数据"""
        try:
            if self.data_source:
                return self.data_source.get_historical_data(code, start_date, end_date)
            else:
                stock_data = self.mock_data["stocks"].get(code)
                if not stock_data:
                    index_data = self.mock_data["indices"].get(code)
                    if index_data:
                        history = index_data.get("history", {})
                        return {
                            "dates": history.get("dates", []),
                            "prices": history.get("prices", [])
                        }
                else:
                    history = stock_data.get("history", {})
                    return {
                        "dates": history.get("dates", []),
                        "prices": history.get("prices", []),
                        "volumes": history.get("volumes", [])
                    }
                return {}
        except Exception as e:
            self.logger.error(f"获取历史数据失败 ({code}): {e}")
            return {}
    
    def search_stocks(self, keyword):
        """搜索股票"""
        try:
            if self.data_source:
                return self.data_source.search_stocks(keyword)
            else:
                results = []
                for code, stock in self.mock_data["stocks"].items():
                    if keyword in code or keyword in stock.get("name", ""):
                        results.append({
                            "code": code,
                            "name": stock.get("name", ""),
                            "price": stock.get("price", 0.0),
                            "change": stock.get("change", 0.0)
                        })
                return results
        except Exception as e:
            self.logger.error(f"搜索股票失败 ({keyword}): {e}")
            return []
    
    def get_account_data(self):
        """获取账户数据"""
        # 模拟账户数据
        return {
            "total_asset": 1000000.0,
            "available_cash": 500000.0,
            "market_value": 500000.0,
            "daily_profit": 50000.0,
            "daily_profit_pct": 0.05,
            "total_profit": 100000.0,
            "total_profit_pct": 0.1,
            "max_drawdown": 0.052,
            "sharpe": 2.35,
            "volatility": 0.158,
            "var": 25000.0
        }
    
    def get_positions(self):
        """获取持仓数据"""
        # 模拟持仓数据
        positions = []
        for i, (code, stock) in enumerate(list(self.mock_data["stocks"].items())[:5]):
            quantity = np.random.randint(1000, 10000) // 100 * 100
            cost = stock["price"] * (1 - np.random.uniform(-0.1, 0.1))
            positions.append({
                "code": code,
                "name": stock["name"],
                "quantity": quantity,
                "available": quantity,
                "cost": cost,
                "price": stock["price"],
                "profit": (stock["price"] - cost) * quantity
            })
        return positions
    
    def get_allocation_data(self):
        """获取资产配置数据"""
        # 模拟资产配置数据
        return [
            {'name': '金融', 'value': 0.25, 'color': (255, 0, 0)},
            {'name': '科技', 'value': 0.30, 'color': (0, 255, 0)},
            {'name': '医药', 'value': 0.15, 'color': (0, 0, 255)},
            {'name': '消费', 'value': 0.20, 'color': (255, 255, 0)},
            {'name': '其他', 'value': 0.10, 'color': (128, 128, 128)}
        ]
    
    def get_performance_data(self):
        """获取绩效数据"""
        # 模拟绩效数据
        days = 100
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days).tolist()
        portfolio_values = [1000000 * (1 + 0.001 * i + 0.002 * np.sin(i/10)) for i in range(days)]
        benchmark_values = [1000000 * (1 + 0.0008 * i) for i in range(days)]
        
        return {
            "dates": dates,
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "annual_return": 0.158,
            "alpha": 0.052,
            "beta": 0.85,
            "sortino": 1.95,
            "win_rate": 0.652,
            "profit_loss_ratio": 2.5
        }
    
    def get_strategy_data(self):
        """获取策略数据"""
        # 模拟策略数据
        days = 200
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days).tolist()
        
        strategy1 = [100 * (1 + 0.001 * i + 0.003 * np.sin(i/10)) for i in range(days)]
        strategy2 = [100 * (1 + 0.0015 * i + 0.002 * np.cos(i/12)) for i in range(days)]
        strategy3 = [100 * (1 + 0.0012 * i - 0.001 * np.sin(i/15)) for i in range(days)]
        benchmark = [100 * (1 + 0.0008 * i) for i in range(days)]
        
        return {
            'dates': dates,
            'strategies': {
                '量子动量策略': strategy1,
                '分形套利策略': strategy2,
                '波动跟踪策略': strategy3,
                '基准': benchmark
            }
        }
    
    def get_strategy_stats(self):
        """获取策略统计数据"""
        # 模拟策略统计数据
        return [
            {
                'name': '量子动量策略',
                'annual_return': 0.268,
                'sharpe': 2.35,
                'max_drawdown': 0.12,
                'volatility': 0.15,
                'win_rate': 0.68,
                'avg_return': 0.021
            },
            {
                'name': '分形套利策略',
                'annual_return': 0.312,
                'sharpe': 2.56,
                'max_drawdown': 0.15,
                'volatility': 0.18,
                'win_rate': 0.72,
                'avg_return': 0.025
            },
            {
                'name': '波动跟踪策略',
                'annual_return': 0.245,
                'sharpe': 2.18,
                'max_drawdown': 0.10,
                'volatility': 0.13,
                'win_rate': 0.65,
                'avg_return': 0.019
            }
        ]
    
    def get_correlation_data(self):
        """获取相关性数据"""
        # 模拟相关性数据
        stocks = ["工商银行", "茅台", "腾讯", "阿里巴巴", "平安保险", "中国石油", "中国移动", "恒瑞医药", "格力电器", "万科A"]
        n = len(stocks)
        
        # 创建相关性矩阵
        correlation_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # 生成一个-0.5到0.9之间的随机相关系数
                    correlation_matrix[i, j] = np.random.uniform(-0.5, 0.9)
                    correlation_matrix[j, i] = correlation_matrix[i, j]  # 确保对称
        
        return {
            'stocks': stocks,
            'matrix': correlation_matrix
        }
    
    def get_risk_decomposition(self):
        """获取风险分解数据"""
        # 模拟风险分解数据
        return [
            {'name': '市场风险', 'value': 0.45, 'color': (255, 0, 0)},
            {'name': '特异风险', 'value': 0.25, 'color': (0, 255, 0)},
            {'name': '行业风险', 'value': 0.15, 'color': (0, 0, 255)},
            {'name': '风格风险', 'value': 0.10, 'color': (255, 255, 0)},
            {'name': '其他风险', 'value': 0.05, 'color': (128, 128, 128)}
        ]
    
    def get_risk_metrics(self):
        """获取风险指标数据"""
        # 模拟风险指标数据
        return {
            'var': 0.025,  # 95% VaR
            'cvar': 0.035,  # 95% CVaR
            'volatility': 0.15,  # 年化波动率
            'max_drawdown': 0.12,  # 最大回撤
            'downside_risk': 0.08,  # 下行风险
            'beta': 0.85,  # Beta
            'tracking_error': 0.05  # 跟踪误差
        }
    
    def get_network_status(self):
        """获取网络状态"""
        # 模拟网络状态数据
        return {
            'segments': 5,
            'agents': 25,
            'learning': True,
            'evolution': 3,
            'performance': 85.2
        }
    
    def load_initial_data(self):
        """加载初始数据，用于GUI启动时"""
        # 初始化数据源
        self.initialize()
        
        # 准备返回的数据集
        data = {
            "market_status": self.get_market_status(),
            "recommended_stocks": self.get_recommended_stocks(),
            "index_data": self.get_index_data(),
            "positions": self.get_positions(),
            "orders": [],  # 空的委托列表
            "account_data": self.get_account_data(),
            "allocation_data": self.get_allocation_data(),
            "performance_data": self.get_performance_data(),
            "strategy_data": self.get_strategy_data(),
            "strategy_stats": self.get_strategy_stats(),
            "correlation_data": self.get_correlation_data(),
            "risk_decomposition": self.get_risk_decomposition(),
            "risk_metrics": self.get_risk_metrics(),
            "network_status": self.get_network_status()
        }
        
        return data 