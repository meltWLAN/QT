#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 交易控制器
管理交易相关功能，包括下单、撤单、查询等
"""

import logging
import uuid
from datetime import datetime
import numpy as np

# 导入量子网络
try:
    from quantum_symbiotic_network.network import QuantumSymbioticNetwork
    QUANTUM_NETWORK_AVAILABLE = True
except ImportError:
    QUANTUM_NETWORK_AVAILABLE = False
    logging.warning("无法导入QuantumSymbioticNetwork，将使用模拟交易功能")


class Order:
    """订单类，表示一个交易订单"""
    
    def __init__(self, code, name, direction, price, quantity, order_type="限价单"):
        """初始化订单"""
        self.order_id = str(uuid.uuid4())[:8]  # 生成订单ID
        self.code = code
        self.name = name
        self.direction = direction  # "买入" 或 "卖出"
        self.price = price
        self.quantity = quantity
        self.order_type = order_type
        self.status = "未成交"
        self.create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_time = self.create_time
        self.canceled = False
        self.filled_quantity = 0
    
    def to_dict(self):
        """转换为字典"""
        return {
            "order_id": self.order_id,
            "code": self.code,
            "name": self.name,
            "direction": self.direction,
            "price": self.price,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "time": self.create_time,
            "update_time": self.update_time,
            "canceled": self.canceled,
            "filled_quantity": self.filled_quantity
        }
    
    def cancel(self):
        """撤销订单"""
        if self.status == "已成交":
            return False
        
        self.status = "已撤单"
        self.canceled = True
        self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True
    
    def fill(self, quantity=None):
        """成交订单"""
        if self.canceled:
            return False
        
        if quantity is None:
            quantity = self.quantity
        
        self.filled_quantity += quantity
        if self.filled_quantity >= self.quantity:
            self.status = "已成交"
        else:
            self.status = "部分成交"
        
        self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True


class Position:
    """持仓类，表示一个股票持仓"""
    
    def __init__(self, code, name, quantity=0, cost=0.0):
        """初始化持仓"""
        self.code = code
        self.name = name
        self.quantity = quantity
        self.available = quantity
        self.cost = cost
        self.price = cost
        self.profit = 0.0
    
    def to_dict(self):
        """转换为字典"""
        return {
            "code": self.code,
            "name": self.name,
            "quantity": self.quantity,
            "available": self.available,
            "cost": self.cost,
            "price": self.price,
            "profit": self.profit
        }
    
    def update_price(self, price):
        """更新价格"""
        self.price = price
        self.profit = (self.price - self.cost) * self.quantity
    
    def buy(self, quantity, price):
        """买入"""
        if quantity <= 0:
            return False
            
        # 计算新的成本
        total_cost = self.cost * self.quantity + price * quantity
        self.quantity += quantity
        self.available += quantity
        
        if self.quantity > 0:
            self.cost = total_cost / self.quantity
        else:
            self.cost = 0.0
            
        self.update_price(price)
        return True
    
    def sell(self, quantity, price):
        """卖出"""
        if quantity <= 0 or quantity > self.available:
            return False
            
        self.quantity -= quantity
        self.available -= quantity
        
        # 更新利润
        profit = (price - self.cost) * quantity
        
        self.update_price(price)
        return profit


class Account:
    """账户类，表示交易账户"""
    
    def __init__(self, initial_capital=1000000.0):
        """初始化账户"""
        self.initial_capital = initial_capital
        self.available_cash = initial_capital
        self.total_asset = initial_capital
        self.market_value = 0.0
        self.positions = {}  # code -> Position
        self.daily_profit = 0.0
        self.total_profit = 0.0
        self.daily_profit_pct = 0.0
        self.total_profit_pct = 0.0
    
    def to_dict(self):
        """转换为字典"""
        return {
            "total_asset": self.total_asset,
            "available_cash": self.available_cash,
            "market_value": self.market_value,
            "daily_profit": self.daily_profit,
            "daily_profit_pct": self.daily_profit_pct,
            "total_profit": self.total_profit,
            "total_profit_pct": self.total_profit_pct,
            "max_drawdown": 0.052,  # 模拟数据
            "sharpe": 2.35,  # 模拟数据
            "volatility": 0.158,  # 模拟数据
            "var": 25000.0  # 模拟数据
        }
    
    def get_positions(self):
        """获取持仓列表"""
        return [position.to_dict() for position in self.positions.values()]
    
    def buy_stock(self, code, name, price, quantity):
        """买入股票"""
        # 检查资金是否足够
        cost = price * quantity
        if cost > self.available_cash:
            return False, "可用资金不足"
        
        # 更新资金
        self.available_cash -= cost
        
        # 更新持仓
        if code in self.positions:
            self.positions[code].buy(quantity, price)
        else:
            position = Position(code, name, quantity, price)
            self.positions[code] = position
        
        # 更新总资产
        self._update_total_asset()
        return True, "买入成功"
    
    def sell_stock(self, code, price, quantity):
        """卖出股票"""
        # 检查持仓是否足够
        if code not in self.positions or self.positions[code].available < quantity:
            return False, "可用持仓不足"
        
        # 更新持仓
        profit = self.positions[code].sell(quantity, price)
        
        # 如果卖光了，删除持仓
        if self.positions[code].quantity == 0:
            del self.positions[code]
        
        # 更新资金
        sales_amount = price * quantity
        self.available_cash += sales_amount
        
        # 更新利润
        self.daily_profit += profit
        self.total_profit += profit
        
        # 更新总资产
        self._update_total_asset()
        return True, "卖出成功"
    
    def _update_total_asset(self):
        """更新总资产"""
        market_value = 0.0
        for position in self.positions.values():
            market_value += position.price * position.quantity
        
        self.market_value = market_value
        self.total_asset = self.available_cash + market_value
        
        # 更新收益率
        if self.initial_capital > 0:
            self.daily_profit_pct = self.daily_profit / self.initial_capital
            self.total_profit_pct = self.total_profit / self.initial_capital
    
    def update_position_prices(self, price_dict):
        """更新持仓价格"""
        for code, price in price_dict.items():
            if code in self.positions:
                self.positions[code].update_price(price)
        
        # 更新总资产
        self._update_total_asset()


class TradingController:
    """交易控制器，负责管理交易功能"""
    
    def __init__(self):
        """初始化交易控制器"""
        self.logger = logging.getLogger("TradingController")
        self.account = Account()
        self.orders = []  # 所有订单
        self.active_orders = []  # 活跃订单
        self.quantum_network = None
        
        # 尝试初始化量子网络
        if QUANTUM_NETWORK_AVAILABLE:
            try:
                self.quantum_network = QuantumSymbioticNetwork()
                self.logger.info("量子网络初始化成功")
            except Exception as e:
                self.logger.error(f"量子网络初始化失败: {e}")
    
    def place_order(self, code, name, direction, price, quantity, order_type="限价单"):
        """下单"""
        try:
            # 创建订单
            order = Order(code, name, direction, price, quantity, order_type)
            
            # 检查账户状态
            if direction == "买入":
                cost = price * quantity
                if cost > self.account.available_cash:
                    order.status = "已拒绝"
                    order.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.orders.append(order)
                    return False, "可用资金不足", order.to_dict()
            elif direction == "卖出":
                if code not in self.account.positions or self.account.positions[code].available < quantity:
                    order.status = "已拒绝"
                    order.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.orders.append(order)
                    return False, "可用持仓不足", order.to_dict()
            
            # 添加订单
            self.orders.append(order)
            self.active_orders.append(order)
            
            # 模拟成交（实际环境中这里应该是异步的）
            self._process_order(order)
            
            return True, "下单成功", order.to_dict()
        except Exception as e:
            self.logger.error(f"下单失败: {e}")
            return False, f"下单失败: {str(e)}", None
    
    def _process_order(self, order):
        """处理订单（模拟成交）"""
        # 模拟成交率
        fill_prob = 0.8
        
        if np.random.random() < fill_prob:
            # 模拟成交
            if order.direction == "买入":
                success, message = self.account.buy_stock(order.code, order.name, order.price, order.quantity)
                if success:
                    order.fill()
                    if order not in self.active_orders:
                        self.active_orders.remove(order)
            elif order.direction == "卖出":
                success, message = self.account.sell_stock(order.code, order.price, order.quantity)
                if success:
                    order.fill()
                    if order in self.active_orders:
                        self.active_orders.remove(order)
    
    def cancel_order(self, order_id):
        """撤单"""
        try:
            # 查找订单
            order = None
            for o in self.orders:
                if o.order_id == order_id:
                    order = o
                    break
            
            if not order:
                return False, "订单不存在"
            
            # 如果已经成交，不能撤单
            if order.status == "已成交":
                return False, "订单已成交，无法撤单"
            
            # 撤销订单
            if order.cancel():
                if order in self.active_orders:
                    self.active_orders.remove(order)
                return True, "撤单成功"
            else:
                return False, "撤单失败"
        except Exception as e:
            self.logger.error(f"撤单失败: {e}")
            return False, f"撤单失败: {str(e)}"
    
    def get_order_list(self):
        """获取订单列表"""
        return [order.to_dict() for order in self.orders]
    
    def get_active_order_list(self):
        """获取活跃订单列表"""
        return [order.to_dict() for order in self.active_orders]
    
    def get_stock_recommendation(self, code=None):
        """获取股票推荐"""
        if self.quantum_network:
            # 如果有量子网络，使用量子网络做推荐
            try:
                recommendation = self.quantum_network.recommend_stock(code)
                return recommendation
            except Exception as e:
                self.logger.error(f"获取股票推荐失败: {e}")
        
        # 如果没有量子网络或者调用失败，返回模拟推荐
        if code:
            return {
                "buy_probability": np.random.uniform(0.5, 0.9),
                "confidence": np.random.uniform(0.6, 0.95),
                "expected_return": np.random.uniform(0.05, 0.2)
            }
        else:
            # 返回一组推荐股票
            recommendations = []
            stock_codes = ["600000", "600036", "601398", "600519", "600276"]
            stock_names = ["浦发银行", "招商银行", "工商银行", "贵州茅台", "恒瑞医药"]
            
            for code, name in zip(stock_codes, stock_names):
                recommendations.append({
                    "code": code,
                    "name": name,
                    "buy_probability": np.random.uniform(0.5, 0.9),
                    "confidence": np.random.uniform(0.6, 0.95),
                    "expected_return": np.random.uniform(0.05, 0.2)
                })
            
            return recommendations
    
    def get_account_info(self):
        """获取账户信息"""
        return self.account.to_dict()
    
    def get_position_list(self):
        """获取持仓列表"""
        return self.account.get_positions()
    
    def initialize_mock_positions(self):
        """初始化模拟持仓（用于演示）"""
        # 创建一些模拟持仓
        stocks = [
            {"code": "600000", "name": "浦发银行", "price": 10.5, "quantity": 1000},
            {"code": "600036", "name": "招商银行", "price": 45.8, "quantity": 500},
            {"code": "601398", "name": "工商银行", "price": 5.3, "quantity": 2000},
            {"code": "600519", "name": "贵州茅台", "price": 1800.0, "quantity": 10},
            {"code": "600276", "name": "恒瑞医药", "price": 32.5, "quantity": 800}
        ]
        
        for stock in stocks:
            self.account.buy_stock(
                stock["code"], stock["name"], stock["price"], stock["quantity"]
            )
    
    def initialize_mock_orders(self):
        """初始化模拟订单（用于演示）"""
        # 创建一些模拟订单
        orders = [
            {"code": "600000", "name": "浦发银行", "direction": "买入", "price": 10.2, "quantity": 500},
            {"code": "600036", "name": "招商银行", "direction": "卖出", "price": 45.0, "quantity": 200},
            {"code": "601398", "name": "工商银行", "direction": "买入", "price": 5.1, "quantity": 1000}
        ]
        
        # 设置一个较早的时间
        earlier_time = datetime.now()
        
        for order_data in orders:
            order = Order(
                order_data["code"],
                order_data["name"],
                order_data["direction"],
                order_data["price"],
                order_data["quantity"]
            )
            
            # 设置一些为已成交状态
            if np.random.random() < 0.7:
                order.fill()
            
            # 设置创建时间
            order.create_time = earlier_time.strftime("%Y-%m-%d %H:%M:%S")
            order.update_time = earlier_time.strftime("%Y-%m-%d %H:%M:%S")
            
            self.orders.append(order)
            if order.status != "已成交":
                self.active_orders.append(order) 