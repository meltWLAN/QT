#!/usr/bin/env python3
"""
超神系统 - 市场控制器
处理市场数据获取、分析和预测控制逻辑
"""

import logging
import traceback
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import os
import sys
from threading import Thread, Lock

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入数据分析和AI模块
try:
    from .quantum_ai import QuantumAIEngine
    HAS_QUANTUM_AI = True
except ImportError:
    logger.warning("无法导入量子AI模块，将使用基础分析")
    HAS_QUANTUM_AI = False

try:
    from .data_sources import (
        get_index_data, 
        get_north_flow,
        get_sector_data,
        get_stock_data
    )
    HAS_DATA_SOURCE = True
except ImportError:
    logger.warning("无法导入市场数据源模块，将使用本地数据")
    HAS_DATA_SOURCE = False


class MarketDataController:
    """市场数据控制器 - 处理市场数据分析和预测"""
    
    def __init__(self, config=None, use_ai=True):
        """初始化市场数据控制器
        
        Args:
            config: 配置字典
            use_ai: 是否使用AI预测功能
        """
        self.config = config or {}
        self.use_ai = use_ai and HAS_QUANTUM_AI
        self.market_data = {}
        self.latest_prediction = {}
        self.data_lock = Lock()
        
        # 初始化AI引擎
        if self.use_ai:
            try:
                self.ai_engine = QuantumAIEngine(config=self.config.get('ai_config', {}))
                logger.info("量子AI引擎初始化成功")
            except Exception as e:
                logger.error(f"量子AI引擎初始化失败: {str(e)}")
                self.use_ai = False
        
        # 设置数据目录
        self.data_dir = self.config.get('data_dir', os.path.expanduser('~/超神系统/market_data'))
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
                logger.info(f"创建市场数据目录: {self.data_dir}")
            except Exception as e:
                logger.error(f"创建市场数据目录失败: {str(e)}")
                self.data_dir = os.path.expanduser('~/market_data')
                if not os.path.exists(self.data_dir):
                    os.makedirs(self.data_dir)
    
    def update_market_data(self):
        """更新市场数据"""
        logger.info("开始更新市场数据...")
        
        with self.data_lock:
            try:
                # 获取市场指数数据
                self._update_market_indices()
                
                # 获取北向资金数据
                self._update_north_flow()
                
                # 获取板块数据
                self._update_sector_data()
                
                # 保存数据到本地
                self._save_market_data()
                
                logger.info("市场数据更新完成")
                return True
            except Exception as e:
                logger.error(f"更新市场数据失败: {str(e)}")
                logger.error(traceback.format_exc())
                return False
    
    def _update_market_indices(self):
        """更新市场指数数据"""
        try:
            if HAS_DATA_SOURCE:
                # 获取上证指数数据
                sh_data = get_index_data('000001.SH')
                # 获取深证成指数据
                sz_data = get_index_data('399001.SZ')
                # 获取创业板指数据
                cyb_data = get_index_data('399006.SZ')
            else:
                # 使用模拟数据
                sh_data = self._generate_mock_index_data('上证指数')
                sz_data = self._generate_mock_index_data('深证成指')
                cyb_data = self._generate_mock_index_data('创业板指')
            
            # 更新市场数据
            self.market_data['sh_index'] = sh_data
            self.market_data['sz_index'] = sz_data
            self.market_data['cyb_index'] = cyb_data
            
            logger.info("市场指数数据更新成功")
        except Exception as e:
            logger.error(f"更新市场指数数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _update_north_flow(self):
        """更新北向资金数据"""
        try:
            if HAS_DATA_SOURCE:
                # 获取北向资金数据
                north_flow = get_north_flow()
            else:
                # 使用模拟数据
                north_flow = self._generate_mock_north_flow()
            
            # 更新北向资金数据
            self.latest_prediction['north_flow'] = north_flow
            
            logger.info("北向资金数据更新成功")
        except Exception as e:
            logger.error(f"更新北向资金数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 这里不抛出异常，因为北向资金不是必须的
    
    def _update_sector_data(self):
        """更新板块数据"""
        try:
            if HAS_DATA_SOURCE:
                # 获取板块数据
                sector_data = get_sector_data()
            else:
                # 使用模拟数据
                sector_data = self._generate_mock_sector_data()
            
            # 更新板块数据
            self.market_data['sectors'] = sector_data
            
            logger.info("板块数据更新成功")
        except Exception as e:
            logger.error(f"更新板块数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 这里不抛出异常，因为板块数据不是必须的
    
    def _save_market_data(self):
        """保存市场数据到本地"""
        try:
            # 创建文件名 (使用日期)
            date_str = datetime.now().strftime('%Y%m%d')
            file_path = os.path.join(self.data_dir, f'market_data_{date_str}.json')
            
            # 保存数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.market_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"市场数据保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存市场数据失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    def predict_market_trend(self):
        """预测市场趋势"""
        logger.info("开始市场趋势预测...")
        
        with self.data_lock:
            try:
                if self.use_ai:
                    # 使用量子AI引擎进行预测
                    prediction = self.ai_engine.predict_market(self.market_data)
                else:
                    # 使用基础分析进行预测
                    prediction = self._basic_market_analysis()
                
                # 更新最新预测结果
                self.latest_prediction.update(prediction)
                
                # 保存预测结果
                self._save_prediction()
                
                logger.info("市场趋势预测完成")
                return self.latest_prediction
            except Exception as e:
                logger.error(f"市场趋势预测失败: {str(e)}")
                logger.error(traceback.format_exc())
                return {}
    
    def _basic_market_analysis(self):
        """基本市场分析 (当AI不可用时)"""
        # 注意：这只是一个简化的市场分析示例
        
        # 获取指数数据
        sh_index = self.market_data.get('sh_index', {})
        sz_index = self.market_data.get('sz_index', {})
        
        # 计算市场整体状态
        sh_change = sh_index.get('change_pct', 0)
        sz_change = sz_index.get('change_pct', 0)
        
        # 平均涨跌幅
        avg_change = (sh_change + sz_change) / 2 if sh_change is not None and sz_change is not None else 0
        
        # 确定市场趋势
        if avg_change > 1.5:
            trend = "强势上涨"
            risk = 0.3  # 低风险
        elif avg_change > 0:
            trend = "小幅上涨"
            risk = 0.4
        elif avg_change > -1.5:
            trend = "小幅下跌"
            risk = 0.6
        else:
            trend = "下跌趋势"
            risk = 0.8  # 高风险
        
        # 构建预测结果
        prediction = {
            'risk_analysis': {
                'overall_risk': risk,
                'risk_trend': trend,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'sector_rotation': {
                'current_hot_sectors': self._get_hot_sectors(),
                'next_sectors_prediction': self._predict_next_sectors()
            }
        }
        
        return prediction
    
    def get_portfolio_suggestion(self):
        """获取投资组合建议"""
        logger.info("生成投资组合建议...")
        
        try:
            # 获取最新预测
            prediction = self.latest_prediction
            
            if not prediction:
                # 如果没有预测，先运行一次
                prediction = self.predict_market_trend()
            
            # 获取风险分析
            risk_analysis = prediction.get('risk_analysis', {})
            overall_risk = risk_analysis.get('overall_risk', 0.5)
            
            # 根据风险调整仓位
            max_position = 1.0 - overall_risk
            
            # 获取当前热点板块
            sector_rotation = prediction.get('sector_rotation', {})
            hot_sectors = sector_rotation.get('current_hot_sectors', [])
            
            # 构建投资组合建议
            portfolio = {
                'max_position': max_position,
                'sector_allocation': self._generate_sector_allocation(hot_sectors),
                'stock_suggestions': self._generate_stock_suggestions(hot_sectors)
            }
            
            # 缓存建议
            self.latest_prediction['portfolio'] = portfolio
            
            logger.info("投资组合建议生成完成")
            return portfolio
        except Exception as e:
            logger.error(f"生成投资组合建议失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'max_position': 0.5,
                'sector_allocation': [],
                'stock_suggestions': []
            }
    
    def _generate_sector_allocation(self, hot_sectors):
        """生成行业配置建议"""
        allocation = []
        
        # 确保至少有一些板块
        if not hot_sectors:
            hot_sectors = self._get_hot_sectors()
        
        # 为前3个热点行业分配较高权重
        total_weight = 0
        for i, sector in enumerate(hot_sectors[:3]):
            # 权重递减
            weight = 0.4 - (i * 0.1)
            allocation.append({
                'sector': sector,
                'weight': weight
            })
            total_weight += weight
        
        # 分配其余权重
        remaining_weight = 1.0 - total_weight
        if len(hot_sectors) > 3:
            for i, sector in enumerate(hot_sectors[3:]):
                # 平均分配剩余权重
                weight = remaining_weight / (len(hot_sectors) - 3)
                allocation.append({
                    'sector': sector,
                    'weight': weight
                })
        
        return allocation
    
    def _generate_stock_suggestions(self, hot_sectors):
        """生成个股推荐"""
        suggestions = []
        
        # 确保至少有一些板块
        if not hot_sectors:
            hot_sectors = self._get_hot_sectors()
        
        # 为每个热点行业推荐1-2只股票
        for sector in hot_sectors[:5]:  # 只取前5个板块
            # 获取该行业的股票
            if HAS_DATA_SOURCE:
                # 从数据源获取
                sector_stocks = get_stock_data(sector=sector)
            else:
                # 使用模拟数据
                sector_stocks = self._generate_mock_stocks(sector)
            
            # 添加推荐
            for stock in sector_stocks[:2]:  # 每个行业选2只
                # 随机生成一个操作建议
                action_types = ["买入", "关注", "持有"]
                action = np.random.choice(action_types, p=[0.5, 0.3, 0.2])
                
                # 随机生成风险评级
                risk_levels = ["低风险", "中等风险", "高风险"]
                risk_level = np.random.choice(risk_levels, p=[0.3, 0.5, 0.2])
                
                suggestions.append({
                    'stock': stock.get('code', ''),
                    'name': stock.get('name', ''),
                    'sector': sector,
                    'action': action,
                    'current_price': stock.get('price', 0),
                    'risk_level': risk_level
                })
        
        return suggestions
    
    def _save_prediction(self):
        """保存预测结果"""
        try:
            # 创建文件名 (使用日期和时间)
            datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.data_dir, f'prediction_{datetime_str}.json')
            
            # 保存数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.latest_prediction, f, ensure_ascii=False, indent=2)
            
            logger.info(f"预测结果保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存预测结果失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    # ---- 模拟数据生成方法 (用于开发和测试) ----
    
    def _generate_mock_index_data(self, name):
        """生成模拟指数数据"""
        # 模拟当前价格
        if name == '上证指数':
            base_price = 3200 + np.random.normal(0, 50)
        elif name == '深证成指':
            base_price = 10500 + np.random.normal(0, 200)
        else:  # 创业板指
            base_price = 2200 + np.random.normal(0, 40)
        
        # 模拟涨跌幅
        change_pct = np.random.normal(0, 1.2)
        
        # 构建指数数据
        index_data = {
            'name': name,
            'code': '000001.SH' if name == '上证指数' else ('399001.SZ' if name == '深证成指' else '399006.SZ'),
            'close': base_price,
            'change_pct': change_pct,
            'open': base_price * (1 - np.random.normal(0, 0.005)),
            'high': base_price * (1 + np.random.normal(0.005, 0.003)),
            'low': base_price * (1 - np.random.normal(0.005, 0.003)),
            'volume': np.random.randint(100000, 500000) * 10000,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return index_data
    
    def _generate_mock_north_flow(self):
        """生成模拟北向资金数据"""
        # 模拟今日净流入 (单位：元)
        inflow = np.random.normal(0, 5) * 100000000
        
        # 模拟5日净流入
        flow_5d = inflow * 3 + np.random.normal(0, 2) * 100000000
        
        # 模拟资金趋势
        if inflow > 300000000:
            trend = "强势流入"
        elif inflow > 0:
            trend = "小幅流入"
        elif inflow > -300000000:
            trend = "小幅流出"
        else:
            trend = "大幅流出"
        
        # 构建北向资金数据
        north_flow = {
            'total_inflow': inflow,
            'total_flow_5d': flow_5d,
            'flow_trend': trend,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return north_flow
    
    def _get_hot_sectors(self):
        """获取热点板块列表"""
        # 如果有真实数据，使用真实数据
        sectors = self.market_data.get('sectors', {}).get('hot_sectors', [])
        
        # 如果没有数据，使用模拟数据
        if not sectors:
            sectors = [
                "半导体", "人工智能", "新能源", "医药生物", 
                "军工", "消费电子", "数字经济", "金融科技"
            ]
            # 随机排序
            np.random.shuffle(sectors)
        
        return sectors[:8]  # 返回最多8个热点板块
    
    def _predict_next_sectors(self):
        """预测下一轮热点板块"""
        # 这里使用一个简单的方法：从所有可能的板块中随机选择一些作为下一轮热点
        all_possible_sectors = [
            "半导体", "人工智能", "新能源", "医药生物", "军工", 
            "消费电子", "数字经济", "金融科技", "碳中和", 
            "云计算", "区块链", "元宇宙", "光伏", "储能", 
            "创新药", "数字货币", "智能汽车", "机器人"
        ]
        
        # 排除当前热点
        current_hot = self._get_hot_sectors()
        next_candidates = [s for s in all_possible_sectors if s not in current_hot]
        
        # 随机选择3-5个
        n_sectors = np.random.randint(3, 6)
        if len(next_candidates) > n_sectors:
            next_sectors = np.random.choice(next_candidates, size=n_sectors, replace=False).tolist()
        else:
            next_sectors = next_candidates
        
        return next_sectors
    
    def _generate_mock_sector_data(self):
        """生成模拟板块数据"""
        # 可能的板块列表
        all_sectors = [
            "半导体", "人工智能", "新能源", "医药生物", "军工", 
            "消费电子", "数字经济", "金融科技", "碳中和", 
            "云计算", "区块链", "元宇宙", "光伏", "储能", 
            "创新药", "数字货币", "智能汽车", "机器人"
        ]
        
        # 随机选择一些作为热点
        n_hot = np.random.randint(5, 9)
        hot_sectors = np.random.choice(all_sectors, size=n_hot, replace=False).tolist()
        
        # 构建板块数据
        sector_data = {
            'hot_sectors': hot_sectors,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return sector_data
    
    def _generate_mock_stocks(self, sector):
        """生成模拟股票数据"""
        # 为指定行业生成模拟股票
        stocks = []
        
        # 每个行业生成3-5只股票
        n_stocks = np.random.randint(3, 6)
        
        # 使用行业前缀生成股票代码和名称
        sector_prefix = sector[:2]
        
        for i in range(n_stocks):
            # 生成股票代码 (沪市以6开头，深市以0或3开头)
            code_prefix = np.random.choice(['6', '0', '3'])
            code = code_prefix + ''.join([str(np.random.randint(0, 10)) for _ in range(5)])
            
            # 生成股票名称
            name = f"{sector_prefix}{np.random.randint(1, 100):02d}"
            
            # 生成股票价格
            price = np.random.uniform(10, 100)
            
            # 构建股票数据
            stock = {
                'code': code,
                'name': name,
                'sector': sector,
                'price': price,
                'change_pct': np.random.normal(0, 2),
                'pe_ratio': np.random.uniform(15, 50),
                'market_cap': price * np.random.randint(5000, 50000) * 10000
            }
            
            stocks.append(stock)
        
        return stocks


class TradeController:
    """交易控制器 - 处理交易执行和管理"""
    
    def __init__(self, market_controller=None, config=None):
        """初始化交易控制器
        
        Args:
            market_controller: 市场数据控制器实例
            config: 配置字典
        """
        self.market_controller = market_controller
        self.config = config or {}
        self.positions = {}
        self.orders = []
        self.trading_enabled = False
        self.trading_mode = "simulation"  # 默认为模拟交易
        
        logger.info("交易控制器初始化成功")
    
    def enable_trading(self, mode="simulation"):
        """启用交易功能
        
        Args:
            mode: 交易模式，可选 "simulation"(模拟) 或 "real"(实盘)
        """
        if mode not in ["simulation", "real"]:
            logger.error(f"不支持的交易模式: {mode}")
            return False
        
        self.trading_mode = mode
        self.trading_enabled = True
        logger.info(f"交易已启用，模式: {mode}")
        return True
    
    def disable_trading(self):
        """禁用交易功能"""
        self.trading_enabled = False
        logger.info("交易已禁用")
        return True
    
    def place_order(self, symbol, direction, quantity, price_type="market", price=None):
        """下单
        
        Args:
            symbol: 股票代码
            direction: 买卖方向，"buy" 或 "sell"
            quantity: 数量
            price_type: 价格类型，"market"(市价) 或 "limit"(限价)
            price: 限价单价格
            
        Returns:
            order_id: 订单ID
        """
        if not self.trading_enabled:
            logger.warning("交易功能未启用，无法下单")
            return None
        
        # 创建订单
        order = {
            "order_id": f"ORDER_{int(time.time())}_{len(self.orders)}",
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price_type": price_type,
            "price": price,
            "status": "submitted",
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加到订单列表
        self.orders.append(order)
        
        # 如果是模拟交易，立即执行
        if self.trading_mode == "simulation":
            self._execute_simulation_order(order)
        
        logger.info(f"订单已提交: {order['order_id']}")
        return order["order_id"]
    
    def _execute_simulation_order(self, order):
        """执行模拟订单"""
        # 更新订单状态
        order["status"] = "filled"
        order["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 更新持仓
        symbol = order["symbol"]
        direction = order["direction"]
        quantity = order["quantity"]
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": 0,
                "cost": 0,
                "current_price": 0
            }
        
        if direction == "buy":
            self.positions[symbol]["quantity"] += quantity
        elif direction == "sell":
            self.positions[symbol]["quantity"] -= quantity
        
        logger.info(f"模拟订单执行完成: {order['order_id']}")
    
    def get_positions(self):
        """获取当前持仓"""
        return self.positions
    
    def get_orders(self, status=None):
        """获取订单列表
        
        Args:
            status: 订单状态筛选
            
        Returns:
            orders: 订单列表
        """
        if status:
            return [order for order in self.orders if order["status"] == status]
        return self.orders
    
    def cancel_order(self, order_id):
        """取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            success: 是否成功
        """
        for order in self.orders:
            if order["order_id"] == order_id and order["status"] == "submitted":
                order["status"] = "cancelled"
                order["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"订单已取消: {order_id}")
                return True
        
        logger.warning(f"找不到可取消的订单: {order_id}")
        return False


# 单元测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建控制器
    controller = MarketDataController(use_ai=False)
    
    # 更新市场数据
    controller.update_market_data()
    
    # 预测市场趋势
    prediction = controller.predict_market_trend()
    
    # 获取投资组合建议
    portfolio = controller.get_portfolio_suggestion()
    
    # 输出结果
    print("\n市场分析结果:")
    print(f"市场风险: {prediction.get('risk_analysis', {}).get('overall_risk', 0)}")
    print(f"趋势: {prediction.get('risk_analysis', {}).get('risk_trend', '')}")
    print("\n热点板块:")
    for sector in prediction.get('sector_rotation', {}).get('current_hot_sectors', []):
        print(f"- {sector}")
    
    print("\n投资建议:")
    print(f"建议仓位: {portfolio.get('max_position', 0) * 100:.0f}%")
    print("\n个股推荐:")
    for stock in portfolio.get('stock_suggestions', [])[:5]:
        print(f"- {stock.get('name', '')} ({stock.get('stock', '')}): {stock.get('action', '')} 风险:{stock.get('risk_level', '')}") 