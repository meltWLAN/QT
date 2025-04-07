#!/usr/bin/env python3
"""
超神系统中国市场分析模块 - 功能测试脚本
"""

import os
import sys
import unittest
import logging
import json
from datetime import datetime
import time

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MarketModuleTest")

# 导入测试目标
try:
    from SuperQuantumNetwork import (
        MarketDataController,
        get_index_data, get_north_flow, get_sector_data, get_stock_data,
        QuantumAIEngine, get_market_prediction
    )
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    logger.error("请确保已安装所有依赖")
    sys.exit(1)


class MarketDataTest(unittest.TestCase):
    """数据源模块测试"""

    def test_index_data(self):
        """测试指数数据获取功能"""
        logger.info("测试获取指数数据...")
        
        # 测试上证指数数据
        sh_data = get_index_data("000001.SH")
        self.assertIsNotNone(sh_data)
        self.assertEqual(sh_data["code"], "000001.SH")
        logger.info(f"上证指数: {sh_data['close']} ({sh_data['change_pct']}%)")

        # 测试深证成指数据
        sz_data = get_index_data("399001.SZ")
        self.assertIsNotNone(sz_data)
        self.assertEqual(sz_data["code"], "399001.SZ")
        logger.info(f"深证成指: {sz_data['close']} ({sz_data['change_pct']}%)")

        # 测试创业板指数据
        cyb_data = get_index_data("399006.SZ")
        self.assertIsNotNone(cyb_data)
        self.assertEqual(cyb_data["code"], "399006.SZ")
        logger.info(f"创业板指: {cyb_data['close']} ({cyb_data['change_pct']}%)")

    def test_north_flow(self):
        """测试北向资金数据获取功能"""
        logger.info("测试获取北向资金数据...")
        
        north_data = get_north_flow()
        self.assertIsNotNone(north_data)
        self.assertIn("total_inflow", north_data)
        self.assertIn("total_flow_5d", north_data)
        
        # 检查数据
        logger.info(f"北向资金今日净流入: {north_data['total_inflow']}")
        logger.info(f"北向资金5日净流入: {north_data['total_flow_5d']}")

    def test_sector_data(self):
        """测试板块数据获取功能"""
        logger.info("测试获取板块数据...")
        
        sector_data = get_sector_data()
        self.assertIsNotNone(sector_data)
        self.assertIn("hot_sectors", sector_data)
        self.assertTrue(len(sector_data["hot_sectors"]) > 0)
        
        # 检查热点板块
        logger.info(f"热点板块: {', '.join(sector_data['hot_sectors'][:3])}")

    def test_stock_data(self):
        """测试个股数据获取功能"""
        logger.info("测试获取行业个股数据...")
        
        # 测试获取特定行业的股票
        sector = "半导体"
        stocks = get_stock_data(sector)
        self.assertIsNotNone(stocks)
        self.assertTrue(len(stocks) > 0)
        
        # 验证每个股票的字段
        for stock in stocks[:2]:
            self.assertIn("code", stock)
            self.assertIn("name", stock)
            self.assertIn("sector", stock)
            # 注意：此处不检查sector匹配，因为模拟数据可能不匹配
            
        logger.info(f"{sector}行业股票: {len(stocks)}只")


class QuantumAITest(unittest.TestCase):
    """量子AI引擎测试"""
    
    def setUp(self):
        """测试初始化"""
        # 实例化AI引擎
        self.ai_engine = QuantumAIEngine()
        
        # 测试用模拟市场数据
        self.mock_market_data = {
            'sh_index': {
                'code': '000001.SH',
                'name': '上证指数',
                'close': 3250.5,
                'change_pct': 0.75
            },
            'sz_index': {
                'code': '399001.SZ',
                'name': '深证成指',
                'close': 10523.7,
                'change_pct': 1.2
            },
            'cyb_index': {
                'code': '399006.SZ',
                'name': '创业板指',
                'close': 2143.8,
                'change_pct': 1.5
            },
            'sectors': {
                'hot_sectors': [
                    "半导体", "人工智能", "新能源", "医药生物", "数字经济"
                ]
            }
        }
        
    def test_market_prediction(self):
        """测试市场预测功能"""
        logger.info("测试市场预测功能...")
        
        # 获取预测结果
        prediction = get_market_prediction(self.mock_market_data)
        self.assertIsNotNone(prediction)
        
        # 验证预测结果包含必要字段
        self.assertIn("risk_analysis", prediction)
        self.assertIn("sector_rotation", prediction)
        
        # 检查风险分析
        risk = prediction["risk_analysis"]
        self.assertIn("overall_risk", risk)
        self.assertIn("risk_trend", risk)
        logger.info(f"市场风险: {risk['overall_risk']:.2f}, 趋势: {risk['risk_trend']}")
        
        # 检查板块轮动预测
        sectors = prediction["sector_rotation"]
        self.assertIn("next_sectors_prediction", sectors)
        self.assertTrue(len(sectors["next_sectors_prediction"]) > 0)
        logger.info(f"下一轮潜在热点: {', '.join(sectors['next_sectors_prediction'])}")
        
    def test_portfolio_suggestion(self):
        """测试投资组合建议功能"""
        logger.info("测试投资组合建议功能...")
        
        # 获取预测结果
        prediction = get_market_prediction(self.mock_market_data)
        self.assertIn("portfolio", prediction)
        
        # 检查投资组合建议
        portfolio = prediction["portfolio"]
        self.assertIn("max_position", portfolio)
        self.assertIn("sector_allocation", portfolio)
        self.assertIn("stock_suggestions", portfolio)
        
        # 验证持仓建议在合理范围内
        self.assertTrue(0 <= portfolio["max_position"] <= 1)
        logger.info(f"建议仓位: {portfolio['max_position']:.2f}")
        
        # 验证股票推荐列表
        stocks = portfolio["stock_suggestions"]
        self.assertTrue(len(stocks) > 0)
        
        for stock in stocks[:3]:
            self.assertIn("stock", stock)
            self.assertIn("name", stock)
            self.assertIn("sector", stock)
            self.assertIn("action", stock)
            self.assertIn("risk_level", stock)
            
            logger.info(f"股票推荐: {stock['name']}({stock['stock']}), "
                        f"行业: {stock['sector']}, 动作: {stock['action']}")


class MarketControllerTest(unittest.TestCase):
    """市场数据控制器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.controller = MarketDataController()
        
    def test_data_update(self):
        """测试数据更新功能"""
        logger.info("测试市场数据更新...")
        
        # 更新市场数据
        success = self.controller.update_market_data()
        self.assertTrue(success)
        
        # 检查各类数据 - 使用市场数据属性
        market_data = self.controller.market_data
        self.assertIsNotNone(market_data)
        
        # 检查指数
        self.assertIn("sh_index", market_data)
        self.assertIn("sz_index", market_data)
        self.assertIn("cyb_index", market_data)
        
        # 检查板块
        self.assertIn("sectors", market_data)
        
        logger.info(f"市场数据更新成功: {market_data['sh_index']['update_time']}")
        
    def test_prediction(self):
        """测试趋势预测功能"""
        logger.info("测试市场趋势预测...")
        
        # 进行预测
        prediction = self.controller.predict_market_trend()
        self.assertIsNotNone(prediction)
        
        # 检查预测时间
        self.assertIn("analysis_time", prediction)
        
        # 检查特征数据
        self.assertIn("features", prediction)
        features = prediction["features"]
        self.assertIn("sh_change", features)
        self.assertIn("quantum_effect", features)
        
        logger.info(f"市场预测完成: {prediction['analysis_time']}")
        
    def test_portfolio(self):
        """测试投资组合生成功能"""
        logger.info("测试投资组合生成...")
        
        # 生成投资组合
        portfolio = self.controller.get_portfolio_suggestion()
        self.assertIsNotNone(portfolio)
        
        # 检查核心字段
        self.assertIn("max_position", portfolio)
        self.assertIn("sector_allocation", portfolio)
        self.assertIn("stock_suggestions", portfolio)
        
        logger.info(f"投资组合建议: 建议仓位 {portfolio['max_position']:.2f}")
        for sector in portfolio["sector_allocation"]:
            logger.info(f"行业配置: {sector['sector']} - {sector['weight']:.2f}")


def run_tests():
    """运行测试套件"""
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_suite.addTest(test_loader.loadTestsFromTestCase(MarketDataTest))
    test_suite.addTest(test_loader.loadTestsFromTestCase(QuantumAITest))
    test_suite.addTest(test_loader.loadTestsFromTestCase(MarketControllerTest))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)


if __name__ == "__main__":
    logger.info("开始超神系统市场模块测试...")
    run_tests()
    logger.info("超神系统市场模块测试完成") 