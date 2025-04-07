#!/usr/bin/env python3
"""
超神系统 - 桌面版中国市场分析模块
"""

import logging
import os
import sys

# 设置版本信息
__version__ = '1.0.0'
__author__ = '超神系统开发团队'

# 确保日志目录存在
log_dir = os.path.expanduser('~/超神系统/logs')
if not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"创建日志目录失败: {str(e)}")

# 确保本地日志目录也存在
local_log_dir = 'logs'
if not os.path.exists(local_log_dir):
    try:
        os.makedirs(local_log_dir, exist_ok=True)
    except Exception as e:
        print(f"创建本地日志目录失败: {str(e)}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, 'market_module.log'), encoding='utf-8'),
    ]
)

# 创建日志对象
logger = logging.getLogger('SuperQuantumNetwork')

# 导入主要组件
try:
    from .china_market_view import ChinaMarketWidget, create_market_view
    from .market_controllers import MarketDataController
    from .data_sources import get_index_data, get_north_flow, get_sector_data, get_stock_data
    from .quantum_ai import QuantumAIEngine, get_market_prediction
    
    __all__ = [
        'ChinaMarketWidget', 
        'create_market_view',
        'MarketDataController',
        'get_index_data',
        'get_north_flow', 
        'get_sector_data', 
        'get_stock_data',
        'QuantumAIEngine',
        'get_market_prediction'
    ]
    
    # 初始化成功
    logger.info(f"超神系统中国市场分析模块 v{__version__} 初始化成功")
    
except ImportError as e:
    logger.error(f"模块导入失败: {str(e)}")
    logger.warning("部分功能可能不可用")


def initialize_market_module(main_window, tab_widget):
    """初始化市场模块
    
    Args:
        main_window: 主窗口对象
        tab_widget: 标签页组件
    
    Returns:
        bool: 是否成功初始化
    """
    try:
        # 创建市场数据控制器
        controller = MarketDataController()
        
        # 创建市场视图
        success = create_market_view(main_window, tab_widget, controller)
        
        return success
    except Exception as e:
        logger.error(f"初始化市场模块失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False 