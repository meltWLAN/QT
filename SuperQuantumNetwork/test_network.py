#!/usr/bin/env python3
"""
测试量子共生网络类
"""

import sys
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestNetwork")

# 导入量子共生网络
from quantum_symbiotic_network.network import QuantumSymbioticNetwork

def test_network():
    """测试网络类"""
    logger.info("创建网络...")
    network = QuantumSymbioticNetwork()
    
    logger.info("初始化网络...")
    try:
        network.initialize(["test_segment"])
        logger.info("网络初始化成功")
    except Exception as e:
        logger.error(f"网络初始化失败: {e}")
        return
    
    # 创建模拟市场数据
    market_data = {
        "test_stock": {
            "price": 100.0,
            "volume": 1000,
            "ma5": 98.0,
            "ma10": 97.0,
            "ma20": 95.0,
            "rsi": 60.0,
            "macd": 2.0
        }
    }
    
    logger.info("测试step方法...")
    try:
        decision = network.step(market_data)
        logger.info(f"决策结果: {decision}")
    except Exception as e:
        logger.error(f"step方法失败: {e}")
        
    logger.info("测试完成")
    
if __name__ == "__main__":
    test_network() 