#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神系统 - 初始化脚本
用于创建必要的目录结构和初始配置文件
"""

import os
import json
import logging
import shutil
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('setup')

# 默认配置
DEFAULT_CONFIG = {
    "cache_dir": "./cache/china_market",
    "quantum_dimensions": 8,
    "learning_rate": 0.01,
    "entanglement_factor": 0.3,
    "policy_influence_weight": 0.4,
    "north_fund_weight": 0.3,
    "sector_rotation_weight": 0.3,
    "default_stocks": ["600519", "000858", "601318", "600036", "000333"],
    "watched_sectors": ["银行", "医药", "食品饮料", "新能源", "半导体", "军工", "房地产"],
    "data_update_interval": 60,
    "risk_threshold": 0.7,
    "max_position_high_risk": 0.2,
    "max_position_medium_risk": 0.5,
    "max_position_low_risk": 0.8
}

# 创建目录结构
def create_directories():
    """创建超神系统所需的目录结构"""
    
    logger.info("开始创建目录结构...")
    
    # 缓存目录
    directories = [
        './cache',
        './cache/china_market',
        './cache/china_market/results',
        './cache/china_market/data',
        './logs',
        './docs',
        './docs/images',
        './config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"已创建目录: {directory}")
    
    return True

# 创建默认配置文件
def create_default_config():
    """创建默认配置文件"""
    
    logger.info("开始创建默认配置文件...")
    
    config_path = './config/default_config.json'
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    
    logger.info(f"默认配置文件已保存到: {config_path}")
    
    return True

# 创建初始目录结构
def create_module_structure():
    """创建模块结构（如果不存在）"""
    
    modules = [
        'SuperQuantumNetwork',
        'SuperQuantumNetwork/quantum_symbiotic_network',
        'SuperQuantumNetwork/quantum_symbiotic_network/core',
        'SuperQuantumNetwork/china_market',
        'SuperQuantumNetwork/china_market/data_sources',
        'SuperQuantumNetwork/china_market/core',
        'SuperQuantumNetwork/china_market/strategies',
        'SuperQuantumNetwork/china_market/controllers',
        'SuperQuantumNetwork/china_market/risk'
    ]
    
    for module in modules:
        if not os.path.exists(module):
            os.makedirs(module, exist_ok=True)
            # 创建__init__.py文件
            with open(f"{module}/__init__.py", 'w', encoding='utf-8') as f:
                f.write(f"# {module} 模块\n")
            
            logger.info(f"已创建模块目录: {module}")
    
    return True

# 检查依赖项
def check_dependencies():
    """检查依赖项是否已安装"""
    try:
        import numpy
        import pandas
        import matplotlib
        import requests
        import akshare
        
        logger.info("核心依赖检查通过")
        return True
    except ImportError as e:
        logger.error(f"依赖检查失败: {e}")
        logger.error("请先运行: pip install -r requirements.txt")
        return False

# 创建示例数据
def create_sample_data():
    """创建示例数据文件"""
    
    sample_data = {
        "market_stats": {
            "up_count": 2103,
            "down_count": 1523,
            "limit_up_count": 58,
            "limit_down_count": 12,
            "total_volume": 5254871025,
            "total_amount": 627856932144.5
        },
        "sh_index": {
            "close": 3452.28,
            "change_pct": 0.0086
        },
        "sz_index": {
            "close": 11502.36,
            "change_pct": 0.0125
        },
        "cyb_index": {
            "close": 2450.21,
            "change_pct": 0.0185
        }
    }
    
    sample_data_path = './cache/china_market/data/sample_market_data.json'
    
    with open(sample_data_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"示例数据已保存到: {sample_data_path}")
    
    return True

# 主函数
def main():
    """主函数"""
    
    logger.info("开始初始化超神系统...")
    
    # 检查目录是否存在，如果不存在则创建
    if create_directories():
        logger.info("目录结构创建成功")
    else:
        logger.error("目录结构创建失败")
        return False
    
    # 创建默认配置文件
    if create_default_config():
        logger.info("默认配置文件创建成功")
    else:
        logger.error("默认配置文件创建失败")
        return False
    
    # 创建模块结构
    if create_module_structure():
        logger.info("模块结构创建成功")
    else:
        logger.error("模块结构创建失败")
        return False
    
    # 检查依赖项
    if not check_dependencies():
        logger.warning("依赖项检查未通过，请安装必要的依赖")
    
    # 创建示例数据
    if create_sample_data():
        logger.info("示例数据创建成功")
    else:
        logger.error("示例数据创建失败")
    
    logger.info("超神系统初始化完成！")
    logger.info("您可以通过运行 'python run_supergod_system.py' 启动系统")
    
    return True

if __name__ == "__main__":
    main() 