#!/usr/bin/env python3
"""
超神系统 - 市场数据源
处理各种市场数据的获取和预处理
"""

import logging
import traceback
import time
from datetime import datetime, timedelta
import numpy as np
import json
import os
import sys
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import requests

# 配置日志
logger = logging.getLogger(__name__)

# 全局配置
CONFIG = {
    'api_keys': {},
    'cache_dir': os.path.expanduser('~/超神系统/cache'),
    'use_cache': True,
    'cache_expiry': 3600,  # 缓存过期时间 (秒)
}

# 确保缓存目录存在
if not os.path.exists(CONFIG['cache_dir']):
    try:
        os.makedirs(CONFIG['cache_dir'])
    except Exception as e:
        logger.error(f"创建缓存目录失败: {str(e)}")
        CONFIG['cache_dir'] = os.path.expanduser('~/cache')
        if not os.path.exists(CONFIG['cache_dir']):
            os.makedirs(CONFIG['cache_dir'])


def init_config(config: Dict) -> None:
    """初始化配置
    
    Args:
        config: 配置字典
    """
    if not config:
        return
        
    # 更新全局配置
    CONFIG.update(config)
    
    # 确保缓存目录存在
    if not os.path.exists(CONFIG['cache_dir']):
        try:
            os.makedirs(CONFIG['cache_dir'])
        except Exception as e:
            logger.error(f"创建缓存目录失败: {str(e)}")


def get_index_data(index_code: str) -> Dict:
    """获取指数数据
    
    Args:
        index_code: 指数代码，如 '000001.SH'
    
    Returns:
        指数数据字典
    """
    logger.info(f"获取指数数据: {index_code}")
    
    try:
        # 检查缓存
        if CONFIG['use_cache']:
            cached_data = _get_from_cache(f'index_{index_code}')
            if cached_data:
                logger.info(f"使用缓存的指数数据: {index_code}")
                return cached_data
        
        # 真实环境中，这里会调用实际的数据API
        # 示例中使用模拟数据
        index_data = _mock_index_data(index_code)
        
        # 缓存数据
        if CONFIG['use_cache']:
            _save_to_cache(f'index_{index_code}', index_data)
        
        return index_data
    except Exception as e:
        logger.error(f"获取指数数据失败: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'code': index_code,
            'name': index_code,
            'close': 0,
            'change_pct': 0,
            'error': str(e)
        }


def get_north_flow() -> Dict:
    """获取北向资金数据
    
    Returns:
        北向资金数据字典
    """
    logger.info("获取北向资金数据")
    
    try:
        # 检查缓存
        if CONFIG['use_cache']:
            cached_data = _get_from_cache('north_flow')
            if cached_data:
                logger.info("使用缓存的北向资金数据")
                return cached_data
        
        # 真实环境中，这里会调用实际的数据API
        # 示例中使用模拟数据
        flow_data = _mock_north_flow()
        
        # 缓存数据
        if CONFIG['use_cache']:
            _save_to_cache('north_flow', flow_data)
        
        return flow_data
    except Exception as e:
        logger.error(f"获取北向资金数据失败: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'total_inflow': 0,
            'flow_trend': 'unknown',
            'error': str(e)
        }


def get_sector_data() -> Dict:
    """获取板块数据
    
    Returns:
        板块数据字典
    """
    logger.info("获取板块数据")
    
    try:
        # 检查缓存
        if CONFIG['use_cache']:
            cached_data = _get_from_cache('sector_data')
            if cached_data:
                logger.info("使用缓存的板块数据")
                return cached_data
        
        # 真实环境中，这里会调用实际的数据API
        # 示例中使用模拟数据
        sector_data = _mock_sector_data()
        
        # 缓存数据
        if CONFIG['use_cache']:
            _save_to_cache('sector_data', sector_data)
        
        return sector_data
    except Exception as e:
        logger.error(f"获取板块数据失败: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'hot_sectors': [],
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e)
        }


def get_stock_data(stock_code: str = None, sector: str = None) -> List[Dict]:
    """获取股票数据
    
    Args:
        stock_code: 股票代码，如 '600000'
        sector: 行业名称，如 '半导体'
    
    Returns:
        股票数据列表
    """
    if stock_code:
        logger.info(f"获取股票数据: {stock_code}")
        cache_key = f'stock_{stock_code}'
    elif sector:
        logger.info(f"获取行业股票数据: {sector}")
        cache_key = f'sector_stocks_{sector}'
    else:
        logger.warning("获取股票数据需要指定股票代码或行业")
        return []
    
    try:
        # 检查缓存
        if CONFIG['use_cache']:
            cached_data = _get_from_cache(cache_key)
            if cached_data:
                logger.info(f"使用缓存的股票数据: {cache_key}")
                return cached_data
        
        # 真实环境中，这里会调用实际的数据API
        # 示例中使用模拟数据
        if stock_code:
            stock_data = [_mock_stock_data(stock_code)]
        else:
            stock_data = _mock_sector_stocks(sector)
        
        # 缓存数据
        if CONFIG['use_cache']:
            _save_to_cache(cache_key, stock_data)
        
        return stock_data
    except Exception as e:
        logger.error(f"获取股票数据失败: {str(e)}")
        logger.error(traceback.format_exc())
        return []


# ---- 缓存函数 ----

def _get_from_cache(key: str) -> Optional[Any]:
    """从缓存获取数据
    
    Args:
        key: 缓存键
    
    Returns:
        缓存的数据，如果没有或已过期则返回None
    """
    try:
        cache_file = os.path.join(CONFIG['cache_dir'], f'{key}.json')
        
        # 检查文件是否存在
        if not os.path.exists(cache_file):
            return None
        
        # 检查是否过期
        file_mod_time = os.path.getmtime(cache_file)
        if time.time() - file_mod_time > CONFIG['cache_expiry']:
            logger.info(f"缓存已过期: {key}")
            return None
        
        # 读取缓存
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取缓存失败: {str(e)}")
        return None


def _save_to_cache(key: str, data: Any) -> bool:
    """保存数据到缓存
    
    Args:
        key: 缓存键
        data: 要缓存的数据
    
    Returns:
        是否成功
    """
    try:
        cache_file = os.path.join(CONFIG['cache_dir'], f'{key}.json')
        
        # 保存到缓存
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"保存缓存失败: {str(e)}")
        return False


# ---- 模拟数据函数 (开发测试用) ----

def _mock_index_data(index_code: str) -> Dict:
    """生成模拟指数数据
    
    Args:
        index_code: 指数代码
    
    Returns:
        模拟指数数据
    """
    # 根据指数代码确定基础价格和名称
    if index_code == '000001.SH':
        name = '上证指数'
        base_price = 3200 + np.random.normal(0, 50)
    elif index_code == '399001.SZ':
        name = '深证成指'
        base_price = 10500 + np.random.normal(0, 200)
    elif index_code == '399006.SZ':
        name = '创业板指'
        base_price = 2200 + np.random.normal(0, 40)
    else:
        name = f'指数{index_code}'
        base_price = 1000 + np.random.normal(0, 100)
    
    # 模拟涨跌幅
    change_pct = np.random.normal(0, 1.2)
    
    # 构建指数数据
    index_data = {
        'name': name,
        'code': index_code,
        'close': base_price,
        'change_pct': change_pct,
        'open': base_price * (1 - np.random.normal(0, 0.005)),
        'high': base_price * (1 + np.random.normal(0.005, 0.003)),
        'low': base_price * (1 - np.random.normal(0.005, 0.003)),
        'volume': np.random.randint(100000, 500000) * 10000,
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return index_data


def _mock_north_flow() -> Dict:
    """生成模拟北向资金数据
    
    Returns:
        模拟北向资金数据
    """
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


def _mock_sector_data() -> Dict:
    """生成模拟板块数据
    
    Returns:
        模拟板块数据
    """
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


def _mock_stock_data(stock_code: str) -> Dict:
    """生成模拟股票数据
    
    Args:
        stock_code: 股票代码
    
    Returns:
        模拟股票数据
    """
    # 根据股票代码前缀确定交易所
    if stock_code.startswith('6'):
        exchange = 'SH'
        sector = np.random.choice(["半导体", "新能源", "金融科技", "人工智能"])
    elif stock_code.startswith('0'):
        exchange = 'SZ'
        sector = np.random.choice(["医药生物", "消费电子", "云计算", "军工"])
    else:
        exchange = 'SZ'
        sector = np.random.choice(["创新药", "数字经济", "智能汽车", "机器人"])
    
    # 生成股票名称
    name = f"{sector[:2]}{np.random.randint(1, 100):02d}"
    
    # 生成股票价格
    price = np.random.uniform(10, 100)
    
    # 构建股票数据
    stock = {
        'code': stock_code,
        'name': name,
        'exchange': exchange,
        'sector': sector,
        'price': price,
        'change_pct': np.random.normal(0, 2),
        'pe_ratio': np.random.uniform(15, 50),
        'market_cap': price * np.random.randint(5000, 50000) * 10000,
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return stock


def _mock_sector_stocks(sector: str) -> List[Dict]:
    """生成模拟行业股票数据
    
    Args:
        sector: 行业名称
    
    Returns:
        模拟股票数据列表
    """
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
            'market_cap': price * np.random.randint(5000, 50000) * 10000,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        stocks.append(stock)
    
    return stocks


# 单元测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试获取指数数据
    sh_index = get_index_data('000001.SH')
    print(f"\n上证指数: {sh_index['close']:.2f}, 涨跌幅: {sh_index['change_pct']:+.2f}%")
    
    # 测试获取北向资金
    north_flow = get_north_flow()
    print(f"\n北向资金: {north_flow['total_inflow']/100000000:.2f}亿, 趋势: {north_flow['flow_trend']}")
    
    # 测试获取板块数据
    sectors = get_sector_data()
    print("\n热点板块:")
    for sector in sectors['hot_sectors']:
        print(f"- {sector}")
    
    # 测试获取行业股票
    stocks = get_stock_data(sector="半导体")
    print("\n半导体行业股票:")
    for stock in stocks:
        print(f"- {stock['name']} ({stock['code']}), 价格: {stock['price']:.2f}") 