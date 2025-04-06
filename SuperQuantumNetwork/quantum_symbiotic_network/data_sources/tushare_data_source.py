#!/usr/bin/env python3
"""
Tushare数据源 - 用于从Tushare获取真实市场数据
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TushareDataSource")

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    logger.warning("Tushare未安装，将使用模拟数据。请使用pip install tushare安装。")
    TUSHARE_AVAILABLE = False

class TushareDataSource:
    """从Tushare获取市场数据的类"""
    
    def __init__(self, token=None):
        """初始化Tushare数据源
        
        Args:
            token (str): Tushare API令牌
        """
        self.token = token
        self.pro = None
        self.cache_dir = os.path.join("quantum_symbiotic_network", "data", "tushare_cache")
        
        # 创建缓存目录
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # 初始化Tushare
        if TUSHARE_AVAILABLE and token:
            try:
                ts.set_token(token)
                self.pro = ts.pro_api()
                logger.info("Tushare API初始化成功")
            except Exception as e:
                logger.error(f"Tushare API初始化失败: {e}")
                self.pro = None
                
    def _get_cache_path(self, name, start_date, end_date):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{name}_{start_date}_{end_date}.csv")
        
    def _load_from_cache(self, cache_path):
        """从缓存加载数据"""
        if os.path.exists(cache_path):
            try:
                return pd.read_csv(cache_path)
            except Exception as e:
                logger.warning(f"从缓存加载数据失败: {e}")
        return None
        
    def _save_to_cache(self, data, cache_path):
        """保存数据到缓存"""
        try:
            data.to_csv(cache_path, index=False)
        except Exception as e:
            logger.warning(f"保存数据到缓存失败: {e}")
            
    def get_stock_list(self):
        """获取股票列表
        
        Returns:
            list: 股票代码列表
        """
        if not self.pro:
            # 如果Tushare不可用，返回模拟数据
            return [f"000{i:03d}" for i in range(1, 51)]
            
        # 尝试从缓存加载
        cache_path = os.path.join(self.cache_dir, "stock_list.csv")
        df = self._load_from_cache(cache_path)
        
        if df is None:
            try:
                # 获取股票基本信息
                df = self.pro.stock_basic(exchange='', list_status='L', 
                                          fields='ts_code,symbol,name,area,industry,list_date')
                # 保存到缓存
                self._save_to_cache(df, cache_path)
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
                return [f"000{i:03d}" for i in range(1, 51)]
                
        # 返回股票代码列表
        return df['ts_code'].tolist()
        
    def calculate_technical_indicators(self, df):
        """计算技术指标
        
        Args:
            df (DataFrame): 股票数据
            
        Returns:
            DataFrame: 添加了技术指标的数据
        """
        # 确保日期排序
        df = df.sort_values('trade_date')
        
        # 计算移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df
        
    def get_stock_data(self, ts_code, start_date, end_date):
        """获取单个股票的历史数据
        
        Args:
            ts_code (str): 股票代码
            start_date (str): 开始日期，格式YYYYMMDD
            end_date (str): 结束日期，格式YYYYMMDD
            
        Returns:
            DataFrame: 股票历史数据
        """
        if not self.pro:
            # 如果Tushare不可用，返回模拟数据
            return self._generate_mock_data(ts_code, start_date, end_date)
            
        # 尝试从缓存加载
        cache_path = self._get_cache_path(ts_code, start_date, end_date)
        df = self._load_from_cache(cache_path)
        
        if df is None:
            try:
                # 设置延迟，避免API限制
                time.sleep(0.5)
                
                # 获取日线数据
                df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                
                # 如果数据为空，返回模拟数据
                if df.empty:
                    return self._generate_mock_data(ts_code, start_date, end_date)
                    
                # 计算技术指标
                df = self.calculate_technical_indicators(df)
                
                # 保存到缓存
                self._save_to_cache(df, cache_path)
                
            except Exception as e:
                logger.error(f"获取股票 {ts_code} 数据失败: {e}")
                return self._generate_mock_data(ts_code, start_date, end_date)
                
        return df
        
    def get_index_data(self, index_code, start_date, end_date):
        """获取指数历史数据
        
        Args:
            index_code (str): 指数代码
            start_date (str): 开始日期，格式YYYYMMDD
            end_date (str): 结束日期，格式YYYYMMDD
            
        Returns:
            DataFrame: 指数历史数据
        """
        if not self.pro:
            # 如果Tushare不可用，返回模拟数据
            return self._generate_mock_data(index_code, start_date, end_date, is_index=True)
            
        # 尝试从缓存加载
        cache_path = self._get_cache_path(f"idx_{index_code}", start_date, end_date)
        df = self._load_from_cache(cache_path)
        
        if df is None:
            try:
                # 设置延迟，避免API限制
                time.sleep(0.5)
                
                # 获取指数日线数据
                df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                
                # 如果数据为空，返回模拟数据
                if df.empty:
                    return self._generate_mock_data(index_code, start_date, end_date, is_index=True)
                    
                # 计算技术指标
                df = self.calculate_technical_indicators(df)
                
                # 保存到缓存
                self._save_to_cache(df, cache_path)
                
            except Exception as e:
                logger.error(f"获取指数 {index_code} 数据失败: {e}")
                return self._generate_mock_data(index_code, start_date, end_date, is_index=True)
                
        return df
        
    def _generate_mock_data(self, code, start_date, end_date, is_index=False):
        """生成模拟数据
        
        Args:
            code (str): 股票/指数代码
            start_date (str): 开始日期，格式YYYYMMDD
            end_date (str): 结束日期，格式YYYYMMDD
            is_index (bool): 是否为指数
            
        Returns:
            DataFrame: 模拟数据
        """
        logger.warning(f"生成 {code} 的模拟数据")
        
        # 解析日期
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        # 生成日期范围
        date_range = []
        current = start
        while current <= end:
            # 跳过周末
            if current.weekday() < 5:
                date_range.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)
            
        # 生成初始价格，根据代码生成随机种子保持一致性
        seed = int(''.join([d for d in code if d.isdigit()])[:6])
        random.seed(seed)
        
        if is_index:
            initial_price = random.uniform(2000, 5000)
            volatility = 0.01  # 指数波动性较小
        else:
            initial_price = random.uniform(10, 100)
            volatility = 0.02  # 股票波动性较大
            
        # 生成价格序列
        prices = [initial_price]
        for _ in range(1, len(date_range)):
            change = random.normalvariate(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
        # 创建DataFrame
        data = []
        for i, date in enumerate(date_range):
            price = prices[i]
            high = price * (1 + random.uniform(0, 0.02))
            low = price * (1 - random.uniform(0, 0.02))
            open_price = low + random.uniform(0, high - low)
            
            # 交易量，股票代码越大，交易量越大
            vol_factor = int(code[-4:]) if code[-4:].isdigit() else 1000
            volume = vol_factor * random.uniform(100, 1000)
            
            data.append({
                'ts_code': code,
                'trade_date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'vol': volume,
                'amount': volume * price,
                'change': 0 if i == 0 else price - prices[i-1],
                'pct_chg': 0 if i == 0 else (price / prices[i-1] - 1) * 100
            })
            
        df = pd.DataFrame(data)
        
        # 计算技术指标
        df = self.calculate_technical_indicators(df)
        
        return df
        
    def get_market_data(self, start_date=None, end_date=None, sample_size=10, include_indices=True):
        """获取市场数据，包括多只股票和指数
        
        Args:
            start_date (str): 开始日期，格式YYYYMMDD
            end_date (str): 结束日期，格式YYYYMMDD
            sample_size (int): 样本股票数量
            include_indices (bool): 是否包含指数
            
        Returns:
            dict: 市场数据，包括股票和指数
        """
        # 设置默认日期为过去180天
        if not start_date:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
            
        logger.info(f"获取市场数据，时间范围: {start_date} - {end_date}")
        
        # 获取股票列表
        stock_list = self.get_stock_list()
        
        # 随机选择样本股票
        if len(stock_list) > sample_size:
            sampled_stocks = random.sample(stock_list, sample_size)
        else:
            sampled_stocks = stock_list
            
        # 获取股票数据
        stocks_data = {}
        for ts_code in sampled_stocks:
            logger.info(f"获取股票 {ts_code} 数据")
            df = self.get_stock_data(ts_code, start_date, end_date)
            stocks_data[ts_code] = df
            
        # 获取指数数据
        indices_data = {}
        if include_indices and self.pro:
            # 主要指数列表
            indices = ['000001.SH', '399001.SZ', '399006.SZ']  # 上证指数、深证成指、创业板指
            
            for idx_code in indices:
                logger.info(f"获取指数 {idx_code} 数据")
                df = self.get_index_data(idx_code, start_date, end_date)
                indices_data[idx_code] = df
                
        # 返回完整市场数据
        market_data = {
            "stocks": stocks_data,
        }
        
        if indices_data:
            market_data["indices"] = indices_data
            
        return market_data 