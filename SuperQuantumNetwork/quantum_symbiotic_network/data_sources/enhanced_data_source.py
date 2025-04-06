#!/usr/bin/env python3
"""
增强数据源模块 - 整合多维度市场数据，提升预测能力
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import requests

# 导入基础数据源
from .tushare_data_source import TushareDataSource, TUSHARE_AVAILABLE

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedDataSource")

class EnhancedDataSource(TushareDataSource):
    """增强型数据源，整合多维度数据"""
    
    def __init__(self, token=None, config=None):
        """初始化增强型数据源
        
        Args:
            token (str): Tushare API令牌
            config (dict): 配置信息
        """
        super().__init__(token)
        self.config = config or {}
        self.cache_dir = os.path.join("quantum_symbiotic_network", "data", "enhanced_cache")
        
        # 创建缓存目录
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # 初始化额外数据源API（如需要）
        self._init_additional_data_sources()
    
    def _init_additional_data_sources(self):
        """初始化额外的数据源"""
        # 这里可以添加其他数据源的初始化，如新浪财经、东方财富等
        # 目前为占位，后续可扩展
        pass
    
    def get_fundamental_data(self, ts_code, report_type='quarterly'):
        """获取基本面数据
        
        Args:
            ts_code (str): 股票代码
            report_type (str): 报告类型，可选值：quarterly, annual
            
        Returns:
            DataFrame: 基本面数据
        """
        if not self.pro:
            return self._generate_mock_fundamental_data(ts_code)
        
        cache_path = os.path.join(self.cache_dir, f"{ts_code}_fundamental_{report_type}.csv")
        df = self._load_from_cache(cache_path)
        
        if df is None:
            try:
                # 获取最近的财务指标数据
                df = self.pro.fina_indicator(ts_code=ts_code, period_type=report_type)
                
                # 如果数据为空，返回模拟数据
                if df.empty:
                    return self._generate_mock_fundamental_data(ts_code)
                
                # 保存到缓存
                self._save_to_cache(df, cache_path)
                
            except Exception as e:
                logger.error(f"获取股票 {ts_code} 基本面数据失败: {e}")
                return self._generate_mock_fundamental_data(ts_code)
        
        return df
    
    def _generate_mock_fundamental_data(self, ts_code):
        """生成模拟基本面数据"""
        periods = 8  # 最近8个季度
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=90*i)).strftime('%Y%m%d') for i in range(periods)]
        
        data = {
            'ts_code': [ts_code] * periods,
            'ann_date': dates,
            'end_date': dates,
            'eps': np.random.normal(0.5, 0.2, periods),
            'dt_eps': np.random.normal(0.4, 0.15, periods),
            'total_revenue_ps': np.random.normal(5, 1.5, periods),
            'revenue_ps': np.random.normal(4.8, 1.4, periods),
            'capital_rese_ps': np.random.normal(3, 0.8, periods),
            'surplus_rese_ps': np.random.normal(1.5, 0.5, periods),
            'undist_profit_ps': np.random.normal(2, 0.7, periods),
            'extra_item': np.random.normal(0, 0.1, periods),
            'profit_dedt': np.random.normal(50000000, 10000000, periods),
            'gross_margin': np.random.normal(0.3, 0.05, periods),
            'current_ratio': np.random.normal(2, 0.5, periods),
            'quick_ratio': np.random.normal(1.5, 0.3, periods),
            'cash_ratio': np.random.normal(0.8, 0.2, periods),
            'ar_turn': np.random.normal(6, 1, periods),
            'inv_turn': np.random.normal(5, 1, periods),
            'assets_turn': np.random.normal(0.7, 0.1, periods),
            'bps': np.random.normal(5, 1, periods),
            'roe': np.random.normal(0.12, 0.03, periods),
            'roe_yearly': np.random.normal(0.12, 0.03, periods),
            'roe_dt': np.random.normal(0.11, 0.03, periods),
            'roa': np.random.normal(0.06, 0.02, periods),
            'debt_to_assets': np.random.normal(0.5, 0.1, periods),
            'op_income_yoy': np.random.normal(0.1, 0.05, periods),
            'ebt_yoy': np.random.normal(0.1, 0.05, periods),
            'tr_yoy': np.random.normal(0.1, 0.05, periods),
            'or_yoy': np.random.normal(0.1, 0.05, periods),
        }
        
        return pd.DataFrame(data)
    
    def get_industry_data(self, industry_code=None):
        """获取行业数据
        
        Args:
            industry_code (str): 行业代码，为空则获取所有行业
            
        Returns:
            DataFrame: 行业数据
        """
        if not self.pro:
            return self._generate_mock_industry_data()
        
        cache_path = os.path.join(self.cache_dir, f"industry_{industry_code or 'all'}.csv")
        df = self._load_from_cache(cache_path)
        
        if df is None:
            try:
                # 获取行业分类数据
                if industry_code:
                    df = self.pro.index_classify(level='L1', src='SW')
                else:
                    df = self.pro.index_classify(level='L1', src='SW')
                
                # 如果数据为空，返回模拟数据
                if df.empty:
                    return self._generate_mock_industry_data()
                
                # 保存到缓存
                self._save_to_cache(df, cache_path)
                
            except Exception as e:
                logger.error(f"获取行业数据失败: {e}")
                return self._generate_mock_industry_data()
        
        return df
    
    def _generate_mock_industry_data(self):
        """生成模拟行业数据"""
        industries = [
            '银行', '保险', '证券', '多元金融', '房地产', '建筑材料', '建筑装饰', 
            '电力设备', '机械设备', '国防军工', '计算机', '传媒', '通信', 
            '医药生物', '食品饮料', '家用电器', '纺织服装', '汽车', '电子'
        ]
        
        data = {
            'index_code': [f"SW{i:04d}" for i in range(1, len(industries)+1)],
            'industry_name': industries,
            'base_date': ['20100101'] * len(industries),
            'change_date': [(datetime.now() - timedelta(days=np.random.randint(30, 300))).strftime('%Y%m%d') for _ in range(len(industries))],
            'weight': np.random.uniform(0.01, 0.1, len(industries))
        }
        
        return pd.DataFrame(data)
    
    def get_macro_data(self, indicator, start_date=None, end_date=None):
        """获取宏观经济数据
        
        Args:
            indicator (str): 指标名称
            start_date (str): 开始日期
            end_date (str): 结束日期
            
        Returns:
            DataFrame: 宏观经济数据
        """
        if not self.pro:
            return self._generate_mock_macro_data(indicator, start_date, end_date)
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        cache_path = os.path.join(self.cache_dir, f"macro_{indicator}_{start_date}_{end_date}.csv")
        df = self._load_from_cache(cache_path)
        
        if df is None:
            try:
                # 获取宏观数据
                if indicator == 'money_supply':
                    df = self.pro.cn_m(start_date=start_date, end_date=end_date)
                elif indicator == 'gdp':
                    df = self.pro.cn_gdp(start_date=start_date, end_date=end_date)
                elif indicator == 'cpi':
                    df = self.pro.cn_cpi(start_date=start_date, end_date=end_date)
                elif indicator == 'ppi':
                    df = self.pro.cn_ppi(start_date=start_date, end_date=end_date)
                else:
                    return self._generate_mock_macro_data(indicator, start_date, end_date)
                
                # 如果数据为空，返回模拟数据
                if df.empty:
                    return self._generate_mock_macro_data(indicator, start_date, end_date)
                
                # 保存到缓存
                self._save_to_cache(df, cache_path)
                
            except Exception as e:
                logger.error(f"获取宏观经济数据 {indicator} 失败: {e}")
                return self._generate_mock_macro_data(indicator, start_date, end_date)
        
        return df
    
    def _generate_mock_macro_data(self, indicator, start_date=None, end_date=None):
        """生成模拟宏观经济数据"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        # 按月生成数据
        periods = (end.year - start.year) * 12 + end.month - start.month + 1
        dates = [(start.replace(day=1) + timedelta(days=30*i)).strftime('%Y%m%d') for i in range(periods)]
        
        if indicator == 'money_supply':
            data = {
                'month': dates,
                'm0': np.random.normal(90000, 5000, periods) * (np.arange(periods)/50 + 1),
                'm1': np.random.normal(600000, 20000, periods) * (np.arange(periods)/50 + 1),
                'm2': np.random.normal(2200000, 80000, periods) * (np.arange(periods)/50 + 1),
                'm0_yoy': np.random.normal(0.05, 0.02, periods),
                'm1_yoy': np.random.normal(0.08, 0.03, periods),
                'm2_yoy': np.random.normal(0.09, 0.02, periods),
            }
        elif indicator == 'gdp':
            # 生成季度数据
            periods = periods // 3
            dates = [(start.replace(day=1) + timedelta(days=90*i)).strftime('%Y%m%d') for i in range(periods)]
            data = {
                'quarter': dates,
                'gdp': np.random.normal(250000, 20000, periods) * (np.arange(periods)/20 + 1),
                'gdp_yoy': np.random.normal(0.06, 0.01, periods),
                'pi': np.random.normal(30000, 3000, periods) * (np.arange(periods)/20 + 1),
                'pi_yoy': np.random.normal(0.04, 0.02, periods),
                'si': np.random.normal(100000, 10000, periods) * (np.arange(periods)/20 + 1),
                'si_yoy': np.random.normal(0.06, 0.02, periods),
                'ti': np.random.normal(120000, 12000, periods) * (np.arange(periods)/20 + 1),
                'ti_yoy': np.random.normal(0.08, 0.02, periods),
            }
        elif indicator in ['cpi', 'ppi']:
            data = {
                'month': dates,
                'nt_yoy': np.random.normal(0.02, 0.005, periods),
                'nt_mom': np.random.normal(0.001, 0.001, periods),
                'nt': np.random.normal(100, 2, periods) * (np.arange(periods)/500 + 1),
            }
        else:
            data = {
                'month': dates,
                'value': np.random.normal(100, 5, periods) * (np.arange(periods)/100 + 1),
                'yoy': np.random.normal(0.05, 0.02, periods),
                'mom': np.random.normal(0.005, 0.002, periods),
            }
        
        return pd.DataFrame(data)
    
    def get_sentiment_data(self, ts_code=None, start_date=None, end_date=None):
        """获取市场情绪数据
        
        Args:
            ts_code (str): 股票代码，为空则获取市场整体情绪
            start_date (str): 开始日期
            end_date (str): 结束日期
            
        Returns:
            DataFrame: 市场情绪数据
        """
        # 情绪数据需要通过其他API获取，这里模拟数据
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 生成日期序列
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        date_range = [(start + timedelta(days=i)).strftime('%Y%m%d') 
                      for i in range((end-start).days+1)]
        
        # 模拟情绪指标
        media_sentiment = np.clip(np.random.normal(0.6, 0.2, len(date_range)), 0, 1)
        social_sentiment = np.clip(np.random.normal(0.55, 0.25, len(date_range)), 0, 1)
        
        # 加入一些趋势和自相关
        for i in range(1, len(date_range)):
            media_sentiment[i] = 0.7 * media_sentiment[i] + 0.3 * media_sentiment[i-1]
            social_sentiment[i] = 0.6 * social_sentiment[i] + 0.4 * social_sentiment[i-1]
        
        # 计算综合情绪
        composite_sentiment = 0.6 * media_sentiment + 0.4 * social_sentiment
        
        data = {
            'date': date_range,
            'media_sentiment': media_sentiment,
            'social_sentiment': social_sentiment,
            'composite_sentiment': composite_sentiment,
            'trading_activity': np.random.normal(0.5, 0.15, len(date_range)),
            'volatility_index': np.random.normal(20, 5, len(date_range))
        }
        
        return pd.DataFrame(data)
    
    def get_enhanced_market_data(self, start_date=None, end_date=None, sample_size=10, include_macro=True):
        """获取增强版市场数据，包含多维度信息
        
        Args:
            start_date (str): 开始日期
            end_date (str): 结束日期
            sample_size (int): 股票样本数量
            include_macro (bool): 是否包含宏观数据
            
        Returns:
            dict: 增强版市场数据
        """
        # 获取基础市场数据
        market_data = self.get_market_data(start_date, end_date, sample_size, include_indices=True)
        
        # 添加宏观经济数据
        if include_macro:
            try:
                market_data['macro'] = {
                    'money_supply': self.get_macro_data('money_supply', start_date, end_date),
                    'gdp': self.get_macro_data('gdp', start_date, end_date),
                    'cpi': self.get_macro_data('cpi', start_date, end_date),
                    'ppi': self.get_macro_data('ppi', start_date, end_date)
                }
            except Exception as e:
                logger.error(f"获取宏观经济数据失败: {e}")
        
        # 添加市场情绪数据
        try:
            market_data['sentiment'] = self.get_sentiment_data(None, start_date, end_date)
        except Exception as e:
            logger.error(f"获取市场情绪数据失败: {e}")
        
        # 添加行业数据
        try:
            market_data['industry'] = self.get_industry_data()
        except Exception as e:
            logger.error(f"获取行业数据失败: {e}")
        
        # 为每只股票添加基本面数据
        try:
            for ts_code in list(market_data['stocks'].keys())[:min(5, len(market_data['stocks']))]:  # 限制处理数量，避免API过载
                market_data['stocks'][ts_code]['fundamental'] = self.get_fundamental_data(ts_code)
        except Exception as e:
            logger.error(f"获取股票基本面数据失败: {e}")
        
        return market_data
    
    def calculate_advanced_indicators(self, df):
        """计算高级技术指标
        
        Args:
            df (DataFrame): 股票数据
            
        Returns:
            DataFrame: 添加了高级技术指标的数据
        """
        # 首先计算基础指标
        df = self.calculate_technical_indicators(df)
        
        try:
            # 相对强弱指数变化率
            df['rsi_change'] = df['rsi14'].diff()
            
            # 布林带
            df['ma20_std'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['ma20'] + 2 * df['ma20_std']
            df['lower_band'] = df['ma20'] - 2 * df['ma20_std']
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['ma20']
            
            # 量价关系
            df['volume_ma5'] = df['vol'].rolling(window=5).mean()
            df['volume_ma10'] = df['vol'].rolling(window=10).mean()
            df['volume_ratio'] = df['vol'] / df['volume_ma5']
            
            # 价格动量指标
            df['momentum'] = df['close'] - df['close'].shift(10)
            df['rate_of_change'] = (df['close'] / df['close'].shift(10) - 1) * 100
            
            # 自定义震荡指标
            high_low_diff = df['high'] - df['low']
            high_close_diff = abs(df['high'] - df['close'].shift(1))
            low_close_diff = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([high_low_diff, high_close_diff, low_close_diff], axis=1).max(axis=1)
            df['atr14'] = tr.rolling(window=14).mean()
            
            # 计算DMI指标
            plus_dm = df['high'].diff()
            minus_dm = df['low'].shift(1) - df['low']
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            df['plus_di14'] = 100 * (plus_dm.rolling(window=14).mean() / df['atr14'])
            df['minus_di14'] = 100 * (minus_dm.rolling(window=14).mean() / df['atr14'])
            df['adx'] = 100 * abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'])
            df['adx'] = df['adx'].rolling(window=14).mean()
            
            # KDJ指标
            low_min = df['low'].rolling(window=9).min()
            high_max = df['high'].rolling(window=9).max()
            df['rsv'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            df['k_value'] = df['rsv'].rolling(window=3).mean()
            df['d_value'] = df['k_value'].rolling(window=3).mean()
            df['j_value'] = 3 * df['k_value'] - 2 * df['d_value']
            
            # OBV (On-Balance Volume)
            df['obv'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
            
            # 计算VWAP (Volume Weighted Average Price)
            df['vwap'] = (df['vol'] * df['close']).cumsum() / df['vol'].cumsum()
            
            # 计算日内波动率
            df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
            
        except Exception as e:
            logger.error(f"计算高级技术指标失败: {e}")
        
        return df 