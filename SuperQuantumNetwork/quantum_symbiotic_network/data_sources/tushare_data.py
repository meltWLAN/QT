import tushare as ts
import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TushareDataSource:
    """
    TuShare数据源，用于获取A股市场数据
    """
    def __init__(self, token):
        self.token = token
        self.pro = None
        self._connect()
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _connect(self):
        """连接到TuShare API"""
        try:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            logger.info("TuShare API连接成功")
        except Exception as e:
            logger.error(f"TuShare API连接失败: {e}")
            raise e
            
    def get_stock_list(self, market='A股主板'):
        """获取股票列表"""
        try:
            # 缓存文件路径
            cache_file = os.path.join(self.cache_dir, "stock_list.csv")
            
            # 如果缓存文件存在且当天创建的，直接读取
            if os.path.exists(cache_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 1:
                return pd.read_csv(cache_file)
                
            # 否则从API获取
            df = self.pro.stock_basic(exchange='', list_status='L', 
                                  fields='ts_code,symbol,name,area,industry,list_date')
            df.to_csv(cache_file, index=False)
            logger.info(f"获取到 {len(df)} 只股票的基本信息")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            # 如果API失败但存在缓存，尝试使用缓存
            if os.path.exists(cache_file):
                logger.warning("使用缓存的股票列表数据")
                return pd.read_csv(cache_file)
            raise e
            
    def get_daily_data(self, ts_code, start_date=None, end_date=None):
        """获取股票日线数据"""
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                
            # 缓存文件路径
            cache_key = f"{ts_code}_{start_date}_{end_date}"
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")
            
            # 如果缓存存在且是今天的数据，直接读取
            if os.path.exists(cache_file) and datetime.now().strftime('%Y%m%d') == end_date and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).seconds < 3600:
                return pd.read_csv(cache_file)
                
            # 从API获取数据
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            # 添加技术指标
            df = self._add_technical_indicators(df)
            
            # 保存到缓存
            df.to_csv(cache_file, index=False)
            logger.info(f"获取到 {ts_code} 从 {start_date} 到 {end_date} 的日线数据, 共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"获取 {ts_code} 日线数据失败: {e}")
            # 尝试使用缓存
            if os.path.exists(cache_file):
                logger.warning(f"使用缓存的 {ts_code} 日线数据")
                return pd.read_csv(cache_file)
            raise e
            
    def _add_technical_indicators(self, df):
        """添加技术指标到数据框"""
        if len(df) == 0:
            return df
            
        # 确保数据按日期排序
        df = df.sort_values('trade_date')
        
        # SMA - 简单移动平均线 (5, 10, 20, 60 日)
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # 成交量移动平均
        df['volume_ma5'] = df['vol'].rolling(window=5).mean()
        
        # RSI - 相对强弱指标 (14日)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']
        
        # 布林带 (20, 2)
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        
        # ATR - 平均真实范围
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(window=14).mean()
        
        return df
        
    def get_index_data(self, index_code='000001.SH', start_date=None, end_date=None):
        """获取指数数据"""
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                
            # 缓存文件路径
            cache_key = f"IDX_{index_code}_{start_date}_{end_date}"
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")
            
            # 检查缓存
            if os.path.exists(cache_file) and datetime.now().strftime('%Y%m%d') == end_date and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).seconds < 3600:
                return pd.read_csv(cache_file)
                
            # 获取指数数据
            df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
            df = self._add_technical_indicators(df)
            df.to_csv(cache_file, index=False)
            logger.info(f"获取到指数 {index_code} 的数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"获取指数 {index_code} 数据失败: {e}")
            if os.path.exists(cache_file):
                logger.warning(f"使用缓存的指数数据")
                return pd.read_csv(cache_file)
            raise e
            
    def get_limit_list(self, trade_date=None):
        """获取每日涨跌停股票列表"""
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
            
        try:
            # 缓存文件
            cache_file = os.path.join(self.cache_dir, f"limit_list_{trade_date}.csv")
            
            # 检查缓存
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
                
            # 获取数据
            df = self.pro.limit_list(trade_date=trade_date)
            df.to_csv(cache_file, index=False)
            logger.info(f"获取到 {trade_date} 的涨跌停股票列表, 共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"获取涨跌停股票列表失败: {e}")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            return pd.DataFrame()
            
    def get_top_list(self, trade_date=None):
        """获取龙虎榜数据"""
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
            
        try:
            # 缓存文件
            cache_file = os.path.join(self.cache_dir, f"top_list_{trade_date}.csv")
            
            # 检查缓存
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
                
            # 获取数据
            df = self.pro.top_list(trade_date=trade_date)
            df.to_csv(cache_file, index=False)
            logger.info(f"获取到 {trade_date} 的龙虎榜数据, 共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"获取龙虎榜数据失败: {e}")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            return pd.DataFrame()
            
    def get_market_data(self, start_date=None, end_date=None, 
                      sample_size=50, include_indices=True):
        """
        获取市场概况数据，包括样本股票和主要指数
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            sample_size: 样本股票数量
            include_indices: 是否包含指数
            
        Returns:
            市场数据字典
        """
        market_data = {}
        
        # 获取股票列表
        stocks = self.get_stock_list()
        
        # 随机选择样本股票
        if sample_size > len(stocks):
            sample_size = len(stocks)
        
        # 优先选择主要板块代表性股票
        major_stocks = []
        
        # 确保包含主要蓝筹股
        blue_chips = ['600519.SH', '601318.SH', '600036.SH', '000858.SZ', '000333.SZ']
        for code in blue_chips:
            if code in stocks['ts_code'].values:
                major_stocks.append(code)
        
        # 按行业分组，每个行业选择市值最大的股票
        industry_samples = []
        try:
            for industry in stocks['industry'].dropna().unique():
                industry_stocks = stocks[stocks['industry'] == industry].sample(min(3, len(stocks[stocks['industry'] == industry])))
                industry_samples.extend(industry_stocks['ts_code'].tolist())
        except:
            # 如果行业分组失败，随机选择
            pass
            
        # 合并主要股票和行业样本，然后随机补足
        selected_stocks = list(set(major_stocks + industry_samples))
        
        if len(selected_stocks) < sample_size:
            # 随机补足
            remaining = stocks[~stocks['ts_code'].isin(selected_stocks)]
            if len(remaining) > 0:
                random_samples = remaining.sample(min(sample_size - len(selected_stocks), len(remaining)))
                selected_stocks.extend(random_samples['ts_code'].tolist())
        
        # 获取每只股票的数据
        stock_data = {}
        for ts_code in selected_stocks:
            try:
                df = self.get_daily_data(ts_code, start_date, end_date)
                if len(df) > 0:
                    stock_data[ts_code] = df
                    time.sleep(0.3)  # 避免API限制
            except Exception as e:
                logger.warning(f"获取 {ts_code} 数据失败: {e}")
                
        market_data['stocks'] = stock_data
        
        # 获取主要指数数据
        if include_indices:
            indices_data = {}
            indices = ['000001.SH', '399001.SZ', '399006.SZ', '000016.SH', '000300.SH']
            for idx in indices:
                try:
                    df = self.get_index_data(idx, start_date, end_date)
                    if len(df) > 0:
                        indices_data[idx] = df
                        time.sleep(0.3)  # 避免API限制
                except Exception as e:
                    logger.warning(f"获取指数 {idx} 数据失败: {e}")
                    
            market_data['indices'] = indices_data
            
        return market_data

# 测试函数
def test_tushare_data(token):
    """测试TuShare数据源"""
    data_source = TushareDataSource(token)
    
    # 测试获取股票列表
    stocks = data_source.get_stock_list()
    print(f"获取到 {len(stocks)} 只股票")
    
    # 测试获取单只股票数据
    if len(stocks) > 0:
        ts_code = stocks.iloc[0]['ts_code']
        df = data_source.get_daily_data(ts_code)
        print(f"获取到 {ts_code} 的日线数据，共 {len(df)} 条记录")
        
    # 测试获取指数数据
    idx_data = data_source.get_index_data()
    print(f"获取到指数数据，共 {len(idx_data)} 条记录")
    
    # 测试获取市场数据
    market_data = data_source.get_market_data(sample_size=10)
    print(f"获取到 {len(market_data['stocks'])} 只股票的数据")
    
    return data_source

if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 使用测试token
    test_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    # 测试
    test_tushare_data(test_token) 