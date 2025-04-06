#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中国股市数据源 - 负责获取和处理A股市场数据
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ChinaStockDataSource:
    """中国股市数据源"""
    
    def __init__(self, cache_dir="./cache/china_market"):
        self.cache_dir = cache_dir
        self.data_dir = os.path.join(cache_dir, "data")
        
        # 创建缓存目录
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化基础数据
        self.stock_basics = self._get_stock_basics()
        self.sector_mapping = self._get_sector_mapping()
        self.last_update_time = None
        
        logger.info("中国股市数据源初始化完成")
    
    def _get_stock_basics(self):
        """获取A股基本信息"""
        try:
            cache_file = os.path.join(self.cache_dir, "stock_basics.csv")
            
            # 检查缓存是否存在且新鲜
            now = datetime.now()
            need_update = True
            
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                # 股票基本信息通常变化不大，每3天更新一次即可
                need_update = (now - file_time).days > 3
                
            if not need_update and os.path.exists(cache_file):
                logger.debug("从缓存加载股票基本信息")
                basics = pd.read_csv(cache_file)
                return basics
            
            # 模拟数据（实际应用中替换为真实数据API调用）
            logger.info("生成模拟股票基本信息数据")
            
            # 生成一些模拟的股票基本信息
            stock_list = [
                # 上证A股
                {"code": "600519", "name": "贵州茅台", "industry": "食品饮料", "market": "上证A股", "list_date": "2001-08-27"},
                {"code": "601318", "name": "中国平安", "industry": "金融保险", "market": "上证A股", "list_date": "2007-03-01"},
                {"code": "600036", "name": "招商银行", "industry": "银行", "market": "上证A股", "list_date": "2002-04-09"},
                {"code": "600276", "name": "恒瑞医药", "industry": "医药", "market": "上证A股", "list_date": "2000-10-18"},
                {"code": "600887", "name": "伊利股份", "industry": "食品饮料", "market": "上证A股", "list_date": "1998-03-12"},
                
                # 深证A股
                {"code": "000858", "name": "五粮液", "industry": "食品饮料", "market": "深证A股", "list_date": "1998-04-27"},
                {"code": "000333", "name": "美的集团", "industry": "家电", "market": "深证A股", "list_date": "2013-09-18"},
                {"code": "000651", "name": "格力电器", "industry": "家电", "market": "深证A股", "list_date": "1996-11-18"},
                {"code": "000001", "name": "平安银行", "industry": "银行", "market": "深证A股", "list_date": "1991-04-03"},
                {"code": "000002", "name": "万科A", "industry": "房地产", "market": "深证A股", "list_date": "1991-01-29"},
                
                # 创业板
                {"code": "300750", "name": "宁德时代", "industry": "新能源", "market": "创业板", "list_date": "2018-06-11"},
                {"code": "300059", "name": "东方财富", "industry": "金融服务", "market": "创业板", "list_date": "2010-03-19"},
                {"code": "300015", "name": "爱尔眼科", "industry": "医药", "market": "创业板", "list_date": "2009-10-30"},
                {"code": "300124", "name": "汇川技术", "industry": "电气机械", "market": "创业板", "list_date": "2010-09-20"},
                {"code": "300014", "name": "亿纬锂能", "industry": "新能源", "market": "创业板", "list_date": "2009-10-30"}
            ]
            
            # 转换为DataFrame
            basics = pd.DataFrame(stock_list)
            
            # 生成更多模拟数据
            np.random.seed(42)  # 保证可重复性
            
            # 为每只股票添加更多信息
            basics['total_share'] = np.random.uniform(1, 100, size=len(basics)) * 1e8  # 总股本(亿股)
            basics['float_share'] = basics['total_share'] * np.random.uniform(0.3, 0.8, size=len(basics))  # 流通股本
            basics['pe'] = np.random.uniform(10, 50, size=len(basics))  # 市盈率
            basics['pb'] = np.random.uniform(1, 10, size=len(basics))  # 市净率
            basics['total_assets'] = np.random.uniform(1, 1000, size=len(basics)) * 1e8  # 总资产(亿元)
            basics['liquid_assets'] = basics['total_assets'] * np.random.uniform(0.1, 0.5, size=len(basics))  # 流动资产
            basics['fixed_assets'] = basics['total_assets'] * np.random.uniform(0.1, 0.5, size=len(basics))  # 固定资产
            basics['reserved'] = np.random.uniform(1, 50, size=len(basics)) * 1e8  # 公积金
            basics['reserved_per_share'] = basics['reserved'] / basics['total_share']  # 每股公积金
            basics['esp'] = np.random.uniform(0.1, 2, size=len(basics))  # 每股收益
            basics['bvps'] = np.random.uniform(1, 10, size=len(basics))  # 每股净资产
            basics['dividend'] = np.random.uniform(0, 1, size=len(basics))  # 股息率
            
            # 保存到缓存
            basics.to_csv(cache_file, index=False)
            
            return basics
            
        except Exception as e:
            logger.error(f"加载股票基本信息失败: {e}")
            # 返回空DataFrame防止程序崩溃
            return pd.DataFrame()
    
    def _get_sector_mapping(self):
        """获取行业分类映射"""
        try:
            # 尝试从缓存加载
            cache_file = os.path.join(self.data_dir, "sector_mapping.json")
            if os.path.exists(cache_file):
                # 检查文件是否过期（7天更新一次）
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if (datetime.now() - file_time).days < 7:
                    logger.info("从缓存加载行业分类数据")
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            # 模拟数据（实际应用中应替换为真实数据API调用）
            logger.info("生成模拟行业分类数据")
            
            # 定义行业
            sectors = ["银行", "医药", "食品饮料", "新能源", "半导体", "军工", "房地产", 
                      "家电", "汽车", "石油化工", "钢铁", "建筑", "通信", "煤炭", "互联网"]
            
            # 为每个行业随机分配股票
            sector_stocks = {}
            codes = self.stock_basics['code'].tolist()
            
            for sector in sectors:
                # 每个行业随机选择10-30只股票
                num_stocks = np.random.randint(10, 31)
                if len(codes) > num_stocks:
                    selected_stocks = np.random.choice(codes, num_stocks, replace=False).tolist()
                    # 从总池中移除已选股票，确保不重复
                    for stock in selected_stocks:
                        if stock in codes:
                            codes.remove(stock)
                    sector_stocks[sector] = selected_stocks
            
            # 保存到缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(sector_stocks, f, ensure_ascii=False, indent=2)
                
            return sector_stocks
            
        except Exception as e:
            logger.error(f"获取行业分类数据失败: {e}")
            return {}
    
    def get_stock_data(self, symbol, days=30):
        """获取个股历史数据"""
        try:
            # 尝试从缓存加载
            cache_file = os.path.join(self.data_dir, f"stock_{symbol}.csv")
            now = datetime.now()
            
            # 判断是否需要更新缓存
            need_update = True
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                # 交易日期间工作时间每小时更新一次，非交易时间每天更新一次
                is_trading_hour = (now.hour >= 9 and now.hour <= 15) and now.weekday() < 5
                if is_trading_hour:
                    need_update = (now - file_time).seconds > 3600  # 1小时
                else:
                    need_update = (now - file_time).days > 0  # 1天
            
            if not need_update and os.path.exists(cache_file):
                logger.debug(f"从缓存加载股票{symbol}数据")
                stock_data = pd.read_csv(cache_file)
                stock_data['日期'] = pd.to_datetime(stock_data['日期'])
                # 只返回请求的天数
                return stock_data.tail(days)
            
            # 模拟数据（实际应用中应替换为真实数据API调用）
            logger.info(f"生成股票{symbol}模拟数据")
            
            # 生成历史日期序列
            end_date = now.date()
            start_date = end_date - timedelta(days=days * 2)  # 获取更多数据以防有非交易日
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B'表示工作日
            
            # 生成模拟价格数据
            np.random.seed(int(symbol) % 10000)  # 使用股票代码作为随机种子
            
            # 生成起始价格（在合理范围内）
            base_price = np.random.uniform(10, 100)
            
            # 生成模拟波动率（每日波动在-2%到2%之间）
            volatility = np.random.uniform(0.01, 0.03)
            
            # 生成每日涨跌幅
            daily_returns = np.random.normal(0.0005, volatility, size=len(date_range))  # 微小正偏移
            
            # 生成价格序列
            prices = [base_price]
            for ret in daily_returns:
                next_price = prices[-1] * (1 + ret)
                # 确保价格在合理范围内
                next_price = max(1, min(1000, next_price))
                prices.append(next_price)
                
            prices = prices[1:]  # 移除初始价格
            
            # 计算其他价格数据
            opens = [price * (1 + np.random.uniform(-0.01, 0.01)) for price in prices]
            highs = [max(o, c) * (1 + np.random.uniform(0, 0.01)) for o, c in zip(opens, prices)]
            lows = [min(o, c) * (1 - np.random.uniform(0, 0.01)) for o, c in zip(opens, prices)]
            
            # 生成成交量数据（关联价格变化）
            volumes = []
            for i, ret in enumerate(daily_returns):
                base_volume = np.random.uniform(1e6, 1e7)  # 基础成交量
                # 价格涨跌幅大时成交量增加
                volume_multiplier = 1 + abs(ret) * 10
                volumes.append(int(base_volume * volume_multiplier))
            
            # 计算技术指标
            ma5 = np.convolve(prices, np.ones(5)/5, mode='valid')
            ma10 = np.convolve(prices, np.ones(10)/10, mode='valid')
            ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
            
            # 填充前面的均线数据
            ma5 = np.concatenate([np.array([prices[0]] * 4), ma5])
            ma10 = np.concatenate([np.array([prices[0]] * 9), ma10])
            ma20 = np.concatenate([np.array([prices[0]] * 19), ma20])
            
            # 计算涨跌幅
            price_change = np.diff(prices, prepend=prices[0])
            price_change_pct = price_change / np.concatenate([[prices[0]], prices[:-1]])
            
            # 计算换手率
            turnover_rate = [vol / (np.random.uniform(5e7, 1e8)) for vol in volumes]
            
            # 获取股票名称
            stock_name = "未知"
            if self.stock_basics is not None and not self.stock_basics.empty:
                try:
                    stock_info = self.stock_basics[self.stock_basics['code'] == symbol]
                    if not stock_info.empty:
                        stock_name = stock_info['name'].values[0]
                except KeyError:
                    pass
            
            # 创建DataFrame
            stock_data = pd.DataFrame({
                '日期': date_range,
                '开盘': opens,
                '收盘': prices,
                '最高': highs,
                '最低': lows,
                '成交量': volumes,
                'ma5': ma5,
                'ma10': ma10,
                'ma20': ma20,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'turnover_rate': turnover_rate,
                'name': stock_name  # 添加股票名称
            })
            
            # 保存到缓存
            stock_data.to_csv(cache_file, index=False)
            
            # 只返回请求的天数
            return stock_data.tail(days)
            
        except Exception as e:
            logger.error(f"获取股票{symbol}数据失败: {e}")
            # 返回空DataFrame防止程序崩溃
            return pd.DataFrame()
    
    def get_index_data(self, index_code="000001", days=30):
        """获取指数数据"""
        try:
            # 尝试从缓存加载
            cache_file = os.path.join(self.data_dir, f"index_{index_code}.csv")
            now = datetime.now()
            
            # 判断是否需要更新缓存
            need_update = True
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                # 交易日期间工作时间每小时更新一次，非交易时间每天更新一次
                is_trading_hour = (now.hour >= 9 and now.hour <= 15) and now.weekday() < 5
                if is_trading_hour:
                    need_update = (now - file_time).seconds > 3600  # 1小时
                else:
                    need_update = (now - file_time).days > 0  # 1天
            
            if not need_update and os.path.exists(cache_file):
                logger.debug(f"从缓存加载指数{index_code}数据")
                index_data = pd.read_csv(cache_file)
                index_data['date'] = pd.to_datetime(index_data['date'])
                # 只返回请求的天数
                return index_data.tail(days)
            
            # 模拟数据（实际应用中应替换为真实数据API调用）
            logger.info(f"生成指数{index_code}模拟数据")
            
            # 指数基准值映射
            index_base = {
                "000001": 3000,  # 上证指数
                "399001": 10000,  # 深证成指
                "399006": 2000   # 创业板指
            }
            
            base = index_base.get(index_code, 2000)
            
            # 生成历史日期序列
            end_date = now.date()
            start_date = end_date - timedelta(days=days*2)  # 生成更多数据以备不时之需
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
            
            # 生成随机价格序列
            np.random.seed(int(index_code))  # 使相同指数生成相同序列
            
            # 生成带趋势的随机价格
            trend = np.random.choice([-1, 1]) * np.random.uniform(0.0001, 0.0003)  # 趋势因子
            daily_returns = np.random.normal(trend, 0.01, len(date_range))  # 每日收益率
            
            # 累积收益率
            cum_returns = np.cumprod(1 + daily_returns)
            closes = base * cum_returns
            
            # 开盘价、最高价、最低价
            opens = closes * (1 + np.random.normal(0, 0.003, len(closes)))
            highs = np.maximum(closes, opens) * (1 + np.random.uniform(0, 0.005, len(closes)))
            lows = np.minimum(closes, opens) * (1 - np.random.uniform(0, 0.005, len(closes)))
            
            # 成交量和成交额
            base_volume = np.random.randint(10000000, 100000000)
            volumes = np.random.normal(base_volume, base_volume * 0.2, len(closes))
            volumes = np.maximum(volumes, 10000).astype(int)  # 确保成交量为正整数
            
            amounts = volumes * closes * np.random.uniform(0.01, 0.02, len(closes))
            
            # 创建DataFrame
            index_data = pd.DataFrame({
                'date': date_range,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'amount': amounts
            })
            
            # 计算技术指标
            index_data['ma5'] = index_data['close'].rolling(5).mean()
            index_data['ma10'] = index_data['close'].rolling(10).mean()
            index_data['ma20'] = index_data['close'].rolling(20).mean()
            
            # 添加涨跌幅
            index_data['price_change_pct'] = index_data['close'].pct_change()
            
            # 保存到缓存
            index_data.to_csv(cache_file, index=False)
            
            # 只返回请求的天数
            return index_data.tail(days)
            
        except Exception as e:
            logger.error(f"获取指数{index_code}数据失败: {e}")
            return pd.DataFrame()
    
    def get_north_bound_flow(self, days=30):
        """获取北向资金流向数据"""
        try:
            # 尝试从缓存加载
            cache_file = os.path.join(self.data_dir, "northbound_flow.json")
            now = datetime.now()
            
            # 判断是否需要更新缓存
            need_update = True
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                # 交易日期间工作时间每小时更新一次，非交易时间每天更新一次
                is_trading_hour = (now.hour >= 9 and now.hour <= 15) and now.weekday() < 5
                if is_trading_hour:
                    need_update = (now - file_time).seconds > 3600  # 1小时
                else:
                    need_update = (now - file_time).days > 0  # 1天
            
            if not need_update and os.path.exists(cache_file):
                logger.debug("从缓存加载北向资金数据")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            
            # 模拟数据（实际应用中应替换为真实数据API调用）
            logger.info("生成模拟北向资金数据")
            
            # 生成历史日期序列
            end_date = now.date()
            date_list = []
            for i in range(days):
                day = end_date - timedelta(days=i)
                # 跳过周末
                if day.weekday() < 5:
                    date_list.append(day.strftime("%Y-%m-%d"))
            date_list.reverse()
            
            # 生成北向资金每日净流入数据
            np.random.seed(int(now.timestamp()) % 10000)
            
            # 生成沪股通数据
            sh_flows = np.random.normal(0, 5e8, len(date_list))  # 5亿标准差
            # 生成深股通数据
            sz_flows = np.random.normal(0, 4e8, len(date_list))  # 4亿标准差
            
            # 添加趋势
            trend = np.random.choice([-1, 1]) * np.random.uniform(1e7, 5e7)  # 趋势因子
            for i in range(len(date_list)):
                sh_flows[i] += trend * i
                sz_flows[i] += trend * i
            
            # 合并数据
            combined_flow = []
            for i, date in enumerate(date_list):
                combined_flow.append({
                    'date': date,
                    'sh_flow': sh_flows[i],
                    'sz_flow': sz_flows[i],
                    'total_flow': sh_flows[i] + sz_flows[i]
                })
            
            # 生成当日十大活跃股
            active_stocks = []
            stock_codes = self.stock_basics['code'].sample(10).tolist()
            stock_names = []
            for code in stock_codes:
                idx = self.stock_basics[self.stock_basics['code'] == code].index[0]
                stock_names.append(self.stock_basics.loc[idx, 'name'])
            
            prices = np.random.uniform(10, 100, 10)
            changes = np.random.normal(0, 0.02, 10)
            net_buys = np.random.normal(0, 1e8, 10)
            buy_ratios = np.abs(net_buys) / (prices * np.random.uniform(1e6, 1e7, 10))
            
            for i in range(10):
                active_stocks.append({
                    'code': stock_codes[i],
                    'name': stock_names[i],
                    'price': prices[i],
                    'change': changes[i],
                    'net_buy': net_buys[i],
                    'buy_ratio': buy_ratios[i]
                })
            
            # 按净买入金额排序
            active_stocks.sort(key=lambda x: abs(x['net_buy']), reverse=True)
            
            # 创建股票流向映射
            stock_flows = {}
            for stock in active_stocks:
                stock_flows[stock['code']] = stock['net_buy']
            
            # 创建行业资金流向映射
            sector_flows = {}
            for sector in self.sector_mapping.keys():
                # 行业流向随机生成
                sector_flows[sector] = np.random.normal(0, 2e8)  # 2亿标准差
            
            # 整合数据
            flow_data = {
                'daily_flow': combined_flow,
                'top_stocks': active_stocks,
                'total_inflow': combined_flow[-1]['total_flow'],
                'stock_flows': stock_flows,
                'sector_flows': sector_flows,
                'date': now.strftime("%Y-%m-%d")
            }
            
            # 保存到缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(flow_data, f, ensure_ascii=False, indent=2)
                
            return flow_data
            
        except Exception as e:
            logger.error(f"获取北向资金数据失败: {e}")
            return {'total_inflow': 0, 'stock_flows': {}, 'sector_flows': {}}
    
    def get_sector_performance(self):
        """获取行业板块表现"""
        try:
            # 尝试从缓存加载
            cache_file = os.path.join(self.data_dir, "sector_performance.json")
            now = datetime.now()
            
            # 判断是否需要更新缓存
            need_update = True
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                # 交易日期间工作时间每小时更新一次，非交易时间每天更新一次
                is_trading_hour = (now.hour >= 9 and now.hour <= 15) and now.weekday() < 5
                if is_trading_hour:
                    need_update = (now - file_time).seconds > 3600  # 1小时
                else:
                    need_update = (now - file_time).days > 0  # 1天
            
            if not need_update and os.path.exists(cache_file):
                logger.debug("从缓存加载行业板块表现数据")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            
            # 模拟数据（实际应用中应替换为真实数据API调用）
            logger.info("生成模拟行业板块表现数据")
            
            # 获取北向资金数据，以获取行业流向
            north_flow = self.get_north_bound_flow()
            sector_flows = north_flow.get('sector_flows', {})
            
            # 生成行业表现数据
            sector_perf = {}
            for sector in self.sector_mapping.keys():
                # 涨跌幅受北向资金影响
                flow = sector_flows.get(sector, 0)
                flow_impact = flow / 1e9 * 0.01  # 每10亿影响涨跌幅1%
                
                # 行业涨跌幅
                price_change = np.random.normal(0, 0.02) + flow_impact
                
                # 其他指标
                turnover_rate = np.random.uniform(0.01, 0.05)  # 换手率
                pe_ratio = np.random.uniform(15, 40)  # 市盈率
                pb_ratio = np.random.uniform(1, 5)  # 市净率
                market_cap = np.random.uniform(1000, 10000)  # 市值（亿元）
                
                sector_perf[sector] = {
                    'price_change_pct': price_change,
                    'turnover_rate': turnover_rate,
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'market_cap': market_cap,
                    'north_bound_ratio': flow / (market_cap * 1e8) if market_cap > 0 else 0
                }
            
            # 保存到缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(sector_perf, f, ensure_ascii=False, indent=2)
                
            return sector_perf
            
        except Exception as e:
            logger.error(f"获取行业板块表现数据失败: {e}")
            return {}
    
    def get_market_status(self):
        """获取当前市场状态"""
        try:
            # 尝试从缓存加载
            cache_file = os.path.join(self.data_dir, "market_status.json")
            now = datetime.now()
            
            # 判断是否需要更新缓存
            need_update = True
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                # 交易日期间工作时间每15分钟更新一次，非交易时间每天更新一次
                is_trading_hour = (now.hour >= 9 and now.hour <= 15) and now.weekday() < 5
                if is_trading_hour:
                    need_update = (now - file_time).seconds > 900  # 15分钟
                else:
                    need_update = (now - file_time).days > 0  # 1天
            
            if not need_update and os.path.exists(cache_file):
                logger.debug("从缓存加载市场状态数据")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            
            # 获取上证指数
            sh_index = self.get_index_data("000001", days=1)
            # 获取深证成指
            sz_index = self.get_index_data("399001", days=1)
            # 获取创业板指
            cyb_index = self.get_index_data("399006", days=1)
            
            # 为模拟数据生成合理的涨跌家数
            np.random.seed(int(now.timestamp()) % 10000)
            
            total_stocks = 4000  # 假设市场有4000只股票
            # 上涨家数 - 基于大盘涨跌幅适当偏移
            if not sh_index.empty and 'price_change_pct' in sh_index.columns:
                market_change = sh_index['price_change_pct'].iloc[-1]
                # 大盘上涨时，上涨家数更多
                up_ratio = 0.5 + market_change * 10  # 基准50%，每涨1%增加10%的上涨家数
                up_ratio = max(0.2, min(0.8, up_ratio))  # 限制在20%-80%之间
            else:
                up_ratio = 0.5  # 默认50%
                
            up_count = int(total_stocks * up_ratio)
            down_count = total_stocks - up_count
            
            # 涨停和跌停家数
            limit_up_ratio = max(0.01, min(0.05, up_ratio * 0.1))  # 1%-5%
            limit_down_ratio = max(0.01, min(0.05, (1-up_ratio) * 0.1))  # 1%-5%
            
            limit_up_count = int(total_stocks * limit_up_ratio)
            limit_down_count = int(total_stocks * limit_down_ratio)
            
            # 获取历史成交量作为基准
            avg_volume = 5e11  # 默认5000亿
            avg_amount = 5e11  # 默认5000亿
            
            # 当日成交量和成交额
            volume_change = np.random.normal(0, 0.15)  # 相比平均值的变化
            current_volume = avg_volume * (1 + volume_change)
            current_amount = avg_amount * (1 + volume_change)
            
            # 获取北向资金数据
            north_flow = self.get_north_bound_flow()
            
            # 构建市场状态数据
            market_status = {
                'date': now.strftime("%Y-%m-%d"),
                'time': now.strftime("%H:%M:%S"),
                'sh_index': {
                    'close': sh_index['close'].iloc[-1] if not sh_index.empty else 3000,
                    'change_pct': sh_index['price_change_pct'].iloc[-1] if not sh_index.empty else 0
                },
                'sz_index': {
                    'close': sz_index['close'].iloc[-1] if not sz_index.empty else 10000,
                    'change_pct': sz_index['price_change_pct'].iloc[-1] if not sz_index.empty else 0
                },
                'cyb_index': {
                    'close': cyb_index['close'].iloc[-1] if not cyb_index.empty else 2000,
                    'change_pct': cyb_index['price_change_pct'].iloc[-1] if not cyb_index.empty else 0
                },
                'market_stats': {
                    'up_count': up_count,
                    'down_count': down_count,
                    'limit_up_count': limit_up_count,
                    'limit_down_count': limit_down_count,
                    'total_volume': current_volume,
                    'total_amount': current_amount
                },
                'avg_volume': avg_volume,
                'north_bound_flow': north_flow.get('total_inflow', 0)
            }
            
            # 保存到缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(market_status, f, ensure_ascii=False, indent=2)
                
            self.last_update_time = now
            return market_status
            
        except Exception as e:
            logger.error(f"获取市场状态数据失败: {e}")
            # 返回基本数据防止程序崩溃
            return {
                'date': now.strftime("%Y-%m-%d"),
                'time': now.strftime("%H:%M:%S"),
                'sh_index': {'close': 3000, 'change_pct': 0},
                'sz_index': {'close': 10000, 'change_pct': 0},
                'cyb_index': {'close': 2000, 'change_pct': 0},
                'market_stats': {
                    'up_count': 2000,
                    'down_count': 2000,
                    'limit_up_count': 50,
                    'limit_down_count': 50,
                    'total_volume': 5e11,
                    'total_amount': 5e11
                }
            } 