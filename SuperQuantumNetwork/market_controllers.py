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
import threading
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# 配置日志
logger = logging.getLogger(__name__)

# 全局线程池 - 用于优化并行请求
_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MarketController")

# 尝试导入数据分析和AI模块
try:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from quantum_ai import QuantumAIEngine
    HAS_QUANTUM_AI = True
except ImportError:
    logger.warning("无法导入量子AI模块，将使用基础分析")
    HAS_QUANTUM_AI = False
    # 创建模拟的QuantumAIEngine供应急使用
    class QuantumAIEngine:
        def __init__(self, config=None):
            self.config = config or {}
            logger.info("初始化模拟量子AI引擎")
        
        def analyze(self, data):
            import random
            return {
                "prediction": "上涨" if random.random() > 0.5 else "下跌",
                "confidence": random.uniform(0.6, 0.95),
                "trends": {
                    "short_term": random.choice(["强烈看涨", "看涨", "中性", "看跌"]),
                    "mid_term": random.choice(["强烈看涨", "看涨", "中性", "看跌"]),
                    "long_term": random.choice(["强烈看涨", "看涨", "中性", "看跌"]),
                }
            }
        
        def set_quantum_engine(self, engine):
            pass

try:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_sources import (
        get_index_data, 
        get_north_flow,
        get_sector_data,
        get_stock_data
    )
    HAS_DATA_SOURCE = True
except ImportError:
    logger.warning("无法导入市场数据源模块，将使用本地数据")
    HAS_DATA_SOURCE = False
    
    # 创建模拟数据函数
    def get_index_data():
        import random
        from datetime import datetime
        
        indices = {
            "000001.SH": {
                "name": "上证指数",
                "current": 3250.55 + random.uniform(-20, 20),
                "change": random.uniform(-1.5, 2.0),
                "volume": 32560000000 + random.uniform(-5000000000, 5000000000),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "399001.SZ": {
                "name": "深证成指",
                "current": 10765.31 + random.uniform(-50, 60),
                "change": random.uniform(-1.2, 2.2),
                "volume": 42580000000 + random.uniform(-6000000000, 6000000000),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "399006.SZ": {
                "name": "创业板指",
                "current": 2156.48 + random.uniform(-15, 25),
                "change": random.uniform(-1.8, 2.5),
                "volume": 15680000000 + random.uniform(-2000000000, 3000000000),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        return indices
    
    def get_north_flow():
        import random
        from datetime import datetime, timedelta
        
        today = datetime.now()
        data = []
        
        # 生成最近7天的数据
        for i in range(7):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            data.append({
                "date": date,
                "value": random.uniform(-5, 8) * 1000000000,  # 单位：亿元
                "accumulated": random.uniform(20, 50) * 1000000000
            })
        
        return data
    
    def get_sector_data():
        import random
        
        sectors = [
            {"name": "新能源", "change": random.uniform(0.5, 3.5)},
            {"name": "半导体", "change": random.uniform(0.8, 4.2)},
            {"name": "人工智能", "change": random.uniform(1.2, 4.8)},
            {"name": "医药生物", "change": random.uniform(-1.0, 2.8)},
            {"name": "金融", "change": random.uniform(-0.8, 1.5)},
            {"name": "消费", "change": random.uniform(-1.2, 1.8)},
            {"name": "储能", "change": random.uniform(0.2, 3.2)},
            {"name": "光伏", "change": random.uniform(-0.5, 2.5)}
        ]
        
        # 按涨幅排序
        sectors.sort(key=lambda x: x["change"], reverse=True)
        return sectors
    
    def get_stock_data(stock_code):
        import random
        from datetime import datetime, timedelta
        
        today = datetime.now()
        stock_name = f"模拟股票{stock_code[-4:]}"
        current_price = random.uniform(10, 100)
        
        return {
            "code": stock_code,
            "name": stock_name,
            "current": current_price,
            "change": random.uniform(-5, 8),
            "open": current_price * (1 - random.uniform(-0.02, 0.02)),
            "high": current_price * (1 + random.uniform(0, 0.05)),
            "low": current_price * (1 - random.uniform(0, 0.05)),
            "volume": random.uniform(10000, 1000000),
            "turnover": random.uniform(50000000, 5000000000),
            "pe": random.uniform(10, 50),
            "pb": random.uniform(1, 10),
            "update_time": today.strftime("%Y-%m-%d %H:%M:%S")
        }


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
        
        # 性能优化：缓存最近的数据以减少文件I/O
        self._data_cache = {}  # 键: 文件路径, 值: (数据, 时间戳)
        self._cache_lifetime = 300  # 缓存有效期(秒)
        self._last_save_time = 0  # 上次保存数据的时间
        self._min_save_interval = 60  # 最小保存间隔(秒)
        
        # 高级缓存策略
        self._memory_cache = {}  # 内存缓存
        self._memory_cache_ttl = {}  # 缓存过期时间
        self._last_update_time = 0  # 上次更新时间
        self._min_update_interval = 10  # 最小更新间隔(秒)
        
        # 延迟初始化标志
        self._ai_engine_initialized = False
        
        # 量子引擎引用 - 初始为None，通过set_quantum_engine设置
        self.quantum_engine = None
        
        # 初始化AI引擎 - 仅在明确请求时加载
        if self.use_ai:
            # 使用后台线程初始化AI引擎，避免阻塞UI
            self._ai_init_thread = Thread(target=self._initialize_ai_engine)
            self._ai_init_thread.daemon = True
            self._ai_init_thread.start()
        
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
    
    def set_quantum_engine(self, quantum_engine):
        """设置量子引擎实例，用于进行高级纠缠预测
        
        Args:
            quantum_engine: 量子引擎实例
        """
        self.quantum_engine = quantum_engine
        logger.info("市场数据控制器成功关联量子引擎")
        
        # 如果AI引擎已初始化，也为其设置量子引擎
        if hasattr(self, 'ai_engine') and self.ai_engine is not None:
            if hasattr(self.ai_engine, 'set_quantum_engine'):
                self.ai_engine.set_quantum_engine(quantum_engine)
                logger.info("AI引擎成功关联量子引擎")

    def _initialize_ai_engine(self):
        """延迟初始化AI引擎"""
        if not self._ai_engine_initialized and HAS_QUANTUM_AI:
            try:
                logger.info("开始初始化量子AI引擎...")
                self.ai_engine = QuantumAIEngine(config=self.config.get('ai_config', {}))
                self._ai_engine_initialized = True
                logger.info("量子AI引擎初始化成功")
            except Exception as e:
                logger.error(f"量子AI引擎初始化失败: {str(e)}")
                self.use_ai = False
    
    def enable_ai_prediction(self):
        """启用AI预测功能"""
        if not self._ai_engine_initialized:
            self.use_ai = True
            self._initialize_ai_engine()
            logger.info("启用AI预测功能")
    
    def update_market_data(self, force_save=False, force_update=False):
        """更新市场数据
        
        Args:
            force_save: 是否强制保存到磁盘
            force_update: 是否强制更新（忽略缓存）
        
        Returns:
            更新是否成功
        """
        logger.info("开始更新市场数据...")
        
        # 检查更新间隔，避免频繁请求
        current_time = time.time()
        if not force_update and (current_time - self._last_update_time) < self._min_update_interval:
            logger.info(f"距离上次更新不足{self._min_update_interval}秒，使用缓存数据")
            return True
        
        with self.data_lock:
            try:
                # 使用线程池并行获取各类数据
                futures = []
                
                # 获取市场指数数据
                futures.append(_THREAD_POOL.submit(self._fetch_market_indices))
                
                # 获取北向资金数据
                futures.append(_THREAD_POOL.submit(self._fetch_north_flow))
                
                # 获取板块数据
                futures.append(_THREAD_POOL.submit(self._fetch_sector_data))
                
                # 等待所有数据获取完成
                for future in as_completed(futures):
                    try:
                        data_type, data = future.result()
                        if data_type == 'indices':
                            # 更新市场指数数据
                            self.market_data.update(data)
                        elif data_type == 'north_flow':
                            # 更新北向资金数据
                            self.latest_prediction['north_flow'] = data
                        elif data_type == 'sectors':
                            # 更新板块数据
                            self.market_data['sectors'] = data
                    except Exception as e:
                        logger.error(f"处理数据更新任务异常: {str(e)}")
                
                # 更新时间戳
                self._last_update_time = current_time
                
                # 保存数据到本地（根据条件）
                if force_save or (current_time - self._last_save_time) > self._min_save_interval:
                    self._save_market_data()
                    self._last_save_time = current_time
                
                logger.info("市场数据更新完成")
                return True
            except Exception as e:
                logger.error(f"更新市场数据失败: {str(e)}")
                logger.error(traceback.format_exc())
                return False
    
    def _fetch_market_indices(self):
        """获取市场指数数据（线程池工作函数）"""
        try:
            # 检查缓存
            cache_key = 'market_indices'
            if cache_key in self._memory_cache and time.time() < self._memory_cache_ttl.get(cache_key, 0):
                logger.info("使用缓存的市场指数数据")
                return 'indices', self._memory_cache[cache_key]
            
            if HAS_DATA_SOURCE:
                # 并行获取三大指数数据
                indices_futures = []
                indices_futures.append(_THREAD_POOL.submit(get_index_data, '000001.SH'))
                indices_futures.append(_THREAD_POOL.submit(get_index_data, '399001.SZ'))
                indices_futures.append(_THREAD_POOL.submit(get_index_data, '399006.SZ'))
                
                # 获取结果
                sh_data, sz_data, cyb_data = None, None, None
                for i, future in enumerate(as_completed(indices_futures)):
                    try:
                        result = future.result()
                        if '000001.SH' in str(result.get('code', '')):
                            sh_data = result
                        elif '399001.SZ' in str(result.get('code', '')):
                            sz_data = result
                        elif '399006.SZ' in str(result.get('code', '')):
                            cyb_data = result
                    except Exception as e:
                        logger.error(f"获取指数数据异常: {str(e)}")
                
                # 如果有缺失数据，使用模拟数据填充
                if not sh_data:
                    sh_data = self._generate_mock_index_data('上证指数')
                if not sz_data:
                    sz_data = self._generate_mock_index_data('深证成指')
                if not cyb_data:
                    cyb_data = self._generate_mock_index_data('创业板指')
            else:
                # 使用模拟数据
                sh_data = self._generate_mock_index_data('上证指数')
                sz_data = self._generate_mock_index_data('深证成指')
                cyb_data = self._generate_mock_index_data('创业板指')
            
            # 组装数据
            indices_data = {
                'sh_index': sh_data,
                'sz_index': sz_data,
                'cyb_index': cyb_data
            }
            
            # 更新缓存
            self._memory_cache[cache_key] = indices_data
            self._memory_cache_ttl[cache_key] = time.time() + 60  # 缓存1分钟
            
            logger.info("市场指数数据获取成功")
            return 'indices', indices_data
        except Exception as e:
            logger.error(f"获取市场指数数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回空数据而不是抛出异常，以避免整个更新过程失败
            return 'indices', {}
    
    def _fetch_north_flow(self):
        """获取北向资金数据（线程池工作函数）"""
        try:
            # 检查缓存
            cache_key = 'north_flow'
            if cache_key in self._memory_cache and time.time() < self._memory_cache_ttl.get(cache_key, 0):
                logger.info("使用缓存的北向资金数据")
                return 'north_flow', self._memory_cache[cache_key]
                
            if HAS_DATA_SOURCE:
                # 获取北向资金数据
                north_flow = get_north_flow()
            else:
                # 使用模拟数据
                north_flow = self._generate_mock_north_flow()
            
            # 更新缓存
            self._memory_cache[cache_key] = north_flow
            self._memory_cache_ttl[cache_key] = time.time() + 300  # 缓存5分钟
            
            logger.info("北向资金数据获取成功")
            return 'north_flow', north_flow
        except Exception as e:
            logger.error(f"获取北向资金数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回空数据
            return 'north_flow', {}
    
    def _fetch_sector_data(self):
        """获取板块数据（线程池工作函数）"""
        try:
            # 检查缓存
            cache_key = 'sectors'
            if cache_key in self._memory_cache and time.time() < self._memory_cache_ttl.get(cache_key, 0):
                logger.info("使用缓存的板块数据")
                return 'sectors', self._memory_cache[cache_key]
                
            if HAS_DATA_SOURCE:
                # 获取板块数据
                sector_data = get_sector_data()
            else:
                # 使用模拟数据
                sector_data = self._generate_mock_sector_data()
            
            # 更新缓存
            self._memory_cache[cache_key] = sector_data
            self._memory_cache_ttl[cache_key] = time.time() + 300  # 缓存5分钟
            
            logger.info("板块数据获取成功")
            return 'sectors', sector_data
        except Exception as e:
            logger.error(f"获取板块数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回空数据
            return 'sectors', {}
    
    def _update_market_indices(self):
        """更新市场指数数据（兼容旧代码）"""
        data_type, indices_data = self._fetch_market_indices()
        self.market_data.update(indices_data)
    
    def _update_north_flow(self):
        """更新北向资金数据（兼容旧代码）"""
        data_type, north_flow = self._fetch_north_flow()
        self.latest_prediction['north_flow'] = north_flow
    
    def _update_sector_data(self):
        """更新板块数据（兼容旧代码）"""
        data_type, sector_data = self._fetch_sector_data()
        self.market_data['sectors'] = sector_data
    
    def _save_market_data(self):
        """保存市场数据到本地"""
        try:
            # 创建文件名 (使用日期)
            date_str = datetime.now().strftime('%Y%m%d')
            file_path = os.path.join(self.data_dir, f'market_data_{date_str}.json')
            
            # 异步保存数据
            def _async_save():
                try:
                    # 创建临时副本以避免长时间锁定
                    with self.data_lock:
                        data_copy = self.market_data.copy()
                    
                    # 保存数据
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data_copy, f, ensure_ascii=False, indent=2)
                    
                    # 更新缓存
                    self._data_cache[file_path] = (data_copy, time.time())
                    
                    logger.info(f"市场数据保存到: {file_path}")
                except Exception as e:
                    logger.error(f"异步保存市场数据失败: {str(e)}")
            
            # 启动异步保存线程
            save_thread = Thread(target=_async_save)
            save_thread.daemon = True
            save_thread.start()
        except Exception as e:
            logger.error(f"保存市场数据失败: {str(e)}")
            logger.error(traceback.format_exc())
    @lru_cache(maxsize=32)
    def _load_data_from_file(self, file_path):
        """从文件加载数据，使用Python内置LRU缓存优化性能"""
        # 从文件加载
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"加载文件失败: {file_path} - {str(e)}")
            return None
        
    def predict_market_trend(self, use_quantum=True):
        """预测市场趋势
        
        Args:
            use_quantum: 是否使用量子引擎增强预测
        
        Returns:
            预测结果，包含市场趋势、风险等级等信息
        """
        try:
            prediction_result = {}
            
            # 如果启用AI并且AI引擎已初始化，使用AI预测
            if self.use_ai and self._ai_engine_initialized:
                prediction_result = self.ai_engine.predict_market(self.market_data)
                logger.info("使用AI引擎完成市场预测")
            else:
                # 使用基本分析进行预测
                prediction_result = self._basic_market_prediction()
                logger.info("使用基本分析完成市场预测")
            
            # 如果启用量子引擎并且量子引擎可用，增强预测
            if use_quantum and self.quantum_engine is not None:
                prediction_result = self._enhance_prediction_with_quantum(prediction_result)
                logger.info("使用量子引擎增强市场预测")
                
            # 保存预测结果
            self.latest_prediction = prediction_result
            
            # 保存预测结果到文件
            self._save_prediction(prediction_result)
                
            logger.info("市场趋势预测完成")
            return prediction_result
        except Exception as e:
            logger.error(f"市场趋势预测失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 返回基本预测
            return {
                'status': 'error',
                'message': str(e),
                'risk_level': 0.5,
                'market_trend': '未知',
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'is_quantum_enhanced': False
            }
    
    def _enhance_prediction_with_quantum(self, basic_prediction):
        """使用量子引擎增强预测结果
        
        Args:
            basic_prediction: 基本预测结果
        
        Returns:
            增强后的预测结果
        """
        try:
            # 从基本预测中获取关键指标
            risk_level = basic_prediction.get('risk_level', 0.5)
            market_trend = basic_prediction.get('market_trend', '未知')
            
            # 准备市场实体
            market_entities = ['000001.SH', '399001.SZ', '399006.SZ']
            
            # 如果量子引擎有market_resonance方法，使用它计算市场共振
            if hasattr(self.quantum_engine, 'calculate_market_resonance'):
                resonance = self.quantum_engine.calculate_market_resonance(market_entities)
                logger.info(f"量子市场共振系数: {resonance:.4f}")
                
                # 修改风险评估
                quantum_risk_adjustment = (resonance - 0.5) * 0.3  # 量子调整因子
                risk_level = max(0.1, min(0.9, risk_level - quantum_risk_adjustment))
                
                # 如果量子引擎有predict_future方法，使用它进行前瞻性预测
                if hasattr(self.quantum_engine, 'predict_future'):
                    future_states = self.quantum_engine.predict_future(
                        market_entities, 
                        steps=5,
                        current_state=self.market_data
                    )
                    
                    # 分析未来状态
                    if future_states and len(future_states) > 0:
                        future_trends = []
                        for state in future_states:
                            if isinstance(state, dict) and 'trend' in state:
                                future_trends.append(state['trend'])
                        
                        # 如果有未来趋势数据，使用它们调整市场趋势预测
                        if future_trends:
                            # 计算未来趋势的加权平均（近期权重高）
                            weights = [0.4, 0.3, 0.15, 0.1, 0.05][:len(future_trends)]
                            norm_weights = [w/sum(weights) for w in weights]
                            
                            avg_future_trend = sum(t * w for t, w in zip(future_trends, norm_weights))
                            
                            # 根据未来趋势调整市场趋势预测
                            trend_map = {
                                '强势上涨': 2,
                                '温和上涨': 1,
                                '横盘震荡': 0,
                                '温和下跌': -1,
                                '强势下跌': -2
                            }
                            
                            # 当前趋势的数值表示
                            current_trend_value = trend_map.get(market_trend, 0)
                            
                            # 融合当前趋势和未来趋势（70%当前，30%未来）
                            combined_trend_value = current_trend_value * 0.7 + avg_future_trend * 0.3
                            
                            # 映射回趋势描述
                            reverse_trend_map = {v: k for k, v in trend_map.items()}
                            # 找到最接近的趋势值
                            closest_trend = min(reverse_trend_map.keys(), key=lambda x: abs(x - combined_trend_value))
                            market_trend = reverse_trend_map[closest_trend]
            
            # 创建增强的预测结果
            enhanced_prediction = basic_prediction.copy()
            enhanced_prediction.update({
                'risk_level': risk_level,
                'market_trend': market_trend,
                'is_quantum_enhanced': True,
                'quantum_enhancement_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # 如果基本预测包含详细预测，调整它们
            if 'detailed_predictions' in basic_prediction:
                detailed = basic_prediction['detailed_predictions']
                
                # 遍历并调整每个详细预测
                for entity, pred in detailed.items():
                    if isinstance(pred, dict) and 'prediction' in pred:
                        # 应用量子调整
                        quantum_factor = self._calculate_quantum_factor(entity)
                        pred['prediction'] = pred['prediction'] * (1 + quantum_factor * 0.2)
                        pred['is_quantum_enhanced'] = True
                
                enhanced_prediction['detailed_predictions'] = detailed
            
            return enhanced_prediction
        except Exception as e:
            logger.error(f"量子增强预测失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回原始预测
            return basic_prediction
    
    def _calculate_quantum_factor(self, entity):
        """计算特定实体的量子因子
        
        Args:
            entity: 市场实体标识符
        
        Returns:
            量子因子 (-0.2 to 0.2)
        """
        try:
            if self.quantum_engine is None:
                return 0.0
                
            # 如果量子引擎有get_entity_phase方法，使用它
            if hasattr(self.quantum_engine, 'get_entity_phase'):
                phase = self.quantum_engine.get_entity_phase(entity)
                return 0.2 * np.sin(phase)
            
            # 如果量子引擎有get_quantum_state方法，使用它
            elif hasattr(self.quantum_engine, 'get_quantum_state'):
                state = self.quantum_engine.get_quantum_state(entity)
                if isinstance(state, dict) and 'phase' in state:
                    return 0.2 * np.sin(state['phase'])
            
            return 0.0
        except Exception:
            return 0.0
    
    def _basic_market_prediction(self):
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
    
    def _save_prediction(self, prediction):
        """保存预测结果"""
        try:
            # 创建文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.data_dir, f'prediction_{timestamp}.json')
            
            # 保存数据 - 使用基本格式以提高性能
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(prediction, f, ensure_ascii=False)
            
            logger.info(f"预测结果保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存预测结果失败: {str(e)}")
            # 不抛出异常，因为这是非关键操作
    
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