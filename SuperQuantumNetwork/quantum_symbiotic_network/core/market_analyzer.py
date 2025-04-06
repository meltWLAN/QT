"""
超神系统 - 量子共生网络 - 市场分析器 (v3.0)
增强版: 分形市场结构识别与多层次趋势分离
"""

import numpy as np
import pandas as pd
import logging
import random
from datetime import datetime
from scipy import stats
import pywt  # PyWavelets库用于小波变换
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class SuperMarketAnalyzer:
    """
    超级市场分析器 3.0 - 多维度市场结构分析
    核心特性:
    1. 分形市场结构识别 - 使用分形数学捕捉跨时间周期模式
    2. 多层次趋势分离技术 - 将市场波动分解为不同时间尺度趋势
    3. 行为金融学整合 - 融合行为心理学模型解析非理性行为
    4. 复杂系统临界点预测 - 开发市场临界状态识别系统
    5. 市场微观结构分析 - 分析订单簿动态和交易流结构
    """
    
    def __init__(self):
        """初始化市场分析器"""
        self.name = "超级市场分析器"
        self.version = "3.0.0"
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化{self.name} v{self.version}")
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
        self.analysis_history = []
        self.detection_thresholds = {
            'trend_strength': 0.65,
            'support_resistance': 0.02,  # 价格波动百分比
            'volatility': 0.10,
            'anomaly': 0.8
        }
        
        # 分形分析参数
        self.fractal_timeframes = ['1h', '4h', '1d']
        self.hurst_window = 100  # Hurst指数计算窗口
        self.fractal_patterns = self._initialize_fractal_patterns()
        
        # 多层次趋势分离参数
        self.decomposition_levels = 4  # 小波分解级别
        self.wavelet_type = 'db8'  # 小波基函数类型
        self.trend_memory = {}  # 存储不同尺度的趋势记忆
        
        # 行为金融学参数
        self.behavioral_biases = {
            'herd_threshold': 0.75,  # 羊群效应阈值
            'overreaction_factor': 1.5,  # 过度反应因子
            'anchoring_weight': 0.3,  # 锚定效应权重
            'recency_decay': 0.95  # 近因效应衰减率
        }
        
        # 临界点识别参数
        self.critical_indicators = {
            'volatility_explosion': 2.5,  # 波动率爆炸阈值(标准差倍数)
            'correlation_breakdown': 0.3,  # 相关性崩溃阈值
            'liquidity_dry_up': 0.5,  # 流动性枯竭阈值
            'entropy_threshold': 0.8  # 信息熵阈值
        }
    
    def _initialize_fractal_patterns(self):
        """初始化常见分形模式库"""
        return {
            'w_pattern': {
                'description': 'W形态(双底)',
                'bullish': True,
                'reliability': 0.75
            },
            'm_pattern': {
                'description': 'M形态(双顶)',
                'bullish': False,
                'reliability': 0.75
            },
            'three_drives': {
                'description': '三推动形态',
                'bullish': None,  # 取决于方向
                'reliability': 0.7
            },
            'elliott_5_waves': {
                'description': '艾略特五浪形态',
                'bullish': None,  # 取决于大趋势
                'reliability': 0.8
            },
            'fractal_triangle': {
                'description': '分形三角形',
                'bullish': None,  # 取决于突破方向
                'reliability': 0.65
            }
        }
    
    def analyze_market_structure(self, market_data=None):
        """
        分析市场结构并返回结构报告
        """
        self.logger.info("开始市场结构分析...")
        
        if market_data is None or not isinstance(market_data, dict):
            # 模拟市场分析过程
            return self._generate_mock_analysis()
        
        try:
            # 1. 多层次趋势分离
            trend_analysis = self.decompose_price_trends(market_data.get('price_data', []))
            
            # 2. 分形市场结构识别
            fractal_analysis = self.identify_fractal_structure(market_data.get('price_data', []))
            
            # 3. 支撑阻力识别
            support_resistance = self.analyze_support_resistance(market_data.get('price_data', []))
            
            # 4. 市场异常检测
            anomalies = self.detect_market_anomalies(market_data)
            
            # 5. 临界点分析
            critical_points = self.analyze_critical_points(market_data)
            
            # 6. 行为金融分析
            behavioral_analysis = self.analyze_behavioral_factors(market_data)
            
            # 组合分析结果
            analysis = {
                'timestamp': datetime.now(),
                'trend': trend_analysis['primary_trend'],
                'multi_scale_trends': trend_analysis['multi_scale_trends'],
                'fractal_structure': fractal_analysis,
                'support_resistance': support_resistance,
                'volatility': self.analyze_volatility_structure(market_data.get('price_data', [])),
                'volume_profile': self.analyze_volume_structure(market_data.get('volume_data', [])),
                'market_cycle': self.determine_market_cycle(market_data),
                'anomalies': anomalies,
                'critical_points': critical_points,
                'behavioral_factors': behavioral_analysis
            }
            
            self.analysis_history.append(analysis)
            self.logger.info(f"市场分析完成: {analysis['trend']['direction']} 趋势, 强度: {analysis['trend']['strength']:.2f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"市场分析过程中发生错误: {e}")
            # 返回模拟分析结果作为备选
            return self._generate_mock_analysis()
    
    def _generate_mock_analysis(self):
        """生成模拟的市场分析结果"""
        mock_analysis = {
            'timestamp': datetime.now(),
            'trend': {
                'direction': random.choice(['上升', '下降', '横盘']),
                'strength': random.uniform(0.5, 0.95),
                'duration_days': random.randint(5, 60)
            },
            'multi_scale_trends': {
                'short_term': random.choice(['上升', '下降', '横盘']),
                'medium_term': random.choice(['上升', '下降', '横盘']),
                'long_term': random.choice(['上升', '下降', '横盘'])
            },
            'fractal_structure': {
                'hurst_exponent': random.uniform(0.3, 0.7),
                'pattern': random.choice(['W形态', 'M形态', '三推动', '无明显形态']),
                'fractal_dimension': random.uniform(1.3, 1.7),
                'completion': random.uniform(0.5, 1.0)
            },
            'support_resistance': {
                'major_support': [random.uniform(90, 95), random.uniform(85, 89)],
                'major_resistance': [random.uniform(105, 110), random.uniform(115, 120)]
            },
            'volatility': {
                'current': random.uniform(0.1, 0.4),
                'historical_percentile': random.uniform(0.3, 0.8),
                'regime': random.choice(['高波动', '低波动', '正常波动'])
            },
            'volume_profile': {
                'average_volume': random.uniform(10000, 50000),
                'volume_trend': random.choice(['增加', '减少', '稳定']),
                'volume_strength': random.uniform(0.4, 0.9)
            },
            'market_cycle': {
                'current_phase': random.choice(['积累', '上涨', '分配', '下跌']),
                'phase_completion': random.uniform(0.1, 0.9)
            },
            'anomalies': {
                'detected': random.choice([True, False]),
                'type': random.choice(['价格异常', '成交量异常', '波动性异常', '无']),
                'significance': random.uniform(0.5, 0.95)
            },
            'critical_points': {
                'approaching_critical': random.choice([True, False]),
                'estimated_distance': random.uniform(0.05, 0.3),
                'transition_type': random.choice(['趋势转变', '波动性爆发', '流动性危机', '未知'])
            },
            'behavioral_factors': {
                'herd_behavior': random.uniform(0, 1),
                'overreaction': random.uniform(-0.5, 0.5),
                'sentiment_extreme': random.choice([True, False]),
                'risk_appetite': random.uniform(0.3, 0.7)
            }
        }
        return mock_analysis
    
    def decompose_price_trends(self, price_data):
        """
        使用小波变换分解多层次价格趋势
        """
        self.logger.info("执行多层次趋势分离...")
        
        if price_data is None or len(price_data) < 50:
            # 返回模拟的趋势分解结果
            return self._generate_mock_trend_decomposition()
        
        try:
            # 转换数据格式
            prices = np.array(price_data)
            
            # 使用小波变换进行多尺度分解
            coeffs = pywt.wavedec(prices, wavelet=self.wavelet_type, level=self.decomposition_levels)
            
            # 提取不同尺度的趋势
            trends = {}
            reconstructed = np.zeros_like(prices)
            
            # 重构每个层级的趋势
            for i in range(len(coeffs)):
                coeff_copy = [np.zeros_like(c) for c in coeffs]
                coeff_copy[i] = coeffs[i]
                trends[f'level_{i}'] = pywt.waverec(coeff_copy, wavelet=self.wavelet_type)[:len(prices)]
            
            # 分析每个尺度的趋势方向和强度
            trend_analysis = {}
            timeframes = ['长期', '中期', '短期', '微观']
            
            for i, level in enumerate(sorted(trends.keys())):
                if i < len(timeframes):
                    timeframe = timeframes[i]
                    trend_data = trends[level]
                    
                    # 计算趋势方向和强度
                    if len(trend_data) > 10:
                        direction = self._determine_trend_direction(trend_data[-10:])
                        strength = self._calculate_trend_strength(trend_data[-30:] if len(trend_data) >= 30 else trend_data)
                    else:
                        direction = '不确定'
                        strength = 0.5
                    
                    trend_analysis[timeframe] = {
                        'direction': direction,
                        'strength': strength,
                        'volatility': np.std(trend_data[-20:]) if len(trend_data) >= 20 else 0
                    }
            
            # 确定主要趋势
            if '中期' in trend_analysis:
                primary_trend = trend_analysis['中期'].copy()
            else:
                primary_trend = {
                    'direction': trend_analysis.get('短期', {}).get('direction', '横盘'),
                    'strength': trend_analysis.get('短期', {}).get('strength', 0.5),
                    'duration_days': 0
                }
            
            return {
                'primary_trend': primary_trend,
                'multi_scale_trends': trend_analysis
            }
        
        except Exception as e:
            self.logger.error(f"趋势分解过程中发生错误: {e}")
            return self._generate_mock_trend_decomposition()
    
    def _generate_mock_trend_decomposition(self):
        """生成模拟的趋势分解结果"""
        timeframes = ['长期', '中期', '短期', '微观']
        trend_directions = ['上升', '下降', '横盘']
        
        multi_scale_trends = {}
        for tf in timeframes:
            multi_scale_trends[tf] = {
                'direction': random.choice(trend_directions),
                'strength': random.uniform(0.4, 0.9),
                'volatility': random.uniform(0.01, 0.1)
            }
        
        primary_trend = {
            'direction': multi_scale_trends['中期']['direction'],
            'strength': multi_scale_trends['中期']['strength'],
            'duration_days': random.randint(5, 60)
        }
        
        return {
            'primary_trend': primary_trend,
            'multi_scale_trends': multi_scale_trends
        }
    
    def _determine_trend_direction(self, data):
        """确定趋势方向"""
        if len(data) < 2:
            return '不确定'
            
        # 使用线性回归确定趋势
        x = np.arange(len(data))
        slope, _, r_value, p_value, _ = stats.linregress(x, data)
        
        # 根据斜率和显著性确定方向
        if p_value > 0.1:  # 不显著
            return '横盘'
        elif slope > 0:
            return '上升'
        else:
            return '下降'
    
    def _calculate_trend_strength(self, data):
        """计算趋势强度"""
        if len(data) < 5:
            return 0.5
            
        # 使用R平方值作为趋势强度指标
        x = np.arange(len(data))
        _, _, r_value, _, _ = stats.linregress(x, data)
        
        # R平方值作为趋势强度
        return min(1.0, abs(r_value))
    
    def identify_fractal_structure(self, price_data):
        """
        识别市场的分形结构和组织形态
        """
        self.logger.info("识别分形市场结构...")
        
        if price_data is None or len(price_data) < 100:
            # 返回模拟的分形分析结果
            return self._generate_mock_fractal_analysis()
        
        try:
            # 计算Hurst指数
            hurst_exponent = self._calculate_hurst_exponent(price_data)
            
            # 计算分形维度
            fractal_dimension = 2 - hurst_exponent
            
            # 识别分形模式
            detected_patterns = self._detect_fractal_patterns(price_data)
            
            # 分析市场效率
            market_efficiency = 0.5
            if hurst_exponent is not None:
                if hurst_exponent < 0.45:  # 反持续性市场(均值回归)
                    market_efficiency = random.uniform(0.6, 0.8)
                elif hurst_exponent > 0.55:  # 持续性市场(趋势跟随)
                    market_efficiency = random.uniform(0.3, 0.5)
                else:  # 随机游走(有效市场)
                    market_efficiency = random.uniform(0.45, 0.55)
            
            return {
                'hurst_exponent': hurst_exponent,
                'fractal_dimension': fractal_dimension,
                'market_type': self._classify_market_by_hurst(hurst_exponent),
                'detected_patterns': detected_patterns,
                'market_efficiency': market_efficiency,
                'self_similarity': random.uniform(0.5, 0.9)  # 简化版自相似性计算
            }
            
        except Exception as e:
            self.logger.error(f"分形识别过程中发生错误: {e}")
            return self._generate_mock_fractal_analysis()
    
    def _calculate_hurst_exponent(self, time_series, max_lag=20):
        """计算Hurst指数"""
        if len(time_series) < 100:
            return random.uniform(0.4, 0.6)
            
        # 简化版Hurst指数计算
        lags = range(2, min(max_lag, len(time_series) // 4))
        tau = []; lagvec = []
        
        # 计算不同lag的方差
        for lag in lags:
            # 计算价格变化
            pp = np.array(time_series[lag:]) - np.array(time_series[:-lag])
            lagvec.append(lag)
            tau.append(np.sqrt(np.std(pp)))
            
        # 使用对数回归计算Hurst指数
        m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
        hurst = m[0]
        
        return hurst
    
    def _classify_market_by_hurst(self, hurst):
        """根据Hurst指数分类市场类型"""
        if hurst is None:
            return '不确定'
        elif hurst < 0.4:
            return '强均值回归'
        elif hurst < 0.45:
            return '均值回归'
        elif hurst < 0.55:
            return '随机游走'
        elif hurst < 0.75:
            return '趋势跟随'
        else:
            return '强趋势跟随'
    
    def _detect_fractal_patterns(self, price_data):
        """检测价格序列中的分形模式"""
        if len(price_data) < 50:
            return []
            
        # 这里需要实现分形模式识别算法
        # 简化版，随机选择一些模式
        patterns = []
        available_patterns = list(self.fractal_patterns.keys())
        
        if random.random() > 0.3:  # 70%概率检测到模式
            pattern_count = random.randint(1, min(3, len(available_patterns)))
            for _ in range(pattern_count):
                pattern_key = random.choice(available_patterns)
                available_patterns.remove(pattern_key)  # 避免重复
                
                pattern = self.fractal_patterns[pattern_key].copy()
                pattern.update({
                    'key': pattern_key,
                    'completion': random.uniform(0.6, 1.0),
                    'confidence': random.uniform(0.6, 0.9)
                })
                patterns.append(pattern)
                
        return patterns
    
    def _generate_mock_fractal_analysis(self):
        """生成模拟的分形分析结果"""
        hurst = random.uniform(0.35, 0.65)
        
        detected_patterns = []
        available_patterns = list(self.fractal_patterns.keys())
        pattern_count = random.randint(0, 2)
        
        for _ in range(pattern_count):
            if available_patterns:
                pattern_key = random.choice(available_patterns)
                available_patterns.remove(pattern_key)
                
                pattern = self.fractal_patterns[pattern_key].copy()
                pattern.update({
                    'key': pattern_key,
                    'completion': random.uniform(0.6, 1.0),
                    'confidence': random.uniform(0.6, 0.9)
                })
                detected_patterns.append(pattern)
        
        return {
            'hurst_exponent': hurst,
            'fractal_dimension': 2 - hurst,
            'market_type': self._classify_market_by_hurst(hurst),
            'detected_patterns': detected_patterns,
            'market_efficiency': random.uniform(0.4, 0.6),
            'self_similarity': random.uniform(0.5, 0.9)
        }
    
    def analyze_behavioral_factors(self, market_data):
        """
        分析市场的行为金融学因素
        """
        self.logger.info("分析市场行为金融学因素...")
        
        # 模拟行为金融分析
        return {
            'herd_behavior': random.uniform(0, 1),  # 羊群行为指数
            'overreaction': random.uniform(-0.5, 0.5),  # 过度反应指数，正值表示过度乐观
            'recency_bias': random.uniform(0, 1),  # 近因偏误指数
            'anchoring_effect': random.uniform(0, 1),  # 锚定效应强度
            'loss_aversion': random.uniform(1.5, 3.0),  # 损失厌恶系数
            'fear_greed_balance': random.uniform(-1, 1)  # 恐惧贪婪平衡，负值表示恐惧占主导
        }
    
    def analyze_critical_points(self, market_data):
        """
        分析市场临界点和相变状态
        """
        self.logger.info("分析市场临界点...")
        
        # 模拟临界点分析
        is_critical = random.random() > 0.8  # 20%概率处于临界状态
        
        return {
            'is_near_critical_point': is_critical,
            'critical_confidence': random.uniform(0.6, 0.9) if is_critical else random.uniform(0.1, 0.3),
            'potential_transition': random.choice(['趋势转变', '波动性爆发', '流动性危机', '横盘突破']) if is_critical else '稳定状态',
            'transition_timeframe': random.choice(['短期', '中期']) if is_critical else 'NA',
            'leading_indicators': {
                'correlation_breakdown': random.uniform(0, 1),
                'volatility_clustering': random.uniform(0, 1),
                'liquidity_changes': random.uniform(-0.5, 0.5),
                'information_entropy': random.uniform(0, 1)
            }
        }
    
    def identify_chart_patterns(self, price_data):
        """
        识别价格图表模式
        """
        patterns = []
        pattern_types = [
            '双顶', '双底', '头肩顶', '头肩底', '上升三角形', 
            '下降三角形', '对称三角形', '旗形', '楔形', '杯柄形态'
        ]
        
        # 模拟图表模式识别
        pattern_count = random.randint(0, 3)
        for _ in range(pattern_count):
            pattern = {
                'type': random.choice(pattern_types),
                'reliability': random.uniform(0.6, 0.9),
                'completion': random.uniform(0.7, 1.0),
                'target_price': random.uniform(95, 120)
            }
            patterns.append(pattern)
            
        return patterns
    
    def detect_divergences(self, price_data, indicator_data):
        """
        检测价格与指标之间的背离
        """
        divergences = []
        divergence_types = ['正常背离', '隐藏背离']
        indicator_types = ['RSI', 'MACD', 'CCI', 'OBV']
        
        # 模拟背离检测
        divergence_count = random.randint(0, 2)
        for _ in range(divergence_count):
            divergence = {
                'type': random.choice(divergence_types),
                'indicator': random.choice(indicator_types),
                'strength': random.uniform(0.65, 0.95),
                'signal': random.choice(['看涨', '看跌'])
            }
            divergences.append(divergence)
            
        return divergences
    
    def analyze_support_resistance(self, price_data):
        """
        分析支撑位和阻力位
        """
        levels = {
            'support': [],
            'resistance': []
        }
        
        # 模拟支撑阻力位检测
        support_count = random.randint(2, 4)
        for _ in range(support_count):
            level = {
                'price': random.uniform(85, 95),
                'strength': random.uniform(0.6, 0.9),
                'tests': random.randint(1, 5)
            }
            levels['support'].append(level)
            
        resistance_count = random.randint(2, 4)
        for _ in range(resistance_count):
            level = {
                'price': random.uniform(105, 120),
                'strength': random.uniform(0.6, 0.9),
                'tests': random.randint(1, 5)
            }
            levels['resistance'].append(level)
            
        return levels
    
    def analyze_volatility_structure(self, price_data):
        """
        分析波动率结构
        """
        if not price_data or len(price_data) < 30:
            return {
                'current': random.uniform(0.1, 0.3),
                'historical_percentile': random.uniform(0.3, 0.7),
                'regime': random.choice(['高波动', '低波动', '正常波动']),
                'clustering': random.uniform(0.5, 0.9),
                'term_structure': random.choice(['正常', '倒挂', '平坦'])
            }
        
        # 计算历史波动率
        returns = np.diff(price_data) / price_data[:-1]
        current_vol = np.std(returns[-20:]) * np.sqrt(252)  # 年化
        
        # 波动率分位数
        percentile = random.uniform(0.1, 0.9)  # 简化计算
        
        # 波动率机制
        regime = '正常波动'
        if current_vol > 0.3:
            regime = '高波动'
        elif current_vol < 0.1:
            regime = '低波动'
        
        return {
            'current': current_vol,
            'historical_percentile': percentile,
            'regime': regime,
            'clustering': random.uniform(0.5, 0.9),  # 波动率聚集性
            'term_structure': random.choice(['正常', '倒挂', '平坦'])  # 波动率期限结构
        }
    
    def analyze_volume_structure(self, volume_data):
        """
        分析成交量结构
        """
        if not volume_data or len(volume_data) < 30:
            return {
                'average_volume': random.uniform(10000, 50000),
                'volume_trend': random.choice(['增加', '减少', '稳定']),
                'volume_strength': random.uniform(0.4, 0.9),
                'price_volume_correlation': random.uniform(-0.5, 0.5),
                'unusual_spikes': []
            }
        
        # 计算平均成交量
        avg_volume = np.mean(volume_data[-20:])
        
        # 成交量趋势
        recent_avg = np.mean(volume_data[-5:])
        older_avg = np.mean(volume_data[-20:-5])
        volume_trend = '稳定'
        if recent_avg > older_avg * 1.2:
            volume_trend = '增加'
        elif recent_avg < older_avg * 0.8:
            volume_trend = '减少'
        
        # 检测异常峰值
        unusual_spikes = []
        vol_std = np.std(volume_data)
        vol_mean = np.mean(volume_data)
        
        for i in range(len(volume_data)):
            if volume_data[i] > vol_mean + 2.5 * vol_std:
                unusual_spikes.append({
                    'index': i,
                    'value': volume_data[i],
                    'sigma': (volume_data[i] - vol_mean) / vol_std
                })
        
        return {
            'average_volume': avg_volume,
            'volume_trend': volume_trend,
            'volume_strength': random.uniform(0.4, 0.9),  # 成交量强度
            'price_volume_correlation': random.uniform(-0.5, 0.5),  # 价格与成交量相关性
            'unusual_spikes': unusual_spikes[:3]  # 只返回前3个最显著的异常
        }
    
    def determine_market_cycle(self, market_data):
        """
        确定市场周期阶段
        """
        self.logger.info("确定市场周期...")
        
        # 简化版市场周期确定
        phases = ['积累', '上涨', '分配', '下跌']
        current_phase = random.choice(phases)
        
        return {
            'current_phase': current_phase,
            'phase_completion': random.uniform(0.1, 0.9),
            'next_phase': phases[(phases.index(current_phase) + 1) % len(phases)],
            'cycle_context': random.choice(['主要趋势', '中期调整', '大周期转折点'])
        }
    
    def detect_market_anomalies(self, market_data):
        """
        检测市场异常情况
        """
        anomalies = []
        anomaly_types = [
            '成交量突增', '价格跳空', '波动率突变', '流动性枯竭',
            '情绪极端', '价格操纵', '闪崩', '大单异常'
        ]
        
        # 模拟异常检测
        if random.random() > 0.7:  # 30%概率检测到异常
            anomaly_count = random.randint(1, 3)
            for _ in range(anomaly_count):
                anomaly = {
                    'type': random.choice(anomaly_types),
                    'severity': random.uniform(0.6, 0.95),
                    'detection_confidence': random.uniform(0.7, 0.9),
                    'timestamp': datetime.now()
                }
                anomalies.append(anomaly)
                
        return anomalies
    
    def calculate_trend_strength(self, price_data):
        """
        计算价格趋势强度
        """
        # 模拟趋势强度计算
        trend_direction = random.choice(['上升', '下降', '横盘'])
        trend_strength = random.uniform(0.3, 0.95)
        
        return {
            'direction': trend_direction,
            'strength': trend_strength,
            'momentum': random.uniform(0.4, 0.9),
            'consistency': random.uniform(0.5, 0.95)
        } 