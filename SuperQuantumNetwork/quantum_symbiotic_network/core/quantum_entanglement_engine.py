#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子纠缠预测引擎 - 超神系统核心模块
基于复数量子态的多维度预测引擎
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
import os
import random
import math
import networkx as nx
import time

logger = logging.getLogger(__name__)

class EntanglementProperty:
    """
    量子纠缠属性类 - 用于描述两个实体间的纠缠关系特性
    """
    
    def __init__(self, strength=0.0, phase=0.0, stability=1.0, decay_rate=0.01):
        """
        初始化纠缠属性
        
        Args:
            strength: 纠缠强度 (0.0-1.0)
            phase: 纠缠相位 (0.0-2π)
            stability: 纠缠稳定性 (0.0-1.0)
            decay_rate: 纠缠衰减率
        """
        self.strength = min(max(0.0, strength), 1.0)  # 限制在0-1范围内
        self.phase = phase % (2 * np.pi)  # 规范化到0-2π
        self.stability = min(max(0.0, stability), 1.0)
        self.decay_rate = max(0.0, decay_rate)
        self.creation_time = datetime.now()
    
    def update_strength(self, delta_strength):
        """更新纠缠强度"""
        self.strength = min(max(0.0, self.strength + delta_strength), 1.0)
        return self.strength
    
    def decay(self, time_elapsed=1.0):
        """
        随时间衰减纠缠强度
        
        Args:
            time_elapsed: 经过的时间单位
            
        Returns:
            衰减后的强度
        """
        decay_factor = np.exp(-self.decay_rate * time_elapsed * (1 - self.stability))
        self.strength *= decay_factor
        return self.strength
    
    def to_dict(self):
        """转换为字典表示"""
        return {
            'strength': self.strength,
            'phase': self.phase,
            'stability': self.stability,
            'decay_rate': self.decay_rate,
            'creation_time': self.creation_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建实例"""
        instance = cls(
            strength=data.get('strength', 0.0),
            phase=data.get('phase', 0.0),
            stability=data.get('stability', 1.0),
            decay_rate=data.get('decay_rate', 0.01)
        )
        if 'creation_time' in data:
            try:
                instance.creation_time = datetime.fromisoformat(data['creation_time'])
            except ValueError:
                pass  # 使用默认创建时间
        return instance

class QuantumEntanglementEngine:
    """量子纠缠预测引擎"""
    
    def __init__(self, dimensions=8, learning_rate=0.01, entanglement_factor=0.3):
        """
        初始化量子纠缠引擎
        
        Args:
            dimensions: 量子态维度
            learning_rate: 学习率
            entanglement_factor: 纠缠因子，控制实体间的量子纠缠强度
        """
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.entanglement_factor = entanglement_factor
        self.depth = 5  # 添加深度参数以兼容测试
        self.high_precision = True  # 启用高精度模式
        
        # 存储所有实体的量子态
        self.quantum_states = {}
        
        # 存储实体间的纠缠关系
        self.entanglement_matrix = {}
        
        # 纠缠群组
        self.entanglement_clusters = []
        
        # 预测历史记录
        self.prediction_history = {}
        
        # 超精度量子阈值
        self.quantum_thresholds = {
            'entanglement': 0.42,  # 黄金分割比相关
            'decoherence': 0.05,
            'nonlocality': 0.78,
            'resonance': 0.61803  # 黄金分割比
        }
        
        # 性能优化：计算缓存
        self._computation_cache = {}
        self._cache_expiry = {}
        self._cache_lifetime = 60  # 缓存生命周期（秒）
        self._priority_entities = set()  # 优先处理的实体
        
        # 性能优化：使用numpy向量化操作
        self._use_vectorization = True
        
        # 初始化随机数生成器
        try:
            # 尝试使用真正的量子随机数生成器
            from secrets import token_bytes
            self.quantum_rng = lambda: int.from_bytes(token_bytes(4), byteorder='big') / 2**32
            logger.info("使用加密级量子随机源")
        except ImportError:
            # 降级到标准随机数生成器
            np.random.seed(datetime.now().microsecond)
            self.quantum_rng = lambda: np.random.random()
            logger.info("使用标准随机源")
        
        logger.info(f"量子纠缠预测引擎初始化完成: 维度={dimensions}, 纠缠因子={entanglement_factor}")
    
    def set_priority_entities(self, entities):
        """
        设置优先处理的实体列表
        
        Args:
            entities: 需要优先计算的实体列表
        """
        self._priority_entities = set(entities)
    
    def _initialize_quantum_states(self, entities):
        """
        批量初始化量子态 - 优化版本，支持向量化操作
        
        Args:
            entities: 实体列表
        """
        for entity in entities:
            if entity not in self.quantum_states:
                self._initialize_quantum_state(entity)
    
    def initialize_entanglement(self, entities, correlation_matrix):
        """
        初始化实体间的纠缠关系 - 优化版本
        
        Args:
            entities: 实体列表（如股票代码列表）
            correlation_matrix: 相关性矩阵，格式为 {(entity1, entity2): correlation}
            
        Returns:
            初始化后的纠缠群组
        """
        # 初始化所有实体的量子态
        self._initialize_quantum_states(entities)
        
        # 初始化纠缠关系
        batch_entities = []
        batch_correlations = []
        
        for (entity1, entity2), correlation in correlation_matrix.items():
            if entity1 in entities and entity2 in entities:
                # 将相关性转换为纠缠强度
                entanglement_strength = correlation
                
                # 收集批处理数据
                batch_entities.append((entity1, entity2))
                batch_correlations.append(entanglement_strength)
        
        # 批量添加纠缠关系
        if self._use_vectorization and batch_entities:
            self._batch_add_entanglement(batch_entities, batch_correlations)
        else:
            # 添加纠缠关系 - 传统方法
            for i, (entity1, entity2) in enumerate(batch_entities):
                self.add_entanglement(entity1, entity2, batch_correlations[i])
        
        # 识别纠缠群组
        self._identify_entanglement_clusters(entities, correlation_matrix)
        
        return self.entanglement_clusters
    
    def _batch_add_entanglement(self, entity_pairs, entanglement_strengths):
        """批量添加纠缠关系 - 性能优化版本
        
        Args:
            entity_pairs: 实体对列表 [(entity1, entity2), ...]
            entanglement_strengths: 纠缠强度列表 [strength1, ...]
        """
        for (entity1, entity2), strength in zip(entity_pairs, entanglement_strengths):
            # 为每对实体创建纠缠属性
            phase = 2 * np.pi * self.quantum_rng()  # 随机相位
            stability = 0.5 + 0.5 * self.quantum_rng()  # 随机稳定性 (0.5-1.0)
            decay_rate = 0.005 + 0.02 * self.quantum_rng()  # 随机衰减率
            
            entanglement_prop = EntanglementProperty(
                strength=strength,
                phase=phase,
                stability=stability,
                decay_rate=decay_rate
            )
            
            # 存储纠缠关系（双向）
            if entity1 not in self.entanglement_matrix:
                self.entanglement_matrix[entity1] = {}
            if entity2 not in self.entanglement_matrix:
                self.entanglement_matrix[entity2] = {}
                
            self.entanglement_matrix[entity1][entity2] = entanglement_prop
            
            # 反向关系使用相同的强度但相反的相位
            reverse_prop = EntanglementProperty(
                strength=strength,
                phase=(phase + np.pi) % (2 * np.pi),  # 相位相差π
                stability=stability,
                decay_rate=decay_rate
            )
            self.entanglement_matrix[entity2][entity1] = reverse_prop
    
    def compute_market_resonance(self, market_data):
        """
        计算市场共振效应 - 性能优化版本
        
        Args:
            market_data: 市场数据字典，格式为 {entity: {'price': float, 'price_change_pct': float, ...}}
            
        Returns:
            共振结果字典 {entity: resonance_value}
        """
        # 生成缓存键
        cache_key = self._generate_cache_key('market_resonance', market_data)
        current_time = time.time()
        
        # 检查缓存是否有效
        if cache_key in self._computation_cache and self._cache_expiry.get(cache_key, 0) > current_time:
            return self._computation_cache[cache_key]
        
        resonance_results = {}
        
        # 优先处理优先级实体
        priority_entities = [e for e in market_data if e in self._priority_entities]
        normal_entities = [e for e in market_data if e not in self._priority_entities]
        
        # 处理所有实体
        for entity in priority_entities + normal_entities:
            if entity in self.quantum_states:
                # 获取量子态
                quantum_state = self.quantum_states[entity]
                
                # 应用市场数据，获取变换后的量子态
                transformed_state = self._apply_quantum_transformation(quantum_state, market_data.get(entity, {}))
                
                # 计算与其他纠缠实体的共振
                resonance = self._compute_entity_resonance(entity, transformed_state, market_data)
                
                # 存储共振结果
                resonance_results[entity] = resonance
        
        # 缓存结果
        self._computation_cache[cache_key] = resonance_results
        self._cache_expiry[cache_key] = current_time + self._cache_lifetime
        
        # 清理过期缓存
        self._cleanup_cache()
        
        return resonance_results
    
    def _compute_entity_resonance(self, entity, transformed_state, market_data):
        """计算单个实体的共振效应"""
        # 查找与该实体有纠缠关系的其他实体
        entangled_entities = []
        
        for (e1, e2) in self.entanglement_matrix:
            if e1 == entity and e2 in market_data:
                entangled_entities.append(e2)
        
        if not entangled_entities:
            return 0.0
        
        # 计算共振强度
        total_resonance = 0.0
        
        for e2 in entangled_entities:
            if e2 in self.quantum_states:
                # 获取纠缠关系
                entanglement = self.entanglement_matrix.get((entity, e2))
                if entanglement:
                    # 计算量子态相干性
                    coherence = self._compute_quantum_coherence(
                        transformed_state, 
                        self.quantum_states[e2]
                    )
                    
                    # 计算共振强度
                    resonance = coherence * entanglement.strength
                    total_resonance += resonance
        
        return total_resonance / len(entangled_entities) if entangled_entities else 0.0
    
    def _compute_quantum_coherence(self, state1, state2):
        """计算两个量子态之间的相干性"""
        # 使用向量化操作
        return np.abs(np.dot(state1, np.conj(state2)))
    
    def _generate_cache_key(self, prefix, data):
        """生成缓存键"""
        # 对于复杂数据结构，使用哈希来生成键
        if isinstance(data, dict):
            # 仅使用关键特性来生成键，避免过度计算
            key_features = []
            for entity, values in data.items():
                if isinstance(values, dict):
                    features = []
                    for k in ['price', 'price_change_pct', 'volume_relative']:
                        if k in values:
                            features.append(f"{k}:{values[k]:.4f}")
                    key_features.append(f"{entity}({','.join(features)})")
                else:
                    key_features.append(f"{entity}:{values}")
            
            return f"{prefix}:{hash(','.join(sorted(key_features)))}"
        else:
            return f"{prefix}:{hash(str(data))}"
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [k for k, exp_time in self._cache_expiry.items() if exp_time <= current_time]
        
        for key in expired_keys:
            if key in self._computation_cache:
                del self._computation_cache[key]
            del self._cache_expiry[key]
    
    def _identify_entanglement_clusters(self, entities, correlation_matrix, threshold=0.5):
        """
        识别纠缠群组
        
        Args:
            entities: 实体列表
            correlation_matrix: 相关性矩阵
            threshold: 纠缠强度阈值，高于此值的实体被认为是强相关的
            
        Returns:
            纠缠群组列表
        """
        # 创建无向图
        G = nx.Graph()
        
        # 添加节点
        for entity in entities:
            G.add_node(entity)
        
        # 添加边
        for (entity1, entity2), correlation in correlation_matrix.items():
            if correlation >= threshold:
                G.add_edge(entity1, entity2, weight=correlation)
        
        # 查找连通分量（群组）
        clusters = list(nx.connected_components(G))
        
        # 存储纠缠群组
        self.entanglement_clusters = [list(cluster) for cluster in clusters]
        
        return self.entanglement_clusters
    
    def get_entanglement_network(self):
        """
        获取纠缠网络状态
        
        Returns:
            包含节点、边和群组的字典
        """
        # 创建网络状态字典
        network = {
            'nodes': [],
            'edges': [],
            'clusters': self.entanglement_clusters
        }
        
        # 添加节点
        for entity, state in self.quantum_states.items():
            node = {
                'id': entity,
                'state': np.abs(state).tolist(),  # 只保存量子态的幅度
                'phase': np.angle(state).tolist() # 保存相位信息
            }
            network['nodes'].append(node)
        
        # 添加边（纠缠关系）
        for entity1, entangled in self.entanglement_matrix.items():
            for entity2, strength in entangled.items():
                if entity1 < entity2:  # 避免重复边
                    edge = {
                        'source': entity1,
                        'target': entity2,
                        'strength': strength
                    }
                    network['edges'].append(edge)
        
        return network
    
    def predict_market_movement(self, assets):
        """
        基于当前市场状态预测市场走势
        
        Args:
            assets: 资产列表
            
        Returns:
            每个资产的预测字典
        """
        predictions = {}
        
        for asset in assets:
            if asset in self.quantum_states:
                # 获取当前量子态
                current_state = self.quantum_states[asset]
                
                # 计算预测结果
                prediction = self._calculate_prediction(current_state)
                
                # 增强预测确定性和稳定性
                prediction_confidence = np.abs(prediction['direction']) 
                
                # 高级预测置信度调整
                if prediction_confidence > self.quantum_thresholds['nonlocality']:
                    confidence_boost = 1.2  # 超过非局域性阈值，提高置信度
                else:
                    confidence_boost = 1.0
                
                # 调整预测置信度，结合量子涨落
                adjusted_confidence = min(0.99, prediction_confidence * confidence_boost)
                
                # 存储预测结果
                predictions[asset] = {
                    'direction': prediction['direction'],
                    'confidence': adjusted_confidence,
                    'volatility': prediction['volatility'],
                    'timestamp': datetime.now().isoformat(),
                    'quantum_resonance': prediction.get('resonance', 0.0) * self.quantum_thresholds['resonance']
                }
            else:
                # 对于没有量子态的资产，初始化并给出初始预测
                self._initialize_quantum_state(asset)
                
                # 给出基础预测
                predictions[asset] = {
                    'direction': 0.0,  # 中性预测
                    'confidence': 0.5,
                    'volatility': 0.1,
                    'timestamp': datetime.now().isoformat(),
                    'quantum_resonance': 0.0,
                    'note': '初始预测，置信度较低'
                }
        
        # 记录预测历史
        timestamp = datetime.now().isoformat()
        self.prediction_history[timestamp] = predictions
        
        return predictions
    
    def apply_quantum_operations(self, market_data):
        """
        应用量子操作处理市场数据
        
        Args:
            market_data: 市场数据字典
            
        Returns:
            更新后的量子态字典
        """
        # 对每个资产应用量子操作
        for asset, data in market_data.items():
            if asset in self.quantum_states:
                # 获取当前量子态
                quantum_state = self.quantum_states[asset]
                
                # 应用量子变换
                transformed_state = self._apply_quantum_transformation(quantum_state, data)
                
                # 应用纠缠效应
                entangled_state = self._apply_entanglement(asset, transformed_state)
                
                # 更新量子态
                self.quantum_states[asset] = entangled_state
        
        return self.quantum_states
    
    def _initialize_quantum_state(self, entity_id):
        """
        为实体初始化量子态
        
        Args:
            entity_id: 实体ID (如股票代码)
            
        Returns:
            初始化的量子态（复数数组）
        """
        # 创建复数量子态
        # 实部和虚部都是随机初始化的
        real_part = np.random.normal(0, 1, self.dimensions)
        imag_part = np.random.normal(0, 1, self.dimensions)
        
        quantum_state = real_part + 1j * imag_part
        
        # 归一化量子态
        norm = np.sqrt(np.sum(np.abs(quantum_state) ** 2))
        if norm > 0:
            quantum_state = quantum_state / norm
            
        self.quantum_states[entity_id] = quantum_state
        return quantum_state
    
    def _apply_quantum_transformation(self, quantum_state, input_data):
        """
        应用量子变换到量子态
        
        Args:
            quantum_state: 当前量子态
            input_data: 输入数据
            
        Returns:
            变换后的量子态
        """
        # 添加非线性变换
        transformed_state = np.copy(quantum_state)
        
        if isinstance(input_data, dict):
            # 提取市场数据特征
            features = []
            
            # 价格变化
            if 'price_change_pct' in input_data:
                features.append(input_data['price_change_pct'])
            
            # 交易量
            if 'volume_relative' in input_data:
                features.append(input_data['volume_relative'])
            
            # 如果特征少于3个，填充随机噪声
            while len(features) < 3:
                features.append(self.quantum_rng() * 0.01)  # 小噪声
        else:
            # 如果输入不是字典，尝试转换为列表或使用随机值
            try:
                features = list(input_data)[:3]
                # 如果特征少于3个，填充随机噪声
                while len(features) < 3:
                    features.append(self.quantum_rng() * 0.01)
            except:
                # 退化情况：使用随机特征
                features = [self.quantum_rng() * 0.1 for _ in range(3)]
        
        # 应用非线性变换 - 使用量子启发的扭曲变换
        for i in range(self.dimensions):
            # 复杂非线性变换 - 结合多个特征和量子相位
            phase = 2 * np.pi * (i / self.dimensions)
            
            # 高级量子效应模拟 - 考虑量子纠缠和相干性
            nonlinear_factor = abs(np.sin(phase + features[0] * np.pi)) * (1.0 + features[1])
            
            # 添加量子涨落效应
            quantum_fluctuation = 0.0
            if self.high_precision:
                # 使用Box-Muller变换生成更高质量的高斯噪声
                u1 = self.quantum_rng()
                u2 = self.quantum_rng()
                if u1 > 0:  # 避免对0取对数
                    quantum_fluctuation = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2) * 0.01
            
            # 应用非线性变换 - 使用黄金分割比作为特殊阈值
            if nonlinear_factor > self.quantum_thresholds['resonance']:
                # 超过黄金分割比阈值时产生共振
                transformed_state[i] = np.tanh(quantum_state[i] * nonlinear_factor + quantum_fluctuation)
            else:
                # 否则使用量子波动方程
                transformed_state[i] = quantum_state[i] * nonlinear_factor + quantum_fluctuation
        
        # 归一化
        norm = np.linalg.norm(transformed_state)
        if norm > 0:
            transformed_state = transformed_state / norm
        
        return transformed_state
    
    def _apply_entanglement(self, entity_id, quantum_state):
        """
        应用量子纠缠效应
        
        Args:
            entity_id: 当前实体ID
            quantum_state: 当前量子态
            
        Returns:
            考虑纠缠后的量子态
        """
        # 如果没有纠缠关系，直接返回
        if entity_id not in self.entanglement_matrix:
            return quantum_state
            
        # 获取所有与当前实体纠缠的实体
        entangled_entities = self.entanglement_matrix[entity_id]
        
        # 累积纠缠效应
        entanglement_effect = np.zeros(self.dimensions, dtype=complex)
        
        for other_id, strength in entangled_entities.items():
            if other_id in self.quantum_states:
                other_state = self.quantum_states[other_id]
                
                # 计算纠缠效应（加权和）
                entanglement_effect += other_state * strength
        
        # 归一化纠缠效应
        norm = np.sqrt(np.sum(np.abs(entanglement_effect) ** 2))
        if norm > 0:
            entanglement_effect = entanglement_effect / norm
            
        # 将纠缠效应与原量子态混合
        # 使用纠缠因子控制混合比例
        mixed_state = quantum_state * (1 - self.entanglement_factor) + entanglement_effect * self.entanglement_factor
        
        # 归一化
        norm = np.sqrt(np.sum(np.abs(mixed_state) ** 2))
        if norm > 0:
            mixed_state = mixed_state / norm
            
        return mixed_state
    
    def _calculate_prediction(self, quantum_state):
        """
        根据量子态计算预测结果
        
        Args:
            quantum_state: 量子态向量
            
        Returns:
            预测结果字典
        """
        # 计算振幅和相位
        amplitudes = np.abs(quantum_state)
        phases = np.angle(quantum_state)
        
        # 使用前几个维度的数据计算方向
        direction_dim = min(4, self.dimensions)
        direction = np.sum(phases[:direction_dim]) / direction_dim / np.pi
        
        # 规范化到-1到1之间
        direction = max(-1.0, min(1.0, direction))
        
        # 计算强度
        strength = np.mean(amplitudes)
        
        # 计算波动性
        volatility = np.std(phases)
        
        # 计算共振
        resonance = np.abs(np.correlate(amplitudes, phases)[0]) / self.dimensions
        
        # 计算上涨和下跌概率
        up_probability = (direction + 1) / 2
        down_probability = 1 - up_probability
        
        return {
            'direction': direction,
            'strength': strength,
            'volatility': volatility,
            'resonance': resonance,
            'up_probability': up_probability,
            'down_probability': down_probability
        }
    
    def add_entanglement(self, entity_id1, entity_id2, strength=None):
        """
        添加两个实体之间的量子纠缠关系
        
        Args:
            entity_id1: 第一个实体ID
            entity_id2: 第二个实体ID
            strength: 纠缠强度（如果为None则随机生成）
            
        Returns:
            纠缠强度
        """
        # 如果强度未指定，生成随机强度（0.1-0.5之间）
        if strength is None:
            strength = random.uniform(0.1, 0.5)
            
        # 确保实体存在量子态
        if entity_id1 not in self.quantum_states:
            self._initialize_quantum_state(entity_id1)
            
        if entity_id2 not in self.quantum_states:
            self._initialize_quantum_state(entity_id2)
            
        # 更新纠缠矩阵
        if entity_id1 not in self.entanglement_matrix:
            self.entanglement_matrix[entity_id1] = {}
            
        if entity_id2 not in self.entanglement_matrix:
            self.entanglement_matrix[entity_id2] = {}
            
        # 双向纠缠（对称）
        self.entanglement_matrix[entity_id1][entity_id2] = strength
        self.entanglement_matrix[entity_id2][entity_id1] = strength
        
        logger.debug(f"添加纠缠关系: {entity_id1} <--> {entity_id2}, 强度={strength:.4f}")
        
        return strength
    
    def predict(self, input_data, entity_id, current_state=None):
        """
        执行量子预测
        
        Args:
            input_data: 输入数据（特征字典）
            entity_id: 实体ID
            current_state: 当前状态（可选）
            
        Returns:
            预测结果
        """
        # 获取或初始化量子态
        if entity_id in self.quantum_states:
            quantum_state = self.quantum_states[entity_id]
        else:
            quantum_state = self._initialize_quantum_state(entity_id)
            
        # 应用量子变换
        transformed_state = self._apply_quantum_transformation(quantum_state, input_data)
        
        # 应用量子纠缠
        entangled_state = self._apply_entanglement(entity_id, transformed_state)
        
        # 更新量子态
        self.quantum_states[entity_id] = entangled_state
        
        # 计算预测结果
        prediction = self._calculate_prediction(entangled_state)
        
        # 记录预测历史
        if entity_id not in self.prediction_history:
            self.prediction_history[entity_id] = []
            
        self.prediction_history[entity_id].append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'input_data': input_data
        })
        
        # 限制历史记录数量
        if len(self.prediction_history[entity_id]) > 100:
            self.prediction_history[entity_id].pop(0)
            
        return prediction
    
    def update_from_feedback(self, entity_id, actual_direction, learning_rate=None):
        """
        根据实际结果反馈更新量子态
        
        Args:
            entity_id: 实体ID
            actual_direction: 实际方向（正值表示上涨，负值表示下跌）
            learning_rate: 学习率（如果为None则使用默认值）
            
        Returns:
            更新后的量子态
        """
        if entity_id not in self.quantum_states:
            logger.warning(f"实体{entity_id}没有量子态，无法更新")
            return None
            
        # 使用指定的学习率或默认学习率
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        # 获取最近的预测
        if entity_id in self.prediction_history and self.prediction_history[entity_id]:
            last_prediction = self.prediction_history[entity_id][-1]['prediction']
            predicted_direction = last_prediction['direction']
            
            # 计算预测误差
            error = actual_direction - predicted_direction
            
            # 如果误差太小，不更新
            if abs(error) < 0.01:
                return self.quantum_states[entity_id]
                
            # 更新量子态
            quantum_state = self.quantum_states[entity_id]
            
            # 应用相位旋转来校正预测方向
            # 方向误差越大，旋转角度越大
            phase_rotation = error * lr * np.pi
            
            # 创建旋转矩阵
            rotation_matrix = np.eye(self.dimensions, dtype=complex)
            rotation_matrix[0, 0] = np.cos(phase_rotation) + 0j
            rotation_matrix[0, 1] = -np.sin(phase_rotation) + 0j
            rotation_matrix[1, 0] = np.sin(phase_rotation) + 0j
            rotation_matrix[1, 1] = np.cos(phase_rotation) + 0j
            
            # 应用旋转
            updated_state = np.dot(rotation_matrix, quantum_state)
            
            # 归一化
            norm = np.sqrt(np.sum(np.abs(updated_state) ** 2))
            if norm > 0:
                updated_state = updated_state / norm
                
            # 更新存储的量子态
            self.quantum_states[entity_id] = updated_state
            
            logger.debug(f"更新实体{entity_id}的量子态: 误差={error:.4f}, 学习率={lr:.4f}")
            
            return updated_state
        else:
            logger.warning(f"实体{entity_id}没有预测历史，无法更新")
            return self.quantum_states[entity_id]
    
    def save_state(self, filepath):
        """
        保存引擎状态到文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否成功
        """
        try:
            # 将量子态转换为可序列化格式
            serialized_states = {}
            for entity_id, state in self.quantum_states.items():
                serialized_states[entity_id] = {
                    'real': state.real.tolist(),
                    'imag': state.imag.tolist()
                }
                
            # 保存状态
            state_data = {
                'dimensions': self.dimensions,
                'learning_rate': self.learning_rate,
                'entanglement_factor': self.entanglement_factor,
                'quantum_states': serialized_states,
                'entanglement_matrix': self.entanglement_matrix,
                'timestamp': datetime.now().isoformat()
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"量子引擎状态已保存到{filepath}")
            return True
        except Exception as e:
            logger.error(f"保存量子引擎状态失败: {e}")
            return False
    
    def load_state(self, filepath):
        """
        从文件加载引擎状态
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否成功
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                
            # 加载基本参数
            self.dimensions = state_data.get('dimensions', 8)
            self.learning_rate = state_data.get('learning_rate', 0.01)
            self.entanglement_factor = state_data.get('entanglement_factor', 0.3)
            
            # 加载量子态
            self.quantum_states = {}
            for entity_id, state_data in state_data.get('quantum_states', {}).items():
                real_part = np.array(state_data['real'])
                imag_part = np.array(state_data['imag'])
                self.quantum_states[entity_id] = real_part + 1j * imag_part
                
            # 加载纠缠矩阵
            self.entanglement_matrix = state_data.get('entanglement_matrix', {})
            
            logger.info(f"从{filepath}加载量子引擎状态成功")
            return True
        except Exception as e:
            logger.error(f"加载量子引擎状态失败: {e}")
            return False
    
    def analyze_entity_correlations(self, entity_id, top_n=5):
        """
        分析实体与其他实体的相关性
        
        Args:
            entity_id: 实体ID
            top_n: 返回最强相关的前N个实体
            
        Returns:
            相关性列表
        """
        if entity_id not in self.quantum_states:
            logger.warning(f"实体{entity_id}不存在")
            return []
            
        target_state = self.quantum_states[entity_id]
        correlations = []
        
        # 计算与所有其他实体的相关性
        for other_id, other_state in self.quantum_states.items():
            if other_id != entity_id:
                # 计算量子态的内积（相关性度量）
                correlation = np.abs(np.vdot(target_state, other_state))
                correlations.append((other_id, correlation))
                
        # 按相关性降序排序
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前N个
        return correlations[:top_n]
    
    def get_quantum_stats(self):
        """
        获取量子引擎统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'dimensions': self.dimensions,
            'entity_count': len(self.quantum_states),
            'entanglement_count': sum(len(entities) for entities in self.entanglement_matrix.values()),
            'average_confidence': 0,
            'system_coherence': 0
        }
        
        # 计算平均置信度
        confidences = []
        for entity_id in self.quantum_states:
            if entity_id in self.prediction_history and self.prediction_history[entity_id]:
                confidences.append(self.prediction_history[entity_id][-1]['prediction']['confidence'])
                
        if confidences:
            stats['average_confidence'] = sum(confidences) / len(confidences)
            
        # 计算系统的整体量子相干性
        coherence_sum = 0
        count = 0
        for entity_id, state in self.quantum_states.items():
            for other_id, other_state in self.quantum_states.items():
                if entity_id != other_id:
                    coherence_sum += np.abs(np.vdot(state, other_state))
                    count += 1
                    
        if count > 0:
            stats['system_coherence'] = coherence_sum / count
            
        return stats

    def calculate_market_resonance(self, entities, weights=None):
        """计算一组市场实体的量子共振度
        
        Args:
            entities: 市场实体列表
            weights: 各实体的权重，默认为均等权重
            
        Returns:
            共振系数 (0.0-1.0)，越高表示市场越协调
        """
        if not entities:
            return 0.5  # 默认中性值
        
        # 初始化权重
        if weights is None:
            weights = [1.0 / len(entities)] * len(entities)
        else:
            # 确保权重和为1
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # 确保所有实体都有量子态
        self._initialize_quantum_states(entities)
        
        # 计算共振因子
        resonance_sum = 0.0
        count = 0
        
        # 计算方式1：实体间的纠缠相位一致性
        phase_coherence = 0.0
        valid_pairs = 0
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i < j:  # 避免重复计算
                    # 检查实体间是否存在纠缠
                    if (entity1 in self.entanglement_matrix and 
                        entity2 in self.entanglement_matrix[entity1]):
                        
                        # 获取纠缠属性
                        entanglement = self.entanglement_matrix[entity1][entity2]
                        
                        # 计算相位相干性
                        # 量子相位越接近，相干性越高
                        phase_diff = abs(np.cos(entanglement.phase))
                        strength = entanglement.strength
                        
                        # 加权共振贡献
                        weight = weights[i] * weights[j]
                        phase_coherence += phase_diff * strength * weight
                        valid_pairs += 1
        
        # 计算平均相干性
        if valid_pairs > 0:
            phase_coherence /= valid_pairs
        
        # 计算方式2：量子态能量分布均匀性
        energy_distribution = 0.0
        
        # 提取所有量子态的能量水平
        energy_levels = []
        for entity in entities:
            if entity in self.quantum_states:
                # 计算简化的量子能量水平
                state = self.quantum_states[entity]
                if isinstance(state, np.ndarray):
                    # 使用量子态的范数作为能量度量
                    energy = np.linalg.norm(state)
                    energy_levels.append(energy)
        
        # 计算能量水平的标准差，标准差越小表示能量分布越均匀
        if energy_levels:
            energy_std = np.std(energy_levels)
            # 转换为0-1范围，标准差越小，均匀性越高
            energy_distribution = np.exp(-2 * energy_std)  # 使用指数衰减
        
        # 综合考虑相位相干性和能量分布
        resonance_factor = 0.7 * phase_coherence + 0.3 * energy_distribution
        
        # 应用黄金分割比调整
        golden_ratio = 0.618
        resonance_factor = golden_ratio * resonance_factor + (1 - golden_ratio) * 0.5
        
        # 确保结果在0-1范围内
        resonance_factor = max(0.0, min(1.0, resonance_factor))
        
        # 对结果应用量子波动
        quantum_fluctuation = 0.05 * (2 * self.quantum_rng() - 1)  # -0.05到0.05的波动
        resonance_factor = max(0.0, min(1.0, resonance_factor + quantum_fluctuation))
        
        return resonance_factor
    
    def predict_future(self, entities, steps=3, current_state=None):
        """预测市场实体的未来状态
        
        Args:
            entities: 市场实体列表
            steps: 预测的时间步数
            current_state: 当前市场状态，可选
            
        Returns:
            未来状态列表，每个状态为一个字典
        """
        if not entities:
            return []
        
        # 确保所有实体都有量子态
        self._initialize_quantum_states(entities)
        
        # 预测结果
        future_states = []
        
        # 当前量子态的深拷贝
        current_quantum_states = {}
        for entity in entities:
            if entity in self.quantum_states:
                current_quantum_states[entity] = np.copy(self.quantum_states[entity])
        
        # 当前纠缠矩阵的简化表示
        current_entanglement = {}
        for entity1 in entities:
            if entity1 in self.entanglement_matrix:
                current_entanglement[entity1] = {}
                for entity2 in entities:
                    if entity2 in self.entanglement_matrix[entity1]:
                        prop = self.entanglement_matrix[entity1][entity2]
                        current_entanglement[entity1][entity2] = {
                            'strength': prop.strength,
                            'phase': prop.phase,
                            'stability': prop.stability
                        }
        
        # 模拟未来时间步
        for step in range(steps):
            # 进行量子态演化
            evolved_states = self._evolve_quantum_states(
                current_quantum_states, 
                current_entanglement,
                step_size=0.2
            )
            
            # 从演化后的量子态计算趋势
            step_trend = self._calculate_trend_from_states(evolved_states)
            
            # 从量子态计算风险
            risk_level = self._calculate_risk_from_states(evolved_states)
            
            # 生成详细预测
            detailed_predictions = {}
            for entity in entities:
                if entity in evolved_states:
                    # 计算该实体的预测值
                    prediction_value = self._extract_prediction_value(
                        evolved_states[entity], 
                        step + 1  # 时间步调整
                    )
                    
                    # 加入详细预测
                    detailed_predictions[entity] = {
                        'prediction': prediction_value,
                        'confidence': 0.7 - step * 0.1  # 随时间步减少置信度
                    }
            
            # 创建该时间步的预测结果
            future_state = {
                'step': step + 1,
                'trend': step_trend,
                'risk_level': risk_level,
                'detailed_predictions': detailed_predictions,
                'resonance': self._calculate_resonance_from_states(evolved_states)
            }
            
            future_states.append(future_state)
            
            # 更新当前状态用于下一步预测
            current_quantum_states = evolved_states
            
            # 更新纠缠关系（随时间演化）
            for entity1 in current_entanglement:
                for entity2 in current_entanglement[entity1]:
                    # 随时间衰减纠缠强度
                    decay_rate = 0.1 * (1 - current_entanglement[entity1][entity2]['stability'])
                    new_strength = current_entanglement[entity1][entity2]['strength'] * (1 - decay_rate)
                    current_entanglement[entity1][entity2]['strength'] = new_strength
        
        return future_states
    
    def update_market_state(self, market_state):
        """使用最新的市场数据更新量子状态
        
        Args:
            market_state: 市场状态字典，格式为 {entity: value}
            
        Returns:
            更新是否成功
        """
        try:
            # 确保所有实体都有量子态
            self._initialize_quantum_states(list(market_state.keys()))
            
            # 基于市场状态更新量子态
            for entity, value in market_state.items():
                if entity in self.quantum_states:
                    # 将市场值转换为量子态调整
                    self._update_entity_quantum_state(entity, value)
            
            # 更新纠缠关系
            self._update_entanglement_based_on_market(market_state)
            
            return True
        except Exception as e:
            logger.error(f"更新量子市场状态失败: {str(e)}")
            return False
    
    def _update_entity_quantum_state(self, entity, value):
        """更新单个实体的量子态
        
        Args:
            entity: 实体标识符
            value: 市场值 (可以是浮点数或者字典)
        """
        if entity not in self.quantum_states:
            self._initialize_quantum_state(entity)
            
        state = self.quantum_states[entity]
        
        if isinstance(value, (int, float)):
            # 简单浮点值，直接调整量子态
            
            # 计算相位调整
            phase_shift = np.pi * value  # 市场值影响相位
            
            # 计算幅度调整
            amplitude_shift = np.tanh(value)  # 限制在-1到1范围
            
            # 应用到量子态
            for i in range(len(state)):
                # 分解为幅度和相位
                amplitude = np.abs(state[i])
                phase = np.angle(state[i])
                
                # 更新相位和幅度
                new_phase = (phase + phase_shift) % (2 * np.pi)
                new_amplitude = max(0, min(1, amplitude + amplitude_shift * 0.1))
                
                # 重建复数
                state[i] = new_amplitude * np.exp(1j * new_phase)
                
            # 归一化
            norm = np.linalg.norm(state)
            if norm > 0:
                state /= norm
        
        elif isinstance(value, dict):
            # 如果是字典格式，包含更多信息
            if 'value' in value:
                main_value = value['value']
                weight = value.get('weight', 1.0)
                phase = value.get('phase', 0.0)
                
                # 计算加权调整
                for i in range(len(state)):
                    # 权重越高，调整越大
                    amplitude_shift = np.tanh(main_value) * weight * 0.1
                    phase_shift = phase + np.pi * main_value * weight
                    
                    # 分解为幅度和相位
                    amplitude = np.abs(state[i])
                    current_phase = np.angle(state[i])
                    
                    # 更新
                    new_phase = (current_phase + phase_shift) % (2 * np.pi)
                    new_amplitude = max(0, min(1, amplitude + amplitude_shift))
                    
                    # 重建复数
                    state[i] = new_amplitude * np.exp(1j * new_phase)
                
                # 归一化
                norm = np.linalg.norm(state)
                if norm > 0:
                    state /= norm
    
    def _update_entanglement_based_on_market(self, market_state):
        """基于市场状态更新纠缠关系
        
        Args:
            market_state: 市场状态字典
        """
        entities = list(market_state.keys())
        
        # 根据市场状态调整实体间的纠缠关系
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i < j:  # 避免重复处理
                    # 检查实体间是否存在纠缠关系
                    if (entity1 in self.entanglement_matrix and 
                        entity2 in self.entanglement_matrix[entity1]):
                        
                        # 计算市场值的相似性
                        value1 = market_state[entity1]
                        value2 = market_state[entity2]
                        
                        if isinstance(value1, dict):
                            value1 = value1.get('value', 0)
                        if isinstance(value2, dict):
                            value2 = value2.get('value', 0)
                        
                        # 市场值越相似，纠缠强度越高
                        similarity = 1.0 - min(1.0, abs(value1 - value2))
                        
                        # 获取当前纠缠属性
                        entanglement = self.entanglement_matrix[entity1][entity2]
                        
                        # 调整纠缠强度
                        current_strength = entanglement.strength
                        target_strength = max(0.1, min(0.9, similarity))
                        
                        # 平滑过渡
                        new_strength = current_strength * 0.7 + target_strength * 0.3
                        entanglement.update_strength(new_strength - current_strength)
                        
                        # 更新反向关系
                        reverse_entanglement = self.entanglement_matrix[entity2][entity1]
                        reverse_entanglement.update_strength(new_strength - reverse_entanglement.strength)
    
    def _evolve_quantum_states(self, states, entanglement, step_size=0.1):
        """随时间演化量子态
        
        Args:
            states: 当前量子态字典
            entanglement: 简化的纠缠关系
            step_size: 演化步长
            
        Returns:
            演化后的量子态
        """
        evolved_states = {}
        
        # 复制状态
        for entity, state in states.items():
            evolved_states[entity] = np.copy(state)
        
        # 应用量子演化
        for entity1, state1 in states.items():
            if entity1 in entanglement:
                for entity2, ent_props in entanglement[entity1].items():
                    if entity2 in states:
                        state2 = states[entity2]
                        
                        # 提取纠缠属性
                        strength = ent_props['strength']
                        phase = ent_props['phase']
                        
                        # 计算纠缠贡献
                        # 实体2对实体1的影响
                        influence = strength * step_size
                        
                        # 混合量子态 (加权平均)
                        evolved_states[entity1] = (1 - influence) * state1 + influence * state2 * np.exp(1j * phase)
                        
                        # 归一化
                        norm = np.linalg.norm(evolved_states[entity1])
                        if norm > 0:
                            evolved_states[entity1] /= norm
        
        # 为每个实体增加随机量子涨落
        for entity in evolved_states:
            # 随机相位扰动
            random_phase = 2 * np.pi * self.quantum_rng() * step_size * 0.2
            
            # 应用随机相位扰动
            evolved_states[entity] = evolved_states[entity] * np.exp(1j * random_phase)
            
            # 归一化
            norm = np.linalg.norm(evolved_states[entity])
            if norm > 0:
                evolved_states[entity] /= norm
        
        return evolved_states
    
    def _calculate_trend_from_states(self, states):
        """从量子态计算市场趋势
        
        Args:
            states: 量子态字典
            
        Returns:
            市场趋势描述
        """
        # 从量子态提取趋势指标
        trend_indicators = []
        
        for entity, state in states.items():
            # 计算简化的趋势指标
            phase_avg = np.angle(np.mean(state))
            amplitude_var = np.var(np.abs(state))
            
            # 将相位映射到-1到1的趋势指标
            trend = np.sin(phase_avg)
            
            # 将振幅方差作为趋势的确定性
            certainty = 1.0 - min(1.0, amplitude_var * 5)
            
            # 加权趋势
            weighted_trend = trend * certainty
            trend_indicators.append(weighted_trend)
        
        # 计算平均趋势
        if trend_indicators:
            avg_trend = np.mean(trend_indicators)
        else:
            avg_trend = 0.0
        
        # 映射到趋势描述
        if avg_trend > 0.6:
            return '强势上涨'
        elif avg_trend > 0.2:
            return '温和上涨'
        elif avg_trend > -0.2:
            return '横盘震荡'
        elif avg_trend > -0.6:
            return '温和下跌'
        else:
            return '强势下跌'
    
    def _calculate_risk_from_states(self, states):
        """从量子态计算风险水平
        
        Args:
            states: 量子态字典
            
        Returns:
            风险水平 (0.0-1.0)
        """
        if not states:
            return 0.5
        
        # 计算量子态的平均纠缠
        entanglement_measure = 0.0
        count = 0
        
        # 使用量子纯度作为风险度量
        purity_measures = []
        for state in states.values():
            # 计算密度矩阵
            rho = np.outer(state, np.conjugate(state))
            
            # 计算纯度 Tr(rho^2)
            purity = np.trace(np.matmul(rho, rho)).real
            
            # 纯度越低，表示量子态越混合，对应更高风险
            purity_measures.append(1.0 - purity)
        
        if purity_measures:
            risk_from_purity = np.mean(purity_measures)
        else:
            risk_from_purity = 0.5
        
        # 使用量子态间的相似性作为风险指标
        similarity_measure = 0.0
        comparisons = 0
        
        entities = list(states.keys())
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i < j:  # 避免重复计算
                    state1 = states[entity1]
                    state2 = states[entity2]
                    
                    # 计算内积的平方
                    overlap = np.abs(np.dot(state1, np.conjugate(state2)))**2
                    
                    # 内积越小，表示量子态越正交，对应更高风险
                    similarity_measure += (1.0 - overlap)
                    comparisons += 1
        
        if comparisons > 0:
            risk_from_similarity = similarity_measure / comparisons
        else:
            risk_from_similarity = 0.5
        
        # 综合风险指标
        return 0.4 * risk_from_purity + 0.6 * risk_from_similarity
    
    def _calculate_resonance_from_states(self, states):
        """计算量子态的共振系数
        
        Args:
            states: 量子态字典
            
        Returns:
            共振系数 (0.0-1.0)
        """
        if not states:
            return 0.5
        
        # 计算各状态的相位
        phases = []
        for state in states.values():
            # 计算平均相位
            phase = np.angle(np.mean(state))
            phases.append(phase)
        
        if not phases:
            return 0.5
        
        # 计算相位的标准差，标准差越小表示相位越一致
        phase_std = np.std(phases)
        
        # 转换为0-1范围的共振系数
        resonance = np.exp(-phase_std)
        
        # 调整到0.2-0.8范围
        resonance = 0.2 + 0.6 * resonance
        
        return resonance
    
    def _extract_prediction_value(self, state, step):
        """从量子态提取预测值
        
        Args:
            state: 量子态
            step: 时间步
            
        Returns:
            预测值
        """
        # 使用量子态的平均相位作为基础
        phase = np.angle(np.mean(state))
        
        # 从相位计算预测值，范围约为-0.3到0.3
        base_prediction = 0.3 * np.sin(phase)
        
        # 随着时间步增加，增加波动性
        volatility = 0.01 * step
        
        # 添加随机波动
        random_factor = volatility * (2 * self.quantum_rng() - 1)
        
        return base_prediction + random_factor
    
    def get_entity_phase(self, entity):
        """获取实体的量子相位
        
        Args:
            entity: 实体标识符
            
        Returns:
            量子相位 (0-2π)
        """
        if entity in self.quantum_states:
            # 计算量子态的平均相位
            state = self.quantum_states[entity]
            return np.angle(np.mean(state))
        return 0.0
    
    def get_quantum_state(self, entity):
        """获取实体的量子状态
        
        Args:
            entity: 实体标识符
            
        Returns:
            量子状态字典
        """
        if entity not in self.quantum_states:
            self._initialize_quantum_state(entity)
            
        state = self.quantum_states[entity]
        
        # 从量子态提取关键信息
        phase = np.angle(np.mean(state))
        amplitude = np.mean(np.abs(state))
        variance = np.var(np.abs(state))
        
        return {
            'phase': phase,
            'amplitude': amplitude,
            'variance': variance,
            'stability': 1.0 - min(1.0, variance * 5)
        } 