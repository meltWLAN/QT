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
        
        # 存储所有实体的量子态
        self.quantum_states = {}
        
        # 存储实体间的纠缠关系
        self.entanglement_matrix = {}
        
        # 纠缠群组
        self.entanglement_clusters = []
        
        # 预测历史记录
        self.prediction_history = {}
        
        # 初始化随机数生成器
        np.random.seed(datetime.now().microsecond)
        
        logger.info(f"量子纠缠预测引擎初始化完成: 维度={dimensions}, 纠缠因子={entanglement_factor}")
    
    def initialize_entanglement(self, entities, correlation_matrix):
        """
        初始化实体间的纠缠关系
        
        Args:
            entities: 实体列表（如股票代码列表）
            correlation_matrix: 相关性矩阵，格式为 {(entity1, entity2): correlation}
            
        Returns:
            初始化后的纠缠群组
        """
        # 初始化所有实体的量子态
        for entity in entities:
            if entity not in self.quantum_states:
                self._initialize_quantum_state(entity)
        
        # 初始化纠缠关系
        for (entity1, entity2), correlation in correlation_matrix.items():
            if entity1 in entities and entity2 in entities:
                # 将相关性转换为纠缠强度
                entanglement_strength = correlation
                
                # 添加纠缠关系
                self.add_entanglement(entity1, entity2, entanglement_strength)
        
        # 识别纠缠群组
        self._identify_entanglement_clusters(entities, correlation_matrix)
        
        return self.entanglement_clusters
    
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
    
    def compute_market_resonance(self, market_data):
        """
        计算市场共振状态
        
        Args:
            market_data: 市场数据，格式为 {asset: {feature1: value1, ...}}
            
        Returns:
            共振状态字典，格式为 {asset: resonance_value}
        """
        resonance_state = {}
        
        # 对每个资产计算共振
        for asset, data in market_data.items():
            if asset in self.quantum_states:
                # 获取量子态
                state = self.quantum_states[asset]
                
                # 计算振幅平方和（概率）
                amplitudes = np.abs(state) ** 2
                
                # 计算共振值（熵的逆）
                entropy = -np.sum(amplitudes * np.log(amplitudes + 1e-10))
                max_entropy = np.log(self.dimensions)
                
                # 归一化共振值（0-1范围）
                # 熵越低，共振越高
                resonance = 1.0 - (entropy / max_entropy)
                
                # 考虑市场数据的影响
                price_change = data.get('price_change_pct', 0)
                volume = data.get('volume_relative', 1.0)
                
                # 调整共振值
                resonance_adjustment = abs(price_change) * volume * 0.2
                resonance = min(1.0, resonance + resonance_adjustment)
                
                resonance_state[asset] = resonance
        
        return resonance_state
    
    def predict_market_movement(self, assets):
        """
        预测市场走势
        
        Args:
            assets: 资产列表
            
        Returns:
            预测结果字典
        """
        predictions = {}
        
        for asset in assets:
            if asset in self.quantum_states:
                # 获取量子态
                state = self.quantum_states[asset]
                
                # 计算振幅和相位
                amplitudes = np.abs(state)
                phases = np.angle(state)
                
                # 使用前几个维度的数据计算方向
                direction_dim = min(4, self.dimensions)
                direction = np.sum(phases[:direction_dim]) / direction_dim / np.pi
                
                # 规范化到-1到1之间
                direction = max(-1.0, min(1.0, direction))
                
                # 计算强度
                strength = np.mean(amplitudes)
                
                # 计算上涨和下跌概率
                up_probability = (direction + 1) / 2
                down_probability = 1 - up_probability
                
                predictions[asset] = {
                    'direction': direction,
                    'strength': strength,
                    'up_probability': up_probability,
                    'down_probability': down_probability
                }
        
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
            input_data: 输入数据字典
            
        Returns:
            变换后的量子态
        """
        # 创建量子门（变换矩阵）
        transformation_matrix = np.eye(self.dimensions, dtype=complex)
        
        # 处理输入数据中的各个特征
        for feature, value in input_data.items():
            # 根据特征和值计算相位变化
            phase_shift = value * np.pi * 0.1  # 缩放到合理范围
            
            # 为不同特征选择不同的量子门行为
            if feature == 'price_change':
                # 价格变化使用旋转门
                rotation_idx = 0
                transformation_matrix[rotation_idx, rotation_idx] = np.cos(phase_shift) + 0j
                rotation_idx2 = 1
                transformation_matrix[rotation_idx, rotation_idx2] = -np.sin(phase_shift) + 0j
                transformation_matrix[rotation_idx2, rotation_idx] = np.sin(phase_shift) + 0j
                transformation_matrix[rotation_idx2, rotation_idx2] = np.cos(phase_shift) + 0j
            
            elif feature == 'volume_change':
                # 成交量变化使用相位门
                phase_idx = 2
                transformation_matrix[phase_idx, phase_idx] = np.exp(1j * phase_shift)
            
            elif feature == 'ma5_diff':
                # 5日均线差异使用相位门
                phase_idx = 3
                transformation_matrix[phase_idx, phase_idx] = np.exp(1j * phase_shift)
            
            elif feature == 'ma10_diff':
                # 10日均线差异使用相位门
                phase_idx = 4
                transformation_matrix[phase_idx, phase_idx] = np.exp(1j * phase_shift)
            
            elif feature == 'ma20_diff':
                # 20日均线差异使用相位门
                phase_idx = 5
                transformation_matrix[phase_idx, phase_idx] = np.exp(1j * phase_shift)
            
            elif feature == 'volatility':
                # 波动率使用哈达玛门混合
                had_idx = 6
                transformation_matrix[had_idx, had_idx] = np.cos(phase_shift) + 0j
                had_idx2 = 7
                if had_idx2 < self.dimensions:
                    transformation_matrix[had_idx, had_idx2] = np.sin(phase_shift) + 0j
                    transformation_matrix[had_idx2, had_idx] = np.sin(phase_shift) + 0j
                    transformation_matrix[had_idx2, had_idx2] = -np.cos(phase_shift) + 0j
            
            # 自适应门：对于未指定的特征，动态分配量子门
            else:
                # 通过特征名的哈希值确定索引，确保相同特征总是使用相同的门
                feat_hash = abs(hash(feature)) % (self.dimensions - 1)
                transformation_matrix[feat_hash, feat_hash] = np.exp(1j * phase_shift)
        
        # 应用变换矩阵到量子态
        transformed_state = np.dot(transformation_matrix, quantum_state)
        
        # 归一化
        norm = np.sqrt(np.sum(np.abs(transformed_state) ** 2))
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
        计算预测结果
        
        Args:
            quantum_state: 量子态
            
        Returns:
            预测结果字典，包含方向、强度和置信度
        """
        # 分离量子态的实部和虚部
        real_part = np.real(quantum_state)
        imag_part = np.imag(quantum_state)
        
        # 计算方向（实部的加权和）
        direction_weights = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
        if len(direction_weights) > len(real_part):
            direction_weights = direction_weights[:len(real_part)]
        elif len(direction_weights) < len(real_part):
            additional_weights = np.ones(len(real_part) - len(direction_weights)) * 0.05
            direction_weights = np.concatenate([direction_weights, additional_weights])
            direction_weights = direction_weights / np.sum(direction_weights)  # 重新归一化
            
        # 计算方向分数（正值表示上涨，负值表示下跌）
        direction = np.sum(real_part * direction_weights)
        
        # 使用量子态的模计算预测强度（0-1之间）
        # 使用前4个维度的幅度平均值
        amplitudes = np.abs(quantum_state[:4]) if len(quantum_state) >= 4 else np.abs(quantum_state)
        strength = np.mean(amplitudes)
        
        # 计算置信度（量子态的相干性）
        # 量子态越纯粹，置信度越高
        coherence = 0
        for i in range(len(quantum_state)):
            for j in range(i+1, len(quantum_state)):
                coherence += np.abs(quantum_state[i] * np.conj(quantum_state[j]))
        
        # 归一化置信度到0-1之间
        max_coherence = (len(quantum_state) * (len(quantum_state) - 1)) / 2
        confidence = coherence / max_coherence if max_coherence > 0 else 0
        
        return {
            'predicted_direction': direction,  # 预测方向（>0上涨，<0下跌）
            'prediction_strength': strength,    # 预测强度（0-1）
            'confidence': confidence,           # 置信度（0-1）
            'quantum_amplitudes': list(np.abs(quantum_state)),  # 量子态幅度
            'quantum_phases': list(np.angle(quantum_state))     # 量子态相位
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
            predicted_direction = last_prediction['predicted_direction']
            
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