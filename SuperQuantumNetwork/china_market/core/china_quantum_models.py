#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中国股市量子模型 - 超神系统专用模型
包含政策量子场、板块轮动共振器和北向资金探测器
"""

import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field

# 设置日志
logger = logging.getLogger(__name__)

class PolicyQuantumField:
    """政策量子场 - 捕捉政策变化对市场的影响"""
    
    def __init__(self, dimensions=8):
        self.dimensions = dimensions
        self.policy_vectors = {}  # 政策向量空间
        self.market_policy_alignment = {}  # 市场与政策的对齐度
        logger.info(f"政策量子场初始化完成，维度={dimensions}")
    
    def add_policy_event(self, policy_type, strength, affected_sectors, description=""):
        """添加新的政策事件"""
        # 生成政策量子向量
        policy_vector = np.random.normal(0, 1, (self.dimensions, 2))
        policy_complex = policy_vector[:, 0] + 1j * policy_vector[:, 1]
        
        # 归一化
        norm = np.sqrt(np.sum(np.abs(policy_complex) ** 2))
        policy_complex = policy_complex / norm * strength
        
        # 存储政策向量
        policy_id = f"{policy_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.policy_vectors[policy_id] = {
            'vector': policy_complex,
            'strength': strength,
            'affected_sectors': affected_sectors,
            'timestamp': datetime.now(),
            'decay_rate': 0.05,  # 每天衰减5%
            'description': description
        }
        
        logger.info(f"添加政策事件: {policy_type}, 强度={strength}, 影响行业={affected_sectors}")
        return policy_id
        
    def calculate_policy_impact(self, sector, quantum_state):
        """计算政策对特定行业的量子态影响"""
        impact = np.zeros(self.dimensions, dtype=complex)
        
        for policy_id, policy_data in self.policy_vectors.items():
            if sector in policy_data['affected_sectors']:
                # 计算时间衰减
                days_passed = (datetime.now() - policy_data['timestamp']).days
                decay = (1 - policy_data['decay_rate']) ** days_passed
                
                # 应用衰减后的政策向量
                impact += policy_data['vector'] * decay
        
        # 计算政策场与量子态的干涉
        alignment = np.abs(np.vdot(quantum_state, impact)) / \
                   (np.linalg.norm(quantum_state) * np.linalg.norm(impact) + 1e-10)
        
        self.market_policy_alignment[sector] = alignment
        return alignment, impact
    
    def get_active_policies(self, threshold=0.1):
        """获取当前活跃的政策（影响力超过阈值）"""
        active_policies = []
        for policy_id, policy_data in self.policy_vectors.items():
            # 计算时间衰减
            days_passed = (datetime.now() - policy_data['timestamp']).days
            decay = (1 - policy_data['decay_rate']) ** days_passed
            
            effective_strength = policy_data['strength'] * decay
            if effective_strength > threshold:
                active_policies.append({
                    'policy_id': policy_id,
                    'effective_strength': effective_strength,
                    'affected_sectors': policy_data['affected_sectors'],
                    'description': policy_data['description'],
                    'days_active': days_passed
                })
        
        return sorted(active_policies, key=lambda x: x['effective_strength'], reverse=True)


class SectorRotationResonator:
    """板块轮动量子共振器 - 检测和预测A股市场板块轮动"""
    
    def __init__(self, sectors):
        self.sectors = sectors
        self.sector_states = {sector: np.zeros(8, dtype=complex) for sector in sectors}
        # 初始化量子态
        for sector in sectors:
            state = np.random.normal(0, 1, (8, 2))
            complex_state = state[:, 0] + 1j * state[:, 1]
            norm = np.sqrt(np.sum(np.abs(complex_state) ** 2))
            if norm > 0:
                self.sector_states[sector] = complex_state / norm
                
        self.rotation_phase = 0.0
        self.momentum_scores = {sector: 0.0 for sector in sectors}
        self.resonance_history = []
        logger.info(f"板块轮动量子共振器初始化完成，跟踪{len(sectors)}个行业板块")
    
    def update_sector_states(self, sector_data, policy_field=None):
        """更新各板块量子态"""
        for sector, data in sector_data.items():
            if sector not in self.sector_states:
                continue
                
            # 基于板块表现更新量子态
            price_change = data.get('price_change_pct', 0)
            volume_change = data.get('turnover_rate', 0) # 使用换手率代替成交量变化
            north_flow = data.get('north_bound_ratio', 0)  # 北向资金占比
            
            # 计算相位旋转
            phase_shift = price_change * np.pi * 0.2
            
            # 创建旋转矩阵
            rotation = np.eye(8, dtype=complex)
            rotation[0, 0] = np.cos(phase_shift) + 0j
            rotation[0, 1] = -np.sin(phase_shift) + 0j
            rotation[1, 0] = np.sin(phase_shift) + 0j
            rotation[1, 1] = np.cos(phase_shift) + 0j
            
            # 应用旋转
            self.sector_states[sector] = rotation @ self.sector_states[sector]
            
            # 应用政策影响（如果有）
            if policy_field:
                policy_alignment, policy_impact = policy_field.calculate_policy_impact(sector, self.sector_states[sector])
                self.sector_states[sector] += policy_impact * 0.1  # 添加10%政策影响
            
            # 归一化
            norm = np.sqrt(np.sum(np.abs(self.sector_states[sector]) ** 2))
            if norm > 0:
                self.sector_states[sector] /= norm
            
            # 计算动量得分
            momentum = price_change * 0.5 + volume_change * 0.3 + north_flow * 0.2
            self.momentum_scores[sector] = momentum
    
    def detect_rotation(self):
        """检测板块轮动模式"""
        # 排序板块动量从高到低
        sorted_sectors = sorted(self.momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 计算轮动相位
        current_phase = sum([i * score for i, (_, score) in enumerate(sorted_sectors)]) / len(sorted_sectors)
        phase_change = current_phase - self.rotation_phase
        self.rotation_phase = current_phase
        
        # 计算板块间量子干涉
        interference_matrix = {}
        for s1 in self.sectors:
            for s2 in self.sectors:
                if s1 != s2:
                    # 计算量子态重叠
                    overlap = np.abs(np.vdot(self.sector_states[s1], self.sector_states[s2]))
                    # 使用字符串键而不是元组
                    interference_matrix[f"{s1}_{s2}"] = overlap
        
        # 记录到历史
        self.resonance_history.append({
            'timestamp': datetime.now(),
            'top_sectors': [s for s, _ in sorted_sectors[:3]],
            'bottom_sectors': [s for s, _ in sorted_sectors[-3:]],
            'phase': current_phase,
            'phase_change': phase_change
        })
        
        # 保持历史记录在合理范围内
        if len(self.resonance_history) > 30:
            self.resonance_history.pop(0)
        
        rotation_result = {
            'current_hot_sectors': [s for s, _ in sorted_sectors[:3]],
            'rotation_phase': current_phase,
            'rotation_speed': phase_change,
            'interference': interference_matrix,
            'next_sectors_prediction': self._predict_next_rotation()
        }
        
        logger.info(f"板块轮动检测: 当前热点={rotation_result['current_hot_sectors']}, 预测下一轮热点={rotation_result['next_sectors_prediction']}")
        return rotation_result
    
    def _predict_next_rotation(self):
        """预测下一轮热点板块"""
        # A股板块轮动分析算法
        if len(self.resonance_history) < 5:
            return []
            
        # 近期热点板块
        recent_hot = set()
        for record in self.resonance_history[-5:]:
            for sector in record['top_sectors']:
                recent_hot.add(sector)
        
        # 计算动量变化率
        momentum_change = {}
        if len(self.resonance_history) >= 3:
            for sector in self.sectors:
                # 检查该板块是否出现在最近3天的记录中
                appearances = []
                for i, record in enumerate(self.resonance_history[-3:]):
                    if sector in record['top_sectors']:
                        appearances.append((i, record['top_sectors'].index(sector)))
                
                # 计算出现位置的变化趋势
                if appearances:
                    # 板块位置上升表示动量增加
                    position_change = 0
                    for i in range(len(appearances)-1):
                        curr_day, curr_pos = appearances[i]
                        next_day, next_pos = appearances[i+1]
                        # 位置变化（越小越好）
                        pos_change = next_pos - curr_pos
                        # 乘以时间因子
                        day_change = next_day - curr_day
                        position_change += pos_change * (0.7 ** day_change)
                    
                    momentum_change[sector] = -position_change  # 负的位置变化表示排名提升
        
        # 找出当前动量正在增加但还未成为热点的板块
        candidates = []
        for sector, momentum in self.momentum_scores.items():
            if sector not in recent_hot and momentum > 0:
                # 基础得分是当前动量
                score = momentum
                
                # 如果有动量变化数据，进一步调整得分
                if sector in momentum_change:
                    score += momentum_change[sector] * 0.5
                    
                candidates.append((sector, score))
        
        # 返回得分最高的几个板块
        return [s for s, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:3]]


class NorthboundFlowDetector:
    """北向资金量子探测器 - 分析北向资金对A股的影响"""
    
    def __init__(self):
        self.flow_history = []
        self.stock_preference = {}  # 北向资金对个股的偏好
        self.sector_flows = {}  # 行业资金流向
        logger.info("北向资金量子探测器初始化完成")
    
    def update_flows(self, today_flow_data):
        """更新北向资金数据"""
        self.flow_history.append({
            'date': datetime.now().date(),
            'total_inflow': today_flow_data['total_inflow'],
            'stock_flows': today_flow_data['stock_flows'],
            'sector_flows': today_flow_data.get('sector_flows', {})
        })
        
        # 保持历史记录在合理范围内
        if len(self.flow_history) > 60:  # 保留60天数据
            self.flow_history.pop(0)
        
        # 更新个股偏好
        for stock, flow in today_flow_data['stock_flows'].items():
            if stock not in self.stock_preference:
                self.stock_preference[stock] = []
            
            # 添加新数据
            self.stock_preference[stock].append(flow)
            
            # 保留最近30天数据
            if len(self.stock_preference[stock]) > 30:
                self.stock_preference[stock].pop(0)
        
        # 更新行业资金流向
        for sector, flow in today_flow_data.get('sector_flows', {}).items():
            if sector not in self.sector_flows:
                self.sector_flows[sector] = []
                
            self.sector_flows[sector].append(flow)
            
            if len(self.sector_flows[sector]) > 30:
                self.sector_flows[sector].pop(0)
                
        logger.info(f"更新北向资金数据: 净流入{today_flow_data['total_inflow']/100000000:.2f}亿元")
    
    def calculate_flow_momentum(self):
        """计算北向资金动量"""
        if len(self.flow_history) < 5:
            return {}
            
        # 计算5日、10日、20日资金净流入
        flow_5d = sum([day['total_inflow'] for day in self.flow_history[-5:]])
        flow_10d = sum([day['total_inflow'] for day in self.flow_history[-10:]]) if len(self.flow_history) >= 10 else 0
        flow_20d = sum([day['total_inflow'] for day in self.flow_history[-20:]]) if len(self.flow_history) >= 20 else 0
        
        # 计算行业资金流向动量
        sector_momentum = {}
        for sector, flows in self.sector_flows.items():
            if len(flows) >= 5:
                short_term = sum(flows[-5:])
                long_term = sum(flows[-10:]) / 2 if len(flows) >= 10 else 0
                
                # 计算短期与长期的差异，正值表示资金流入加速
                sector_momentum[sector] = short_term - long_term
        
        # 计算个股资金流向动量
        stock_momentum = {}
        for stock, flows in self.stock_preference.items():
            if len(flows) >= 5:
                # 使用指数加权平均计算近期趋势
                weights = np.exp(np.linspace(0, 1, len(flows[-5:])))
                weights = weights / weights.sum()
                weighted_flows = np.dot(flows[-5:], weights)
                
                stock_momentum[stock] = weighted_flows
                
        # 判断整体趋势
        flow_trend = "加速流入" if flow_5d > flow_10d/2 > flow_20d/4 else \
                    "持续流入" if flow_5d > 0 and flow_10d > 0 else \
                    "加速流出" if flow_5d < flow_10d/2 < flow_20d/4 else \
                    "持续流出" if flow_5d < 0 and flow_10d < 0 else "震荡"
                    
        result = {
            'total_flow_5d': flow_5d,
            'total_flow_10d': flow_10d,
            'total_flow_20d': flow_20d,
            'sector_momentum': sector_momentum,
            'stock_momentum': stock_momentum,
            'flow_trend': flow_trend
        }
        
        logger.info(f"北向资金动量: 5日{flow_5d/100000000:.2f}亿, 趋势={flow_trend}")
        return result
    
    def get_favored_stocks(self, top_n=10):
        """获取北向资金偏好的个股"""
        if not self.stock_preference:
            return []
            
        # 计算每只股票的整体得分
        stock_scores = {}
        for stock, flows in self.stock_preference.items():
            if len(flows) < 5:
                continue
                
            # 最近5日平均净买入
            recent_avg = sum(flows[-5:]) / 5
            
            # 趋势得分（最近3日是否加速）
            trend_score = 0
            if len(flows) >= 3:
                if flows[-1] > flows[-2] > flows[-3]:
                    trend_score = 1  # 加速流入
                elif flows[-1] < flows[-2] < flows[-3]:
                    trend_score = -1  # 加速流出
            
            # 连续性得分（连续净买入/卖出天数）
            continuity = 0
            for flow in reversed(flows):
                if (flow > 0 and continuity >= 0) or (flow < 0 and continuity <= 0):
                    continuity += 1 if flow > 0 else -1
                else:
                    break
                    
            # 综合得分
            score = recent_avg * 0.6 + abs(trend_score) * 0.2 + abs(continuity) * 0.2
            if trend_score < 0 or continuity < 0:
                score = -score  # 负值表示卖出倾向
                
            stock_scores[stock] = score
            
        # 排序并返回前N个
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_stocks[:top_n] 