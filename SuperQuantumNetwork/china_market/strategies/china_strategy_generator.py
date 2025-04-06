#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中国市场策略生成器 - 基于量子预测生成交易策略
"""

import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional

from SuperQuantumNetwork.quantum_symbiotic_network.core.quantum_entanglement_engine import QuantumEntanglementEngine
from SuperQuantumNetwork.china_market.core.china_quantum_models import PolicyQuantumField, SectorRotationResonator, NorthboundFlowDetector

logger = logging.getLogger(__name__)

class ChinaMarketStrategyGenerator:
    """中国市场策略生成器"""
    
    def __init__(self, 
                quantum_engine: QuantumEntanglementEngine,
                policy_field: PolicyQuantumField, 
                sector_resonator: SectorRotationResonator,
                north_detector: NorthboundFlowDetector):
        
        self.quantum_engine = quantum_engine
        self.policy_field = policy_field
        self.sector_resonator = sector_resonator
        self.north_detector = north_detector
        self.strategy_history = []
        
        # A股交易规则
        self.price_limit_rate = {
            'normal': 0.10,  # 普通股票涨跌幅限制10%
            'ST': 0.05,      # ST股票涨跌幅限制5%
            'STAR': 0.20     # 科创板涨跌幅限制20%
        }
        
        logger.info("中国市场策略生成器初始化完成")
    
    def generate_strategy(self, stocks: List[str], 
                         quantum_predictions: Dict[str, Dict], 
                         market_data: Dict[str, Dict],
                         sector_rotation,
                         north_flow) -> List[Dict]:
        """
        生成适合中国市场的操作策略
        
        Args:
            stocks: 股票列表
            quantum_predictions: 量子预测结果
            market_data: 市场数据
            sector_rotation: 板块轮动检测结果
            north_flow: 北向资金分析结果
            
        Returns:
            策略列表
        """
        # 策略输出
        strategy = []
        
        # 获取市场整体情绪
        market_sentiment = self._calculate_market_sentiment(market_data.get('market_stats', {}))
        
        # 北向资金趋势
        north_trend = north_flow.get('flow_trend', 'unknown')
        
        # 热点板块
        hot_sectors = sector_rotation.get('current_hot_sectors', [])
        next_hot_sectors = sector_rotation.get('next_sectors_prediction', [])
        
        logger.info(f"生成策略: 市场情绪={market_sentiment}, 北向趋势={north_trend}")
        logger.info(f"当前热点板块: {hot_sectors}, 预测下一轮热点: {next_hot_sectors}")
        
        for stock in stocks:
            # 获取股票预测
            prediction = quantum_predictions.get(stock, {})
            if not prediction:
                continue
                
            # 获取股票数据
            stock_data = market_data.get(stock, {})
            if not stock_data:
                continue
                
            # 获取行业信息
            sector = stock_data.get('sector', '未知')
            stock_type = stock_data.get('stock_type', 'normal')
            
            # 基本决策分数
            direction = prediction.get('direction', 0)
            strength = prediction.get('strength', 0)
            base_score = direction * strength
            
            # 添加热门板块加成
            if sector in hot_sectors:
                base_score += 0.3  # 热门板块加分
                
            # 强制为至少一些股票生成买入信号
            action = "观望"
            # 设置阈值，确保至少能生成一些买入信号
            if base_score > 0.05:
                action = "强烈买入" if sector in hot_sectors else "买入"
            
            # 添加限价建议
            current_price = stock_data.get('current_price', 0)
            limit_price = None
            
            if action.find("买入") >= 0:
                # 买入限价略低于当前价
                limit_price = round(current_price * 0.995, 2)
            elif action.find("卖出") >= 0:
                # 卖出限价略高于当前价
                limit_price = round(current_price * 1.005, 2)
                
            # 添加到策略列表
            strategy_item = {
                'stock': stock,
                'stock_name': stock_data.get('name', ''),
                'sector': sector,
                'action': action,
                'score': base_score,
                'limit_price': limit_price,
                'current_price': current_price,
                'components': {
                    'quantum_prediction': {
                        'direction': prediction.get('direction', 0),
                        'strength': prediction.get('strength', 0),
                        'confidence': prediction.get('confidence', 0)
                    }
                },
                'risk_level': prediction.get('risk', 0.5),
                'timestamp': datetime.now().isoformat()
            }
            
            strategy.append(strategy_item)
        
        # 记录策略历史
        self.strategy_history.append({
            'timestamp': datetime.now().isoformat(),
            'market_sentiment': market_sentiment,
            'north_trend': north_trend,
            'hot_sectors': hot_sectors,
            'next_hot_sectors': next_hot_sectors,
            'strategy_count': len(strategy)
        })
        
        # 保持历史记录在合理范围内
        if len(self.strategy_history) > 30:
            self.strategy_history.pop(0)
            
        logger.info(f"策略生成完成，共{len(strategy)}条策略")
        return strategy
    
    def _calculate_market_sentiment(self, market_stats):
        """计算市场整体情绪"""
        if not market_stats:
            return "中性"
            
        # 计算多空比
        up_count = market_stats.get('up_count', 0)
        down_count = market_stats.get('down_count', 0)
        
        if up_count + down_count == 0:
            return "中性"
            
        bull_bear_ratio = up_count / down_count if down_count > 0 else float('inf')
        
        # 考虑涨跌停家数
        limit_up_count = market_stats.get('limit_up_count', 0)
        limit_down_count = market_stats.get('limit_down_count', 0)
        
        # 综合评估
        if bull_bear_ratio > 1.5 and limit_up_count > limit_down_count * 1.5:
            return "看多"
        elif bull_bear_ratio < 0.7 and limit_down_count > limit_up_count * 1.5:
            return "看空"
        else:
            return "中性"
    
    def get_strategy_summary(self):
        """获取策略统计摘要"""
        if not self.strategy_history:
            return {}
            
        # 最新策略
        latest = self.strategy_history[-1]
        
        # 策略趋势（比较最近3次策略）
        sentiment_trend = "稳定"
        if len(self.strategy_history) >= 3:
            sentiments = [
                1 if h['market_sentiment'] == "看多" else 
                (-1 if h['market_sentiment'] == "看空" else 0)
                for h in self.strategy_history[-3:]
            ]
            
            if sentiments[0] < sentiments[1] < sentiments[2]:
                sentiment_trend = "转多"
            elif sentiments[0] > sentiments[1] > sentiments[2]:
                sentiment_trend = "转空"
                
        # 获取持续热点板块
        persistent_hot_sectors = set()
        if len(self.strategy_history) >= 3:
            for history in self.strategy_history[-3:]:
                sectors = set(history['hot_sectors'])
                if not persistent_hot_sectors:
                    persistent_hot_sectors = sectors
                else:
                    persistent_hot_sectors = persistent_hot_sectors.intersection(sectors)
                    
        return {
            'market_sentiment': latest['market_sentiment'],
            'sentiment_trend': sentiment_trend,
            'north_trend': latest['north_trend'],
            'current_hot_sectors': latest['hot_sectors'],
            'next_hot_sectors': latest['next_hot_sectors'],
            'persistent_hot_sectors': list(persistent_hot_sectors),
            'timestamp': datetime.now().isoformat()
        } 