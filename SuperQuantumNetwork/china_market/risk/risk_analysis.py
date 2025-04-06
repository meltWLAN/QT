import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class ChinaMarketRiskAnalyzer:
    """中国市场风险分析器"""
    
    def __init__(self):
        self.risk_factors = {
            'market_volatility': 0.0,  # 市场波动性
            'liquidity_risk': 0.0,     # 流动性风险
            'policy_risk': 0.0,        # 政策风险
            'sectoral_risk': {},       # 行业风险
            'stock_specific_risk': {}  # 个股风险
        }
        self.risk_history = []         # 风险历史
        self.overall_risk = 0.0        # 整体风险水平
        logger.info("风险分析器初始化完成")
    
    def analyze_market_risk(self, market_data, 
                          policy_field, 
                          north_flow, 
                          sector_rotation):
        """分析市场整体风险"""
        # 获取市场统计数据
        stats = market_data.get('market_stats', {})
        indices = {
            'sh': market_data.get('sh_index', {}),
            'sz': market_data.get('sz_index', {}),
            'cyb': market_data.get('cyb_index', {})
        }
        
        # 1. 市场波动性风险
        # 计算指数波动率
        volatility = 0
        count = 0
        for idx_name, idx_data in indices.items():
            if idx_data and 'change_pct' in idx_data:
                # 确保change_pct是有效数值
                change_pct = idx_data.get('change_pct', 0)
                if isinstance(change_pct, (int, float)) and not np.isnan(change_pct):
                    volatility += abs(change_pct) * 0.33
                    count += 1
                
        # 如果没有有效数据，设置默认值
        if count == 0:
            volatility = 0.33  # 默认中等波动率
                
        # 计算涨跌停板数量占比
        total_stocks = stats.get('up_count', 0) + stats.get('down_count', 0)
        if total_stocks > 0:
            limit_ratio = (stats.get('limit_up_count', 0) + 
                          stats.get('limit_down_count', 0)) / total_stocks
            volatility += limit_ratio * 5  # 放大涨跌停影响
            
        self.risk_factors['market_volatility'] = min(volatility, 1.0)
        
        # 2. 流动性风险
        liquidity_risk = 0
        # 成交量变化
        avg_volume = market_data.get('avg_volume', 0)
        current_volume = stats.get('total_volume', 0)
        if avg_volume > 0:
            volume_change = current_volume / avg_volume - 1
            # 成交量过低表示流动性风险
            if volume_change < -0.3:
                liquidity_risk += abs(volume_change) * 0.5
        
        # 北向资金流向
        if north_flow.get('flow_trend') in ['加速流出', '持续流出']:
            liquidity_risk += 0.3
            
        self.risk_factors['liquidity_risk'] = min(liquidity_risk, 1.0)
        
        # 3. 政策风险
        policy_risk = 0
        active_policies = policy_field.get_active_policies(threshold=0.05)
        # 活跃政策数量越多，不确定性越高
        policy_risk += min(len(active_policies) * 0.05, 0.3)
        
        # 政策对市场的对齐度
        avg_alignment = 0
        alignments = list(policy_field.market_policy_alignment.values())
        if alignments:
            avg_alignment = sum(alignments) / len(alignments)
            # 对齐度低表示政策与市场方向不一致，风险更高
            policy_risk += (1 - avg_alignment) * 0.7
            
        self.risk_factors['policy_risk'] = min(policy_risk, 1.0)
        
        # 4. 行业风险
        sector_risks = {}
        rotation_data = sector_rotation.detect_rotation()
        
        # 根据板块轮动计算行业风险
        # 热点板块可能有泡沫风险
        for sector in rotation_data.get('current_hot_sectors', []):
            sector_risks[sector] = 0.6
            
        # 轮动速度快的行业风险高
        rotation_speed = abs(rotation_data.get('rotation_speed', 0))
        if rotation_speed > 0.5:
            for sector in rotation_data.get('current_hot_sectors', []):
                if sector in sector_risks:
                    sector_risks[sector] += 0.2
                    
        self.risk_factors['sectoral_risk'] = sector_risks
        
        # 计算整体风险 - 确保没有NaN值
        market_vol = self.risk_factors.get('market_volatility', 0.33)
        liquidity = self.risk_factors.get('liquidity_risk', 0.0)
        policy = self.risk_factors.get('policy_risk', 0.5)
        
        # 使用有效值计算整体风险
        if np.isnan(market_vol):
            market_vol = 0.33
        if np.isnan(liquidity):
            liquidity = 0.0
        if np.isnan(policy):
            policy = 0.5
            
        self.overall_risk = (
            market_vol * 0.3 +
            liquidity * 0.3 +
            policy * 0.4
        )
        
        # 保证整体风险是有效值
        if np.isnan(self.overall_risk):
            self.overall_risk = 0.5  # 默认中等风险水平
        
        # 记录风险历史
        self.risk_history.append({
            'timestamp': datetime.now(),
            'overall_risk': self.overall_risk,
            'factors': {k: v for k, v in self.risk_factors.items() 
                       if k != 'sectoral_risk' and k != 'stock_specific_risk'},
            'top_sector_risks': sorted(sector_risks.items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
        })
        
        # 保持历史记录在合理范围内
        if len(self.risk_history) > 30:
            self.risk_history.pop(0)
            
        logger.info(f"风险分析完成: 整体风险={self.overall_risk:.2f}, 市场波动={self.risk_factors['market_volatility']:.2f}, 流动性={self.risk_factors['liquidity_risk']:.2f}, 政策={self.risk_factors['policy_risk']:.2f}")
        
        return {
            'overall_risk': self.overall_risk,
            'risk_level': self._risk_level_description(),
            'risk_factors': self.risk_factors,
            'risk_trend': self._calculate_risk_trend(),
            'high_risk_sectors': [s for s, r in sector_risks.items() if r > 0.6]
        }
    
    def analyze_stock_risk(self, stock, stock_data, quantum_prediction):
        """分析个股风险"""
        if not stock_data:
            return 0.5
            
        # 基础风险分数
        base_risk = 0.5
        
        # 1. 波动性风险
        volatility = stock_data.get('volatility', 0)
        volatility_risk = min(volatility * 5, 1.0)  # 波动率越高风险越大
        
        # 2. 流动性风险
        turnover = stock_data.get('turnover_rate', 0)
        liquidity_risk = 0
        if turnover < 0.01:  # 换手率低于1%
            liquidity_risk = 0.8
        elif turnover < 0.03:  # 换手率低于3%
            liquidity_risk = 0.5
        elif turnover < 0.05:  # 换手率低于5%
            liquidity_risk = 0.3
        else:
            liquidity_risk = 0.1
            
        # 3. 趋势违背风险
        trend_risk = 0
        if quantum_prediction and 'confidence' in quantum_prediction:
            # 预测信心度低表示不确定性高
            trend_risk = 1 - quantum_prediction['confidence']
            
        # 4. 行业风险
        sector = stock_data.get('sector', '')
        sector_risk = self.risk_factors['sectoral_risk'].get(sector, 0.5)
        
        # 5. 突发事件风险
        event_risk = stock_data.get('event_risk', 0)
        
        # 综合风险计算
        stock_risk = (
            base_risk * 0.1 +
            volatility_risk * 0.2 +
            liquidity_risk * 0.2 +
            trend_risk * 0.2 +
            sector_risk * 0.2 +
            event_risk * 0.1
        )
        
        # 保存结果
        self.risk_factors['stock_specific_risk'][stock] = stock_risk
        
        return stock_risk
    
    def _risk_level_description(self):
        """返回风险水平描述"""
        if self.overall_risk < 0.2:
            return "低风险"
        elif self.overall_risk < 0.4:
            return "中低风险"
        elif self.overall_risk < 0.6:
            return "中等风险"
        elif self.overall_risk < 0.8:
            return "中高风险"
        else:
            return "高风险"
    
    def _calculate_risk_trend(self):
        """计算风险趋势"""
        if len(self.risk_history) < 3:
            return "稳定"
            
        recent_risks = [record['overall_risk'] for record in self.risk_history[-3:]]
        
        if recent_risks[-1] > recent_risks[-2] > recent_risks[-3]:
            return "上升"
        elif recent_risks[-1] < recent_risks[-2] < recent_risks[-3]:
            return "下降"
        else:
            return "震荡"
    
    def get_risk_alerts(self, threshold=0.7):
        """获取风险预警"""
        alerts = []
        
        # 整体市场风险预警
        if self.overall_risk > threshold:
            alerts.append({
                'type': 'market',
                'level': 'high',
                'message': f"市场整体风险较高({self.overall_risk:.2f})，建议降低仓位"
            })
            
        # 行业风险预警
        for sector, risk in self.risk_factors['sectoral_risk'].items():
            if risk > threshold:
                alerts.append({
                    'type': 'sector',
                    'level': 'high',
                    'sector': sector,
                    'message': f"行业{sector}风险较高({risk:.2f})，建议减少该行业持仓"
                })
                
        # 个股风险预警
        for stock, risk in self.risk_factors['stock_specific_risk'].items():
            if risk > threshold:
                alerts.append({
                    'type': 'stock',
                    'level': 'high',
                    'stock': stock,
                    'message': f"个股{stock}风险较高({risk:.2f})，建议减持或观望"
                })
                
        # 政策风险预警
        if self.risk_factors['policy_risk'] > threshold:
            alerts.append({
                'type': 'policy',
                'level': 'high',
                'message': f"政策不确定性风险较高({self.risk_factors['policy_risk']:.2f})，建议关注政策动向"
            })
            
        # 流动性风险预警
        if self.risk_factors['liquidity_risk'] > threshold:
            alerts.append({
                'type': 'liquidity',
                'level': 'high',
                'message': f"市场流动性风险较高({self.risk_factors['liquidity_risk']:.2f})，建议避免交易低流动性股票"
            })
            
        return alerts 