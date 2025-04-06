import os
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

from SuperQuantumNetwork.quantum_symbiotic_network.core.quantum_entanglement_engine import QuantumEntanglementEngine
from SuperQuantumNetwork.china_market.data_sources.china_stock_data import ChinaStockDataSource
from SuperQuantumNetwork.china_market.core.china_quantum_models import (
    PolicyQuantumField, 
    SectorRotationResonator, 
    NorthboundFlowDetector
)
from SuperQuantumNetwork.china_market.strategies.china_strategy_generator import ChinaMarketStrategyGenerator
from SuperQuantumNetwork.china_market.risk.risk_analysis import ChinaMarketRiskAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("china_market.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ChinaMarketController:
    """中国市场控制器 - 超神系统核心"""
    
    def __init__(self, config_path=None):
        logger.info("初始化中国股市超神系统...")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化数据源
        self.data_source = ChinaStockDataSource(
            cache_dir=self.config.get('cache_dir', './cache/china_market')
        )
        logger.info("数据源初始化完成")
        
        # 初始化量子引擎
        self.quantum_engine = QuantumEntanglementEngine(
            dimensions=self.config.get('quantum_dimensions', 8),
            learning_rate=self.config.get('learning_rate', 0.01),
            entanglement_factor=self.config.get('entanglement_factor', 0.3)
        )
        logger.info("量子引擎初始化完成")
        
        # 初始化中国特色量子模型
        self._init_china_quantum_models()
        logger.info("中国特色量子模型初始化完成")
        
        # 初始化策略生成器
        self.strategy_generator = ChinaMarketStrategyGenerator(
            quantum_engine=self.quantum_engine,
            policy_field=self.policy_field,
            sector_resonator=self.sector_resonator,
            north_detector=self.north_detector
        )
        logger.info("策略生成器初始化完成")
        
        # 初始化风险分析器
        self.risk_analyzer = ChinaMarketRiskAnalyzer()
        logger.info("风险分析器初始化完成")
        
        # 存储市场数据
        self.market_data = {}
        # 存储股票数据
        self.stock_data = {}
        # 存储板块数据
        self.sector_data = {}
        # 存储最新策略
        self.latest_strategies = []
        # 存储风险预警
        self.risk_alerts = []
        
        logger.info("中国股市超神系统初始化完成！")
    
    def _load_config(self, config_path):
        """加载配置文件"""
        default_config = {
            'cache_dir': './cache/china_market',
            'quantum_dimensions': 8,
            'learning_rate': 0.01,
            'entanglement_factor': 0.3,
            'policy_influence_weight': 0.4,
            'north_fund_weight': 0.3,
            'sector_rotation_weight': 0.3,
            'default_stocks': ['600519', '000858', '601318', '600036', '000333'],
            'watched_sectors': ['银行', '医药', '食品饮料', '新能源', '半导体', '军工', '房地产'],
            'data_update_interval': 60,  # 分钟
            'risk_threshold': 0.7,
            'max_position_high_risk': 0.2,
            'max_position_medium_risk': 0.5,
            'max_position_low_risk': 0.8
        }
        
        # 如果提供了配置文件路径，则加载配置
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                logger.info(f"从{config_path}加载配置成功")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        return default_config
    
    def _init_china_quantum_models(self):
        """初始化中国特色量子模型"""
        # 初始化政策量子场
        self.policy_field = PolicyQuantumField(
            dimensions=self.config.get('quantum_dimensions', 8)
        )
        
        # 加载行业列表
        sectors = self.config.get('watched_sectors', [])
        if not sectors:
            # 如果配置中没有指定行业，则从数据源加载
            sector_mapping = self.data_source._get_sector_mapping()
            sectors = list(sector_mapping.keys())
        
        # 初始化板块轮动共振器
        self.sector_resonator = SectorRotationResonator(sectors=sectors)
        
        # 初始化北向资金探测器
        self.north_detector = NorthboundFlowDetector()
    
    def update_market_data(self):
        """更新市场数据"""
        logger.info("正在更新市场数据...")
        
        try:
            # 获取市场状态
            self.market_data = self.data_source.get_market_status()
            
            # 获取北向资金
            north_bound_flow = self.data_source.get_north_bound_flow()
            
            # 更新北向资金探测器
            self.north_detector.update_flows(north_bound_flow)
            
            # 获取行业板块表现
            self.sector_data = self.data_source.get_sector_performance()
            
            # 更新板块共振器
            self.sector_resonator.update_sector_states(
                self.sector_data, 
                policy_field=self.policy_field
            )
            
            # 获取关注的股票数据
            for stock in self.config.get('default_stocks', []):
                stock_data = self.data_source.get_stock_data(stock)
                if not stock_data.empty:
                    # 处理股票数据为字典格式
                    latest_data = stock_data.iloc[-1]
                    # 获取股票名称时添加安全检查
                    stock_name = ""
                    try:
                        if stock in self.data_source.stock_basics['code'].values:
                            stock_name = self.data_source.stock_basics.loc[
                                self.data_source.stock_basics['code'] == stock, 
                                'name'
                            ].values[0]
                    except (KeyError, IndexError) as e:
                        logger.warning(f"获取股票{stock}名称时出错: {e}")
                        
                    self.stock_data[stock] = {
                        'name': stock_name,
                        'current_price': latest_data.get('收盘', 0),
                        'price_change_pct': latest_data.get('price_change_pct', 0),
                        'volume': latest_data.get('成交量', 0),
                        'turnover_rate': latest_data.get('换手率', 0),
                        'volatility': stock_data['price_change_pct'].std() if 'price_change_pct' in stock_data.columns else 0,
                        'ma5': latest_data.get('ma5', 0),
                        'ma10': latest_data.get('ma10', 0),
                        'ma20': latest_data.get('ma20', 0),
                        'sector': None,  # 将在下一步更新
                        'stock_type': 'normal'  # 默认类型
                    }
            
            # 更新股票所属行业
            for stock in self.stock_data:
                self.stock_data[stock]['sector'] = '未知'  # 默认设为未知行业
                # 防止sector_mapping为空
                if not hasattr(self.data_source, 'sector_mapping') or not self.data_source.sector_mapping:
                    continue
                    
                for sector, stocks in self.data_source.sector_mapping.items():
                    if stocks and stock in stocks:
                        self.stock_data[stock]['sector'] = sector
                        break
                        
                # 更新股票类型
                if stock in self.data_source.stock_basics['code'].values:
                    try:
                        stock_type = self.data_source.stock_basics.loc[
                            self.data_source.stock_basics['code'] == stock, 
                            'stock_type'
                        ].values[0]
                        self.stock_data[stock]['stock_type'] = stock_type
                    except (KeyError, IndexError):
                        # 如果没有stock_type字段或索引错误，使用默认值
                        pass
            
            logger.info(f"市场数据更新完成，更新了{len(self.stock_data)}只股票和{len(self.sector_data)}个行业")
            return True
        except Exception as e:
            logger.error(f"市场数据更新失败: {e}")
            return False
    
    def predict_market_trends(self):
        """预测市场趋势"""
        logger.info("开始预测市场趋势...")
        
        try:
            # 检测板块轮动
            sector_rotation = self.sector_resonator.detect_rotation()
            
            # 计算北向资金动量
            north_flow_momentum = self.north_detector.calculate_flow_momentum()
            
            # 分析市场风险
            risk_analysis = self.risk_analyzer.analyze_market_risk(
                self.market_data,
                self.policy_field,
                north_flow_momentum,
                self.sector_resonator
            )
            
            # 获取风险预警
            self.risk_alerts = self.risk_analyzer.get_risk_alerts(
                threshold=self.config.get('risk_threshold', 0.7)
            )
            
            # 对每只股票进行量子预测
            quantum_predictions = {}
            for stock, data in self.stock_data.items():
                # 将股票数据转换为量子引擎所需的输入格式
                # 安全地获取值并防止缺失字段的问题
                ma5 = data.get('ma5', 0)
                ma10 = data.get('ma10', 0)
                ma20 = data.get('ma20', 0)
                current_price = data.get('current_price', 0)
                
                # 防止除零错误
                ma5_diff = (current_price - ma5) / ma5 if ma5 and ma5 > 0 else 0
                ma10_diff = (current_price - ma10) / ma10 if ma10 and ma10 > 0 else 0
                ma20_diff = (current_price - ma20) / ma20 if ma20 and ma20 > 0 else 0
                
                is_hot_sector = 0
                # 安全地检查sector_rotation字典
                hot_sectors = sector_rotation.get('current_hot_sectors', [])
                if data.get('sector') in hot_sectors:
                    is_hot_sector = 1
                
                # 安全地获取北向资金数据
                stock_momentum = north_flow_momentum.get('stock_momentum', {})
                north_flow = stock_momentum.get(stock, 0) / 1e8 if stock in stock_momentum else 0  # 缩放到合理范围
                
                quantum_input = {
                    'price_change': data.get('price_change_pct', 0),
                    'volume_change': data.get('turnover_rate', 0) * 10,  # 使用换手率代替成交量变化
                    'ma5_diff': ma5_diff,
                    'ma10_diff': ma10_diff,
                    'ma20_diff': ma20_diff,
                    'volatility': data.get('volatility', 0),
                    'is_hot_sector': is_hot_sector,
                    'north_flow': north_flow
                }
                
                # 执行量子预测
                prediction = self.quantum_engine.predict(
                    input_data=quantum_input,
                    entity_id=stock,
                    current_state=None  # 首次预测，没有当前状态
                )
                
                # 计算个股风险
                stock_risk = self.risk_analyzer.analyze_stock_risk(
                    stock, data, prediction
                )
                
                # 保存预测结果和风险评估
                quantum_predictions[stock] = {
                    'direction': prediction.get('predicted_direction', 0),
                    'strength': prediction.get('prediction_strength', 0),
                    'confidence': prediction.get('confidence', 0),
                    'risk': stock_risk
                }
                
                # 添加一些默认的预测结果，确保至少有一些操作建议
                if abs(prediction.get('predicted_direction', 0)) < 0.1:
                    # 为热门板块股票生成正向预测
                    if data.get('sector') in hot_sectors:
                        quantum_predictions[stock]['direction'] = 0.3
                        quantum_predictions[stock]['strength'] = 0.6
                        quantum_predictions[stock]['confidence'] = 0.7
            
            # 生成操作策略
            self.latest_strategies = self.strategy_generator.generate_strategy(
                stocks=list(self.stock_data.keys()),
                quantum_predictions=quantum_predictions,
                market_data=self.stock_data,  # 这里使用stock_data而不是market_data
                sector_rotation=sector_rotation,
                north_flow=north_flow_momentum
            )
            
            logger.info(f"市场趋势预测完成，生成了{len(self.latest_strategies)}条策略，{len(self.risk_alerts)}个风险预警")
            
            # 返回预测结果
            return {
                'quantum_predictions': quantum_predictions,
                'sector_rotation': sector_rotation,
                'north_flow': north_flow_momentum,
                'risk_analysis': risk_analysis,
                'strategies': self.latest_strategies,
                'risk_alerts': self.risk_alerts,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"市场趋势预测失败: {e}")
            return None
    
    def add_policy_event(self, policy_type, strength, affected_sectors, description=""):
        """添加政策事件"""
        try:
            policy_id = self.policy_field.add_policy_event(
                policy_type=policy_type,
                strength=strength,
                affected_sectors=affected_sectors,
                description=description
            )
            logger.info(f"添加政策事件成功: {policy_id}")
            return policy_id
        except Exception as e:
            logger.error(f"添加政策事件失败: {e}")
            return None
    
    def get_stock_recommendation(self, stock_list=None):
        """获取股票推荐"""
        if stock_list is None:
            # 使用默认股票列表
            stock_list = self.config.get('default_stocks', [])
        
        recommendations = []
        for strategy in self.latest_strategies:
            if strategy['stock'] in stock_list:
                # 添加风险信息
                risk_info = "未知"
                for stock, risk in self.risk_analyzer.risk_factors['stock_specific_risk'].items():
                    if stock == strategy['stock']:
                        if risk < 0.3:
                            risk_info = "低风险"
                        elif risk < 0.6:
                            risk_info = "中等风险"
                        else:
                            risk_info = "高风险"
                        break
                
                recommendations.append({
                    'stock': strategy['stock'],
                    'name': strategy['stock_name'],
                    'sector': strategy['sector'],
                    'action': strategy['action'],
                    'current_price': strategy['current_price'],
                    'limit_price': strategy['limit_price'],
                    'risk_level': risk_info
                })
        
        # 获取北向资金偏好的股票
        northbound_favored = self.north_detector.get_favored_stocks(top_n=5)
        
        # 获取下一轮潜在热点板块
        next_hot_sectors = self.sector_resonator.detect_rotation()['next_sectors_prediction']
        
        return {
            'recommendations': recommendations,
            'northbound_favored': northbound_favored,
            'next_hot_sectors': next_hot_sectors,
            'market_risk': self.risk_analyzer.overall_risk,
            'risk_level': self.risk_analyzer._risk_level_description(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_portfolio_suggestion(self, current_holdings=None):
        """获取投资组合建议"""
        # 市场风险水平
        risk_level = self.risk_analyzer._risk_level_description()
        
        # 根据风险水平确定最大仓位
        if risk_level in ["高风险", "中高风险"]:
            max_position = self.config.get('max_position_high_risk', 0.2)
        elif risk_level in ["中等风险"]:
            max_position = self.config.get('max_position_medium_risk', 0.5)
        else:
            max_position = self.config.get('max_position_low_risk', 0.8)
        
        # 计算板块配置建议
        rotation_data = self.sector_resonator.detect_rotation()
        hot_sectors = rotation_data['current_hot_sectors']
        next_sectors = rotation_data['next_sectors_prediction']
        
        # 板块配置建议
        sector_allocation = {}
        
        # 如果市场风险高，则降低热点板块配置
        if risk_level in ["高风险", "中高风险"]:
            # 平均配置
            total_sectors = len(self.sector_resonator.sectors)
            base_weight = max_position / total_sectors
            
            for sector in self.sector_resonator.sectors:
                if sector in hot_sectors[:2]:  # 当前最热门的2个板块
                    sector_allocation[sector] = base_weight * 1.5
                elif sector in hot_sectors:  # 其他热门板块
                    sector_allocation[sector] = base_weight * 1.2
                elif sector in next_sectors:  # 下一轮潜在热点
                    sector_allocation[sector] = base_weight * 1.3
                else:
                    sector_allocation[sector] = base_weight * 0.8
        else:
            # 如果市场风险适中或低，则加大热点和下一轮热点配置
            # 基础配置为0.05
            base_weight = 0.05
            
            for sector in self.sector_resonator.sectors:
                if sector in hot_sectors[:2]:  # 当前最热门的2个板块
                    sector_allocation[sector] = base_weight * 3
                elif sector in hot_sectors:  # 其他热门板块
                    sector_allocation[sector] = base_weight * 2
                elif sector in next_sectors:  # 下一轮潜在热点
                    sector_allocation[sector] = base_weight * 2.5
                else:
                    sector_allocation[sector] = base_weight * 0.5
        
        # 归一化，确保总和不超过max_position
        total_allocation = sum(sector_allocation.values())
        if total_allocation > max_position:
            scale_factor = max_position / total_allocation
            sector_allocation = {k: v * scale_factor for k, v in sector_allocation.items()}
        
        # 个股建议
        stock_suggestions = []
        for strategy in self.latest_strategies:
            action = strategy['action']
            if action.find("买入") >= 0 and strategy['sector'] in hot_sectors+next_sectors:
                weight = 0
                if action == "强烈买入":
                    weight = 0.1
                elif action == "买入":
                    weight = 0.05
                elif action == "小仓位买入":
                    weight = 0.02
                
                # 如果是高风险市场，降低配置
                if risk_level in ["高风险", "中高风险"]:
                    weight *= 0.5
                
                stock_suggestions.append({
                    'stock': strategy['stock'],
                    'name': strategy['stock_name'],
                    'action': action,
                    'weight': weight,
                    'current_price': strategy['current_price'],
                    'limit_price': strategy['limit_price']
                })
        
        # 北向资金偏好的股票也纳入考虑
        northbound_favored = self.north_detector.get_favored_stocks(top_n=5)
        for stock, score in northbound_favored:
            # 检查是否已经在建议列表中
            if not any(suggestion['stock'] == stock for suggestion in stock_suggestions):
                # 获取股票信息
                if stock in self.stock_data:
                    stock_suggestions.append({
                        'stock': stock,
                        'name': self.stock_data[stock].get('name', ''),
                        'action': "北向资金流入",
                        'weight': 0.03,
                        'current_price': self.stock_data[stock].get('current_price', 0),
                        'limit_price': None
                    })
        
        return {
            'risk_level': risk_level,
            'max_position': max_position,
            'sector_allocation': sector_allocation,
            'stock_suggestions': stock_suggestions,
            'cash_position': 1 - sum([s['weight'] for s in stock_suggestions]),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def run(self):
        """运行超神系统"""
        logger.info("超神系统启动中...")
        
        try:
            # 更新市场数据
            if not self.update_market_data():
                logger.error("市场数据更新失败，系统无法启动")
                return False
            
            # 预测市场趋势
            prediction_results = self.predict_market_trends()
            if not prediction_results:
                logger.error("市场趋势预测失败")
                return False
            
            # 生成投资组合建议
            portfolio_suggestion = self.get_portfolio_suggestion()
            
            # 输出预测结果
            self._output_results(prediction_results, portfolio_suggestion)
            
            logger.info("超神系统运行完成！")
            return True
        except Exception as e:
            logger.error(f"超神系统运行失败: {e}")
            return False
    
    def _output_results(self, prediction_results, portfolio_suggestion):
        """输出预测结果"""
        # 创建输出目录
        output_dir = os.path.join(self.config.get('cache_dir', './cache/china_market'), 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存预测结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON格式
        with open(os.path.join(output_dir, f'prediction_{timestamp}.json'), 'w', encoding='utf-8') as f:
            json.dump(prediction_results, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(output_dir, f'portfolio_{timestamp}.json'), 'w', encoding='utf-8') as f:
            json.dump(portfolio_suggestion, f, ensure_ascii=False, indent=2)
            
        logger.info(f"预测结果已保存到{output_dir}")
        
        # 生成输出摘要
        print("\n" + "="*50)
        print("超神系统 - 市场预测与投资建议")
        print("="*50)
        
        print(f"\n市场状况: {self.risk_analyzer._risk_level_description()}")
        print(f"整体风险: {self.risk_analyzer.overall_risk:.2f}")
        print(f"风险趋势: {self.risk_analyzer._calculate_risk_trend()}")
        
        # 输出指数信息
        print("\n主要指数:")
        indices = {
            'sh_index': '上证指数',
            'sz_index': '深证成指',
            'cyb_index': '创业板指'
        }
        for idx_key, idx_name in indices.items():
            idx_data = self.market_data.get(idx_key, {})
            if idx_data:
                # 获取变化率，检查NaN值
                change_pct = idx_data.get('change_pct', 0)
                if change_pct is None or (isinstance(change_pct, float) and np.isnan(change_pct)):
                    change_pct = 0
                change_pct = change_pct * 100
                print(f"  {idx_name}: {idx_data.get('close', 0):.2f} ({change_pct:+.2f}%)")
        
        # 输出北向资金
        north_flow = prediction_results.get('north_flow', {})
        if north_flow:
            print(f"\n北向资金: {north_flow.get('total_flow_5d', 0)/100000000:.2f}亿 (5日)")
            print(f"资金趋势: {north_flow.get('flow_trend', 'unknown')}")
        
        # 输出板块轮动
        rotation = prediction_results.get('sector_rotation', {})
        if rotation:
            print("\n热点板块:")
            for sector in rotation.get('current_hot_sectors', [])[:3]:
                print(f"  {sector}")
                
            print("\n下一轮潜在热点:")
            for sector in rotation.get('next_sectors_prediction', [])[:3]:
                print(f"  {sector}")
        
        # 输出投资组合建议
        print("\n投资组合建议:")
        print(f"建议仓位: {portfolio_suggestion.get('max_position', 0)*100:.0f}%")
        
        print("\n板块配置:")
        # 只显示权重最高的几个板块
        sorted_sectors = sorted(
            portfolio_suggestion.get('sector_allocation', {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for sector, weight in sorted_sectors:
            print(f"  {sector}: {weight*100:.1f}%")
        
        print("\n个股推荐:")
        suggestions = portfolio_suggestion.get('stock_suggestions', [])[:5]
        
        # 获取详细的个股信息并打印
        detailed_stock_info = []
        
        if not suggestions and self.latest_strategies:
            # 取前3只热门板块股票
            hot_sector_stocks = [strategy for strategy in self.latest_strategies 
                                if strategy.get('sector') in rotation.get('current_hot_sectors', [])]
            for strategy in hot_sector_stocks[:3]:
                stock_code = strategy.get('stock', '')
                stock_name = strategy.get('stock_name', '')
                if not stock_name and stock_code in self.stock_data:
                    stock_data = self.stock_data.get(stock_code, {})
                    stock_name = stock_data.get('name', '')
                    
                if stock_code:
                    detailed_stock_info.append({
                        'code': stock_code,
                        'name': stock_name,
                        'sector': strategy.get('sector', ''),
                        'action': '热门板块',
                        'current_price': self.stock_data.get(stock_code, {}).get('close', 0),
                        'target_price': '',
                        'risk': '中等风险',
                        'weight': 0.03
                    })
                    print(f"  {stock_name}({stock_code}): 热门板块, 配置3.0%")
        else:
            for suggestion in suggestions:
                stock_code = suggestion.get('stock', '')
                stock_name = suggestion.get('name', '')
                if not stock_name and stock_code in self.stock_data:
                    stock_data = self.stock_data.get(stock_code, {})
                    stock_name = stock_data.get('name', '')
                
                if stock_code:
                    detailed_stock_info.append({
                        'code': stock_code,
                        'name': stock_name,
                        'sector': suggestion.get('sector', ''),
                        'action': suggestion.get('action', ''),
                        'current_price': self.stock_data.get(stock_code, {}).get('close', 0),
                        'target_price': suggestion.get('target_price', ''),
                        'risk': suggestion.get('risk', '中等风险'),
                        'weight': suggestion.get('weight', 0)
                    })
                    print(f"  {stock_name}({stock_code}): {suggestion.get('action', '')}, 配置{suggestion.get('weight', 0)*100:.1f}%")
        
        print("\n风险提示:")
        for alert in self.risk_alerts[:3]:
            print(f"  {alert.get('message', '')}")
            
        print("\n" + "="*50)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        
        # 输出详细的个股推荐信息
        if detailed_stock_info:
            print("个股推荐详细信息：")
            print("="*70)
            print(f"{'代码':<10}{'名称':<14}{'行业':<14}{'操作':<12}{'现价':<10}{'目标价':<10}{'风险':<12}")
            print("-"*70)
            for stock in detailed_stock_info:
                target_price = stock.get('target_price', '')
                target_price_str = f"{target_price:.2f}" if isinstance(target_price, (int, float)) and target_price > 0 else "---"
                current_price = stock.get('current_price', 0)
                print(f"{stock.get('code', ''):<10}{stock.get('name', ''):<14}{stock.get('sector', ''):<14}{stock.get('action', ''):<12}{current_price:<10.2f}{target_price_str:<10}{stock.get('risk', ''):<12}")
            print("="*70)
            
            print("\n感谢使用超神系统！\n")


if __name__ == "__main__":
    controller = ChinaMarketController()
    controller.run() 