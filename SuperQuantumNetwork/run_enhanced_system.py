#!/usr/bin/env python3
"""
增强版量子共生网络交易系统 - 运行脚本
集成多维度数据、深度学习预测和高级风险管理
"""

import os
import logging
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_quantum_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("EnhancedQuantumSystem")

# 导入自定义模块
from quantum_symbiotic_network.network import QuantumSymbioticNetwork
from quantum_symbiotic_network.data_sources.enhanced_data_source import EnhancedDataSource
from quantum_symbiotic_network.simulation import MarketSimulator
from quantum_symbiotic_network.models.deep_market_predictor import DeepMarketPredictor
from quantum_symbiotic_network.core.risk_management import RiskManager
from quantum_symbiotic_network.visualization import PerformanceVisualizer

def setup_arg_parser():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description="增强版量子共生网络交易系统")
    
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'paper', 'live', 'train'],
                       help='运行模式：回测、模拟交易、实盘交易或训练模型')
    
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    
    parser.add_argument('--start-date', type=str, default='',
                       help='回测开始日期 (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default='',
                       help='回测结束日期 (YYYY-MM-DD)')
    
    parser.add_argument('--capital', type=float, default=1000000.0,
                       help='初始资金')
    
    parser.add_argument('--stocks', type=str, default='',
                       help='股票代码列表，用逗号分隔')
    
    parser.add_argument('--output', type=str, default='results',
                       help='结果输出目录')
    
    parser.add_argument('--token', type=str, default='',
                       help='Tushare API令牌')
    
    parser.add_argument('--no-visualization', action='store_true',
                       help='禁用可视化')
    
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    return parser

def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return {}
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

def train_models(args, config):
    """训练模型模式"""
    logger.info("启动模型训练模式")
    
    # 解析日期
    today = datetime.now()
    
    if args.start_date:
        start_date = args.start_date.replace('-', '')
    else:
        # 默认使用一年前的日期
        start_date = (today - timedelta(days=365)).strftime('%Y%m%d')
        
    if args.end_date:
        end_date = args.end_date.replace('-', '')
    else:
        # 默认使用今天的日期
        end_date = today.strftime('%Y%m%d')
    
    # 获取Token
    token = args.token or config.get('tushare_token', '')
    
    # 初始化数据源
    try:
        data_config = config.get('data_config', {})
        data_source = EnhancedDataSource(token=token, config=data_config)
        logger.info("增强数据源初始化完成")
    except Exception as e:
        logger.error(f"数据源初始化失败: {e}")
        return
    
    # 获取股票列表
    if args.stocks:
        stock_list = args.stocks.split(',')
    else:
        try:
            # 获取股票列表
            stock_list = data_source.get_stock_list()
            # 随机选择一部分股票用于训练
            from random import sample
            sample_size = min(config.get('sample_size', 50), len(stock_list))
            stock_list = sample(stock_list, sample_size)
            logger.info(f"随机选择了 {sample_size} 只股票用于训练")
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return
    
    # 获取市场数据
    try:
        logger.info("获取增强市场数据...")
        market_data = data_source.get_enhanced_market_data(
            start_date=start_date,
            end_date=end_date,
            sample_size=config.get('sample_size', 50),
            include_macro=data_config.get('include_macro', True)
        )
        logger.info(f"获取到市场数据，包含 {len(market_data.get('stocks', {}))} 只股票")
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        return
    
    # 初始化深度学习预测模型
    try:
        model_config = config.get('model_config', {}).get('deep_predictor', {})
        predictor = DeepMarketPredictor(config=model_config)
        logger.info("深度学习预测模型初始化完成")
    except Exception as e:
        logger.error(f"预测模型初始化失败: {e}")
        return
    
    # 训练每只股票的模型
    success_count = 0
    for symbol, stock_data in market_data.get('stocks', {}).items():
        try:
            logger.info(f"训练股票 {symbol} 的预测模型...")
            
            # 预处理数据，添加高级技术指标
            if data_config.get('technical_indicators', {}).get('use_advanced_indicators', True):
                stock_data = data_source.calculate_advanced_indicators(stock_data)
            
            # 训练模型
            history = predictor.train(stock_data, market_data.get('macro', None))
            
            if history:
                logger.info(f"股票 {symbol} 模型训练完成")
                success_count += 1
            else:
                logger.warning(f"股票 {symbol} 模型训练失败")
        except Exception as e:
            logger.error(f"训练股票 {symbol} 模型时出错: {e}")
    
    logger.info(f"模型训练完成，成功训练 {success_count} 个模型")

def run_backtest(args, config):
    """运行回测模式"""
    logger.info("启动回测模式")
    
    # 解析日期
    today = datetime.now()
    
    if args.start_date:
        start_date = args.start_date.replace('-', '')
    else:
        # 默认使用一年前的日期
        start_date = (today - timedelta(days=365)).strftime('%Y%m%d')
        
    if args.end_date:
        end_date = args.end_date.replace('-', '')
    else:
        # 默认使用今天的日期
        end_date = today.strftime('%Y%m%d')
    
    # 获取Token
    token = args.token or config.get('tushare_token', '')
    
    # 初始化数据源
    try:
        data_config = config.get('data_config', {})
        data_source = EnhancedDataSource(token=token, config=data_config)
        logger.info("增强数据源初始化完成")
    except Exception as e:
        logger.error(f"数据源初始化失败: {e}")
        return
    
    # 获取股票列表
    if args.stocks:
        stock_list = args.stocks.split(',')
    else:
        try:
            # 获取股票列表
            stock_list = data_source.get_stock_list()
            # 随机选择一部分股票用于回测
            from random import sample
            sample_size = min(config.get('sample_size', 50), len(stock_list))
            stock_list = sample(stock_list, sample_size)
            logger.info(f"随机选择了 {sample_size} 只股票用于回测")
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return
    
    # 获取市场数据
    try:
        logger.info("获取增强市场数据...")
        market_data = data_source.get_enhanced_market_data(
            start_date=start_date,
            end_date=end_date,
            sample_size=config.get('sample_size', 50),
            include_macro=data_config.get('include_macro', True)
        )
        logger.info(f"获取到市场数据，包含 {len(market_data.get('stocks', {}))} 只股票")
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        return
    
    # 初始化模拟器
    simulator_config = {
        "initial_capital": args.capital,
        "transaction_fee_rate": config.get('transaction_fee_rate', 0.0003),
        "slippage_rate": config.get('slippage', 0.0001),
        "risk_free_rate": config.get('risk_free_rate', 0.03) / 252  # 转换为日利率
    }
    
    simulator = MarketSimulator(config=simulator_config)
    
    # 加载市场数据
    simulator.load_real_data(market_data)
    logger.info("市场模拟器初始化完成")
    
    # 初始化深度学习预测模型
    try:
        model_config = config.get('model_config', {}).get('deep_predictor', {})
        predictor = DeepMarketPredictor(config=model_config)
        logger.info("深度学习预测模型初始化完成")
    except Exception as e:
        logger.error(f"预测模型初始化失败: {e}")
        return
    
    # 初始化风险管理器
    try:
        risk_config = config.get('risk_config', {})
        risk_manager = RiskManager(config=risk_config)
        logger.info("风险管理器初始化完成")
    except Exception as e:
        logger.error(f"风险管理器初始化失败: {e}")
        return
    
    # 初始化量子共生网络
    quantum_network = QuantumSymbioticNetwork(config.get('network_config', {}))
    
    # 初始化网络
    stock_symbols = list(market_data.get('stocks', {}).keys())
    market_segments = [stock[:2] for stock in stock_symbols]  # 简单根据股票代码前两位分类
    unique_segments = list(set(market_segments))
    
    try:
        quantum_network.initialize(unique_segments)
        logger.info("量子共生网络初始化完成")
    except Exception as e:
        logger.error(f"量子共生网络初始化失败: {e}")
        return
    
    # 运行回测
    logger.info(f"开始回测 - 从 {start_date} 到 {end_date}")
    
    total_days = simulator.days
    current_day = 0
    
    # 回测性能记录
    performance_history = []
    decision_history = []
    positions_history = []
    portfolio_value_history = []
    
    # 定义再平衡周期
    rebalance_interval = config.get('backtest_config', {}).get('rebalance_interval', 5)
    
    while True:
        try:
            # 获取当前市场数据
            current_data, done = simulator.step()
            current_day += 1
            
            if current_day % 10 == 0:
                logger.info(f"回测进度: {current_day}/{total_days} ({current_day/total_days*100:.1f}%)")
            
            # 记录当前投资组合价值
            portfolio_value = simulator.portfolio_value
            portfolio_value_history.append({
                'date': current_data.get('date', ''),
                'portfolio_value': portfolio_value
            })
            
            # 检查是否是再平衡日
            is_rebalance_day = current_day % rebalance_interval == 0
            
            # 在再平衡日使用深度学习模型进行预测
            predictions = {}
            if is_rebalance_day:
                for symbol, data in current_data.get('stocks', {}).items():
                    try:
                        # 预处理数据，添加高级技术指标
                        if data_config.get('technical_indicators', {}).get('use_advanced_indicators', True):
                            data = data_source.calculate_advanced_indicators(data)
                        
                        # 进行预测
                        prediction = predictor.predict(data, current_data.get('macro', None))
                        
                        if prediction:
                            # 提取预测结果
                            price_pred = prediction.get('price_change_percent', [0])[0]
                            volatility_pred = prediction.get('volatility_prediction', [0])[0]
                            trend_pred = prediction.get('trend_prediction', [0])[0]
                            confidence = prediction.get('confidence', [0.5])[0]
                            
                            predictions[symbol] = {
                                'expected_return': price_pred,
                                'volatility': volatility_pred,
                                'trend': trend_pred,
                                'confidence': confidence
                            }
                    except Exception as e:
                        logger.error(f"预测股票 {symbol} 时出错: {e}")
            
            # 获取当前持仓和历史数据
            current_positions = simulator.positions
            historical_data = simulator.historical_data
            
            # 分析风险
            risk_analysis = risk_manager.check_risk_limits(
                portfolio_value, 
                current_positions,
                historical_data
            )
            
            # 记录持仓历史
            positions_history.append({
                'date': current_data.get('date', ''),
                'positions': current_positions,
                'risk_analysis': risk_analysis
            })
            
            # 在再平衡日生成交易决策
            decision = {}
            if is_rebalance_day:
                # 优化仓位配置
                optimized_positions = risk_manager.optimize_position_sizes(
                    portfolio_value,
                    current_data.get('stocks', {}),
                    predictions,
                    max_positions=risk_config.get('max_positions', 10)
                )
                
                # 生成交易决策
                for symbol, position in optimized_positions.items():
                    current_shares = current_positions.get(symbol, {}).get('shares', 0)
                    target_shares = position.get('shares', 0)
                    
                    # 计算需要交易的股数
                    trade_shares = target_shares - current_shares
                    
                    if trade_shares != 0:
                        decision[symbol] = {
                            'action': 'buy' if trade_shares > 0 else 'sell',
                            'shares': abs(trade_shares),
                            'reason': 'portfolio_optimization',
                            'confidence': position.get('confidence', 0.5)
                        }
            
            # 使用量子共生网络进行辅助决策
            qsn_decision = quantum_network.step(current_data)
            
            # 合并决策（优先使用优化后的决策）
            for symbol, qsn_action in qsn_decision.items():
                if symbol not in decision:
                    # 量子网络决策只有在风险管理未覆盖该股票时才使用
                    decision[symbol] = qsn_action
            
            # 记录决策历史
            decision_history.append({
                'date': current_data.get('date', ''),
                'decision': decision
            })
            
            # 执行交易决策
            execution_result = simulator.execute_action(decision)
            
            # 提供反馈
            quantum_network.provide_feedback({
                "performance": execution_result.get("daily_return", 0),
                "trade_result": "success" if execution_result.get("success", False) else "failed"
            })
            
            # 记录性能
            performance_history.append({
                'date': current_data.get('date', ''),
                'daily_return': execution_result.get('daily_return', 0),
                'portfolio_value': simulator.portfolio_value,
                'success': execution_result.get('success', False)
            })
            
            if done:
                break
        except Exception as e:
            logger.error(f"回测过程中出错: {e}")
            logger.error(traceback.format_exc())
            break
    
    # 计算并显示回测结果
    performance = simulator.calculate_performance()
    logger.info(f"回测完成 - 最终资金: {simulator.portfolio_value:.2f}, 收益率: {(simulator.portfolio_value / args.capital - 1) * 100:.2f}%")
    logger.info(f"年化收益率: {performance.get('annual_return', 0) * 100:.2f}%, 夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
    logger.info(f"最大回撤: {performance.get('max_drawdown', 0) * 100:.2f}%")
    
    # 保存回测结果
    results_dir = args.output
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 保存性能记录
    performance_df = pd.DataFrame(performance_history)
    performance_df.to_csv(os.path.join(results_dir, 'performance_history.csv'), index=False)
    
    # 保存决策历史
    with open(os.path.join(results_dir, 'decision_history.json'), 'w', encoding='utf-8') as f:
        json.dump(decision_history, f, indent=4)
    
    # 保存持仓历史
    with open(os.path.join(results_dir, 'positions_history.json'), 'w', encoding='utf-8') as f:
        json.dump(positions_history, f, indent=4)
    
    # 保存投资组合价值历史
    portfolio_value_df = pd.DataFrame(portfolio_value_history)
    portfolio_value_df.to_csv(os.path.join(results_dir, 'portfolio_value_history.csv'), index=False)
    
    # 保存总体性能指标
    with open(os.path.join(results_dir, 'performance_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(performance, f, indent=4)
    
    # 可视化结果
    if not args.no_visualization:
        try:
            visualizer = PerformanceVisualizer()
            visualizer.plot_portfolio_performance(portfolio_value_df, results_dir)
            visualizer.plot_return_distribution(performance_df, results_dir)
            visualizer.plot_drawdown(performance, results_dir)
            logger.info(f"可视化结果已保存到 {results_dir}")
        except Exception as e:
            logger.error(f"可视化结果时出错: {e}")

def run_paper_trading(args, config):
    """运行模拟交易模式"""
    logger.info("启动模拟交易模式")
    logger.warning("模拟交易模式尚未实现，请选择其他模式")

def run_live_trading(args, config):
    """运行实盘交易模式"""
    logger.info("启动实盘交易模式")
    logger.warning("实盘交易模式尚未实现，请选择其他模式")

def main():
    """主函数"""
    # 解析命令行参数
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 加载配置
    config = load_config(args.config)
    
    # 根据模式运行不同的功能
    if args.mode == 'train':
        train_models(args, config)
    elif args.mode == 'backtest':
        run_backtest(args, config)
    elif args.mode == 'paper':
        run_paper_trading(args, config)
    elif args.mode == 'live':
        run_live_trading(args, config)

if __name__ == "__main__":
    main() 