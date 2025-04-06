#!/usr/bin/env python3
"""
量子共生网络交易系统 - 主脚本
基于Tushare数据的量子共生网络交易系统
"""

import argparse
import logging
import os
import json
import pandas as pd
from datetime import datetime, timedelta

# 直接从模块导入
from quantum_symbiotic_network.network import QuantumSymbioticNetwork
from quantum_symbiotic_network.data_sources import TushareDataSource
from quantum_symbiotic_network.simulation import MarketSimulator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("QuantumTradingSystem")


def setup_arg_parser():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description="量子共生网络交易系统")
    
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'paper', 'live'],
                       help='运行模式：回测、模拟交易或实盘交易')
    
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
        data_source = TushareDataSource(token=token)
        logger.info("数据源初始化完成")
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
        market_data = data_source.get_market_data(
            start_date=start_date,
            end_date=end_date,
            sample_size=config.get('sample_size', 50)
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
    
    # 初始化量子共生网络
    quantum_network = QuantumSymbioticNetwork(config.get('network_config', {}))
    
    # 初始化网络
    stock_symbols = list(market_data.get('stocks', {}).keys())
    market_segments = [stock[:2] for stock in stock_symbols]  # 简单根据股票代码前两位分类
    unique_segments = list(set(market_segments))
    
    try:
        # 使用列表而不是命名参数传递
        quantum_network.initialize(unique_segments)
        logger.info("量子共生网络初始化完成")
    except Exception as e:
        logger.error(f"量子共生网络初始化失败: {e}")
        return
        
    # 运行回测
    logger.info(f"开始回测 - 从 {start_date} 到 {end_date}")
    
    total_days = simulator.days
    current_day = 0
    
    while True:
        # 获取当前市场数据
        current_data, done = simulator.step()
        current_day += 1
        
        if current_day % 10 == 0:
            logger.info(f"回测进度: {current_day}/{total_days} ({current_day/total_days*100:.1f}%)")
        
        # 获取交易决策
        decision = quantum_network.step(current_data)
        
        # 执行交易决策
        execution_result = simulator.execute_action(decision)
        
        # 提供反馈
        quantum_network.provide_feedback({
            "performance": execution_result.get("daily_return", 0),
            "trade_result": "success" if execution_result.get("success", False) else "failed"
        })
        
        if done:
            break
            
    # 计算并显示回测结果
    performance = simulator.calculate_performance()
    
    # 保存回测结果
    os.makedirs(args.output, exist_ok=True)
    
    # 保存性能报告
    report_path = os.path.join(args.output, "performance_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(performance, f, indent=2)
        
    # 绘制回测结果图表
    chart_path = os.path.join(args.output, "backtest_chart.png")
    simulator.plot_performance(performance, chart_path)
    
    # 显示关键指标
    logger.info("-" * 50)
    logger.info("回测结果")
    logger.info("-" * 50)
    logger.info(f"总收益率: {performance['total_return']*100:.2f}%")
    logger.info(f"年化收益率: {performance['annual_return']*100:.2f}%")
    logger.info(f"最大回撤: {performance['max_drawdown']*100:.2f}%")
    logger.info(f"夏普比率: {performance['sharpe']:.2f}")
    logger.info(f"胜率: {performance['win_rate']*100:.2f}%")
    logger.info(f"总交易次数: {performance['trade_count']}")
    logger.info("-" * 50)
    
    return performance


def run_paper_trading(args, config):
    """运行模拟交易模式"""
    logger.info("启动模拟交易模式")
    logger.warning("模拟交易模式尚未实现")
    # TODO: 实现模拟交易逻辑


def run_live_trading(args, config):
    """运行实盘交易模式"""
    logger.info("启动实盘交易模式")
    logger.warning("实盘交易模式尚未实现")
    # TODO: 实现实盘交易逻辑


def main():
    """主函数"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 根据模式运行不同的函数
    if args.mode == 'backtest':
        run_backtest(args, config)
    elif args.mode == 'paper':
        run_paper_trading(args, config)
    elif args.mode == 'live':
        run_live_trading(args, config)
    else:
        logger.error(f"不支持的模式: {args.mode}")


if __name__ == "__main__":
    main() 