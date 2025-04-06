#!/usr/bin/env python3
"""
Tushare数据源测试脚本 - 使用真实市场数据测试量子共生网络
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TushareTest")

# 导入自定义模块
from quantum_symbiotic_network.data_sources import TushareDataSource
from quantum_symbiotic_network.simulation import MarketSimulator
from quantum_symbiotic_network.strategies import create_default_strategy_ensemble
from quantum_symbiotic_network import QuantumSymbioticNetwork

# Tushare Token
TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"

def test_tushare_data_integration():
    """测试Tushare数据源集成"""
    logger.info("开始测试Tushare数据源集成")
    
    # 创建Tushare数据源
    data_source = TushareDataSource(TUSHARE_TOKEN)
    
    # 获取示例数据
    try:
        # 获取股票列表
        stocks = data_source.get_stock_list()
        logger.info(f"获取到 {len(stocks)} 只股票")
        
        # 获取10只样本股票的数据
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')
        market_data = data_source.get_market_data(
            start_date=start_date,
            end_date=end_date,
            sample_size=10
        )
        
        logger.info(f"获取到 {len(market_data['stocks'])} 只股票的数据")
        for symbol, df in market_data['stocks'].items():
            logger.info(f"股票 {symbol}: {len(df)} 条记录")
            
        # 获取指数数据
        for idx, df in market_data.get('indices', {}).items():
            logger.info(f"指数 {idx}: {len(df)} 条记录")
            
        return market_data
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return None
        
def test_strategy_signals(market_data):
    """测试策略信号生成"""
    logger.info("开始测试策略信号生成")
    
    if not market_data or not market_data.get('stocks'):
        logger.error("没有可用的市场数据进行测试")
        return
        
    # 创建策略集
    strategy_ensemble = create_default_strategy_ensemble()
    
    # 测试一个样本股票的信号
    sample_symbol = list(market_data['stocks'].keys())[0]
    sample_data = market_data['stocks'][sample_symbol]
    
    # 生成策略信号
    signals = []
    buy_count = 0
    sell_count = 0
    hold_count = 0
    
    # 按日期排序
    sample_data = sample_data.sort_values('trade_date')
    
    for i in range(20, len(sample_data)):
        # 获取当天和历史数据
        current_row = sample_data.iloc[i]
        history = sample_data.iloc[i-20:i]
        
        # 准备数据
        data = {
            "symbol": sample_symbol,
            "date": str(current_row["trade_date"]),
            "open": float(current_row["open"]),
            "high": float(current_row["high"]),
            "low": float(current_row["low"]),
            "close": float(current_row["close"]),
            "volume": float(current_row["vol"]),
            "history": []
        }
        
        # 添加技术指标
        if "ma5" in current_row:
            data["ma5"] = float(current_row["ma5"]) if not np.isnan(current_row["ma5"]) else None
        if "ma10" in current_row:
            data["ma10"] = float(current_row["ma10"]) if not np.isnan(current_row["ma10"]) else None
        if "ma20" in current_row:
            data["ma20"] = float(current_row["ma20"]) if not np.isnan(current_row["ma20"]) else None
        if "rsi14" in current_row:
            data["rsi"] = float(current_row["rsi14"]) if not np.isnan(current_row["rsi14"]) else None
        if "macd" in current_row:
            data["macd"] = float(current_row["macd"]) if not np.isnan(current_row["macd"]) else None
        if "signal" in current_row:
            data["signal"] = float(current_row["signal"]) if not np.isnan(current_row["signal"]) else None
        
        # 添加历史数据
        for j in range(len(history)):
            hist_row = history.iloc[j]
            history_data = {
                "date": str(hist_row["trade_date"]),
                "open": float(hist_row["open"]),
                "high": float(hist_row["high"]),
                "low": float(hist_row["low"]),
                "close": float(hist_row["close"]),
                "volume": float(hist_row["vol"])
            }
            data["history"].append(history_data)
            
        # 生成信号
        signal = strategy_ensemble.generate_signal(data)
        signals.append({
            "date": current_row["trade_date"],
            "action": signal["action"],
            "confidence": signal["confidence"],
            "close": data["close"]
        })
        
        # 统计信号类型
        if signal["action"] == "buy":
            buy_count += 1
        elif signal["action"] == "sell":
            sell_count += 1
        else:
            hold_count += 1
            
    logger.info(f"生成 {len(signals)} 个交易信号")
    logger.info(f"买入: {buy_count}, 卖出: {sell_count}, 持仓: {hold_count}")
    
    # 可视化信号
    dates = [s["date"] for s in signals]
    closes = [s["close"] for s in signals]
    actions = [s["action"] for s in signals]
    confidences = [s["confidence"] for s in signals]
    
    # 绘制价格和信号
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, closes, label="价格")
    
    # 标记买入信号
    buy_dates = [dates[i] for i in range(len(dates)) if actions[i] == "buy"]
    buy_prices = [closes[i] for i in range(len(dates)) if actions[i] == "buy"]
    plt.scatter(buy_dates, buy_prices, color="green", marker="^", s=100, label="买入")
    
    # 标记卖出信号
    sell_dates = [dates[i] for i in range(len(dates)) if actions[i] == "sell"]
    sell_prices = [closes[i] for i in range(len(dates)) if actions[i] == "sell"]
    plt.scatter(sell_dates, sell_prices, color="red", marker="v", s=100, label="卖出")
    
    plt.title(f"股票 {sample_symbol} 价格和交易信号")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # 绘制信号置信度
    plt.subplot(2, 1, 2)
    plt.bar(dates, confidences, color="blue", alpha=0.6)
    plt.title("信号置信度")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig("strategy_signals.png")
    logger.info("已保存策略信号图表到 strategy_signals.png")
    
    return signals
    
def test_market_simulator(market_data):
    """测试市场模拟器"""
    logger.info("开始测试市场模拟器")
    
    if not market_data or not market_data.get('stocks'):
        logger.error("没有可用的市场数据进行测试")
        return
        
    # 创建市场模拟器
    simulator = MarketSimulator({
        "initial_capital": 100000.0
    })
    
    # 加载真实市场数据
    simulator.load_real_data(market_data)
    
    # 创建策略集
    strategy_ensemble = create_default_strategy_ensemble()
    
    # 运行回测
    logger.info(f"开始回测 {len(simulator.symbols)} 只股票，周期 {simulator.days} 天")
    
    actions = []
    day = 0
    is_done = False
    
    while not is_done:
        # 获取当天数据
        data, is_done = simulator.step()
        
        # 对每个股票生成信号并执行
        for symbol, symbol_data in data.items():
            signal = strategy_ensemble.generate_signal(symbol_data)
            
            if signal["action"] != "hold" and signal["confidence"] > 0.3:
                size = signal["confidence"] * 2  # 根据置信度调整仓位
                
                action = {
                    "action": signal["action"],
                    "symbol": symbol,
                    "size": size
                }
                
                # 执行交易
                result = simulator.execute_action(action)
                if result["status"] == "success":
                    actions.append(action)
                    logger.info(f"Day {day}: {result['message']}")
                    
        day += 1
        
    # 计算性能
    performance = simulator.calculate_performance()
    
    logger.info("\n====== 回测性能报告 ======")
    logger.info(f"初始资金: {performance['initial_capital']:.2f}")
    logger.info(f"最终价值: {performance['final_value']:.2f}")
    logger.info(f"总收益率: {performance['total_return']*100:.2f}%")
    logger.info(f"夏普比率: {performance['sharpe']:.2f}")
    logger.info(f"最大回撤: {performance['max_drawdown']*100:.2f}%")
    logger.info(f"交易次数: {len(performance['trades'])}")
    
    # 保存性能报告
    simulator.save_performance_report(performance)
    
    # 绘制性能图表
    simulator.plot_performance(performance)
    
    return performance
    
def test_quantum_network(market_data):
    """测试量子共生网络"""
    logger.info("开始测试量子共生网络")
    
    if not market_data or not market_data.get('stocks'):
        logger.error("没有可用的市场数据进行测试")
        return
        
    # 创建市场模拟器
    simulator = MarketSimulator({
        "initial_capital": 100000.0
    })
    
    # 加载真实市场数据
    simulator.load_real_data(market_data)
    
    # 创建量子共生网络
    config = {
        "use_strategy_ensemble": True,
        "fractal_network": {
            "use_strategy_ensemble": True,
            "micro_agents_per_segment": 8,
            "self_modify_interval": 30
        },
        "quantum_trading": {
            "collapse_threshold": 0.3,  # 更低的阈值产生更多交易
            "uncertainty_decay": 0.8
        },
        "neural_evolution": {
            "learning_rate": 0.01,
            "batch_size": 16
        }
    }
    
    network = QuantumSymbioticNetwork(config)
    
    # 准备市场分段和特征
    market_segments = list(market_data['stocks'].keys())
    features = {}
    
    for symbol in market_segments:
        features[symbol] = ["price", "volume", "ma5", "ma10", "ma20", "rsi", "macd"]
    
    # 初始化网络
    network.initialize(market_segments, features)
    
    # 运行回测
    logger.info(f"开始量子共生网络回测，周期 {simulator.days} 天")
    
    actions = []
    day = 0
    is_done = False
    
    while not is_done:
        # 获取当天数据
        market_data, is_done = simulator.step()
        
        # 使用量子共生网络生成决策
        decision = network.step(market_data)
        
        # 根据决策执行交易
        action = {
            "action": decision["action"],
            "symbol": decision.get("symbol", simulator.symbols[0]),
            "size": decision.get("confidence", 0.5) * 2  # 根据置信度调整仓位
        }
        
        if action["action"] != "hold":
            # 执行交易
            result = simulator.execute_action(action)
            if result["status"] == "success":
                actions.append(action)
                logger.info(f"Day {day}: {result['message']}")
        
        # 提供反馈
        feedback = {
            "performance": 0.0,  # 简化的反馈
            "metrics": {}
        }
        network.provide_feedback(feedback)
        
        day += 1
        
    # 计算性能
    performance = simulator.calculate_performance()
    
    logger.info("\n====== 量子共生网络回测性能报告 ======")
    logger.info(f"初始资金: {performance['initial_capital']:.2f}")
    logger.info(f"最终价值: {performance['final_value']:.2f}")
    logger.info(f"总收益率: {performance['total_return']*100:.2f}%")
    logger.info(f"夏普比率: {performance['sharpe']:.2f}")
    logger.info(f"最大回撤: {performance['max_drawdown']*100:.2f}%")
    logger.info(f"交易次数: {len(performance['trades'])}")
    
    # 保存性能报告
    report_path = os.path.join("quantum_symbiotic_network", "data", "quantum_performance_report.txt")
    simulator.save_performance_report(performance, report_path)
    
    # 绘制性能图表
    simulator.plot_performance(performance)
    plt.savefig(os.path.join("quantum_symbiotic_network", "data", "quantum_performance.png"))
    
    return performance

def run_complete_test():
    """运行完整测试流程"""
    logger.info("====== 开始量子共生网络完整测试 ======")
    
    # 1. 测试Tushare数据源
    market_data = test_tushare_data_integration()
    if not market_data:
        logger.error("Tushare数据源测试失败，无法继续后续测试")
        return
        
    # 2. 测试策略信号
    signals = test_strategy_signals(market_data)
    
    # 3. 测试市场模拟器
    simulator_performance = test_market_simulator(market_data)
    
    # 4. 测试量子共生网络
    quantum_performance = test_quantum_network(market_data)
    
    # 比较结果
    if simulator_performance and quantum_performance:
        logger.info("\n====== 性能比较 ======")
        logger.info(f"策略集回测收益率: {simulator_performance['total_return']*100:.2f}%")
        logger.info(f"量子共生网络收益率: {quantum_performance['total_return']*100:.2f}%")
        logger.info(f"策略集夏普比率: {simulator_performance['sharpe']:.2f}")
        logger.info(f"量子共生网络夏普比率: {quantum_performance['sharpe']:.2f}")
        
        # 绘制比较图表
        plt.figure(figsize=(12, 6))
        plt.plot(simulator_performance["value_history"], label="策略集回测")
        plt.plot(quantum_performance["value_history"], label="量子共生网络")
        plt.title("策略集与量子共生网络性能比较")
        plt.xlabel("交易日")
        plt.ylabel("资产价值")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("quantum_symbiotic_network", "data", "performance_comparison.png"))
        logger.info("性能比较图表已保存")
        
    logger.info("====== 测试完成 ======")

if __name__ == "__main__":
    run_complete_test() 