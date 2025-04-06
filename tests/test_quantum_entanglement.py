#!/usr/bin/env python
"""
超神系统能力验证脚本 - 简化版
测试量子纠缠引擎的基本功能
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 直接导入测试所需类
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SuperQuantumNetwork.quantum_symbiotic_network.core.quantum_entanglement_engine import (
    QuantumEntanglementEngine, EntanglementProperty
)

def print_header(title):
    """打印测试标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_quantum_entanglement_engine():
    """测试量子纠缠引擎基本功能"""
    print_header("测试: 量子纠缠引擎基本功能")
    
    # 初始化引擎 - 添加空配置参数
    engine = QuantumEntanglementEngine(dimensions=8, depth=5, config={})
    print(f"引擎初始化: 维度={engine.dimensions}, 深度={engine.depth}")
    
    # 初始化资产列表
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BABA"]
    print(f"测试资产: {', '.join(assets)}")
    
    # 手动创建相关性矩阵（模拟实际市场数据）
    correlation_matrix = {}
    for i, asset_i in enumerate(assets):
        for j, asset_j in enumerate(assets):
            if i < j:  # 只处理上三角矩阵
                # 模拟相关行业有更高相关性
                if (asset_i in ["AAPL", "MSFT", "GOOGL"] and asset_j in ["AAPL", "MSFT", "GOOGL"]) or \
                   (asset_i in ["AMZN", "BABA"] and asset_j in ["AMZN", "BABA"]) or \
                   (asset_i in ["TSLA", "NVDA"] and asset_j in ["TSLA", "NVDA"]):
                    correlation = 0.7 + np.random.random() * 0.2  # 0.7-0.9
                else:
                    correlation = 0.2 + np.random.random() * 0.3  # 0.2-0.5
                
                correlation_matrix[(asset_i, asset_j)] = correlation
                correlation_matrix[(asset_j, asset_i)] = correlation
    
    # 初始化纠缠关系
    engine.initialize_entanglement(assets, correlation_matrix)
    print(f"纠缠关系初始化完成，共{len(correlation_matrix)//2}对相互关系")
    
    # 检查纠缠群组
    print(f"检测到{len(engine.entanglement_clusters)}个纠缠群组:")
    for i, cluster in enumerate(engine.entanglement_clusters):
        print(f"  群组{i+1}: {', '.join(cluster)}")
    
    # 模拟市场数据
    market_data = {}
    for asset in assets:
        price_change = np.random.normal(0, 0.02)  # 标准正态分布，均值0，标准差2%
        volume_relative = 0.8 + np.random.random() * 0.4  # 0.8-1.2
        momentum = price_change * 3 + np.random.normal(0, 0.01)  # 带一些噪声的动量
        
        market_data[asset] = {
            "price_change_pct": price_change,
            "volume_relative": volume_relative,
            "momentum": momentum
        }
    
    # 应用量子操作
    engine.apply_quantum_operations(market_data)
    
    # 计算市场共振
    resonance_state = engine.compute_market_resonance(market_data)
    
    print("\n市场共振状态:")
    for asset, resonance in resonance_state.items():
        print(f"  {asset}: 共振度={resonance:.4f}")
    
    # 市场预测
    predictions = engine.predict_market_movement(assets)
    
    print("\n市场预测结果:")
    for asset, prediction in predictions.items():
        direction = prediction["direction"]
        strength = prediction["strength"]
        up_prob = prediction["up_probability"]
        down_prob = prediction["down_probability"]
        
        signal = "🔴 卖出" if direction < -0.2 else "🟢 买入" if direction > 0.2 else "⚪ 持有"
        
        print(f"  {asset}: {signal} 方向={direction:.4f} 强度={strength:.4f} (上涨概率:{up_prob:.2f} 下跌概率:{down_prob:.2f})")
    
    # 获取纠缠网络状态（用于可视化）
    network_state = engine.get_entanglement_network()
    print(f"\n获取到纠缠网络状态: {len(network_state['nodes'])}个节点, {len(network_state['edges'])}条边, {len(network_state['clusters'])}个群组")
    
    return engine, assets, market_data, predictions

def test_market_anomaly_detection(engine, assets):
    """测试市场异常检测功能"""
    print_header("测试: 市场异常检测功能")
    
    # 创建包含异常的市场数据
    market_data = {}
    for asset in assets:
        is_anomaly = False
        
        # 为TSLA和NVDA创建异常数据
        if asset in ["TSLA", "NVDA"]:
            price_change = 0.08 if asset == "TSLA" else -0.11  # 异常大的价格变化
            volume_relative = 3.2 if asset == "TSLA" else 2.8  # 异常大的成交量
            momentum = price_change * 2
            is_anomaly = True
        else:
            price_change = np.random.normal(0, 0.015)  # 标准正态分布，均值0，标准差1.5%
            volume_relative = 0.9 + np.random.random() * 0.2  # 0.9-1.1
            momentum = price_change * 3 + np.random.normal(0, 0.01)  # 带一些噪声的动量
        
        market_data[asset] = {
            "price_change_pct": price_change,
            "volume_relative": volume_relative,
            "momentum": momentum,
            "is_anomaly": is_anomaly  # 标记异常（仅用于测试验证）
        }
    
    # 应用量子操作
    engine.apply_quantum_operations(market_data)
    
    # 计算市场共振
    resonance_state = engine.compute_market_resonance(market_data)
    
    # 手动计算统计量以检测异常
    resonance_values = list(resonance_state.values())
    mean_resonance = np.mean(resonance_values)
    std_resonance = np.std(resonance_values)
    
    print("\n市场共振状态统计: 均值={:.4f}, 标准差={:.4f}".format(mean_resonance, std_resonance))
    print("\n共振状态与异常检测:")
    
    for asset, resonance in resonance_state.items():
        z_score = (resonance - mean_resonance) / std_resonance if std_resonance > 0 else 0
        is_anomaly = abs(z_score) > 2.0
        actual_anomaly = market_data[asset].get("is_anomaly", False)
        
        # 评估检测结果
        status = "正确检测到异常 ✓" if is_anomaly and actual_anomaly else \
                 "正确识别为正常 ✓" if not is_anomaly and not actual_anomaly else \
                 "误报 ✗" if is_anomaly and not actual_anomaly else "漏报 ✗"
        
        print(f"  {asset}: 共振度={resonance:.4f}, Z分数={z_score:.2f}, 是否异常: {is_anomaly}, {status}")

if __name__ == "__main__":
    # 运行测试
    engine, assets, market_data, predictions = test_quantum_entanglement_engine()
    test_market_anomaly_detection(engine, assets)
    
    print("\n" + "="*80)
    print("  超神系统能力验证完成")
    print("="*80)
