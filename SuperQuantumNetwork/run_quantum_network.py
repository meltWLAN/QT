#!/usr/bin/env python
"""
量子共生网络 - 启动脚本
运行此脚本启动量子共生网络系统的演示
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(BASE_DIR))

# 导入演示模块
from quantum_symbiotic_network.demo import run_simulation

if __name__ == "__main__":
    print("="*80)
    print("       量子共生网络交易系统演示       ")
    print("="*80)
    print("这个系统整合了：")
    print("1. 分形智能结构 - 微观智能体网络形成宏观决策能力")
    print("2. 量子概率交易框架 - 保持决策在叠加态直到观测")
    print("3. 自进化神经架构 - 能自我重构、适应市场的神经网络")
    print("4. 反熵知识引擎 - 不断挑战自身假设的知识系统")
    print("="*80)
    print("正在启动模拟...")
    print()
    
    # 运行模拟
    performance = run_simulation()
    
    print()
    print("="*80)
    print("模拟完成！")
    print(f"总收益: {performance['total_return']:.2%}")
    print(f"夏普比率: {performance['sharpe']:.2f}")
    print(f"最大回撤: {performance['max_drawdown']:.2%}")
    print(f"最终资产: {performance['final_value']:.2f}")
    print("="*80)
    print(f"性能图表已保存到: {os.path.join(BASE_DIR, 'quantum_symbiotic_network', 'data', 'performance.png')}")
    print("="*80) 