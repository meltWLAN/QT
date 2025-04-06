"""


量子共生网络 - 配置文件
这个文件包含系统的核心参数和配置
"""

import os
from pathlib import Path

# 系统路径
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

# 确保路径存在
for dir_path in [DATA_DIR, LOG_DIR, MODEL_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# 分形智能结构配置
FRACTAL_CONFIG = {
    "micro_agents": {
        "initial_count": 50,  # 初始微型智能体数量
        "max_count": 5000,    # 最大智能体数量
        "specialization_factor": 0.8,  # 专注度因子(0-1)
        "mutation_rate": 0.05,  # 智能体变异率
    },
    "mid_agents": {
        "initial_count": 5,   # 初始中层智能体
        "max_count": 50,      # 最大中层智能体
        "aggregation_threshold": 0.6,  # 聚合阈值
    },
    "meta_agents": {
        "initial_count": 1,   # 顶层决策智能体
        "max_count": 5,       # 最大决策智能体
    }
}

# 量子概率框架配置
QUANTUM_CONFIG = {
    "superposition_states": 5,  # 每个决策的叠加状态数
    "decoherence_rate": 0.1,    # 退相干率（决策具体化速率）
    "entanglement_depth": 3,    # 决策关联深度
}

# 自进化神经架构配置
EVOLUTION_CONFIG = {
    "self_modify_interval": 100,  # 自我修改间隔（轮次）
    "preservation_threshold": 0.7,  # 有价值结构保存阈值
    "novelty_bias": 0.3,          # 新颖性偏好度
}

# 市场接口配置
MARKET_CONFIG = {
    "data_sources": ["qlib", "custom"],  # 数据源
    "update_frequency": 60,  # 数据更新频率(秒)
    "initial_symbols": ["000001.SZ", "000002.SZ", "000063.SZ"],  # 初始股票池
}

# 系统运行配置
SYSTEM_CONFIG = {
    "seed_mode": True,  # 是否处于种子模式（初始训练阶段）
    "simulation_steps": 1000,  # 模拟步骤数
    "learning_rate": 0.01,  # 系统整体学习率
    "entropy_target": 0.7,  # 目标熵值（混沌与秩序平衡）
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "backup_count": 5,
    "max_size_mb": 10,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
} 