"""
量子共生网络 - 革命性的交易系统
整合分形智能结构、量子概率交易框架和自进化神经架构，用于构建先进的交易系统

这个系统整合了：
1. 分形智能结构 - 微观智能体网络形成宏观决策能力
2. 量子概率交易框架 - 保持决策在叠加态直到观测
3. 自进化神经架构 - 能自我重构、适应市场的神经网络
4. 反熵知识引擎 - 不断挑战自身假设的知识系统

作者: Claude 3.7
日期: 2024-04-05
"""

import os
import sys
import logging
from pathlib import Path
from quantum_symbiotic_network.network import QuantumSymbioticNetwork

# 设置系统路径
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(BASE_DIR.parent))

# 设置版本
__version__ = "0.2.0"
__author__ = "Claude 3.7"

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumSymbioticNetwork")
logger.info(f"量子共生网络 v{__version__} 已初始化")
logger.info("初始化量子共生网络系统...")

# 导入核心组件
from quantum_symbiotic_network.core import (
    FractalIntelligenceNetwork, 
    QuantumProbabilityFramework,
    SelfEvolvingNetwork
)

# 配置日志
def setup_logging(level=logging.INFO, log_file=None):
    """
    设置日志系统
    
    Args:
        level: 日志级别
        log_file: 日志文件路径，如果为None则只输出到控制台
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=log_format)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        
    logger = logging.getLogger("QuantumSymbioticNetwork")
    logger.info(f"量子共生网络 v{__version__} 已初始化")
    
    return logger
    
# 系统核心类
class QuantumSymbioticNetwork:
    """量子共生网络 - 系统主类"""
    
    def __init__(self, config_path=None):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        from quantum_symbiotic_network.config import (
            FRACTAL_CONFIG, QUANTUM_CONFIG, EVOLUTION_CONFIG, 
            MARKET_CONFIG, SYSTEM_CONFIG, LOG_CONFIG
        )
        
        # 设置日志
        log_file = os.path.join(BASE_DIR, "logs", "system.log")
        self.logger = setup_logging(
            level=getattr(logging, LOG_CONFIG["level"]),
            log_file=log_file
        )
        
        self.logger.info("初始化量子共生网络系统...")
        
        # 配置
        self.fractal_config = FRACTAL_CONFIG
        self.quantum_config = QUANTUM_CONFIG
        self.evolution_config = EVOLUTION_CONFIG
        self.market_config = MARKET_CONFIG
        self.system_config = SYSTEM_CONFIG
        
        # 初始化核心组件
        self.fractal_network = FractalIntelligenceNetwork(FRACTAL_CONFIG)
        self.quantum_executor = QuantumProbabilityFramework(QUANTUM_CONFIG)
        
        self.logger.info("核心组件初始化完成")
        
    def initialize(self):
        """初始化系统，准备市场环境和初始智能体网络"""
        self.logger.info("准备创建初始智能体网络...")
        
        # 模拟市场分段
        market_segments = self.market_config["initial_symbols"]
        
        # 模拟特征
        features = {}
        common_features = ["open", "high", "low", "close", "volume", "ma5", "ma10", "ma20", "rsi", "macd"]
        
        for segment in market_segments:
            # 每个分段有一些通用特征和一些特殊特征
            segment_features = common_features.copy()
            # 添加一些特殊特征
            special_features = [f"{segment}_special_{i}" for i in range(3)]
            segment_features.extend(special_features)
            features[segment] = segment_features
            
        # 创建初始网络
        self.fractal_network.create_initial_network(market_segments, features)
        
        self.logger.info(f"初始网络创建完成，包含 {len(self.fractal_network.micro_agents)} 个微观智能体和 {len(self.fractal_network.mid_agents)} 个中层智能体")
        
    def step(self, market_data):
        """
        系统单步运行
        
        Args:
            market_data: 市场数据
            
        Returns:
            系统决策结果
        """
        self.logger.info("执行系统步骤...")
        
        # 通过分形网络获取决策
        decision = self.fractal_network.step(market_data)
        
        # 使用量子执行器准备交易
        trade_id = f"trade_{self.fractal_network.current_time}"
        
        if decision["action"] != "no_action":
            # 准备交易但不立即执行
            self.quantum_executor.prepare_trade(
                trade_id=trade_id,
                symbol=decision.get("symbol", "default"),
                signal=decision
            )
            
            # 更新量子状态
            self.quantum_executor.update()
            
            # 模拟市场因素的影响
            if "market_volatility" in market_data.get("global_market", {}):
                volatility = market_data["global_market"]["volatility"]
                self.quantum_executor.apply_market_factor("volatility", volatility)
                
        self.logger.info(f"步骤完成，当前决策: {decision.get('action', 'unknown')}")
        
        return {
            "decision": decision,
            "trade_id": trade_id if decision["action"] != "no_action" else None,
            "timestamp": self.fractal_network.current_time
        }
        
    def provide_feedback(self, feedback):
        """
        向系统提供反馈
        
        Args:
            feedback: 包含性能评估的反馈
        """
        self.logger.info("接收系统反馈...")
        
        # 传递给分形网络
        self.fractal_network.provide_feedback(feedback)
        
        self.logger.info("反馈处理完成")
        
    def execute_trade(self, trade_id):
        """
        执行交易（坍缩量子态）
        
        Args:
            trade_id: 交易ID
            
        Returns:
            执行结果
        """
        self.logger.info(f"执行交易 {trade_id}...")
        
        result = self.quantum_executor.execute_trade(trade_id)
        
        self.logger.info(f"交易执行完成: {result.get('action', 'unknown')}")
        
        return result 

__all__ = ['QuantumSymbioticNetwork'] 