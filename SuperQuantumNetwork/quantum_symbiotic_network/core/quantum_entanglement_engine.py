"""
量子纠缠引擎 - 高级量子纠缠计算和市场共振分析
实现多维度量子纠缠状态处理，提高市场预测准确度
"""

import numpy as np
import scipy.linalg as la
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
from datetime import datetime
import uuid
import math
from dataclasses import dataclass, field

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class EntanglementProperty:
    """量子纠缠性质"""
    # 纠缠强度（0-1）
    strength: float = 0.0
    # 纠缠相位
    phase: float = 0.0
    # 共振频率
    resonance_frequency: float = 0.0
    # 相关性历史
    correlation_history: List[float] = field(default_factory=list)
    # 最后更新时间
    last_updated: datetime = field(default_factory=datetime.now)
    # 量子效率
    efficiency: float = 0.0
    # 纠缠半衰期（小时）
    half_life: float = 24.0
    
    def update_strength(self, new_correlation: float, alpha: float = 0.3) -> None:
        """更新纠缠强度，使用指数移动平均"""
        if self.strength == 0.0:
            self.strength = new_correlation
        else:
            self.strength = alpha * new_correlation + (1 - alpha) * self.strength
        
        # 记录历史
        self.correlation_history.append(new_correlation)
        if len(self.correlation_history) > 100:  # 限制历史记录长度
            self.correlation_history.pop(0)
            
        # 更新时间
        self.last_updated = datetime.now()
    
    def calculate_efficiency(self) -> float:
        """计算纠缠效率 - 基于历史稳定性"""
        if len(self.correlation_history) < 3:
            return 0.5
        
        # 计算相关性的稳定性（标准差的倒数）
        std = np.std(self.correlation_history[-10:]) if len(self.correlation_history) >= 10 else np.std(self.correlation_history)
        stability = 1.0 / (1.0 + std * 10)  # 标准化到0-1
        
        # 计算纠缠效率
        self.efficiency = self.strength * stability
        return self.efficiency
    
    def decay(self, hours_passed: float) -> None:
        """纠缠强度随时间衰减"""
        if hours_passed <= 0:
            return
            
        # 计算衰减因子
        decay_factor = math.pow(0.5, hours_passed / self.half_life)
        self.strength *= decay_factor


class QuantumEntanglementEngine:
    """高级量子纠缠引擎，提高市场预测准确度"""
    
    def __init__(self, dimensions=12, depth=7, config: Dict[str, Any] = None):
        """
        初始化量子纠缠引擎
        
        Args:
            dimensions: 量子空间维度，捕捉更多市场特征
            depth: 量子电路深度，提高预测的时间跨度
            config: 配置参数
        """
        self.dimensions = dimensions
        self.depth = depth
        self.config = config or {}
        
        # 纠缠矩阵 - 存储资产间的纠缠关系
        self.entanglement_matrix: Dict[Tuple[str, str], EntanglementProperty] = {}
        
        # 资产的量子状态
        self.quantum_states: Dict[str, np.ndarray] = {}
        
        # 市场共振状态
        self.resonance_state: Dict[str, float] = {}
        
        # 纠缠群组 - 存储高度纠缠的资产群组
        self.entanglement_clusters: List[Set[str]] = []
        
        # 相位参考
        self.phase_reference = np.random.random() * 2 * np.pi
        
        # 量子共振频率
        self.base_resonance_frequency = config.get("base_resonance_frequency", 0.1)
        
        logger.info(f"量子纠缠引擎初始化完成，维度={dimensions}，深度={depth}")
    
    def initialize_entanglement(self, assets: List[str], 
                              correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> None:
        """
        初始化量子纠缠矩阵
        
        Args:
            assets: 资产列表
            correlation_matrix: 相关性矩阵，如果为None则根据资产ID初始化
        """
        logger.info(f"初始化量子纠缠矩阵，共{len(assets)}个资产")
        
        # 初始化资产的量子状态
        for asset in assets:
            self._initialize_quantum_state(asset)
        
        # 初始化纠缠关系
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i != j:
                    pair = (asset_i, asset_j)
                    
                    # 获取初始相关性
                    if correlation_matrix and pair in correlation_matrix:
                        initial_correlation = abs(correlation_matrix[pair])
                    else:
                        # 根据资产ID生成一个初始相关性
                        seed = hash(asset_i + asset_j) % 1000 / 1000.0
                        initial_correlation = 0.1 + seed * 0.3  # 0.1-0.4的范围
                    
                    # 创建纠缠属性
                    entanglement = EntanglementProperty(
                        strength=initial_correlation,
                        phase=np.random.random() * 2 * np.pi,
                        resonance_frequency=self.base_resonance_frequency * (0.8 + 0.4 * np.random.random()),
                        correlation_history=[initial_correlation],
                        efficiency=0.5,
                        half_life=12.0 + np.random.random() * 36.0  # 12-48小时不等
                    )
                    
                    self.entanglement_matrix[pair] = entanglement
        
        # 更新纠缠群组
        self._update_entanglement_clusters(threshold=0.3)
        
        logger.info(f"量子纠缠矩阵初始化完成，包含{len(self.entanglement_matrix)}个纠缠关系")
    
    def _initialize_quantum_state(self, asset: str) -> None:
        """
        初始化资产的量子状态
        
        Args:
            asset: 资产ID
        """
        # 创建一个随机的量子态矢量（复数向量）
        state = np.random.normal(0, 1, (self.dimensions, 2))
        complex_state = state[:, 0] + 1j * state[:, 1]
        
        # 归一化
        norm = np.sqrt(np.sum(np.abs(complex_state) ** 2))
        if norm > 0:
            complex_state = complex_state / norm
        
        self.quantum_states[asset] = complex_state
    
    def update_correlations(self, new_correlations: Dict[Tuple[str, str], float]) -> None:
        """
        更新资产间的相关性
        
        Args:
            new_correlations: 新的相关性矩阵 {(asset_i, asset_j): correlation}
        """
        # 当前时间
        now = datetime.now()
        
        # 更新纠缠矩阵
        for pair, correlation in new_correlations.items():
            asset_i, asset_j = pair
            correlation_value = abs(correlation)  # 使用相关性的绝对值
            
            # 检查是否需要创建新的纠缠关系
            if pair not in self.entanglement_matrix:
                self.entanglement_matrix[pair] = EntanglementProperty(
                    strength=correlation_value,
                    phase=np.random.random() * 2 * np.pi,
                    resonance_frequency=self.base_resonance_frequency * (0.8 + 0.4 * np.random.random()),
                    correlation_history=[correlation_value],
                    efficiency=0.5
                )
                continue
            
            # 获取现有纠缠属性
            entanglement = self.entanglement_matrix[pair]
            
            # 计算时间衰减
            hours_passed = (now - entanglement.last_updated).total_seconds() / 3600.0
            entanglement.decay(hours_passed)
            
            # 更新纠缠强度
            entanglement.update_strength(correlation_value)
            
            # 计算纠缠效率
            entanglement.calculate_efficiency()
            
            # 更新共振频率 - 随相关性波动
            frequency_shift = (correlation_value - entanglement.strength) * 0.1
            entanglement.resonance_frequency += frequency_shift
            entanglement.resonance_frequency = max(0.01, min(0.5, entanglement.resonance_frequency))
        
        # 更新纠缠群组
        self._update_entanglement_clusters()
        
        logger.debug(f"已更新{len(new_correlations)}个纠缠关系")
    
    def _update_entanglement_clusters(self, threshold: float = 0.5) -> None:
        """
        更新纠缠群组，将高度纠缠的资产归为同一群组
        
        Args:
            threshold: 纠缠强度阈值，超过此值的资产对被视为高度纠缠
        """
        # 收集所有资产
        all_assets = set()
        for pair in self.entanglement_matrix:
            all_assets.add(pair[0])
            all_assets.add(pair[1])
        
        # 构建邻接矩阵
        adjacency = {asset: set() for asset in all_assets}
        for (asset_i, asset_j), entanglement in self.entanglement_matrix.items():
            if entanglement.strength >= threshold:
                adjacency[asset_i].add(asset_j)
                adjacency[asset_j].add(asset_i)
        
        # 找出所有连通分量（群组）
        visited = set()
        self.entanglement_clusters = []
        
        for asset in all_assets:
            if asset not in visited:
                cluster = set()
                queue = [asset]
                visited.add(asset)
                
                while queue:
                    current = queue.pop(0)
                    cluster.add(current)
                    
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                if len(cluster) > 1:  # 只保留至少有两个资产的群组
                    self.entanglement_clusters.append(cluster)
        
        logger.debug(f"更新了纠缠群组，共{len(self.entanglement_clusters)}个群组")
    
    def apply_quantum_operations(self, market_data: Dict[str, Any]) -> None:
        """
        应用量子门操作，演化量子状态
        
        Args:
            market_data: 市场数据
        """
        # 获取所有资产
        assets = list(self.quantum_states.keys())
        
        # 应用单量子比特门操作
        for asset in assets:
            if asset in market_data:
                asset_data = market_data[asset]
                # 根据市场数据调整量子态
                self._apply_single_qubit_gate(asset, asset_data)
        
        # 应用两量子比特门操作（纠缠门）
        for (asset_i, asset_j), entanglement in self.entanglement_matrix.items():
            if asset_i in assets and asset_j in assets:
                self._apply_entanglement_gate(asset_i, asset_j, entanglement)
        
        # 应用多量子比特门操作（群组操作）
        for cluster in self.entanglement_clusters:
            cluster_assets = list(cluster)
            if len(cluster_assets) >= 3:  # 至少需要3个资产
                self._apply_cluster_operation(cluster_assets)
    
    def _apply_single_qubit_gate(self, asset: str, asset_data: Dict[str, Any]) -> None:
        """
        应用单量子比特门，根据市场数据调整量子态
        
        Args:
            asset: 资产ID
            asset_data: 资产市场数据
        """
        # 获取当前量子态
        state = self.quantum_states[asset]
        
        # 根据价格变化应用旋转门
        price_change = asset_data.get("price_change_pct", 0.0)
        angle = price_change * np.pi * 0.1  # 将价格变化转换为旋转角度
        
        # 构造旋转矩阵（在第一个维度上）
        rotation = np.eye(self.dimensions, dtype=complex)
        rotation[0, 0] = np.cos(angle) + 0j
        rotation[0, 1] = -np.sin(angle) + 0j
        rotation[1, 0] = np.sin(angle) + 0j
        rotation[1, 1] = np.cos(angle) + 0j
        
        # 应用旋转
        new_state = rotation @ state
        
        # 根据成交量应用相位门
        volume = asset_data.get("volume_relative", 1.0)
        phase_angle = volume * np.pi * 0.2
        
        # 构造相位矩阵
        phase = np.eye(self.dimensions, dtype=complex)
        phase[1, 1] = np.exp(1j * phase_angle)
        
        # 应用相位
        new_state = phase @ new_state
        
        # 归一化
        norm = np.sqrt(np.sum(np.abs(new_state) ** 2))
        if norm > 0:
            new_state = new_state / norm
        
        # 更新量子态
        self.quantum_states[asset] = new_state
    
    def _apply_entanglement_gate(self, asset_i: str, asset_j: str, 
                               entanglement: EntanglementProperty) -> None:
        """
        应用纠缠门，调整两个资产间的量子纠缠
        
        Args:
            asset_i: 第一个资产
            asset_j: 第二个资产
            entanglement: 纠缠属性
        """
        # 获取当前量子态
        state_i = self.quantum_states[asset_i]
        state_j = self.quantum_states[asset_j]
        
        # 纠缠强度
        strength = entanglement.strength
        
        # 创建CNOT类似的门（根据纠缠强度）
        # 当第一个资产的第一个维度为主导时，翻转第二个资产的第一个维度
        if abs(state_i[0]) > 0.7:
            # 构造翻转矩阵
            flip = np.eye(self.dimensions, dtype=complex)
            flip[0, 0] = np.cos(strength * np.pi / 2)
            flip[0, 1] = np.sin(strength * np.pi / 2)
            flip[1, 0] = np.sin(strength * np.pi / 2)
            flip[1, 1] = np.cos(strength * np.pi / 2)
            
            # 应用到第二个资产
            new_state_j = flip @ state_j
            
            # 归一化
            norm = np.sqrt(np.sum(np.abs(new_state_j) ** 2))
            if norm > 0:
                new_state_j = new_state_j / norm
            
            # 更新量子态
            self.quantum_states[asset_j] = new_state_j
        
        # 相位纠缠 - 根据两个资产的相对相位调整
        phase_diff = np.angle(state_i[0]) - np.angle(state_j[0])
        if abs(phase_diff) < 0.5:  # 相位接近时增强纠缠
            # 增加纠缠强度
            entanglement.strength = min(1.0, entanglement.strength * 1.01)
        else:  # 相位差距大时减弱纠缠
            # 减少纠缠强度
            entanglement.strength = max(0.1, entanglement.strength * 0.99)
    
    def _apply_cluster_operation(self, assets: List[str]) -> None:
        """
        应用群组量子操作，处理高度纠缠的资产群组
        
        Args:
            assets: 资产列表
        """
        if len(assets) < 3:
            return
            
        # 收集所有量子态
        states = [self.quantum_states[asset] for asset in assets]
        
        # 计算群组平均相位
        avg_phase = np.mean([np.angle(state[0]) for state in states])
        
        # 计算与平均相位的偏差
        for i, asset in enumerate(assets):
            state = states[i]
            phase_diff = np.angle(state[0]) - avg_phase
            
            # 应用相位调整，使群组更加同步
            adjustment = -phase_diff * 0.1  # 小幅调整，避免过度同步
            phase = np.exp(1j * adjustment)
            
            # 只调整第一个维度
            new_state = state.copy()
            new_state[0] = state[0] * phase
            
            # 归一化
            norm = np.sqrt(np.sum(np.abs(new_state) ** 2))
            if norm > 0:
                new_state = new_state / norm
            
            # 更新量子态
            self.quantum_states[asset] = new_state
    
    def compute_market_resonance(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        计算市场共振状态，识别市场异常模式
        
        Args:
            market_data: 市场数据
            
        Returns:
            市场共振状态评分字典
        """
        # 重置共振状态
        self.resonance_state = {}
        
        # 计算整体市场相位
        market_phase = self._compute_global_phase(market_data)
        
        # 更新全局相位参考
        phase_diff = market_phase - self.phase_reference
        self.phase_reference = market_phase
        
        # 计算每个资产的共振度
        for asset, state in self.quantum_states.items():
            # 资产相位与市场相位的关系
            asset_phase = np.angle(state[0])
            phase_alignment = 1.0 - abs(np.sin((asset_phase - market_phase) / 2))
            
            # 量子态的熵 - 低熵表示更确定的状态
            probabilities = np.abs(state) ** 2
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropy_factor = 1.0 - entropy / np.log2(self.dimensions)
            
            # 计算资产的共振评分
            resonance = phase_alignment * entropy_factor
            
            # 应用市场数据因子
            if asset in market_data:
                volume_factor = market_data[asset].get("volume_relative", 1.0)
                momentum = market_data[asset].get("momentum", 0.0)
                
                # 量化动量对共振的影响
                momentum_factor = 0.5 + 0.5 * min(1.0, abs(momentum) / 3.0)
                
                # 调整共振度
                resonance = resonance * volume_factor * momentum_factor
            
            # 记录共振状态
            self.resonance_state[asset] = resonance
        
        # 识别异常共振模式
        self._detect_resonance_anomalies()
        
        return self.resonance_state
    
    def _compute_global_phase(self, market_data: Dict[str, Any]) -> float:
        """
        计算全局市场相位
        
        Args:
            market_data: 市场数据
            
        Returns:
            市场全局相位
        """
        # 获取所有资产的价格变化
        price_changes = []
        for asset, data in market_data.items():
            if asset in self.quantum_states:
                change = data.get("price_change_pct", 0.0)
                price_changes.append(change)
        
        if not price_changes:
            return self.phase_reference
        
        # 计算平均价格变化和标准差
        mean_change = np.mean(price_changes)
        std_change = np.std(price_changes) if len(price_changes) > 1 else 0.01
        
        # 将价格变化转换为相位变化
        phase_change = mean_change * np.pi / (std_change + 0.01)
        
        # 更新全局相位
        new_phase = self.phase_reference + phase_change
        return new_phase % (2 * np.pi)
    
    def _detect_resonance_anomalies(self) -> Dict[str, float]:
        """
        检测共振异常，识别潜在的市场异常
        
        Returns:
            异常评分字典
        """
        # 计算共振度分布
        values = list(self.resonance_state.values())
        if not values:
            return {}
            
        mean_resonance = np.mean(values)
        std_resonance = np.std(values) if len(values) > 1 else 0.1
        
        # 识别异常共振的资产
        anomalies = {}
        for asset, resonance in self.resonance_state.items():
            # 计算z-分数
            z_score = (resonance - mean_resonance) / std_resonance if std_resonance > 0 else 0
            
            # 高于2个标准差视为异常
            if abs(z_score) > 2.0:
                anomalies[asset] = z_score
        
        # 记录异常
        if anomalies:
            logger.info(f"检测到{len(anomalies)}个共振异常: {anomalies}")
        
        return anomalies
    
    def predict_market_movement(self, assets: List[str]) -> Dict[str, Dict[str, float]]:
        """
        基于量子纠缠状态预测市场走势
        
        Args:
            assets: 需要预测的资产列表
            
        Returns:
            预测结果字典 {资产: {方向: 概率, 强度: 强度值}}
        """
        predictions = {}
        
        for asset in assets:
            if asset not in self.quantum_states:
                continue
                
            state = self.quantum_states[asset]
            
            # 使用量子态的前两个维度预测方向
            up_probability = abs(state[0]) ** 2
            down_probability = abs(state[1]) ** 2
            
            # 归一化
            total = up_probability + down_probability
            if total > 0:
                up_probability /= total
                down_probability /= total
            
            # 方向 - 上涨概率与下跌概率的差值
            direction = up_probability - down_probability
            
            # 强度 - 基于量子态的确定性
            probabilities = np.abs(state) ** 2
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(self.dimensions)
            certainty = 1.0 - entropy / max_entropy
            
            # 应用共振因子
            resonance = self.resonance_state.get(asset, 0.5)
            strength = certainty * (0.5 + 0.5 * resonance)
            
            # 创建预测结果
            predictions[asset] = {
                "direction": direction,
                "strength": strength,
                "up_probability": up_probability,
                "down_probability": down_probability,
                "resonance": resonance
            }
        
        return predictions
    
    def get_entanglement_network(self) -> Dict[str, Any]:
        """
        获取纠缠网络状态，用于可视化
        
        Returns:
            纠缠网络状态信息
        """
        # 收集所有资产
        all_assets = set()
        for pair in self.entanglement_matrix:
            all_assets.add(pair[0])
            all_assets.add(pair[1])
        
        # 创建节点数据
        nodes = []
        for asset in all_assets:
            # 计算节点的共振度
            resonance = self.resonance_state.get(asset, 0.5)
            
            # 获取量子态
            state = self.quantum_states.get(asset, np.zeros(self.dimensions))
            
            # 从量子态计算方向倾向
            direction = 0.0
            if len(state) >= 2:
                up_prob = abs(state[0]) ** 2
                down_prob = abs(state[1]) ** 2
                total = up_prob + down_prob
                if total > 0:
                    direction = (up_prob - down_prob) / total
            
            nodes.append({
                "id": asset,
                "resonance": resonance,
                "direction": direction
            })
        
        # 创建边数据
        edges = []
        for (asset_i, asset_j), entanglement in self.entanglement_matrix.items():
            if entanglement.strength > 0.1:  # 只显示有意义的纠缠
                edges.append({
                    "source": asset_i,
                    "target": asset_j,
                    "strength": entanglement.strength,
                    "efficiency": entanglement.efficiency
                })
        
        # 创建群组数据
        clusters = []
        for i, cluster in enumerate(self.entanglement_clusters):
            clusters.append({
                "id": i,
                "assets": list(cluster),
                "size": len(cluster)
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "timestamp": datetime.now().isoformat()
        } 