"""
自进化神经架构
实现可以重构自身连接、创建新节点、移除无效节点的神经网络系统
"""

import numpy as np
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import random
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class NeuralNode:
    """神经网络节点"""
    
    def __init__(self, node_id: Optional[str] = None, node_type: str = "hidden", 
                 activation: str = "relu", bias: float = 0.0):
        """
        初始化神经节点
        
        Args:
            node_id: 节点ID，如果为None则自动生成
            node_type: 节点类型 (input, hidden, output)
            activation: 激活函数类型
            bias: 偏置值
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.node_type = node_type
        self.activation = activation
        self.bias = bias
        self.incoming_connections = {}  # 源节点ID -> 权重
        self.outgoing_connections = {}  # 目标节点ID -> 权重
        self.value = 0.0
        self.gradient = 0.0
        self.creation_time = datetime.now()
        self.last_active = datetime.now()
        self.importance_score = 0.5  # 节点重要性评分(0-1)
        self.mutation_history = []
        
    def connect_to(self, target_node: 'NeuralNode', weight: float = None) -> None:
        """
        连接到目标节点
        
        Args:
            target_node: 目标节点
            weight: 连接权重，如果为None则随机初始化
        """
        if weight is None:
            weight = np.random.normal(0, 0.1)
            
        self.outgoing_connections[target_node.node_id] = weight
        target_node.incoming_connections[self.node_id] = weight
        
    def disconnect_from(self, target_node: 'NeuralNode') -> None:
        """
        断开与目标节点的连接
        
        Args:
            target_node: 目标节点
        """
        if target_node.node_id in self.outgoing_connections:
            del self.outgoing_connections[target_node.node_id]
            
        if self.node_id in target_node.incoming_connections:
            del target_node.incoming_connections[self.node_id]
            
    def calculate_activation(self, x: float) -> float:
        """
        计算激活函数值
        
        Args:
            x: 输入值
            
        Returns:
            激活后的值
        """
        if self.activation == "relu":
            return max(0, x)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "linear":
            return x
        else:
            return max(0, x)  # 默认使用ReLU
            
    def forward(self, inputs: Dict[str, float]) -> float:
        """
        前向传播计算节点值
        
        Args:
            inputs: 输入值字典 {节点ID: 值}
            
        Returns:
            节点输出值
        """
        # 对于输入节点，直接返回输入值
        if self.node_type == "input" and self.node_id in inputs:
            self.value = inputs[self.node_id]
            return self.value
            
        # 计算加权和
        weighted_sum = self.bias
        for node_id, weight in self.incoming_connections.items():
            if node_id in inputs:
                weighted_sum += inputs[node_id] * weight
                
        # 应用激活函数
        self.value = self.calculate_activation(weighted_sum)
        
        # 更新最后活跃时间
        self.last_active = datetime.now()
        
        return self.value
        
    def backward(self, gradient: float, learning_rate: float = 0.01) -> Dict[str, float]:
        """
        反向传播更新权重
        
        Args:
            gradient: 梯度值
            learning_rate: 学习率
            
        Returns:
            前一层节点的梯度字典 {节点ID: 梯度}
        """
        # 计算激活函数的导数
        if self.activation == "relu":
            activation_gradient = 1.0 if self.value > 0 else 0.0
        elif self.activation == "sigmoid":
            activation_gradient = self.value * (1 - self.value)
        elif self.activation == "tanh":
            activation_gradient = 1 - self.value ** 2
        elif self.activation == "linear":
            activation_gradient = 1.0
        else:
            activation_gradient = 1.0 if self.value > 0 else 0.0
            
        # 计算当前节点的梯度
        self.gradient = gradient * activation_gradient
        
        # 更新权重和偏置
        self.bias -= learning_rate * self.gradient
        
        # 计算前一层节点的梯度
        gradients = {}
        for node_id, weight in self.incoming_connections.items():
            # 更新权重
            weight_update = learning_rate * self.gradient
            self.incoming_connections[node_id] -= weight_update
            
            # 计算并传递梯度
            gradients[node_id] = self.gradient * weight
            
        return gradients
        
    def mutate(self, mutation_rate: float = 0.1) -> bool:
        """
        变异节点参数
        
        Args:
            mutation_rate: 变异率
            
        Returns:
            是否发生变异
        """
        mutated = False
        
        # 变异偏置
        if np.random.random() < mutation_rate:
            old_bias = self.bias
            self.bias += np.random.normal(0, 0.1)
            mutated = True
            self.mutation_history.append(("bias", old_bias, self.bias))
            
        # 变异激活函数
        if np.random.random() < mutation_rate / 2:  # 较低概率
            old_activation = self.activation
            activations = ["relu", "sigmoid", "tanh", "linear"]
            self.activation = np.random.choice([a for a in activations if a != self.activation])
            mutated = True
            self.mutation_history.append(("activation", old_activation, self.activation))
            
        # 变异连接权重
        for node_id in list(self.incoming_connections.keys()):
            if np.random.random() < mutation_rate:
                old_weight = self.incoming_connections[node_id]
                self.incoming_connections[node_id] += np.random.normal(0, 0.1)
                mutated = True
                self.mutation_history.append(("weight", node_id, old_weight, self.incoming_connections[node_id]))
                
        return mutated


class SelfEvolvingNetwork:
    """自进化神经网络 - 可以自动调整结构和参数"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化自进化网络
        
        Args:
            config: 网络配置
        """
        self.config = config
        self.nodes = {}  # ID -> 节点
        self.input_nodes = []  # 输入节点ID列表
        self.output_nodes = []  # 输出节点ID列表
        self.hidden_layers = []  # 每个元素是一个隐藏层节点ID列表
        self.creation_time = datetime.now()
        self.last_evolved = self.creation_time
        self.evolution_generation = 0
        self.performance_history = []
        self.structure_mutations = []
        
    def create_standard_network(self, input_size: int, hidden_sizes: List[int], output_size: int) -> None:
        """
        创建标准前馈神经网络结构
        
        Args:
            input_size: 输入层大小
            hidden_sizes: 各隐藏层大小列表
            output_size: 输出层大小
        """
        # 创建输入节点
        self.input_nodes = []
        for i in range(input_size):
            node_id = f"input_{i}"
            node = NeuralNode(node_id=node_id, node_type="input", activation="linear")
            self.nodes[node_id] = node
            self.input_nodes.append(node_id)
            
        # 创建隐藏层
        self.hidden_layers = []
        prev_layer = self.input_nodes
        
        for h, size in enumerate(hidden_sizes):
            layer = []
            for i in range(size):
                node_id = f"hidden_{h}_{i}"
                node = NeuralNode(node_id=node_id, node_type="hidden", activation="relu")
                self.nodes[node_id] = node
                layer.append(node_id)
                
                # 连接到上一层的所有节点
                for prev_id in prev_layer:
                    self.nodes[prev_id].connect_to(node)
                    
            self.hidden_layers.append(layer)
            prev_layer = layer
            
        # 创建输出节点
        self.output_nodes = []
        for i in range(output_size):
            node_id = f"output_{i}"
            node = NeuralNode(node_id=node_id, node_type="output", activation="sigmoid")
            self.nodes[node_id] = node
            self.output_nodes.append(node_id)
            
            # 连接到最后一个隐藏层
            for prev_id in prev_layer:
                self.nodes[prev_id].connect_to(node)
                
        logger.info(f"Created network with {input_size} inputs, {hidden_sizes} hidden, {output_size} outputs")
        
    def forward(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        前向传播计算输出
        
        Args:
            inputs: 输入值字典 {输入节点ID索引: 值}
            
        Returns:
            输出值字典 {输出节点ID: 值}
        """
        # 准备完整的节点值字典
        node_values = {}
        
        # 设置输入值
        for i, node_id in enumerate(self.input_nodes):
            if i in inputs:
                node_values[node_id] = inputs[i]
            else:
                node_values[node_id] = 0.0
                
        # 逐层前向传播
        for layer in self.hidden_layers:
            for node_id in layer:
                node = self.nodes[node_id]
                node_values[node_id] = node.forward(node_values)
                
        # 输出层
        outputs = {}
        for i, node_id in enumerate(self.output_nodes):
            node = self.nodes[node_id]
            value = node.forward(node_values)
            outputs[i] = value
            node_values[node_id] = value
            
        return outputs
        
    def backward(self, target_outputs: Dict[int, float], learning_rate: float = 0.01) -> float:
        """
        反向传播更新权重
        
        Args:
            target_outputs: 目标输出值 {输出节点索引: 目标值}
            learning_rate: 学习率
            
        Returns:
            损失值
        """
        # 计算输出层梯度和损失
        loss = 0.0
        output_gradients = {}
        
        for i, node_id in enumerate(self.output_nodes):
            if i in target_outputs:
                node = self.nodes[node_id]
                target = target_outputs[i]
                output = node.value
                
                # 计算均方误差损失
                error = output - target
                loss += error ** 2 / 2
                
                # 输出层梯度 = 误差
                output_gradients[node_id] = error
                
        # 反向传播
        layer_gradients = output_gradients
        
        # 从输出层向输入层反向传播
        layers = self.hidden_layers.copy()
        layers.reverse()
        
        for layer in layers:
            next_gradients = {}
            
            for node_id in layer:
                node = self.nodes[node_id]
                if node_id in layer_gradients:
                    gradient = layer_gradients[node_id]
                    # 反向传播并获取前一层的梯度
                    prev_gradients = node.backward(gradient, learning_rate)
                    
                    # 合并梯度
                    for prev_id, grad in prev_gradients.items():
                        if prev_id in next_gradients:
                            next_gradients[prev_id] += grad
                        else:
                            next_gradients[prev_id] = grad
                            
            layer_gradients = next_gradients
            
        return loss
        
    def train(self, inputs: List[Dict[int, float]], targets: List[Dict[int, float]], 
              epochs: int = 100, learning_rate: float = 0.01, batch_size: int = 32) -> List[float]:
        """
        训练网络
        
        Args:
            inputs: 输入数据列表
            targets: 目标输出列表
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批量大小
            
        Returns:
            每个epoch的损失列表
        """
        epoch_losses = []
        n_samples = len(inputs)
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            
            total_loss = 0.0
            # 批量训练
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                batch_loss = 0.0
                
                for idx in batch_indices:
                    # 前向传播
                    self.forward(inputs[idx])
                    
                    # 反向传播
                    loss = self.backward(targets[idx], learning_rate)
                    batch_loss += loss
                    
                total_loss += batch_loss
                
            avg_loss = total_loss / n_samples
            epoch_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
                
        # 训练后更新节点重要性
        self._update_node_importance()
        
        return epoch_losses
        
    def _update_node_importance(self) -> None:
        """更新节点重要性评分"""
        # 基于连接权重和活动度评估节点重要性
        for node_id, node in self.nodes.items():
            if node.node_type == "input" or node.node_type == "output":
                node.importance_score = 1.0  # 输入输出节点始终重要
                continue
                
            # 计算连接权重的平均绝对值
            incoming_weights = list(node.incoming_connections.values())
            outgoing_weights = list(node.outgoing_connections.values())
            all_weights = incoming_weights + outgoing_weights
            
            if all_weights:
                weight_importance = np.mean(np.abs(all_weights))
            else:
                weight_importance = 0.0
                
            # 最后活跃时间的衰减因子
            time_since_active = (datetime.now() - node.last_active).total_seconds()
            activity_factor = np.exp(-time_since_active / (24 * 3600))  # 一天的时间常数
            
            # 计算最终重要性
            node.importance_score = 0.7 * weight_importance + 0.3 * activity_factor
            
    def evolve(self) -> None:
        """进化网络结构和参数"""
        self.evolution_generation += 1
        logger.info(f"Evolving network - Generation {self.evolution_generation}")
        
        # 更新节点重要性评分
        self._update_node_importance()
        
        # 1. 参数变异
        self._mutate_parameters()
        
        # 2. 结构变异
        if self.evolution_generation % 5 == 0:  # 每5代进行一次结构变异
            self._mutate_structure()
            
        # 3. 移除不重要的节点
        if self.evolution_generation % 10 == 0:  # 每10代进行一次修剪
            self._prune_unimportant_nodes()
            
        self.last_evolved = datetime.now()
        
    def _mutate_parameters(self) -> None:
        """变异网络参数"""
        mutation_rate = self.config.get("mutation_rate", 0.05)
        
        for node_id, node in self.nodes.items():
            if node.node_type != "input":  # 不变异输入节点
                mutated = node.mutate(mutation_rate)
                if mutated:
                    logger.debug(f"Mutated parameters of node {node_id}")
                    
    def _mutate_structure(self) -> None:
        """变异网络结构"""
        # 1. 添加新节点（分裂连接）
        self._add_nodes()
        
        # 2. 添加新连接
        self._add_connections()
        
        # 记录结构变异
        self.structure_mutations.append({
            "generation": self.evolution_generation,
            "time": datetime.now(),
            "node_count": len(self.nodes),
            "connection_count": sum(len(node.outgoing_connections) for node in self.nodes.values())
        })
        
    def _add_nodes(self) -> None:
        """添加新节点（分裂现有连接）"""
        novelty_bias = self.config.get("novelty_bias", 0.3)
        
        # 选择可能的分裂连接
        splittable_connections = []
        for source_id, source_node in self.nodes.items():
            for target_id in source_node.outgoing_connections.keys():
                if source_id not in self.output_nodes and target_id not in self.input_nodes:
                    splittable_connections.append((source_id, target_id))
                    
        # 确定要添加的节点数
        n_connections = len(splittable_connections)
        if n_connections == 0:
            return
            
        # 基于当前大小确定可添加的节点数量
        current_hidden_nodes = sum(len(layer) for layer in self.hidden_layers)
        max_new_nodes = max(1, int(current_hidden_nodes * novelty_bias * 0.2))
        n_new_nodes = min(max_new_nodes, n_connections)
        
        if n_new_nodes == 0:
            return
            
        # 随机选择要分裂的连接
        connections_to_split = random.sample(splittable_connections, n_new_nodes)
        
        for source_id, target_id in connections_to_split:
            # 创建新节点
            new_id = f"hidden_new_{uuid.uuid4().hex[:8]}"
            new_node = NeuralNode(node_id=new_id, node_type="hidden", 
                                 activation=np.random.choice(["relu", "sigmoid", "tanh"]))
            
            # 获取原始连接权重并移除
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            original_weight = source_node.outgoing_connections[target_id]
            source_node.disconnect_from(target_node)
            
            # 连接源节点到新节点，以及新节点到目标节点
            w1 = original_weight * (0.8 + 0.4 * np.random.random())  # 轻微随机化权重
            w2 = original_weight * (0.8 + 0.4 * np.random.random())
            
            source_node.connect_to(new_node, w1)
            new_node.connect_to(target_node, w2)
            
            # 添加到网络
            self.nodes[new_id] = new_node
            
            # 确定新节点应该属于哪个隐藏层
            source_layer = -1
            target_layer = -1
            
            for i, layer in enumerate(self.hidden_layers):
                if source_id in layer:
                    source_layer = i
                if target_id in layer:
                    target_layer = i
                    
            if source_id in self.input_nodes:
                source_layer = -1
            if target_id in self.output_nodes:
                target_layer = len(self.hidden_layers)
                
            # 如果源和目标在相邻层，将新节点放入目标层前面的新层
            if target_layer - source_layer == 1:
                # 创建新层
                new_layer = [new_id]
                self.hidden_layers.insert(source_layer + 1, new_layer)
            # 如果源和目标在同一层，放入同一层
            elif source_layer == target_layer and source_layer >= 0:
                self.hidden_layers[source_layer].append(new_id)
            # 否则放入源层后的下一层
            elif source_layer >= 0:
                if source_layer + 1 < len(self.hidden_layers):
                    self.hidden_layers[source_layer + 1].append(new_id)
                else:
                    # 添加新层
                    self.hidden_layers.append([new_id])
            else:
                # 边缘情况，添加到第一个隐藏层
                if self.hidden_layers:
                    self.hidden_layers[0].append(new_id)
                else:
                    self.hidden_layers.append([new_id])
                    
            logger.debug(f"Added new node {new_id} by splitting connection {source_id}->{target_id}")
                
    def _add_connections(self) -> None:
        """添加新连接"""
        max_new_connections = int(len(self.nodes) * 0.1)  # 最多添加10%的新连接
        
        if max_new_connections == 0:
            return
            
        n_new_connections = np.random.randint(1, max_new_connections + 1)
        
        # 尝试添加新连接
        for _ in range(n_new_connections):
            # 选择源节点（不能是输出节点）
            valid_sources = [node_id for node_id, node in self.nodes.items() 
                            if node_id not in self.output_nodes]
            
            if not valid_sources:
                continue
                
            source_id = np.random.choice(valid_sources)
            source_node = self.nodes[source_id]
            
            # 选择目标节点（不能是输入节点且不能已经连接）
            valid_targets = [node_id for node_id in self.nodes 
                            if node_id not in self.input_nodes 
                            and node_id not in source_node.outgoing_connections]
            
            if not valid_targets:
                continue
                
            target_id = np.random.choice(valid_targets)
            target_node = self.nodes[target_id]
            
            # 添加连接
            weight = np.random.normal(0, 0.1)
            source_node.connect_to(target_node, weight)
            
            logger.debug(f"Added new connection from {source_id} to {target_id} with weight {weight:.4f}")
                
    def _prune_unimportant_nodes(self) -> None:
        """移除不重要的节点"""
        preservation_threshold = self.config.get("preservation_threshold", 0.7)
        
        # 不移除输入输出节点
        prunable_nodes = []
        
        for node_id, node in self.nodes.items():
            if node.node_type == "hidden" and node.importance_score < preservation_threshold:
                prunable_nodes.append(node_id)
                
        # 最多移除10%的节点
        max_prune = max(1, int(len(prunable_nodes) * 0.1))
        
        if not prunable_nodes:
            return
            
        # 按重要性排序，移除最不重要的节点
        nodes_to_prune = sorted([(node_id, self.nodes[node_id].importance_score) 
                                for node_id in prunable_nodes], 
                               key=lambda x: x[1])[:max_prune]
        
        for node_id, score in nodes_to_prune:
            self._remove_node(node_id)
            logger.debug(f"Pruned node {node_id} with importance score {score:.4f}")
            
    def _remove_node(self, node_id: str) -> None:
        """
        从网络中移除节点
        
        Args:
            node_id: 要移除的节点ID
        """
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        # 连接输入节点和输出节点，跳过被移除的节点
        for source_id in node.incoming_connections:
            if source_id in self.nodes:
                source_node = self.nodes[source_id]
                
                for target_id in node.outgoing_connections:
                    if target_id in self.nodes:
                        target_node = self.nodes[target_id]
                        
                        # 创建新连接，权重为原始连接权重的乘积
                        weight = node.incoming_connections[source_id] * node.outgoing_connections[target_id]
                        source_node.connect_to(target_node, weight)
                        
        # 从所有层列表中移除
        for layer in self.hidden_layers:
            if node_id in layer:
                layer.remove(node_id)
                
        # 移除空的隐藏层
        self.hidden_layers = [layer for layer in self.hidden_layers if layer]
        
        # 删除节点
        del self.nodes[node_id]
        
    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        return {
            "node_count": len(self.nodes),
            "input_count": len(self.input_nodes),
            "output_count": len(self.output_nodes),
            "hidden_layers": [len(layer) for layer in self.hidden_layers],
            "connection_count": sum(len(node.outgoing_connections) for node in self.nodes.values()),
            "evolution_generation": self.evolution_generation,
            "last_evolved": self.last_evolved.isoformat()
        }
        
    def save_checkpoint(self) -> Dict[str, Any]:
        """保存网络状态"""
        checkpoint = {
            "nodes": {},
            "input_nodes": self.input_nodes,
            "output_nodes": self.output_nodes,
            "hidden_layers": self.hidden_layers,
            "evolution_generation": self.evolution_generation,
            "creation_time": self.creation_time.isoformat(),
            "last_evolved": self.last_evolved.isoformat(),
            "performance_history": self.performance_history,
            "structure_mutations": self.structure_mutations
        }
        
        # 序列化节点
        for node_id, node in self.nodes.items():
            checkpoint["nodes"][node_id] = {
                "node_type": node.node_type,
                "activation": node.activation,
                "bias": node.bias,
                "incoming_connections": node.incoming_connections,
                "outgoing_connections": node.outgoing_connections,
                "importance_score": node.importance_score,
                "creation_time": node.creation_time.isoformat(),
                "last_active": node.last_active.isoformat(),
                "mutation_history": node.mutation_history
            }
            
        return checkpoint
        
    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        从检查点恢复网络
        
        Args:
            checkpoint: 网络检查点
        """
        self.nodes = {}
        self.input_nodes = checkpoint["input_nodes"]
        self.output_nodes = checkpoint["output_nodes"]
        self.hidden_layers = checkpoint["hidden_layers"]
        self.evolution_generation = checkpoint["evolution_generation"]
        self.creation_time = datetime.fromisoformat(checkpoint["creation_time"])
        self.last_evolved = datetime.fromisoformat(checkpoint["last_evolved"])
        self.performance_history = checkpoint["performance_history"]
        self.structure_mutations = checkpoint["structure_mutations"]
        
        # 反序列化节点
        for node_id, node_data in checkpoint["nodes"].items():
            node = NeuralNode(
                node_id=node_id,
                node_type=node_data["node_type"],
                activation=node_data["activation"],
                bias=node_data["bias"]
            )
            
            node.incoming_connections = node_data["incoming_connections"]
            node.outgoing_connections = node_data["outgoing_connections"]
            node.importance_score = node_data["importance_score"]
            node.creation_time = datetime.fromisoformat(node_data["creation_time"])
            node.last_active = datetime.fromisoformat(node_data["last_active"])
            node.mutation_history = node_data["mutation_history"]
            
            self.nodes[node_id] = node
            
        logger.info(f"Loaded network with {len(self.nodes)} nodes and {len(self.hidden_layers)} hidden layers") 