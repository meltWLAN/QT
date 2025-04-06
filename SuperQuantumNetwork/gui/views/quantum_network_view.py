#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 量子网络视图
可视化量子共生网络的状态和决策过程
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QGroupBox, QFormLayout, QSlider, QSpinBox,
    QTabWidget, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush
import pyqtgraph as pg
import numpy as np
import pandas as pd
from datetime import datetime


class NetworkVisualizer(QWidget):
    """网络可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建示例数据
        self.nodes = []
        self.edges = []
        self._generate_sample_data()
        
        # 设置UI
        self._setup_ui()
    
    def _generate_sample_data(self):
        """生成示例数据"""
        # 创建节点
        for i in range(20):
            self.nodes.append({
                'id': i,
                'name': f'节点{i}',
                'type': np.random.choice(['input', 'hidden', 'output']),
                'pos': [np.random.uniform(0, 100), np.random.uniform(0, 100)],
                'value': np.random.normal(0, 1)
            })
        
        # 创建边
        for i in range(30):
            source = np.random.randint(0, 20)
            target = np.random.randint(0, 20)
            if source != target:
                self.edges.append({
                    'source': source,
                    'target': target,
                    'weight': np.random.normal(0, 1)
                })
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建图表布局
        self.graph_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graph_widget)
        
        # 设置背景颜色
        self.graph_widget.setBackground('k')
        
        # 创建散点图
        self.plot = self.graph_widget.addPlot()
        self.plot.setAspectLocked()
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        
        # 绘制节点
        self._draw_nodes()
        
        # 绘制边
        self._draw_edges()
    
    def _draw_nodes(self):
        """绘制节点"""
        # 节点位置
        pos = np.array([node['pos'] for node in self.nodes])
        
        # 节点颜色 - 修复颜色列表的创建方式
        colors = []
        for node in self.nodes:
            if node['type'] == 'input':
                colors.append((0, 0, 255))  # 蓝色表示输入节点
            elif node['type'] == 'hidden':
                colors.append((255, 255, 255))  # 白色表示隐藏节点
            else:
                colors.append((255, 0, 0))  # 红色表示输出节点
        
        # 节点大小
        sizes = 10 * np.ones(len(self.nodes))
        
        # 创建散点图
        self.scatter = pg.ScatterPlotItem(
            pos=pos,
            size=sizes,
            pen=pg.mkPen(None),
            brush=colors
        )
        self.plot.addItem(self.scatter)
        
        # 添加节点标签
        for i, node in enumerate(self.nodes):
            text = pg.TextItem(text=node['name'], color='w')
            text.setPos(node['pos'][0], node['pos'][1] + 5)
            self.plot.addItem(text)
    
    def _draw_edges(self):
        """绘制边"""
        for edge in self.edges:
            source = self.nodes[edge['source']]
            target = self.nodes[edge['target']]
            
            # 计算边的颜色
            if edge['weight'] > 0:
                color = pg.mkColor(255, 0, 0, 100 + int(abs(edge['weight']) * 100))  # 正权重为红色
            else:
                color = pg.mkColor(0, 255, 0, 100 + int(abs(edge['weight']) * 100))  # 负权重为绿色
            
            # 创建线条
            line = pg.PlotCurveItem(
                x=[source['pos'][0], target['pos'][0]],
                y=[source['pos'][1], target['pos'][1]],
                pen=pg.mkPen(color, width=1 + abs(edge['weight']))
            )
            self.plot.addItem(line)
    
    def update_network_data(self, nodes, edges):
        """更新网络数据"""
        self.nodes = nodes
        self.edges = edges
        
        # 清空图表
        self.plot.clear()
        
        # 重新绘制
        self._draw_nodes()
        self._draw_edges()


class NetworkStatusWidget(QWidget):
    """网络状态组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建分组框
        status_group = QGroupBox("量子共生网络状态")
        status_layout = QFormLayout(status_group)
        
        # 添加状态标签
        self.active_label = QLabel("活跃")
        self.active_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addRow("状态:", self.active_label)
        
        self.segments_label = QLabel("5")
        status_layout.addRow("市场分段:", self.segments_label)
        
        self.agents_label = QLabel("25")
        status_layout.addRow("智能体数量:", self.agents_label)
        
        self.learning_label = QLabel("已开启")
        status_layout.addRow("自学习:", self.learning_label)
        
        self.evolution_label = QLabel("第3代")
        status_layout.addRow("进化阶段:", self.evolution_label)
        
        self.performance_label = QLabel("85.2%")
        status_layout.addRow("性能评分:", self.performance_label)
        
        # 添加状态分组到主布局
        main_layout.addWidget(status_group)
        
        # 创建控制分组框
        control_group = QGroupBox("网络控制")
        control_layout = QFormLayout(control_group)
        
        # 添加控制组件
        self.learning_rate_slider = QSlider(Qt.Horizontal)
        self.learning_rate_slider.setRange(1, 100)
        self.learning_rate_slider.setValue(10)
        control_layout.addRow("学习率:", self.learning_rate_slider)
        
        self.mutation_rate_slider = QSlider(Qt.Horizontal)
        self.mutation_rate_slider.setRange(1, 100)
        self.mutation_rate_slider.setValue(5)
        control_layout.addRow("变异率:", self.mutation_rate_slider)
        
        self.agents_spinbox = QSpinBox()
        self.agents_spinbox.setRange(1, 100)
        self.agents_spinbox.setValue(5)
        control_layout.addRow("每段智能体:", self.agents_spinbox)
        
        self.reset_button = QPushButton("重置网络")
        control_layout.addRow("操作:", self.reset_button)
        
        # 添加控制分组到主布局
        main_layout.addWidget(control_group)
        
        # 添加弹簧，使组件靠上对齐
        main_layout.addStretch(1)
    
    def update_status(self, status_data):
        """更新状态数据"""
        # 更新状态标签
        self.segments_label.setText(str(status_data.get('segments', 0)))
        self.agents_label.setText(str(status_data.get('agents', 0)))
        self.learning_label.setText("已开启" if status_data.get('learning', False) else "已关闭")
        self.evolution_label.setText(f"第{status_data.get('evolution', 0)}代")
        self.performance_label.setText(f"{status_data.get('performance', 0.0):.1f}%")


class NetworkInsightWidget(QWidget):
    """网络洞察组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建决策日志
        log_group = QGroupBox("量子决策日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # 添加示例日志
        self.log_text.append("[09:30:15] 量子网络初始化完成")
        self.log_text.append("[09:30:20] 开始分析市场数据")
        self.log_text.append("[09:31:05] 检测到隐藏市场模式：波动收敛")
        self.log_text.append("[09:32:10] 发现潜在交易机会：科技板块")
        self.log_text.append("[09:33:22] 量子概率收敛，推荐买入：600000")
        
        # 添加日志分组到主布局
        main_layout.addWidget(log_group)
    
    def add_log(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_text.append(f"{timestamp} {message}")
        
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )


class QuantumNetworkView(QWidget):
    """量子网络视图"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建选项卡部件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 创建可视化选项卡
        self.visualizer = NetworkVisualizer()
        self.tab_widget.addTab(self.visualizer, "网络可视化")
        
        # 创建状态选项卡
        self.status_widget = NetworkStatusWidget()
        self.tab_widget.addTab(self.status_widget, "网络状态")
        
        # 创建洞察选项卡
        self.insight_widget = NetworkInsightWidget()
        self.tab_widget.addTab(self.insight_widget, "决策洞察")
    
    def initialize_with_data(self, data):
        """使用数据初始化视图"""
        # 更新网络状态
        network_status = data.get("network_status", {})
        self.status_widget.update_status(network_status)
        
        # 添加决策日志
        self.insight_widget.add_log("量子网络数据加载完成") 