#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 量子视图模块
提供系统量子网络状态可视化
"""

import sys
import os
import logging
import traceback
import random
import math
from datetime import datetime, timedelta
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout,
    QFrame, QSizePolicy, QSpacerItem, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QComboBox, QGraphicsDropShadowEffect,
    QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget, QSplitter
)
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QRectF, QPointF
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QBrush, QRadialGradient, QPainterPath

logger = logging.getLogger(__name__)

# 尝试导入pyqtgraph (用于高级图表)
try:
    import pyqtgraph as pg
    import numpy as np
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    logger.warning("未找到pyqtgraph，将使用简化图表")


class QuantumStateVisualizer(QWidget):
    """量子状态可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置背景色
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(10, 15, 35))
        self.setPalette(palette)
        
        # 设置最小高度
        self.setMinimumHeight(250)
        
        # 量子状态数据
        self.states = []
        for i in range(8):  # 8个量子态
            angle = random.uniform(0, 2 * math.pi)
            self.states.append({
                'angle': angle,
                'amplitude': random.uniform(0.7, 1.0),
                'phase': random.uniform(0, 2 * math.pi),
                'color': (
                    int(128 + 127 * math.sin(i * 0.7)),
                    int(128 + 127 * math.sin(i * 0.7 + 2)),
                    int(128 + 127 * math.sin(i * 0.7 + 4))
                )
            })
        
        # 启动动画定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_animation)
        self.timer.start(50)  # 50ms刷新一次
    
    def _update_animation(self):
        """更新动画"""
        for state in self.states:
            # 缓慢改变相位
            state['phase'] += 0.02
            if state['phase'] > 2 * math.pi:
                state['phase'] -= 2 * math.pi
            
            # 缓慢改变振幅
            state['amplitude'] += random.uniform(-0.01, 0.01)
            state['amplitude'] = max(0.5, min(1.0, state['amplitude']))
        
        self.update()
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        center_x, center_y = w / 2, h / 2
        radius = min(w, h) / 2 - 20
        
        # 绘制背景
        painter.fillRect(self.rect(), QBrush(QColor(10, 15, 35)))
        
        # 绘制坐标轴
        painter.setPen(QPen(QColor(80, 80, 100), 1, Qt.DashLine))
        painter.drawLine(QPointF(center_x - radius, center_y), QPointF(center_x + radius, center_y))
        painter.drawLine(QPointF(center_x, center_y - radius), QPointF(center_x, center_y + radius))
        
        # 绘制圆
        painter.setPen(QPen(QColor(50, 50, 70), 1))
        painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))
        
        # 绘制量子态
        for i, state in enumerate(self.states):
            # 计算位置
            angle = state['angle']
            amplitude = state['amplitude']
            phase = state['phase']
            
            x = center_x + radius * amplitude * math.cos(angle)
            y = center_y + radius * amplitude * math.sin(angle)
            
            # 绘制到原点的线
            painter.setPen(QPen(QColor(*state['color'], 100), 1, Qt.DashLine))
            painter.drawLine(QPointF(center_x, center_y), QPointF(x, y))
            
            # 绘制状态点
            painter.setPen(Qt.NoPen)
            
            # 使用相位创建脉动效果
            pulse = (math.sin(phase) + 1) / 2
            size = 5 + pulse * 5
            
            # 创建径向渐变
            gradient = QRadialGradient(x, y, size * 2)
            r, g, b = state['color']
            gradient.setColorAt(0, QColor(r, g, b, 255))
            gradient.setColorAt(0.5, QColor(r, g, b, 150))
            gradient.setColorAt(1, QColor(r, g, b, 0))
            
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(QRectF(x - size, y - size, size * 2, size * 2))
            
            # 绘制状态标签
            painter.setPen(QColor(200, 200, 220))
            painter.drawText(x + 10, y + 10, f"|ψ{i}>")


class EntanglementGraphWidget(QWidget):
    """量子纠缠图组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置背景色
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(10, 15, 35))
        self.setPalette(palette)
        
        # 设置最小高度
        self.setMinimumHeight(250)
        
        # 节点数据
        self.nodes = []
        node_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BABA"]
        
        for i, name in enumerate(node_names):
            angle = i * (2 * math.pi / len(node_names))
            x = 0.5 + 0.4 * math.cos(angle)
            y = 0.5 + 0.4 * math.sin(angle)
            
            self.nodes.append({
                'name': name,
                'x': x,
                'y': y,
                'size': random.uniform(0.03, 0.05),
                'color': (
                    int(100 + 155 * (i / len(node_names))),
                    int(100 + 155 * ((i + 3) / len(node_names)) % 1),
                    int(100 + 155 * ((i + 6) / len(node_names)) % 1)
                ),
                'pulse': 0,
                'pulse_dir': 1,
                'pulse_speed': random.uniform(0.001, 0.003),
                'value': random.uniform(0.5, 1.5)
            })
        
        # 边数据 (纠缠关系)
        self.edges = []
        
        # 生成随机纠缠关系
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                # 随机决定是否有纠缠关系
                if random.random() < 0.3:
                    strength = random.uniform(0.2, 0.9)
                    self.edges.append({
                        'from': i,
                        'to': j,
                        'strength': strength
                    })
        
        # 启动动画定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_animation)
        self.timer.start(50)  # 50ms刷新一次
    
    def _update_animation(self):
        """更新动画"""
        for node in self.nodes:
            # 更新脉动
            node['pulse'] += node['pulse_speed'] * node['pulse_dir']
            if node['pulse'] > 0.5 or node['pulse'] < -0.5:
                node['pulse_dir'] *= -1
        
        # 随机改变边的强度
        for edge in self.edges:
            edge['strength'] += random.uniform(-0.01, 0.01)
            edge['strength'] = max(0.1, min(0.9, edge['strength']))
        
        self.update()
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 绘制背景
        painter.fillRect(self.rect(), QBrush(QColor(10, 15, 35)))
        
        # 绘制边 (纠缠关系)
        for edge in self.edges:
            from_node = self.nodes[edge['from']]
            to_node = self.nodes[edge['to']]
            
            x1, y1 = from_node['x'] * w, from_node['y'] * h
            x2, y2 = to_node['x'] * w, to_node['y'] * h
            
            # 纠缠强度决定线的颜色和宽度
            strength = edge['strength']
            color = QColor(0, int(150 * strength), int(255 * strength), int(150 * strength))
            
            pen = QPen(color)
            pen.setWidthF(1 + strength * 3)  # 线宽与强度成正比
            painter.setPen(pen)
            
            # 绘制曲线
            c1x = (x1 + x2) / 2 + random.uniform(-30, 30)
            c1y = (y1 + y2) / 2 + random.uniform(-30, 30)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        
        # 绘制节点
        for node in self.nodes:
            x, y = node['x'] * w, node['y'] * h
            base_size = node['size'] * min(w, h)
            pulse = node['pulse']
            
            # 调整大小
            size = base_size * (1 + pulse * 0.3)
            
            # 创建径向渐变
            r, g, b = node['color']
            gradient = QRadialGradient(x, y, size * 1.5)
            gradient.setColorAt(0, QColor(r, g, b, 255))
            gradient.setColorAt(0.7, QColor(r, g, b, 180))
            gradient.setColorAt(1, QColor(r, g, b, 0))
            
            # 绘制节点
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(QRectF(x - size, y - size, size * 2, size * 2))
            
            # 绘制节点名称
            painter.setPen(QColor(230, 230, 240))
            painter.setFont(QFont("Arial", 9))
            
            # 计算文本位置
            text_x = x - painter.fontMetrics().horizontalAdvance(node['name']) / 2
            text_y = y + size * 2 + 15
            
            painter.drawText(text_x, text_y, node['name'])
            
            # 绘制价值标签
            value_text = f"{node['value']:.2f}"
            val_width = painter.fontMetrics().horizontalAdvance(value_text)
            
            painter.setPen(QColor(r, g, b))
            painter.drawText(x - val_width / 2, y - size * 2 - 5, value_text)


class QuantumControlPanel(QGroupBox):
    """量子控制面板"""
    
    def __init__(self, parent=None):
        super().__init__("量子引擎控制", parent)
        
        # 设置样式
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3050AA;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: #00AAFF;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 3px;
            }
            
            QPushButton {
                background-color: #0A0A28;
                color: #00DDFF;
                border: 1px solid #0055AA;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #101038;
                border: 1px solid #00AAFF;
            }
            
            QPushButton:pressed {
                background-color: #080818;
                border: 2px solid #00FFFF;
            }
            
            QLabel {
                color: #AAAAEE;
            }
        """)
        
        # 设置布局
        layout = QGridLayout(self)
        
        # 添加控制选项
        layout.addWidget(QLabel("维度:"), 0, 0)
        self.dim_spinner = QSpinBox()
        self.dim_spinner.setMinimum(2)
        self.dim_spinner.setMaximum(32)
        self.dim_spinner.setValue(8)
        layout.addWidget(self.dim_spinner, 0, 1)
        
        layout.addWidget(QLabel("纠缠因子:"), 0, 2)
        self.entangle_spinner = QDoubleSpinBox()
        self.entangle_spinner.setMinimum(0.0)
        self.entangle_spinner.setMaximum(1.0)
        self.entangle_spinner.setSingleStep(0.05)
        self.entangle_spinner.setValue(0.3)
        layout.addWidget(self.entangle_spinner, 0, 3)
        
        layout.addWidget(QLabel("自学习:"), 1, 0)
        self.learning_check = QCheckBox()
        self.learning_check.setChecked(True)
        layout.addWidget(self.learning_check, 1, 1)
        
        layout.addWidget(QLabel("优化方法:"), 1, 2)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["量子梯度下降", "QAOA", "VQE", "随机搜索"])
        layout.addWidget(self.optimizer, 1, 3)
        
        # 添加控制按钮
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("运行计算")
        self.stop_button = QPushButton("停止")
        self.reset_button = QPushButton("重置系统")
        
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout, 2, 0, 1, 4)
        
        # 添加进度条
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress, 3, 0, 1, 4)


class QuantumResultsWidget(QGroupBox):
    """量子计算结果组件"""
    
    def __init__(self, parent=None):
        super().__init__("计算结果", parent)
        
        # 设置样式
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3050AA;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: #00AAFF;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 3px;
            }
            
            QTableWidget {
                background-color: #0A0A18;
                alternate-background-color: #101020;
                color: #CCCCFF;
                gridline-color: #303050;
                border: 1px solid #303050;
                border-radius: 4px;
            }
            
            QTableWidget::item:selected {
                background-color: #003366;
                color: #FFFFFF;
            }
            
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                           stop:0 #202040, stop:1 #101030);
                color: #00DDFF;
                padding: 4px;
                border: 1px solid #303050;
                font-weight: bold;
            }
        """)
        
        # 设置布局
        layout = QVBoxLayout(self)
        
        # 创建表格
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["资产", "纠缠度", "共振状态", "预测", "概率"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        
        layout.addWidget(self.table)
        
        # 填充随机数据
        self._populate_random_data()
    
    def _populate_random_data(self):
        """填充随机数据"""
        assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BABA"]
        predictions = ["买入", "持有", "卖出"]
        colors = {
            "买入": "#00FF99",
            "持有": "#FFCC33",
            "卖出": "#FF6666"
        }
        
        self.table.setRowCount(len(assets))
        
        for i, asset in enumerate(assets):
            # 资产名称
            self.table.setItem(i, 0, QTableWidgetItem(asset))
            
            # 纠缠度
            entanglement = random.uniform(0.1, 0.9)
            item = QTableWidgetItem(f"{entanglement:.2f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 1, item)
            
            # 共振状态
            resonance = random.uniform(-1.0, 1.0)
            item = QTableWidgetItem(f"{resonance:.2f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 2, item)
            
            # 预测
            pred_idx = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
            prediction = predictions[pred_idx]
            item = QTableWidgetItem(prediction)
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QBrush(QColor(colors[prediction])))
            self.table.setItem(i, 3, item)
            
            # 概率
            prob = random.uniform(0.55, 0.95)
            item = QTableWidgetItem(f"{prob:.1%}")
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 4, item)


class QuantumNetworkVisualizer(QWidget):
    """量子网络可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 设置背景色
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor('#0A0A1E'))
        self.setPalette(p)
        
        # 网络节点和连接
        self.nodes = []
        self.connections = []
        
        # 量子体系参数
        self.quantum_particles = []
        self.quantum_resonance = 0.0
        self.entanglement_count = 0
        self.coherence = 0.85
        
        # 临界量子阈值
        self.quantum_thresholds = {
            'entanglement': 0.42,  # 黄金分割比相关
            'decoherence': 0.05,
            'nonlocality': 0.78,
            'resonance': 0.61803  # 黄金分割比
        }
        
        # 高精度模式
        self.high_precision = True
        
        # 动画定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)
        
        self.phase = 0.0
        
        # 初始化节点和粒子
        self._init_quantum_system()
    
    def _init_quantum_system(self):
        """初始化量子系统元素"""
        # 创建节点
        self.nodes = []
        for i in range(8):
            angle = i * np.pi / 4
            self.nodes.append({
                'x': 0.5 + 0.35 * np.cos(angle),
                'y': 0.5 + 0.35 * np.sin(angle),
                'energy': 0.5 + 0.5 * np.random.random(),
                'phase': np.random.random() * 2 * np.pi
            })
            
        # 创建连接
        self.connections = []
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                # 只连接部分节点
                if np.random.random() < 0.6:
                    self.connections.append({
                        'source': i,
                        'target': j,
                        'strength': 0.3 + 0.7 * np.random.random(),
                        'phase': np.random.random() * 2 * np.pi
                    })
        
        # 创建量子粒子
        self.quantum_particles = []
        for _ in range(20):
            self.quantum_particles.append({
                'x': np.random.random(),
                'y': np.random.random(),
                'vx': 0.002 * (np.random.random() - 0.5),
                'vy': 0.002 * (np.random.random() - 0.5),
                'energy': 0.3 + 0.7 * np.random.random(),
                'size': 2 + 3 * np.random.random()
            })
    
    def update_animation(self):
        """更新动画状态"""
        self.phase += 0.05
        
        # 更新量子共振值
        if self.high_precision:
            # 使用更复杂的量子共振模型
            golden_ratio = 1.61803398875
            self.quantum_resonance = 0.3 + 0.2 * np.sin(self.phase * golden_ratio) + 0.1 * np.sin(self.phase * 2.71828)
            
            # 随机量子涨落
            from secrets import token_bytes
            quantum_fluctuation = int.from_bytes(token_bytes(2), byteorder='big') / 2**16 * 0.05
            self.quantum_resonance += quantum_fluctuation
        else:
            # 基本共振模型
            self.quantum_resonance = 0.4 + 0.3 * np.sin(self.phase)
        
        # 量子体系参数更新
        self.coherence = 0.7 + 0.15 * np.sin(self.phase * 0.2)
        
        # 检查是否达到临界值
        if self.quantum_resonance > self.quantum_thresholds['resonance']:
            # 临界共振状态
            self.entanglement_count = min(24, self.entanglement_count + 1)
        else:
            # 保持标准纠缠数量
            self.entanglement_count = max(12, self.entanglement_count - 1)
        
        # 更新节点
        for node in self.nodes:
            node['phase'] += 0.02 + 0.01 * self.quantum_resonance
            node['energy'] = 0.4 + 0.3 * np.sin(node['phase'])
        
        # 更新连接
        for conn in self.connections:
            conn['phase'] += 0.03
            conn['strength'] = 0.3 + 0.3 * np.sin(conn['phase']) + 0.2 * self.quantum_resonance
        
        # 更新量子粒子
        for particle in self.quantum_particles:
            # 移动粒子
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # 边界处理
            if particle['x'] < 0 or particle['x'] > 1:
                particle['vx'] *= -1
            if particle['y'] < 0 or particle['y'] > 1:
                particle['vy'] *= -1
            
            # 量子共振影响粒子能量
            particle['energy'] = 0.3 + 0.5 * np.sin(self.phase * particle['size'] / 3)
            if self.quantum_resonance > self.quantum_thresholds['resonance']:
                particle['energy'] += 0.2  # 临界状态能量提升
        
        # 更新视图
        self.update()
    
    def paintEvent(self, event):
        """绘制量子网络"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取组件尺寸
        width = self.width()
        height = self.height()
        
        # 绘制背景
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(10, 10, 40))
        gradient.setColorAt(1, QColor(5, 5, 20))
        painter.fillRect(0, 0, width, height, gradient)
        
        # 绘制量子场
        self._draw_quantum_field(painter, width, height)
        
        # 绘制节点和连接
        self._draw_nodes_and_connections(painter, width, height)
        
        # 绘制量子参数
        self._draw_quantum_parameters(painter, width, height)
        
        # 绘制临界阈值指示器
        if self.high_precision:
            self._draw_threshold_indicators(painter, width, height)
    
    def _draw_quantum_field(self, painter, width, height):
        """绘制量子场背景"""
        # 绘制网格线
        painter.setPen(QPen(QColor(30, 30, 80, 30), 1, Qt.SolidLine))
        
        # 横线
        step = height / 10
        for i in range(1, 10):
            y = int(i * step)  # 转换为整数
            painter.drawLine(0, y, int(width), y)
        
        # 竖线
        step = width / 15
        for i in range(1, 15):
            x = int(i * step)  # 转换为整数
            painter.drawLine(x, 0, x, int(height))
        
        # 绘制量子粒子
        for particle in self.quantum_particles:
            x = int(particle['x'] * width)  # 转换为整数
            y = int(particle['y'] * height)  # 转换为整数
            size = int(particle['size'])  # 转换为整数
            
            # 计算颜色
            energy = particle['energy']
            
            # 根据能量选择颜色
            if energy > 0.7:  # 高能量
                color = QColor(100, 200, 255, int(200 * energy))
            else:  # 低能量
                color = QColor(50, 100, 200, int(150 * energy))
            
            # 绘制粒子
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(x - size//2, y - size//2, size, size)
            
            # 绘制光晕
            radial = QRadialGradient(x, y, size * 3)
            radial.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 50))
            radial.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
            painter.setBrush(radial)
            painter.drawEllipse(x - size*3, y - size*3, size*6, size*6)
    
    def _draw_nodes_and_connections(self, painter, width, height):
        """绘制节点和连接"""
        # 先绘制连接
        for conn in self.connections:
            source = self.nodes[conn['source']]
            target = self.nodes[conn['target']]
            
            x1 = int(source['x'] * width)  # 转换为整数
            y1 = int(source['y'] * height)  # 转换为整数
            x2 = int(target['x'] * width)  # 转换为整数
            y2 = int(target['y'] * height)  # 转换为整数
            
            # 连接强度决定线宽和透明度
            strength = conn['strength']
            
            # 量子纠缠线 - 使用渐变色
            gradient = QLinearGradient(x1, y1, x2, y2)
            
            # 根据共振状态决定连接颜色
            if self.quantum_resonance > self.quantum_thresholds['resonance']:
                # 临界状态 - 更明亮的颜色
                gradient.setColorAt(0, QColor(0, 150, 255, int(200 * strength)))
                gradient.setColorAt(1, QColor(150, 100, 255, int(200 * strength)))
            else:
                # 正常状态
                gradient.setColorAt(0, QColor(0, 100, 200, int(150 * strength)))
                gradient.setColorAt(1, QColor(50, 50, 150, int(150 * strength)))
            
            painter.setPen(QPen(gradient, 1 + int(2 * strength), Qt.SolidLine))
            painter.drawLine(x1, y1, x2, y2)
        
        # 绘制节点
        for node in self.nodes:
            x = int(node['x'] * width)  # 转换为整数
            y = int(node['y'] * height)  # 转换为整数
            energy = node['energy']
            
            # 节点大小
            size = int(10 + 10 * energy)  # 转换为整数
            
            # 节点颜色随能量变化
            if self.quantum_resonance > self.quantum_thresholds['resonance']:
                # 临界状态下的颜色
                color = QColor(50, 100 + int(155 * energy), 200 + int(55 * energy))
            else:
                # 正常状态下的颜色
                color = QColor(30, 50 + int(100 * energy), 150 + int(105 * energy))
            
            # 绘制节点
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.setBrush(color)
            painter.drawEllipse(x - size//2, y - size//2, size, size)
            
            # 绘制内环
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
            painter.setBrush(Qt.NoBrush)
            inner_size = int(size * 0.7)  # 转换为整数
            painter.drawEllipse(x - inner_size//2, y - inner_size//2, inner_size, inner_size)
    
    def _draw_quantum_parameters(self, painter, width, height):
        """绘制量子参数信息"""
        # 在左上角绘制主要参数
        margin = 15
        font = QFont("Arial", 8)
        painter.setFont(font)
        
        # 计算颜色
        if self.quantum_resonance > self.quantum_thresholds['resonance']:
            # 临界状态 - 使用更明亮的颜色
            text_color = QColor(100, 255, 255)
        else:
            # 正常状态
            text_color = QColor(100, 200, 230)
        
        painter.setPen(text_color)
        painter.drawText(margin, margin + 12, f"共振强度: {self.quantum_resonance:.3f}")
        painter.drawText(margin, margin + 27, f"相干度: {self.coherence:.2f}")
        painter.drawText(margin, margin + 42, f"纠缠数量: {self.entanglement_count}")
    
    def _draw_threshold_indicators(self, painter, width, height):
        """绘制临界阈值指示器"""
        margin = 10
        indicator_height = 20
        indicator_width = width - 2 * margin
        y_pos = int(height - margin - indicator_height)  # 转换为整数
        
        # 绘制阈值条背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(30, 30, 60, 150))
        painter.drawRoundedRect(margin, y_pos, int(indicator_width), indicator_height, 5, 5)
        
        # 绘制当前共振值
        resonance_width = int(indicator_width * self.quantum_resonance)  # 转换为整数
        
        # 创建渐变色
        if self.quantum_resonance > self.quantum_thresholds['resonance']:
            # 临界状态 - 高能量
            gradient = QLinearGradient(margin, 0, margin + resonance_width, 0)
            gradient.setColorAt(0, QColor(0, 200, 255, 200))
            gradient.setColorAt(1, QColor(200, 100, 255, 200))
        else:
            # 正常状态
            gradient = QLinearGradient(margin, 0, margin + resonance_width, 0)
            gradient.setColorAt(0, QColor(0, 150, 200, 200))
            gradient.setColorAt(1, QColor(0, 200, 150, 200))
        
        painter.setBrush(gradient)
        painter.drawRoundedRect(margin, y_pos, resonance_width, indicator_height, 5, 5)
        
        # 绘制临界阈值线
        threshold_x = int(margin + indicator_width * self.quantum_thresholds['resonance'])  # 转换为整数
        painter.setPen(QPen(QColor(255, 220, 50, 200), 2, Qt.DashLine))
        painter.drawLine(threshold_x, y_pos - 3, threshold_x, y_pos + indicator_height + 3)
        
        # 绘制阈值标签
        painter.setPen(QColor(255, 220, 50))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(threshold_x - 25, y_pos - 5, "临界阈值")
        
        # 绘制当前值
        painter.setPen(QColor(220, 220, 255))
        value_text = f"量子共振: {self.quantum_resonance:.3f}"
        painter.drawText(margin + 5, y_pos + indicator_height - 5, value_text)


class QuantumViewWidget(QWidget):
    """量子视图主组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 顶部标题
        title_layout = QHBoxLayout()
        
        title = QLabel("量子引擎 - 预测与纠缠分析")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #00DDFF;")
        title_layout.addWidget(title)
        
        # 状态指示器
        self.status_label = QLabel("● 运行中")
        self.status_label.setStyleSheet("color: #00FF99; font-weight: bold;")
        title_layout.addWidget(self.status_label)
        
        title_layout.addStretch()
        
        # 更新时间
        self.time_label = QLabel(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
        self.time_label.setStyleSheet("color: #AAAAEE;")
        title_layout.addWidget(self.time_label)
        
        layout.addLayout(title_layout)
        
        # 添加控制面板
        self.control_panel = QuantumControlPanel()
        layout.addWidget(self.control_panel)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        
        # 添加状态可视化选项卡
        states_tab = QWidget()
        states_layout = QVBoxLayout(states_tab)
        self.state_viz = QuantumStateVisualizer()
        states_layout.addWidget(self.state_viz)
        self.tabs.addTab(states_tab, "量子态")
        
        # 添加纠缠图选项卡
        entangle_tab = QWidget()
        entangle_layout = QVBoxLayout(entangle_tab)
        self.entangle_viz = EntanglementGraphWidget()
        entangle_layout.addWidget(self.entangle_viz)
        self.tabs.addTab(entangle_tab, "纠缠网络")
        
        # 添加选项卡到布局
        layout.addWidget(self.tabs)
        
        # 添加结果表格
        self.results = QuantumResultsWidget()
        layout.addWidget(self.results)
        
        # 更新定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)  # 每秒更新一次
        
        # 连接按钮信号
        self.control_panel.run_button.clicked.connect(self._run_calculation)
        self.control_panel.stop_button.clicked.connect(self._stop_calculation)
        self.control_panel.reset_button.clicked.connect(self._reset_system)
    
    def _update_time(self):
        """更新时间"""
        self.time_label.setText(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
    
    def _run_calculation(self):
        """运行计算"""
        # 模拟计算进度
        self.calculation_timer = QTimer(self)
        self.calculation_timer.timeout.connect(self._update_calculation)
        self.calculation_timer.start(100)
        
        # 禁用运行按钮
        self.control_panel.run_button.setEnabled(False)
        self.control_panel.reset_button.setEnabled(False)
    
    def _update_calculation(self):
        """更新计算进度"""
        current = self.control_panel.progress.value()
        
        if current < 100:
            self.control_panel.progress.setValue(current + 1)
        else:
            self.calculation_timer.stop()
            self.control_panel.run_button.setEnabled(True)
            self.control_panel.reset_button.setEnabled(True)
            
            # 更新结果
            self.results._populate_random_data()
    
    def _stop_calculation(self):
        """停止计算"""
        if hasattr(self, 'calculation_timer') and self.calculation_timer.isActive():
            self.calculation_timer.stop()
            self.control_panel.run_button.setEnabled(True)
            self.control_panel.reset_button.setEnabled(True)
    
    def _reset_system(self):
        """重置系统"""
        self.control_panel.progress.setValue(0)
        self.status_label.setText("● 就绪")
        self.status_label.setStyleSheet("color: #FFAA33; font-weight: bold;")


def create_quantum_view(parent=None):
    """创建量子视图"""
    frame = QFrame(parent)
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setStyleSheet("""
        QFrame {
            background-color: #070720;
            border: 1px solid #1A1A40;
            border-radius: 8px;
        }
    """)
    
    # 添加阴影效果
    shadow = QGraphicsDropShadowEffect(frame)
    shadow.setBlurRadius(15)
    shadow.setColor(QColor(0, 0, 0, 80))
    shadow.setOffset(0, 0)
    frame.setGraphicsEffect(shadow)
    
    # 布局
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(15, 15, 15, 15)
    
    # 添加标题
    title = QLabel("量子纠缠网络实时视图")
    title.setStyleSheet("color: #00DDFF; font-size: 18px; font-weight: bold;")
    layout.addWidget(title)
    
    # 添加高精度模式标签
    precision_label = QLabel("[ 高精度量子模式 ]")
    precision_label.setStyleSheet("color: #50FFDD; font-size: 12px;")
    precision_label.setAlignment(Qt.AlignRight)
    layout.addWidget(precision_label)
    
    # 创建量子网络可视化
    quantum_visualizer = QuantumNetworkVisualizer()
    quantum_visualizer.setMinimumHeight(200)
    layout.addWidget(quantum_visualizer)
    
    # 添加量子网络信息面板
    info_frame = QFrame()
    info_frame.setStyleSheet("background-color: #0A0A2A; border-radius: 5px;")
    info_layout = QVBoxLayout(info_frame)
    
    # 添加关键指标
    metrics_layout = QHBoxLayout()
    
    # 量子纠缠对数
    entanglement_metric = create_metric_widget("纠缠实体对", "28", "#00DDFF")
    metrics_layout.addWidget(entanglement_metric)
    
    # 量子相干度
    coherence_metric = create_metric_widget("量子相干性", "87.2%", "#00FFAA")
    metrics_layout.addWidget(coherence_metric)
    
    # 非局域性强度
    nonlocality_metric = create_metric_widget("非局域强度", "高", "#AA99FF")
    metrics_layout.addWidget(nonlocality_metric)
    
    # 临界熵值
    entropy_metric = create_metric_widget("临界熵", "0.458", "#FFAA33")
    metrics_layout.addWidget(entropy_metric)
    
    info_layout.addLayout(metrics_layout)
    
    # 添加提示信息
    tip_label = QLabel("系统运行在量子共振预测模式，超临界状态将生成高精度市场预测")
    tip_label.setStyleSheet("color: #AAAAEE; font-size: 11px; font-style: italic;")
    tip_label.setAlignment(Qt.AlignCenter)
    info_layout.addWidget(tip_label)
    
    layout.addWidget(info_frame)
    
    return frame

def create_metric_widget(title, value, color):
    """创建指标小部件"""
    widget = QFrame()
    widget.setStyleSheet(f"border: 1px solid {color}; border-radius: 5px;")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    
    title_label = QLabel(title)
    title_label.setStyleSheet("color: #BBBBEE; font-size: 11px;")
    title_label.setAlignment(Qt.AlignCenter)
    
    value_label = QLabel(value)
    value_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
    value_label.setAlignment(Qt.AlignCenter)
    
    layout.addWidget(title_label)
    layout.addWidget(value_label)
    
    widget.setFixedHeight(70)
    
    return widget 