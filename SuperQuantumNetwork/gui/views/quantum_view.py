#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 超神级量子网络视图
实现高级量子网络可视化和控制功能
"""

import logging
import traceback
import random
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QGroupBox, QFormLayout, QSlider, QSpinBox,
    QTabWidget, QTextEdit, QSplitter, QFrame, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient

# 尝试导入可能缺失的依赖
try:
    import numpy as np
    import pandas as pd
    import pyqtgraph as pg
    DEPS_AVAILABLE = True
except ImportError as e:
    logging.error(f"量子网络视图依赖导入错误: {str(e)}\n{traceback.format_exc()}")
    DEPS_AVAILABLE = False

# 尝试导入qtawesome，如果没有则使用备选图标
try:
    import qtawesome as qta
    QTA_AVAILABLE = True
except ImportError:
    logging.warning("qtawesome库未找到，将使用简化图标")
    QTA_AVAILABLE = False

# 只有在依赖可用时才定义完整功能的类
if DEPS_AVAILABLE:
    class QuantumNetworkNode:
        """量子网络节点"""
        
        def __init__(self, node_id, type_name, importance=1.0):
            self.id = node_id
            self.type = type_name  # input, hidden, output
            self.importance = importance
            self.connections = []
            self.value = 0.0
            self.position = np.array([0.0, 0.0])
            
            # 节点可视化属性
            self.size = 10.0 + importance * 5.0
            if type_name == "input":
                self.color = (30, 144, 255)  # 蓝色
            elif type_name == "hidden":
                self.color = (255, 255, 255)  # 白色
            else:
                self.color = (255, 50, 50)    # 红色


    class QuantumNetworkConnection:
        """量子网络连接"""
        
        def __init__(self, source, target, weight=0.0):
            self.source = source
            self.target = target
            self.weight = weight
            
            # 连接可视化属性
            self.width = 1.0 + abs(weight) * 2.0
            if weight > 0:
                self.color = (0, 255, 0, 150)  # 绿色表示正影响
            else:
                self.color = (255, 0, 0, 150)  # 红色表示负影响


    class QuantumNetworkGraph(pg.GraphicsLayoutWidget):
        """量子网络图形组件"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            # 创建示例网络数据
            self.nodes = []
            self.connections = []
            self._generate_sample_network()
            
            # 初始化UI
            self._setup_ui()
            
            # 创建动画定时器
            self.animation_timer = QTimer(self)
            self.animation_timer.timeout.connect(self._update_network)
            self.animation_timer.start(50)  # 每50ms更新一次
        
        def _generate_sample_network(self):
            """生成样本网络"""
            # 创建节点
            input_nodes = []
            for i in range(5):
                node = QuantumNetworkNode(i, "input", random.uniform(0.8, 1.2))
                node.position = np.array([100, 100 + i * 60])
                node.value = random.uniform(-1, 1)
                self.nodes.append(node)
                input_nodes.append(node)
            
            hidden_nodes = []
            for i in range(8):
                node = QuantumNetworkNode(i + 5, "hidden", random.uniform(0.8, 1.2))
                node.position = np.array([250, 70 + i * 45])
                node.value = random.uniform(-1, 1)
                self.nodes.append(node)
                hidden_nodes.append(node)
            
            output_nodes = []
            for i in range(3):
                node = QuantumNetworkNode(i + 13, "output", random.uniform(0.8, 1.2))
                node.position = np.array([400, 130 + i * 80])
                node.value = random.uniform(-1, 1)
                self.nodes.append(node)
                output_nodes.append(node)
            
            # 创建连接
            # 输入层到隐藏层
            for input_node in input_nodes:
                for hidden_node in hidden_nodes:
                    if random.random() < 0.7:  # 70%概率创建连接
                        weight = random.uniform(-1, 1)
                        conn = QuantumNetworkConnection(input_node, hidden_node, weight)
                        self.connections.append(conn)
                        input_node.connections.append(conn)
            
            # 隐藏层到输出层
            for hidden_node in hidden_nodes:
                for output_node in output_nodes:
                    if random.random() < 0.7:  # 70%概率创建连接
                        weight = random.uniform(-1, 1)
                        conn = QuantumNetworkConnection(hidden_node, output_node, weight)
                        self.connections.append(conn)
                        hidden_node.connections.append(conn)
        
        def _setup_ui(self):
            """设置UI"""
            # 设置背景为黑色
            self.setBackground('k')
            
            # 创建绘图项
            self.plot = self.addPlot()
            self.plot.setAspectLocked(False)
            self.plot.hideAxis('left')
            self.plot.hideAxis('bottom')
            self.plot.setRange(xRange=(0, 500), yRange=(0, 400))
            
            # 创建节点散点图
            self.node_scatter = pg.ScatterPlotItem()
            self.plot.addItem(self.node_scatter)
            
            # 创建边线项
            self.edges = []
            
            # 更新可视化
            self._update_visualization()
        
        def _update_visualization(self):
            """更新可视化"""
            # 清除旧的边
            for edge in self.edges:
                self.plot.removeItem(edge)
            self.edges.clear()
            
            # 绘制新的边
            for conn in self.connections:
                source_pos = conn.source.position
                target_pos = conn.target.position
                
                # 创建连接线
                edge = pg.PlotCurveItem(
                    x=[source_pos[0], target_pos[0]],
                    y=[source_pos[1], target_pos[1]],
                    pen=pg.mkPen(color=conn.color, width=conn.width)
                )
                self.plot.addItem(edge)
                self.edges.append(edge)
            
            # 更新节点
            positions = np.array([node.position for node in self.nodes])
            sizes = np.array([node.size for node in self.nodes])
            brushes = [node.color for node in self.nodes]
            
            self.node_scatter.setData(pos=positions, size=sizes, brush=brushes, pxMode=True)
            
            # 更新节点标签
            for node in self.nodes:
                # 计算显示值
                display_value = "{:.2f}".format(node.value)
                
                # 查找现有标签或创建新标签
                label_found = False
                for item in self.plot.items:
                    if isinstance(item, pg.TextItem) and hasattr(item, 'node_id') and item.node_id == node.id:
                        item.setText(display_value)
                        item.setPos(node.position[0], node.position[1] + node.size / 2 + 5)
                        label_found = True
                        break
                
                if not label_found:
                    text = pg.TextItem(text=display_value, color='w', anchor=(0.5, 0))
                    text.node_id = node.id
                    text.setPos(node.position[0], node.position[1] + node.size / 2 + 5)
                    self.plot.addItem(text)
        
        def _update_network(self):
            """更新网络动画"""
            # 更新节点值
            for node in self.nodes:
                # 添加一些随机波动
                node.value += random.uniform(-0.1, 0.1)
                node.value = max(-1.0, min(1.0, node.value))  # 限制在[-1, 1]范围内
                
                # 根据值调整大小
                value_factor = 1.0 + abs(node.value) * 0.5
                node.size = (10.0 + node.importance * 5.0) * value_factor
            
            # 更新连接权重
            for conn in self.connections:
                # 偶尔随机调整权重
                if random.random() < 0.05:  # 5%概率更新
                    conn.weight += random.uniform(-0.1, 0.1)
                    conn.weight = max(-1.0, min(1.0, conn.weight))  # 限制在[-1, 1]范围内
                    
                    # 更新连接可视化属性
                    conn.width = 1.0 + abs(conn.weight) * 2.0
                    if conn.weight > 0:
                        conn.color = (0, 255, 0, 100 + int(abs(conn.weight) * 150))  # 绿色表示正影响
                    else:
                        conn.color = (255, 0, 0, 100 + int(abs(conn.weight) * 150))  # 红色表示负影响
            
            # 更新可视化
            self._update_visualization()


    class QuantumNetworkControls(QWidget):
        """量子网络控制面板"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._setup_ui()
        
        def _setup_ui(self):
            """设置UI"""
            # 创建主布局
            main_layout = QVBoxLayout(self)
            
            # 创建状态组
            status_group = QGroupBox("网络状态")
            status_layout = QFormLayout(status_group)
            
            self.status_label = QLabel("运行中")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            status_layout.addRow("状态:", self.status_label)
            
            self.evolution_label = QLabel("第8代")
            status_layout.addRow("进化阶段:", self.evolution_label)
            
            self.fitness_label = QLabel("92.7%")
            status_layout.addRow("适应度:", self.fitness_label)
            
            self.symmetry_label = QLabel("高")
            status_layout.addRow("量子对称性:", self.symmetry_label)
            
            self.entanglement_label = QLabel("78.3%")
            status_layout.addRow("量子纠缠度:", self.entanglement_label)
            
            # 添加状态组到主布局
            main_layout.addWidget(status_group)
            
            # 创建参数组
            params_group = QGroupBox("网络参数")
            params_layout = QFormLayout(params_group)
            
            self.learning_rate_slider = QSlider(Qt.Horizontal)
            self.learning_rate_slider.setRange(1, 100)
            self.learning_rate_slider.setValue(25)
            params_layout.addRow("学习率:", self.learning_rate_slider)
            
            self.mutation_rate_slider = QSlider(Qt.Horizontal)
            self.mutation_rate_slider.setRange(1, 100)
            self.mutation_rate_slider.setValue(15)
            params_layout.addRow("变异率:", self.mutation_rate_slider)
            
            self.coherence_slider = QSlider(Qt.Horizontal)
            self.coherence_slider.setRange(1, 100)
            self.coherence_slider.setValue(80)
            params_layout.addRow("量子相干性:", self.coherence_slider)
            
            self.superposition_slider = QSlider(Qt.Horizontal)
            self.superposition_slider.setRange(1, 100)
            self.superposition_slider.setValue(70)
            params_layout.addRow("叠加态强度:", self.superposition_slider)
            
            # 添加参数组到主布局
            main_layout.addWidget(params_group)
            
            # 创建操作按钮组
            actions_group = QGroupBox("网络操作")
            actions_layout = QVBoxLayout(actions_group)
            
            self.reset_button = QPushButton("重置网络")
            self.reset_button.setIcon(qta.icon('fa5s.redo'))
            actions_layout.addWidget(self.reset_button)
            
            self.optimize_button = QPushButton("优化参数")
            self.optimize_button.setIcon(qta.icon('fa5s.magic'))
            actions_layout.addWidget(self.optimize_button)
            
            self.save_button = QPushButton("保存网络")
            self.save_button.setIcon(qta.icon('fa5s.save'))
            actions_layout.addWidget(self.save_button)
            
            # 添加操作组到主布局
            main_layout.addWidget(actions_group)
            
            # 添加弹性空间
            main_layout.addStretch(1)


    class QuantumNetworkMonitor(QWidget):
        """量子网络监控面板"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            # 初始化日志数据
            self.log_data = [
                (datetime.now().strftime("%H:%M:%S"), "量子网络初始化完成", "信息"),
                (datetime.now().strftime("%H:%M:%S"), "检测到隐藏市场模式", "信息"),
                (datetime.now().strftime("%H:%M:%S"), "量子纠缠度提升到78.3%", "信息"),
                (datetime.now().strftime("%H:%M:%S"), "发现潜在交易机会: 科技板块", "交易"),
                (datetime.now().strftime("%H:%M:%S"), "预测未来4小时市场波动", "预测"),
                (datetime.now().strftime("%H:%M:%S"), "自适应参数优化完成", "系统")
            ]
            
            # 设置UI
            self._setup_ui()
            
            # 设置定时器，模拟实时日志更新
            self.log_timer = QTimer(self)
            self.log_timer.timeout.connect(self._add_random_log)
            self.log_timer.start(5000)  # 每5秒添加一条日志
        
        def _setup_ui(self):
            """设置UI"""
            # 主布局
            main_layout = QVBoxLayout(self)
            
            # 创建日志表格
            self.log_table = QTableWidget()
            self.log_table.setColumnCount(3)
            self.log_table.setHorizontalHeaderLabels(["时间", "事件", "类型"])
            self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.log_table.setSelectionBehavior(QTableWidget.SelectRows)
            
            # 填充日志数据
            self._update_log_table()
            
            # 添加表格到布局
            main_layout.addWidget(self.log_table)
            
            # 创建性能指标组
            metrics_group = QGroupBox("网络性能指标")
            metrics_layout = QGridLayout(metrics_group)
            
            # 添加各种性能指标
            metrics = [
                ("预测准确率", "92.7%", "green"),
                ("量子状态稳定性", "高", "green"),
                ("市场捕捉率", "87.3%", "green"),
                ("决策延迟", "12ms", "blue"),
                ("误报率", "2.3%", "orange"),
                ("学习速度", "快", "green")
            ]
            
            for i, (name, value, color) in enumerate(metrics):
                row, col = i // 3, i % 3
                
                label = QLabel(name + ":")
                value_label = QLabel(value)
                value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
                
                metrics_layout.addWidget(label, row, col*2)
                metrics_layout.addWidget(value_label, row, col*2+1)
            
            # 添加性能指标组到主布局
            main_layout.addWidget(metrics_group)
        
        def _update_log_table(self):
            """更新日志表格"""
            self.log_table.setRowCount(len(self.log_data))
            
            for i, (time, event, event_type) in enumerate(self.log_data):
                self.log_table.setItem(i, 0, QTableWidgetItem(time))
                self.log_table.setItem(i, 1, QTableWidgetItem(event))
                
                type_item = QTableWidgetItem(event_type)
                if event_type == "交易":
                    type_item.setForeground(QColor("orange"))
                elif event_type == "预测":
                    type_item.setForeground(QColor("cyan"))
                elif event_type == "警告":
                    type_item.setForeground(QColor("yellow"))
                elif event_type == "错误":
                    type_item.setForeground(QColor("red"))
                
                self.log_table.setItem(i, 2, type_item)
            
            # 滚动到最新记录
            self.log_table.scrollToBottom()
        
        def _add_random_log(self):
            """添加随机日志条目"""
            event_types = ["信息", "交易", "预测", "系统", "警告"]
            events = [
                "检测到市场异常波动",
                "发现做市商行为模式",
                "预测指数短期反弹",
                "自动调整量子相干性",
                "股票相关性矩阵更新",
                "完成市场微观结构分析",
                "发现新交易机会",
                "开始深度学习环节",
                "量子信号强度增加",
                "完成交易策略评估"
            ]
            
            new_event = random.choice(events)
            new_type = random.choice(event_types)
            new_time = datetime.now().strftime("%H:%M:%S")
            
            self.log_data.append((new_time, new_event, new_type))
            
            # 保持日志长度合理
            if len(self.log_data) > 100:
                self.log_data.pop(0)
            
            self._update_log_table()


    class QuantumPredictionView(QWidget):
        """量子预测视图"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._setup_ui()
        
        def _setup_ui(self):
            """设置UI"""
            # 主布局
            main_layout = QVBoxLayout(self)
            
            # 创建预测图表
            self.prediction_plot = pg.PlotWidget()
            self.prediction_plot.setBackground('k')
            self.prediction_plot.setLabel('left', '价格')
            self.prediction_plot.setLabel('bottom', '时间')
            self.prediction_plot.showGrid(x=True, y=True, alpha=0.3)
            
            # 添加图表到布局
            main_layout.addWidget(self.prediction_plot)
            
            # 创建样本数据
            self._create_sample_data()
            
            # 绘制图表
            self._plot_prediction_data()
            
            # 添加预测结果表格
            results_group = QGroupBox("量子预测结果")
            results_layout = QVBoxLayout(results_group)
            
            self.results_table = QTableWidget()
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels(["时间范围", "预测趋势", "置信度", "建议"])
            self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            
            # 填充预测结果
            predictions = [
                ("短期 (1小时)", "上涨", "87.3%", "观望"),
                ("中期 (1天)", "震荡上行", "76.5%", "少量买入"),
                ("长期 (1周)", "上涨", "92.1%", "买入")
            ]
            
            self.results_table.setRowCount(len(predictions))
            for i, (timeframe, trend, confidence, advice) in enumerate(predictions):
                self.results_table.setItem(i, 0, QTableWidgetItem(timeframe))
                
                trend_item = QTableWidgetItem(trend)
                if "上涨" in trend:
                    trend_item.setForeground(QColor("red"))
                elif "下跌" in trend:
                    trend_item.setForeground(QColor("green"))
                self.results_table.setItem(i, 1, trend_item)
                
                confidence_item = QTableWidgetItem(confidence)
                confidence_val = float(confidence.strip('%')) / 100
                if confidence_val > 0.8:
                    confidence_item.setForeground(QColor("lime"))
                elif confidence_val > 0.6:
                    confidence_item.setForeground(QColor("yellow"))
                else:
                    confidence_item.setForeground(QColor("orange"))
                self.results_table.setItem(i, 2, confidence_item)
                
                advice_item = QTableWidgetItem(advice)
                if advice == "买入":
                    advice_item.setForeground(QColor("red"))
                elif advice == "卖出":
                    advice_item.setForeground(QColor("green"))
                self.results_table.setItem(i, 3, advice_item)
            
            results_layout.addWidget(self.results_table)
            main_layout.addWidget(results_group)
            
            # 调整比例
            main_layout.setStretchFactor(self.prediction_plot, 7)
            main_layout.setStretchFactor(results_group, 3)
        
        def _create_sample_data(self):
            """创建样本数据"""
            # 历史数据点
            self.x_data = np.arange(100)
            self.y_data = np.cumsum(np.random.normal(0, 1, 100)) + 100
            
            # 预测数据点
            self.x_pred = np.arange(100, 130)
            
            # 模拟三种不同的预测路径及其概率
            self.y_pred_up = self.y_data[-1] + np.cumsum(np.random.normal(0.2, 1, 30))
            self.y_pred_flat = self.y_data[-1] + np.cumsum(np.random.normal(0, 1, 30))
            self.y_pred_down = self.y_data[-1] + np.cumsum(np.random.normal(-0.2, 1, 30))
            
            # 每条路径的概率
            self.prob_up = 0.60    # 60%概率上涨
            self.prob_flat = 0.25  # 25%概率横盘
            self.prob_down = 0.15  # 15%概率下跌
        
        def _plot_prediction_data(self):
            """绘制预测数据"""
            # 清除图表
            self.prediction_plot.clear()
            
            # 历史数据
            history_curve = pg.PlotDataItem(
                self.x_data, self.y_data, 
                pen=pg.mkPen(color=(255, 255, 255), width=2)
            )
            self.prediction_plot.addItem(history_curve)
            
            # 预测数据 - 上涨路径
            up_curve = pg.PlotDataItem(
                self.x_pred, self.y_pred_up,
                pen=pg.mkPen(color=(255, 0, 0, 150), width=2, style=Qt.DashLine)
            )
            self.prediction_plot.addItem(up_curve)
            
            # 添加上涨概率标签
            up_label = pg.TextItem(
                text=f"上涨 ({self.prob_up*100:.1f}%)", 
                color=(255, 0, 0),
                anchor=(0, 0)
            )
            up_label.setPos(self.x_pred[-1], self.y_pred_up[-1])
            self.prediction_plot.addItem(up_label)
            
            # 预测数据 - 横盘路径
            flat_curve = pg.PlotDataItem(
                self.x_pred, self.y_pred_flat,
                pen=pg.mkPen(color=(255, 255, 0, 150), width=2, style=Qt.DashLine)
            )
            self.prediction_plot.addItem(flat_curve)
            
            # 添加横盘概率标签
            flat_label = pg.TextItem(
                text=f"震荡 ({self.prob_flat*100:.1f}%)", 
                color=(255, 255, 0),
                anchor=(0, 0.5)
            )
            flat_label.setPos(self.x_pred[-1], self.y_pred_flat[-1])
            self.prediction_plot.addItem(flat_label)
            
            # 预测数据 - 下跌路径
            down_curve = pg.PlotDataItem(
                self.x_pred, self.y_pred_down,
                pen=pg.mkPen(color=(0, 255, 0, 150), width=2, style=Qt.DashLine)
            )
            self.prediction_plot.addItem(down_curve)
            
            # 添加下跌概率标签
            down_label = pg.TextItem(
                text=f"下跌 ({self.prob_down*100:.1f}%)", 
                color=(0, 255, 0),
                anchor=(0, 1)
            )
            down_label.setPos(self.x_pred[-1], self.y_pred_down[-1])
            self.prediction_plot.addItem(down_label)
            
            # 在历史和预测的交界处添加垂直线
            vline = pg.InfiniteLine(
                pos=self.x_data[-1], 
                angle=90, 
                pen=pg.mkPen(color=(255, 255, 255, 100), width=1, style=Qt.DashLine)
            )
            self.prediction_plot.addItem(vline)
            
            # 添加当前时间标记
            now_label = pg.TextItem(
                text="当前", 
                color=(255, 255, 255),
                anchor=(0.5, 1)
            )
            now_label.setPos(self.x_data[-1], self.y_data[-1] - 5)
            self.prediction_plot.addItem(now_label)


    class SuperQuantumNetworkView(QWidget):
        """超级量子网络视图"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            # 检查依赖是否可用
            if DEPS_AVAILABLE:
                self._setup_advanced_ui()
            else:
                self._setup_basic_ui()
        
        def _setup_basic_ui(self):
            """设置基本UI (依赖不可用时)"""
            main_layout = QVBoxLayout(self)
            
            # 标题标签
            title_label = QLabel("超神量子网络")
            title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: cyan;")
            title_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title_label)
            
            # 添加信息标签
            info_label = QLabel("简化版超神量子网络视图")
            info_label.setStyleSheet("font-size: 16px;")
            info_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(info_label)
            
            # 添加安装依赖提示
            tip_label = QLabel("请安装依赖以获取完整功能：\npip install pyqtgraph numpy pandas")
            tip_label.setStyleSheet("color: yellow;")
            tip_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(tip_label)
            
            # 添加模拟数据视图
            sim_frame = QFrame()
            sim_frame.setFrameShape(QFrame.StyledPanel)
            sim_frame.setStyleSheet("background-color: #1a1a1a;")
            sim_layout = QVBoxLayout(sim_frame)
            
            sim_status = QLabel("量子网络已初始化")
            sim_status.setStyleSheet("color: green; font-weight: bold;")
            sim_layout.addWidget(sim_status)
            
            sim_progress = QProgressBar()
            sim_progress.setRange(0, 100)
            sim_progress.setValue(87)
            sim_progress.setFormat("量子纠缠度: %p%")
            sim_layout.addWidget(sim_progress)
            
            # 一些基本的状态信息
            status_group = QGroupBox("网络状态")
            status_layout = QFormLayout(status_group)
            
            status_layout.addRow("状态:", QLabel("运行中"))
            status_layout.addRow("预测准确率:", QLabel("92.7%"))
            status_layout.addRow("量子状态:", QLabel("高度相干"))
            
            sim_layout.addWidget(status_group)
            main_layout.addWidget(sim_frame)
            
            # 添加按钮
            button_layout = QHBoxLayout()
            refresh_button = QPushButton("刷新")
            optimize_button = QPushButton("优化")
            button_layout.addWidget(refresh_button)
            button_layout.addWidget(optimize_button)
            main_layout.addLayout(button_layout)
            
            # 添加弹性空间
            main_layout.addStretch(1)
        
        def _setup_advanced_ui(self):
            """设置高级UI (依赖可用时)"""
            # 主布局
            main_layout = QVBoxLayout(self)
            
            # 创建顶部控件
            top_layout = QHBoxLayout()
            
            self.view_label = QLabel("超神量子网络视图")
            self.view_label.setStyleSheet("font-size: 16px; font-weight: bold; color: cyan;")
            top_layout.addWidget(self.view_label)
            
            self.view_combo = QComboBox()
            self.view_combo.addItems(["标准视图", "深度视图", "量子纠缠视图", "交易预测视图"])
            top_layout.addWidget(self.view_combo)
            
            top_layout.addStretch(1)
            
            self.refresh_button = QPushButton("刷新")
            if QTA_AVAILABLE:
                self.refresh_button.setIcon(qta.icon('fa5s.sync'))
            top_layout.addWidget(self.refresh_button)
            
            main_layout.addLayout(top_layout)
            
            # 创建主分割器
            self.main_splitter = QSplitter(Qt.Horizontal)
            
            # 左侧为网络图
            self.network_graph = QuantumNetworkGraph()
            self.main_splitter.addWidget(self.network_graph)
            
            # 右侧为控制面板
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            
            # 添加控制面板
            self.network_controls = QuantumNetworkControls()
            right_layout.addWidget(self.network_controls)
            
            # 添加监控面板
            self.network_monitor = QuantumNetworkMonitor()
            right_layout.addWidget(self.network_monitor)
            
            self.main_splitter.addWidget(right_panel)
            
            # 设置分割器初始大小
            self.main_splitter.setSizes([700, 400])
            
            # 添加主分割器到布局
            main_layout.addWidget(self.main_splitter)
            
            # 创建底部状态面板
            status_panel = QFrame()
            status_panel.setFrameShape(QFrame.StyledPanel)
            status_panel.setMaximumHeight(100)
            status_layout = QHBoxLayout(status_panel)
            
            # 添加预测视图
            self.prediction_view = QuantumPredictionView()
            
            # 添加预测标签
            prediction_label = QLabel("当前预测: 市场将在2小时内上涨1.7%")
            prediction_label.setStyleSheet("color: green; font-weight: bold;")
            status_layout.addWidget(prediction_label)
            
            # 添加信号强度
            signal_label = QLabel("量子信号强度:")
            status_layout.addWidget(signal_label)
            
            signal_bar = QProgressBar()
            signal_bar.setRange(0, 100)
            signal_bar.setValue(87)
            signal_bar.setMaximumWidth(150)
            status_layout.addWidget(signal_bar)
            
            # 添加置信度
            confidence_label = QLabel("置信度: 92.7%")
            confidence_label.setStyleSheet("color: cyan;")
            status_layout.addWidget(confidence_label)
            
            status_layout.addStretch(1)
            
            # 添加时间戳
            time_label = QLabel(datetime.now().strftime("最后更新: %Y-%m-%d %H:%M:%S"))
            status_layout.addWidget(time_label)
            
            # 添加状态面板到主布局
            main_layout.addWidget(status_panel)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    view = SuperQuantumNetworkView()
    view.setStyleSheet("""
        QWidget {
            background-color: #1a1a2e;
            color: #e6e6e6;
        }
        QTabWidget::pane {
            border: 1px solid #3f3f5a;
            background-color: #252538;
        }
        QTabBar::tab {
            background-color: #1a1a2e;
            color: #e6e6e6;
            padding: 8px 16px;
            border: 1px solid #3f3f5a;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #252538;
            border-bottom: none;
        }
        QTabBar::tab:hover {
            background-color: #30304d;
        }
        QGroupBox {
            border: 1px solid #3f3f5a;
            border-radius: 5px;
            margin-top: 1em;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            padding: 0px 5px;
        }
    """)
    view.resize(1200, 800)
    view.show()
    
    sys.exit(app.exec_()) 