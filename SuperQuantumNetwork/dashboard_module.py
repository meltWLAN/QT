#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 高级仪表盘模块
提供系统核心指标和市场全景可视化
"""

import sys
import os
import logging
import traceback
import random
import math
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout,
    QFrame, QGraphicsDropShadowEffect, QSizePolicy, QSpacerItem, QProgressBar
)
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QRectF, QPoint, QPointF
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


class InfoCard(QFrame):
    """信息卡片组件"""
    
    def __init__(self, title, value, icon=None, color="#00AAEE", parent=None):
        super().__init__(parent)
        
        # 设置样式
        self.setObjectName("infoCard")
        self.setStyleSheet(f"""
            #infoCard {{
                background-color: #0F1028;
                border: 1px solid {color};
                border-radius: 8px;
            }}
            
            #cardTitle {{
                color: #AAAAEE;
                font-size: 12px;
            }}
            
            #cardValue {{
                color: {color};
                font-size: 24px;
                font-weight: bold;
            }}
            
            #cardTrend {{
                color: #BBBBBB;
                font-size: 11px;
            }}
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 70))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # 设置布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题和图标
        header_layout = QHBoxLayout()
        
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        header_layout.addWidget(title_label)
        
        if icon:
            header_layout.addWidget(icon)
        
        header_layout.addStretch()
        
        # 添加到主布局
        layout.addLayout(header_layout)
        
        # 添加值标签
        self.value_label = QLabel(value)
        self.value_label.setObjectName("cardValue")
        self.value_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.value_label)
        
        # 添加趋势标签
        self.trend_label = QLabel("数据更新中...")
        self.trend_label.setObjectName("cardTrend")
        layout.addWidget(self.trend_label)
        
        # 设置尺寸策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(120)
    
    def update_value(self, value, trend=None):
        """更新卡片值"""
        self.value_label.setText(str(value))
        
        if trend:
            self.trend_label.setText(trend)


class QuantumStatusWidget(QFrame):
    """量子网络状态组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置样式
        self.setObjectName("quantumStatus")
        self.setStyleSheet("""
            #quantumStatus {
                background-color: #0F1028;
                border: 1px solid #0066AA;
                border-radius: 8px;
            }
            
            QLabel {
                color: #AAAAEE;
            }
            
            QLabel[important="true"] {
                color: #00DDFF;
                font-weight: bold;
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 70))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # 设置布局
        layout = QVBoxLayout(self)
        
        # 添加标题
        title = QLabel("量子网络状态")
        title.setObjectName("sectionTitle")
        title.setProperty("important", "true")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 创建状态网格
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        # 添加各种状态指标
        metrics = [
            ("量子纠缠对", "28", 0, 0),
            ("并行维度", "8", 0, 1),
            ("运行状态", "正常", 1, 0),
            ("网络温度", "0.32 K", 1, 1),
            ("熵值", "0.458", 2, 0),
            ("收敛率", "99.2%", 2, 1),
            ("量子位", "32 qubits", 3, 0),
            ("超导温度", "-273.12°C", 3, 1)
        ]
        
        self.metric_values = {}
        
        for name, value, row, col in metrics:
            label = QLabel(name + ":")
            value_label = QLabel(value)
            value_label.setProperty("important", "true")
            
            grid_layout.addWidget(label, row, col * 2)
            grid_layout.addWidget(value_label, row, col * 2 + 1)
            
            self.metric_values[name] = value_label
        
        layout.addLayout(grid_layout)
        
        # 添加量子网络可视化
        viz_label = QLabel("量子网络可视化")
        viz_label.setProperty("important", "true")
        layout.addWidget(viz_label)
        
        # 添加自定义绘图区域
        self.visualization = QuantumNetworkVisualization()
        self.visualization.setFixedHeight(150)
        layout.addWidget(self.visualization)
        
        # 启动动画定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_data)
        self.timer.start(2000)  # 每2秒更新一次
    
    def _update_data(self):
        """更新数据"""
        # 更新随机指标
        self.metric_values["熵值"].setText(f"{random.uniform(0.4, 0.6):.3f}")
        self.metric_values["收敛率"].setText(f"{random.uniform(98.0, 99.9):.1f}%")
        self.metric_values["网络温度"].setText(f"{random.uniform(0.28, 0.35):.2f} K")
        
        # 更新可视化
        self.visualization.update()


class QuantumNetworkVisualization(QWidget):
    """量子网络可视化组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置背景色
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(10, 15, 35))
        self.setPalette(palette)
        
        # 节点和连接数据
        self.nodes = []
        self.connections = []
        
        # 生成随机节点
        for i in range(12):
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            size = random.uniform(0.02, 0.04)
            pulse_speed = random.uniform(0.001, 0.003)
            
            # 根据位置选择颜色
            h = (x + y) * 180  # 色相
            s = 0.7  # 饱和度
            v = 0.9  # 明度
            color = self._hsv_to_rgb(h, s, v)
            
            self.nodes.append({
                'x': x,
                'y': y,
                'size': size,
                'color': color,
                'pulse': 0,
                'pulse_dir': 1,
                'pulse_speed': pulse_speed
            })
        
        # 生成连接
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                # 随机决定是否连接
                if random.random() < 0.3:
                    strength = random.uniform(0.2, 0.9)
                    self.connections.append({
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
            
            # 小幅调整位置
            node['x'] += random.uniform(-0.002, 0.002)
            node['y'] += random.uniform(-0.002, 0.002)
            
            # 确保在边界内
            node['x'] = max(0.05, min(0.95, node['x']))
            node['y'] = max(0.05, min(0.95, node['y']))
        
        self.update()
    
    def _hsv_to_rgb(self, h, s, v):
        """HSV颜色转RGB"""
        if s == 0.0:
            return (v, v, v)
        
        h /= 60.0
        i = int(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        if i == 0:
            return (v, t, p)
        elif i == 1:
            return (q, v, p)
        elif i == 2:
            return (p, v, t)
        elif i == 3:
            return (p, q, v)
        elif i == 4:
            return (t, p, v)
        else:
            return (v, p, q)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 绘制背景渐变
        gradient = QLinearGradient(0, 0, w, h)
        gradient.setColorAt(0, QColor(10, 15, 35))
        gradient.setColorAt(1, QColor(15, 25, 50))
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # 绘制连接
        for conn in self.connections:
            from_node = self.nodes[conn['from']]
            to_node = self.nodes[conn['to']]
            
            x1, y1 = from_node['x'] * w, from_node['y'] * h
            x2, y2 = to_node['x'] * w, to_node['y'] * h
            
            # 设置线条颜色和宽度
            strength = conn['strength']
            color = QColor(0, int(170 * strength), int(255 * strength), int(180 * strength))
            pen = QPen(color)
            pen.setWidthF(strength * 2)
            painter.setPen(pen)
            
            # 绘制连接线
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
            gradient = QRadialGradient(x, y, size)
            gradient.setColorAt(0, QColor(int(r * 255), int(g * 255), int(b * 255), 255))
            gradient.setColorAt(0.7, QColor(int(r * 255 * 0.8), int(g * 255 * 0.8), int(b * 255 * 0.8), 180))
            gradient.setColorAt(1, QColor(int(r * 255 * 0.6), int(g * 255 * 0.6), int(b * 255 * 0.6), 0))
            
            # 绘制节点
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(QRectF(x - size, y - size, size * 2, size * 2))


class MarketPulseWidget(QFrame):
    """市场脉搏组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置样式
        self.setObjectName("marketPulse")
        self.setStyleSheet("""
            #marketPulse {
                background-color: #0F1028;
                border: 1px solid #AA5500;
                border-radius: 8px;
            }
            
            QLabel {
                color: #AAAAEE;
            }
            
            QLabel[important="true"] {
                color: #FFBB33;
                font-weight: bold;
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 70))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # 设置布局
        layout = QVBoxLayout(self)
        
        # 添加标题
        title = QLabel("市场脉搏")
        title.setObjectName("sectionTitle")
        title.setProperty("important", "true")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 市场状态指标
        status_layout = QHBoxLayout()
        
        # 股票市场
        stock_frame = QFrame()
        stock_frame.setFrameShape(QFrame.StyledPanel)
        stock_frame.setStyleSheet("background-color: rgba(255, 150, 50, 20); border-radius: 5px; padding: 5px;")
        stock_layout = QVBoxLayout(stock_frame)
        
        stock_title = QLabel("股票市场")
        stock_title.setProperty("important", "true")
        stock_layout.addWidget(stock_title)
        
        self.stock_indicator = QLabel("90 多头")
        stock_layout.addWidget(self.stock_indicator)
        
        # 外汇市场
        forex_frame = QFrame()
        forex_frame.setFrameShape(QFrame.StyledPanel)
        forex_frame.setStyleSheet("background-color: rgba(50, 150, 255, 20); border-radius: 5px; padding: 5px;")
        forex_layout = QVBoxLayout(forex_frame)
        
        forex_title = QLabel("外汇市场")
        forex_title.setProperty("important", "true")
        forex_layout.addWidget(forex_title)
        
        self.forex_indicator = QLabel("45 中性")
        forex_layout.addWidget(self.forex_indicator)
        
        # 商品市场
        commodity_frame = QFrame()
        commodity_frame.setFrameShape(QFrame.StyledPanel)
        commodity_frame.setStyleSheet("background-color: rgba(50, 255, 150, 20); border-radius: 5px; padding: 5px;")
        commodity_layout = QVBoxLayout(commodity_frame)
        
        commodity_title = QLabel("商品市场")
        commodity_title.setProperty("important", "true")
        commodity_layout.addWidget(commodity_title)
        
        self.commodity_indicator = QLabel("60 中性偏多")
        commodity_layout.addWidget(self.commodity_indicator)
        
        # 加密货币
        crypto_frame = QFrame()
        crypto_frame.setFrameShape(QFrame.StyledPanel)
        crypto_frame.setStyleSheet("background-color: rgba(200, 100, 255, 20); border-radius: 5px; padding: 5px;")
        crypto_layout = QVBoxLayout(crypto_frame)
        
        crypto_title = QLabel("加密货币")
        crypto_title.setProperty("important", "true")
        crypto_layout.addWidget(crypto_title)
        
        self.crypto_indicator = QLabel("75 偏多")
        crypto_layout.addWidget(self.crypto_indicator)
        
        # 添加到水平布局
        status_layout.addWidget(stock_frame)
        status_layout.addWidget(forex_frame)
        status_layout.addWidget(commodity_frame)
        status_layout.addWidget(crypto_frame)
        
        layout.addLayout(status_layout)
        
        # 市场图表
        if HAS_PYQTGRAPH:
            self._create_market_chart(layout)
        else:
            chart_label = QLabel("市场趋势图 (需要pyqtgraph)")
            chart_label.setAlignment(Qt.AlignCenter)
            chart_label.setStyleSheet("color: #AAAAAA; font-style: italic;")
            layout.addWidget(chart_label)
        
        # 更新定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_data)
        self.timer.start(3000)  # 每3秒更新一次
    
    def _create_market_chart(self, layout):
        """创建市场图表"""
        # 设置样式
        pg.setConfigOption('background', QColor(15, 20, 40))
        pg.setConfigOption('foreground', QColor(150, 150, 200))
        
        # 创建图表窗口
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(180)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # 设置样式
        self.plot_widget.setLabel('left', '指数值')
        self.plot_widget.setLabel('bottom', '时间')
        
        # 创建曲线
        self.curves = {
            'stock': self.plot_widget.plot(pen=pg.mkPen(color='#FF9632', width=2), name="股票"),
            'forex': self.plot_widget.plot(pen=pg.mkPen(color='#3296FF', width=2), name="外汇"),
            'commodity': self.plot_widget.plot(pen=pg.mkPen(color='#32FF96', width=2), name="商品"),
            'crypto': self.plot_widget.plot(pen=pg.mkPen(color='#C864FF', width=2), name="加密货币")
        }
        
        # 初始数据
        self.data = {
            'stock': np.array([90, 91, 93, 92, 94, 93, 95, 98, 96, 97]),
            'forex': np.array([45, 44, 46, 48, 47, 45, 44, 43, 45, 46]),
            'commodity': np.array([60, 59, 58, 60, 62, 63, 61, 60, 62, 64]),
            'crypto': np.array([75, 77, 74, 72, 75, 78, 80, 77, 75, 76])
        }
        
        # 设置数据
        for market, curve in self.curves.items():
            curve.setData(self.data[market])
        
        # 添加图例
        self.plot_widget.addLegend()
        
        # 添加到布局
        layout.addWidget(self.plot_widget)
    
    def _update_data(self):
        """更新数据"""
        # 更新市场指标
        stock_value = random.randint(80, 100)
        self.stock_indicator.setText(f"{stock_value} {'多头' if stock_value > 90 else '偏多'}")
        
        forex_value = random.randint(40, 50)
        self.forex_indicator.setText(f"{forex_value} 中性")
        
        commodity_value = random.randint(55, 70)
        self.commodity_indicator.setText(f"{commodity_value} {'中性偏多' if commodity_value > 60 else '中性'}")
        
        crypto_value = random.randint(70, 85)
        self.crypto_indicator.setText(f"{crypto_value} {'偏多' if crypto_value < 80 else '多头'}")
        
        # 更新图表
        if HAS_PYQTGRAPH:
            # 对每个市场，添加新值并更新曲线
            self.data['stock'] = np.append(self.data['stock'][1:], stock_value)
            self.data['forex'] = np.append(self.data['forex'][1:], forex_value)
            self.data['commodity'] = np.append(self.data['commodity'][1:], commodity_value)
            self.data['crypto'] = np.append(self.data['crypto'][1:], crypto_value)
            
            for market, curve in self.curves.items():
                curve.setData(self.data[market])


class QuantumPredictionWidget(QFrame):
    """量子预测组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置样式
        self.setObjectName("quantumPrediction")
        self.setStyleSheet("""
            #quantumPrediction {
                background-color: #0F1028;
                border: 1px solid #5500AA;
                border-radius: 8px;
            }
            
            QLabel {
                color: #AAAAEE;
            }
            
            QLabel[important="true"] {
                color: #BB33FF;
                font-weight: bold;
            }
            
            QLabel[positive="true"] {
                color: #33FF66;
            }
            
            QLabel[negative="true"] {
                color: #FF3366;
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 70))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # 设置布局
        layout = QVBoxLayout(self)
        
        # 添加标题
        title = QLabel("量子预测")
        title.setObjectName("sectionTitle")
        title.setProperty("important", "true")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 预测内容
        content_layout = QVBoxLayout()
        
        # 短期预测
        short_label = QLabel("短期预测 (1-3天)")
        short_label.setProperty("important", "true")
        content_layout.addWidget(short_label)
        
        self.short_content = QLabel("市场将维持震荡上行趋势，建议关注科技和医疗板块。")
        content_layout.addWidget(self.short_content)
        
        # 中期预测
        mid_label = QLabel("中期预测 (1-2周)")
        mid_label.setProperty("important", "true")
        content_layout.addWidget(mid_label)
        
        self.mid_content = QLabel("量子分析显示市场可能进入调整阶段，建议降低仓位，关注价值型股票。")
        content_layout.addWidget(self.mid_content)
        
        # 长期预测
        long_label = QLabel("长期预测 (1-3月)")
        long_label.setProperty("important", "true")
        content_layout.addWidget(long_label)
        
        self.long_content = QLabel("量子计算预测长期走势向好，经济基本面将支持市场扩张。")
        content_layout.addWidget(self.long_content)
        
        # 预测准确率
        accuracy_layout = QHBoxLayout()
        
        accuracy_label = QLabel("预测准确率:")
        accuracy_layout.addWidget(accuracy_label)
        
        self.accuracy_value = QLabel("87.5%")
        self.accuracy_value.setProperty("positive", "true")
        self.accuracy_value.setFont(QFont("Arial", 10, QFont.Bold))
        accuracy_layout.addWidget(self.accuracy_value)
        
        accuracy_layout.addStretch()
        
        content_layout.addLayout(accuracy_layout)
        
        # 最后更新时间
        update_layout = QHBoxLayout()
        
        update_label = QLabel("最后更新:")
        update_layout.addWidget(update_label)
        
        self.update_time = QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        update_layout.addWidget(self.update_time)
        
        update_layout.addStretch()
        
        content_layout.addLayout(update_layout)
        
        layout.addLayout(content_layout)
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新预测")
        self.refresh_button.clicked.connect(self._update_predictions)
        layout.addWidget(self.refresh_button)
    
    def _update_predictions(self):
        """更新预测"""
        # 更新预测内容 (随机选择预设内容)
        short_predictions = [
            "市场将维持震荡上行趋势，建议关注科技和医疗板块。",
            "量子信号显示短期可能出现回调，建议降低高估值股票仓位。",
            "短期内市场情绪积极，量子计算模型预测指数将走高。",
            "波动率将增加，建议做好风险管理，关注低估值蓝筹股。"
        ]
        
        mid_predictions = [
            "量子分析显示市场可能进入调整阶段，建议降低仓位，关注价值型股票。",
            "中期市场趋势向好，量子模型预测创新科技和新能源板块将领涨。",
            "宏观经济数据改善，预计市场将在1-2周内突破前高。",
            "警惕通胀压力增大，利率敏感板块可能承压，建议关注必需消费品和医疗板块。"
        ]
        
        long_predictions = [
            "量子计算预测长期走势向好，经济基本面将支持市场扩张。",
            "长期看好创新驱动领域，人工智能、量子计算和新能源将成为投资热点。",
            "量子分析显示全球流动性收紧的风险增加，长期投资需更加关注企业基本面。",
            "长期来看，全球经济复苏将支撑市场，但不同地区表现将出现分化。"
        ]
        
        # 随机选择预测
        self.short_content.setText(random.choice(short_predictions))
        self.mid_content.setText(random.choice(mid_predictions))
        self.long_content.setText(random.choice(long_predictions))
        
        # 更新准确率
        accuracy = random.uniform(85.0, 92.0)
        self.accuracy_value.setText(f"{accuracy:.1f}%")
        
        # 更新时间
        self.update_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class DashboardWidget(QWidget):
    """超神系统高级仪表盘"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # 添加欢迎信息
        welcome_layout = QHBoxLayout()
        
        welcome_label = QLabel("欢迎回来，超神系统运行中")
        welcome_label.setFont(QFont("Arial", 20, QFont.Bold))
        welcome_label.setStyleSheet("color: #00DDFF;")
        welcome_layout.addWidget(welcome_label)
        
        self.date_time = QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.date_time.setStyleSheet("color: #AAAAEE;")
        self.date_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        welcome_layout.addWidget(self.date_time)
        
        main_layout.addLayout(welcome_layout)
        
        # 添加信息卡片
        cards_layout = QHBoxLayout()
        
        self.market_card = InfoCard("市场指数", "3,600.25", color="#FF6600")
        self.volume_card = InfoCard("成交量", "1.25万亿", color="#00CC66")
        self.risk_card = InfoCard("风险指数", "32.5", color="#3366FF")
        self.trend_card = InfoCard("趋势强度", "75.8", color="#FF33CC")
        
        cards_layout.addWidget(self.market_card)
        cards_layout.addWidget(self.volume_card)
        cards_layout.addWidget(self.risk_card)
        cards_layout.addWidget(self.trend_card)
        
        main_layout.addLayout(cards_layout)
        
        # 添加主要内容区
        content_layout = QHBoxLayout()
        
        # 左侧 - 量子网络状态
        self.quantum_status = QuantumStatusWidget()
        content_layout.addWidget(self.quantum_status, 1)
        
        # 右侧垂直分栏
        right_layout = QVBoxLayout()
        
        # 右上 - 市场脉搏
        self.market_pulse = MarketPulseWidget()
        right_layout.addWidget(self.market_pulse)
        
        # 右下 - 量子预测
        self.quantum_prediction = QuantumPredictionWidget()
        right_layout.addWidget(self.quantum_prediction)
        
        content_layout.addLayout(right_layout, 1)
        
        main_layout.addLayout(content_layout)
        
        # 启动日期时间更新定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_datetime)
        self.timer.start(1000)  # 每秒更新
        
        # 数据更新定时器
        self.data_timer = QTimer(self)
        self.data_timer.timeout.connect(self._update_card_data)
        self.data_timer.start(5000)  # 每5秒更新
    
    def _update_datetime(self):
        """更新日期时间"""
        self.date_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _update_card_data(self):
        """更新卡片数据"""
        # 市场指数
        index_value = 3600.25 + random.uniform(-20, 20)
        change = (index_value - 3600.25) / 3600.25 * 100
        trend = f"{'↑' if change >= 0 else '↓'} {abs(change):.2f}% 今日"
        self.market_card.update_value(f"{index_value:.2f}", trend)
        
        # 成交量
        volume = 1.25 + random.uniform(-0.1, 0.1)
        volume_change = (volume - 1.25) / 1.25 * 100
        volume_trend = f"{'↑' if volume_change >= 0 else '↓'} {abs(volume_change):.2f}% 较昨日"
        self.volume_card.update_value(f"{volume:.2f}万亿", volume_trend)
        
        # 风险指数
        risk = 32.5 + random.uniform(-2, 2)
        risk_trend = "低风险" if risk < 30 else "中等风险" if risk < 50 else "高风险"
        self.risk_card.update_value(f"{risk:.1f}", f"{risk_trend}")
        
        # 趋势强度
        trend = 75.8 + random.uniform(-5, 5)
        trend_direction = "看涨" if trend > 70 else "中性" if trend > 40 else "看跌"
        self.trend_card.update_value(f"{trend:.1f}", f"{trend_direction}")


def create_dashboard(parent=None):
    """创建并返回仪表盘组件"""
    return DashboardWidget(parent) 