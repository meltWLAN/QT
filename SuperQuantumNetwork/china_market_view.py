#!/usr/bin/env python3
"""
超神系统 - 中国市场分析视图
高级版本，适用于超神桌面系统
"""

import logging
import traceback
from functools import lru_cache
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QTableWidget, QTableWidgetItem, 
                           QHeaderView, QGroupBox, QMessageBox, QProgressBar,
                           QSplitter, QFrame, QComboBox, QSpacerItem, QSizePolicy,
                           QCheckBox, QApplication, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QSize, QTimer, QObject, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve, QRectF, QPointF, QLineF
from PyQt5.QtGui import QFont, QColor, QIcon, QPainter, QBrush, QLinearGradient, QPen, QRadialGradient
import os
import sys
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)


class DataUpdateWorker(QObject):
    """数据更新工作线程"""
    dataUpdated = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
    def run(self):
        """执行数据更新"""
        try:
            # 更新市场数据
            self.controller.update_market_data()
            
            # 获取市场数据
            market_data = self.controller.market_data
            
            # 发送数据更新信号
            self.dataUpdated.emit(market_data)
        except Exception as e:
            logger.error(f"数据更新线程异常: {str(e)}")
            logger.error(traceback.format_exc())
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class PredictionWorker(QObject):
    """预测执行工作线程"""
    predictionComplete = pyqtSignal(dict, dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
    def run(self):
        """执行市场预测"""
        try:
            # 运行预测
            prediction = self.controller.predict_market_trend()
            
            # 获取投资组合建议
            portfolio = self.controller.get_portfolio_suggestion()
            
            # 发送预测完成信号
            self.predictionComplete.emit(prediction, portfolio)
        except Exception as e:
            logger.error(f"预测线程异常: {str(e)}")
            logger.error(traceback.format_exc())
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class QuantumParticleEffect(QWidget):
    """量子粒子效果组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 80)
        self.particles = []
        self.animations = []
        
        # 创建粒子
        for i in range(5):
            # 创建粒子位置和大小
            self.particles.append({
                'x': 30 + i * 50,
                'y': 40,
                'size': 10,
                'color': QColor(0, 180 + i*15, 255, 180),
                'phase': i * 0.5
            })
        
        # 启动动画定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        
        # 设置背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)
    
    def paintEvent(self, event):
        """绘制粒子效果"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算动画
        current_time = time.time()
        
        for p in self.particles:
            # 波动动画
            wave = np.sin(current_time * 2 + p['phase']) * 10
            
            # 绘制光晕
            glowRect = QRectF(
                p['x'] - p['size'] * 2.5, 
                p['y'] + wave - p['size'] * 2.5, 
                p['size'] * 5, 
                p['size'] * 5
            )
            gradient = QRadialGradient(p['x'], p['y'] + wave, p['size'] * 2.5)
            glow_color = QColor(p['color'])
            glow_color.setAlpha(40)
            gradient.setColorAt(0, glow_color)
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(glowRect)
            
            # 绘制粒子
            particleRect = QRectF(
                p['x'] - p['size'], 
                p['y'] + wave - p['size'], 
                p['size'] * 2, 
                p['size'] * 2
            )
            gradient = QRadialGradient(p['x'], p['y'] + wave, p['size'])
            gradient.setColorAt(0, p['color'])
            gradient.setColorAt(1, QColor(0, 100, 200, 180))
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(particleRect)
            
            # 绘制连接线
            if p != self.particles[-1]:
                next_p = self.particles[self.particles.index(p) + 1]
                next_wave = np.sin(current_time * 2 + next_p['phase']) * 10
                
                pen = QPen(QColor(0, 150, 220, 100), 2)
                painter.setPen(pen)
                # 使用QLineF代替直接传递坐标
                line = QLineF(
                    QPointF(p['x'], p['y'] + wave),
                    QPointF(next_p['x'], next_p['y'] + next_wave)
                )
                painter.drawLine(line)


class ChinaMarketWidget(QWidget):
    """超神系统中国市场分析组件 - 高级版"""
    
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        
        # 缓存数据
        self.cached_market_data = {}
        self.cached_prediction = {}
        self.cached_portfolio = {}
        self.last_update_time = 0
        self.last_prediction_time = 0
        
        # 初始化线程
        self.data_thread = None
        self.prediction_thread = None
        
        self.init_ui()
        
        # 设置对象名，方便样式表定位
        self.setObjectName("chinaMarketWidget")
        
        # 初始刷新
        QTimer.singleShot(500, self.refresh_data)
        
        # 设置整体样式
        self.setStyleSheet("""
            #chinaMarketWidget {
                background-color: #050510;
                background-image: radial-gradient(circle at 50% 30%, #101040 0%, #050510 70%);
            }
            
            QGroupBox {
                border: 1px solid #222244;
                border-radius: 6px;
                margin-top: 12px;
                background-color: rgba(10, 10, 30, 0.6);
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #00DDFF;
                font-weight: bold;
            }
            
            QLabel {
                color: #BBBBDD;
            }
            
            QLabel[class="index-value"] {
                color: #00DDFF;
                font-size: 22px;
                font-weight: bold;
            }
        """)
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # 创建顶部面板 - 显示市场指数
        self.create_market_index_panel()
        
        # 创建左右分栏
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #222244;
            }
        """)
        
        # 创建左侧面板 - 热点板块和北向资金
        self.create_left_panel()
        
        # 创建右侧面板 - 个股推荐列表
        self.create_right_panel()
        
        # 添加分栏到主布局
        self.main_layout.addWidget(self.splitter)
        
        # 创建底部控制面板
        self.create_bottom_panel()
        
        # 创建状态面板
        self.create_status_panel()
    
    def create_market_index_panel(self):
        """创建市场指数面板"""
        index_group = QGroupBox("市场指数")
        index_layout = QVBoxLayout(index_group)
        
        # 添加量子粒子效果
        particle_effect = QuantumParticleEffect()
        index_layout.addWidget(particle_effect, alignment=Qt.AlignCenter)
        
        # 创建指数信息面板
        index_info_layout = QHBoxLayout()
        
        # 上证指数
        sh_box = QGroupBox("上证指数")
        sh_box.setProperty("class", "index-box")
        sh_box.setStyleSheet("""
            QGroupBox[class="index-box"] {
                background-color: rgba(10, 10, 35, 0.8);
                border: 1px solid #2A2A4A;
                border-radius: 8px;
            }
        """)
        
        sh_layout = QVBoxLayout(sh_box)
        self.sh_value_label = QLabel("--")
        self.sh_value_label.setAlignment(Qt.AlignCenter)
        self.sh_value_label.setProperty("class", "index-value")
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 170, 255, 150))
        shadow.setOffset(0, 0)
        self.sh_value_label.setGraphicsEffect(shadow)
        
        self.sh_change_label = QLabel("--")
        self.sh_change_label.setAlignment(Qt.AlignCenter)
        sh_layout.addWidget(self.sh_value_label)
        sh_layout.addWidget(self.sh_change_label)
        
        # 深证成指
        sz_box = QGroupBox("深证成指")
        sz_box.setProperty("class", "index-box")
        sz_box.setStyleSheet("""
            QGroupBox[class="index-box"] {
                background-color: rgba(10, 10, 35, 0.8);
                border: 1px solid #2A2A4A;
                border-radius: 8px;
            }
        """)
        
        sz_layout = QVBoxLayout(sz_box)
        self.sz_value_label = QLabel("--")
        self.sz_value_label.setAlignment(Qt.AlignCenter)
        self.sz_value_label.setProperty("class", "index-value")
        
        # 添加阴影效果
        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(15)
        shadow2.setColor(QColor(0, 170, 255, 150))
        shadow2.setOffset(0, 0)
        self.sz_value_label.setGraphicsEffect(shadow2)
        
        self.sz_change_label = QLabel("--")
        self.sz_change_label.setAlignment(Qt.AlignCenter)
        sz_layout.addWidget(self.sz_value_label)
        sz_layout.addWidget(self.sz_change_label)
        
        # 创业板指
        cyb_box = QGroupBox("创业板指")
        cyb_box.setProperty("class", "index-box")
        cyb_box.setStyleSheet("""
            QGroupBox[class="index-box"] {
                background-color: rgba(10, 10, 35, 0.8);
                border: 1px solid #2A2A4A;
                border-radius: 8px;
            }
        """)
        
        cyb_layout = QVBoxLayout(cyb_box)
        self.cyb_value_label = QLabel("--")
        self.cyb_value_label.setAlignment(Qt.AlignCenter)
        self.cyb_value_label.setProperty("class", "index-value")
        
        # 添加阴影效果
        shadow3 = QGraphicsDropShadowEffect()
        shadow3.setBlurRadius(15)
        shadow3.setColor(QColor(0, 170, 255, 150))
        shadow3.setOffset(0, 0)
        self.cyb_value_label.setGraphicsEffect(shadow3)
        
        self.cyb_change_label = QLabel("--")
        self.cyb_change_label.setAlignment(Qt.AlignCenter)
        cyb_layout.addWidget(self.cyb_value_label)
        cyb_layout.addWidget(self.cyb_change_label)
        
        # 添加到布局
        index_info_layout.addWidget(sh_box)
        index_info_layout.addWidget(sz_box)
        index_info_layout.addWidget(cyb_box)
        
        # 添加到指数面板
        index_layout.addLayout(index_info_layout)
        
        # 市场风险和趋势
        risk_layout = QHBoxLayout()
        
        # 风险评级
        risk_frame = QFrame()
        risk_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 40, 0.6);
                border-radius: 6px;
                padding: 5px;
            }
        """)
        risk_frame_layout = QHBoxLayout(risk_frame)
        risk_frame_layout.setContentsMargins(10, 5, 10, 5)
        
        self.risk_label = QLabel("市场风险: --")
        self.risk_label.setProperty("class", "risk-label")
        self.risk_label.setStyleSheet("color: #FFAA44; font-weight: bold;")
        risk_frame_layout.addWidget(self.risk_label)
        
        # 市场趋势
        self.trend_label = QLabel("趋势: --")
        self.trend_label.setProperty("class", "trend-label")
        self.trend_label.setStyleSheet("color: #00DDFF; font-weight: bold;")
        risk_frame_layout.addWidget(self.trend_label)
        
        risk_layout.addWidget(risk_frame)
        
        # 最后更新时间
        update_frame = QFrame()
        update_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 40, 0.6);
                border-radius: 6px;
                padding: 5px;
            }
        """)
        update_frame_layout = QHBoxLayout(update_frame)
        update_frame_layout.setContentsMargins(10, 5, 10, 5)
        
        self.update_time_label = QLabel("最后更新: --")
        self.update_time_label.setStyleSheet("color: #8888AA;")
        update_frame_layout.addWidget(self.update_time_label)
        
        risk_layout.addWidget(update_frame)
        
        index_layout.addLayout(risk_layout)
        
        # 添加到主布局
        self.main_layout.addWidget(index_group)
    
    def create_left_panel(self):
        """创建左侧面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 热点板块
        hot_group = QGroupBox("热点板块")
        hot_layout = QVBoxLayout(hot_group)
        self.sector_table = QTableWidget(0, 2)
        self.sector_table.setHorizontalHeaderLabels(["板块", "强度"])
        self.sector_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.sector_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.sector_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sector_table.setStyleSheet("""
            QTableWidget {
                background-color: rgba(8, 8, 24, 0.8);
                alternate-background-color: rgba(12, 12, 30, 0.8);
                gridline-color: #222244;
                border: none;
                border-radius: 4px;
            }
            
            QTableWidget::item {
                padding: 5px;
            }
            
            QTableWidget::item:selected {
                background-color: rgba(0, 120, 215, 0.6);
                color: white;
            }
            
            QHeaderView::section {
                background-color: #101030;
                color: #8888AA;
                border: none;
                padding: 5px;
                font-weight: bold;
            }
        """)
        hot_layout.addWidget(self.sector_table)
        
        # 下一轮热点预测
        next_group = QGroupBox("下一轮潜在热点")
        next_layout = QVBoxLayout(next_group)
        self.next_sector_table = QTableWidget(0, 1)
        self.next_sector_table.setHorizontalHeaderLabels(["板块"])
        self.next_sector_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.next_sector_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.next_sector_table.setStyleSheet("""
            QTableWidget {
                background-color: rgba(8, 8, 24, 0.8);
                alternate-background-color: rgba(12, 12, 30, 0.8);
                gridline-color: #222244;
                border: none;
                border-radius: 4px;
            }
            
            QTableWidget::item {
                padding: 5px;
            }
            
            QTableWidget::item:selected {
                background-color: rgba(0, 120, 215, 0.6);
                color: white;
            }
            
            QHeaderView::section {
                background-color: #101030;
                color: #8888AA;
                border: none;
                padding: 5px;
                font-weight: bold;
            }
        """)
        next_layout.addWidget(self.next_sector_table)
        
        # 北向资金
        north_group = QGroupBox("北向资金")
        north_layout = QVBoxLayout(north_group)
        
        self.north_flow_label = QLabel("今日净流入: --")
        self.north_flow_label.setStyleSheet("font-size: 14px; padding: 5px;")
        
        self.north_flow_5d_label = QLabel("5日净流入: --")
        self.north_flow_5d_label.setStyleSheet("font-size: 14px; padding: 5px;")
        
        self.north_trend_label = QLabel("资金趋势: --")
        self.north_trend_label.setStyleSheet("font-size: 14px; padding: 5px; font-weight: bold;")
        
        north_layout.addWidget(self.north_flow_label)
        north_layout.addWidget(self.north_flow_5d_label)
        north_layout.addWidget(self.north_trend_label)
        
        # 添加到左侧布局
        left_layout.addWidget(hot_group)
        left_layout.addWidget(next_group)
        left_layout.addWidget(north_group)
        
        # 添加到分栏
        self.splitter.addWidget(left_widget)
    
    def create_right_panel(self):
        """创建右侧面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 个股推荐
        stock_group = QGroupBox("量子预测个股推荐")
        stock_layout = QVBoxLayout(stock_group)
        
        # 创建表格
        self.stock_table = QTableWidget(0, 6)
        self.stock_table.setHorizontalHeaderLabels(["代码", "名称", "行业", "操作", "当前价", "风险"])
        self.stock_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stock_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stock_table.setAlternatingRowColors(True)
        self.stock_table.setStyleSheet("""
            QTableWidget {
                background-color: rgba(8, 8, 24, 0.8);
                alternate-background-color: rgba(12, 12, 30, 0.8);
                gridline-color: #222244;
                border: none;
                border-radius: 4px;
                color: #BBBBDD;
            }
            
            QTableWidget::item {
                padding: 5px;
            }
            
            QTableWidget::item:selected {
                background-color: rgba(0, 120, 215, 0.6);
                color: white;
            }
            
            QHeaderView::section {
                background-color: #101030;
                color: #8888AA;
                border: none;
                padding: 5px;
                font-weight: bold;
            }
        """)
        stock_layout.addWidget(self.stock_table)
        
        # 投资建议
        advice_frame = QFrame()
        advice_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 40, 0.6);
                border-radius: 6px;
                padding: 5px;
            }
        """)
        advice_layout = QHBoxLayout(advice_frame)
        advice_layout.setContentsMargins(10, 5, 10, 5)
        
        self.position_label = QLabel("建议仓位: --")
        self.position_label.setProperty("class", "advice-label")
        self.position_label.setStyleSheet("color: #00FFAA; font-weight: bold; font-size: 16px;")
        advice_layout.addWidget(self.position_label)
        
        stock_layout.addWidget(advice_frame)
        
        # 添加到右侧布局
        right_layout.addWidget(stock_group)
        
        # 添加到分栏
        self.splitter.addWidget(right_widget)
    
    def create_bottom_panel(self):
        """创建底部控制面板"""
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(10, 10, 30, 0.7);
                border-radius: 8px;
                border: 1px solid #222244;
            }
            
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                             stop:0 #1A1A4A, stop:1 #0A0A2A);
                color: #BBBBDD;
                border: 1px solid #2A2A5A;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                             stop:0 #2A2A6A, stop:1 #1A1A4A);
                color: #00DDFF;
            }
            
            QPushButton:pressed {
                background-color: #151540;
            }
            
            QCheckBox {
                color: #BBBBDD;
            }
            
            QComboBox {
                background-color: #0A0A2A;
                color: #BBBBDD;
                border: 1px solid #222244;
                border-radius: 4px;
                padding: 3px 5px;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #222244;
            }
        """)
        
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(15, 10, 15, 10)
        
        # 操作按钮
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.refresh_button.clicked.connect(self.refresh_data)
        
        self.predict_button = QPushButton("超神预测")
        self.predict_button.setIcon(QIcon.fromTheme("system-run"))
        self.predict_button.clicked.connect(self.run_prediction)
        
        # 其他控制
        self.auto_refresh = QCheckBox("自动刷新")
        self.auto_refresh.setChecked(False)
        self.auto_refresh.stateChanged.connect(self.toggle_auto_refresh)
        
        self.refresh_interval = QComboBox()
        self.refresh_interval.addItems(["1分钟", "5分钟", "10分钟", "30分钟"])
        self.refresh_interval.setCurrentIndex(1)  # 默认5分钟
        
        # 添加到布局
        bottom_layout.addWidget(self.refresh_button)
        bottom_layout.addWidget(self.predict_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.auto_refresh)
        bottom_layout.addWidget(self.refresh_interval)
        
        # 添加到主布局
        self.main_layout.addWidget(bottom_frame)
        
        # 自动刷新定时器
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
    
    def create_status_panel(self):
        """创建状态面板"""
        self.status_bar = QProgressBar()
        self.status_bar.setTextVisible(True)
        self.status_bar.setFormat("%p% %v")
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setValue(0)
        self.status_bar.setMaximum(100)
        self.status_bar.setVisible(False)
        self.status_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #222244;
                border-radius: 4px;
                background-color: rgba(8, 8, 24, 0.7);
                color: #BBBBDD;
                height: 8px;
                text-align: center;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #0055AA, stop:0.4 #00AAEE, 
                                           stop:0.6 #00DDFF, stop:1 #0088CC);
                border-radius: 3px;
            }
        """)
        
        self.main_layout.addWidget(self.status_bar)
    
    def toggle_auto_refresh(self, state):
        """切换自动刷新状态"""
        if state == Qt.Checked:
            # 获取刷新间隔（分钟）
            interval_text = self.refresh_interval.currentText()
            interval_min = int(interval_text.split("分")[0])
            interval_ms = interval_min * 60 * 1000
            
            # 启动定时器
            self.refresh_timer.start(interval_ms)
            logger.info(f"已启用自动刷新，间隔: {interval_min}分钟")
        else:
            # 停止定时器
            self.refresh_timer.stop()
            logger.info("已停用自动刷新")
    
    def refresh_data(self):
        """刷新市场数据"""
        if not self.controller:
            return
            
        # 检查缓存时间，如果距上次更新不足30秒，则使用缓存
        current_time = time.time()
        if current_time - self.last_update_time < 30 and self.cached_market_data:
            logger.info("使用缓存的市场数据（30秒内）")
            self._update_index_display(self.cached_market_data)
            self._update_north_flow()
            return
            
        # 启动数据加载动画
        self.status_bar.setVisible(True)
        self.status_bar.setValue(0)
        self.status_bar.setFormat("正在加载市场数据... %p%")
        
        # 设置按钮状态
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText("刷新中...")
        
        # 创建工作线程
        self.data_worker = DataUpdateWorker(self.controller)
        self.data_thread = QThread()
        self.data_worker.moveToThread(self.data_thread)
        
        # 连接信号
        self.data_thread.started.connect(self.data_worker.run)
        self.data_worker.dataUpdated.connect(self.handle_data_updated)
        self.data_worker.error.connect(self.handle_data_error)
        self.data_worker.finished.connect(self.data_thread.quit)
        self.data_worker.finished.connect(self.data_worker.deleteLater)
        self.data_thread.finished.connect(self.data_thread.deleteLater)
        self.data_thread.finished.connect(self.on_data_update_finished)
        
        # 启动线程
        self.data_thread.start()
        
        # 启动进度条更新
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._update_progress)
        self.progress_timer.start(100)  # 每100ms更新一次进度条
        
    def _update_progress(self):
        """更新进度条"""
        current = self.status_bar.value()
        if current < 90:  # 保留最后10%给实际数据处理
            self.status_bar.setValue(current + 1)
    
    def handle_data_updated(self, market_data):
        """处理数据更新结果"""
        # 缓存数据
        self.cached_market_data = market_data
        self.last_update_time = time.time()
        
        # 更新UI
        self._update_index_display(market_data)
        self._update_north_flow()
        
        # 更新进度条
        self.status_bar.setValue(100)
        self.status_bar.setFormat("数据加载完成 100%")
        
        # 更新时间
        from datetime import datetime
        self.update_time_label.setText(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
    
    def handle_data_error(self, error_message):
        """处理数据更新错误"""
        # 显示错误消息
        QMessageBox.warning(self, "刷新失败", f"刷新市场数据失败: {error_message}")
    
    def on_data_update_finished(self):
        """数据更新线程完成"""
        # 恢复按钮状态
        self.refresh_button.setEnabled(True)
        self.refresh_button.setText("刷新数据")
        
        # 停止进度条更新定时器
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()
        
        # 隐藏进度条
        QTimer.singleShot(1000, lambda: self.status_bar.setVisible(False))
    
    def _update_index_display(self, market_data):
        """更新指数显示"""
        # 更新UI元素前，确保界面响应
        QApplication.processEvents()
        
        # 上证指数
        sh_data = market_data.get('sh_index', {})
        sh_close = sh_data.get('close', 0)
        sh_change = sh_data.get('change_pct', 0)
        
        if sh_change is None or (isinstance(sh_change, float) and np.isnan(sh_change)):
            sh_change = 0
            
        self.sh_value_label.setText(f"{sh_close:.2f}")
        self.sh_change_label.setText(f"{sh_change:+.2f}%")
        
        if sh_change >= 0:
            self.sh_change_label.setStyleSheet("color: #FF4444; font-weight: bold;")
        else:
            self.sh_change_label.setStyleSheet("color: #44FF44; font-weight: bold;")
        
        # 深证成指
        sz_data = market_data.get('sz_index', {})
        sz_close = sz_data.get('close', 0)
        sz_change = sz_data.get('change_pct', 0)
        
        if sz_change is None or (isinstance(sz_change, float) and np.isnan(sz_change)):
            sz_change = 0
            
        self.sz_value_label.setText(f"{sz_close:.2f}")
        self.sz_change_label.setText(f"{sz_change:+.2f}%")
        
        if sz_change >= 0:
            self.sz_change_label.setStyleSheet("color: #FF4444; font-weight: bold;")
        else:
            self.sz_change_label.setStyleSheet("color: #44FF44; font-weight: bold;")
        
        # 创业板指
        cyb_data = market_data.get('cyb_index', {})
        cyb_close = cyb_data.get('close', 0)
        cyb_change = cyb_data.get('change_pct', 0)
        
        if cyb_change is None or (isinstance(cyb_change, float) and np.isnan(cyb_change)):
            cyb_change = 0
            
        self.cyb_value_label.setText(f"{cyb_close:.2f}")
        self.cyb_change_label.setText(f"{cyb_change:+.2f}%")
        
        if cyb_change >= 0:
            self.cyb_change_label.setStyleSheet("color: #FF4444; font-weight: bold;")
        else:
            self.cyb_change_label.setStyleSheet("color: #44FF44; font-weight: bold;")
        
        # 确保界面更新
        QApplication.processEvents()
    
    def _update_north_flow(self):
        """更新北向资金数据"""
        try:
            # 获取北向资金数据
            prediction = getattr(self.controller, 'latest_prediction', {})
            north_flow = prediction.get('north_flow', {})
            
            if north_flow:
                # 今日净流入
                daily_flow = north_flow.get('total_inflow', 0) / 100000000  # 单位：亿
                self.north_flow_label.setText(f"今日净流入: {daily_flow:.2f}亿")
                
                # 5日净流入
                flow_5d = north_flow.get('total_flow_5d', 0) / 100000000  # 单位：亿
                self.north_flow_5d_label.setText(f"5日净流入: {flow_5d:.2f}亿")
                
                # 趋势
                trend = north_flow.get('flow_trend', 'unknown')
                self.north_trend_label.setText(f"资金趋势: {trend}")
                
                # 根据正负设置颜色
                if daily_flow > 0:
                    self.north_flow_label.setStyleSheet("color: #FF4444;")
                else:
                    self.north_flow_label.setStyleSheet("color: #44FF44;")
                    
                if flow_5d > 0:
                    self.north_flow_5d_label.setStyleSheet("color: #FF4444;")
                else:
                    self.north_flow_5d_label.setStyleSheet("color: #44FF44;")
        except Exception as e:
            logger.error(f"更新北向资金失败: {str(e)}")
    
    def run_prediction(self):
        """运行市场预测"""
        if not self.controller:
            return
            
        # 检查缓存时间，如果距上次预测不足5分钟，则使用缓存
        current_time = time.time()
        cache_valid = (current_time - self.last_prediction_time < 300) and self.cached_prediction and self.cached_portfolio
        
        if cache_valid:
            logger.info("使用缓存的预测结果（5分钟内）")
            # 直接更新UI
            self._update_prediction_ui(self.cached_prediction, self.cached_portfolio)
            return
            
        # 启动预测加载动画
        self.status_bar.setVisible(True)
        self.status_bar.setValue(0)
        self.status_bar.setFormat("正在进行超神预测... %p%")
        
        # 修改按钮状态
        self.predict_button.setEnabled(False)
        self.predict_button.setText("预测中...")
        
        # 创建工作线程
        self.prediction_worker = PredictionWorker(self.controller)
        self.prediction_thread = QThread()
        self.prediction_worker.moveToThread(self.prediction_thread)
        
        # 连接信号
        self.prediction_thread.started.connect(self.prediction_worker.run)
        self.prediction_worker.predictionComplete.connect(self.handle_prediction_complete)
        self.prediction_worker.error.connect(self.handle_prediction_error)
        self.prediction_worker.finished.connect(self.prediction_thread.quit)
        self.prediction_worker.finished.connect(self.prediction_worker.deleteLater)
        self.prediction_thread.finished.connect(self.prediction_thread.deleteLater)
        self.prediction_thread.finished.connect(self.on_prediction_finished)
        
        # 启动线程
        self.prediction_thread.start()
        
        # 启动进度条更新
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._update_progress)
        self.progress_timer.start(100)
    
    def handle_prediction_complete(self, prediction, portfolio):
        """处理预测完成"""
        # 缓存结果
        self.cached_prediction = prediction
        self.cached_portfolio = portfolio
        self.last_prediction_time = time.time()
        
        # 更新UI
        self._update_prediction_ui(prediction, portfolio)
        
        # 更新进度条
        self.status_bar.setValue(100)
        self.status_bar.setFormat("超神预测完成 100%")
        
        # 显示预测完成消息
        risk_analysis = prediction.get('risk_analysis', {})
        overall_risk = risk_analysis.get('overall_risk', 0)
        risk_trend = risk_analysis.get('risk_trend', '')
        QMessageBox.information(self, "预测完成", 
                                f"超神量子预测完成！\n市场风险评级: {overall_risk:.2f}\n趋势: {risk_trend}")
    
    def _update_prediction_ui(self, prediction, portfolio):
        """更新预测相关UI元素"""
        # 获取预测结果
        sector_rotation = prediction.get('sector_rotation', {})
        risk_analysis = prediction.get('risk_analysis', {})
        
        # 更新热点板块
        self._update_hot_sectors(sector_rotation)
        
        # 更新风险评级
        overall_risk = risk_analysis.get('overall_risk', 0)
        risk_trend = risk_analysis.get('risk_trend', '')
        self.risk_label.setText(f"市场风险: {overall_risk:.2f}")
        self.trend_label.setText(f"趋势: {risk_trend}")
        
        # 更新建议仓位
        position = portfolio.get('max_position', 0) * 100
        self.position_label.setText(f"建议仓位: {position:.0f}%")
        
        # 更新个股推荐
        self._update_stock_recommendations(portfolio)
    
    def handle_prediction_error(self, error_message):
        """处理预测错误"""
        # 显示错误消息
        QMessageBox.warning(self, "预测失败", f"运行市场预测失败: {error_message}")
    
    def on_prediction_finished(self):
        """预测线程完成"""
        # 恢复按钮状态
        self.predict_button.setEnabled(True)
        self.predict_button.setText("超神预测")
        
        # 停止进度条更新定时器
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()
        
        # 隐藏进度条
        QTimer.singleShot(1000, lambda: self.status_bar.setVisible(False))
    
    def _update_hot_sectors(self, sector_rotation):
        """更新热点板块"""
        # 当前热点
        hot_sectors = sector_rotation.get('current_hot_sectors', [])
        
        # 清空表格
        self.sector_table.setRowCount(0)
        
        # 添加热点板块
        for i, sector in enumerate(hot_sectors):
            self.sector_table.insertRow(i)
            self.sector_table.setItem(i, 0, QTableWidgetItem(sector))
            
            # 强度 - 使用简单的星级
            strength = QTableWidgetItem("⭐" * (5 - i))  # 越靠前的热度越高
            self.sector_table.setItem(i, 1, strength)
        
        # 下一轮热点
        next_sectors = sector_rotation.get('next_sectors_prediction', [])
        
        # 清空表格
        self.next_sector_table.setRowCount(0)
        
        # 添加下一轮热点
        for i, sector in enumerate(next_sectors):
            self.next_sector_table.insertRow(i)
            self.next_sector_table.setItem(i, 0, QTableWidgetItem(sector))
    
    def _update_stock_recommendations(self, portfolio):
        """更新个股推荐"""
        # 获取推荐列表
        stock_suggestions = portfolio.get('stock_suggestions', [])
        
        # 清空表格
        self.stock_table.setRowCount(0)
        
        # 添加推荐股票
        for i, stock in enumerate(stock_suggestions):
            self.stock_table.insertRow(i)
            
            # 股票代码
            code_item = QTableWidgetItem(stock.get('stock', ''))
            self.stock_table.setItem(i, 0, code_item)
            
            # 股票名称
            name_item = QTableWidgetItem(stock.get('name', ''))
            self.stock_table.setItem(i, 1, name_item)
            
            # 行业
            sector_item = QTableWidgetItem(stock.get('sector', ''))
            self.stock_table.setItem(i, 2, sector_item)
            
            # 操作建议
            action = stock.get('action', '')
            action_item = QTableWidgetItem(action)
            
            # 根据操作设置颜色
            if '买入' in action:
                action_item.setBackground(QColor('#334433'))
                action_item.setForeground(QColor('#55FF55'))
            elif '卖出' in action:
                action_item.setBackground(QColor('#443333'))
                action_item.setForeground(QColor('#FF5555'))
                
            self.stock_table.setItem(i, 3, action_item)
            
            # 当前价格
            price_item = QTableWidgetItem(f"{stock.get('current_price', 0):.2f}")
            self.stock_table.setItem(i, 4, price_item)
            
            # 风险评级
            risk_item = QTableWidgetItem(stock.get('risk_level', ''))
            
            # 根据风险设置颜色
            risk_level = stock.get('risk_level', '')
            if '低' in risk_level:
                risk_item.setForeground(QColor('#44FF44'))
            elif '高' in risk_level:
                risk_item.setForeground(QColor('#FF4444'))
            else:
                risk_item.setForeground(QColor('#FFFF44'))
                
            self.stock_table.setItem(i, 5, risk_item)


def create_market_view(main_window, tab_widget, controller=None):
    """创建中国市场视图
    
    Args:
        main_window: 主窗口
        tab_widget: 标签页容器
        controller: 市场数据控制器
    
    Returns:
        创建的中国市场视图组件
    """
    try:
        market_widget = ChinaMarketWidget(parent=main_window, controller=controller)
        return market_widget
    except Exception as e:
        logger.error(f"创建中国市场视图失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 创建一个简单的替代视图
        fallback_widget = QWidget()
        fallback_layout = QVBoxLayout(fallback_widget)
        
        # 错误信息
        error_label = QLabel(f"加载中国市场分析组件失败: {str(e)}")
        error_label.setStyleSheet("color: red;")
        fallback_layout.addWidget(error_label)
        
        # 创建简单市场指数面板
        indices_group = QGroupBox("市场指数")
        indices_layout = QVBoxLayout(indices_group)
        
        # 添加一些示例数据
        indices = [
            {"name": "上证指数", "code": "000001.SH", "price": "3250.55", "change": "+0.75%"},
            {"name": "深证成指", "code": "399001.SZ", "price": "10765.31", "change": "+1.25%"},
            {"name": "创业板指", "code": "399006.SZ", "price": "2156.48", "change": "+1.56%"}
        ]
        
        for index in indices:
            index_widget = QWidget()
            index_layout = QHBoxLayout(index_widget)
            
            name_label = QLabel(f"{index['name']} ({index['code']})")
            index_layout.addWidget(name_label)
            
            price_label = QLabel(index['price'])
            index_layout.addWidget(price_label)
            
            change_label = QLabel(index['change'])
            if "+" in index['change']:
                change_label.setStyleSheet("color: red;")
            else:
                change_label.setStyleSheet("color: green;")
            index_layout.addWidget(change_label)
            
            indices_layout.addWidget(index_widget)
        
        fallback_layout.addWidget(indices_group)
        
        # 添加刷新按钮
        refresh_btn = QPushButton("刷新市场数据")
        fallback_layout.addWidget(refresh_btn)
        
        # 添加弹性空间
        fallback_layout.addStretch(1)
        
        return fallback_widget


if __name__ == "__main__":
    # 独立测试
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("中国市场分析视图测试")
    layout = QVBoxLayout(window)
    
    widget = ChinaMarketWidget(window)
    layout.addWidget(widget)
    
    window.setMinimumSize(1024, 768)
    window.show()
    
    sys.exit(app.exec_()) 