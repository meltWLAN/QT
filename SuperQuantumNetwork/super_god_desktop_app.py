#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 豪华版桌面应用
集成高级量子预测引擎、可视化分析模块和中国股市分析模块
"""

import sys
import os
import logging
import traceback
import time
import random
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QSplashScreen, QWidget, 
    QVBoxLayout, QHBoxLayout, QLabel, QStatusBar, QMessageBox, 
    QPushButton, QDockWidget, QAction, QToolBar, QSplitter, 
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QFormLayout, QMenu, QSystemTrayIcon, QProgressBar, QFrame,
    QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect, QDesktopWidget,
    QComboBox, QScrollArea, QProgressDialog, QListWidgetItem, QTextEdit,
    QLineEdit, QSpinBox, QGridLayout
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve, QObject, QPoint, QDateTime, QRect
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QPainter, QBrush, QLinearGradient, QPen
import threading
import socket
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='super_god_desktop.log'
)
logger = logging.getLogger("SuperGodSystem")


def import_or_install(package_name, import_name=None):
    """尝试导入模块，如果不存在则提示安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        return module
    except ImportError:
        print(f"缺少{package_name}模块，正在尝试安装...")
        
        try:
            from pip import main as pip_main
            pip_main(['install', package_name])
            return __import__(import_name)
        except:
            print(f"自动安装失败，请手动安装: pip install {package_name}")
            return None


# 尝试导入高级依赖 - 使用延迟导入方式，不阻塞主线程
_pyqtgraph = None
_qdarkstyle = None
_qt_material = None
_qtawesome = None
_pandas = None
_numpy = None

def get_pyqtgraph():
    global _pyqtgraph
    if _pyqtgraph is None:
        _pyqtgraph = import_or_install('pyqtgraph')
    return _pyqtgraph

def get_qdarkstyle():
    global _qdarkstyle
    if _qdarkstyle is None:
        _qdarkstyle = import_or_install('qdarkstyle')
    return _qdarkstyle

def get_qt_material():
    global _qt_material
    if _qt_material is None:
        _qt_material = import_or_install('qt-material', 'qt_material')
    return _qt_material

def get_qtawesome():
    global _qtawesome
    if _qtawesome is None:
        _qtawesome = import_or_install('qtawesome')
    return _qtawesome

def get_pandas():
    global _pandas
    if _pandas is None:
        _pandas = import_or_install('pandas')
    return _pandas

def get_numpy():
    global _numpy 
    if _numpy is None:
        _numpy = import_or_install('numpy')
    return _numpy


# 预先导入应用真正需要的库
import numpy as np
import time

# 性能追踪记录
PERFORMANCE_LOG = []

def track_performance(func):
    """装饰器：追踪函数执行时间"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        PERFORMANCE_LOG.append((func.__name__, elapsed))
        if elapsed > 0.5:  # 记录执行时间超过0.5秒的函数
            logger.info(f"性能追踪: {func.__name__} 执行时间 {elapsed:.3f}秒")
        return result
    return wrapper


# 添加初始化工作线程类
class InitializationWorker(QObject):
    """初始化任务工作器 - 在单独线程中工作"""
    finished = pyqtSignal()  # 完成信号
    modules_loaded = pyqtSignal()  # 模块加载完成信号
    controllers_initialized = pyqtSignal()  # 控制器初始化完成信号
    loading_progress = pyqtSignal(str)  # 加载进度信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
    
    def run(self):
        """执行初始化过程"""
        try:
            # 更新加载信息
            self.loading_progress.emit("导入核心模块...")
            
            # 导入模块
            self.app._import_modules()
            self.modules_loaded.emit()
            
            # 更新加载信息
            self.loading_progress.emit("初始化控制器...")
            
            # 初始化后台服务
            self.app._initialize_background_services()
            self.controllers_initialized.emit()
            
            # 更新加载信息
            self.loading_progress.emit("准备就绪")
            
            # 发出完成信号
            self.finished.emit()
        except Exception as e:
            logger.error(f"延迟初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.error_occurred.emit(f"初始化失败: {str(e)}")


class SuperGodDesktopApp(QMainWindow):
    """超神豪华版桌面应用主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化日志
        self.logger = logging.getLogger("SuperGodDesktopApp")
        # 初始化托盘图标为None，将在_setup_tray_icon中创建
        self.tray_icon = None
        
        # 记录启动时间
        self.start_time = time.time()
        self.startup_time_start = time.time()
        
        # 设置窗口属性
        self.setWindowTitle("超神量子共生网络交易系统 - 豪华版 v2.0.1 (高性能优化版)")
        self.setMinimumSize(1280, 800)
        
        # 调整默认大小以适应内容
        desktop = QDesktopWidget().availableGeometry()
        self.resize(int(min(1600, desktop.width() * 0.8)), int(min(900, desktop.height() * 0.8)))
        self.move((desktop.width() - self.width()) // 2, (desktop.height() - self.height()) // 2)
        
        # 去除默认边框，使用自定义边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # 使用延迟加载
        self.modules_loaded = False
        self.controllers_initialized = False
        
        # 显示加载画面
        self._show_loading_screen()
        
        # 初始化线程和工作对象
        self._delayed_initialization()
        
        # 创建简单的状态栏
        self.statusBar().showMessage("正在初始化系统...", 0)
        
        # 设置定时检查器
        self._init_check_timer = QTimer(self)
        self._init_check_timer.timeout.connect(self._check_initialization)
        self._init_check_timer.start(100)  # 每100ms检查一次
    
    def _delayed_initialization(self):
        """初始化和启动工作线程"""
        # 创建工作对象
        self.worker = InitializationWorker(self)
        
        # 连接信号
        self.worker.finished.connect(self._complete_initialization)
        self.worker.modules_loaded.connect(self._modules_loaded)
        self.worker.controllers_initialized.connect(self._controllers_initialized)
        self.worker.loading_progress.connect(self._update_loading_progress)
        self.worker.error_occurred.connect(self._handle_error)
        
        # 创建并配置线程
        self.init_thread = QThread()
        self.worker.moveToThread(self.init_thread)
        
        # 配置线程启动和完成处理
        self.init_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.init_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.init_thread.finished.connect(self.init_thread.deleteLater)
        
        # 启动线程
        self.init_thread.start()
    
    def _show_loading_screen(self):
        """显示加载画面"""
        # 创建临时中央部件
        temp_widget = QWidget()
        self.setCentralWidget(temp_widget)
        
        # 设置布局
        layout = QVBoxLayout(temp_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建标题
        title = QLabel("超神量子共生网络交易系统")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setStyleSheet("color: #00DDFF;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 添加子标题
        subtitle = QLabel("正在加载量子核心组件...")
        subtitle.setFont(QFont("Arial", 16))
        subtitle.setStyleSheet("color: #AAAAEE; margin-bottom: 20px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # 创建量子风格的动画组件
        quantum_widget = QWidget()
        quantum_layout = QHBoxLayout(quantum_widget)
        
        # 添加量子粒子效果
        for i in range(5):
            particle = QLabel()
            particle.setFixedSize(20, 20)
            particle.setStyleSheet(f"""
                background-color: #00DDFF;
                border-radius: 10px;
                margin: 10px;
            """)
            quantum_layout.addWidget(particle)
            
            # 创建呼吸动画效果
            animation = QPropertyAnimation(particle, b"geometry")
            animation.setDuration(1500 + i*300)
            animation.setStartValue(QRect(0, 0, 20, 20))
            animation.setEndValue(QRect(0, 0, 14, 14))
            animation.setLoopCount(-1)
            animation.setEasingCurve(QEasingCurve.InOutQuad)
            animation.start()
        
        layout.addWidget(quantum_widget, alignment=Qt.AlignCenter)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(10)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #0A0A2A;
                border-radius: 6px;
                background-color: #0F1028;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                              stop:0 #0055AA, stop:0.4 #00AAEE, 
                                              stop:0.6 #00DDFF, stop:1 #0088CC);
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 添加加载信息标签
        self.loading_info = QLabel("正在预加载系统组件...")
        self.loading_info.setAlignment(Qt.AlignCenter)
        self.loading_info.setStyleSheet("color: #888888; margin-top: 10px;")
        layout.addWidget(self.loading_info)
        
        # 添加底部空间
        layout.addStretch()
        
        # 设置背景样式
        temp_widget.setStyleSheet("""
            background-color: #070720;
            background-image: radial-gradient(circle at 50% 30%, #101040 0%, #070720 70%);
        """)
    
    def _check_initialization(self):
        """检查初始化状态"""
        # 当收到信号后直接设置标志，不再操作进度条
        if not self.modules_loaded:
            return
            
        if not self.controllers_initialized:
            return
            
        # 初始化完成，停止检查器
        self._init_check_timer.stop()
        
        # 在完成时不要从此方法直接调用_complete_initialization
        # 因为_complete_initialization已经通过信号连接到worker.finished，会被自动调用
    
    def _complete_initialization(self):
        """完成初始化 - 在主UI线程中执行"""
        # 应用样式
        self._apply_stylesheet()
        
        # 设置UI
        self._setup_ui()
        
        # 设置系统托盘
        self._setup_tray_icon()
        
        # 整合所有模块，建立模块间连接
        self._integrate_modules()
        
        # 状态栏更新定时器
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start(5000)  # 每5秒更新一次
        
        # 自动刷新市场数据 - 降低刷新频率以提高性能
        self._market_timer = QTimer(self)
        self._market_timer.timeout.connect(self._refresh_market_data)
        self._market_timer.start(300000)  # 每5分钟刷新一次
        
        # 显示启动完成消息和性能信息
        startup_time = time.time() - self.start_time
        self.statusBar().showMessage(f"超神系统启动完成，量子引擎就绪。启动用时: {startup_time:.2f}秒", 10000)
        logger.info(f"超神系统启动完成，用时: {startup_time:.2f}秒")
    
    def _modules_loaded(self):
        """模块加载完成"""
        self.modules_loaded = True
    
    def _controllers_initialized(self):
        """控制器初始化完成"""
        self.controllers_initialized = True
    
    def _update_loading_progress(self, message):
        """更新加载进度"""
        self.loading_info.setText(message)
    
    def _handle_error(self, error_message):
        """处理错误"""
        logger.error(f"延迟初始化失败: {error_message}")
        logger.error(traceback.format_exc())
        self.loading_info.setText(f"初始化失败: {error_message}")
    
    def _apply_stylesheet(self):
        """应用全局样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #050510;
                background-image: radial-gradient(circle at 50% 30%, #0A0A30 0%, #050510 70%);
            }
            
            QWidget {
                color: #BBBBDD;
            }
            
            QLabel {
                color: #BBBBDD;
            }
            
            QPushButton {
                background-color: rgba(20, 20, 50, 0.7);
                color: #BBBBDD;
                border: 1px solid #2A2A5A;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: rgba(30, 30, 70, 0.8);
                color: #00DDFF;
                border-color: #3A3A7A;
            }
            
            QPushButton:pressed {
                background-color: rgba(10, 10, 30, 0.8);
            }
            
            QGroupBox {
                border: 1px solid #222244;
                border-radius: 6px;
                margin-top: 12px;
                padding: 5px;
                background-color: rgba(10, 10, 30, 0.5);
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #00DDFF;
                font-weight: bold;
            }
            
            QLineEdit, QComboBox, QSpinBox {
                background-color: rgba(5, 5, 15, 0.7);
                color: #BBBBDD;
                border: 1px solid #222244;
                border-radius: 4px;
                padding: 3px 5px;
            }
            
            QTableWidget {
                background-color: rgba(5, 5, 15, 0.7);
                gridline-color: #222244;
                color: #BBBBDD;
                border: 1px solid #222244;
                border-radius: 4px;
            }
            
            QHeaderView::section {
                background-color: rgba(20, 20, 50, 0.7);
                color: #8888AA;
                border: 1px solid #222244;
                padding: 4px;
                font-weight: bold;
            }
            
            QScrollBar:vertical {
                background-color: rgba(10, 10, 30, 0.5);
                width: 12px;
                margin: 0px;
            }
            
            QScrollBar::handle:vertical {
                background-color: rgba(40, 40, 80, 0.7);
                min-height: 20px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: rgba(60, 60, 120, 0.9);
            }
            
            QScrollBar:horizontal {
                background-color: rgba(10, 10, 30, 0.5);
                height: 12px;
                margin: 0px;
            }
            
            QScrollBar::handle:horizontal {
                background-color: rgba(40, 40, 80, 0.7);
                min-width: 20px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background-color: rgba(60, 60, 120, 0.9);
            }
            
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
                height: 0px;
            }
            
            QScrollBar::add-page, QScrollBar::sub-page {
                background: none;
            }
        """)
    
    def _setup_ui(self):
        """设置主界面"""
        try:
            # 创建主窗口容器
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # 主布局
            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            
            # 创建自定义标题栏
            self._create_titlebar(main_layout)
            
            # 创建内容区域
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(10, 10, 10, 10)
            content_layout.setSpacing(10)
            
            # 创建选项卡
            self.tab_widget = QTabWidget()
            self.tab_widget.setDocumentMode(True)
            self.tab_widget.setTabPosition(QTabWidget.North)
            self.tab_widget.setMovable(True)
            
            # 设置选项卡样式
            self.tab_widget.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #222244;
                    background-color: rgba(8, 8, 24, 0.7);
                    border-radius: 5px;
                    padding: 5px;
                }
                
                QTabBar::tab {
                    background-color: rgba(20, 20, 40, 0.7);
                    color: #AAAACC;
                    border: 1px solid #222244;
                    border-bottom: none;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    padding: 8px 12px;
                    margin-right: 2px;
                    font-weight: bold;
                }
                
                QTabBar::tab:selected {
                    background-color: rgba(30, 30, 60, 0.8);
                    color: #00DDFF;
                    border-bottom: 2px solid #00DDFF;
                }
                
                QTabBar::tab:hover:!selected {
                    background-color: rgba(25, 25, 50, 0.8);
                    color: #BBBBEE;
                }
            """)
            
            # 添加各个标签页
            self._create_dashboard_tab()
            self._create_market_tab()
            self._create_trading_tab()
            self._create_analysis_tab()
            self._create_quantum_tab()
            self._create_settings_tab()
            
            # 添加标签页到内容区域
            content_layout.addWidget(self.tab_widget)
            
            # 添加内容区域到主布局
            main_layout.addWidget(content_widget)
            
            # 设置状态栏
            self._setup_statusbar()
            
            # 设置工具栏
            self._setup_toolbar()
            
            # 设置菜单
            self._setup_menu()
            
            # 设置托盘图标
            self._setup_tray_icon()
            
            # 设置窗口拖动
            self._setup_window_drag()
            
            # 启动周期性状态更新
            self.status_update_timer = QTimer(self)
            self.status_update_timer.timeout.connect(self._update_status_bar)
            self.status_update_timer.start(1000)  # 每秒更新状态
            
            # 设置样式表
            self._apply_stylesheet()
            
            logger.info("主界面创建完成")
        except Exception as e:
            logger.error(f"设置主界面失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _create_titlebar(self, layout):
        """创建自定义标题栏"""
        title_widget = QWidget()
        title_widget.setObjectName("titleBar")
        title_widget.setFixedHeight(50)
        
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(10, 0, 10, 0)
        
        # 系统图标 - 使用量子风格图标
        icon_label = QLabel()
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        
        # 绘制量子风格图标
        gradient = QLinearGradient(0, 0, 32, 32)
        gradient.setColorAt(0, QColor(0, 170, 255))
        gradient.setColorAt(1, QColor(0, 100, 200))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(4, 4, 24, 24)
        
        gradient = QLinearGradient(0, 0, 32, 32)
        gradient.setColorAt(0, QColor(0, 220, 255))
        gradient.setColorAt(1, QColor(0, 150, 220))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(8, 8, 16, 16)
        painter.end()
        
        icon_label.setPixmap(pixmap)
        title_layout.addWidget(icon_label)
        
        # 系统名称
        title_label = QLabel("超神量子共生网络交易系统")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("color: #00DDFF;")
        title_layout.addWidget(title_label)
        
        # 系统版本
        version_label = QLabel("v2.0.1")
        version_label.setStyleSheet("color: #5588AA; margin-left: 5px;")
        title_layout.addWidget(version_label)
        
        # 系统状态指示器
        self.system_status = QLabel("● 系统正常")
        self.system_status.setStyleSheet("color: #00FF88; margin-left: 20px;")
        title_layout.addWidget(self.system_status)
        
        title_layout.addStretch()
        
        # 控制按钮组
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)
        
        # 最小化按钮
        min_btn = QPushButton("—")
        min_btn.setFixedSize(36, 36)
        min_btn.setObjectName("minButton")
        min_btn.setStyleSheet("""
            #minButton {
                background-color: #0F1028;
                color: #AAAACC;
                border: 1px solid #222244;
                border-radius: 18px;
            }
            #minButton:hover {
                background-color: #1A1A3A;
            }
            #minButton:pressed {
                background-color: #151530;
            }
        """)
        min_btn.clicked.connect(self.showMinimized)
        
        # 最大化按钮
        max_btn = QPushButton("□")
        max_btn.setFixedSize(36, 36)
        max_btn.setObjectName("maxButton")
        max_btn.setStyleSheet("""
            #maxButton {
                background-color: #0F1028;
                color: #AAAACC;
                border: 1px solid #222244;
                border-radius: 18px;
            }
            #maxButton:hover {
                background-color: #1A1A3A;
            }
            #maxButton:pressed {
                background-color: #151530;
            }
        """)
        max_btn.clicked.connect(self._toggle_maximize)
        
        # 关闭按钮
        close_btn = QPushButton("×")
        close_btn.setFixedSize(36, 36)
        close_btn.setObjectName("closeButton")
        close_btn.setStyleSheet("""
            #closeButton {
                background-color: #0F1028;
                color: #FFAAAA;
                border: 1px solid #222244;
                border-radius: 18px;
                font-size: 16px;
            }
            #closeButton:hover {
                background-color: #AA3333;
                color: white;
            }
            #closeButton:pressed {
                background-color: #882222;
            }
        """)
        close_btn.clicked.connect(self.close)
        
        buttons_layout.addWidget(min_btn)
        buttons_layout.addWidget(max_btn)
        buttons_layout.addWidget(close_btn)
        
        title_layout.addWidget(buttons_widget)
        
        # 设置标题栏样式
        title_widget.setStyleSheet("""
            #titleBar {
                background-color: #08081A;
                border-bottom: 1px solid #222244;
            }
        """)
        
        layout.addWidget(title_widget)
    
    def _create_dashboard_tab(self):
        """创建仪表盘选项卡"""
        try:
            # 使用高级仪表盘组件
            dashboard_widget = self.create_dashboard()
            self.tab_widget.addTab(dashboard_widget, "仪表盘")
        except Exception as e:
            logger.error(f"创建仪表盘选项卡失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 创建一个空的仪表盘选项卡
            dashboard_widget = QWidget()
            dashboard_layout = QVBoxLayout(dashboard_widget)
            error_label = QLabel(f"加载仪表盘组件失败: {str(e)}")
            error_label.setStyleSheet("color: red;")
            dashboard_layout.addWidget(error_label)
            
            self.tab_widget.addTab(dashboard_widget, "仪表盘")
    
    def _create_market_tab(self):
        """创建市场分析标签页"""
        try:
            logger.info("创建市场分析标签页...")
            
            # 创建标签页容器
            market_tab = QWidget()
            market_layout = QVBoxLayout(market_tab)
            market_layout.setContentsMargins(15, 15, 15, 15)
            market_layout.setSpacing(10)
            
            # 顶部状态面板
            status_panel = QWidget()
            status_layout = QHBoxLayout(status_panel)
            status_layout.setContentsMargins(5, 5, 5, 5)
            status_layout.setSpacing(20)
            
            # 市场状态标签
            self.market_status_label = QLabel("市场状态: 加载中...")
            self.market_status_label.setStyleSheet("color: #00DDFF; font-weight: bold;")
            status_layout.addWidget(self.market_status_label)
            
            # 更新时间标签
            self.market_time_label = QLabel("更新时间: --")
            self.market_time_label.setStyleSheet("color: #00DDFF; font-weight: bold;")
            status_layout.addWidget(self.market_time_label)
            
            # 添加刷新按钮
            refresh_btn = QPushButton("刷新数据")
            refresh_btn.setFixedWidth(120)
            refresh_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 60, 120, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(0, 80, 160, 0.8);
                }
                QPushButton:pressed {
                    background-color: rgba(0, 40, 100, 0.9);
                }
            """)
            refresh_btn.clicked.connect(self._refresh_market_data)
            status_layout.addWidget(refresh_btn)
            
            # 添加到主布局
            market_layout.addWidget(status_panel)
            
            # 指数面板
            indices_group = QGroupBox("市场指数")
            indices_group.setStyleSheet("""
                QGroupBox {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            indices_layout = QVBoxLayout(indices_group)
            
            # 添加三个指数
            indices = [
                {"name": "上证指数", "code": "000001.SH", "price": "3250.55", "change": "+0.75%", "volume": "325.6亿"},
                {"name": "深证成指", "code": "399001.SZ", "price": "10765.31", "change": "+1.25%", "volume": "425.8亿"},
                {"name": "创业板指", "code": "399006.SZ", "price": "2156.48", "change": "+1.56%", "volume": "156.8亿"}
            ]
            
            for index in indices:
                index_widget = QWidget()
                index_layout = QHBoxLayout(index_widget)
                index_layout.setContentsMargins(5, 5, 5, 5)
                
                # 指数名称
                name_label = QLabel(f"{index['name']} ({index['code']})")
                name_label.setStyleSheet("color: #FFFFFF; font-weight: bold;")
                index_layout.addWidget(name_label)
                
                # 指数价格
                price_label = QLabel(index['price'])
                price_label.setStyleSheet("color: #FFFFFF; font-weight: bold; font-size: 16px;")
                index_layout.addWidget(price_label)
                
                # 涨跌幅
                change_label = QLabel(index['change'])
                if "+" in index['change']:
                    change_label.setStyleSheet("color: #FF4444; font-weight: bold;")
                else:
                    change_label.setStyleSheet("color: #44FF44; font-weight: bold;")
                index_layout.addWidget(change_label)
                
                # 成交量
                volume_label = QLabel(f"成交量: {index['volume']}")
                volume_label.setStyleSheet("color: #AAAACC;")
                index_layout.addWidget(volume_label)
                
                # 添加到指数布局
                indices_layout.addWidget(index_widget)
            
            # 添加到主布局
            market_layout.addWidget(indices_group)
            
            # 热点板块面板
            sectors_group = QGroupBox("热点板块")
            sectors_group.setStyleSheet("""
                QGroupBox {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            sectors_layout = QVBoxLayout(sectors_group)
            
            # 添加热点板块
            sectors = [
                "半导体: +2.35%",
                "新能源: +1.85%",
                "医药生物: +1.25%",
                "人工智能: +3.12%",
                "金融科技: +0.86%"
            ]
            
            for i, sector in enumerate(sectors):
                sector_label = QLabel(f"{i+1}. {sector}")
                if i < 3:
                    sector_label.setStyleSheet("color: #00FFA0; font-weight: bold;")
                else:
                    sector_label.setStyleSheet("color: #00FFCC;")
                sectors_layout.addWidget(sector_label)
            
            # 添加到主布局
            market_layout.addWidget(sectors_group)
            
            # 市场预测面板
            prediction_group = QGroupBox("量子预测")
            prediction_group.setStyleSheet("""
                QGroupBox {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            prediction_layout = QVBoxLayout(prediction_group)
            
            # 预测结果文本
            self.prediction_text = QTextEdit()
            self.prediction_text.setReadOnly(True)
            self.prediction_text.setStyleSheet("""
                QTextEdit {
                    background-color: rgba(0, 30, 60, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    padding: 5px;
                    font-family: 'Arial';
                }
            """)
            self.prediction_text.setHtml("""
                <h3 style="color:#00FFCC;">市场预测</h3>
                <p>点击"刷新数据"按钮获取最新市场数据，并进行量子预测分析。</p>
                <p>当前预测结果将显示在此处。</p>
            """)
            prediction_layout.addWidget(self.prediction_text)
            
            # 添加到主布局
            market_layout.addWidget(prediction_group)
            
            # 设置弹性空间
            market_layout.addStretch(1)
            
            # 创建定时器以定期更新市场时间
            self.market_timer = QTimer(self)
            self.market_timer.timeout.connect(self._update_market_time)
            self.market_timer.start(30000)  # 每30秒更新一次
            
            # 初始设置市场时间
            self._update_market_time()
            
            logger.info("市场分析标签页创建完成")
            
            # 添加到标签页组件
            self.tab_widget.addTab(market_tab, "中国市场")
            
            return market_tab
        except Exception as e:
            logger.error(f"创建市场分析标签页失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 创建一个空白页面并显示错误
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_label = QLabel(f"加载市场分析模块失败: {str(e)}")
            error_label.setStyleSheet("color: red;")
            error_layout.addWidget(error_label)
            
            # 添加到标签页组件
            self.tab_widget.addTab(error_tab, "中国市场")
            
            return error_tab
    
    def _create_index_panel(self, title, data_key):
        """创建指数面板"""
        panel = QGroupBox(title)
        panel.setObjectName(f"{data_key}Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 15, 10, 15)
        layout.setSpacing(8)
        
        # 指数价格
        price_label = QLabel("--")
        price_label.setObjectName(f"{data_key}Price")
        price_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(price_label)
        
        # 涨跌幅
        change_label = QLabel("--")
        change_label.setObjectName(f"{data_key}Change")
        change_label.setStyleSheet("font-size: 18px; color: #FFFFFF;")
        layout.addWidget(change_label)
        
        # 成交量
        volume_label = QLabel("成交量: --")
        volume_label.setObjectName(f"{data_key}Volume")
        layout.addWidget(volume_label)
        
        # 设置对象名称以便后续引用
        panel.price_label = price_label
        panel.change_label = change_label
        panel.volume_label = volume_label
        
        return panel
    
    def _create_sectors_panel(self):
        """创建热点板块面板"""
        panel = QGroupBox("热点板块")
        panel.setObjectName("sectorsPanel")
        layout = QVBoxLayout(panel)
        
        # 使用QTextEdit代替QListWidget
        self.sectors_list = QTextEdit()
        self.sectors_list.setObjectName("sectorsList")
        self.sectors_list.setReadOnly(True)
        self.sectors_list.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 20, 40, 0.7);
                color: #00FFCC;
                border: 1px solid #0088AA;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.sectors_list)
        
        return panel
    
    def _create_north_flow_panel(self):
        """创建北向资金面板"""
        panel = QGroupBox("北向资金")
        panel.setObjectName("northFlowPanel")
        layout = QVBoxLayout(panel)
        
        # 创建北向资金标签
        self.north_flow_label = QLabel("净流入: --")
        self.north_flow_label.setObjectName("northFlowValue")
        self.north_flow_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.north_flow_label)
        
        # 五日净流入
        self.north_flow_5d_label = QLabel("五日净流入: --")
        self.north_flow_5d_label.setObjectName("northFlow5d")
        layout.addWidget(self.north_flow_5d_label)
        
        # 资金趋势
        self.north_flow_trend_label = QLabel("趋势: --")
        self.north_flow_trend_label.setObjectName("northFlowTrend")
        layout.addWidget(self.north_flow_trend_label)
        
        return panel
    
    def _load_initial_market_data(self):
        """初始加载市场数据"""
        # 使用后台线程加载数据
        threading.Thread(target=self._async_load_market_data, daemon=True).start()
    
    def _async_load_market_data(self):
        """异步加载市场数据"""
        try:
            # 确保市场控制器已初始化
            if hasattr(self, 'market_controller') and self.market_controller:
                # 更新市场数据
                self.market_controller.update_market_data(force_update=True)
                
                # 数据更新完成后，在主线程中更新UI
                QTimer.singleShot(0, self._update_market_ui)
            else:
                logger.warning("市场控制器尚未初始化，无法加载市场数据")
        except Exception as e:
            logger.error(f"异步加载市场数据失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _update_market_ui(self):
        """更新市场UI"""
        try:
            # 确保市场控制器已初始化
            if not hasattr(self, 'market_controller') or not self.market_controller:
                return
            
            # 获取市场数据
            market_data = self.market_controller.market_data
            
            # 更新指数面板
            if hasattr(self, 'sh_index_panel'):
                self._update_index_panel(self.sh_index_panel, market_data.get('sh_index', {}))
            if hasattr(self, 'sz_index_panel'):    
                self._update_index_panel(self.sz_index_panel, market_data.get('sz_index', {}))
            if hasattr(self, 'cyb_index_panel'):
                self._update_index_panel(self.cyb_index_panel, market_data.get('cyb_index', {}))
            
            # 更新热点板块
            if hasattr(self, 'sectors_list'):
                sectors = market_data.get('sectors', {}).get('hot_sectors', [])
                self._update_sectors_list(sectors)
            
            # 更新北向资金
            if hasattr(self, 'north_flow_label'):
                north_flow = self.market_controller.latest_prediction.get('north_flow', {})
                self._update_north_flow_panel(north_flow)
            
            # 更新市场状态
            self._update_market_status()
            
            # 更新时间标签
            self._update_market_time()
            
            logger.info("市场UI更新完成")
        except Exception as e:
            logger.error(f"更新市场UI失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _update_index_panel(self, panel, data):
        """更新指数面板"""
        if not data or not hasattr(panel, 'price_label'):
            return
        
        # 更新价格
        price = data.get('close', '--')
        if isinstance(price, (int, float)):
            panel.price_label.setText(f"{price:.2f}")
        else:
            panel.price_label.setText(str(price))
        
        # 更新涨跌幅
        change_pct = data.get('change_pct', 0)
        if isinstance(change_pct, (int, float)):
            change_text = f"{change_pct:+.2f}%"
            if change_pct > 0:
                panel.change_label.setStyleSheet("font-size: 18px; color: #FF4444;")
            elif change_pct < 0:
                panel.change_label.setStyleSheet("font-size: 18px; color: #44FF44;")
            else:
                panel.change_label.setStyleSheet("font-size: 18px; color: #FFFFFF;")
            panel.change_label.setText(change_text)
        
        # 更新成交量
        volume = data.get('volume', 0)
        if isinstance(volume, (int, float)):
            if volume > 100000000:
                volume_text = f"成交量: {volume/100000000:.2f}亿手"
            else:
                volume_text = f"成交量: {volume/10000:.2f}万手"
            panel.volume_label.setText(volume_text)
    
    def _update_sectors_list(self, sectors):
        """更新热点板块列表"""
        # 确保sectors_list存在
        if not hasattr(self, 'sectors_list'):
            return
            
        # 准备HTML内容
        html = "<html><body style='color: #00FFCC;'>"
        
        # 添加热点板块
        for i, sector in enumerate(sectors):
            if i < 3:
                # 前三名使用不同颜色
                html += f"<p style='color: #00FFA0;'><b>{i+1}. {sector}</b></p>"
            else:
                html += f"<p>{i+1}. {sector}</p>"
        
        html += "</body></html>"
        
        # 设置HTML内容
        self.sectors_list.setHtml(html)
    
    def _update_north_flow_panel(self, data):
        """更新北向资金面板"""
        if not data:
            return
        
        # 净流入
        if hasattr(self, 'north_flow_label'):
            inflow = data.get('total_inflow', 0)
            if isinstance(inflow, (int, float)):
                if abs(inflow) > 100000000:
                    inflow_text = f"净流入: {inflow/100000000:.2f}亿"
                else:
                    inflow_text = f"净流入: {inflow/10000:.2f}万"
                
                # 设置颜色
                if inflow > 0:
                    self.north_flow_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF4444;")
                else:
                    self.north_flow_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #44FF44;")
                
                self.north_flow_label.setText(inflow_text)
        
        # 5日净流入
        if hasattr(self, 'north_flow_5d_label'):
            flow_5d = data.get('total_flow_5d', 0)
            if isinstance(flow_5d, (int, float)):
                if abs(flow_5d) > 100000000:
                    flow_5d_text = f"五日净流入: {flow_5d/100000000:.2f}亿"
                else:
                    flow_5d_text = f"五日净流入: {flow_5d/10000:.2f}万"
                self.north_flow_5d_label.setText(flow_5d_text)
        
        # 资金趋势
        if hasattr(self, 'north_flow_trend_label'):
            trend = data.get('flow_trend', '--')
            self.north_flow_trend_label.setText(f"趋势: {trend}")
    
    def _update_market_status(self):
        """更新市场状态"""
        try:
            # 如果有预测结果，显示预测的市场状态
            if hasattr(self, 'market_controller') and self.market_controller and hasattr(self, 'market_status_label'):
                prediction = self.market_controller.latest_prediction
                if prediction and 'risk_analysis' in prediction:
                    risk = prediction['risk_analysis'].get('overall_risk', 0.5)
                    trend = prediction['risk_analysis'].get('risk_trend', '未知')
                    
                    # 基于风险水平设置状态颜色
                    if risk < 0.3:
                        status_color = "#00FF00"  # 绿色 - 低风险
                    elif risk < 0.7:
                        status_color = "#FFFF00"  # 黄色 - 中等风险
                    else:
                        status_color = "#FF0000"  # 红色 - 高风险
                    
                    status_text = f"市场状态: {trend} (风险指数: {risk:.2f})"
                    self.market_status_label.setText(status_text)
                    self.market_status_label.setStyleSheet(f"color: {status_color};")
                else:
                    # 没有预测结果，显示基本状态
                    self.market_status_label.setText("市场状态: 已更新")
                    self.market_status_label.setStyleSheet("color: #FFFFFF;")
        except Exception as e:
            logger.error(f"更新市场状态失败: {str(e)}")
    
    def _run_market_prediction(self):
        """运行市场预测"""
        try:
            # 确保市场控制器已初始化
            if not hasattr(self, 'market_controller') or not self.market_controller:
                QMessageBox.warning(self, "预测失败", "市场控制器尚未初始化")
                return
            
            # 显示加载进度对话框
            progress_dialog = QProgressDialog("正在进行量子预测...", "取消", 0, 100, self)
            progress_dialog.setWindowTitle("量子预测")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setValue(10)
            
            # 使用后台线程进行预测
            def run_prediction():
                try:
                    # 更新市场数据
                    self.market_controller.update_market_data()
                    progress_dialog.setValue(30)
                    
                    # 使用量子引擎进行预测
                    prediction = self.market_controller.predict_market_trend(use_quantum=True)
                    progress_dialog.setValue(80)
                    
                    # 预测完成后在主线程更新UI
                    QTimer.singleShot(0, lambda: self._update_prediction_ui(prediction))
                    progress_dialog.setValue(100)
                except Exception as e:
                    logger.error(f"市场预测失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    QTimer.singleShot(0, lambda: QMessageBox.critical(self, "预测失败", f"市场预测出错: {str(e)}"))
            
            # 启动预测线程
            prediction_thread = threading.Thread(target=run_prediction)
            prediction_thread.daemon = True
            prediction_thread.start()
            
        except Exception as e:
            logger.error(f"启动市场预测失败: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "预测失败", f"启动预测出错: {str(e)}")
    
    def _update_prediction_ui(self, prediction):
        """更新预测UI显示"""
        try:
            # 更新市场状态
            self._update_market_status()
            
            # 更新预测文本
            if prediction:
                # 格式化预测结果为可读文本
                prediction_html = self._format_prediction_html(prediction)
                self.prediction_text.setHtml(prediction_html)
            else:
                self.prediction_text.setPlainText("未能获取预测结果")
        except Exception as e:
            logger.error(f"更新预测UI失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _format_prediction_html(self, prediction):
        """将预测结果格式化为HTML以提升显示效果"""
        try:
            html = "<html><body style='color:#00FFFF; font-family:Microsoft YaHei,Arial;'>"
            
            # 添加标题
            html += "<h2 style='color:#00DDFF;'>量子纠缠市场预测</h2>"
            
            # 预测时间
            prediction_time = prediction.get('quantum_enhancement_time', 
                             prediction.get('prediction_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            html += f"<p>预测时间: {prediction_time}</p>"
            
            # 是否使用量子增强
            is_quantum = prediction.get('is_quantum_enhanced', False)
            html += f"<p>预测模式: " + ('<span style="color:#00FF99;">量子增强</span>' if is_quantum else '基础分析') + "</p>"
            
            # 风险分析部分
            html += "<h3 style='color:#00CCFF;'>风险分析</h3>"
            
            # 市场趋势
            market_trend = prediction.get('market_trend', '未知')
            html += f"<p>市场趋势: <span style='font-weight:bold;'>{market_trend}</span></p>"
            
            # 风险水平
            risk_level = prediction.get('risk_level', 0.5)
            risk_color = self._get_risk_color(risk_level)
            html += f"<p>风险水平: <span style='color:{risk_color};font-weight:bold;'>{risk_level:.2f}</span></p>"
            
            # 板块轮动
            html += "<h3 style='color:#00CCFF;'>板块轮动分析</h3>"
            
            # 当前热点板块
            if 'sector_rotation' in prediction and 'current_hot_sectors' in prediction['sector_rotation']:
                hot_sectors = prediction['sector_rotation']['current_hot_sectors']
                html += "<p>当前热点板块:</p><ul>"
                for sector in hot_sectors[:5]:  # 只显示前5个
                    html += f"<li>{sector}</li>"
                html += "</ul>"
            
            # 下一轮预测板块
            if 'sector_rotation' in prediction and 'next_sectors_prediction' in prediction['sector_rotation']:
                next_sectors = prediction['sector_rotation']['next_sectors_prediction']
                html += "<p>下一轮可能热点:</p><ul>"
                for sector in next_sectors[:3]:  # 只显示前3个
                    html += f"<li>{sector}</li>"
                html += "</ul>"
            
            # 详细预测
            if 'detailed_predictions' in prediction:
                html += "<h3 style='color:#00CCFF;'>详细预测</h3>"
                html += "<table border='0' cellspacing='5'>"
                html += "<tr><th>标的</th><th>预测值</th><th>置信度</th></tr>"
                
                for entity, pred in prediction['detailed_predictions'].items():
                    if isinstance(pred, dict):
                        pred_value = pred.get('prediction', 0)
                        confidence = pred.get('confidence', 0)
                        
                        # 设置颜色
                        if pred_value > 0:
                            color = "#FF4444"  # 红色 - 正向
                        else:
                            color = "#44FF44"  # 绿色 - 负向
                        
                        html += f"<tr><td>{entity}</td><td style='color:{color};'>{pred_value:+.2f}%</td><td>{confidence:.2f}</td></tr>"
                
                html += "</table>"
            
            html += "</body></html>"
            return html
        except Exception as e:
            logger.error(f"格式化预测HTML失败: {str(e)}")
            return f"<html><body>预测结果格式化失败: {str(e)}</body></html>"
    
    def _get_risk_color(self, risk_level):
        """根据风险水平获取颜色"""
        if risk_level < 0.3:
            return "#00FF00"  # 绿色 - 低风险
        elif risk_level < 0.7:
            return "#FFFF00"  # 黄色 - 中等风险
        else:
            return "#FF0000"  # 红色 - 高风险

    def _create_trading_tab(self):
        """创建交易选项卡"""
        try:
            logger.info("创建交易标签页...")
            trading_widget = QWidget()
            trading_layout = QVBoxLayout(trading_widget)
            trading_layout.setContentsMargins(15, 15, 15, 15)
            trading_layout.setSpacing(10)
            
            # 顶部状态面板
            status_panel = QWidget()
            status_layout = QHBoxLayout(status_panel)
            status_layout.setContentsMargins(5, 5, 5, 5)
            
            # 交易状态
            trade_status_label = QLabel("交易状态: 模拟交易模式")
            trade_status_label.setObjectName("tradeStatusLabel")
            trade_status_label.setStyleSheet("color: #00DDFF; font-weight: bold;")
            status_layout.addWidget(trade_status_label)
            
            # 添加按钮
            refresh_btn = QPushButton("刷新数据")
            refresh_btn.setObjectName("tradeRefreshButton")
            refresh_btn.setFixedWidth(120)
            refresh_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 60, 120, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(0, 80, 160, 0.8);
                }
                QPushButton:pressed {
                    background-color: rgba(0, 40, 100, 0.9);
                }
            """)
            refresh_btn.clicked.connect(self._refresh_trading_data)
            status_layout.addWidget(refresh_btn)
            
            # 添加到主布局
            trading_layout.addWidget(status_panel)
            
            # 交易面板
            trade_group = QGroupBox("交易操作")
            trade_group.setObjectName("tradeOperationPanel")
            trade_group.setStyleSheet("""
                QGroupBox {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            trade_layout = QGridLayout(trade_group)
            
            # 股票代码输入
            trade_layout.addWidget(QLabel("股票代码:"), 0, 0)
            self.stock_code_input = QLineEdit()
            self.stock_code_input.setPlaceholderText("输入股票代码，如: 600001")
            self.stock_code_input.setStyleSheet("""
                QLineEdit {
                    background-color: rgba(0, 30, 60, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 4px;
                    padding: 5px;
                }
            """)
            trade_layout.addWidget(self.stock_code_input, 0, 1, 1, 2)
            
            # 数量输入
            trade_layout.addWidget(QLabel("数量:"), 1, 0)
            self.quantity_input = QSpinBox()
            self.quantity_input.setMinimum(100)
            self.quantity_input.setMaximum(100000)
            self.quantity_input.setSingleStep(100)
            self.quantity_input.setValue(100)
            self.quantity_input.setStyleSheet("""
                QSpinBox {
                    background-color: rgba(0, 30, 60, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 4px;
                    padding: 5px;
                }
            """)
            trade_layout.addWidget(self.quantity_input, 1, 1, 1, 2)
            
            # 交易按钮
            buy_btn = QPushButton("买入")
            buy_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 120, 0, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #00AA00;
                    border-radius: 4px;
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(0, 150, 0, 0.8);
                }
            """)
            
            sell_btn = QPushButton("卖出")
            sell_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(180, 0, 0, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #CC0000;
                    border-radius: 4px;
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(220, 0, 0, 0.8);
                }
            """)
            
            trade_layout.addWidget(buy_btn, 2, 1)
            trade_layout.addWidget(sell_btn, 2, 2)
            
            trading_layout.addWidget(trade_group)
            
            # 持仓面板
            positions_group = QGroupBox("当前持仓")
            positions_group.setObjectName("positionsPanel")
            positions_group.setStyleSheet("""
                QGroupBox {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            positions_layout = QVBoxLayout(positions_group)
            
            # 持仓表格
            self.positions_table = QTableWidget()
            self.positions_table.setColumnCount(5)
            self.positions_table.setHorizontalHeaderLabels(["股票代码", "股票名称", "持仓数量", "当前价格", "市值"])
            self.positions_table.setStyleSheet("""
                QTableWidget {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    gridline-color: #0066AA;
                }
                QHeaderView::section {
                    background-color: rgba(0, 40, 80, 0.8);
                    color: white;
                    border: 1px solid #0066AA;
                    padding: 4px;
                }
                QTableWidget::item {
                    border-bottom: 1px solid #0066AA;
                }
            """)
            positions_layout.addWidget(self.positions_table)
            
            trading_layout.addWidget(positions_group)
            
            # 委托订单面板
            orders_group = QGroupBox("委托订单")
            orders_group.setObjectName("ordersPanel")
            orders_group.setStyleSheet("""
                QGroupBox {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
            orders_layout = QVBoxLayout(orders_group)
            
            # 订单表格
            self.orders_table = QTableWidget()
            self.orders_table.setColumnCount(6)
            self.orders_table.setHorizontalHeaderLabels(["订单编号", "股票代码", "方向", "数量", "状态", "创建时间"])
            self.orders_table.setStyleSheet("""
                QTableWidget {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    gridline-color: #0066AA;
                }
                QHeaderView::section {
                    background-color: rgba(0, 40, 80, 0.8);
                    color: white;
                    border: 1px solid #0066AA;
                    padding: 4px;
                }
                QTableWidget::item {
                    border-bottom: 1px solid #0066AA;
                }
            """)
            orders_layout.addWidget(self.orders_table)
            
            trading_layout.addWidget(orders_group)
            
            # 填充模拟数据
            self._populate_trading_demo_data()
            
            logger.info("交易标签页创建完成")
            self.tab_widget.addTab(trading_widget, "交易")
            return trading_widget
        except Exception as e:
            logger.error(f"创建交易标签页失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 创建错误页面
            error_widget = QWidget()
            error_layout = QVBoxLayout(error_widget)
            error_label = QLabel(f"加载交易标签页失败: {str(e)}")
            error_label.setStyleSheet("color: red;")
            error_layout.addWidget(error_label)
            
            self.tab_widget.addTab(error_widget, "交易")
            return error_widget

    def _populate_trading_demo_data(self):
        """填充交易模块的演示数据"""
        try:
            # 填充持仓数据
            self.positions_table.setRowCount(3)
            
            # 模拟持仓数据
            positions = [
                {"code": "600000", "name": "浦发银行", "quantity": 1000, "price": 12.34, "value": 12340},
                {"code": "000001", "name": "平安银行", "quantity": 500, "price": 15.67, "value": 7835},
                {"code": "601318", "name": "中国平安", "quantity": 200, "price": 42.56, "value": 8512}
            ]
            
            for i, pos in enumerate(positions):
                self.positions_table.setItem(i, 0, QTableWidgetItem(pos["code"]))
                self.positions_table.setItem(i, 1, QTableWidgetItem(pos["name"]))
                self.positions_table.setItem(i, 2, QTableWidgetItem(str(pos["quantity"])))
                self.positions_table.setItem(i, 3, QTableWidgetItem(f"{pos['price']:.2f}"))
                self.positions_table.setItem(i, 4, QTableWidgetItem(f"{pos['value']:.2f}"))
            
            # 调整列宽
            self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            
            # 填充订单数据
            self.orders_table.setRowCount(4)
            
            # 模拟订单数据
            orders = [
                {"id": "ORD202311060001", "code": "600000", "direction": "买入", "quantity": 1000, "status": "已成交", "time": "2023-11-06 09:30:15"},
                {"id": "ORD202311060002", "code": "000001", "direction": "买入", "quantity": 500, "status": "已成交", "time": "2023-11-06 10:15:32"},
                {"id": "ORD202311060003", "code": "601318", "direction": "买入", "quantity": 200, "status": "已成交", "time": "2023-11-06 13:45:21"},
                {"id": "ORD202311060004", "code": "600036", "direction": "买入", "quantity": 300, "status": "委托中", "time": "2023-11-06 14:30:45"}
            ]
            
            for i, order in enumerate(orders):
                self.orders_table.setItem(i, 0, QTableWidgetItem(order["id"]))
                self.orders_table.setItem(i, 1, QTableWidgetItem(order["code"]))
                self.orders_table.setItem(i, 2, QTableWidgetItem(order["direction"]))
                self.orders_table.setItem(i, 3, QTableWidgetItem(str(order["quantity"])))
                self.orders_table.setItem(i, 4, QTableWidgetItem(order["status"]))
                self.orders_table.setItem(i, 5, QTableWidgetItem(order["time"]))
                
                # 设置已成交的行为绿色，委托中的为黄色
                if order["status"] == "已成交":
                    for j in range(6):
                        self.orders_table.item(i, j).setForeground(QColor(0, 255, 150))
                elif order["status"] == "委托中":
                    for j in range(6):
                        self.orders_table.item(i, j).setForeground(QColor(255, 200, 0))
            
            # 调整列宽
            self.orders_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            
            logger.info("交易演示数据加载完成")
        except Exception as e:
            logger.error(f"填充交易演示数据失败: {str(e)}")

    def _refresh_trading_data(self):
        """刷新交易数据"""
        try:
            # 重新加载模拟数据
            self._populate_trading_demo_data()
            QMessageBox.information(self, "刷新成功", "交易数据已更新")
            
            logger.info("交易数据已刷新")
        except Exception as e:
            logger.error(f"刷新交易数据失败: {str(e)}")
            QMessageBox.warning(self, "刷新失败", f"刷新交易数据时发生错误: {str(e)}")
    
    def _create_analysis_tab(self):
        """创建分析选项卡"""
        try:
            logger.info("创建分析标签页...")
            analysis_widget = QWidget()
            analysis_layout = QVBoxLayout(analysis_widget)
            analysis_layout.setContentsMargins(15, 15, 15, 15)
            
            # 顶部控制面板
            control_panel = QWidget()
            control_layout = QHBoxLayout(control_panel)
            
            # 股票代码输入
            control_layout.addWidget(QLabel("股票代码:"))
            self.stock_code_input = QLineEdit()
            self.stock_code_input.setPlaceholderText("输入股票代码")
            self.stock_code_input.setMaximumWidth(150)
            control_layout.addWidget(self.stock_code_input)
            
            # 分析类型选择
            control_layout.addWidget(QLabel("分析类型:"))
            self.analysis_type = QComboBox()
            self.analysis_type.addItems(["技术分析", "基本面分析", "量子分析"])
            control_layout.addWidget(self.analysis_type)
            
            # 分析按钮
            analyze_btn = QPushButton("开始分析")
            analyze_btn.clicked.connect(self._run_analysis)
            control_layout.addWidget(analyze_btn)
            
            # 添加到主布局
            analysis_layout.addWidget(control_panel)
            
            # 分析结果面板
            self.result_text = QTextEdit()
            self.result_text.setReadOnly(True)
            self.result_text.setStyleSheet("""
                QTextEdit {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                }
            """)
            analysis_layout.addWidget(self.result_text)
            
            # 添加示例分析结果
            self.result_text.setHtml("""
                <h2 style="color:#00DDFF;">股票分析报告</h2>
                <p>请在上方输入股票代码并选择分析类型，然后点击"开始分析"按钮。</p>
                <p>分析完成后，结果将显示在此处。</p>
            """)
            
            logger.info("分析标签页创建完成")
            self.tab_widget.addTab(analysis_widget, "分析")
        except Exception as e:
            logger.error(f"创建分析标签页失败: {str(e)}")
            
            # 创建一个空白页面并显示错误
            analysis_widget = QWidget()
            error_layout = QVBoxLayout(analysis_widget)
            error_label = QLabel(f"加载分析模块失败: {str(e)}")
            error_label.setStyleSheet("color: red;")
            error_layout.addWidget(error_label)
            
            self.tab_widget.addTab(analysis_widget, "分析")

    def _run_analysis(self):
        """运行股票分析"""
        stock_code = self.stock_code_input.text()
        analysis_type = self.analysis_type.currentText()
        
        if not stock_code:
            QMessageBox.warning(self, "输入错误", "请输入股票代码")
            return
        
        # 生成示例分析结果
        stock_name = "测试股票"  # 实际应用中应从数据源获取
        html_result = f"""
        <h2 style="color:#00DDFF;">{stock_name} ({stock_code}) 分析报告</h2>
        <p>分析类型: {analysis_type}</p>
        <p>分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <hr style="border-color:#0088CC;">
        
        <h3 style="color:#00FFCC;">分析结果摘要</h3>
        <p>该股票当前呈现震荡上行趋势，技术指标表现良好，建议关注。</p>
        
        <h3 style="color:#00FFCC;">详细指标</h3>
        <ul>
            <li>MA(5): 25.34 (↑)</li>
            <li>MA(10): 24.56 (↑)</li>
            <li>MA(20): 23.89 (↑)</li>
            <li>MACD: 0.45 (金叉)</li>
            <li>KDJ: K值:75.3 D值:68.2 (偏强)</li>
        </ul>
        
        <h3 style="color:#00FFCC;">风险评估</h3>
        <p>风险等级: <span style="color:#FFCC00;">中等</span></p>
        
        <hr style="border-color:#0088CC;">
        <p style="color:#AAAAAA;font-size:smaller;">本分析由超神系统生成，仅供参考，不构成投资建议</p>
        """
        
        self.result_text.setHtml(html_result)
        logger.info(f"完成对股票 {stock_code} 的{analysis_type}")
    
    def _create_quantum_tab(self):
        """创建量子引擎选项卡"""
        try:
            # 使用量子视图组件
            quantum_widget = self.create_quantum_view()
            self.tab_widget.addTab(quantum_widget, "量子引擎")
        except Exception as e:
            logger.error(f"创建量子引擎选项卡失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 创建一个空的量子引擎选项卡
            quantum_widget = QWidget()
            quantum_layout = QVBoxLayout(quantum_widget)
            error_label = QLabel(f"加载量子引擎组件失败: {str(e)}")
            error_label.setStyleSheet("color: red;")
            quantum_layout.addWidget(error_label)
            
            self.tab_widget.addTab(quantum_widget, "量子引擎")
    
    def _create_settings_tab(self):
        """创建设置选项卡"""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # 在这里添加设置组件
        
        self.tab_widget.addTab(settings_widget, "设置")
    
    def _setup_statusbar(self):
        """设置状态栏"""
        self.statusbar = self.statusBar()
        self.statusbar.setFixedHeight(28)
        
        # 设置状态栏样式
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #05051A;
                color: #AAAACC;
                border-top: 1px solid #222244;
            }
            
            QStatusBar::item {
                border: none;
                border-left: 1px solid #222244;
                padding: 2px 10px;
            }
        """)
        
        # 创建状态栏组件
        self.market_status = QLabel("市场状态: 未知")
        self.market_status.setStyleSheet("color: #AAAACC;")
        
        self.quantum_status = QLabel("量子引擎: 就绪")
        self.quantum_status.setStyleSheet("color: #00DDFF;")
        
        self.connection_status = QLabel("网络连接: 正常")
        self.connection_status.setStyleSheet("color: #00FF88;")
        
        self.memory_usage = QLabel("内存占用: 0MB")
        self.memory_usage.setStyleSheet("color: #AAAACC;")
        
        self.cpu_usage = QLabel("CPU: 0%")
        self.cpu_usage.setStyleSheet("color: #AAAACC;")
        
        # 添加组件到状态栏
        self.statusbar.addPermanentWidget(self.market_status)
        self.statusbar.addPermanentWidget(self.quantum_status)
        self.statusbar.addPermanentWidget(self.connection_status)
        self.statusbar.addPermanentWidget(self.memory_usage)
        self.statusbar.addPermanentWidget(self.cpu_usage)
        
        # 创建状态更新定时器
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_indicators)
        self._status_timer.start(3000)  # 每3秒更新一次
    
    def _update_status_indicators(self):
        """更新状态指示器"""
        # 这里可以添加真实的系统状态检测逻辑
        import random
        
        # 随机更新CPU和内存使用
        cpu = random.randint(5, 30)
        memory = random.randint(200, 800)
        
        self.cpu_usage.setText(f"CPU: {cpu}%")
        self.memory_usage.setText(f"内存占用: {memory}MB")
        
        # 更新市场状态
        market_states = ["开盘", "收盘", "午休", "盘前准备", "盘后统计"]
        market_state = random.choice(market_states)
        self.market_status.setText(f"市场状态: {market_state}")
        
        # 模拟量子引擎状态
        quantum_states = ["运行中", "就绪", "高负载"]
        quantum_weights = [0.7, 0.2, 0.1]  # 加权概率
        quantum_state = random.choices(quantum_states, weights=quantum_weights, k=1)[0]
        
        self.quantum_status.setText(f"量子引擎: {quantum_state}")
        if quantum_state == "高负载":
            self.quantum_status.setStyleSheet("color: #FFAA44;")
        else:
            self.quantum_status.setStyleSheet("color: #00DDFF;")
    
    def _setup_toolbar(self):
        """设置工具栏"""
        # 创建工具栏
        self.toolbar = self.addToolBar("主工具栏")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        # 设置工具栏样式
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #08081A;
                border: none;
                border-bottom: 1px solid #222244;
                spacing: 10px;
                padding: 5px;
            }
            
            QToolButton {
                background-color: transparent;
                color: #AAAACC;
                border: none;
                border-radius: 4px;
                padding: 5px;
            }
            
            QToolButton:hover {
                background-color: #151530;
            }
            
            QToolButton:pressed {
                background-color: #101025;
            }
        """)
        
        # 添加工具栏按钮
        refresh_action = QAction(QIcon.fromTheme("view-refresh"), "刷新数据", self)
        refresh_action.triggered.connect(self._refresh_market_data)
        self.toolbar.addAction(refresh_action)
        
        # 添加分析按钮
        analyze_action = QAction(QIcon.fromTheme("format-justify-fill"), "量子分析", self)
        analyze_action.triggered.connect(self._run_quantum_analysis)
        self.toolbar.addAction(analyze_action)
        
        # 添加导出按钮
        export_action = QAction(QIcon.fromTheme("document-save"), "导出报告", self)
        export_action.triggered.connect(self._export_data)
        self.toolbar.addAction(export_action)
        
        # 添加分隔符
        self.toolbar.addSeparator()
        
        # 添加设置按钮
        settings_action = QAction(QIcon.fromTheme("preferences-system"), "设置", self)
        settings_action.triggered.connect(self._show_settings)
        self.toolbar.addAction(settings_action)
    
    def _setup_window_drag(self):
        """设置窗口拖动"""
        self._is_dragging = False
        self._drag_position = QPoint()

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 检查是否在标题栏区域
            title_bar = self.findChild(QWidget, "titleBar")
            if title_bar and title_bar.geometry().contains(event.pos()):
                self._is_dragging = True
                self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if event.buttons() == Qt.LeftButton and self._is_dragging:
            self.move(event.globalPos() - self._drag_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
        super().mouseReleaseEvent(event)
    
    def _setup_tray_icon(self):
        """设置系统托盘图标"""
        # 创建一个自定义图标
        icon_pixmap = QPixmap(32, 32)
        icon_pixmap.fill(QColor(0, 128, 255))
        
        # 使用自定义图标
        self.tray_icon = QSystemTrayIcon(QIcon(icon_pixmap), self)
        
        # 创建托盘菜单
        tray_menu = QMenu()
        
        # 显示/隐藏
        show_action = QAction("显示窗口", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)
        
        # 退出
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        
        # 点击托盘图标显示窗口
        self.tray_icon.activated.connect(self._tray_icon_activated)
    
    def _toggle_maximize(self):
        """切换最大化/正常窗口状态"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
    
    def _tray_icon_activated(self, reason):
        """托盘图标被激活"""
        if reason == QSystemTrayIcon.Trigger:
            if self.isHidden():
                self.show()
            else:
                self.hide()
    
    def _update_status_bar(self):
        """更新状态栏信息"""
        try:
            # 更新内存使用
            import psutil
            process = psutil.Process(os.getpid())
            memory = process.memory_info().rss / 1024 / 1024
            cpu = psutil.cpu_percent()
            
            self.memory_usage.setText(f"内存占用: {memory:.1f}MB")
            self.cpu_usage.setText(f"CPU: {cpu}%")
            
            # 更新量子引擎状态
            quantum_status = "活跃" if random.random() > 0.2 else "就绪"
            self.quantum_status.setText(f"量子引擎: {quantum_status}")
            
            # 市场状态
            market_states = ["开盘", "收盘", "午休", "盘前准备", "盘后统计"]
            market_state = random.choice(market_states)
            self.market_status.setText(f"市场状态: {market_state}")
        except Exception as e:
            logger.error(f"更新状态栏失败: {str(e)}")
    
    def _refresh_market_data(self):
        """刷新市场数据"""
        self.statusBar().showMessage("正在刷新市场数据...", 2000)
        # 在这里实现数据刷新逻辑
    
    def _export_data(self):
        """导出数据"""
        # 在这里实现数据导出逻辑
        pass
    
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于超神系统",
                         """<b>超神量子共生网络交易系统 - 豪华版</b>
                         <p>版本: v2.0.0</p>
                         <p>超神系统是一款集成量子计算和人工智能的高级交易平台</p>
                         <p>©2024 超神系统开发团队</p>""")
    
    def closeEvent(self, event):
        """处理窗口关闭事件"""
        if hasattr(self, 'tray_icon') and self.tray_icon and self.tray_icon.isVisible():
            QMessageBox.information(self, "提示", "应用将继续在系统托盘中运行。")
            self.hide()
            event.ignore()
        else:
            event.accept()

    @track_performance
    def _import_modules(self):
        """导入必要模块"""
        try:
            # 导入各种控制器和视图
            from advanced_splash import SuperGodSplashScreen
            from china_market_view import ChinaMarketWidget
            from quantum_symbiotic_network.core.quantum_entanglement_engine import QuantumEntanglementEngine
            from market_controllers import MarketDataController, TradeController
            from dashboard_module import create_dashboard
            from quantum_view import create_quantum_view
            
            # 存储引用
            self.splash_screen_class = SuperGodSplashScreen
            self.china_market_widget_class = ChinaMarketWidget
            self.quantum_engine_class = QuantumEntanglementEngine
            self.market_controller_class = MarketDataController
            self.trade_controller_class = TradeController
            self.create_dashboard = create_dashboard
            self.create_quantum_view = create_quantum_view
            
            logger.info("成功导入所有模块")
        except Exception as e:
            logger.error(f"模块导入失败: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "初始化失败", f"模块导入失败: {str(e)}")
    
    @track_performance
    def _initialize_background_services(self):
        """初始化后台服务"""
        try:
            # 创建控制器实例 - 使用惰性初始化
            self.market_controller = self.market_controller_class(use_ai=False)  # 初始不使用AI降低启动时间
            self.trade_controller = self.trade_controller_class(market_controller=self.market_controller)
            
            # 创建量子引擎但延迟初始化量子态
            self.quantum_engine = self.quantum_engine_class(dimensions=8, entanglement_factor=0.3)
            
            # 定时加载剩余服务
            QTimer.singleShot(5000, self._initialize_advanced_services)
            
            logger.info("初始化基础后台服务完成")
        except Exception as e:
            logger.error(f"后台服务初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "初始化失败", f"后台服务初始化失败: {str(e)}")
    
    def _initialize_advanced_services(self):
        """初始化高级服务 - 会在主界面显示后的5秒后执行"""
        try:
            # 启用AI预测功能
            if hasattr(self.market_controller, 'enable_ai_prediction'):
                self.market_controller.enable_ai_prediction()
            
            # 初始化量子态
            market_entities = ["AAPL", "MSFT", "GOOGL", "AMZN", "000001.SH", "399001.SZ"]
            
            # 设置优先实体
            if hasattr(self.quantum_engine, 'set_priority_entities'):
                self.quantum_engine.set_priority_entities(["000001.SH", "399001.SZ"])
                
            # 初始化量子态
            if hasattr(self.quantum_engine, '_initialize_quantum_states'):
                self.quantum_engine._initialize_quantum_states(market_entities)
            
            logger.info("高级服务初始化完成")
            
            # 使用QTimer.singleShot确保在主线程中更新UI
            QTimer.singleShot(0, lambda: self.statusBar().showMessage("高级量子服务初始化完成", 3000))
        except Exception as e:
            logger.error(f"高级服务初始化失败: {str(e)}")

    def _run_quantum_analysis(self):
        """运行量子分析"""
        QMessageBox.information(self, "量子分析", "量子分析功能即将推出...")

    def _show_settings(self):
        """显示设置对话框"""
        QMessageBox.information(self, "设置", "设置功能即将推出...")

    def _setup_menu(self):
        """设置菜单栏"""
        # 创建菜单栏
        menu_bar = self.menuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu("文件")
        
        # 导出数据
        export_action = QAction("导出数据", self)
        export_action.triggered.connect(self._export_data)
        file_menu.addAction(export_action)
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menu_bar.addMenu("视图")
        
        # 刷新数据
        refresh_action = QAction("刷新数据", self)
        refresh_action.triggered.connect(self._refresh_market_data)
        view_menu.addAction(refresh_action)
        
        # 工具菜单
        tools_menu = menu_bar.addMenu("工具")
        
        # 量子分析
        analysis_action = QAction("量子分析", self)
        analysis_action.triggered.connect(self._run_quantum_analysis)
        tools_menu.addAction(analysis_action)
        
        # 设置
        settings_action = QAction("设置", self)
        settings_action.triggered.connect(self._show_settings)
        tools_menu.addAction(settings_action)
        
        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _integrate_modules(self):
        """整合所有模块，创建模块间的量子纠缠连接"""
        try:
            logger.info("开始整合系统模块，建立量子纠缠...")
            
            # 1. 将市场控制器与量子引擎连接
            if hasattr(self, 'market_controller') and hasattr(self, 'quantum_engine'):
                # 如果市场控制器支持设置量子引擎
                if hasattr(self.market_controller, 'set_quantum_engine'):
                    self.market_controller.set_quantum_engine(self.quantum_engine)
                    logger.info("市场控制器与量子引擎连接成功")
                
                # 如果存在量子AI模块，设置关联
                if hasattr(self.market_controller, 'ai_engine') and hasattr(self.market_controller.ai_engine, 'set_quantum_engine'):
                    self.market_controller.ai_engine.set_quantum_engine(self.quantum_engine)
                    logger.info("量子AI引擎与量子引擎连接成功")
            
            # 2. 连接交易控制器与量子引擎
            if hasattr(self, 'trade_controller') and hasattr(self, 'quantum_engine'):
                if hasattr(self.trade_controller, 'set_quantum_engine'):
                    self.trade_controller.set_quantum_engine(self.quantum_engine)
                    logger.info("交易控制器与量子引擎连接成功")
            
            # 3. 整合市场和交易模块
            if hasattr(self, 'market_controller') and hasattr(self, 'trade_controller'):
                # 已经在初始化中设置了market_controller，确保再次检查
                if not hasattr(self.trade_controller, 'market_controller') or self.trade_controller.market_controller is None:
                    self.trade_controller.market_controller = self.market_controller
                    logger.info("交易控制器与市场控制器连接成功")
            
            # 4. 连接UI组件与后端控制器
            # 标签页内的市场分析组件
            if hasattr(self, 'market_widget'):
                if self.market_widget.controller is None:
                    self.market_widget.controller = self.market_controller
                    logger.info("市场分析界面与市场控制器连接成功")
            
            # 5. 初始化量子纠缠网络
            if hasattr(self, 'quantum_engine'):
                # 创建示例市场实体列表
                market_entities = [
                    "000001.SH", "399001.SZ", "399006.SZ",  # 主要指数
                    "600000.SH", "600030.SH", "600036.SH",  # 样本股票
                    "AAPL", "MSFT", "GOOGL"                # 国际股票
                ]
                
                # 创建基础关联矩阵
                correlation_matrix = {}
                for i, entity1 in enumerate(market_entities):
                    for j, entity2 in enumerate(market_entities):
                        if i != j:
                            # 生成一个基于位置的相关性值
                            correlation = 0.3 + 0.4 * np.exp(-abs(i-j)/3)
                            correlation_matrix[(entity1, entity2)] = correlation
                
                # 初始化量子纠缠关系
                self.quantum_engine.initialize_entanglement(market_entities, correlation_matrix)
                logger.info(f"量子纠缠网络初始化完成，纠缠了{len(market_entities)}个市场实体")
            
            # 6. 启用所有UI组件的自动刷新
            self._enable_auto_refresh()
            
            logger.info("所有模块整合完成，系统处于共生状态")
            self.statusBar().showMessage("量子纠缠网络初始化完成，所有模块处于共生状态", 5000)
        except Exception as e:
            logger.error(f"模块整合失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.statusBar().showMessage(f"模块整合部分失败: {str(e)}", 5000)
    
    def _enable_auto_refresh(self):
        """启用所有相关组件的自动刷新功能"""
        try:
            # 为市场视图启用自动刷新
            if hasattr(self, 'market_widget') and hasattr(self.market_widget, 'set_auto_refresh'):
                self.market_widget.set_auto_refresh(True, interval_minutes=1)
                logger.info("市场数据自动刷新已启用")
            
            # 为量子视图启用自动刷新
            if self.tab_widget.count() >= 5:  # 如果有足够的标签页
                quantum_tab = self.tab_widget.widget(4)  # 量子引擎标签页
                if hasattr(quantum_tab, 'set_auto_refresh'):
                    quantum_tab.set_auto_refresh(True, interval_minutes=2)
                    logger.info("量子引擎视图自动刷新已启用")
        except Exception as e:
            logger.error(f"启用自动刷新失败: {str(e)}")
    
    def _refresh_market_data(self):
        """刷新市场数据"""
        try:
            self.statusBar().showMessage("正在刷新市场数据...", 2000)
            
            # 刷新市场数据
            if hasattr(self, 'market_controller'):
                self.market_controller.update_market_data(force_update=True)
                logger.info("市场数据已刷新")
            
            # 如果存在市场视图，刷新界面
            if hasattr(self, 'market_widget') and hasattr(self.market_widget, 'refresh_data'):
                self.market_widget.refresh_data()
                logger.info("市场视图已刷新")
            
            # 更新市场时间
            self._update_market_time()
            
            # 触发量子纠缠更新
            if hasattr(self, 'quantum_engine') and hasattr(self.market_controller, 'market_data'):
                # 使用市场数据更新量子状态
                try:
                    self._update_quantum_state_with_market_data()
                    logger.info("量子状态已根据市场数据更新")
                except Exception as e:
                    logger.error(f"更新量子状态失败: {str(e)}")
            
            self.statusBar().showMessage("市场数据刷新完成", 2000)
        except Exception as e:
            logger.error(f"刷新市场数据失败: {str(e)}")
            self.statusBar().showMessage(f"刷新失败: {str(e)}", 2000)
    
    def _update_quantum_state_with_market_data(self):
        """使用最新的市场数据更新量子状态"""
        if not hasattr(self, 'quantum_engine') or not hasattr(self, 'market_controller'):
            return
            
        market_data = self.market_controller.market_data
        if not market_data:
            return
            
        # 提取指数变化百分比
        sh_change = market_data.get('sh_index', {}).get('change_pct', 0)
        sz_change = market_data.get('sz_index', {}).get('change_pct', 0)
        cyb_change = market_data.get('cyb_index', {}).get('change_pct', 0)
        
        # 提取北向资金数据
        north_flow = market_data.get('north_flow', 0)
        
        # 创建市场状态向量
        market_state = {
            '000001.SH': sh_change / 100.0,  # 转换为小数
            '399001.SZ': sz_change / 100.0,
            '399006.SZ': cyb_change / 100.0,
            'north_flow': north_flow / 10000.0 if north_flow else 0  # 归一化
        }
        
        # 如果量子引擎有更新市场状态的方法，调用它
        if hasattr(self.quantum_engine, 'update_market_state'):
            self.quantum_engine.update_market_state(market_state)
        elif hasattr(self.quantum_engine, 'update_quantum_states'):
            # 转换为量子引擎可接受的格式
            states = {}
            for entity, value in market_state.items():
                states[entity] = {
                    'value': value,
                    'weight': 1.0,
                    'phase': 0.0
                }
            self.quantum_engine.update_quantum_states(states)

    def _update_market_time(self):
        """更新市场时间"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.market_time_label.setText(f"更新时间: {current_time}")

    def _apply_quantum_style_to(self, widget):
        """应用量子风格到指定组件"""
        if isinstance(widget, QLabel):
            widget.setStyleSheet("""
                color: #00DDFF;
                font-weight: bold;
                font-family: 'Microsoft YaHei', Arial;
            """)
        elif isinstance(widget, QPushButton):
            widget.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 60, 120, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 4px;
                    padding: 5px;
                    font-family: 'Microsoft YaHei', Arial;
                }
                QPushButton:hover {
                    background-color: rgba(0, 80, 160, 0.8);
                }
                QPushButton:pressed {
                    background-color: rgba(0, 40, 100, 0.9);
                }
            """)
        elif isinstance(widget, QLineEdit):
            widget.setStyleSheet("""
                QLineEdit {
                    background-color: rgba(0, 30, 60, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 4px;
                    padding: 5px;
                    font-family: 'Microsoft YaHei', Arial;
                }
            """)
        elif isinstance(widget, QComboBox):
            widget.setStyleSheet("""
                QComboBox {
                    background-color: rgba(0, 30, 60, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 150px;
                    font-family: 'Microsoft YaHei', Arial;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 20px;
                    border-left: 1px solid #0088CC;
                }
                QComboBox QAbstractItemView {
                    background-color: rgba(0, 20, 40, 0.9);
                    color: #00DDFF;
                    selection-background-color: rgba(0, 80, 120, 0.8);
                }
            """)
        elif isinstance(widget, QGroupBox):
            widget.setStyleSheet("""
                QGroupBox {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #FFFFFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                    font-family: 'Microsoft YaHei', Arial;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
        elif isinstance(widget, QTextEdit):
            widget.setStyleSheet("""
                QTextEdit {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    padding: 5px;
                    font-family: 'Microsoft YaHei', Arial;
                }
            """)
        elif isinstance(widget, QTableWidget):
            widget.setStyleSheet("""
                QTableWidget {
                    background-color: rgba(0, 20, 40, 0.7);
                    color: #00DDFF;
                    border: 1px solid #0088CC;
                    border-radius: 5px;
                    gridline-color: #0066AA;
                    font-family: 'Microsoft YaHei', Arial;
                }
                QHeaderView::section {
                    background-color: rgba(0, 40, 80, 0.8);
                    color: white;
                    border: 1px solid #0066AA;
                    padding: 4px;
                }
                QTableWidget::item {
                    border-bottom: 1px solid #0066AA;
                }
            """)
        else:
            # 为其他类型的组件应用通用样式
            widget.setStyleSheet("""
                background-color: rgba(0, 20, 40, 0.7);
                color: #00DDFF;
                font-family: 'Microsoft YaHei', Arial;
            """)
                
        return widget


def show_splash_screen():
    """显示启动画面"""
    try:
        # 尝试导入高级启动画面
        from advanced_splash import SuperGodSplashScreen
        
        splash = SuperGodSplashScreen()
        splash.show()
        
        # 模拟加载过程
        for i in range(0, 101, 5):
            if i == 0:
                splash.progressChanged.emit(i, "正在初始化系统...")
            elif i == 20:
                splash.progressChanged.emit(i, "加载量子共生网络...")
            elif i == 40:
                splash.progressChanged.emit(i, "初始化市场数据模块...")
            elif i == 60:
                splash.progressChanged.emit(i, "启动量子预测引擎...")
            elif i == 80:
                splash.progressChanged.emit(i, "连接交易接口...")
            elif i == 95:
                splash.progressChanged.emit(i, "准备就绪...")
            else:
                splash.progressChanged.emit(i, "")
            
            # 处理事件和延时
            QApplication.processEvents()
            time.sleep(0.1)
        
        # 完成
        splash.progressChanged.emit(100, "启动完成")
        time.sleep(0.5)
        splash.finished.emit()
        
        return splash
    
    except Exception as e:
        # 使用基本的启动画面
        logger.error(f"高级启动画面加载失败，使用简单启动画面: {str(e)}")
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor(10, 10, 24))
        
        # 绘制简单的启动画面
        painter = QPainter(pixmap)
        painter.setPen(QColor(0, 170, 255))
        painter.setFont(QFont("Arial", 20, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "超神系统启动中...")
        painter.end()
        
        splash = QSplashScreen(pixmap)
        splash.show()
        
        # 模拟加载过程
        for i in range(0, 101, 10):
            splash.showMessage(f"加载中... {i}%", Qt.AlignBottom | Qt.AlignCenter, QColor(0, 221, 255))
            QApplication.processEvents()
            time.sleep(0.2)
        
        return splash


# 添加单实例检查功能
def is_already_running():
    """检查程序是否已经在运行
    
    Returns:
        bool: 如果程序已经在运行返回True，否则返回False
    """
    # 使用锁文件方式
    lock_file_path = os.path.join(tempfile.gettempdir(), "super_god_desktop.lock")
    
    # 检查锁文件是否存在
    if os.path.exists(lock_file_path):
        try:
            # 尝试读取锁文件中的进程ID
            with open(lock_file_path, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查该进程是否还在运行
            try:
                # 在Unix系统中，如果进程存在，os.kill(pid, 0)不会发送信号但会做权限检查
                os.kill(pid, 0)
                # 如果执行到这里，进程仍在运行
                return True
            except OSError:
                # 进程不存在，可以删除旧的锁文件
                os.remove(lock_file_path)
                # 继续创建新的锁文件
        except (ValueError, IOError):
            # 锁文件损坏，删除它
            os.remove(lock_file_path)
    
    # 创建新的锁文件
    try:
        with open(lock_file_path, 'w') as f:
            f.write(str(os.getpid()))
        return False
    except IOError:
        logger.error("无法创建锁文件")
        return False

def cleanup_lock_file():
    """清理锁文件"""
    lock_file_path = os.path.join(tempfile.gettempdir(), "super_god_desktop.lock")
    if os.path.exists(lock_file_path):
        try:
            os.remove(lock_file_path)
        except OSError:
            pass

def main():
    """主函数"""
    try:
        # 检查是否已有实例在运行
        if is_already_running():
            print("超神系统已经在运行中，请勿重复启动")
            QMessageBox.warning(None, "重复启动", "超神系统已经在运行中，请勿重复启动")
            return 0
        
        # 注册退出时清理锁文件
        import atexit
        atexit.register(cleanup_lock_file)
        
        # 创建应用
        app = QApplication(sys.argv)
        app.setApplicationName("超神量子共生网络交易系统")
        app.setOrganizationName("SuperGodDev")
        
        # 显示启动画面
        splash = show_splash_screen()
        
        # 创建主窗口
        main_window = SuperGodDesktopApp()
        
        # 关闭启动画面，显示主窗口
        splash.close()
        main_window.show()
        
        # 运行应用
        return app.exec_()
    
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 显示错误消息
        if 'app' in locals():
            QMessageBox.critical(None, "启动失败", f"应用启动失败: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 