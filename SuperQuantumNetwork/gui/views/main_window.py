#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 主窗口
实现系统的主界面，整合各个功能模块
"""

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QStatusBar, QAction, QToolBar, QSplitter, QMessageBox, QFrame, QProgressBar, QPushButton, QSizePolicy, QApplication,
    QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QSequentialAnimationGroup, QPoint
from PyQt5.QtGui import QIcon, QFont
import qtawesome as qta
import logging
from datetime import datetime

class SuperTradingMainWindow(QMainWindow):
    """超级交易系统主窗口"""
    
    def __init__(self, data_controller, trading_controller, parent=None):
        super().__init__(parent)
        
        # 保存控制器引用
        self.data_controller = data_controller
        self.trading_controller = trading_controller
        
        # 设置基本属性
        self.setWindowTitle("超神量子共生网络交易系统 v0.2.0")
        self.resize(1280, 800)
        
        # 设置UI
        self._setup_ui()
        
        # 设置响应式布局
        self._setup_responsive_layout()
        
        # 显示欢迎消息
        self.statusBar().showMessage("系统准备就绪")
    
    def _setup_ui(self):
        """设置UI"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 顶部信息栏 - 显示关键市场指标
        top_info_bar = QFrame()
        top_info_bar.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0A0A18, stop:0.5 #101028, stop:1 #0A0A18); border-bottom: 1px solid #303050;")
        top_info_bar.setFixedHeight(40)
        top_info_bar_layout = QHBoxLayout(top_info_bar)
        top_info_bar_layout.setContentsMargins(10, 0, 10, 0)
        
        # 添加几个关键市场指标标签
        for indicator, value, color in [
            ("量子共生指数", "2186.45 (+2.41%)", "#00FF88"),
            ("市场纠缠度", "78.6%", "#00DDFF"),
            ("全球市场状态", "活跃", "#FFAA00"),
            ("量子预测准确度", "97.3%", "#00FF88"),
            ("神经网络强度", "9.8/10", "#00DDFF")
        ]:
            indicator_widget = QFrame()
            indicator_layout = QHBoxLayout(indicator_widget)
            indicator_layout.setContentsMargins(0, 0, 0, 0)
            indicator_layout.setSpacing(5)
            
            indicator_name = QLabel(f"{indicator}:")
            indicator_name.setStyleSheet("color: #AAAAFF; font-weight: bold;")
            indicator_layout.addWidget(indicator_name)
            
            indicator_value = QLabel(value)
            indicator_value.setStyleSheet(f"color: {color}; font-weight: bold;")
            indicator_layout.addWidget(indicator_value)
            
            top_info_bar_layout.addWidget(indicator_widget)
        
        top_info_bar_layout.addStretch()
        
        # 添加时间和刷新按钮
        current_time = QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        current_time.setStyleSheet("color: #FFFFFF;")
        top_info_bar_layout.addWidget(current_time)
        
        refresh_btn = QPushButton()
        refresh_btn.setIcon(qta.icon('fa5s.sync'))
        refresh_btn.setFixedSize(30, 30)
        refresh_btn.setStyleSheet("background-color: transparent; border: none;")
        refresh_btn.clicked.connect(self._refresh_data)
        top_info_bar_layout.addWidget(refresh_btn)
        
        # 每秒更新时间
        timer = QTimer(self)
        timer.timeout.connect(lambda: current_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        timer.start(1000)
        
        main_layout.addWidget(top_info_bar)
        
        # 创建主选项卡部件
        self.main_tab = QTabWidget()
        self.main_tab.setTabPosition(QTabWidget.North)
        self.main_tab.setDocumentMode(True)
        main_layout.addWidget(self.main_tab)
        
        # 添加选项卡切换动画
        self.main_tab.currentChanged.connect(self._animate_tab_change)
        
        # 创建并添加市场视图选项卡
        try:
            from gui.views.market_view import MarketView
            self.market_view = MarketView(self.data_controller)
            self.main_tab.addTab(self.market_view, qta.icon('fa5s.chart-line'), "市场")
            logging.info("成功加载市场视图")
        except Exception as e:
            market_tab = QWidget()
            market_layout = QVBoxLayout(market_tab)
            market_layout.addWidget(QLabel(f"市场视图加载失败: {str(e)}"))
            self.main_tab.addTab(market_tab, qta.icon('fa5s.chart-line'), "市场")
            logging.error(f"加载市场视图失败: {str(e)}")
        
        # 创建并添加交易视图选项卡
        try:
            from gui.views.trading_view import TradingView
            self.trading_view = TradingView(self.trading_controller)
            self.main_tab.addTab(self.trading_view, qta.icon('fa5s.exchange-alt'), "交易")
            logging.info("成功加载交易视图")
        except Exception as e:
            trading_tab = QWidget()
            trading_layout = QVBoxLayout(trading_tab)
            trading_layout.addWidget(QLabel(f"交易视图加载失败: {str(e)}"))
            self.main_tab.addTab(trading_tab, qta.icon('fa5s.exchange-alt'), "交易")
            logging.error(f"加载交易视图失败: {str(e)}")
        
        # 创建并添加量子网络视图选项卡 - 使用超神级视图
        try:
            from gui.views.quantum_view import SuperQuantumNetworkView
            self.quantum_view = SuperQuantumNetworkView()
            self.main_tab.addTab(self.quantum_view, qta.icon('fa5s.atom'), "超神量子网络")
            logging.info("成功加载超神量子网络视图")
        except ImportError as e:
            logging.error(f"缺少量子网络视图依赖: {str(e)}")
            # 创建一个简单的备用量子网络视图
            quantum_tab = QWidget()
            quantum_layout = QVBoxLayout(quantum_tab)
            
            title_label = QLabel("超神量子网络")
            title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: cyan;")
            title_label.setAlignment(Qt.AlignCenter)
            quantum_layout.addWidget(title_label)
            
            error_label = QLabel(f"量子网络视图加载失败: 缺少必要依赖\n{str(e)}")
            error_label.setStyleSheet("color: red; margin: 20px;")
            error_label.setAlignment(Qt.AlignCenter)
            quantum_layout.addWidget(error_label)
            
            install_label = QLabel("请安装以下依赖后重试:\npip install pyqtgraph numpy pandas")
            install_label.setStyleSheet("color: yellow; font-weight: bold; margin: 10px;")
            install_label.setAlignment(Qt.AlignCenter)
            quantum_layout.addWidget(install_label)
            
            # 添加一个简单的进度指示器
            status_frame = QFrame()
            status_frame.setFrameShape(QFrame.StyledPanel)
            status_frame.setStyleSheet("background-color: #1a1a1a; padding: 20px;")
            status_layout = QVBoxLayout(status_frame)
            
            status_label = QLabel("量子网络状态: 简化模式")
            status_label.setStyleSheet("color: orange; font-weight: bold;")
            status_layout.addWidget(status_label)
            
            for metric in ["量子纠缠度", "预测精度", "网络性能"]:
                metric_layout = QHBoxLayout()
                metric_label = QLabel(f"{metric}:")
                metric_layout.addWidget(metric_label)
                
                progress = QProgressBar()
                progress.setRange(0, 100)
                progress.setValue(80)
                metric_layout.addWidget(progress)
                
                status_layout.addLayout(metric_layout)
            
            quantum_layout.addWidget(status_frame)
            
            # 创建一个模拟刷新按钮
            refresh_button = QPushButton("刷新量子网络")
            refresh_button.clicked.connect(lambda: self.statusBar().showMessage("量子网络简化模式下无法刷新，请安装依赖后重试", 3000))
            quantum_layout.addWidget(refresh_button)
            
            quantum_layout.addStretch(1)
            self.main_tab.addTab(quantum_tab, qta.icon('fa5s.atom'), "量子网络")
        except Exception as e:
            quantum_tab = QWidget()
            quantum_layout = QVBoxLayout(quantum_tab)
            quantum_layout.addWidget(QLabel(f"量子网络视图加载失败: {str(e)}"))
            self.main_tab.addTab(quantum_tab, qta.icon('fa5s.atom'), "量子网络")
            logging.error(f"加载量子网络视图失败: {str(e)}")
        
        # 创建并添加投资组合视图选项卡
        try:
            from gui.views.portfolio_view import PortfolioView
            self.portfolio_view = PortfolioView()
            self.main_tab.addTab(self.portfolio_view, qta.icon('fa5s.briefcase'), "投资组合")
            logging.info("成功加载投资组合视图")
        except Exception as e:
            portfolio_tab = QWidget()
            portfolio_layout = QVBoxLayout(portfolio_tab)
            portfolio_layout.addWidget(QLabel(f"投资组合视图加载失败: {str(e)}"))
            self.main_tab.addTab(portfolio_tab, qta.icon('fa5s.briefcase'), "投资组合")
            logging.error(f"加载投资组合视图失败: {str(e)}")
        
        # 创建底部状态信息面板
        bottom_status = QWidget()
        bottom_status.setFixedHeight(30)
        bottom_status.setStyleSheet("background-color: #0A0A18; border-top: 1px solid #303050;")
        bottom_layout = QHBoxLayout(bottom_status)
        bottom_layout.setContentsMargins(10, 0, 10, 0)
        bottom_layout.setSpacing(20)
        
        # 添加状态信息
        status_icons = [
            ("连接状态", "已连接", "green", "fa5s.plug"),
            ("量子网络", "活跃", "cyan", "fa5s.atom"),
            ("AI引擎", "运行中", "#00FF88", "fa5s.brain"),
            ("数据同步", "已同步", "#AAFFAA", "fa5s.sync"),
            ("系统负载", "32%", "#FFAA00", "fa5s.microchip")
        ]
        
        for label, value, color, icon in status_icons:
            status_widget = QFrame()
            status_layout = QHBoxLayout(status_widget)
            status_layout.setContentsMargins(0, 0, 0, 0)
            status_layout.setSpacing(5)
            
            # 添加图标
            icon_label = QLabel()
            icon_label.setPixmap(qta.icon(icon, color=color).pixmap(16, 16))
            status_layout.addWidget(icon_label)
            
            # 添加状态名称
            name_label = QLabel(f"{label}:")
            name_label.setStyleSheet("color: #AAAAAA;")
            status_layout.addWidget(name_label)
            
            # 添加状态值
            value_label = QLabel(value)
            value_label.setStyleSheet(f"color: {color};")
            status_layout.addWidget(value_label)
            
            bottom_layout.addWidget(status_widget)
        
        # 添加伸展
        bottom_layout.addStretch(1)
        
        # 添加版本信息
        version_label = QLabel("超神量子共生网络交易系统 v0.2.0")
        version_label.setStyleSheet("color: #AAAAAA;")
        bottom_layout.addWidget(version_label)
        
        main_layout.addWidget(bottom_status)
        
        # 创建状态栏
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # 创建工具栏
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setMovable(False)
        toolbar.setStyleSheet("background-color: #0A0A18; border-bottom: 1px solid #303050;")
        self.addToolBar(toolbar)
        
        # 添加工具栏按钮
        toolbar_actions = [
            ("fa5s.cog", "设置", self._show_settings),
            ("fa5s.question-circle", "帮助", self._show_help),
            ("fa5s.sync", "刷新", self._refresh_data),
            ("fa5s.search", "搜索", lambda: self.statusBar().showMessage("搜索功能尚未实现", 3000)),
            ("fa5s.bell", "通知", lambda: self.statusBar().showMessage("通知功能尚未实现", 3000)),
            ("fa5s.user-circle", "用户", lambda: self.statusBar().showMessage("用户功能尚未实现", 3000))
        ]
        
        for icon_name, tooltip, callback in toolbar_actions:
            action = QAction(qta.icon(icon_name), tooltip, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)
            
        # 添加分隔符和伸展
        toolbar.addSeparator()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)
        
        # 添加网络状态指示器
        network_status = QLabel("超神网络: 在线")
        network_status.setStyleSheet("color: #00FF88; font-weight: bold; padding-right: 10px;")
        toolbar.addWidget(network_status)
    
    def _setup_responsive_layout(self):
        """设置响应式布局，使界面能够适应不同屏幕尺寸"""
        # 获取屏幕尺寸
        screen_size = QApplication.desktop().availableGeometry().size()
        
        # 根据屏幕宽度进行适配
        if screen_size.width() <= 1024:
            # 小屏幕模式
            self.setMinimumSize(800, 600)
            font = self.font()
            font.setPointSize(font.pointSize() - 1)  # 减小字体
            self.setFont(font)
            logging.info(f"应用小屏幕模式: {screen_size.width()}x{screen_size.height()}")
            
        elif screen_size.width() <= 1440:
            # 中等屏幕模式
            self.setMinimumSize(1024, 720)
            logging.info(f"应用中等屏幕模式: {screen_size.width()}x{screen_size.height()}")
            
        else:
            # 大屏幕模式
            self.setMinimumSize(1280, 800)
            font = self.font()
            font.setPointSize(font.pointSize() + 1)  # 增大字体
            self.setFont(font)
            logging.info(f"应用大屏幕模式: {screen_size.width()}x{screen_size.height()}")
        
        # 设置是否自动最大化
        if screen_size.width() >= 1920:
            self.showMaximized()
        
        # 设置分隔器比例
        for i in range(self.main_tab.count()):
            tab = self.main_tab.widget(i)
            # 查找所有分隔器并设置合适的比例
            self._adjust_splitters(tab)
    
    def _adjust_splitters(self, widget):
        """递归调整控件中的所有分隔器"""
        if isinstance(widget, QSplitter):
            # 对于垂直分隔器，通常是上方占60%，下方占40%
            if widget.orientation() == Qt.Vertical:
                widget.setStretchFactor(0, 60)
                widget.setStretchFactor(1, 40)
            # 对于水平分隔器，通常是左侧占30%，右侧占70%
            else:
                widget.setStretchFactor(0, 30)
                widget.setStretchFactor(1, 70)
        
        # 递归查找所有子控件
        for child in widget.findChildren(QWidget):
            self._adjust_splitters(child)
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 当窗口尺寸改变时，调整UI布局
        window_width = event.size().width()
        
        # 根据窗口宽度调整布局
        if window_width < 1000:
            # 小宽度模式 - 某些元素可能需要隐藏或折叠
            pass
        elif window_width < 1400:
            # 中等宽度模式
            pass
        else:
            # 大宽度模式
            pass
    
    def _show_settings(self):
        """显示设置对话框"""
        QMessageBox.information(self, "设置", "设置功能尚未实现")
    
    def _show_help(self):
        """显示帮助对话框"""
        QMessageBox.information(self, "帮助", "超神量子共生网络交易系统 v0.2.0\n\n这是一个基于量子共生网络的交易系统。\n请参阅README_DESKTOP.md获取更多信息。")
    
    def _animate_tab_change(self, index):
        """选项卡切换动画"""
        current_widget = self.main_tab.widget(index)
        if current_widget is None:
            return
            
        # 创建不透明度动画
        opacity_effect = QGraphicsOpacityEffect(current_widget)
        current_widget.setGraphicsEffect(opacity_effect)
        
        opacity_anim = QPropertyAnimation(opacity_effect, b"opacity")
        opacity_anim.setDuration(300)  # 300毫秒
        opacity_anim.setStartValue(0.3)
        opacity_anim.setEndValue(1.0)
        opacity_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        # 启动动画
        opacity_anim.start()
        
        # 记录当前切换的选项卡
        title = self.main_tab.tabText(index)
        self.statusBar().showMessage(f"已切换到{title}视图", 2000)
    
    def _animate_refresh(self):
        """刷新数据时的动画效果"""
        # 创建一个并行动画组
        anim_group = QParallelAnimationGroup()
        
        # 对底部状态条中的所有图标进行动画
        for widget in self.findChildren(QLabel):
            if not widget.pixmap() or widget.pixmap().isNull():
                continue
                
            # 创建抖动动画
            pos_anim = QPropertyAnimation(widget, b"pos")
            pos_anim.setDuration(300)
            
            # 获取当前位置
            start_pos = widget.pos()
            
            # 设置抖动位置
            pos_anim.setStartValue(start_pos)
            pos_anim.setKeyValueAt(0.25, QPoint(start_pos.x() - 2, start_pos.y()))
            pos_anim.setKeyValueAt(0.5, QPoint(start_pos.x() + 2, start_pos.y()))
            pos_anim.setKeyValueAt(0.75, QPoint(start_pos.x() - 1, start_pos.y()))
            pos_anim.setEndValue(start_pos)
            
            anim_group.addAnimation(pos_anim)
        
        # 启动动画组
        anim_group.start()
    
    def _refresh_data(self):
        """刷新数据"""
        self.statusBar().showMessage("正在刷新数据...")
        
        # 播放刷新动画
        self._animate_refresh()
        
        # 实际应用中，这里会调用控制器的刷新方法
        logging.info("刷新数据")
        
        # 刷新当前选项卡的数据
        current_index = self.main_tab.currentIndex()
        current_widget = self.main_tab.widget(current_index)
        
        if hasattr(current_widget, 'refresh_data') and callable(getattr(current_widget, 'refresh_data')):
            try:
                current_widget.refresh_data()
            except Exception as e:
                logging.error(f"刷新数据失败: {str(e)}")
        
        self.statusBar().showMessage("数据刷新完成", 3000)
    
    def initialize_with_data(self, data):
        """初始化界面数据"""
        # 各个视图的初始化
        logging.info("初始化界面数据")
        
        # 初始化市场视图
        if hasattr(self, 'market_view') and hasattr(self.market_view, 'initialize_with_data'):
            try:
                market_data = data.get('market_data', {})
                self.market_view.initialize_with_data(market_data)
                logging.info("市场视图数据初始化完成")
            except Exception as e:
                logging.error(f"市场视图数据初始化失败: {str(e)}")
        
        # 初始化交易视图
        if hasattr(self, 'trading_view') and hasattr(self.trading_view, 'initialize_with_data'):
            try:
                trading_data = data.get('trading_data', {})
                self.trading_view.initialize_with_data(trading_data)
                logging.info("交易视图数据初始化完成")
            except Exception as e:
                logging.error(f"交易视图数据初始化失败: {str(e)}")
        
        # 初始化量子网络视图
        if hasattr(self, 'quantum_view') and hasattr(self.quantum_view, 'initialize_with_data'):
            try:
                quantum_data = data.get('quantum_data', {})
                self.quantum_view.initialize_with_data(quantum_data)
                logging.info("量子网络视图数据初始化完成")
            except Exception as e:
                logging.error(f"量子网络视图数据初始化失败: {str(e)}")
        
        # 初始化投资组合视图
        if hasattr(self, 'portfolio_view') and hasattr(self.portfolio_view, 'initialize_with_data'):
            try:
                portfolio_data = data.get('portfolio_data', {})
                self.portfolio_view.initialize_with_data(portfolio_data)
                logging.info("投资组合视图数据初始化完成")
            except Exception as e:
                logging.error(f"投资组合视图数据初始化失败: {str(e)}")
    
    def update_loading_progress(self, progress, message):
        """更新加载进度"""
        self.statusBar().showMessage(f"加载中: {message} ({progress}%)")
        logging.info(f"加载进度: {progress}% - {message}") 