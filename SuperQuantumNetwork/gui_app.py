#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 桌面客户端
高级交易系统的现代化桌面界面
"""

import sys
import os
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplashScreen, QMessageBox
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QFont
import qdarkstyle
from qt_material import apply_stylesheet

# 导入自定义模块
from gui.views.main_window import SuperTradingMainWindow
from gui.controllers.data_controller import DataController
from gui.controllers.trading_controller import TradingController

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_trading_gui.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SuperTradingGUI")


class DataLoadingThread(QThread):
    """数据加载线程"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, data_controller, trading_controller):
        super().__init__()
        self.data_controller = data_controller
        self.trading_controller = trading_controller
        
    def run(self):
        try:
            # 初始化数据控制器
            self.progress_signal.emit(10, "初始化数据源...")
            self.data_controller.initialize()
            
            # 加载初始数据
            self.progress_signal.emit(30, "加载市场数据...")
            data = self.data_controller.load_initial_data()
            
            # 初始化交易控制器
            self.progress_signal.emit(50, "初始化交易系统...")
            self.trading_controller.initialize_mock_positions()
            self.trading_controller.initialize_mock_orders()
            
            # 获取账户和订单数据
            self.progress_signal.emit(70, "获取账户数据...")
            data["positions"] = self.trading_controller.get_position_list()
            data["orders"] = self.trading_controller.get_order_list()
            
            # 初始化量子网络
            self.progress_signal.emit(90, "初始化量子共生网络...")
            
            # 发送完成信号
            self.progress_signal.emit(100, "加载完成!")
            self.finished_signal.emit(data)
            
        except Exception as e:
            error_message = f"数据加载失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)
            self.error_signal.emit(error_message)


def load_stylesheet(theme="dark_teal"):
    """加载应用样式表"""
    try:
        return apply_stylesheet(QApplication.instance(), theme=theme)
    except:
        return qdarkstyle.load_stylesheet_pyqt5()


def main():
    """应用主函数"""
    # 创建应用实例
    app = QApplication(sys.argv)
    app.setApplicationName("超神量子共生网络交易系统")
    app.setOrganizationName("QuantumSymbioticTeam")
    
    # 设置应用图标和字体
    app.setWindowIcon(QIcon("gui/resources/icon.png"))
    # 使用系统默认字体
    # app.setFont(QFont("Microsoft YaHei UI", 9))
    
    # 加载样式表
    load_stylesheet("dark_cyan")  # 尝试不同的主题
    
    # 显示启动画面
    splash_pix = QPixmap("gui/resources/splash.png")
    if not splash_pix.isNull():
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.show()
        app.processEvents()
    else:
        splash = None
    
    # 创建控制器
    data_controller = DataController()
    trading_controller = TradingController()
    
    # 创建主窗口
    main_window = SuperTradingMainWindow(data_controller, trading_controller)
    
    # 创建数据加载线程，并保存引用以防止过早销毁
    loading_thread = DataLoadingThread(data_controller, trading_controller)
    # 保存线程引用到主窗口
    main_window.loading_thread = loading_thread
    
    # 延迟显示主窗口
    def show_main_window():
        if splash:
            splash.finish(main_window)
        main_window.showMaximized()
        
        # 连接信号
        loading_thread.progress_signal.connect(main_window.update_loading_progress)
        loading_thread.finished_signal.connect(main_window.initialize_with_data)
        loading_thread.error_signal.connect(lambda msg: QMessageBox.critical(main_window, "错误", msg))
        # 启动线程
        loading_thread.start()
    
    QTimer.singleShot(1000, show_main_window)
    
    # 执行应用
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main()) 