#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神系统 - 中国市场分析桌面版启动入口
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import traceback
import tempfile

# 确保能够导入SuperQuantumNetwork包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QMessageBox, QSplashScreen
    from PyQt5.QtGui import QPixmap, QFont
    from PyQt5.QtCore import Qt, QTimer
    
    # 导入超神系统模块
    from SuperQuantumNetwork import initialize_market_module
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所需的依赖项: PyQt5, numpy, pandas")
    sys.exit(1)

# 检查是否已经有实例在运行
def check_running_instance():
    """检查是否已有超神系统实例在运行
    
    Returns:
        bool: 如果已有实例在运行则返回True，否则返回False
    """
    # 检查豪华版锁文件
    lock_file_path = os.path.join(tempfile.gettempdir(), "super_god_desktop.lock")
    
    if os.path.exists(lock_file_path):
        try:
            # 读取锁文件中的进程ID
            with open(lock_file_path, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                # 进程不存在，可以忽略
                pass
        except (ValueError, IOError):
            # 锁文件损坏，忽略
            pass
    
    # 检查基础版锁文件
    lock_file_path = os.path.join(tempfile.gettempdir(), "supergod_basic.lock")
    
    if os.path.exists(lock_file_path):
        try:
            # 读取锁文件中的进程ID
            with open(lock_file_path, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                # 进程不存在，可以删除旧的锁文件
                os.remove(lock_file_path)
        except (ValueError, IOError):
            # 锁文件损坏，删除它
            os.remove(lock_file_path)
    
    # 创建锁文件
    try:
        with open(lock_file_path, 'w') as f:
            f.write(str(os.getpid()))
    except IOError:
        print("警告: 无法创建锁文件")
    
    return False

def cleanup_lock_file():
    """退出时清理锁文件"""
    lock_file_path = os.path.join(tempfile.gettempdir(), "supergod_basic.lock")
    if os.path.exists(lock_file_path):
        try:
            os.remove(lock_file_path)
        except OSError:
            pass

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='超神系统 - 中国市场分析桌面版')
    
    parser.add_argument('--config', '-c', type=str, 
                      help='配置文件路径')
    parser.add_argument('--debug', '-d', action='store_true',
                      help='启用调试模式')
    
    return parser.parse_args()

def print_banner():
    """打印超神系统横幅"""
    banner = """
    ███████╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗  ██████╗ ██████╗ 
    ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔════╝ ██╔═══██╗██╔══██╗
    ███████╗██║   ██║██████╔╝█████╗  ██████╔╝██║  ███╗██║   ██║██║  ██║
    ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██║   ██║██║   ██║██║  ██║
    ███████║╚██████╔╝██║     ███████╗██║  ██║╚██████╔╝╚██████╔╝██████╔╝
    ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═════╝ 
                                                                       
              中国市场分析系统 v1.0 - 桌面超神版
    ==================================================================
    """
    print(banner)

def initialize_logging(debug=False):
    """初始化日志配置"""
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'supergod_{timestamp}.log')
    
    # 设置日志级别
    log_level = logging.DEBUG if debug else logging.INFO
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('supergod')


class MainWindow(QMainWindow):
    """超神系统主窗口"""
    
    def __init__(self, logger, debug=False):
        super().__init__()
        self.logger = logger
        self.debug = debug
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口属性
        self.setWindowTitle("超神系统 - 中国市场分析")
        self.resize(1280, 800)
        
        # 创建中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页组件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 初始化市场分析模块
        self.init_market_module()
    
    def init_market_module(self):
        """初始化市场分析模块"""
        try:
            success = initialize_market_module(self, self.tab_widget)
            
            if not success:
                self.logger.error("市场分析模块初始化失败")
                QMessageBox.warning(self, "初始化失败", "市场分析模块初始化失败，请查看日志了解详情。")
        
        except Exception as e:
            self.logger.error(f"初始化市场分析模块时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"初始化市场分析模块时出错: {str(e)}")


def show_splash_screen():
    """显示启动画面"""
    # 创建启动画面
    splash_pixmap = QPixmap(500, 300)
    splash_pixmap.fill(Qt.black)
    
    splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
    
    # 设置启动画面文本
    splash.setFont(QFont('Arial', 14))
    splash.showMessage("超神系统启动中...", Qt.AlignCenter | Qt.AlignBottom, Qt.white)
    
    return splash


def main():
    """主函数"""
    # 检查是否已有实例在运行
    if check_running_instance():
        print("\n超神系统已经在运行中，请勿重复启动！\n")
        QMessageBox.warning(None, "重复启动", "超神系统已经在运行中，请勿重复启动！")
        return 1
    
    # 注册退出时清理锁文件
    import atexit
    atexit.register(cleanup_lock_file)
    
    # 打印横幅
    print_banner()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 初始化日志
    logger = initialize_logging(debug=args.debug)
    logger.info("超神系统启动中...")
    
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    # 显示启动画面
    splash = show_splash_screen()
    splash.show()
    app.processEvents()
    
    try:
        # 创建主窗口
        main_window = MainWindow(logger, debug=args.debug)
        
        # 设置一个定时器以在适当时间关闭启动画面
        def finish_splash():
            splash.finish(main_window)
            main_window.show()
        
        QTimer.singleShot(2000, finish_splash)
        
        # 运行应用
        return app.exec_()
    
    except Exception as e:
        logger.error(f"启动超神系统时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 关闭启动画面
        splash.close()
        
        # 显示错误消息
        QMessageBox.critical(None, "启动错误", f"启动超神系统时发生错误:\n{str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 