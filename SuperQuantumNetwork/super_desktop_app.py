#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 桌面应用入口
集成了高级启动画面、量子预测引擎和中国股市分析模块
"""

import sys
import os
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox, QAction, QMenu
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SuperQuantumSystem")


def import_or_install(package_name, import_name=None):
    """尝试导入模块，如果不存在则提示安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        return __import__(import_name)
    except ImportError:
        print(f"缺少{package_name}模块，请安装后重试")
        print(f"可使用命令: pip install {package_name}")
        return None


def load_controllers():
    """加载控制器"""
    try:
        # 导入控制器
        from simple_gui_app import DataController, TradingController
        
        # 创建控制器实例
        data_controller = DataController()
        trading_controller = TradingController()
        
        return data_controller, trading_controller
    except Exception as e:
        logger.error(f"加载控制器失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def try_load_advanced_splash():
    """尝试加载高级启动画面"""
    try:
        from advanced_splash import SuperGodSplashScreen
        return SuperGodSplashScreen
    except ImportError:
        # 如果高级启动画面不可用，使用简单版本
        from simple_gui_app import SimpleSplashScreen
        return SimpleSplashScreen


def try_load_advanced_window():
    """尝试加载高级主窗口"""
    try:
        # 尝试导入高级UI依赖
        pyqtgraph = import_or_install('pyqtgraph')
        qdarkstyle = import_or_install('qdarkstyle')
        qt_material = import_or_install('qt-material', 'qt_material')
        qtawesome = import_or_install('qtawesome')
        
        if all([pyqtgraph, qdarkstyle, qt_material, qtawesome]):
            # 所有依赖都已安装，使用完整版
            try:
                from gui.views.main_window import SuperTradingMainWindow
                logger.info("加载完整版主窗口成功")
                return SuperTradingMainWindow
            except Exception as e:
                logger.error(f"加载完整版主窗口失败: {str(e)}")
                logger.error(traceback.format_exc())
                # 回退到简单版本
    
    except Exception as e:
        logger.error(f"检查高级UI依赖时出错: {str(e)}")
    
    # 回退到简单版本
    try:
        from simple_gui_app import SimpleMainWindow
        logger.info("加载简化版主窗口")
        return SimpleMainWindow
    except Exception as e:
        logger.error(f"加载简化版主窗口失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def load_stylesheet():
    """加载应用样式表"""
    try:
        # 尝试使用qt-material样式
        try:
            import qt_material
            return None  # 返回None表示将使用apply_stylesheet而不是setStyleSheet
        except ImportError:
            pass
        
        # 尝试使用qdarkstyle
        try:
            import qdarkstyle
            return qdarkstyle.load_stylesheet_pyqt5()
        except ImportError:
            pass
        
        # 回退到简单样式表
        from simple_gui_app import load_stylesheet
        return load_stylesheet()
    
    except Exception as e:
        logger.error(f"加载样式表失败: {str(e)}")
        # 提供超神系统风格的高级样式表
        return """
        /* 超神量子共生网络交易系统 - 高级暗黑主题 */
        QMainWindow, QWidget {
            background-color: #121218;
            color: #E1E1E1;
        }
        
        /* 状态栏样式 - 量子能量效果 */
        QStatusBar {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                      stop:0 #050517, stop:0.5 #0a2050, stop:1 #050517);
            color: #00DDFF;
            font-weight: bold;
            border-top: 1px solid #00AAFF;
        }
        
        /* 选项卡样式 - 超神风格 */
        QTabWidget::pane {
            border: 1px solid #202030;
            background-color: #121220;
        }
        
        QTabBar::tab {
            background-color: #101018;
            color: #AAAAFF;
            border: 1px solid #202030;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 8px 12px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #202038, stop:1 #101028);
            color: #00DDFF;
            border-bottom: 2px solid #00AAFF;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #1A1A28;
            color: #CCCCFF;
        }
        
        /* 按钮样式 - 量子能量效果 */
        QPushButton {
            background-color: #0A0A28;
            color: #00DDFF;
            border: 1px solid #0055AA;
            border-radius: 4px;
            padding: 5px 15px;
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
        
        /* 进度条样式 - 量子能量流动效果 */
        QProgressBar {
            border: 1px solid #00AAFF;
            border-radius: 2px;
            background-color: #0A0A18;
            color: #FFFFFF;
            height: 6px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                      stop:0 #0055AA, stop:0.5 #00DDFF, stop:1 #0055AA);
            width: 5px;
        }
        
        /* 分组框样式 - 先进科技感 */
        QGroupBox {
            border: 1px solid #303050;
            border-radius: 5px;
            margin-top: 1.5ex;
            padding-top: 15px;
            padding-bottom: 8px;
            background-color: rgba(16, 16, 32, 150);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
            color: #00DDFF;
            font-weight: bold;
        }
        
        /* 表格样式 - 数据可视化增强 */
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
            padding: 6px;
            border: 1px solid #303050;
            font-weight: bold;
        }
        
        /* 标签样式 - 核心指标突出显示 */
        QLabel {
            color: #CCCCFF;
        }
        
        QLabel[important="true"] {
            color: #00DDFF;
            font-weight: bold;
        }
        
        /* 滚动条样式 - 简约现代 */
        QScrollBar:vertical {
            border: none;
            background: #0A0A18;
            width: 10px;
            margin: 10px 0 10px 0;
        }
        
        QScrollBar::handle:vertical {
            background: #303050;
            border-radius: 5px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #4040A0;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        /* 水平滚动条 */
        QScrollBar:horizontal {
            border: none;
            background: #0A0A18;
            height: 10px;
            margin: 0 10px 0 10px;
        }
        
        QScrollBar::handle:horizontal {
            background: #303050;
            border-radius: 5px;
            min-width: 20px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: #4040A0;
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* 下拉框样式 */
        QComboBox {
            background-color: #101028;
            color: #CCCCFF;
            border: 1px solid #303050;
            border-radius: 4px;
            padding: 4px 10px;
        }
        
        QComboBox:hover {
            border: 1px solid #00AAFF;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #303050;
        }
        
        QComboBox QAbstractItemView {
            background-color: #101028;
            border: 1px solid #303050;
            selection-background-color: #003366;
            selection-color: #FFFFFF;
        }
        
        /* 中国市场分析模块专用样式 */
        #chinaMarketWidget {
            background-color: #0A1028;
            border: 1px solid #304060;
            border-radius: 6px;
        }
        
        #marketIndexLabel {
            color: #00FFAA;
            font-weight: bold;
            font-size: 14px;
        }
        
        .StockUpLabel {
            color: #FF4444;
        }
        
        .StockDownLabel {
            color: #44FF44;
        }
        
        .HotSectorLabel {
            color: #FFAA00;
            font-weight: bold;
        }
        
        /* 量子网络可视化面板样式 */
        #quantumVisualizerPanel {
            background-color: #050510;
            border: 1px solid #101040;
            border-radius: 8px;
        }
        """


def main():
    """主函数"""
    # 记录启动信息
    logger.info("超神量子共生网络交易系统 v1.0.0 启动中")
    
    # 确保当前目录是脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # 创建应用
        app = QApplication(sys.argv)
        app.setApplicationName("超神量子共生网络交易系统")
        app.setApplicationVersion("1.0.0")
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)
        
        # 使用Fusion风格，与暗色主题更匹配
        app.setStyle('Fusion')  
        
        # 加载启动画面类
        SplashScreen = try_load_advanced_splash()
        splash = SplashScreen()
        splash.show()
        app.processEvents()
        
        # 加载控制器
        data_controller, trading_controller = load_controllers()
        
        # 尝试加载中国市场控制器
        try:
            from china_market.controllers.market_controller import ChinaMarketController
            china_controller = ChinaMarketController()
            logger.info("中国市场控制器加载成功")
            has_china_market = True
        except Exception as e:
            logger.warning(f"中国市场控制器加载失败: {str(e)}")
            china_controller = None
            has_china_market = False
        
        # 加载主窗口类
        MainWindow = try_load_advanced_window()
        
        # 创建主窗口
        main_window = MainWindow(data_controller, trading_controller)
        
        # 应用样式表
        stylesheet = load_stylesheet()
        if stylesheet:
            app.setStyleSheet(stylesheet)
        else:
            # 使用qt-material样式
            try:
                from qt_material import apply_stylesheet
                apply_stylesheet(app, theme='dark_teal')
            except Exception as e:
                logger.warning(f"应用qt-material样式失败: {str(e)}")
                
                # 尝试使用qdarkstyle
                try:
                    import qdarkstyle
                    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
                except:
                    logger.warning("后备样式加载失败")
        
        # 添加中国市场分析标签页
        if has_china_market:
            try:
                from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QTableWidget, QTableWidgetItem
                
                # 创建中国市场标签页
                china_tab = QWidget()
                main_window.tab_widget.addTab(china_tab, "中国市场")
                
                # 创建布局
                china_layout = QVBoxLayout(china_tab)
                
                # 市场指数面板
                market_index_label = QLabel("市场指数")
                market_index_label.setProperty("class", "title-label")
                china_layout.addWidget(market_index_label)
                
                # 指数信息
                index_layout = QHBoxLayout()
                sh_index = QLabel(f"上证指数: {china_controller.market_data.get('sh_index', {}).get('close', 0):.2f}")
                sz_index = QLabel(f"深证成指: {china_controller.market_data.get('sz_index', {}).get('close', 0):.2f}")
                cy_index = QLabel(f"创业板指: {china_controller.market_data.get('cyb_index', {}).get('close', 0):.2f}")
                
                index_layout.addWidget(sh_index)
                index_layout.addWidget(sz_index)
                index_layout.addWidget(cy_index)
                china_layout.addLayout(index_layout)
                
                # 板块热点
                hot_sector_label = QLabel("热点板块")
                hot_sector_label.setProperty("class", "title-label")
                china_layout.addWidget(hot_sector_label)
                
                # 刷新与预测按钮
                buttons_layout = QHBoxLayout()
                refresh_btn = QPushButton("刷新数据")
                predict_btn = QPushButton("超神预测")
                
                # 连接按钮信号
                refresh_btn.clicked.connect(lambda: china_controller.update_market_data())
                predict_btn.clicked.connect(lambda: china_controller.predict_market_trend())
                
                buttons_layout.addWidget(refresh_btn)
                buttons_layout.addWidget(predict_btn)
                china_layout.addLayout(buttons_layout)
                
                # 添加推荐表
                stock_table = QTableWidget(5, 4)
                stock_table.setHorizontalHeaderLabels(["代码", "名称", "操作", "当前价"])
                china_layout.addWidget(stock_table)
                
                # 记录成功
                logger.info("成功加载中国市场视图")
                
            except Exception as e:
                logger.error(f"加载中国市场标签页失败: {str(e)}")
        
        # 显示主窗口回调
        def on_splash_finished():
            # 显示主窗口
            main_window.show()
            # main_window.showMaximized()  # 全屏显示
        
        # 根据splash的类型连接信号或使用定时器
        if hasattr(splash, 'finished'):
            # 高级启动画面：使用信号
            splash.finished.connect(on_splash_finished)
        else:
            # 简单启动画面：使用定时器
            QTimer.singleShot(3000, lambda: [on_splash_finished(), splash.close()])
        
        # 模拟加载过程
        total_steps = 5
        for i in range(1, total_steps + 1):
            # 更新进度
            progress = i * 100 // total_steps
            if i == 1:
                message = "正在初始化量子网络..."
            elif i == 2:
                message = "加载交易引擎..."
            elif i == 3:
                message = "连接市场数据源..."
            elif i == 4:
                message = "正在加载中国股市分析模块..."
            else:
                message = "即将完成，请稍候..."
                
            # 更新启动画面
            if hasattr(splash, 'showMessage'):
                splash.showMessage(f"{message} {progress}%", Qt.AlignBottom | Qt.AlignHCenter)
            if hasattr(splash, 'setProgress'):
                splash.setProgress(progress)
                
            # 处理事件并等待
            app.processEvents()
            time.sleep(0.5)  # 模拟加载延迟
        
        # 运行应用
        return app.exec_()
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 尝试显示错误消息
        try:
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "超神系统启动失败",
                                f"错误: {str(e)}\n\n{traceback.format_exc()}")
        except:
            print(f"严重错误: {str(e)}")
            print(traceback.format_exc())
            
        return 1


if __name__ == "__main__":
    sys.exit(main())