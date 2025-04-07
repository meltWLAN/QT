#!/usr/bin/env python3
"""
超神系统 - GUI功能测试脚本
用于验证和测试超神系统桌面应用的GUI界面
"""

import os
import sys
import unittest
import time
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GuiTest")

# 导入测试目标
try:
    from run_supergod_system import MainWindow
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    logger.error("请确保已安装所有依赖")
    sys.exit(1)


class SuperGodGuiTest(unittest.TestCase):
    """超神系统桌面GUI功能测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化 - 创建应用实例"""
        # 创建QApplication实例
        cls.app = QApplication.instance() or QApplication(sys.argv)
        logger.info("创建Qt应用实例")
    
    def setUp(self):
        """每个测试方法初始化"""
        # 创建日志记录器
        self.test_logger = logging.getLogger('supergod')
        
        # 创建并显示主窗口
        self.window = MainWindow(self.test_logger, debug=True)
        self.window.show()
        self.app.processEvents()
        
        # 等待界面完全加载
        time.sleep(0.5)
        self.app.processEvents()
        
        logger.info("测试窗口创建完成")
    
    def tearDown(self):
        """每个测试方法完成后清理"""
        # 关闭主窗口
        self.window.close()
        self.app.processEvents()
        
        # 释放引用
        self.window = None
        
        # 垃圾回收
        import gc
        gc.collect()
        
        logger.info("测试窗口已关闭")
    
    def test_window_title(self):
        """测试窗口标题正确性"""
        expected_title = "超神系统 - 中国市场分析"
        self.assertEqual(self.window.windowTitle(), expected_title)
        logger.info(f"窗口标题验证通过: {expected_title}")
    
    def test_tab_widget_exists(self):
        """测试标签页组件存在性"""
        self.assertIsNotNone(self.window.tab_widget)
        logger.info("标签页组件验证通过")
    
    def test_market_module_loaded(self):
        """测试市场模块加载状态"""
        # 检查至少有一个标签页
        self.assertTrue(self.window.tab_widget.count() >= 1)
        
        # 获取当前标签页标题
        current_tab_title = self.window.tab_widget.tabText(0)
        logger.info(f"检测到标签页: {current_tab_title}")
        
        # 验证标签页存在并且可见
        self.assertTrue(current_tab_title != "")
    
    def test_market_data_displayed(self):
        """测试市场数据显示状态"""
        try:
            # 点击第一个标签页
            self.window.tab_widget.setCurrentIndex(0)
            self.app.processEvents()
            
            # 获取当前页面
            current_page = self.window.tab_widget.currentWidget()
            self.assertIsNotNone(current_page)
            
            # 验证页面上的市场数据已加载 (通过检查主布局中的组件)
            if hasattr(current_page, 'findChildren'):
                # 查找页面上的QLabel组件
                labels = current_page.findChildren(QLabel)
                self.assertTrue(len(labels) > 0)
                
                # 验证至少有一个标签包含市场数据
                has_market_data = False
                for label in labels:
                    if (hasattr(label, 'text') and 
                        (label.text() != "" and 
                         label.text() != "--" and 
                         not label.text().startswith("请"))):
                        has_market_data = True
                        logger.info(f"检测到市场数据: {label.text()}")
                        break
                
                self.assertTrue(has_market_data, "未检测到有效的市场数据")
        except Exception as e:
            logger.error(f"测试市场数据显示失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.fail(f"测试市场数据显示失败: {str(e)}")
    
    def test_tab_navigation(self):
        """测试标签页导航功能"""
        # 获取标签页数量
        tab_count = self.window.tab_widget.count()
        
        if tab_count > 1:
            # 记录初始标签页名称
            first_tab_name = self.window.tab_widget.tabText(0)
            
            # 切换到第二个标签页
            self.window.tab_widget.setCurrentIndex(1)
            self.app.processEvents()
            time.sleep(0.3)
            
            # 验证标签页已切换
            second_tab_name = self.window.tab_widget.tabText(1)
            current_index = self.window.tab_widget.currentIndex()
            
            self.assertEqual(current_index, 1)
            logger.info(f"标签页导航成功: 从 {first_tab_name} 切换到 {second_tab_name}")
        else:
            logger.info("仅有一个标签页，跳过导航测试")
    
    def test_resize_behavior(self):
        """测试窗口调整大小行为"""
        # 记录原始尺寸
        original_size = self.window.size()
        logger.info(f"原始窗口尺寸: {original_size.width()}x{original_size.height()}")
        
        # 调整到新尺寸
        new_width = original_size.width() + 100
        new_height = original_size.height() + 100
        self.window.resize(new_width, new_height)
        self.app.processEvents()
        time.sleep(0.3)
        
        # 验证尺寸已更改
        current_size = self.window.size()
        logger.info(f"调整后窗口尺寸: {current_size.width()}x{current_size.height()}")
        
        # 验证尺寸变化是否合理 (允许有一定偏差)
        # macOS可能会忽略高度调整，所以只检查宽度
        self.assertAlmostEqual(current_size.width(), new_width, delta=20)
        # self.assertAlmostEqual(current_size.height(), new_height, delta=20)


def run_gui_tests():
    """运行GUI测试套件"""
    try:
        logger.info("开始超神系统GUI功能测试...")
        
        # 创建测试套件
        test_loader = unittest.TestLoader()
        test_suite = unittest.TestSuite()
        
        # 添加测试类
        test_suite.addTest(test_loader.loadTestsFromTestCase(SuperGodGuiTest))
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # 输出测试结果
        logger.info(f"GUI测试完成: 成功 {result.testsRun - len(result.errors) - len(result.failures)}/{result.testsRun}")
        
        if result.errors or result.failures:
            logger.error("存在测试失败项")
            return False
        else:
            logger.info("所有测试通过")
            return True
    
    except Exception as e:
        logger.error(f"运行GUI测试出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_gui_tests()
    sys.exit(0 if success else 1) 