#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 简化版启动脚本
"""

import sys
import os
import logging
import traceback

# 确保当前工作目录设置在脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    # 导入主函数并运行
    from simple_gui_app import main
    sys.exit(main())
except Exception as e:
    # 记录错误
    logging.error(f"启动失败: {str(e)}")
    logging.error(traceback.format_exc())
    
    # 尝试显示GUI错误消息
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "启动失败", f"错误: {str(e)}\n\n{traceback.format_exc()}")
    except:
        # 如果GUI显示也失败，则打印到控制台
        print(f"严重错误: {str(e)}")
        print(traceback.format_exc())
    
    sys.exit(1) 