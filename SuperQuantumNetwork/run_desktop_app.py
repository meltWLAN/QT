#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 桌面客户端启动脚本
"""

import sys
import os
import logging
import traceback

# 确保脚本可以在任何目录下运行
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    from gui_app import main
    
    # 运行应用
    sys.exit(main())
    
except Exception as e:
    # 显示错误消息
    error_message = f"启动失败: {str(e)}\n{traceback.format_exc()}"
    
    # 记录到日志
    logging.error(error_message)
    
    # 尝试显示GUI错误消息
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "启动错误", error_message)
    except:
        # 如果无法显示GUI错误，打印到控制台
        print("启动错误:", file=sys.stderr)
        print(error_message, file=sys.stderr)
    
    sys.exit(1) 