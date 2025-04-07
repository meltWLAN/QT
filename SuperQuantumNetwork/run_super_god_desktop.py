#!/usr/bin/env python3
"""
超神系统 - 豪华版桌面应用启动脚本
"""

import sys
import os
import logging
import traceback

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='super_god_desktop.log'
)
logger = logging.getLogger("SuperGodDesktop")


def check_dependencies():
    """检查必要的依赖库"""
    required_packages = [
        'PyQt5',
        'numpy',
        'pandas',
        'pyqtgraph',
        'qtawesome',
        'qdarkstyle',
        'qt-material'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages


def install_dependencies(packages):
    """安装缺失的依赖库"""
    try:
        import pip
        for package in packages:
            print(f"正在安装 {package}...")
            pip.main(['install', package])
        
        return True
    except Exception as e:
        print(f"安装依赖失败: {str(e)}")
        return False


def main():
    """主函数"""
    try:
        # 打印欢迎信息
        print("=" * 60)
        print(" 超神量子共生网络交易系统 - 豪华版桌面应用 v2.0.0")
        print("=" * 60)
        
        # 不再检查依赖，直接启动应用
        print("正在启动超神系统豪华版桌面应用...")
        
        # 导入桌面应用
        from super_god_desktop_app import main as run_app
        
        # 运行应用
        return run_app()
    
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"启动失败: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 