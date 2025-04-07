#!/usr/bin/env python3
"""
超神系统 - 豪华版桌面应用启动脚本
性能优化版本
"""

import sys
import os
import logging
import traceback
import time
import importlib
import threading

# 添加性能计时
START_TIME = time.time()

def log_time(message):
    """记录运行时间"""
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:.3f}s] {message}")

log_time("启动脚本初始化")

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

# 预加载常用模块
def preload_modules():
    """预加载常用模块，使用线程避免阻塞主程序"""
    modules_to_preload = [
        'numpy', 
        'pandas', 
        'pyqtgraph', 
        'quantum_symbiotic_network.core.quantum_entanglement_engine',
        'market_controllers',
        'dashboard_module',
        'quantum_view'
    ]
    
    for module_name in modules_to_preload:
        try:
            importlib.import_module(module_name)
        except ImportError:
            logger.warning(f"预加载模块 {module_name} 失败")

# 在后台线程中预加载模块
preload_thread = threading.Thread(target=preload_modules)
preload_thread.daemon = True
preload_thread.start()

def check_dependencies():
    """检查必要的依赖库"""
    log_time("检查依赖")
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
        print(" 超神量子共生网络交易系统 - 豪华版桌面应用 v2.0.1")
        print(" 高性能优化版")
        print("=" * 60)
        
        # 快速检查核心依赖
        log_time("快速检查核心依赖")
        try:
            import PyQt5
            import numpy
        except ImportError as e:
            print(f"缺少核心依赖: {str(e)}")
            missing = check_dependencies()
            if missing:
                print(f"需要安装以下依赖: {', '.join(missing)}")
                if not install_dependencies(missing):
                    print("依赖安装失败，请手动安装后重试")
                    return 1
        
        print("正在启动超神系统豪华版桌面应用...")
        log_time("准备加载主程序")
        
        # 导入桌面应用
        from super_god_desktop_app import main as run_app
        
        log_time("开始运行主程序")
        
        # 设置更详细的异常处理
        try:
            # 运行应用
            return run_app()
        except Exception as e:
            logger.error(f"运行应用失败: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"应用运行时发生错误:\n{str(e)}\n{traceback.format_exc()}")
            return 1
    
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"启动失败: {str(e)}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())