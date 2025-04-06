#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 一键安装与启动脚本
"""

import os
import sys
import subprocess
import platform
import time

WELCOME_TEXT = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      超 神 量 子 共 生 网 络 交 易 系 统                      ║
║                                                              ║
║      Super God-Level Quantum Symbiotic Trading System        ║
║                                                              ║
║      版本: 0.2.0                                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

def print_step(step, message):
    """打印步骤信息"""
    print(f"\n[{step}] {message}")
    time.sleep(0.5)

def check_python_version():
    """检查Python版本"""
    print_step(1, "检查Python版本...")
    ver = sys.version_info
    if ver.major < 3 or (ver.major == 3 and ver.minor < 7):
        print("错误: 需要Python 3.7或更高版本")
        sys.exit(1)
    print(f"√ Python版本兼容: {platform.python_version()}")

def install_dependencies():
    """安装依赖"""
    print_step(2, "检查并安装依赖包...")
    
    # 确定requirements文件路径
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    if not os.path.exists(req_path):
        print("错误: 找不到requirements.txt文件")
        sys.exit(1)
    
    # 安装所有依赖
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_path], check=True)
        print("√ 所有依赖安装完成")
    except subprocess.CalledProcessError:
        print("警告: 部分依赖可能安装失败，尝试继续...")

def check_system():
    """检查系统环境"""
    print_step(3, "检查系统环境...")
    
    system = platform.system()
    print(f"操作系统: {system} {platform.version()}")
    
    if system == "Windows":
        print("√ Windows系统兼容")
    elif system == "Darwin":
        print("√ macOS系统兼容")
    elif system == "Linux":
        print("√ Linux系统兼容")
    else:
        print(f"警告: 未测试的操作系统 {system}，可能存在兼容性问题")

def run_app():
    """运行应用程序"""
    print_step(4, "启动交易系统...")
    
    try:
        # 先尝试运行完整版
        try:
            import PyQt5
            import pyqtgraph
            import qdarkstyle
            import qt_material
            import qtawesome
            
            print("√ 所有UI依赖已满足，启动完整版...")
            subprocess.run([sys.executable, "run_desktop_app.py"])
            return True
        except ImportError as e:
            print(f"注意: 部分UI依赖缺失 ({str(e)})，尝试启动简化版...")
            
        # 运行简化版
        subprocess.run([sys.executable, "run_simple_app.py"])
        return True
    except Exception as e:
        print(f"错误: 启动失败 - {str(e)}")
        return False

def main():
    """主函数"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(WELCOME_TEXT)
    time.sleep(1)
    
    check_python_version()
    check_system()
    
    # 询问是否安装依赖
    install_deps = input("\n是否安装/更新依赖? (y/n, 默认y): ").strip().lower()
    if install_deps != "n":
        install_dependencies()
    
    # 运行应用
    print("\n准备启动超神量子共生网络交易系统...")
    time.sleep(1)
    
    if run_app():
        print("\n应用程序已关闭，感谢使用！")
    else:
        print("\n应用程序启动失败，请检查日志以获取更多信息。")

if __name__ == "__main__":
    main() 