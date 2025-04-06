#!/usr/bin/env python3
"""
创建超神量子共生网络交易系统的桌面启动器
"""

import os
import sys
import platform
import subprocess

def create_windows_shortcut():
    """在Windows上创建桌面快捷方式"""
    import winreg
    import win32com.client
    
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    script_path = os.path.abspath("install_and_run.py")
    shortcut_path = os.path.join(desktop, "超神量子交易系统.lnk")
    
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.TargetPath = sys.executable
    shortcut.Arguments = f'"{script_path}"'
    shortcut.WorkingDirectory = os.path.dirname(script_path)
    shortcut.IconLocation = sys.executable
    shortcut.Description = "超神量子共生网络交易系统"
    shortcut.save()
    
    print(f"桌面快捷方式已创建: {shortcut_path}")

def create_linux_desktop_entry():
    """在Linux上创建桌面启动器"""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    script_path = os.path.abspath("install_and_run.py")
    
    desktop_entry = f"""[Desktop Entry]
Type=Application
Name=超神量子交易系统
Comment=超神量子共生网络交易系统
Exec={sys.executable} "{script_path}"
Terminal=false
Categories=Finance;
"""
    
    desktop_file_path = os.path.join(desktop, "quantum-trading.desktop")
    
    try:
        with open(desktop_file_path, "w") as f:
            f.write(desktop_entry)
        
        os.chmod(desktop_file_path, 0o755)
        print(f"桌面启动器已创建: {desktop_file_path}")
    except Exception as e:
        print(f"创建桌面启动器失败: {str(e)}")

def create_macos_application():
    """在macOS上创建应用程序包"""
    script_path = os.path.abspath("install_and_run.py")
    home = os.path.expanduser("~")
    app_path = os.path.join(home, "Applications", "超神量子交易系统.app")
    
    app_structure = {
        "Contents": {
            "Info.plist": f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>run.sh</string>
    <key>CFBundleIconFile</key>
    <string>applet</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>超神量子交易系统</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>""",
            "MacOS": {
                "run.sh": f"""#!/bin/bash
cd "{os.path.dirname(script_path)}"
"{sys.executable}" "{script_path}"
"""
            },
            "Resources": {
                "applet": ""  # 图标文件，这里留空
            }
        }
    }
    
    # 创建应用程序目录结构
    if os.path.exists(app_path):
        print(f"删除现有应用程序: {app_path}")
        import shutil
        shutil.rmtree(app_path)
    
    # 创建应用目录
    os.makedirs(app_path, exist_ok=True)
    
    # 创建内容
    for path, content in _walk_dict(app_structure, app_path):
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        
        if content:
            with open(path, "w") as f:
                f.write(content)
            if path.endswith("run.sh"):
                os.chmod(path, 0o755)
    
    print(f"macOS应用程序已创建: {app_path}")

def _walk_dict(d, base_path):
    """遍历字典结构，生成文件路径和内容"""
    for k, v in d.items():
        path = os.path.join(base_path, k)
        if isinstance(v, dict):
            if not os.path.exists(path):
                os.makedirs(path)
            yield from _walk_dict(v, path)
        else:
            yield path, v

def main():
    """主函数"""
    system = platform.system()
    
    print(f"正在为{system}系统创建桌面启动器...")
    
    try:
        if system == "Windows":
            try:
                create_windows_shortcut()
            except ImportError:
                print("错误: 需要安装pywin32模块")
                print("请运行: pip install pywin32")
        elif system == "Linux":
            create_linux_desktop_entry()
        elif system == "Darwin":  # macOS
            create_macos_application()
        else:
            print(f"不支持的操作系统: {system}")
            sys.exit(1)
        
        print("桌面启动器创建成功！")
    except Exception as e:
        print(f"创建桌面启动器时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 