#!/usr/bin/env python
"""PyInstaller 构建脚本：打包 thinkWM 可执行文件

Windows 用户：
  # 在 Windows 命令提示符中运行：
  pip install pyinstaller
  python build_exe.py

Linux 用户：
  python build_exe.py

输出路径：dist/thinkWM(.exe)
"""
import PyInstaller.__main__
import os, sys, platform

name = "thinkWM"
script = os.path.join(os.path.dirname(__file__), "thinkWM.py")
is_win = platform.system() == "Windows"

args = [
    script,
    "--name", name,
    "--onefile",                    # 单文件
    "--clean",
    "--noconfirm",
    "--hidden-import", "psychopy",
    "--hidden-import", "numpy",
    "--collect-submodules", "psychopy",
]

if not is_win:
    # Windows 上 --windowed 会生成无控制台的 .exe
    # Linux/mac 上不加，避免缺 wx/gtk 报错
    pass

# Debug 模式（带控制台，可看到 print 输出）
if "debug" in sys.argv:
    if "--windowed" in args:
        args.remove("--windowed")
    print("[build] Debug build (console visible)")
else:
    # 正式版：隐藏控制台
    if is_win:
        args.append("--windowed")
    print("[build] Release build (no console)")

print(f"[build] Target: {platform.system()} {platform.machine()}")
PyInstaller.__main__.run(args)
