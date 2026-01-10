# -*- mode: python ; coding: utf-8 -*-
# DLER Tkinter - PyInstaller Spec File
# Optimized build with AVX2 native decoder

import os
import sys

# Get TKinterModernThemes paths - hardcoded for sylc2 environment
env_path = r'C:\Users\Symphoenix\anaconda3\envs\sylc2'
tkmt_themes = os.path.join(env_path, 'Lib', 'site-packages', 'TKinterModernThemes', 'themes')
tkmt_images = os.path.join(env_path, 'Lib', 'site-packages', 'TKinterModernThemes', 'images')

a = Analysis(
    ['tk_main.py'],
    pathex=['.', 'src'],
    binaries=[
        # AVX2-optimized native yEnc decoder (3 GB/s!)
        ('src/native/yenc_turbo.cp314-win_amd64.pyd', 'src/native'),
        # Bundled tools - no external dependencies needed!
        ('tools/7z.exe', 'tools'),
        ('tools/7z.dll', 'tools'),
        ('tools/par2j64.exe', 'tools'),
    ],
    datas=[
        ('logo.png', '.'),
        # TKinterModernThemes data files
        (tkmt_themes, 'TKinterModernThemes/themes'),
        (tkmt_images, 'TKinterModernThemes/images'),
    ],
    hiddenimports=[
        'TKinterModernThemes',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'src.gui.tkinter_app',
        'src.core',
        'src.core.turbo_engine_v2',
        'src.core.turbo_yenc',
        'src.core.fast_nntp',
        'src.core.post_processor',
        'src.utils.config',
        'yenc_turbo',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude large/problematic packages
        'numpy',          # Bug with Python 3.14 PyInstaller
        'numba',
        'cupy',
        'torch',
        'tensorflow',
        'scipy',
        'matplotlib',
        'pandas',
        'PySide6',
        'PyQt5',
        'PyQt6',
        'pyqtgraph',
        'IPython',
        'jupyter',
        'pytest',
        'opencv',
        'cv2',
    ],
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DLER',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.png'],
    version='version_info.py',  # Windows version info
)
