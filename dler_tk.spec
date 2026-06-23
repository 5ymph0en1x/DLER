# -*- mode: python ; coding: utf-8 -*-
# DLER Tkinter - PyInstaller Spec File
# Single CPU/MP build (onefile, no console). RAM-processing/GPU support removed.
#
# Usage:  uv run --with pyinstaller python build_basic.py
#   -> dist/DLER.exe

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

EXE_NAME = 'DLER'

# =============================================================================
# DATA / BINARY COLLECTION (resolve from the ACTIVE build env)
# =============================================================================
# TKinterModernThemes ships its theme/image data as package data.
tkmt_datas = collect_data_files('TKinterModernThemes')
# tkinterdnd2 needs its native tkdnd library bundled for drag-and-drop.
dnd_datas, dnd_binaries, dnd_hiddenimports = collect_all('tkinterdnd2')
# All of the app's own modules — some are imported dynamically (e.g. mp_engine).
src_hiddenimports = collect_submodules('src')

# =============================================================================
# HIDDEN IMPORTS
# =============================================================================
hiddenimports = [
    'TKinterModernThemes',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'yenc_turbo',
    'numpy',
    'numpy._core',
    'numpy._core._methods',
    'numpy._core._multiarray_umath',
] + src_hiddenimports + dnd_hiddenimports

# =============================================================================
# EXCLUDES (GPU/RAM-processing libs are gone — exclude them outright)
# =============================================================================
excludes = [
    'numba', 'torch', 'tensorflow', 'scipy', 'matplotlib', 'pandas',
    'PySide6', 'PyQt5', 'PyQt6', 'pyqtgraph',
    'IPython', 'jupyter', 'pytest', 'opencv', 'cv2',
    'cupy', 'cupy_backends', 'cupyx', 'fastrlock',
]

# =============================================================================
# ANALYSIS
# =============================================================================
a = Analysis(
    ['tk_main.py'],
    pathex=['.', 'src'],
    binaries=[
        # AVX2-optimized native yEnc decoder (3 GB/s!)
        ('src/native/yenc_turbo.cp314-win_amd64.pyd', 'src/native'),
        # Native RAR extractor (disk extraction backend)
        ('src/native/ram_rar.cp314-win_amd64.pyd', 'src/native'),
        # Bundled tools - no external dependencies needed!
        ('tools/7z.exe', 'tools'),
        ('tools/7z.dll', 'tools'),
        ('tools/par2j64.exe', 'tools'),
        ('tools/UnRAR64.dll', 'tools'),
    ] + dnd_binaries,
    datas=[
        ('logo.png', '.'),
        ('logo.ico', '.'),
    ] + tkmt_datas + dnd_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook_basic.py'],
    excludes=excludes,
    noarchive=False,
    optimize=0,  # Must be 0 for NumPy 2.x
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=EXE_NAME,
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
    icon=['logo.ico'],
    version='version_info.py',
)
