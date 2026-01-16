# -*- mode: python ; coding: utf-8 -*-
# DLER Tkinter - PyInstaller Spec File
# Dual build: Ultimate (CUDA) vs Basic (CPU-only)
#
# Usage:
#   set DLER_BUILD=ultimate && pyinstaller dler_tk.spec   (1 GB, GPU support)
#   set DLER_BUILD=basic && pyinstaller dler_tk.spec      (30 MB, CPU only)

import os
import sys

# =============================================================================
# BUILD CONFIGURATION
# =============================================================================
BUILD_MODE = os.environ.get('DLER_BUILD', 'ultimate').lower()
IS_ULTIMATE = BUILD_MODE == 'ultimate'

print(f"")
print(f"{'='*60}")
print(f"  DLER Build Mode: {'ULTIMATE (GPU/CUDA)' if IS_ULTIMATE else 'BASIC (CPU-only)'}")
print(f"{'='*60}")
print(f"")

# Output name based on build mode
EXE_NAME = 'DLER' if IS_ULTIMATE else 'DLER_Basic'

# =============================================================================
# PATHS
# =============================================================================
env_path = r'C:\Users\Symphoenix\anaconda3\envs\sylc2'
tkmt_themes = os.path.join(env_path, 'Lib', 'site-packages', 'TKinterModernThemes', 'themes')
tkmt_images = os.path.join(env_path, 'Lib', 'site-packages', 'TKinterModernThemes', 'images')

# =============================================================================
# HIDDEN IMPORTS
# =============================================================================
# Base imports (both versions)
base_hiddenimports = [
    'TKinterModernThemes',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'src.gui.tkinter_app',
    'src.gui.speed_graph',
    'src.gui.system_tray',
    'pystray',
    'pystray._win32',
    'src.core',
    'src.core.turbo_engine_v2',
    'src.core.turbo_yenc',
    'src.core.fast_nntp',
    'src.core.post_processor',
    'src.core.ram_processor',
    'src.core.par2_database',
    'src.core.file_verification',
    'src.utils.config',
    'yenc_turbo',
    # NumPy (required for both)
    'numpy',
    'numpy._core',
    'numpy._core._methods',
    'numpy._core._multiarray_umath',
]

# CuPy imports (Ultimate only) - Must be explicit due to dynamic imports
# See: https://github.com/cupy/cupy/issues/5090
cuda_hiddenimports = [
    # Core cupy
    'cupy',
    'cupy.cuda',
    'cupy.cuda.runtime',
    'cupy.cuda.memory',
    'cupy.cuda.device',
    'cupy.cuda.stream',
    # cupy._core - ALL submodules needed
    'cupy._core',
    'cupy._core._accelerator',
    'cupy._core._carray',
    'cupy._core._codeblock',
    'cupy._core._cub_reduction',
    'cupy._core._dtype',
    'cupy._core._fusion_interface',
    'cupy._core._fusion_kernel',
    'cupy._core._fusion_op',
    'cupy._core._fusion_optimization',
    'cupy._core._fusion_thread_local',
    'cupy._core._fusion_trace',
    'cupy._core._fusion_variable',
    'cupy._core._gufuncs',
    'cupy._core._kernel',
    'cupy._core._memory_range',
    'cupy._core._optimize_config',
    'cupy._core._reduction',
    'cupy._core._routines_binary',
    'cupy._core._routines_indexing',
    'cupy._core._routines_linalg',
    'cupy._core._routines_logic',
    'cupy._core._routines_manipulation',
    'cupy._core._routines_math',
    'cupy._core._routines_sorting',
    'cupy._core._routines_statistics',
    'cupy._core._scalar',
    'cupy._core._ufuncs',  # This one specifically causes issues!
    'cupy._core.core',
    'cupy._core.fusion',
    'cupy._core.internal',
    # cupy_backends
    'cupy_backends',
    'cupy_backends.cuda',
    'cupy_backends.cuda._softlink',
    'cupy_backends.cuda.api',
    'cupy_backends.cuda.api._runtime_enum',
    'cupy_backends.cuda.api._driver_enum',
    'cupy_backends.cuda.api.runtime',
    'cupy_backends.cuda.api.driver',
    'cupy_backends.cuda.stream',
    # fastrlock is required by cupy
    'fastrlock',
    'fastrlock.rlock',
]

# Combine based on build mode
hiddenimports = base_hiddenimports + (cuda_hiddenimports if IS_ULTIMATE else [])

# =============================================================================
# EXCLUDES
# =============================================================================
# Base excludes (both versions)
base_excludes = [
    'numba',
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
]

# Additional excludes for Basic version (no CUDA)
basic_excludes = [
    'cupy',
    'cupy_backends',
    'cupyx',
]

# Combine based on build mode
excludes = base_excludes + ([] if IS_ULTIMATE else basic_excludes)

# Runtime hook to set edition at startup
runtime_hook = ['runtime_hook_ultimate.py'] if IS_ULTIMATE else ['runtime_hook_basic.py']

# =============================================================================
# CUPY/CUDA COLLECTION (Ultimate only)
# =============================================================================
from PyInstaller.utils.hooks import collect_all, collect_submodules

cuda_binaries = []
cuda_datas = []
cuda_hiddenimports_extra = []

if IS_ULTIMATE:
    try:
        # Use PyInstaller's collect_all to properly collect cupy package
        cupy_datas, cupy_binaries, cupy_hiddenimports = collect_all('cupy')
        cuda_datas.extend(cupy_datas)
        cuda_binaries.extend(cupy_binaries)
        cuda_hiddenimports_extra.extend(cupy_hiddenimports)

        # Also collect cupy_backends
        try:
            backends_datas, backends_binaries, backends_hiddenimports = collect_all('cupy_backends')
            cuda_datas.extend(backends_datas)
            cuda_binaries.extend(backends_binaries)
            cuda_hiddenimports_extra.extend(backends_hiddenimports)
        except Exception:
            pass

        print(f"  CuPy: {len(cuda_binaries)} binaries, {len(cuda_datas)} data files, {len(cuda_hiddenimports_extra)} imports")
    except Exception as e:
        print(f"  WARNING: CuPy collection failed: {e}")

# =============================================================================
# ANALYSIS
# =============================================================================
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
    ] + cuda_binaries,
    datas=[
        ('logo.png', '.'),
        # TKinterModernThemes data files
        (tkmt_themes, 'TKinterModernThemes/themes'),
        (tkmt_images, 'TKinterModernThemes/images'),
    ] + cuda_datas,
    hiddenimports=hiddenimports + cuda_hiddenimports_extra,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hook,
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
    icon=['logo.png'],
    version='version_info.py',
)
