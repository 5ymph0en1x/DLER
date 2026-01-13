#!/usr/bin/env python
"""Build DLER Ultimate (GPU/CUDA) version."""
import os
import subprocess
import sys

# Set environment variable for ultimate build
os.environ['DLER_BUILD'] = 'ultimate'

# Run PyInstaller
result = subprocess.run(
    [sys.executable, '-m', 'PyInstaller', 'dler_tk.spec', '--noconfirm'],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

sys.exit(result.returncode)
