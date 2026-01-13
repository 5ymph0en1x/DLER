#!/usr/bin/env python
"""Build DLER Basic (CPU-only) version."""
import os
import subprocess
import sys

# Set environment variable for basic build
os.environ['DLER_BUILD'] = 'basic'

# Run PyInstaller
result = subprocess.run(
    [sys.executable, '-m', 'PyInstaller', 'dler_tk.spec', '--noconfirm'],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

sys.exit(result.returncode)
