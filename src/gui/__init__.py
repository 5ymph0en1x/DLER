"""
DLER GUI Package
================

Supports both PySide6 (modern) and Tkinter (lightweight) interfaces.
"""

# Lazy imports - only load PySide6 modules when explicitly requested
# This allows tkinter_app to work without PySide6 installed

__all__ = ['run_app', 'MainWindow', 'DLERApp']


def __getattr__(name):
    """Lazy import PySide6 modules only when accessed."""
    if name == 'run_app':
        from .app import run_app
        return run_app
    elif name == 'MainWindow':
        from .main_window import MainWindow
        return MainWindow
    elif name == 'DLERApp':
        from .tkinter_app import DLERApp
        return DLERApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
