#!/usr/bin/env python3
"""
DLER TKinter GUI - Entry Point
==============================

Marantz-style minimalist interface using TKinterModernThemes.
"""

import sys
import os
from pathlib import Path

# Determine if we're running as a PyInstaller bundle
IS_FROZEN = getattr(sys, 'frozen', False)

if IS_FROZEN:
    BUNDLE_DIR = Path(sys._MEIPASS)
    if str(BUNDLE_DIR) not in sys.path:
        sys.path.insert(0, str(BUNDLE_DIR))
else:
    SCRIPT_DIR = Path(__file__).parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


def setup_logging() -> None:
    """Configure logging for the application."""
    import logging

    log_dir = Path.home() / ".dler" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        file_handler = logging.FileHandler(
            log_dir / "dler_tk.log",
            encoding='utf-8',
            mode='a'
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler]
        )
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")


def main() -> int:
    """Main entry point for TKinter GUI."""
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)
    logger.info("DLER TKinter starting...")

    try:
        # Ensure config directories exist
        from src.utils.config import load_config, ensure_directories
        config = load_config()
        ensure_directories(config)

        logger.info("Configuration loaded, launching TKinter GUI...")

        # Import and run TKinter app
        from src.gui.tkinter_app import DLERApp
        app = DLERApp()
        app.run()

        return 0

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

        # Show error dialog
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "DLER Error",
                f"Fatal error starting DLER:\n\n{e}\n\nCheck logs at:\n{Path.home() / '.dler' / 'logs'}"
            )
        except:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
