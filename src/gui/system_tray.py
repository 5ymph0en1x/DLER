"""
DLER System Tray Integration
=============================

Provides system tray functionality for DLER:
- Minimize to tray instead of closing
- Start minimized in system tray
- Auto-start with Windows
- Tray icon with context menu

Uses pystray for cross-platform tray support.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

# Windows registry for auto-start
if sys.platform == 'win32':
    import winreg

try:
    import pystray
    from PIL import Image
    HAS_PYSTRAY = True
except ImportError:
    HAS_PYSTRAY = False
    pystray = None

if TYPE_CHECKING:
    from pystray import Icon, MenuItem

logger = logging.getLogger(__name__)

# Windows registry key for auto-start
AUTOSTART_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_NAME = "DLER"


class SystemTrayManager:
    """
    Manages system tray icon and related functionality.

    Features:
    - System tray icon with context menu
    - Window show/hide control
    - Auto-start management (Windows registry)
    - Download status in tooltip

    Usage:
        tray = SystemTrayManager(
            icon_path="logo.png",
            on_show=lambda: window.deiconify(),
            on_quit=lambda: app.quit()
        )
        tray.start()
        # ... later ...
        tray.update_tooltip("Downloading: 150 MB/s")
        tray.stop()
    """

    def __init__(
        self,
        icon_path: Optional[Path] = None,
        on_show: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
        on_pause: Optional[Callable[[], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize system tray manager.

        Args:
            icon_path: Path to tray icon image (PNG recommended)
            on_show: Callback when "Show" menu item clicked
            on_quit: Callback when "Quit" menu item clicked
            on_pause: Callback when "Pause" menu item clicked
            on_resume: Callback when "Resume" menu item clicked
        """
        self._icon_path = icon_path
        self._on_show = on_show
        self._on_quit = on_quit
        self._on_pause = on_pause
        self._on_resume = on_resume

        self._icon: Optional[Icon] = None
        self._icon_image: Optional[Image.Image] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._tooltip = "DLER - Ready"
        self._is_paused = False

        # Load icon image
        self._load_icon()

    def _load_icon(self) -> None:
        """Load tray icon image."""
        if not HAS_PYSTRAY:
            logger.warning("pystray not available, system tray disabled")
            return

        if self._icon_path and self._icon_path.exists():
            try:
                self._icon_image = Image.open(self._icon_path)
                # Resize for tray (typically 16x16 or 32x32)
                self._icon_image = self._icon_image.resize((32, 32), Image.Resampling.LANCZOS)
                logger.debug(f"Loaded tray icon from {self._icon_path}")
            except Exception as e:
                logger.warning(f"Failed to load tray icon: {e}")
                self._create_default_icon()
        else:
            self._create_default_icon()

    def _create_default_icon(self) -> None:
        """Create a default icon if none provided."""
        if not HAS_PYSTRAY:
            return

        # Create a simple blue square icon
        self._icon_image = Image.new('RGB', (32, 32), color=(51, 153, 255))
        logger.debug("Created default tray icon")

    def _create_menu(self) -> tuple:
        """Create tray context menu."""
        if not HAS_PYSTRAY:
            return ()

        def on_show_click(icon, item):
            if self._on_show:
                self._on_show()

        def on_pause_click(icon, item):
            if self._is_paused:
                if self._on_resume:
                    self._on_resume()
                self._is_paused = False
            else:
                if self._on_pause:
                    self._on_pause()
                self._is_paused = True
            # Update menu
            self._update_menu()

        def on_quit_click(icon, item):
            if self._on_quit:
                self._on_quit()

        pause_text = "Resume" if self._is_paused else "Pause"

        return pystray.Menu(
            pystray.MenuItem("Show DLER", on_show_click, default=True),
            pystray.MenuItem(pause_text, on_pause_click),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit_click),
        )

    def _update_menu(self) -> None:
        """Update tray menu (e.g., after pause state change)."""
        if self._icon:
            self._icon.menu = self._create_menu()

    def start(self) -> bool:
        """
        Start system tray icon.

        Returns:
            True if started successfully, False otherwise
        """
        if not HAS_PYSTRAY:
            logger.warning("Cannot start tray: pystray not available")
            return False

        if self._running:
            logger.debug("Tray already running")
            return True

        if not self._icon_image:
            logger.error("Cannot start tray: no icon image")
            return False

        try:
            self._icon = pystray.Icon(
                name="DLER",
                icon=self._icon_image,
                title=self._tooltip,
                menu=self._create_menu()
            )

            # Run in background thread
            self._thread = threading.Thread(target=self._icon.run, daemon=True)
            self._thread.start()
            self._running = True

            logger.info("System tray started")
            return True

        except Exception as e:
            logger.error(f"Failed to start system tray: {e}")
            return False

    def stop(self) -> None:
        """Stop system tray icon."""
        if self._icon and self._running:
            try:
                self._icon.stop()
                self._running = False
                logger.info("System tray stopped")
            except Exception as e:
                logger.debug(f"Error stopping tray: {e}")

    def update_tooltip(self, text: str) -> None:
        """
        Update tray icon tooltip text.

        Args:
            text: New tooltip text
        """
        self._tooltip = text
        if self._icon:
            self._icon.title = text

    def set_paused(self, paused: bool) -> None:
        """Update pause state for menu."""
        self._is_paused = paused
        self._update_menu()

    @property
    def is_available(self) -> bool:
        """Check if system tray is available."""
        return HAS_PYSTRAY

    @property
    def is_running(self) -> bool:
        """Check if tray icon is running."""
        return self._running


# =============================================================================
# Windows Auto-Start Management
# =============================================================================

def get_executable_path() -> Path:
    """Get the path to the current executable."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        return Path(sys.executable)
    else:
        # Running as script
        return Path(sys.argv[0]).resolve()


def is_autostart_enabled() -> bool:
    """
    Check if DLER is set to auto-start with Windows.

    Returns:
        True if auto-start is enabled, False otherwise
    """
    if sys.platform != 'win32':
        logger.debug("Auto-start only supported on Windows")
        return False

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            AUTOSTART_KEY,
            0,
            winreg.KEY_READ
        )
        try:
            value, _ = winreg.QueryValueEx(key, APP_NAME)
            winreg.CloseKey(key)
            return True
        except FileNotFoundError:
            winreg.CloseKey(key)
            return False
    except Exception as e:
        logger.debug(f"Error checking auto-start: {e}")
        return False


def enable_autostart(start_minimized: bool = True) -> bool:
    """
    Enable DLER auto-start with Windows.

    Args:
        start_minimized: If True, adds --minimized flag to startup command

    Returns:
        True if successful, False otherwise
    """
    if sys.platform != 'win32':
        logger.warning("Auto-start only supported on Windows")
        return False

    try:
        exe_path = get_executable_path()

        # Build command with optional --minimized flag
        if start_minimized:
            command = f'"{exe_path}" --minimized'
        else:
            command = f'"{exe_path}"'

        # Open registry key
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            AUTOSTART_KEY,
            0,
            winreg.KEY_SET_VALUE
        )

        # Set value
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, command)
        winreg.CloseKey(key)

        logger.info(f"Auto-start enabled: {command}")
        return True

    except Exception as e:
        logger.error(f"Failed to enable auto-start: {e}")
        return False


def disable_autostart() -> bool:
    """
    Disable DLER auto-start with Windows.

    Returns:
        True if successful, False otherwise
    """
    if sys.platform != 'win32':
        logger.warning("Auto-start only supported on Windows")
        return False

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            AUTOSTART_KEY,
            0,
            winreg.KEY_SET_VALUE
        )

        try:
            winreg.DeleteValue(key, APP_NAME)
            logger.info("Auto-start disabled")
        except FileNotFoundError:
            logger.debug("Auto-start was not enabled")

        winreg.CloseKey(key)
        return True

    except Exception as e:
        logger.error(f"Failed to disable auto-start: {e}")
        return False


def set_autostart(enabled: bool, start_minimized: bool = True) -> bool:
    """
    Set auto-start state.

    Args:
        enabled: Whether to enable or disable auto-start
        start_minimized: If enabling, whether to start minimized

    Returns:
        True if successful, False otherwise
    """
    if enabled:
        return enable_autostart(start_minimized)
    else:
        return disable_autostart()
