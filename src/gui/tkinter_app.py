"""
DLER TKinter Application
========================

High-end professional interface using TKinterModernThemes.
Fixed-size frames, adaptive units, elegant design.
"""

from __future__ import annotations

import threading
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import TKinterModernThemes as TKMT
    HAS_TKMT = True
except ImportError:
    HAS_TKMT = False

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from .speed_graph import SpeedGraphWidget
    HAS_SPEED_GRAPH = True
except ImportError:
    HAS_SPEED_GRAPH = False

try:
    from .system_tray import SystemTrayManager, set_autostart, is_autostart_enabled
    HAS_SYSTEM_TRAY = True
except ImportError:
    HAS_SYSTEM_TRAY = False
    SystemTrayManager = None

# Drag and drop support (tkinterdnd2 - works at Tcl/Tk level, no Windows hook conflicts)
HAS_DND = False
DND_FILES = None
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    TkinterDnD = None

HAS_RAM_PROCESSOR = False
try:
    # Test NumPy first - Python 3.14 has compatibility issues
    import numpy as np
    _test = np.array([1, 2, 3])  # Simple test to trigger any delayed errors
    del _test

    from ..core.ram_processor import RamBuffer, RamPostProcessor
    HAS_RAM_PROCESSOR = True
except (ImportError, TypeError, Exception) as e:
    # TypeError can occur with Python 3.14 + NumPy/CuPy (add_docstring bug)
    HAS_RAM_PROCESSOR = False
    logging.getLogger(__name__).warning(f"RAM processor not available: {e}")

# Adaptive Extractor for intelligent NZB classification
HAS_ADAPTIVE_EXTRACTOR = False
try:
    from ..core.adaptive_extractor import ReleaseClassifier, ExtractionStrategy, ReleaseType
    from ..core.nzb_parser import NZBParser
    HAS_ADAPTIVE_EXTRACTOR = True
except ImportError as e:
    logging.getLogger(__name__).debug(f"Adaptive extractor not available: {e}")

# Edition detection (set by runtime hook in PyInstaller build)
# Ultimate = GPU/CUDA support, Basic = CPU-only
import os
IS_ULTIMATE_EDITION = os.environ.get('DLER_EDITION', 'ultimate').lower() == 'ultimate'

# CuPy/CUDA detection for GPU acceleration (only in Ultimate edition)
HAS_CUDA = False
_cuda_error = None
if IS_ULTIMATE_EDITION:
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            HAS_CUDA = True
            logging.getLogger(__name__).info(f"CUDA detected: {device_count} device(s)")
    except ImportError as e:
        _cuda_error = f"CuPy import failed: {e}"
        logging.getLogger(__name__).warning(_cuda_error)
    except Exception as e:
        _cuda_error = f"CUDA detection failed: {e}"
        logging.getLogger(__name__).warning(_cuda_error)

logger = logging.getLogger(__name__)

# Fixed dimensions for consistent layout
SPEED_PANEL_WIDTH = 200
SPEED_PANEL_HEIGHT = 150
PROGRESS_PANEL_HEIGHT = 150
GRAPH_PANEL_HEIGHT = 95
STATS_PANEL_HEIGHT = 85
QUEUE_PANEL_HEIGHT = 120

# Font definitions (consistent throughout)
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_SUBTITLE = ("Segoe UI", 10)
FONT_SPEED_VALUE = ("Consolas", 38, "bold")
FONT_SPEED_UNIT = ("Segoe UI", 12)
FONT_LABEL = ("Segoe UI", 10)
FONT_VALUE = ("Consolas", 11, "bold")
FONT_VALUE_SMALL = ("Consolas", 10)
FONT_SECTION = ("Segoe UI", 9, "bold")


def format_size(bytes_val: int, show_zero: bool = True) -> str:
    """Format bytes with adaptive units (fixed width output)."""
    if bytes_val == 0:
        return "0.00 MB" if show_zero else "---"
    elif bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f} MB"
    elif bytes_val < 1024 * 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024 * 1024):.2f} TB"


def format_speed(mbps: float) -> tuple[str, str]:
    """Format speed with adaptive units. Returns (value, unit)."""
    if mbps < 0.001:
        return "0.0", "KB/s"
    elif mbps < 1.0:
        return f"{mbps * 1024:.1f}", "KB/s"
    elif mbps < 1000.0:
        return f"{mbps:.1f}", "MB/s"
    else:
        return f"{mbps / 1024:.2f}", "GB/s"


@dataclass
class DownloadState:
    """Current download state for UI updates."""
    speed_mbps: float = 0.0
    progress_percent: float = 0.0
    eta_seconds: float = 0.0
    current_file: str = ""
    downloaded_bytes: int = 0
    total_bytes: int = 0
    segments_done: int = 0
    segments_total: int = 0
    nzb_segments_total: int = 0  # Parsed from NZB, never overwritten
    connections_active: int = 0
    connections_total: int = 0
    queue_raw: int = 0
    queue_write: int = 0
    is_downloading: bool = False
    is_post_processing: bool = False
    status_message: str = "Ready"


class DLERApp:
    """DLER NZB Downloader - High-end professional interface."""

    def __init__(self, start_minimized: bool = False):
        self._state = DownloadState()
        self._turbo_engine = None
        self._config = None
        self._download_queue: list[Path] = []
        self._update_job = None
        self._logo_image = None
        self._icon_image = None
        self._is_paused = False
        self._download_complete = False  # Prevents progress updates after completion
        self._speed_graph = None  # Speed graph widget
        self._system_tray: Optional[SystemTrayManager] = None  # System tray manager
        self._start_minimized = start_minimized  # Start minimized to tray

        self._setup_ui()
        self._load_config()
        self._setup_system_tray()
        # Setup drag-and-drop AFTER system tray (they conflict - windnd vs pystray)
        self._setup_drag_and_drop()

    def _setup_ui(self) -> None:
        """Setup the high-end interface with fixed dimensions."""
        if HAS_TKMT:
            self._root = TKMT.ThemedTKinterFrame(
                "DLER",
                theme="park",
                mode="dark",
                usecommandlineargs=False,
                useconfigfile=False
            )
            self._master = self._root.master
        else:
            self._root = tk.Tk()
            self._root.title("DLER")
            self._master = self._root

        self._master.geometry("720x710")
        self._master.resizable(False, False)

        # Set window icon (taskbar icon)
        self._set_window_icon()

        # Handle window close button (X)
        self._master.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Main container
        main_container = ttk.Frame(self._master, padding=20)
        main_container.pack(fill="both", expand=True)

        # Fixed grid weights
        main_container.columnconfigure(0, weight=0, minsize=SPEED_PANEL_WIDTH)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(4, weight=1)  # Queue panel expands

        self._create_header(main_container)
        self._create_speed_panel(main_container)
        self._create_progress_panel(main_container)
        self._create_speed_graph_panel(main_container)  # NEW: Speed graph
        self._create_stats_panel(main_container)
        self._create_queue_panel(main_container)
        self._create_action_bar(main_container)

        # Start UI update loop
        self._schedule_update()

    def _setup_drag_and_drop(self) -> None:
        """Setup drag-and-drop for NZB files using tkinterdnd2.

        tkinterdnd2 works at the Tcl/Tk level (not Windows message hooks),
        so it's compatible with pystray system tray.

        We use Tcl commands directly to work with any existing Tk window
        (including TKinterModernThemes).

        NOTE: We use Tcl callbacks instead of Python bind() to avoid
        the "expected integer but got %#" TclError bug in tkinter.
        See: https://github.com/python/cpython/issues/94861
        """
        if not HAS_DND:
            logger.debug("Drag-and-drop not available (tkinterdnd2 not installed)")
            return

        try:
            # Load TkDND extension into existing Tk interpreter
            TkinterDnD._require(self._master)

            # Register window as drop target using Tcl commands directly
            self._master.tk.call('tkdnd::drop_target', 'register', self._master, DND_FILES)

            # Create a Tcl command that calls our Python handler
            # This avoids the tkinter event substitution bug
            def drop_handler(data):
                self._on_drop_files(data)
                return 'copy'  # Return the action

            # Register the Python function as a Tcl command
            drop_cmd = self._master.register(drop_handler)

            # Bind to TkDND drop events using Tcl eval
            # TkDND generates <<Drop:TYPE>> events for specific types
            # The %D substitution contains dropped file paths
            # We wrap %D in braces to pass it as a single argument (paths may contain spaces)
            for event in ('<<Drop>>', '<<Drop:DND_Files>>'):
                self._master.tk.eval(f'bind {self._master} {event} {{{drop_cmd} {{%D}}}}')

            logger.info("Drag-and-drop enabled for NZB files (tkinterdnd2)")
        except Exception as e:
            logger.warning(f"Could not setup drag-and-drop: {e}")

    def _on_drop_files(self, raw_data: str) -> None:
        """Handle dropped files (NZB files). Called via Tcl callback with file paths.

        tkdnd passes file paths as a string. Paths with spaces are enclosed
        in curly braces: {path with spaces}
        """
        # Parse file paths from tkdnd format
        files = []
        if '{' in raw_data:
            # Handle curly brace notation for paths with spaces
            import re
            brace_paths = re.findall(r'\{([^}]+)\}', raw_data)
            remaining = re.sub(r'\{[^}]+\}', '', raw_data).strip()
            files.extend(brace_paths)
            if remaining:
                files.extend(remaining.split())
        else:
            files = raw_data.split()

        # Filter for NZB files only
        nzb_files = []
        for f in files:
            path = Path(f)
            if path.exists() and path.suffix.lower() == '.nzb':
                nzb_files.append(path)

        if not nzb_files:
            self._log("No valid NZB files dropped", "warning")
            return

        # Add to queue
        for path in nzb_files:
            self._download_queue.append(path)
            self._log(f"Queued: {path.name}", "info")
            logger.info(f"Queued (drag-drop): {path.name}")

        self._update_queue_info()

        # Start download if not already running
        if not self._state.is_downloading:
            self._start_next_download()

    def _get_logo_path(self) -> Optional[Path]:
        """Get the path to logo.png."""
        logo_path = Path(__file__).parent.parent.parent / "logo.png"
        if not logo_path.exists():
            logo_path = Path("C:/dler/logo.png")
        return logo_path if logo_path.exists() else None

    def _set_window_icon(self) -> None:
        """Set the window/taskbar icon."""
        if not HAS_PIL:
            return

        try:
            logo_path = self._get_logo_path()
            if logo_path:
                img = Image.open(logo_path)
                # Create icon in multiple sizes for taskbar
                icon_img = ImageTk.PhotoImage(img.resize((32, 32), Image.Resampling.LANCZOS))
                self._icon_image = icon_img  # Keep reference
                self._master.iconphoto(True, icon_img)
        except Exception as e:
            logger.warning(f"Could not set window icon: {e}")

    def _load_logo(self) -> Optional[ImageTk.PhotoImage]:
        """Load logo.png for header display."""
        if not HAS_PIL:
            return None

        try:
            logo_path = self._get_logo_path()
            if logo_path:
                img = Image.open(logo_path)
                # Resize to fit header (40x40)
                img = img.resize((40, 40), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(img)
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")

        return None

    def _create_header(self, parent) -> None:
        """Create header with logo and title."""
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        header.columnconfigure(1, weight=1)

        # Logo + Title frame
        title_frame = ttk.Frame(header)
        title_frame.grid(row=0, column=0, sticky="w")

        # Load and display logo
        self._logo_image = self._load_logo()
        if self._logo_image:
            logo_label = ttk.Label(title_frame, image=self._logo_image)
            logo_label.pack(side="left", padx=(0, 12))

        # Title text
        text_frame = ttk.Frame(title_frame)
        text_frame.pack(side="left")

        title = ttk.Label(text_frame, text="DLER", font=FONT_TITLE)
        title.pack(anchor="w")

        subtitle = ttk.Label(text_frame, text="NZB Downloader", font=FONT_SUBTITLE)
        subtitle.pack(anchor="w")

        # Settings button
        self._settings_btn = ttk.Button(
            header,
            text="Settings",
            command=self._show_settings,
            width=12
        )
        self._settings_btn.grid(row=0, column=1, sticky="e")

    def _create_speed_panel(self, parent) -> None:
        """Create fixed-size speed display panel."""
        # Outer frame with fixed dimensions
        speed_outer = ttk.Frame(parent, width=SPEED_PANEL_WIDTH, height=SPEED_PANEL_HEIGHT)
        speed_outer.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        speed_outer.grid_propagate(False)  # FIXED SIZE

        # LabelFrame inside
        speed_frame = ttk.LabelFrame(speed_outer, text=" TRANSFER ", padding=8)
        speed_frame.pack(fill="both", expand=True)

        # Inner display (LCD effect)
        display = ttk.Frame(speed_frame, relief="sunken", borderwidth=2)
        display.pack(fill="both", expand=True, padx=4, pady=4)

        # Center content
        center = ttk.Frame(display)
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Speed value (fixed width, larger)
        self._speed_var = tk.StringVar(value="0.0")
        speed_label = ttk.Label(
            center,
            textvariable=self._speed_var,
            font=FONT_SPEED_VALUE,
            width=7,
            anchor="center"
        )
        speed_label.pack()

        # Unit (adaptive)
        self._speed_unit_var = tk.StringVar(value="MB/s")
        unit_label = ttk.Label(
            center,
            textvariable=self._speed_unit_var,
            font=FONT_SPEED_UNIT
        )
        unit_label.pack()

    def _create_progress_panel(self, parent) -> None:
        """Create fixed-size progress panel."""
        # Outer frame with fixed height
        progress_outer = ttk.Frame(parent, height=PROGRESS_PANEL_HEIGHT)
        progress_outer.grid(row=1, column=1, sticky="nsew", pady=(0, 12))
        progress_outer.grid_propagate(False)  # FIXED HEIGHT

        # LabelFrame inside
        progress_frame = ttk.LabelFrame(progress_outer, text=" PROGRESS ", padding=12)
        progress_frame.pack(fill="both", expand=True)
        progress_frame.columnconfigure(0, weight=1)

        # File name (fixed width with ellipsis)
        self._file_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(
            progress_frame,
            textvariable=self._file_var,
            font=("Segoe UI", 10, "bold"),
            width=50,
            anchor="w"
        )
        file_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Progress bar
        bar_frame = ttk.Frame(progress_frame)
        bar_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        bar_frame.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            bar_frame,
            variable=self._progress_var,
            maximum=100,
            mode='determinate',
            length=350
        )
        self._progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 12))

        # Percentage (fixed width)
        self._percent_var = tk.StringVar(value="  0.0%")
        percent_label = ttk.Label(
            bar_frame,
            textvariable=self._percent_var,
            font=FONT_VALUE,
            width=7,
            anchor="e"
        )
        percent_label.grid(row=0, column=1)

        # Bottom row: ETA and size
        bottom = ttk.Frame(progress_frame)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew")
        bottom.columnconfigure(1, weight=1)

        # ETA
        eta_frame = ttk.Frame(bottom)
        eta_frame.grid(row=0, column=0, sticky="w")

        ttk.Label(eta_frame, text="ETA:", font=FONT_LABEL).pack(side="left")
        self._eta_var = tk.StringVar(value="  --:--")
        ttk.Label(
            eta_frame,
            textvariable=self._eta_var,
            font=FONT_VALUE,
            width=10,
            anchor="w"
        ).pack(side="left", padx=(8, 0))

        # Size
        size_frame = ttk.Frame(bottom)
        size_frame.grid(row=0, column=1, sticky="e")

        self._size_var = tk.StringVar(value="0.00 MB / 0.00 MB")
        ttk.Label(
            size_frame,
            textvariable=self._size_var,
            font=FONT_VALUE_SMALL,
            anchor="e"
        ).pack(side="right")

    def _create_speed_graph_panel(self, parent) -> None:
        """Create speed graph panel with Windows 11-style animated chart."""
        if not HAS_SPEED_GRAPH:
            logger.warning("SpeedGraphWidget not available")
            return

        # Outer frame with fixed height
        graph_outer = ttk.Frame(parent, height=GRAPH_PANEL_HEIGHT)
        graph_outer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        graph_outer.grid_propagate(False)  # FIXED HEIGHT

        # LabelFrame inside
        graph_frame = ttk.LabelFrame(graph_outer, text=" SPEED HISTORY ", padding=4)
        graph_frame.pack(fill="both", expand=True)
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)

        # Create the speed graph widget (keeps all data, compresses to fit)
        self._speed_graph = SpeedGraphWidget(
            graph_frame,
            width=650,
            height=70,
            update_interval_ms=100
        )
        self._speed_graph.pack(fill="both", expand=True, padx=1, pady=1)

        # Start animation
        self._speed_graph.start_animation()

    def _create_stats_panel(self, parent) -> None:
        """Create statistics panel with elegant subframes."""
        # Outer frame
        stats_outer = ttk.Frame(parent, height=STATS_PANEL_HEIGHT)
        stats_outer.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        stats_outer.grid_propagate(False)  # FIXED HEIGHT

        # Main container for 3 metric boxes
        stats_container = ttk.Frame(stats_outer)
        stats_container.pack(fill="both", expand=True)

        # Equal distribution
        stats_container.columnconfigure(0, weight=1, uniform="stat")
        stats_container.columnconfigure(1, weight=1, uniform="stat")
        stats_container.columnconfigure(2, weight=1, uniform="stat")
        stats_container.rowconfigure(0, weight=1)

        # Segments subframe
        self._segments_var = tk.StringVar(value="0 / 0")
        self._create_metric_box(stats_container, "SEGMENTS", self._segments_var, 0)

        # Connections subframe
        self._conn_var = tk.StringVar(value="0 / 0")
        self._create_metric_box(stats_container, "CONNECTIONS", self._conn_var, 1)

        # Queues subframe
        self._queues_var = tk.StringVar(value="0 / 0")
        self._create_metric_box(stats_container, "QUEUES", self._queues_var, 2)

    def _create_metric_box(self, parent, title: str, var: tk.StringVar, col: int) -> None:
        """Create an elegant metric box with title and value."""
        # Outer LabelFrame
        box = ttk.LabelFrame(parent, text=f" {title} ", padding=8)
        box.grid(row=0, column=col, sticky="nsew", padx=4, pady=2)

        # Inner display frame (LCD effect)
        display = ttk.Frame(box, relief="sunken", borderwidth=1)
        display.pack(fill="both", expand=True, padx=2, pady=2)

        # Centered value
        value_label = ttk.Label(
            display,
            textvariable=var,
            font=FONT_VALUE,
            anchor="center"
        )
        value_label.pack(expand=True, fill="both", pady=4)

    def _create_queue_panel(self, parent) -> None:
        """Create fixed-size queue panel."""
        # Outer frame
        queue_outer = ttk.Frame(parent)
        queue_outer.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 12))
        queue_outer.columnconfigure(0, weight=1)
        queue_outer.rowconfigure(0, weight=1)

        # LabelFrame for logs
        log_frame = ttk.LabelFrame(queue_outer, text=" LOGS ", padding=10)
        log_frame.pack(fill="both", expand=True)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # Text widget for logs (read-only)
        list_frame = ttk.Frame(log_frame)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._log_text = tk.Text(
            list_frame,
            height=4,
            font=FONT_VALUE_SMALL,
            wrap="none",
            state="disabled",
            cursor="arrow"
        )
        self._log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self._log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._log_text.configure(yscrollcommand=scrollbar.set)

        # Configure log text tags for colors
        self._log_text.tag_configure("time", foreground="#666666")
        self._log_text.tag_configure("info", foreground="#808080")
        self._log_text.tag_configure("success", foreground="#4CAF50")
        self._log_text.tag_configure("warning", foreground="#FFA726")
        self._log_text.tag_configure("error", foreground="#EF5350")

        # Status bar
        status_frame = ttk.Frame(log_frame)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self._queue_info_var = tk.StringVar(value="Ready")
        ttk.Label(
            status_frame,
            textvariable=self._queue_info_var,
            font=FONT_LABEL
        ).pack(side="left")

        # Downloaded total
        self._total_var = tk.StringVar(value="")
        ttk.Label(
            status_frame,
            textvariable=self._total_var,
            font=FONT_VALUE_SMALL
        ).pack(side="right")

        # Keep reference for compatibility (some code still uses _queue_listbox)
        self._queue_listbox = None

    def _create_action_bar(self, parent) -> None:
        """Create action buttons."""
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        action_frame.columnconfigure(1, weight=1)

        # Add NZB
        self._add_btn = ttk.Button(
            action_frame,
            text="+ Add NZB",
            command=self._add_nzb,
            width=14
        )
        self._add_btn.grid(row=0, column=0, sticky="w")

        # Right buttons
        btn_frame = ttk.Frame(action_frame)
        btn_frame.grid(row=0, column=2, sticky="e")

        self._pause_btn = tk.Button(
            btn_frame,
            text="Pause",
            command=self._toggle_pause,
            width=10,
            state="disabled",
            font=FONT_LABEL
        )
        self._pause_btn.pack(side="left", padx=(0, 8))

        self._stop_btn = tk.Button(
            btn_frame,
            text="Stop",
            command=self._stop_download,
            width=10,
            state="disabled",
            font=FONT_LABEL
        )
        self._stop_btn.pack(side="left")

    def _load_config(self) -> None:
        """Load configuration."""
        try:
            from ..utils.config import load_config
            self._config = load_config()
            logger.info("Configuration loaded")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            self._config = None

    def _log(self, message: str, level: str = "info") -> None:
        """
        Add a message to the logs panel.

        Args:
            message: The message to display
            level: One of 'info', 'success', 'warning', 'error'
        """
        from datetime import datetime

        def update():
            self._log_text.configure(state="normal")

            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._log_text.insert("end", f"{timestamp} ", "time")

            # Add message with appropriate tag
            self._log_text.insert("end", f"{message}\n", level)

            # Auto-scroll to bottom
            self._log_text.see("end")

            # Limit to last 100 lines to prevent memory issues
            lines = int(self._log_text.index('end-1c').split('.')[0])
            if lines > 100:
                self._log_text.delete("1.0", f"{lines - 100}.0")

            self._log_text.configure(state="disabled")

        # Schedule on main thread if called from background thread
        self._master.after(0, update)

    def _show_settings(self) -> None:
        """Show settings dialog."""
        SettingsDialog(self._master, self._config, self._on_settings_saved)

    def _on_settings_saved(self, config) -> None:
        """Handle settings save."""
        self._config = config
        self._queue_info_var.set("Settings saved")
        if self._turbo_engine and not self._state.is_downloading:
            self._turbo_engine = None

    def _add_nzb(self) -> None:
        """Add NZB files."""
        files = filedialog.askopenfilenames(
            title="Select NZB Files",
            filetypes=[("NZB Files", "*.nzb"), ("All Files", "*.*")]
        )

        if not files:
            return

        for f in files:
            path = Path(f)
            self._download_queue.append(path)
            self._log(f"Queued: {path.name}", "info")
            logger.info(f"Queued: {path.name}")

        self._update_queue_info()

        if not self._state.is_downloading:
            self._start_next_download()

    def _update_queue_info(self) -> None:
        """Update queue info."""
        count = len(self._download_queue)
        if count == 0:
            self._queue_info_var.set("Queue empty")
        elif count == 1:
            self._queue_info_var.set("1 file in queue")
        else:
            self._queue_info_var.set(f"{count} files in queue")

    def _start_next_download(self) -> None:
        """Start next download."""
        if not self._download_queue:
            self._state.is_downloading = False
            self._stop_btn.config(state="disabled")
            self._pause_btn.config(state="disabled")
            self._queue_info_var.set("Queue complete")
            return

        if not self._config or not self._config.server.host:
            messagebox.showerror("Configuration Error", "Please configure server settings first.")
            return

        nzb_path = self._download_queue.pop(0)
        self._state.current_file = nzb_path.name
        self._state.is_downloading = True
        self._download_complete = False  # Reset for new download
        self._state.progress_percent = 0  # Reset progress
        self._stop_btn.config(state="normal")
        self._pause_btn.config(state="normal")

        # Reset speed graph for new download
        if self._speed_graph:
            self._speed_graph.reset()

        # Log download start
        self._log(f"Starting: {nzb_path.name}", "info")

        thread = threading.Thread(target=self._download_thread, args=(nzb_path,), daemon=True)
        thread.start()

    def _download_thread(self, nzb_path: Path) -> None:
        """Background download."""
        streaming_pp = None  # Streaming post-processor for early PAR2
        streaming_extractor = None  # Streaming RAR extractor
        direct_to_destination = False  # True if downloading directly to final location
        release_title = nzb_path.stem
        has_archives = True  # Default to True for safety
        is_obfuscated = False  # Assume not obfuscated unless detected
        extraction_plan = None  # Adaptive extraction plan (set by classifier)

        try:
            if self._turbo_engine is None:
                self._init_engine()

            if self._turbo_engine is None:
                self._queue_info_var.set("Failed to initialize")
                self._state.is_downloading = False
                return

            # === NZB ANALYSIS: Get metadata before download ===
            nzb_password = None  # Password from NZB metadata

            # Extract password from NZB ALWAYS (needed for RAM mode even if disk post-processing disabled)
            try:
                from ..core.post_processor import PostProcessor
                metadata = PostProcessor.parse_nzb_metadata(nzb_path)
                if metadata.password:
                    nzb_password = metadata.password
                    logger.info(f"[NZB] Password extracted: {'*' * len(nzb_password)}")
            except Exception as e:
                logger.warning(f"[NZB] Could not extract password: {e}")

            if (self._config and
                self._config.postprocess.enabled and
                self._config.postprocess.extract_dir):
                try:
                    from ..core.post_processor import quick_nzb_analysis

                    release_title, has_archives, is_obfuscated = quick_nzb_analysis(nzb_path)

                    # === ADAPTIVE CLASSIFICATION: Analyze NZB before download ===
                    extraction_plan = None
                    if HAS_ADAPTIVE_EXTRACTOR and has_archives:
                        try:
                            nzb_doc = NZBParser.parse(nzb_path)
                            classifier = ReleaseClassifier()
                            classification = classifier.classify(nzb_doc.get_main_files())

                            extraction_plan = ExtractionStrategy.create_plan(classification)

                            logger.info(f"[ADAPTIVE] Release type: {classification.release_type.value} "
                                       f"(confidence: {classification.confidence:.0%})")
                            logger.info(f"[ADAPTIVE] Plan: {len(extraction_plan.stages)} stages, "
                                       f"expects_nested={extraction_plan.expects_nested}")

                            # Override obfuscated detection with classification result
                            if classification.is_obfuscated:
                                is_obfuscated = True
                                logger.info("[ADAPTIVE] Confirmed obfuscated NZB")

                        except Exception as e:
                            logger.warning(f"[ADAPTIVE] Classification failed: {e}, will use reactive extraction")
                            extraction_plan = None

                    if not has_archives:
                        # No RAR/ZIP → download directly to final destination
                        final_dir = Path(self._config.postprocess.extract_dir) / release_title
                        final_dir.mkdir(parents=True, exist_ok=True)

                        self._turbo_engine.output_dir = final_dir
                        direct_to_destination = True
                        logger.info(f"DIRECT-TO-DESTINATION: No archives detected, downloading to {final_dir}")
                    else:
                        # Has archives → use temp dir (normal flow)
                        self._turbo_engine.output_dir = Path(self._config.download.output_dir)
                        logger.info(f"NORMAL FLOW: Archives detected, downloading to temp dir")

                except Exception as e:
                    logger.warning(f"NZB analysis failed: {e}, using normal flow")
                    self._turbo_engine.output_dir = Path(self._config.download.output_dir)

            # Get actual download dir after potential override
            download_dir = self._turbo_engine.output_dir

            # === STREAMING PAR2: Setup early verification ===
            if (self._config and
                self._config.postprocess.enabled and
                getattr(self._config.postprocess, 'streaming_par2', False)):
                try:
                    from ..core.post_processor import StreamingPostProcessor

                    streaming_pp = StreamingPostProcessor(
                        download_dir=download_dir,
                        par2_path=getattr(self._config.postprocess, 'par2_path', None),
                        on_early_issue=lambda msg: self._queue_info_var.set(f"PAR2: {msg[:25]}..."),
                        on_progress=lambda msg, _: logger.debug(f"Streaming PAR2: {msg}")
                    )
                    logger.info("Streaming PAR2 enabled - will verify during download")

                except Exception as e:
                    logger.warning(f"Streaming PAR2 setup failed: {e}")
                    streaming_pp = None

            # === STREAMING EXTRACTION: Setup RAR extraction during download ===
            # IMPORTANT: Disabled for obfuscated NZBs because we can't reliably detect
            # when all RAR parts are complete (filenames aren't known until download)
            if (self._config and
                self._config.postprocess.enabled and
                has_archives and
                not direct_to_destination and
                not is_obfuscated):
                try:
                    from ..core.post_processor import StreamingExtractor

                    extract_dir = Path(self._config.postprocess.extract_dir)

                    streaming_extractor = StreamingExtractor(
                        download_dir=download_dir,
                        extract_dir=extract_dir / release_title,
                        sevenzip_path=getattr(self._config.postprocess, 'sevenzip_path', None),
                        password=nzb_password,  # Password from NZB metadata
                        max_parallel=2,  # Conservative during download
                        threads_per_extraction=4,
                        on_extraction_start=lambda name: self._log(f"Streaming extract: {name}...", "info"),
                        on_extraction_complete=lambda name, ok, n: self._log(
                            f"Extracted: {name} ({n} files)" if ok else f"Extract failed: {name}",
                            "success" if ok else "error"
                        )
                    )
                    if nzb_password:
                        logger.info(f"StreamingExtractor configured with NZB password")
                    logger.info("Streaming Extraction enabled - will extract RAR sets during download")

                except Exception as e:
                    logger.warning(f"Streaming Extraction setup failed: {e}")
                    streaming_extractor = None
            elif is_obfuscated and has_archives:
                logger.info("Streaming Extraction DISABLED for obfuscated NZB - using post-download extraction")

            # === COMBINED CALLBACK: Register with engine ===
            def combined_file_complete(file_path: Path, filename: str) -> None:
                """Combined callback for PAR2 streaming and RAR extraction."""
                if streaming_pp:
                    try:
                        streaming_pp.on_file_complete(file_path, filename)
                    except Exception as e:
                        logger.debug(f"Streaming PAR2 callback error: {e}")

                if streaming_extractor:
                    try:
                        streaming_extractor.on_file_complete(file_path, filename)
                    except Exception as e:
                        logger.debug(f"Streaming Extractor callback error: {e}")

            # Only register callback if at least one streaming feature is active
            if streaming_pp or streaming_extractor:
                self._turbo_engine.on_file_complete = combined_file_complete

            self._parse_nzb_size(nzb_path)

            # === RAM PROCESSING MODE ===
            ram_buffer = None
            use_ram_mode = False
            if (HAS_RAM_PROCESSOR and
                self._config and
                self._config.ram_processing.enabled):
                # Check if NZB fits in RAM limit
                nzb_size_mb = self._state.total_bytes / (1024 * 1024)
                max_ram_mb = self._config.ram_processing.max_size_mb

                if nzb_size_mb <= max_ram_mb:
                    try:
                        ram_buffer = RamBuffer(max_size_mb=max_ram_mb)
                        use_ram_mode = True
                        logger.info(f"[RAM MODE] Enabled: {nzb_size_mb:.0f} MB fits in {max_ram_mb} MB limit")
                        logger.info(f"[RAM MODE] GPU repair: {self._config.ram_processing.gpu_repair}")

                        # Configure engine for RAM mode
                        self._turbo_engine.ram_buffer = ram_buffer
                        self._turbo_engine.ram_mode = True
                    except Exception as e:
                        logger.warning(f"[RAM MODE] Failed to initialize: {e}")
                        ram_buffer = None
                        use_ram_mode = False
                else:
                    logger.info(f"[RAM MODE] Skipped: {nzb_size_mb:.0f} MB exceeds {max_ram_mb} MB limit")

            self._queue_info_var.set(f"Downloading...")
            logger.info(f"Starting: {nzb_path}")

            start_time = time.time()
            success = self._turbo_engine.download_nzb(nzb_path)
            elapsed = time.time() - start_time

            # Mark download as complete BEFORE setting final values
            # This prevents progress callback from overwriting
            self._download_complete = True

            if success:
                # Get final stats from engine
                if self._turbo_engine:
                    final_stats = self._turbo_engine._stats
                    self._state.downloaded_bytes = final_stats.bytes_written
                    self._state.segments_done = final_stats.segments_written
                    if final_stats.segments_total > 0:
                        self._state.segments_total = final_stats.segments_total

                # Force progress to 100% on completion
                self._state.progress_percent = 100.0
                self._state.eta_seconds = 0
                self._state.speed_mbps = 0

                avg_speed = self._state.downloaded_bytes / (elapsed * 1024 * 1024) if elapsed > 0 else 0
                self._queue_info_var.set(f"Complete! {avg_speed:.1f} MB/s avg")

                # Log download completion
                size_gb = self._state.downloaded_bytes / (1024 * 1024 * 1024)
                self._log(f"Download complete: {size_gb:.2f} GB in {elapsed:.0f}s ({avg_speed:.1f} MB/s)", "success")

                # Force immediate UI update on main thread
                self._master.after(0, self._force_100_percent)

                # === WAIT FOR STREAMING EXTRACTIONS ===
                streaming_extracted_files = 0
                remaining_archives = []

                if streaming_extractor:
                    self._log("Waiting for streaming extractions...", "info")
                    self._queue_info_var.set("Finishing extractions...")

                    # Wait for pending extractions (max 5 minutes)
                    streaming_extractor.wait_for_pending(timeout=300)

                    # Get results
                    completed = streaming_extractor.get_completed_extractions()
                    streaming_extracted_files = streaming_extractor.get_total_extracted_files()
                    remaining_archives = streaming_extractor.get_remaining_archives()

                    if completed:
                        success_count = sum(1 for _, ok, _ in completed if ok)
                        logger.info(f"Streaming extraction: {success_count}/{len(completed)} sets extracted, {streaming_extracted_files} files")
                        if success_count > 0:
                            self._log(f"Streaming: {streaming_extracted_files} files from {success_count} archive(s)", "success")

                    if remaining_archives:
                        logger.info(f"Remaining archives for post-processing: {len(remaining_archives)}")

                    # Shutdown extractor
                    streaming_extractor.shutdown()

                # === POST-PROCESSING (PAR2 + Extraction) ===
                logger.info(f"Download complete, checking post-processing...")
                logger.info(f"Config exists: {self._config is not None}")
                if self._config:
                    logger.info(f"Postprocess enabled: {self._config.postprocess.enabled}")

                if self._config and self._config.postprocess.enabled:
                    # === RAM MODE POST-PROCESSING ===
                    if use_ram_mode and ram_buffer and HAS_RAM_PROCESSOR:
                        try:
                            logger.info("[RAM MODE] Starting GPU-accelerated post-processing...")
                            self._log("RAM post-processing...", "info")
                            self._queue_info_var.set("RAM processing...")

                            # Note: Don't add release_title here - process() adds it via release_name
                            extract_dir = Path(self._config.postprocess.extract_dir)
                            extract_dir.mkdir(parents=True, exist_ok=True)

                            # Progress callback to update UI with actual stage
                            def ram_progress_callback(stage: str, pct: float):
                                logger.debug(f"[RAM] {stage}: {pct:.0f}%")
                                # Update UI with current stage (truncate long messages)
                                display_stage = stage[:40] + "..." if len(stage) > 40 else stage
                                self._queue_info_var.set(f"RAM: {display_stage}")

                            ram_pp = RamPostProcessor(
                                ram_buffer=ram_buffer,
                                extract_dir=extract_dir,
                                gpu_device_id=self._config.ram_processing.gpu_device_id,
                                verify_threads=self._config.ram_processing.verify_threads or 0,
                                password=nzb_password,  # Pass NZB password for encrypted archives
                                on_progress=ram_progress_callback
                            )

                            pp_start = time.time()
                            success, message = ram_pp.process(release_name=release_title, extraction_plan=extraction_plan)
                            pp_elapsed = time.time() - pp_start

                            if success:
                                self._log(f"RAM processing: {message} ({pp_elapsed:.1f}s)", "success")
                                logger.info(f"[RAM MODE] Completed: {message} in {pp_elapsed:.1f}s")
                                # Show completion summary popup (if enabled)
                                if self._config and self._config.ui.show_completion_summary:
                                    self._show_ram_completion_summary(
                                        ram_pp=ram_pp,
                                        download_size=self._state.downloaded_bytes,
                                        download_duration=elapsed
                                    )
                            else:
                                self._log(f"RAM processing failed: {message}", "error")
                                logger.error(f"[RAM MODE] Failed: {message}")
                                # Show error popup
                                self._master.after(100, lambda msg=message: messagebox.showerror(
                                    "RAM Processing Failed",
                                    f"{msg}\n\nCheck logs for details."
                                ))

                            # Cleanup RAM buffer
                            ram_buffer.clear()
                            logger.info("[RAM MODE] Buffer cleared")

                        except Exception as e:
                            logger.error(f"[RAM MODE] Post-processing error: {e}")
                            self._log(f"RAM processing error: {e}", "error")
                            # Fallback: flush to disk and use regular post-processing
                            logger.info("[RAM MODE] Falling back to disk-based processing...")
                            if ram_buffer:
                                ram_buffer.flush_to_disk(download_dir)
                                ram_buffer.clear()
                    else:
                        # === STANDARD POST-PROCESSING ===
                        # Get early PAR2 result from streaming verification if available
                        early_par2_result = None
                        if streaming_pp:
                            # Wait briefly for early PAR2 to finish if it's running
                            streaming_pp.wait_for_par2(timeout=5)
                            early_par2_result = streaming_pp.get_early_result()
                            if early_par2_result:
                                verified, _, msg = early_par2_result
                                logger.info(f"Early PAR2 result available: verified={verified}, {msg}")
                                if not verified:
                                    self._log("PAR2: Repair needed", "warning")

                        # Determine if we should skip extraction (streaming handled it)
                        should_skip_extraction = (streaming_extracted_files > 0 and not remaining_archives)

                        if should_skip_extraction:
                            self._log("All archives extracted during download!", "success")
                            logger.info("Streaming extraction handled all archives, skipping post-processing extraction")

                        self._log("Post-processing started...", "info")
                        logger.info("Starting post-processing...")
                        self._run_post_processing(
                            nzb_path,
                            early_par2_result=early_par2_result,
                            download_size=self._state.downloaded_bytes,
                            download_duration=elapsed,
                            direct_to_destination=direct_to_destination,
                            actual_download_dir=download_dir,
                            skip_extraction=should_skip_extraction,
                            streaming_extract_path=Path(self._config.postprocess.extract_dir) / release_title if should_skip_extraction else None
                        )
                    logger.info("Post-processing finished")
                else:
                    logger.info("Post-processing disabled or no config")
            else:
                # Download incomplete - but we can try PAR2 repair!
                logger.warning("Download returned False - attempting PAR2 repair...")
                self._queue_info_var.set("Download incomplete - trying PAR2 repair...")
                self._log("Download incomplete - attempting PAR2 repair", "warning")

                # Still attempt post-processing with PAR2 repair
                if self._config and self._config.postprocess.enabled:
                    # Get downloaded bytes for logging
                    downloaded_bytes = 0
                    if self._turbo_engine and self._turbo_engine._stats:
                        downloaded_bytes = self._turbo_engine._stats.bytes_written
                        segments_ok = self._turbo_engine._stats.segments_written
                        segments_total = self._turbo_engine._stats.segments_total
                        success_rate = segments_ok / max(1, segments_total) * 100
                        logger.info(f"[INCOMPLETE] Downloaded {segments_ok}/{segments_total} segments ({success_rate:.1f}%)")
                        logger.info(f"[INCOMPLETE] {downloaded_bytes / (1024**2):.1f} MB downloaded, attempting repair...")

                    # RAM MODE: Try GPU repair
                    if use_ram_mode and ram_buffer and HAS_RAM_PROCESSOR:
                        try:
                            logger.info("[INCOMPLETE] Attempting GPU-accelerated PAR2 repair...")
                            self._log("Attempting PAR2 repair (GPU)...", "warning")
                            self._queue_info_var.set("PAR2 repair (GPU)...")

                            extract_dir = Path(self._config.postprocess.extract_dir)
                            extract_dir.mkdir(parents=True, exist_ok=True)

                            def repair_progress_callback(stage: str, pct: float):
                                logger.debug(f"[REPAIR] {stage}: {pct:.0f}%")
                                display_stage = stage[:40] + "..." if len(stage) > 40 else stage
                                self._queue_info_var.set(f"Repair: {display_stage}")

                            ram_pp = RamPostProcessor(
                                ram_buffer=ram_buffer,
                                extract_dir=extract_dir,
                                gpu_device_id=self._config.ram_processing.gpu_device_id,
                                verify_threads=self._config.ram_processing.verify_threads or 0,
                                password=nzb_password,
                                on_progress=repair_progress_callback
                            )

                            pp_start = time.time()
                            repair_success, repair_message = ram_pp.process(release_name=release_title, extraction_plan=extraction_plan)
                            pp_elapsed = time.time() - pp_start

                            if repair_success:
                                self._log(f"PAR2 repair successful: {repair_message} ({pp_elapsed:.1f}s)", "success")
                                logger.info(f"[INCOMPLETE] Repair successful: {repair_message}")
                                self._queue_info_var.set("Repair successful!")
                            else:
                                self._log(f"PAR2 repair failed: {repair_message}", "error")
                                logger.error(f"[INCOMPLETE] Repair failed: {repair_message}")
                                self._queue_info_var.set("Repair failed")
                                # Show warning
                                self._master.after(100, lambda msg=repair_message: self._show_warning(
                                    "Download Incomplete - Repair Failed",
                                    f"Download was incomplete and PAR2 repair could not recover the files.\n\n"
                                    f"Error: {msg}\n\n"
                                    "Possible causes:\n"
                                    "- Not enough PAR2 recovery blocks\n"
                                    "- Too many missing segments\n"
                                    "- DMCA takedown"
                                ))

                            # Cleanup RAM buffer
                            ram_buffer.clear()
                            logger.info("[INCOMPLETE] Buffer cleared")

                        except Exception as e:
                            logger.error(f"[INCOMPLETE] Repair error: {e}")
                            self._log(f"Repair error: {e}", "error")
                            self._master.after(100, lambda err=str(e): self._show_warning(
                                "Repair Error",
                                f"An error occurred during PAR2 repair:\n\n{err}"
                            ))
                    else:
                        # Disk mode: try standard post-processing
                        logger.info("[INCOMPLETE] Attempting disk-based PAR2 repair...")
                        self._run_post_processing(
                            nzb_path,
                            early_par2_result=(False, False, "Download incomplete"),
                            download_size=downloaded_bytes,
                            download_duration=elapsed,
                            direct_to_destination=direct_to_destination,
                            actual_download_dir=download_dir
                        )
                else:
                    # Post-processing disabled - just show warning
                    self._master.after(100, lambda: self._show_warning(
                        "Incomplete Download",
                        "Download was incomplete and post-processing is disabled.\n\n"
                        "Enable post-processing to attempt PAR2 repair."
                    ))

        except Exception as e:
            logger.error(f"Download error: {e}")
            self._queue_info_var.set(f"Error: {e}")
            self._log(f"Error: {e}", "error")
        finally:
            # Reset RAM mode for next download
            if self._turbo_engine:
                self._turbo_engine.ram_buffer = None
                self._turbo_engine.ram_mode = False

            self._state.is_downloading = False
            self._state.current_file = ""
            self._update_queue_info()
            self._master.after(500, self._start_next_download)

    def _run_post_processing(
        self,
        nzb_path: Path,
        early_par2_result: Optional[Tuple[bool, bool, str]] = None,
        download_size: int = 0,
        download_duration: float = 0.0,
        direct_to_destination: bool = False,
        actual_download_dir: Optional[Path] = None,
        skip_extraction: bool = False,
        streaming_extract_path: Optional[Path] = None
    ) -> None:
        """
        Run PAR2 verification and extraction.

        Args:
            nzb_path: Path to NZB file
            early_par2_result: Optional (verified, repaired, message) from streaming PAR2
            download_size: Total bytes downloaded
            download_duration: Download time in seconds
            direct_to_destination: If True, files were downloaded directly to final location
            actual_download_dir: Actual directory where files were downloaded
            skip_extraction: If True, skip archive extraction (streaming already did it)
            streaming_extract_path: Path where streaming extractor put files
        """
        logger.info(f"=== _run_post_processing called for {nzb_path.name} ===")
        try:
            from ..core.post_processor import PostProcessor
            logger.info("PostProcessor imported successfully")

            self._queue_info_var.set("Post-processing...")
            logger.info(f"Starting post-processing for {nzb_path.name}")

            # Get download directory (use actual if provided, else config)
            download_dir = actual_download_dir or Path(self._config.download.output_dir)
            extract_dir = Path(self._config.postprocess.extract_dir)

            if direct_to_destination:
                logger.info(f"DIRECT-TO-DESTINATION mode: files already at {download_dir}")

            # Smooth progress animation state
            pp_current_progress = [0.0]  # Use list for mutable closure
            pp_target_progress = [0.0]
            pp_animation_id = [None]

            def animate_progress():
                """Smoothly animate progress bar toward target value."""
                if not self._state.is_post_processing:
                    return

                current = pp_current_progress[0]
                target = pp_target_progress[0]

                if abs(target - current) < 0.1:
                    # Close enough, snap to target
                    pp_current_progress[0] = target
                else:
                    # Move 15% of remaining distance (smooth easing)
                    pp_current_progress[0] = current + (target - current) * 0.15

                # Update UI
                display_val = pp_current_progress[0]
                self._progress_var.set(display_val)
                self._percent_var.set(f"{display_val:5.1f}%")

                # Continue animation
                if self._state.is_post_processing:
                    pp_animation_id[0] = self._master.after(30, animate_progress)

            def progress_callback(msg: str, percent: float):
                """Handle progress updates from post-processor."""
                def update_ui():
                    self._queue_info_var.set(f"PP: {msg}")
                    if percent >= 0:
                        pp_target_progress[0] = percent
                        self._state.progress_percent = percent
                        # Start animation if not running
                        if pp_animation_id[0] is None:
                            animate_progress()
                self._master.after(0, update_ui)
                logger.debug(f"PP Progress: {msg} ({percent}%)")

            # Enter post-processing mode - prevents _update_ui from overwriting progress
            self._state.is_post_processing = True

            # Reset progress bar for post-processing (on main thread)
            def init_pp_ui():
                pp_current_progress[0] = 0.0
                pp_target_progress[0] = 0.0
                self._state.progress_percent = 0
                self._progress_var.set(0)
                self._percent_var.set("  0.0%")
                self._queue_info_var.set("PP: Starting...")
            self._master.after(0, init_pp_ui)

            logger.info(f"Download dir: {download_dir}")
            logger.info(f"Extract dir: {extract_dir}")
            logger.info(f"PAR2 path: {self._config.postprocess.par2_path or 'auto-detect'}")
            logger.info(f"7z path: {self._config.postprocess.sevenzip_path or 'auto-detect'}")
            if early_par2_result:
                logger.info(f"Early PAR2 result: {early_par2_result}")

            # For direct-to-destination, download_dir and extract_dir are the same
            effective_extract_dir = download_dir if direct_to_destination else extract_dir

            processor = PostProcessor(
                download_dir=download_dir,
                extract_dir=effective_extract_dir,
                par2_path=self._config.postprocess.par2_path or None,
                sevenzip_path=self._config.postprocess.sevenzip_path or None,
                cleanup_after_extract=self._config.postprocess.cleanup_after_extract,
                on_progress=progress_callback
            )

            logger.info(f"PostProcessor created. PAR2={processor.par2_path}, 7z={processor.sevenzip_path}")
            logger.info(f"Calling processor.process(direct_to_destination={direct_to_destination}, skip_extraction={skip_extraction})...")

            # Pass early PAR2 result to skip re-verification if already done
            result = processor.process(
                nzb_path,
                early_par2_result=early_par2_result,
                direct_to_destination=direct_to_destination,
                skip_extraction=skip_extraction
            )

            # If streaming extraction handled files, update result path
            if skip_extraction and streaming_extract_path and streaming_extract_path.exists():
                result.extract_path = streaming_extract_path
                # Count extracted files
                result.files_extracted = sum(1 for _ in streaming_extract_path.rglob('*') if _.is_file())

            logger.info(f"processor.process() returned: success={result.success}, msg={result.message}")

            if result.success:
                # Build concise status message
                msg = f"Done! {result.files_extracted} files"
                if result.par2_repaired:
                    msg += " (repaired)"
                self._queue_info_var.set(msg)
                logger.info(f"Post-processing completed: {result.message}")

                # Log success with details
                log_msg = f"Completed: {result.files_extracted} file(s)"
                if result.par2_repaired:
                    log_msg += " (PAR2 repaired)"
                if result.extract_path:
                    log_msg += f" -> {result.extract_path.name}"
                self._log(log_msg, "success")

                # Show single completion summary with all metrics (if enabled)
                if self._config and self._config.ui.show_completion_summary:
                    self._show_completion_summary(result, download_size, download_duration)
            else:
                self._queue_info_var.set(f"PP Error: {result.message[:30]}")
                logger.error(f"Post-processing failed: {result.message}")
                self._log(f"Post-processing failed: {result.message}", "error")
                # Show error details
                self._master.after(100, lambda: self._show_warning(
                    "Post-Processing Failed",
                    f"{result.message}\n\nCheck the logs for more details."
                ))

        except ImportError as e:
            logger.warning(f"Post-processor not available: {e}")
            self._queue_info_var.set("Complete (no PP)")
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
            self._queue_info_var.set(f"PP Error: {str(e)[:30]}")
            self._log(f"Post-processing error: {e}", "error")
        finally:
            # Exit post-processing mode
            self._state.is_post_processing = False

    def _init_engine(self) -> None:
        """Initialize TurboEngineV2."""
        from ..core.turbo_engine_v2 import TurboEngineV2
        from ..core.fast_nntp import ServerConfig

        if not self._config:
            return

        try:
            self._queue_info_var.set(f"Connecting...")

            server_config = ServerConfig(
                host=self._config.server.host,
                port=self._config.server.port,
                username=self._config.server.username,
                password=self._config.server.password,
                use_ssl=self._config.server.use_ssl,
                connections=self._config.server.connections,
                timeout=self._config.server.timeout
            )

            output_dir = Path(self._config.download.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self._turbo_engine = TurboEngineV2(
                server_config=server_config,
                output_dir=output_dir,
                download_threads=self._config.server.connections,
                decoder_threads=self._config.turbo.decoder_threads or 0,
                writer_threads=self._config.turbo.writer_threads or 8,
                pipeline_depth=self._config.turbo.pipeline_depth or 20,
                write_through=self._config.turbo.write_through,
                on_progress=self._on_progress,
                raw_queue_size=self._config.turbo.raw_queue_size,
                write_queue_size=self._config.turbo.write_queue_size
            )

            conn_count = self._turbo_engine.connect()
            self._state.connections_active = conn_count
            self._state.connections_total = conn_count  # Use actual connected count
            self._queue_info_var.set(f"Connected: {conn_count} connections")
            self._log(f"Connected to {self._config.server.host} ({conn_count} connections)", "success")

        except Exception as e:
            logger.error(f"Init failed: {e}")
            self._turbo_engine = None
            self._queue_info_var.set(f"Connection failed")
            self._log(f"Connection failed: {e}", "error")

    def _parse_nzb_size(self, nzb_path: Path) -> None:
        """Parse NZB size."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(nzb_path)
            root = tree.getroot()
            ns = {'nzb': 'http://www.newzbin.com/DTD/2003/nzb'}

            total_bytes = 0
            segment_count = 0
            for seg in root.findall('.//nzb:segment', ns):
                total_bytes += int(seg.get('bytes', 0))
                segment_count += 1

            self._state.total_bytes = total_bytes
            self._state.segments_total = segment_count
            self._state.nzb_segments_total = segment_count  # Store separately, never overwritten
            logger.info(f"NZB parsed: {segment_count} segments, {total_bytes} bytes")
        except Exception as e:
            logger.error(f"Error parsing NZB: {e}")
            self._state.total_bytes = 0
            self._state.segments_total = 0
            self._state.nzb_segments_total = 0

    def _on_progress(self, stats) -> None:
        """Progress callback."""
        # Don't update if download marked as complete
        if self._download_complete:
            return

        # Don't update speed if paused
        if hasattr(self, '_is_paused') and self._is_paused:
            return

        if hasattr(self, '_turbo_engine') and self._turbo_engine:
            self._state.speed_mbps = self._turbo_engine._speed_tracker.speed_mbps
            self._state.queue_raw = self._turbo_engine._raw_queue.qsize()
            self._state.queue_write = self._turbo_engine._write_queue.qsize()
            # Get active connections from engine
            try:
                self._state.connections_active = len([c for c in self._turbo_engine._connections if c and c.socket])
            except:
                pass  # Keep previous value if error

        self._state.downloaded_bytes = stats.bytes_written
        self._state.segments_done = stats.segments_written

        # Use NZB-parsed segment total (engine's value is unreliable)
        # Only use engine value if we don't have NZB value
        if self._state.nzb_segments_total > 0:
            self._state.segments_total = self._state.nzb_segments_total
        elif stats.segments_total > self._state.segments_total:
            self._state.segments_total = stats.segments_total

        # Calculate progress based on segments
        if self._state.segments_total > 0:
            self._state.progress_percent = min(100.0, (self._state.segments_done / self._state.segments_total) * 100)
        elif self._state.total_bytes > 0:
            self._state.progress_percent = min(100.0, (self._state.downloaded_bytes / self._state.total_bytes) * 100)

        if self._state.speed_mbps > 0 and self._state.total_bytes > 0:
            remaining = self._state.total_bytes - self._state.downloaded_bytes
            self._state.eta_seconds = remaining / (self._state.speed_mbps * 1024 * 1024)
        else:
            self._state.eta_seconds = 0

    def _toggle_pause(self) -> None:
        """Toggle pause/resume."""
        if not self._turbo_engine:
            return

        if hasattr(self, '_is_paused') and self._is_paused:
            # Resume
            if hasattr(self._turbo_engine, 'resume'):
                self._turbo_engine.resume()
            self._is_paused = False
            self._pause_btn.config(text="Pause")
            self._queue_info_var.set("Resumed")
        else:
            # Pause
            if hasattr(self._turbo_engine, 'pause'):
                self._turbo_engine.pause()
            self._is_paused = True
            self._state.speed_mbps = 0  # Reset speed display
            # Update graph to show zero speed when paused
            if self._speed_graph:
                self._speed_graph.update_speed(0.0)
            self._pause_btn.config(text="Resume")
            self._queue_info_var.set("Paused")

    def _stop_download(self) -> None:
        """Stop download."""
        if self._turbo_engine:
            if hasattr(self._turbo_engine, 'stop'):
                self._turbo_engine.stop()
            # Disconnect and reset engine
            if hasattr(self._turbo_engine, 'disconnect'):
                self._turbo_engine.disconnect()
            self._turbo_engine = None

        # Reset state
        self._state.is_downloading = False
        self._state.speed_mbps = 0
        self._state.progress_percent = 0
        self._state.eta_seconds = 0
        self._state.current_file = ""
        self._is_paused = False

        # Update graph to show zero speed
        if self._speed_graph:
            self._speed_graph.update_speed(0.0)

        self._download_queue.clear()
        self._log("Download stopped by user", "warning")
        self._stop_btn.config(state="disabled")
        self._pause_btn.config(state="disabled", text="Pause")
        self._queue_info_var.set("Stopped")

    def _show_warning(self, title: str, message: str) -> None:
        """Show a warning dialog to the user."""
        messagebox.showwarning(title, message)

    def _show_post_process_warnings(self, result) -> None:
        """Show post-processing warnings to user."""
        from ..core.post_processor import WarningType

        warnings_text = []
        for warning_type, warning_msg in result.warnings:
            if warning_type == WarningType.ANTIVIRUS_BLOCK:
                warnings_text.append(f"ANTIVIRUS WARNING:\n{warning_msg}")
            elif warning_type == WarningType.PAR2_REPAIR_NEEDED:
                warnings_text.append(f"REPAIR INFO:\n{warning_msg}")
            else:
                warnings_text.append(warning_msg)

        if warnings_text:
            full_message = "\n\n".join(warnings_text)
            self._master.after(100, lambda: messagebox.showwarning(
                "Post-Processing Warnings",
                full_message
            ))

    def _show_completion_summary(
        self,
        result,
        download_size: int,
        download_duration: float
    ) -> None:
        """Show detailed completion summary with metrics."""

        def format_size(bytes_val: int) -> str:
            if bytes_val >= 1024**3:
                return f"{bytes_val / (1024**3):.2f} GB"
            elif bytes_val >= 1024**2:
                return f"{bytes_val / (1024**2):.1f} MB"
            return f"{bytes_val / 1024:.0f} KB"

        def format_duration(seconds: float) -> str:
            if seconds >= 3600:
                h, m = divmod(int(seconds), 3600)
                m, s = divmod(m, 60)
                return f"{h}h {m}m {s}s"
            elif seconds >= 60:
                m, s = divmod(int(seconds), 60)
                return f"{m}m {s}s"
            return f"{seconds:.1f}s"

        lines = []

        # Title based on success/repair status
        if result.par2_repaired:
            title = "Download Complete (Repaired)"
        else:
            title = "Download Complete"

        # Files extracted
        lines.append(f"Files extracted: {result.files_extracted}")

        # Destination
        if result.extract_path:
            lines.append(f"Destination: {result.extract_path}")

        lines.append("")  # Separator

        # === DOWNLOAD METRICS ===
        if download_size > 0 and download_duration > 0:
            dl_speed = download_size / (download_duration * 1024 * 1024)
            lines.append(f"Download: {format_size(download_size)} in {format_duration(download_duration)}")
            lines.append(f"Network speed: {dl_speed:.1f} MB/s avg")

        # === EXTRACTION METRICS ===
        if result.extraction_duration_seconds > 0 and result.extracted_bytes > 0:
            lines.append("")
            lines.append(f"Extraction: {format_size(result.extracted_bytes)} in {format_duration(result.extraction_duration_seconds)}")
            lines.append(f"Disk throughput: {result.extraction_speed_mbs:.1f} MB/s")

        # === REPAIR INFO (only if repair actually happened) ===
        if result.par2_repaired:
            lines.append("")
            lines.append("PAR2 repair was required and successful.")

        # === TOTAL TIME ===
        if result.duration_seconds > 0:
            lines.append("")
            lines.append(f"Total post-processing: {format_duration(result.duration_seconds)}")

        # === WARNINGS (only critical ones) ===
        if result.warnings:
            from ..core.post_processor import WarningType
            critical_warnings = []
            for warning_type, warning_msg in result.warnings:
                if warning_type == WarningType.ANTIVIRUS_BLOCK:
                    critical_warnings.append(f"Antivirus: {warning_msg}")
                elif warning_type == WarningType.CLEANUP_SKIPPED:
                    critical_warnings.append(f"Cleanup: {warning_msg}")

            if critical_warnings:
                lines.append("")
                lines.append("ATTENTION:")
                lines.extend(critical_warnings)

        message = "\n".join(lines)

        self._master.after(100, lambda: messagebox.showinfo(title, message))

    def _show_ram_completion_summary(
        self,
        ram_pp,
        download_size: int,
        download_duration: float
    ) -> None:
        """Show beautiful completion dialog for RAM mode processing."""
        self._master.after(100, lambda: self._create_completion_dialog(
            ram_pp, download_size, download_duration
        ))

    def _create_completion_dialog(
        self,
        ram_pp,
        download_size: int,
        download_duration: float
    ) -> None:
        """Create themed completion dialog showcasing RAM mode performance."""
        # Theme colors (matching park dark theme)
        BG_DARK = "#1e1e1e"
        BG_PANEL = "#2b2b2b"
        BG_ACCENT = "#333333"
        FG_TEXT = "#e0e0e0"
        FG_DIM = "#808080"
        FG_ACCENT = "#2d9254"  # Green accent
        FG_GOLD = "#d4a62a"    # Gold for highlights
        FG_CYAN = "#4fc3f7"    # Cyan for speed

        # Fonts
        FONT_HEADER = ("Segoe UI", 16, "bold")
        FONT_BIG = ("Consolas", 32, "bold")
        FONT_MEDIUM = ("Consolas", 18, "bold")
        FONT_LABEL = ("Segoe UI", 10)
        FONT_VALUE = ("Consolas", 11, "bold")
        FONT_SMALL = ("Segoe UI", 9)

        # Create dialog
        dialog = tk.Toplevel(self._master)
        dialog.title("RAM Mode Complete")
        dialog.configure(bg=BG_DARK)
        dialog.resizable(False, False)
        dialog.transient(self._master)
        dialog.grab_set()

        # Set initial size and allow auto-sizing
        dialog.geometry("480x620")
        dialog.update_idletasks()

        # Center on parent
        x = self._master.winfo_x() + (self._master.winfo_width() - 480) // 2
        y = self._master.winfo_y() + (self._master.winfo_height() - 620) // 2
        dialog.geometry(f"+{x}+{y}")

        # Main container
        main = tk.Frame(dialog, bg=BG_DARK, padx=20, pady=15)
        main.pack(fill="both", expand=True)

        # === HEADER ===
        header = tk.Frame(main, bg=BG_DARK)
        header.pack(fill="x", pady=(0, 15))

        tk.Label(
            header, text="RAM MODE", font=FONT_HEADER,
            fg=FG_ACCENT, bg=BG_DARK
        ).pack(side="left")

        tk.Label(
            header, text="COMPLETE", font=FONT_HEADER,
            fg=FG_TEXT, bg=BG_DARK
        ).pack(side="left", padx=(8, 0))

        # === SPEED SHOWCASE (Big numbers) ===
        speed_frame = tk.Frame(main, bg=BG_PANEL, padx=15, pady=12)
        speed_frame.pack(fill="x", pady=(0, 12))

        # Download speed
        dl_speed = download_size / (download_duration * 1024 * 1024) if download_duration > 0 else 0

        speed_row = tk.Frame(speed_frame, bg=BG_PANEL)
        speed_row.pack(fill="x")

        # Network speed (left)
        net_box = tk.Frame(speed_row, bg=BG_ACCENT, padx=12, pady=8)
        net_box.pack(side="left", fill="both", expand=True, padx=(0, 6))

        tk.Label(
            net_box, text="NETWORK", font=FONT_SMALL,
            fg=FG_DIM, bg=BG_ACCENT
        ).pack(anchor="w")

        net_val_frame = tk.Frame(net_box, bg=BG_ACCENT)
        net_val_frame.pack(anchor="w")
        tk.Label(
            net_val_frame, text=f"{dl_speed:.0f}", font=FONT_BIG,
            fg=FG_CYAN, bg=BG_ACCENT
        ).pack(side="left")
        tk.Label(
            net_val_frame, text=" MB/s", font=FONT_LABEL,
            fg=FG_DIM, bg=BG_ACCENT
        ).pack(side="left", anchor="s", pady=(0, 8))

        # RAM→Disk speed (right)
        ram_box = tk.Frame(speed_row, bg=BG_ACCENT, padx=12, pady=8)
        ram_box.pack(side="left", fill="both", expand=True, padx=(6, 0))

        tk.Label(
            ram_box, text="RAM\u2192DISK", font=FONT_SMALL,
            fg=FG_DIM, bg=BG_ACCENT
        ).pack(anchor="w")

        ram_val_frame = tk.Frame(ram_box, bg=BG_ACCENT)
        ram_val_frame.pack(anchor="w")

        # Calculate RAM→Disk speed based on ACTUAL disk writes
        # This includes both extraction (direct to disk) and flush phases
        total_disk_duration = ram_pp.stats_extract_duration + ram_pp.stats_flush_duration
        total_disk_bytes = ram_pp.stats_flush_bytes

        # For extraction with archives, use extracted size; for direct media, use flush size
        if ram_pp.stats_files_extracted > 0 and ram_pp.stats_extract_duration > 0.1:
            # Archives were extracted - use processing duration with download size
            # (extraction writes decompressed data directly to disk)
            ram_speed = download_size / (1024 * 1024) / total_disk_duration if total_disk_duration > 0 else 0
        elif ram_pp.stats_flush_duration > 0 and total_disk_bytes > 0:
            # No extraction (DIRECT_MEDIA) - use flush stats
            ram_speed = ram_pp.stats_flush_speed_mbs
        else:
            ram_speed = 0

        tk.Label(
            ram_val_frame, text=f"{ram_speed:.0f}", font=FONT_BIG,
            fg=FG_GOLD, bg=BG_ACCENT
        ).pack(side="left")
        tk.Label(
            ram_val_frame, text=" MB/s", font=FONT_LABEL,
            fg=FG_DIM, bg=BG_ACCENT
        ).pack(side="left", anchor="s", pady=(0, 8))

        # === RESULTS PANEL ===
        results = tk.Frame(main, bg=BG_PANEL, padx=15, pady=12)
        results.pack(fill="x", pady=(0, 12))

        def add_result_row(parent, label: str, value: str, highlight: bool = False):
            row = tk.Frame(parent, bg=BG_PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(
                row, text=label, font=FONT_LABEL,
                fg=FG_DIM, bg=BG_PANEL, width=18, anchor="w"
            ).pack(side="left")
            tk.Label(
                row, text=value, font=FONT_VALUE,
                fg=FG_ACCENT if highlight else FG_TEXT, bg=BG_PANEL
            ).pack(side="left")

        # Files extracted (THE MAIN RESULT!)
        extracted = ram_pp.stats_files_extracted
        if extracted > 0:
            add_result_row(results, "Files extracted:", f"{extracted:,}", highlight=True)

        # Download size
        if download_size > 0:
            size_str = f"{download_size / (1024**3):.2f} GB" if download_size >= 1024**3 else f"{download_size / (1024**2):.1f} MB"
            add_result_row(results, "Downloaded:", size_str)

        # Verification
        verified = ram_pp.stats_files_verified
        if verified > 0:
            verify_status = f"{verified} files OK"
            if ram_pp.stats_files_damaged > 0:
                verify_status += f" ({ram_pp.stats_files_repaired} repaired)"
            add_result_row(results, "Verification:", verify_status)

        # === TIMING PANEL ===
        timing = tk.Frame(main, bg=BG_PANEL, padx=15, pady=12)
        timing.pack(fill="x", pady=(0, 12))

        def format_time(seconds: float) -> str:
            if seconds >= 3600:
                h, m = divmod(int(seconds), 3600)
                m, s = divmod(m, 60)
                return f"{h}h {m}m {s}s"
            elif seconds >= 60:
                m, s = divmod(int(seconds), 60)
                return f"{m}m {s}s"
            return f"{seconds:.1f}s"

        def add_timing_row(parent, label: str, value: str, bold: bool = False):
            row = tk.Frame(parent, bg=BG_PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(
                row, text=label, font=FONT_LABEL,
                fg=FG_DIM, bg=BG_PANEL, width=18, anchor="w"
            ).pack(side="left")
            tk.Label(
                row, text=value,
                font=("Consolas", 11, "bold") if bold else FONT_VALUE,
                fg=FG_GOLD if bold else FG_TEXT, bg=BG_PANEL
            ).pack(side="left")

        add_timing_row(timing, "Download time:", format_time(download_duration))
        if ram_pp.stats_total_duration > 0:
            add_timing_row(timing, "Processing time:", format_time(ram_pp.stats_total_duration))

        # Separator
        sep = tk.Frame(timing, bg=BG_ACCENT, height=1)
        sep.pack(fill="x", pady=8)

        # TOTAL TIME (highlighted)
        total_time = download_duration + ram_pp.stats_total_duration
        add_timing_row(timing, "TOTAL TIME:", format_time(total_time), bold=True)

        # === DESTINATION ===
        if ram_pp.stats_extract_path:
            dest_frame = tk.Frame(main, bg=BG_DARK)
            dest_frame.pack(fill="x", pady=(0, 15))

            tk.Label(
                dest_frame, text="Destination:", font=FONT_SMALL,
                fg=FG_DIM, bg=BG_DARK
            ).pack(anchor="w")

            # Truncate path if too long
            path_str = str(ram_pp.stats_extract_path)
            if len(path_str) > 55:
                path_str = "..." + path_str[-52:]

            tk.Label(
                dest_frame, text=path_str, font=("Consolas", 9),
                fg=FG_TEXT, bg=BG_DARK
            ).pack(anchor="w")

        # === OK BUTTON ===
        btn_frame = tk.Frame(main, bg=BG_DARK)
        btn_frame.pack(fill="x", pady=(10, 5))

        ok_btn = tk.Button(
            btn_frame, text="  OK  ", font=("Segoe UI", 11, "bold"),
            padx=30, pady=8,
            bg=FG_ACCENT, fg="white",
            activebackground="#3daf6a", activeforeground="white",
            relief="raised", bd=2, cursor="hand2",
            command=dialog.destroy
        )
        ok_btn.pack()

        # Bind Enter key to close
        dialog.bind("<Return>", lambda e: dialog.destroy())
        dialog.bind("<Escape>", lambda e: dialog.destroy())

        # Focus dialog
        dialog.focus_set()

    def _force_100_percent(self) -> None:
        """Force progress to 100% on main thread."""
        self._state.progress_percent = 100.0
        self._progress_var.set(100.0)
        self._percent_var.set("100.0%")

    def _update_pp_progress(self, percent: float) -> None:
        """Update progress bar during post-processing."""
        self._state.progress_percent = percent
        self._progress_var.set(percent)
        self._percent_var.set(f"{percent:5.1f}%")

    def _schedule_update(self) -> None:
        """Schedule UI update."""
        self._update_ui()
        self._update_job = self._master.after(100, self._schedule_update)

    def _update_ui(self) -> None:
        """Update UI with fixed-width values."""
        # Speed (adaptive units)
        speed_val, speed_unit = format_speed(self._state.speed_mbps)
        self._speed_var.set(speed_val)
        self._speed_unit_var.set(speed_unit)

        # Update speed graph (only during actual download, not post-processing)
        if self._speed_graph and self._state.is_downloading and not self._state.is_post_processing:
            self._speed_graph.update_speed(self._state.speed_mbps)

        # Progress (skip update during post-processing to let PP control it)
        if not self._state.is_post_processing:
            self._progress_var.set(self._state.progress_percent)
            self._percent_var.set(f"{self._state.progress_percent:5.1f}%")

        # File name (truncate to fixed width)
        if self._state.is_downloading and self._state.current_file:
            name = self._state.current_file
            if len(name) > 45:
                name = name[:42] + "..."
            self._file_var.set(name)
        else:
            self._file_var.set("No file selected")

        # ETA (fixed width)
        if hasattr(self, '_is_paused') and self._is_paused:
            self._eta_var.set(" Paused")
        elif self._state.is_downloading and self._state.eta_seconds > 0:
            eta = self._state.eta_seconds
            if eta < 60:
                self._eta_var.set(f"{int(eta):3d}s")
            elif eta < 3600:
                self._eta_var.set(f"{int(eta // 60):2d}m {int(eta % 60):02d}s")
            else:
                self._eta_var.set(f"{int(eta // 3600)}h {int((eta % 3600) // 60):02d}m")
        else:
            self._eta_var.set("  --:--")

        # Size (adaptive units)
        dl_str = format_size(self._state.downloaded_bytes)
        total_str = format_size(self._state.total_bytes) if self._state.total_bytes > 0 else "---"
        self._size_var.set(f"{dl_str} / {total_str}")

        # Statistics
        self._segments_var.set(f"{self._state.segments_done} / {self._state.segments_total}")
        self._conn_var.set(f"{self._state.connections_active} / {self._state.connections_total}")
        self._queues_var.set(f"{self._state.queue_raw} / {self._state.queue_write}")

        # Total downloaded
        if self._state.is_downloading:
            self._total_var.set(f"Downloaded: {format_size(self._state.downloaded_bytes)}")
        else:
            self._total_var.set("")

        # Update system tray tooltip
        self._update_tray_tooltip()

    def run(self) -> None:
        """Start application."""
        # Start minimized to tray if requested
        if self._start_minimized and self._system_tray and self._system_tray.is_running:
            self._master.withdraw()
            logger.info("Started minimized to system tray")

        if HAS_TKMT:
            self._root.run()
        else:
            self._root.mainloop()

    # =========================================================================
    # System Tray Methods
    # =========================================================================

    def _setup_system_tray(self) -> None:
        """Initialize system tray icon."""
        if not HAS_SYSTEM_TRAY:
            logger.debug("System tray not available (pystray not installed)")
            return

        try:
            logo_path = self._get_logo_path()
            self._system_tray = SystemTrayManager(
                icon_path=logo_path,
                on_show=self._show_window,
                on_quit=self._quit_app,
                on_pause=self._toggle_pause,
                on_resume=self._toggle_pause,
            )
            if self._system_tray.start():
                logger.info("System tray initialized")
            else:
                logger.warning("Failed to start system tray")
        except Exception as e:
            logger.error(f"Error setting up system tray: {e}")
            self._system_tray = None

    def _on_window_close(self) -> None:
        """Handle window close button (X) click."""
        # Check if we should minimize to tray instead of closing
        if (self._config and self._config.ui.minimize_to_tray and
                self._system_tray and self._system_tray.is_running):
            self._minimize_to_tray()
        else:
            self._quit_app()

    def _minimize_to_tray(self) -> None:
        """Minimize window to system tray."""
        self._master.withdraw()
        logger.debug("Window minimized to tray")

    def _show_window(self) -> None:
        """Show/restore window from system tray."""
        self._master.deiconify()
        self._master.lift()
        self._master.focus_force()
        logger.debug("Window restored from tray")

    def _quit_app(self) -> None:
        """Quit application completely."""
        logger.info("Application shutting down...")

        # Stop system tray
        if self._system_tray:
            self._system_tray.stop()

        # Cancel scheduled updates
        if self._update_job:
            self._master.after_cancel(self._update_job)

        # Stop any running download
        if self._turbo_engine:
            try:
                self._turbo_engine.cancel()
            except Exception:
                pass

        # Destroy window
        try:
            self._master.quit()
            self._master.destroy()
        except Exception:
            pass

    def _update_tray_tooltip(self) -> None:
        """Update system tray tooltip with current status."""
        if not self._system_tray or not self._system_tray.is_running:
            return

        if self._state.is_downloading:
            speed_val, speed_unit = format_speed(self._state.speed_mbps)
            tooltip = f"DLER - {speed_val} {speed_unit} ({self._state.progress_percent:.0f}%)"
        elif self._state.is_post_processing:
            tooltip = f"DLER - Post-processing..."
        else:
            tooltip = "DLER - Ready"

        self._system_tray.update_tooltip(tooltip)


class SettingsDialog:
    """Settings dialog with tabs."""

    def __init__(self, parent, config, on_save: Callable):
        self._config = config
        self._on_save = on_save

        self._dialog = tk.Toplevel(parent)
        self._dialog.title("Settings")
        self._dialog.geometry("520x680")
        self._dialog.resizable(False, False)
        self._dialog.transient(parent)
        self._dialog.grab_set()

        self._create_ui()

        self._dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 520) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 680) // 2
        self._dialog.geometry(f"+{x}+{y}")

    def _create_ui(self) -> None:
        """Create settings UI."""
        notebook = ttk.Notebook(self._dialog)
        notebook.pack(fill="both", expand=True, padx=15, pady=15)

        # Server tab
        server_frame = ttk.Frame(notebook, padding=20)
        notebook.add(server_frame, text="Server")
        self._create_server_tab(server_frame)

        # Download tab
        download_frame = ttk.Frame(notebook, padding=20)
        notebook.add(download_frame, text="Download")
        self._create_download_tab(download_frame)

        # Performance tab
        perf_frame = ttk.Frame(notebook, padding=20)
        notebook.add(perf_frame, text="Perf")
        self._create_performance_tab(perf_frame)

        # Post-Processing tab
        pp_frame = ttk.Frame(notebook, padding=20)
        notebook.add(pp_frame, text="Post-Process")
        self._create_postprocess_tab(pp_frame)

        # RAM/GPU tab
        ram_frame = ttk.Frame(notebook, padding=20)
        notebook.add(ram_frame, text="RAM/GPU")
        self._create_ram_gpu_tab(ram_frame)

        # System tab (tray, auto-start)
        system_frame = ttk.Frame(notebook, padding=20)
        notebook.add(system_frame, text="System")
        self._create_system_tab(system_frame)

        # Buttons
        btn_frame = ttk.Frame(self._dialog)
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        ttk.Button(btn_frame, text="Save", command=self._save, width=12).pack(side="right", padx=(8, 0))
        ttk.Button(btn_frame, text="Cancel", command=self._dialog.destroy, width=12).pack(side="right")

    def _create_server_tab(self, parent) -> None:
        """Server settings."""
        parent.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(parent, text="Host:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._host_var = tk.StringVar(value=self._config.server.host if self._config else "")
        ttk.Entry(parent, textvariable=self._host_var, font=FONT_VALUE_SMALL).grid(row=row, column=1, sticky="ew", pady=6)
        row += 1

        ttk.Label(parent, text="Port:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._port_var = tk.IntVar(value=self._config.server.port if self._config else 563)
        ttk.Entry(parent, textvariable=self._port_var, font=FONT_VALUE_SMALL).grid(row=row, column=1, sticky="ew", pady=6)
        row += 1

        self._ssl_var = tk.BooleanVar(value=self._config.server.use_ssl if self._config else True)
        ttk.Checkbutton(parent, text="Use SSL/TLS", variable=self._ssl_var).grid(row=row, column=1, sticky="w", pady=6)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
        row += 1

        ttk.Label(parent, text="Username:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._user_var = tk.StringVar(value=self._config.server.username if self._config else "")
        ttk.Entry(parent, textvariable=self._user_var, font=FONT_VALUE_SMALL).grid(row=row, column=1, sticky="ew", pady=6)
        row += 1

        ttk.Label(parent, text="Password:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._pass_var = tk.StringVar(value=self._config.server.password if self._config else "")
        ttk.Entry(parent, textvariable=self._pass_var, font=FONT_VALUE_SMALL, show="*").grid(row=row, column=1, sticky="ew", pady=6)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
        row += 1

        ttk.Label(parent, text="Connections:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._conn_var = tk.IntVar(value=self._config.server.connections if self._config else 30)
        conn_frame = ttk.Frame(parent)
        conn_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(conn_frame, textvariable=self._conn_var, from_=1, to=100, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(conn_frame, text="(1-100)", font=FONT_LABEL).pack(side="left", padx=(10, 0))
        row += 1

        ttk.Label(parent, text="Timeout:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._timeout_var = tk.IntVar(value=self._config.server.timeout if self._config else 30)
        timeout_frame = ttk.Frame(parent)
        timeout_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(timeout_frame, textvariable=self._timeout_var, from_=5, to=120, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(timeout_frame, text="seconds", font=FONT_LABEL).pack(side="left", padx=(10, 0))

    def _create_download_tab(self, parent) -> None:
        """Download settings."""
        parent.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(parent, text="Output Directory:", font=FONT_LABEL).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1

        dir_frame = ttk.Frame(parent)
        dir_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        dir_frame.columnconfigure(0, weight=1)

        self._dir_var = tk.StringVar(value=self._config.download.output_dir if self._config else str(Path.home() / "Downloads"))
        ttk.Entry(dir_frame, textvariable=self._dir_var, font=FONT_VALUE_SMALL).grid(row=0, column=0, sticky="ew")
        ttk.Button(dir_frame, text="Browse...", command=self._browse_dir, width=10).grid(row=0, column=1, padx=(8, 0))
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
        row += 1

        ttk.Label(parent, text="Max Retries:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._retries_var = tk.IntVar(value=self._config.download.max_retries if self._config else 3)
        retry_frame = ttk.Frame(parent)
        retry_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(retry_frame, textvariable=self._retries_var, from_=0, to=10, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(retry_frame, text="per segment", font=FONT_LABEL).pack(side="left", padx=(10, 0))

    def _create_performance_tab(self, parent) -> None:
        """Performance settings."""
        parent.columnconfigure(1, weight=1)

        row = 0
        info = ttk.Label(parent, text="TurboEngine advanced settings.\n0 = auto-detection.", font=FONT_LABEL)
        info.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 15))
        row += 1

        ttk.Label(parent, text="Decoder Threads:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._decoder_var = tk.IntVar(value=self._config.turbo.decoder_threads if self._config else 0)
        dec_frame = ttk.Frame(parent)
        dec_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(dec_frame, textvariable=self._decoder_var, from_=0, to=32, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(dec_frame, text="(0 = auto)", font=FONT_LABEL).pack(side="left", padx=(10, 0))
        row += 1

        ttk.Label(parent, text="Writer Threads:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._writer_var = tk.IntVar(value=self._config.turbo.writer_threads if self._config else 8)
        write_frame = ttk.Frame(parent)
        write_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(write_frame, textvariable=self._writer_var, from_=1, to=32, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(write_frame, text="(recommended: 8)", font=FONT_LABEL).pack(side="left", padx=(10, 0))
        row += 1

        ttk.Label(parent, text="Pipeline Depth:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._pipeline_var = tk.IntVar(value=self._config.turbo.pipeline_depth if self._config else 20)
        pipe_frame = ttk.Frame(parent)
        pipe_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(pipe_frame, textvariable=self._pipeline_var, from_=5, to=50, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(pipe_frame, text="(per connection)", font=FONT_LABEL).pack(side="left", padx=(10, 0))
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
        row += 1

        self._writethrough_var = tk.BooleanVar(value=self._config.turbo.write_through if self._config else False)
        ttk.Checkbutton(parent, text="Write-through (bypass OS cache)", variable=self._writethrough_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)

    def _create_postprocess_tab(self, parent) -> None:
        """Post-processing settings (PAR2 + extraction)."""
        parent.columnconfigure(1, weight=1)

        row = 0
        # Enable post-processing
        self._pp_enabled_var = tk.BooleanVar(
            value=self._config.postprocess.enabled if self._config else True
        )
        ttk.Checkbutton(
            parent,
            text="Enable post-processing (PAR2 verify + extract)",
            variable=self._pp_enabled_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # Extract directory
        ttk.Label(parent, text="Extract Directory:", font=FONT_LABEL).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 2)
        )
        row += 1

        extract_frame = ttk.Frame(parent)
        extract_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        extract_frame.columnconfigure(0, weight=1)

        self._extract_dir_var = tk.StringVar(
            value=self._config.postprocess.extract_dir if self._config else ""
        )
        ttk.Entry(extract_frame, textvariable=self._extract_dir_var, font=FONT_VALUE_SMALL).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(
            extract_frame, text="Browse...", command=self._browse_extract_dir, width=10
        ).grid(row=0, column=1, padx=(8, 0))
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # PAR2 options
        ttk.Label(parent, text="PAR2 Verification:", font=FONT_SECTION).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 4)
        )
        row += 1

        self._par2_verify_var = tk.BooleanVar(
            value=self._config.postprocess.par2_verify if self._config else True
        )
        ttk.Checkbutton(
            parent, text="Verify files with PAR2", variable=self._par2_verify_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)
        row += 1

        self._par2_repair_var = tk.BooleanVar(
            value=self._config.postprocess.par2_repair if self._config else True
        )
        ttk.Checkbutton(
            parent, text="Auto-repair if needed", variable=self._par2_repair_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # Extraction options
        ttk.Label(parent, text="Extraction:", font=FONT_SECTION).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 4)
        )
        row += 1

        self._auto_extract_var = tk.BooleanVar(
            value=self._config.postprocess.auto_extract if self._config else True
        )
        ttk.Checkbutton(
            parent, text="Auto-extract archives (RAR, ZIP, 7z)", variable=self._auto_extract_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)
        row += 1

        self._cleanup_var = tk.BooleanVar(
            value=self._config.postprocess.cleanup_after_extract if self._config else True
        )
        ttk.Checkbutton(
            parent, text="Delete archives after extraction", variable=self._cleanup_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)

    def _create_ram_gpu_tab(self, parent) -> None:
        """RAM/GPU processing settings."""
        parent.columnconfigure(1, weight=1)

        row = 0
        # Header info
        info = ttk.Label(
            parent,
            text="Process downloads in RAM for maximum speed.\nRequires sufficient RAM (recommended: 32+ GB).",
            font=FONT_LABEL
        )
        info.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        # Enable RAM processing
        self._ram_enabled_var = tk.BooleanVar(
            value=self._config.ram_processing.enabled if self._config else False
        )
        ttk.Checkbutton(
            parent,
            text="Enable RAM-based processing",
            variable=self._ram_enabled_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # Max RAM size
        ttk.Label(parent, text="Max RAM Size:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._ram_size_var = tk.IntVar(
            value=self._config.ram_processing.max_size_mb // 1024 if self._config else 32
        )
        ram_frame = ttk.Frame(parent)
        ram_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(ram_frame, textvariable=self._ram_size_var, from_=1, to=128, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(ram_frame, text="GB (for processing)", font=FONT_LABEL).pack(side="left", padx=(10, 0))
        row += 1

        # Flush buffer
        ttk.Label(parent, text="Flush Buffer:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._flush_buffer_var = tk.IntVar(
            value=self._config.ram_processing.flush_buffer_mb if self._config else 256
        )
        flush_frame = ttk.Frame(parent)
        flush_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(flush_frame, textvariable=self._flush_buffer_var, from_=64, to=4096, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(flush_frame, text="MB (disk write buffer)", font=FONT_LABEL).pack(side="left", padx=(10, 0))
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # GPU Section
        # Use IS_ULTIMATE_EDITION for label, HAS_CUDA for functionality
        if not IS_ULTIMATE_EDITION:
            gpu_section_text = "GPU Acceleration: (Basic Edition - not available)"
        elif HAS_CUDA:
            gpu_section_text = "GPU Acceleration:"
        else:
            gpu_section_text = "GPU Acceleration: (CUDA not detected)"
        ttk.Label(parent, text=gpu_section_text, font=FONT_SECTION).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 4)
        )
        row += 1

        # Show CUDA error if any (for debugging)
        if IS_ULTIMATE_EDITION and not HAS_CUDA and _cuda_error:
            error_label = ttk.Label(parent, text=f"Error: {_cuda_error}", font=FONT_LABEL, foreground="red")
            error_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)
            row += 1

        # GPU repair - disabled if Basic edition OR no CUDA
        gpu_enabled = IS_ULTIMATE_EDITION and HAS_CUDA
        self._gpu_repair_var = tk.BooleanVar(
            value=(self._config.ram_processing.gpu_repair if self._config else True) if gpu_enabled else False
        )
        gpu_repair_cb = ttk.Checkbutton(
            parent,
            text="Use GPU for Reed-Solomon repair (CUDA)",
            variable=self._gpu_repair_var
        )
        gpu_repair_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)
        if not gpu_enabled:
            gpu_repair_cb.configure(state="disabled")
        row += 1

        # GPU device
        gpu_device_label = ttk.Label(parent, text="GPU Device ID:", font=FONT_LABEL)
        gpu_device_label.grid(row=row, column=0, sticky="w", padx=(20, 0), pady=6)
        self._gpu_device_var = tk.IntVar(
            value=self._config.ram_processing.gpu_device_id if self._config else 0
        )
        gpu_frame = ttk.Frame(parent)
        gpu_frame.grid(row=row, column=1, sticky="w", pady=6)
        gpu_spinbox = ttk.Spinbox(gpu_frame, textvariable=self._gpu_device_var, from_=0, to=7, width=8, font=FONT_VALUE_SMALL)
        gpu_spinbox.pack(side="left")
        ttk.Label(gpu_frame, text="(0 = first GPU)", font=FONT_LABEL).pack(side="left", padx=(10, 0))
        if not gpu_enabled:
            gpu_spinbox.configure(state="disabled")
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # Verify threads
        ttk.Label(parent, text="Verify Threads:", font=FONT_LABEL).grid(row=row, column=0, sticky="w", pady=6)
        self._verify_threads_var = tk.IntVar(
            value=self._config.ram_processing.verify_threads if self._config else 0
        )
        verify_frame = ttk.Frame(parent)
        verify_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Spinbox(verify_frame, textvariable=self._verify_threads_var, from_=0, to=32, width=8, font=FONT_VALUE_SMALL).pack(side="left")
        ttk.Label(verify_frame, text="(0 = auto, for MD5)", font=FONT_LABEL).pack(side="left", padx=(10, 0))

    def _create_system_tab(self, parent) -> None:
        """System settings (tray, auto-start)."""
        parent.columnconfigure(1, weight=1)

        row = 0

        # System Tray Section
        ttk.Label(parent, text="System Tray:", font=FONT_SECTION).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        row += 1

        # Minimize to tray
        self._minimize_to_tray_var = tk.BooleanVar(
            value=self._config.ui.minimize_to_tray if self._config else True
        )
        ttk.Checkbutton(
            parent,
            text="Minimize to tray instead of closing",
            variable=self._minimize_to_tray_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)
        row += 1

        # Start minimized
        self._start_minimized_var = tk.BooleanVar(
            value=self._config.ui.start_minimized if self._config else False
        )
        ttk.Checkbutton(
            parent,
            text="Start minimized in system tray",
            variable=self._start_minimized_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # Windows Startup Section
        ttk.Label(parent, text="Windows Startup:", font=FONT_SECTION).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        row += 1

        # Start with Windows
        self._start_with_windows_var = tk.BooleanVar(
            value=self._config.ui.start_with_windows if self._config else False
        )
        ttk.Checkbutton(
            parent,
            text="Start DLER when Windows starts",
            variable=self._start_with_windows_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)
        row += 1

        # Note
        note = ttk.Label(
            parent,
            text="(Registry: HKEY_CURRENT_USER, no admin required)",
            font=("Segoe UI", 8),
            foreground="gray"
        )
        note.grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=(5, 0))
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # Notifications Section
        ttk.Label(parent, text="Notifications:", font=FONT_SECTION).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        row += 1

        # Show completion summary
        self._show_completion_summary_var = tk.BooleanVar(
            value=self._config.ui.show_completion_summary if self._config else True
        )
        ttk.Checkbutton(
            parent,
            text="Show completion summary popup after download",
            variable=self._show_completion_summary_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=2)

    def _browse_extract_dir(self) -> None:
        """Browse extraction directory."""
        path = filedialog.askdirectory(initialdir=self._extract_dir_var.get())
        if path:
            self._extract_dir_var.set(path)

    def _browse_dir(self) -> None:
        """Browse directory."""
        path = filedialog.askdirectory(initialdir=self._dir_var.get())
        if path:
            self._dir_var.set(path)

    def _save(self) -> None:
        """Save settings."""
        from ..utils.config import Config, save_config

        if self._config is None:
            self._config = Config()

        self._config.server.host = self._host_var.get()
        self._config.server.port = self._port_var.get()
        self._config.server.use_ssl = self._ssl_var.get()
        self._config.server.username = self._user_var.get()
        self._config.server.password = self._pass_var.get()
        self._config.server.connections = self._conn_var.get()
        self._config.server.timeout = self._timeout_var.get()

        self._config.download.output_dir = self._dir_var.get()
        self._config.download.max_retries = self._retries_var.get()

        self._config.turbo.decoder_threads = self._decoder_var.get()
        self._config.turbo.writer_threads = self._writer_var.get()
        self._config.turbo.pipeline_depth = self._pipeline_var.get()
        self._config.turbo.write_through = self._writethrough_var.get()

        # Post-processing settings
        self._config.postprocess.enabled = self._pp_enabled_var.get()
        self._config.postprocess.extract_dir = self._extract_dir_var.get()
        self._config.postprocess.par2_verify = self._par2_verify_var.get()
        self._config.postprocess.par2_repair = self._par2_repair_var.get()
        self._config.postprocess.auto_extract = self._auto_extract_var.get()
        self._config.postprocess.cleanup_after_extract = self._cleanup_var.get()

        # RAM/GPU processing settings
        self._config.ram_processing.enabled = self._ram_enabled_var.get()
        self._config.ram_processing.max_size_mb = self._ram_size_var.get() * 1024  # GB to MB
        self._config.ram_processing.gpu_repair = self._gpu_repair_var.get()
        self._config.ram_processing.gpu_device_id = self._gpu_device_var.get()
        self._config.ram_processing.verify_threads = self._verify_threads_var.get()
        self._config.ram_processing.flush_buffer_mb = self._flush_buffer_var.get()

        # System settings (tray, auto-start, notifications)
        self._config.ui.minimize_to_tray = self._minimize_to_tray_var.get()
        self._config.ui.start_minimized = self._start_minimized_var.get()
        self._config.ui.start_with_windows = self._start_with_windows_var.get()
        self._config.ui.show_completion_summary = self._show_completion_summary_var.get()

        # Update Windows auto-start registry
        if HAS_SYSTEM_TRAY:
            set_autostart(
                self._config.ui.start_with_windows,
                self._config.ui.start_minimized
            )

        if save_config(self._config):
            self._on_save(self._config)
            self._dialog.destroy()
        else:
            messagebox.showerror("Error", "Failed to save settings")


def main():
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app = DLERApp()
    app.run()


if __name__ == "__main__":
    main()
