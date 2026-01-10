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
from typing import Optional, Callable
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

logger = logging.getLogger(__name__)

# Fixed dimensions for consistent layout
SPEED_PANEL_WIDTH = 200
SPEED_PANEL_HEIGHT = 150
PROGRESS_PANEL_HEIGHT = 150
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
    status_message: str = "Ready"


class DLERApp:
    """DLER NZB Downloader - High-end professional interface."""

    def __init__(self):
        self._state = DownloadState()
        self._turbo_engine = None
        self._config = None
        self._download_queue: list[Path] = []
        self._update_job = None
        self._logo_image = None
        self._icon_image = None
        self._is_paused = False
        self._download_complete = False  # Prevents progress updates after completion

        self._setup_ui()
        self._load_config()

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

        self._master.geometry("720x600")
        self._master.minsize(700, 580)
        self._master.resizable(True, True)

        # Set window icon (taskbar icon)
        self._set_window_icon()

        # Main container
        main_container = ttk.Frame(self._master, padding=20)
        main_container.pack(fill="both", expand=True)

        # Fixed grid weights
        main_container.columnconfigure(0, weight=0, minsize=SPEED_PANEL_WIDTH)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(3, weight=1)

        self._create_header(main_container)
        self._create_speed_panel(main_container)
        self._create_progress_panel(main_container)
        self._create_stats_panel(main_container)
        self._create_queue_panel(main_container)
        self._create_action_bar(main_container)

        # Start UI update loop
        self._schedule_update()

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

    def _create_stats_panel(self, parent) -> None:
        """Create statistics panel with elegant subframes."""
        # Outer frame
        stats_outer = ttk.Frame(parent, height=STATS_PANEL_HEIGHT)
        stats_outer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
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
        queue_outer.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 12))
        queue_outer.columnconfigure(0, weight=1)
        queue_outer.rowconfigure(0, weight=1)

        # LabelFrame inside
        queue_frame = ttk.LabelFrame(queue_outer, text=" QUEUE ", padding=10)
        queue_frame.pack(fill="both", expand=True)
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        # Listbox with fixed font
        list_frame = ttk.Frame(queue_frame)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._queue_listbox = tk.Listbox(
            list_frame,
            height=4,
            font=FONT_VALUE_SMALL,
            selectmode="single",
            activestyle="none"
        )
        self._queue_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self._queue_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._queue_listbox.configure(yscrollcommand=scrollbar.set)

        # Status bar
        status_frame = ttk.Frame(queue_frame)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self._queue_info_var = tk.StringVar(value="Queue empty")
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

    def _create_action_bar(self, parent) -> None:
        """Create action buttons."""
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0))
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
            self._queue_listbox.insert("end", f"  {path.name}")
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

        if self._queue_listbox.size() > 0:
            self._queue_listbox.delete(0)
            self._queue_listbox.insert(0, f"> {nzb_path.name}")
            self._queue_listbox.itemconfig(0, foreground="#4CAF50")

        thread = threading.Thread(target=self._download_thread, args=(nzb_path,), daemon=True)
        thread.start()

    def _download_thread(self, nzb_path: Path) -> None:
        """Background download."""
        try:
            if self._turbo_engine is None:
                self._init_engine()

            if self._turbo_engine is None:
                self._queue_info_var.set("Failed to initialize")
                self._state.is_downloading = False
                return

            self._parse_nzb_size(nzb_path)
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

                # Force immediate UI update on main thread
                self._master.after(0, self._force_100_percent)
                if self._queue_listbox.size() > 0:
                    self._queue_listbox.delete(0)

                # === POST-PROCESSING (PAR2 + Extraction) ===
                logger.info(f"Download complete, checking post-processing...")
                logger.info(f"Config exists: {self._config is not None}")
                if self._config:
                    logger.info(f"Postprocess enabled: {self._config.postprocess.enabled}")

                if self._config and self._config.postprocess.enabled:
                    logger.info("Starting post-processing...")
                    self._run_post_processing(nzb_path)
                    logger.info("Post-processing finished")
                else:
                    logger.info("Post-processing disabled or no config")
            else:
                logger.warning("Download returned False - check [STATS] logs above")
                self._queue_info_var.set("Download incomplete")
                # Show warning for incomplete download
                self._master.after(100, lambda: self._show_warning(
                    "Incomplete Download",
                    "Some segments could not be downloaded.\n\n"
                    "Possible causes:\n"
                    "- DMCA takedown (articles removed from server)\n"
                    "- Server retention expired\n"
                    "- Network issues\n\n"
                    "Check logs for details. PAR2 repair may still recover the files."
                ))

        except Exception as e:
            logger.error(f"Download error: {e}")
            self._queue_info_var.set(f"Error: {e}")
        finally:
            self._state.is_downloading = False
            self._state.current_file = ""
            self._update_queue_info()
            self._master.after(500, self._start_next_download)

    def _run_post_processing(self, nzb_path: Path) -> None:
        """Run PAR2 verification and extraction."""
        logger.info(f"=== _run_post_processing called for {nzb_path.name} ===")
        try:
            from ..core.post_processor import PostProcessor
            logger.info("PostProcessor imported successfully")

            self._queue_info_var.set("Post-processing...")
            logger.info(f"Starting post-processing for {nzb_path.name}")

            # Get download directory
            download_dir = Path(self._config.download.output_dir)
            extract_dir = Path(self._config.postprocess.extract_dir)

            def progress_callback(msg: str, percent: float):
                self._queue_info_var.set(f"PP: {msg}")

            logger.info(f"Download dir: {download_dir}")
            logger.info(f"Extract dir: {extract_dir}")
            logger.info(f"PAR2 path: {self._config.postprocess.par2_path or 'auto-detect'}")
            logger.info(f"7z path: {self._config.postprocess.sevenzip_path or 'auto-detect'}")

            processor = PostProcessor(
                download_dir=download_dir,
                extract_dir=extract_dir,
                par2_path=self._config.postprocess.par2_path or None,
                sevenzip_path=self._config.postprocess.sevenzip_path or None,
                cleanup_after_extract=self._config.postprocess.cleanup_after_extract,
                on_progress=progress_callback
            )

            logger.info(f"PostProcessor created. PAR2={processor.par2_path}, 7z={processor.sevenzip_path}")
            logger.info("Calling processor.process()...")

            result = processor.process(nzb_path)
            logger.info(f"processor.process() returned: success={result.success}, msg={result.message}")

            if result.success:
                msg = f"Done! {result.files_extracted} files"
                if result.par2_repaired:
                    msg += " (repaired)"
                self._queue_info_var.set(msg)
                logger.info(f"Post-processing completed: {result.message}")

                # Show warnings to user if any
                if result.warnings:
                    self._show_post_process_warnings(result)
            else:
                self._queue_info_var.set(f"PP Error: {result.message[:30]}")
                logger.error(f"Post-processing failed: {result.message}")
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
                on_progress=self._on_progress
            )

            conn_count = self._turbo_engine.connect()
            self._state.connections_active = conn_count
            self._state.connections_total = conn_count  # Use actual connected count
            self._queue_info_var.set(f"Connected: {conn_count} connections")

        except Exception as e:
            logger.error(f"Init failed: {e}")
            self._turbo_engine = None
            self._queue_info_var.set(f"Connection failed")

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

        self._download_queue.clear()
        self._queue_listbox.delete(0, "end")
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

    def _force_100_percent(self) -> None:
        """Force progress to 100% on main thread."""
        self._state.progress_percent = 100.0
        self._progress_var.set(100.0)
        self._percent_var.set("100.0%")

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

        # Progress
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

    def run(self) -> None:
        """Start application."""
        if HAS_TKMT:
            self._root.run()
        else:
            self._root.mainloop()


class SettingsDialog:
    """Settings dialog with tabs."""

    def __init__(self, parent, config, on_save: Callable):
        self._config = config
        self._on_save = on_save

        self._dialog = tk.Toplevel(parent)
        self._dialog.title("Settings")
        self._dialog.geometry("520x620")
        self._dialog.resizable(False, False)
        self._dialog.transient(parent)
        self._dialog.grab_set()

        self._create_ui()

        self._dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 520) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 620) // 2
        self._dialog.geometry(f"+{x}+{y}")

    def _create_ui(self) -> None:
        """Create settings UI."""
        notebook = ttk.Notebook(self._dialog)
        notebook.pack(fill="both", expand=True, padx=15, pady=15)

        # Server tab
        server_frame = ttk.Frame(notebook, padding=20)
        notebook.add(server_frame, text="  Server  ")
        self._create_server_tab(server_frame)

        # Download tab
        download_frame = ttk.Frame(notebook, padding=20)
        notebook.add(download_frame, text="  Download  ")
        self._create_download_tab(download_frame)

        # Performance tab
        perf_frame = ttk.Frame(notebook, padding=20)
        notebook.add(perf_frame, text="  Performance  ")
        self._create_performance_tab(perf_frame)

        # Post-Processing tab
        pp_frame = ttk.Frame(notebook, padding=20)
        notebook.add(pp_frame, text="  Post-Process  ")
        self._create_postprocess_tab(pp_frame)

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
