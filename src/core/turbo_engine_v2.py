"""
Turbo Download Engine V2.5 - Maximum Throughput Edition
========================================================

Architecture avec separation complete des threads:

    [Download Threads]     [Decoder Threads]     [Writer Threads]
           |                      |                     |
      Network I/O            CPU (yEnc)            Disk I/O
           |                      |                     |
    +-------------+        +-------------+       +-------------+
    | Raw Queue   |------->| Write Queue |------>| mmap write  |
    | (max: 4000) |        | (max: 2000) |       | lock-free!  |
    +-------------+        +-------------+       +-------------+

Optimisations V2.5:
- LARGE queues (4000+2000) = ~4.8 GB buffer pour 64GB RAM systeme
- Flush reduit (500 segments) pour moins d'I/O
- Finalisation PROGRESSIVE (libere RAM des que fichier complet!)
- del explicite apres write pour forcer liberation memoire
- gc.collect() periodique (toutes les 5s)
- Monitoring RAM en temps reel
- SSL/TLS optimise: TLS 1.3, ciphers AES-GCM

Target: Debit maximal 10 Gbps avec RAM controlee
"""

import os
import sys
import mmap
import time
import logging
import threading
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple
from queue import Queue, Empty
from collections import defaultdict, deque
import xml.etree.ElementTree as ET

# RAM monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# RAM-based processing support
try:
    from .ram_processor import RamBuffer
    HAS_RAM_PROCESSOR = True
except ImportError:
    HAS_RAM_PROCESSOR = False
    RamBuffer = None  # Type hint placeholder

def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    else:
        # Fallback for Windows without psutil
        try:
            import ctypes
            # Use GetProcessMemoryInfo from psapi.dll
            psapi = ctypes.windll.psapi
            kernel32 = ctypes.windll.kernel32

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = kernel32.GetCurrentProcess()

            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / 1024 / 1024
        except Exception:
            pass
        return 0.0

# Robust imports for both development and PyInstaller
def _import_modules():
    import importlib
    import sys
    
    # Try different import paths
    fast_nntp_paths = [
        'src.core.fast_nntp',
        'core.fast_nntp', 
        'fast_nntp',
    ]
    turbo_yenc_paths = [
        'src.core.turbo_yenc',
        'core.turbo_yenc',
        'turbo_yenc',
    ]
    
    fast_nntp = None
    turbo_yenc = None
    
    # Try relative import first
    try:
        from . import fast_nntp as fast_nntp
        from . import turbo_yenc as turbo_yenc
    except ImportError:
        pass
    
    # Try absolute imports
    if fast_nntp is None:
        for path in fast_nntp_paths:
            try:
                fast_nntp = importlib.import_module(path)
                break
            except ImportError:
                continue
    
    if turbo_yenc is None:
        for path in turbo_yenc_paths:
            try:
                turbo_yenc = importlib.import_module(path)
                break
            except ImportError:
                continue
    
    if fast_nntp is None or turbo_yenc is None:
        raise ImportError(f"Could not import fast_nntp ({fast_nntp}) or turbo_yenc ({turbo_yenc})")
    
    return fast_nntp, turbo_yenc

_fast_nntp_mod, _turbo_yenc_mod = _import_modules()
NNTPConnection = _fast_nntp_mod.NNTPConnection
ServerConfig = _fast_nntp_mod.ServerConfig
TurboYEncDecoder = _turbo_yenc_mod.TurboYEncDecoder
NATIVE_AVAILABLE = _turbo_yenc_mod.NATIVE_AVAILABLE
NUMBA_AVAILABLE = _turbo_yenc_mod.NUMBA_AVAILABLE

logger = logging.getLogger(__name__)


class SpeedTracker:
    """
    Calcul de vitesse avec moyenne mobile exponentielle (EMA).
    Evite les fluctuations brutales du débit affiché.
    """

    def __init__(self, window_seconds: float = 3.0, alpha: float = 0.3):
        """
        Args:
            window_seconds: Fenêtre de temps pour la moyenne mobile
            alpha: Facteur de lissage EMA (0.1=lent, 0.5=rapide)
        """
        self._samples: deque = deque()  # (timestamp, bytes)
        self._window = window_seconds
        self._alpha = alpha
        self._ema_speed = 0.0
        self._last_bytes = 0
        self._last_time = time.time()
        self._lock = threading.Lock()

    def update(self, total_bytes: int) -> float:
        """
        Update avec le total de bytes téléchargés.
        Retourne la vitesse lissée en bytes/sec.
        """
        now = time.time()

        with self._lock:
            # Calcul du delta depuis la dernière mise à jour
            delta_bytes = total_bytes - self._last_bytes
            delta_time = now - self._last_time

            if delta_time <= 0:
                return self._ema_speed

            # Vitesse instantanée
            instant_speed = delta_bytes / delta_time

            # Moyenne mobile exponentielle
            if self._ema_speed == 0:
                self._ema_speed = instant_speed
            else:
                self._ema_speed = self._alpha * instant_speed + (1 - self._alpha) * self._ema_speed

            # Ajouter l'échantillon pour la moyenne sur fenêtre
            self._samples.append((now, total_bytes))

            # Nettoyer les vieux échantillons
            cutoff = now - self._window
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

            self._last_bytes = total_bytes
            self._last_time = now

            return self._ema_speed

    def get_window_speed(self) -> float:
        """Calcul de la vitesse moyenne sur la fenêtre de temps."""
        with self._lock:
            if len(self._samples) < 2:
                return self._ema_speed

            oldest_time, oldest_bytes = self._samples[0]
            newest_time, newest_bytes = self._samples[-1]

            dt = newest_time - oldest_time
            if dt <= 0:
                return self._ema_speed

            return (newest_bytes - oldest_bytes) / dt

    @property
    def speed_mbps(self) -> float:
        """Vitesse en MB/s (moyenne entre EMA et fenêtre pour stabilité)."""
        ema = self._ema_speed
        window = self.get_window_speed()
        # Moyenne des deux méthodes pour meilleure stabilité
        return (ema + window) / 2 / 1024 / 1024


@dataclass
class NZBSegment:
    """Segment info from NZB."""
    message_id: str
    number: int
    bytes: int
    file_index: int


@dataclass
class NZBFile:
    """File info from NZB."""
    index: int
    filename: str
    size: int
    segments: List[NZBSegment] = field(default_factory=list)
    obfuscated: bool = False  # True if filename is from obfuscated subject


@dataclass
class DecodedSegment:
    """Decoded segment ready for writing."""
    file_index: int
    position: int  # Byte offset in file
    data: bytes
    yenc_filename: Optional[str] = None  # Real filename from yEnc header


@dataclass
class TurboStatsV2:
    """Real-time statistics."""
    segments_total: int = 0
    bytes_downloaded: int = 0
    segments_downloaded: int = 0
    bytes_decoded: int = 0
    segments_decoded: int = 0
    bytes_written: int = 0
    segments_written: int = 0
    files_total: int = 0
    files_completed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def download_speed_mbps(self) -> float:
        if self.elapsed > 0:
            return self.bytes_downloaded / self.elapsed / 1024 / 1024
        return 0.0


class TurboEngineV2:
    """
    High-performance NZB downloader with separated pipeline.

    Three distinct thread pools:
    1. Download threads - Network I/O only
    2. Decoder threads - CPU-bound yEnc decoding
    3. Writer threads - Disk I/O only
    """

    def __init__(
        self,
        server_config: ServerConfig,
        output_dir: Path = Path("downloads"),
        download_threads: int = 60,  # Increased for 10 Gbps
        decoder_threads: int = 0,    # 0 = auto (24 for 5950X)
        writer_threads: int = 12,    # Increased for throughput
        pipeline_depth: int = 30,    # Deeper pipeline
        write_through: bool = False, # Bypass OS cache
        on_progress: Optional[Callable] = None,
        on_file_complete: Optional[Callable[[Path, str], None]] = None,  # Streaming PAR2 callback
        ram_buffer: Optional['RamBuffer'] = None,  # RAM-based storage (no disk writes)
        raw_queue_size: int = 16000,   # ~12 GB buffer for raw data
        write_queue_size: int = 8000   # ~6 GB buffer for decoded data
    ):
        self.server_config = server_config
        self.output_dir = Path(output_dir)
        self.download_threads = download_threads
        # Optimized for Ryzen 5950X: 24 decoder threads (leave 8 for download/write)
        self.decoder_threads = decoder_threads or min(os.cpu_count() or 8, 24)
        self.writer_threads = writer_threads
        self.pipeline_depth = pipeline_depth
        self.write_through = write_through
        self.on_progress = on_progress
        self.on_file_complete = on_file_complete  # Called when individual file finishes
        self.ram_buffer = ram_buffer  # If set, write to RAM instead of disk
        self.ram_mode = ram_buffer is not None
        self.raw_queue_size = raw_queue_size
        self.write_queue_size = write_queue_size

        # Connections
        self._connections: List[NNTPConnection] = []

        # Decoder
        self._decoder = TurboYEncDecoder()

        # State
        self._running = False
        self._paused = False
        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self._stats = TurboStatsV2()

        # Thread-safe counters
        self._lock = threading.Lock()

        # Speed tracker with smoothing
        self._speed_tracker = SpeedTracker(window_seconds=3.0, alpha=0.25)

        # === LARGE QUEUES FOR MAXIMUM THROUGHPUT ===
        # Buffer size determines how much data can be downloaded before backpressure kicks in
        # Each segment ~750KB, so 16000 segments = ~12GB buffer
        # This allows full-speed download even if disk is slower than network
        self._raw_queue: Queue[Tuple[NZBSegment, bytes]] = Queue(maxsize=self.raw_queue_size)
        self._write_queue: Queue[DecodedSegment] = Queue(maxsize=self.write_queue_size)

        # File handles with mmap for fast writes
        # Dict[file_index] -> [fd, mmap, size, write_count]
        self._file_handles: Dict[int, List] = {}
        self._file_locks: Dict[int, threading.Lock] = {}

        # === PROGRESSIVE FILE FINALIZATION ===
        # Track progress per file to auto-finalize when complete
        # Dict[file_index] -> [segments_written, total_segments]
        self._file_progress: Dict[int, List[int]] = {}
        self._file_info: Dict[int, 'NZBFile'] = {}
        self._progress_lock = threading.Lock()

        # === OBFUSCATED FILE RENAMING ===
        # Track files already renamed: Dict[file_index] -> real_filename
        self._renamed_files: Dict[int, str] = {}
        self._rename_lock = threading.Lock()
        # Track file paths: Dict[file_index] -> current_path
        self._file_paths: Dict[int, Path] = {}
        # Track used filenames to avoid collisions: Set[safe_name]
        self._used_filenames: set = set()

        # Flush mmap every N writes (reduced I/O for throughput)
        # 500 segments ~= 400 MB between flushes
        self._flush_interval = 500

        # Ensure output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> int:
        """Establish all NNTP connections."""
        print(f"Connecting to {self.server_config.host}:{self.server_config.port}...")

        def create_conn(i: int) -> Optional[NNTPConnection]:
            conn = NNTPConnection(self.server_config, i)
            return conn if conn.connect() else None

        # Parallel connection establishment
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=self.server_config.connections) as executor:
            futures = [executor.submit(create_conn, i)
                      for i in range(self.server_config.connections)]
            for future in as_completed(futures):
                conn = future.result()
                if conn:
                    self._connections.append(conn)

        print(f"Connected: {len(self._connections)}/{self.server_config.connections}")
        return len(self._connections)

    def pause(self) -> None:
        """Pause download."""
        self._paused = True
        self._pause_event.clear()
        print("[ENGINE] Paused")

    def resume(self) -> None:
        """Resume download."""
        self._paused = False
        self._pause_event.set()
        print("[ENGINE] Resumed")

    def stop(self) -> None:
        """Stop download."""
        self._stop_requested = True
        self._running = False
        # Unblock any paused threads
        self._pause_event.set()
        print("[ENGINE] Stop requested")

    def disconnect(self) -> None:
        """Disconnect all connections."""
        self._running = False
        self._stop_requested = True
        self._pause_event.set()

        for conn in self._connections:
            try:
                if conn and conn.socket:
                    conn.socket.close()
            except:
                pass
        self._connections.clear()
        print("[ENGINE] Disconnected")

    def download_nzb(self, nzb_path: Path) -> bool:
        """Download all files from NZB."""
        if not self._connections:
            if self.connect() == 0:
                return False

        # Parse NZB
        files = self._parse_nzb(nzb_path)
        if not files:
            print("Failed to parse NZB")
            return False

        # Initialize stats
        self._stats = TurboStatsV2()
        self._stats.files_total = len(files)
        self._stats.segments_total = sum(len(f.segments) for f in files)

        # Reset pause/stop flags
        self._paused = False
        self._stop_requested = False
        self._pause_event.set()

        # Reset speed tracker
        self._speed_tracker = SpeedTracker(window_seconds=3.0, alpha=0.25)

        # Reset filename tracking for this download
        self._used_filenames.clear()
        self._renamed_files.clear()
        self._file_paths.clear()

        logger.info(f"[DOWNLOAD] Starting: {len(files)} files, {self._stats.segments_total} segments")
        logger.info(f"[DOWNLOAD] Pipeline: {len(self._connections)} connections, "
                   f"{self.decoder_threads} decoders, {self.writer_threads} writers")
        print()

        self._running = True

        try:
            # Prepare output files with memory mapping
            # Also initialize progress tracking for progressive finalization
            logger.info(f"[PREPARE] Preparing {len(files)} files... (RAM mode: {self.ram_mode})")
            for nzb_file in files:
                logger.debug(f"[PREPARE] File {nzb_file.index}: '{nzb_file.filename}' ({nzb_file.size} bytes, {len(nzb_file.segments)} segments)")
                self._prepare_file(nzb_file)
                # Verify file handle was created
                if nzb_file.index not in self._file_handles:
                    logger.error(f"[PREPARE] FAILED to create handle for file {nzb_file.index}: '{nzb_file.filename}'")
                else:
                    logger.debug(f"[PREPARE] OK - handle created for file {nzb_file.index}")
                # Initialize progress tracking: [written, total]
                self._file_progress[nzb_file.index] = [0, len(nzb_file.segments)]
                self._file_info[nzb_file.index] = nzb_file

            # Log summary of prepared files
            logger.info(f"[PREPARE] Completed: {len(self._file_handles)} handles created for {len(files)} files")

            # Build work queue and segment map
            work_queue: Queue[NZBSegment] = Queue()
            segment_map: Dict[str, NZBSegment] = {}
            for f in files:
                for seg in f.segments:
                    work_queue.put(seg)
                    segment_map[seg.message_id] = seg

            # === START ALL THREAD POOLS ===
            raw_buffer_gb = self.raw_queue_size * 750 / 1024 / 1024  # ~750KB per segment
            write_buffer_gb = self.write_queue_size * 750 / 1024 / 1024
            logger.info(f"[THREADS] Starting pools: {len(self._connections)} connections, "
                       f"{self.decoder_threads} decoders, {self.writer_threads} writers")
            logger.info(f"[BUFFERS] raw_queue={self.raw_queue_size} (~{raw_buffer_gb:.1f}GB), "
                       f"write_queue={self.write_queue_size} (~{write_buffer_gb:.1f}GB)")

            # 1. Writer threads (start first - consumers)
            writer_threads = []
            for i in range(self.writer_threads):
                t = threading.Thread(
                    target=self._writer_worker,
                    name=f"Writer-{i}",
                    daemon=True
                )
                t.start()
                writer_threads.append(t)

            # 2. Decoder threads (middle - transform)
            decoder_threads = []
            for i in range(self.decoder_threads):
                t = threading.Thread(
                    target=self._decoder_worker,
                    name=f"Decoder-{i}",
                    daemon=True
                )
                t.start()
                decoder_threads.append(t)

            # 3. Download threads (start last - producers)
            download_threads = []
            for i, conn in enumerate(self._connections):
                t = threading.Thread(
                    target=self._download_worker,
                    args=(conn, work_queue, segment_map),
                    name=f"Download-{i}",
                    daemon=True
                )
                t.start()
                download_threads.append(t)
                # Stagger starts for smooth flow
                if i < len(self._connections) - 1:
                    time.sleep(0.003)

            # 4. Progress reporter
            progress_thread = threading.Thread(
                target=self._progress_reporter,
                name="Progress",
                daemon=True
            )
            progress_thread.start()
            logger.info(f"[THREADS] All pools started: {len(download_threads)} downloaders active")

            # Wait for downloads to complete
            logger.info(f"[THREADS] Waiting for {len(download_threads)} download threads...")
            threads_alive = len(download_threads)
            check_interval = 5  # Check every 5 seconds
            max_wait = 600  # 10 minutes max
            waited = 0

            while threads_alive > 0 and waited < max_wait:
                time.sleep(check_interval)
                waited += check_interval
                threads_alive = sum(1 for t in download_threads if t.is_alive())

                # Log progress with queue status
                raw_q = self._raw_queue.qsize()
                write_q = self._write_queue.qsize()
                raw_pct = raw_q * 100 // self.raw_queue_size if self.raw_queue_size else 0
                write_pct = write_q * 100 // self.write_queue_size if self.write_queue_size else 0
                logger.info(f"[PROGRESS] Threads: {threads_alive} active | "
                           f"Queues: raw={raw_pct}%, write={write_pct}% | "
                           f"Segments: {self._stats.segments_downloaded}/{self._stats.segments_total}")

            if threads_alive > 0:
                logger.warning(f"[THREADS] {threads_alive} download threads still alive after {max_wait}s!")
            else:
                logger.info(f"[THREADS] All download threads done in {waited}s")

            # Signal decoders to finish (poison pills)
            logger.info("[SHUTDOWN] Sending poison pills to decoders...")
            for _ in range(self.decoder_threads):
                self._raw_queue.put(None)

            # Wait for decoders
            logger.info("[SHUTDOWN] Waiting for decoder threads...")
            for i, t in enumerate(decoder_threads):
                t.join(timeout=30)
                if t.is_alive():
                    logger.warning(f"[SHUTDOWN] Decoder thread {i} still alive!")
            logger.info("[SHUTDOWN] All decoder threads done")

            # Signal writers to finish
            logger.info("[SHUTDOWN] Sending poison pills to writers...")
            for _ in range(self.writer_threads):
                self._write_queue.put(None)

            # Wait for writers
            logger.info("[SHUTDOWN] Waiting for writer threads...")
            for i, t in enumerate(writer_threads):
                t.join(timeout=30)
                if t.is_alive():
                    logger.warning(f"[SHUTDOWN] Writer thread {i} still alive!")
            logger.info("[SHUTDOWN] All writer threads done")

            self._running = False

            # Finalize any remaining files (not already finalized progressively)
            for nzb_file in files:
                if nzb_file.index in self._file_handles:
                    self._finalize_file(nzb_file)

            # Clear tracking dicts
            self._file_progress.clear()
            self._file_info.clear()

            # Final stats
            logger.info(f"[STATS] Downloaded: {self._stats.segments_downloaded}/{self._stats.segments_total}")
            logger.info(f"[STATS] Decoded: {self._stats.segments_decoded}")
            logger.info(f"[STATS] Written: {self._stats.segments_written}")
            logger.info(f"[STATS] Errors: {self._stats.errors}")
            logger.info(f"[STATS] Files completed: {self._stats.files_completed}/{self._stats.files_total}")

            # Return True if most segments succeeded (allow some missing)
            success_rate = self._stats.segments_written / max(1, self._stats.segments_total)
            success = success_rate >= 0.95  # 95% threshold
            logger.info(f"[STATS] Success rate: {success_rate*100:.1f}%, returning {success}")

            return success

        except KeyboardInterrupt:
            print("\n\nCancelled by user")
            self._running = False
            return False

    def _download_worker(
        self,
        conn: NNTPConnection,
        work_queue: Queue[NZBSegment],
        segment_map: Dict[str, NZBSegment]
    ) -> None:
        """
        Download worker - ONLY network I/O.

        Fetches raw data and puts it in raw_queue.
        Never touches disk, never decodes.
        """
        local_bytes = 0
        local_segs = 0
        local_errors = 0
        update_counter = 0

        def get_next_id():
            try:
                seg = work_queue.get_nowait()
                return seg.message_id
            except Empty:
                return None

        def on_data(msg_id: str, data: Optional[bytes]):
            nonlocal local_bytes, local_segs, local_errors, update_counter

            segment = segment_map.get(msg_id)
            if segment and data:
                local_bytes += len(data)
                local_segs += 1
                # Put in raw queue (may block if queue full - backpressure)
                self._raw_queue.put((segment, data))
            else:
                local_errors += 1

            work_queue.task_done()
            update_counter += 1

            # Update stats frequently for smooth display
            if update_counter >= 5:
                with self._lock:
                    self._stats.bytes_downloaded += local_bytes
                    self._stats.segments_downloaded += local_segs
                    self._stats.errors += local_errors
                local_bytes = 0
                local_segs = 0
                local_errors = 0
                update_counter = 0

        # Use streaming fetch - this blocks until queue is empty
        # Pass pause_event and stop_flag for immediate pause/stop response
        try:
            logger.debug(f"{threading.current_thread().name}: Starting fetch_streaming")
            conn.fetch_streaming(
                get_next_id,
                on_data,
                self.pipeline_depth,
                pause_event=self._pause_event,
                stop_flag=lambda: self._stop_requested
            )
            logger.debug(f"{threading.current_thread().name}: fetch_streaming completed")
        except Exception as e:
            logger.error(f"{threading.current_thread().name}: fetch_streaming error: {e}")

        # Final update
        with self._lock:
            self._stats.bytes_downloaded += local_bytes
            self._stats.segments_downloaded += local_segs
            self._stats.errors += local_errors

    def _decoder_worker(self) -> None:
        """
        Decoder worker - ONLY CPU work.

        Takes raw data from raw_queue, decodes yEnc,
        puts decoded data in write_queue.
        """
        local_bytes = 0
        local_segs = 0
        local_errors = 0
        update_counter = 0

        while True:
            # Check for stop request
            if self._stop_requested:
                break

            # Wait if paused
            self._pause_event.wait()

            try:
                # Short timeout for responsiveness
                item = self._raw_queue.get(timeout=0.05)
            except Empty:
                # Flush stats periodically even when idle
                if update_counter > 0:
                    with self._lock:
                        self._stats.bytes_decoded += local_bytes
                        self._stats.segments_decoded += local_segs
                    local_bytes = 0
                    local_segs = 0
                    update_counter = 0
                if not self._running:
                    break
                continue

            # Poison pill
            if item is None:
                break

            segment, raw_data = item

            try:
                # Decode yEnc (CPU-intensive, but GIL-free with native module)
                result = self._decoder.decode(raw_data)

                # Clear raw data immediately to free RAM
                del raw_data

                if result and result.valid and result.begin > 0:
                    # Pass yEnc filename for obfuscated file detection
                    yenc_name = result.filename if result.filename else None
                    decoded_seg = DecodedSegment(
                        file_index=segment.file_index,
                        position=result.begin - 1,
                        data=result.data,
                        yenc_filename=yenc_name
                    )
                    local_bytes += len(result.data)
                    local_segs += 1
                    # Put in write queue
                    self._write_queue.put(decoded_seg)
                    del decoded_seg  # Clear local reference
                else:
                    local_errors += 1

                del result  # Clear decoder result

            except Exception as e:
                local_errors += 1

            del item  # Clear item reference
            self._raw_queue.task_done()
            update_counter += 1

            # Update stats frequently
            if update_counter >= 10:
                with self._lock:
                    self._stats.bytes_decoded += local_bytes
                    self._stats.segments_decoded += local_segs
                local_bytes = 0
                local_segs = 0
                update_counter = 0

        # Final update
        with self._lock:
            self._stats.bytes_decoded += local_bytes
            self._stats.segments_decoded += local_segs
            self._stats.errors += local_errors

    def _writer_worker(self) -> None:
        """
        Writer worker - ONLY disk I/O.

        Takes decoded data from write_queue, writes to disk.
        Uses memory-mapped files for efficiency.
        """
        local_bytes = 0
        local_segs = 0
        local_errors = 0
        update_counter = 0

        while True:
            # Check for stop request
            if self._stop_requested:
                break

            # Wait if paused
            self._pause_event.wait()

            try:
                # Short timeout for responsiveness
                item = self._write_queue.get(timeout=0.05)
            except Empty:
                # Flush stats periodically even when idle
                if update_counter > 0:
                    with self._lock:
                        self._stats.bytes_written += local_bytes
                        self._stats.segments_written += local_segs
                    local_bytes = 0
                    local_segs = 0
                    update_counter = 0
                if not self._running:
                    break
                continue

            # Poison pill
            if item is None:
                break

            try:
                file_idx = item.file_index

                # === OBFUSCATED FILE RENAMING ===
                # If we got a real filename from yEnc header, store it for later rename
                if item.yenc_filename and file_idx not in self._renamed_files:
                    with self._rename_lock:
                        if file_idx not in self._renamed_files:
                            self._renamed_files[file_idx] = item.yenc_filename

                if file_idx not in self._file_handles:
                    logger.warning(f"[WRITER] No file handle for file_idx={file_idx}, segment dropped!")
                    local_errors += 1
                    continue

                handle = self._file_handles[file_idx]
                pos = item.position
                data = item.data
                data_len = len(data)
                file_size = handle[2]

                # Skip writes for zero-size files (dummy handles)
                if file_size == 0:
                    logger.debug(f"[WRITER] Skipping write for zero-size file {file_idx}")
                    local_segs += 1  # Count as success for progress
                    # Still track progress for finalization
                    with self._progress_lock:
                        if file_idx in self._file_progress:
                            progress = self._file_progress[file_idx]
                            progress[0] += 1
                            if progress[0] >= progress[1]:
                                nzb_file = self._file_info.get(file_idx)
                                if nzb_file:
                                    self._finalize_file(nzb_file)
                                    del self._file_progress[file_idx]
                                    del self._file_info[file_idx]
                    continue

                if pos + data_len <= file_size:
                    if self.ram_mode:
                        # === RAM MODE: Write to BytesIO buffer ===
                        # MUST use lock - BytesIO is not thread-safe for concurrent seek+write
                        ram_file = handle[0]  # RamFile object

                        # Debug: Log first segment write for each file (pos=0)
                        if pos == 0:
                            first_bytes = data[:16].hex() if len(data) >= 16 else data.hex()
                            logger.debug(f"[RAM WRITE] file_idx={file_idx} pos=0 len={data_len} "
                                        f"first_16={first_bytes} filename={ram_file.filename}")

                        with self._file_locks[file_idx]:
                            ram_file.data.seek(pos)
                            ram_file.data.write(data)
                            # Track actual bytes written (max position reached)
                            ram_file.update_actual_size(pos, data_len)
                        local_bytes += data_len
                        local_segs += 1
                        # No flush needed for RAM
                    elif self.write_through:
                        # WriteThrough mode: direct file write (DyMaxIO compatible)
                        f = handle[0]
                        with self._file_locks[file_idx]:
                            f.seek(pos)
                            f.write(data)
                        local_bytes += data_len
                        local_segs += 1
                    else:
                        # mmap mode: fast memory-mapped write
                        mm = handle[1]
                        if mm:
                            mm[pos:pos + data_len] = data
                            local_bytes += data_len
                            local_segs += 1

                            # Periodic flush
                            write_count = handle[3] + 1
                            handle[3] = write_count
                            if write_count % self._flush_interval == 0:
                                with self._file_locks[file_idx]:
                                    mm.flush()

                    # === PROGRESSIVE FILE FINALIZATION ===
                    # Track progress and auto-finalize when file is complete
                    with self._progress_lock:
                        if file_idx in self._file_progress:
                            progress = self._file_progress[file_idx]
                            progress[0] += 1  # Increment written count
                            if progress[0] >= progress[1]:
                                # File complete! Finalize to release RAM
                                nzb_file = self._file_info.get(file_idx)
                                if nzb_file:
                                    self._finalize_file(nzb_file)
                                    del self._file_progress[file_idx]
                                    del self._file_info[file_idx]
                else:
                    logger.warning(f"[WRITER] Position out of bounds: pos={pos}, data_len={data_len}, file_size={file_size}")
                    local_errors += 1

                # Clear reference to release memory immediately
                del item
                del data

            except Exception as e:
                local_errors += 1
                logger.warning(f"[WRITER] Error writing segment: {e}")

            self._write_queue.task_done()
            update_counter += 1

            # Update stats frequently
            if update_counter >= 5:
                with self._lock:
                    self._stats.bytes_written += local_bytes
                    self._stats.segments_written += local_segs
                    self._stats.errors += local_errors
                local_bytes = 0
                local_segs = 0
                local_errors = 0
                update_counter = 0

        # Final update
        with self._lock:
            self._stats.bytes_written += local_bytes
            self._stats.segments_written += local_segs
            self._stats.errors += local_errors

    def _progress_reporter(self) -> None:
        """Report progress periodically with smoothed speed."""
        # Wait a moment for data to start flowing
        time.sleep(0.5)

        gc_counter = 0
        while self._running:
            time.sleep(0.1)  # Update more frequently for smoother display

            # Periodic garbage collection to release memory
            gc_counter += 1
            if gc_counter >= 50:  # Every 5 seconds
                gc.collect()
                gc_counter = 0

            with self._lock:
                dl_bytes = self._stats.bytes_downloaded
                dl_segs = self._stats.segments_downloaded
                dec_segs = self._stats.segments_decoded
                wr_segs = self._stats.segments_written
                errors = self._stats.errors
                total = self._stats.segments_total

            # Use smoothed speed calculation
            self._speed_tracker.update(dl_bytes)
            speed = self._speed_tracker.speed_mbps

            percent = (wr_segs / total * 100) if total > 0 else 0

            # Show queue depths
            raw_q = self._raw_queue.qsize()
            write_q = self._write_queue.qsize()

            # Visual indicator of queue health (queues are 4000/2000)
            if raw_q > 3600:
                q_status = "FULL"
            elif raw_q > 2400:
                q_status = "HIGH"
            elif raw_q < 400:
                q_status = "LOW"
            else:
                q_status = "OK"

            # Get RAM usage
            ram_mb = get_process_memory_mb()

            print(f"\r[{percent:5.1f}%] {speed:7.1f} MB/s | "
                  f"DL:{dl_segs} -> Dec:{dec_segs} -> Wr:{wr_segs}/{total} | "
                  f"Q[{raw_q}/{write_q}] | RAM:{ram_mb:.0f}MB | Err:{errors}   ",
                  end="", flush=True)

            if self.on_progress:
                self.on_progress(self._stats)

    def _parse_nzb(self, nzb_path: Path) -> List[NZBFile]:
        """Parse NZB file."""
        files = []
        try:
            tree = ET.parse(nzb_path)
            root = tree.getroot()
            ns = {'nzb': 'http://www.newzbin.com/DTD/2003/nzb'}

            for idx, file_elem in enumerate(root.findall('.//nzb:file', ns)):
                subject = file_elem.get('subject', '')

                # Extract filename from subject
                filename = subject
                obfuscated = False

                if '"' in subject:
                    parts = subject.split('"')
                    if len(parts) >= 2 and parts[1].strip():
                        filename = parts[1]
                    else:
                        # Quotes but empty/invalid filename = obfuscated
                        obfuscated = True
                else:
                    # No quotes in subject = likely obfuscated
                    # Check if subject looks like random string (no spaces, no extension)
                    if ' ' not in subject and '.' not in subject and len(subject) > 10:
                        obfuscated = True

                segments = []
                total_bytes = 0

                for seg_elem in file_elem.findall('.//nzb:segment', ns):
                    seg_bytes = int(seg_elem.get('bytes', 0))
                    seg_number = int(seg_elem.get('number', 0))
                    message_id = seg_elem.text.strip() if seg_elem.text else ""

                    if message_id:
                        segments.append(NZBSegment(
                            message_id=message_id,
                            number=seg_number,
                            bytes=seg_bytes,
                            file_index=idx
                        ))
                        total_bytes += seg_bytes

                segments.sort(key=lambda s: s.number)

                files.append(NZBFile(
                    index=idx,
                    filename=filename,
                    size=total_bytes,
                    segments=segments,
                    obfuscated=obfuscated
                ))

        except Exception as e:
            logger.error(f"Failed to parse NZB: {e}")

        return files

    def _prepare_file(self, nzb_file: NZBFile) -> None:
        """Prepare output file for writing (disk or RAM mode)."""
        try:
            # Handle empty filename
            if not nzb_file.filename or not nzb_file.filename.strip():
                safe_name = f"file_{nzb_file.index:04d}"
                logger.warning(f"[PREPARE] Empty filename for file {nzb_file.index}, using: {safe_name}")
            else:
                safe_name = "".join(c if c.isalnum() or c in '.-_' else '_'
                                   for c in nzb_file.filename)

            # === RAM MODE: Store in memory instead of disk ===
            if self.ram_mode and self.ram_buffer is not None:
                # Use unique key with file index to avoid collisions when NZB has duplicate names
                # (common with obfuscated releases where all files have same subject)
                ram_key = f"{safe_name}__idx{nzb_file.index:04d}"

                # Store filename for later (original name without index suffix)
                self._file_paths[nzb_file.index] = Path(safe_name)  # Virtual path (just the name)

                # Create RAM buffer for this file with unique key
                # Pass safe_name as display_name so ram_file.filename has the clean name
                ram_file = self.ram_buffer.create_file(ram_key, max(nzb_file.size, 1), display_name=safe_name)
                if ram_file:
                    # Store reference: [ram_file, ram_key, size, write_count]
                    self._file_handles[nzb_file.index] = [ram_file, ram_key, nzb_file.size, 0]
                    self._file_locks[nzb_file.index] = threading.Lock()
                else:
                    logger.error(f"[RAM] Failed to create buffer for {ram_key}")
                return  # Skip disk file creation

            # Ensure safe_name is not empty after sanitization
            if not safe_name or not safe_name.strip():
                safe_name = f"file_{nzb_file.index:04d}"
                logger.warning(f"[PREPARE] Sanitized filename empty for file {nzb_file.index}, using: {safe_name}")

            # === HANDLE DUPLICATE FILENAMES ===
            # If this filename was already used, add index suffix
            original_safe_name = safe_name
            if safe_name in self._used_filenames:
                # Find unique name by adding index suffix
                base_name = safe_name
                ext = ""
                # Try to preserve extension
                if '.' in safe_name:
                    last_dot = safe_name.rfind('.')
                    base_name = safe_name[:last_dot]
                    ext = safe_name[last_dot:]

                counter = 1
                while safe_name in self._used_filenames:
                    safe_name = f"{base_name}_{nzb_file.index:03d}{ext}"
                    counter += 1
                    if counter > 1000:  # Safety limit
                        safe_name = f"file_{nzb_file.index:04d}{ext}"
                        break

                logger.info(f"[PREPARE] Duplicate filename '{original_safe_name}' -> using '{safe_name}' for file {nzb_file.index}")

            # Mark this filename as used
            self._used_filenames.add(safe_name)

            filepath = self.output_dir / safe_name
            logger.debug(f"[PREPARE] Creating file: {filepath}")

            # Delete existing file if present (from previous run, not current session)
            if filepath.exists():
                try:
                    filepath.unlink()
                    logger.debug(f"[PREPARE] Deleted existing file: {filepath}")
                except Exception as del_e:
                    # File locked - add unique suffix
                    base_name = safe_name
                    ext = ""
                    if '.' in safe_name:
                        last_dot = safe_name.rfind('.')
                        base_name = safe_name[:last_dot]
                        ext = safe_name[last_dot:]
                    safe_name = f"{base_name}_{nzb_file.index:03d}_new{ext}"
                    filepath = self.output_dir / safe_name
                    self._used_filenames.add(safe_name)
                    logger.warning(f"[PREPARE] Could not delete existing file, using: {safe_name}")

            # Handle zero-size files
            if nzb_file.size <= 0:
                logger.warning(f"[PREPARE] File {nzb_file.index} has size {nzb_file.size}, creating empty placeholder")
                # Create empty file but still add to handles so writer doesn't error
                with open(filepath, 'wb') as f:
                    pass
                self._file_paths[nzb_file.index] = filepath
                # Create dummy handle entry so writer can find it (no-op writes)
                self._file_handles[nzb_file.index] = [None, None, 0, 0]
                self._file_locks[nzb_file.index] = threading.Lock()
                return

            # Create sparse file with correct size
            with open(filepath, 'wb') as f:
                f.seek(nzb_file.size - 1)
                f.write(b'\x00')
            logger.debug(f"[PREPARE] Created sparse file: {filepath} ({nzb_file.size} bytes)")

            # Track file path for potential renaming (obfuscated files)
            self._file_paths[nzb_file.index] = filepath

            if self.write_through:
                # WriteThrough mode: bypass OS cache (DyMaxIO compatible)
                # Use Python file handle with unbuffered writes
                f = open(filepath, 'r+b', buffering=0)  # Unbuffered
                # Store: [file_handle, None (no mmap), size, write_count]
                self._file_handles[nzb_file.index] = [f, None, nzb_file.size, 0]
                logger.debug(f"[PREPARE] WriteThrough handle created for file {nzb_file.index}")
            else:
                # mmap mode: fast but uses more RAM
                flags = os.O_RDWR | getattr(os, 'O_BINARY', 0)
                fd = os.open(str(filepath), flags)

                mm = mmap.mmap(fd, nzb_file.size, access=mmap.ACCESS_WRITE)

                # Store: [fd, mmap, size, write_count]
                self._file_handles[nzb_file.index] = [fd, mm, nzb_file.size, 0]
                logger.debug(f"[PREPARE] mmap handle created for file {nzb_file.index}")

            self._file_locks[nzb_file.index] = threading.Lock()

        except Exception as e:
            logger.error(f"[PREPARE] EXCEPTION for file {nzb_file.index} '{nzb_file.filename}': {e}")
            import traceback
            logger.error(f"[PREPARE] Traceback: {traceback.format_exc()}")

    def _finalize_file(self, nzb_file: NZBFile) -> None:
        """Close file handles, flush, and rename if obfuscated."""
        file_idx = nzb_file.index

        if file_idx in self._file_handles:
            handle = self._file_handles[file_idx]
            try:
                if self.ram_mode:
                    # RAM mode: no disk operations needed
                    # Data stays in RamBuffer, just clean up tracking
                    ram_file = handle[0]
                    logger.debug(f"[FINALIZE-RAM] File complete: {ram_file.filename} ({ram_file.size} bytes)")
                elif self.write_through:
                    # WriteThrough mode: just close file handle
                    f = handle[0]
                    f.flush()
                    f.close()
                else:
                    # mmap mode
                    fd, mm, _, _ = handle
                    if mm:
                        mm.flush()
                        mm.close()
                    os.close(fd)
            except:
                pass
            del self._file_handles[file_idx]

        # === RENAME OBFUSCATED FILE TO REAL NAME ===
        if file_idx in self._renamed_files and file_idx in self._file_paths:
            yenc_name = self._renamed_files[file_idx]
            old_path = self._file_paths[file_idx]
            current_name = old_path.name

            # Determine which name is "real" (has proper extension, not random string)
            def looks_like_real_filename(name: str) -> bool:
                """Check if filename looks real (has known extension)."""
                if not name:
                    return False
                # Known archive/media extensions = real filename
                extensions = {'.rar', '.r00', '.r01', '.r02', '.zip', '.7z', '.par2',
                             '.nfo', '.sfv', '.mkv', '.avi', '.mp4', '.iso', '.nzb',
                             '.txt', '.exe', '.msi', '.part1.rar', '.part2.rar', '.part3.rar'}
                name_lower = name.lower()

                # Check known extensions first
                for ext in extensions:
                    if name_lower.endswith(ext):
                        return True

                # Check .rXX pattern (split RAR)
                import re
                if re.search(r'\.[r]\d{2}$', name_lower):
                    return True

                # Check .partXX.rar pattern
                if re.search(r'\.part\d+\.rar$', name_lower):
                    return True

                # No known extension = probably obfuscated
                return False

            current_is_real = looks_like_real_filename(current_name)
            yenc_is_real = looks_like_real_filename(yenc_name)

            logger.debug(f"Rename check: current='{current_name}' (real={current_is_real}), yenc='{yenc_name}' (real={yenc_is_real})")

            # Only rename if current name is obfuscated AND yEnc name is real
            if not current_is_real and yenc_is_real:
                # Clean the real filename
                safe_name = "".join(c if c.isalnum() or c in '.-_() ' else '_'
                                   for c in yenc_name)

                if self.ram_mode and self.ram_buffer is not None:
                    # RAM mode: update filename in RamBuffer
                    try:
                        # Get the actual RAM key from _file_handles (includes index suffix)
                        if file_idx in self._file_handles:
                            old_name = self._file_handles[file_idx][1]  # ram_key with index
                        else:
                            old_name = str(old_path)
                        ram_file = self.ram_buffer.get_file(old_name)
                        if ram_file:
                            # Update the filename in place
                            ram_file.filename = safe_name
                            # Move to new key in the buffer dict
                            self.ram_buffer._files[safe_name] = ram_file
                            if old_name in self.ram_buffer._files:
                                del self.ram_buffer._files[old_name]
                            # Update both references
                            self._file_paths[file_idx] = Path(safe_name)
                            self._file_handles[file_idx][1] = safe_name  # Update handle reference
                            logger.info(f"Renamed (RAM): {old_name} -> {safe_name}")
                    except Exception as e:
                        logger.warning(f"Failed to rename in RAM {old_path}: {e}")
                else:
                    # Disk mode: rename file on filesystem
                    new_path = self.output_dir / safe_name
                    try:
                        # Handle duplicate names
                        if new_path.exists() and new_path != old_path:
                            base = new_path.stem
                            ext = new_path.suffix
                            counter = 1
                            while new_path.exists():
                                new_path = self.output_dir / f"{base}_{counter}{ext}"
                                counter += 1

                        if old_path.exists():
                            old_path.rename(new_path)
                            self._file_paths[file_idx] = new_path
                            logger.info(f"Renamed: {old_path.name} -> {new_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to rename {old_path.name}: {e}")

        self._stats.files_completed += 1

        # === STREAMING POST-PROCESSING CALLBACK ===
        # Notify listeners that a file is complete (for early PAR2 verification)
        if self.on_file_complete:
            try:
                file_path = self._file_paths.get(file_idx)
                if self.ram_mode:
                    # RAM mode: pass virtual path (filename only)
                    if file_path:
                        self.on_file_complete(file_path, nzb_file.filename)
                elif file_path and file_path.exists():
                    # Disk mode: pass real path
                    self.on_file_complete(file_path, nzb_file.filename)
            except Exception as e:
                logger.debug(f"on_file_complete callback error: {e}")

    def close(self) -> None:
        """Clean up."""
        self._running = False
        for conn in self._connections:
            conn.close()
        self._connections.clear()


# Entry point for testing
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="DLER Turbo V2 - Separated Pipeline")
    parser.add_argument("nzb", help="NZB file to download")
    parser.add_argument("-c", "--connections", type=int, default=50, help="Download connections")
    parser.add_argument("-d", "--decoders", type=int, default=0, help="Decoder threads (0=auto)")
    parser.add_argument("-w", "--writers", type=int, default=4, help="Writer threads")
    parser.add_argument("-o", "--output", default="downloads", help="Output directory")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        server = config.get('server', {})
    else:
        print("No config.json found!")
        sys.exit(1)

    server_config = ServerConfig(
        host=server['host'],
        port=server.get('port', 563),
        username=server.get('username', ''),
        password=server.get('password', ''),
        connections=args.connections
    )

    engine = TurboEngineV2(
        server_config=server_config,
        output_dir=Path(args.output),
        download_threads=args.connections,
        decoder_threads=args.decoders,
        writer_threads=args.writers
    )

    try:
        success = engine.download_nzb(Path(args.nzb))
        sys.exit(0 if success else 1)
    finally:
        engine.close()
