"""
Post-Processor Module
=====================

Ultra-fast PAR2 verification and archive extraction.

Features:
- Multi-threaded PAR2 verification with par2cmdline-turbo
- Parallel archive extraction with 7-Zip
- Password extraction from NZB metadata
- Smart file detection (RAR, ZIP, 7z)
- Progressive status reporting
- Automatic cleanup of temp files

Performance targets:
- PAR2 verification: ~500 MB/s (par2cmdline-turbo with AVX2)
- Extraction: Limited by disk I/O (~200+ MB/s on NVMe)
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import logging
import threading
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Windows: hide console windows for subprocess calls
if sys.platform == 'win32':
    SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW
else:
    SUBPROCESS_FLAGS = 0


def _extract_filename_from_subject(subject: str) -> Optional[str]:
    """
    Extract filename from NZB subject line.

    Subject formats vary but commonly include:
    - "release name" filename.ext yEnc (1/10)
    - [group] release - "filename.ext" yEnc
    - release.name.part01.rar yEnc (1/100)
    - [#a]release.name.part01.rar yEnc
    """
    if not subject:
        return None

    # Try to find filename in quotes
    quoted = re.search(r'"([^"]+\.[a-zA-Z0-9]{2,7})"', subject)
    if quoted:
        return quoted.group(1)

    # Try to find filename before yEnc marker
    yenc_match = re.search(r'([^\s"\[\]]+\.[a-zA-Z0-9]{2,7})\s*yEnc', subject, re.IGNORECASE)
    if yenc_match:
        return yenc_match.group(1)

    # Try to find part/vol pattern (common in RAR sets)
    part_match = re.search(r'([^\s"\[\]]+\.(?:part\d+|vol\d+[+\d]*)\.[a-zA-Z0-9]{2,7})', subject, re.IGNORECASE)
    if part_match:
        return part_match.group(1)

    # Try to find any filename pattern with common extensions
    file_match = re.search(r'([^\s"\[\]<>|]+\.(?:rar|r\d{2,3}|zip|7z|par2|nfo|sfv|nzb|mkv|avi|mp4|iso|m2ts|ts))', subject, re.IGNORECASE)
    if file_match:
        return file_match.group(1)

    return None


def _normalize_filename(filename: str) -> str:
    """
    Normalize filename for comparison.

    Handles common Usenet filename variations:
    - Spaces ↔ underscores
    - Parentheses () → underscores
    - Multiple underscores → single underscore
    """
    normalized = filename.lower()
    # Replace spaces and parentheses with underscores
    normalized = normalized.replace(' ', '_')
    normalized = normalized.replace('(', '_')
    normalized = normalized.replace(')', '_')
    # Collapse multiple underscores
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    return normalized


class PostProcessStatus(Enum):
    """Post-processing status."""
    PENDING = "pending"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WarningType(Enum):
    """Types of warnings for user notification."""
    ANTIVIRUS_BLOCK = "antivirus_block"      # AV blocked file operations
    PAR2_REPAIR_NEEDED = "par2_repair"       # Files needed repair
    INCOMPLETE_DOWNLOAD = "incomplete"        # Missing segments (DMCA?)
    PASSWORD_REQUIRED = "password_required"   # Archive needs password
    EXTRACTION_PARTIAL = "extraction_partial" # Some files failed to extract
    DISK_SPACE_LOW = "disk_space"            # Low disk space warning
    CLEANUP_SKIPPED = "cleanup_skipped"      # Cleanup skipped (NZB parsing failed)


@dataclass
class PostProcessResult:
    """Result of post-processing."""
    success: bool
    status: PostProcessStatus
    message: str
    par2_verified: bool = False
    par2_repaired: bool = False
    files_extracted: int = 0
    extract_path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[Tuple[WarningType, str]] = field(default_factory=list)
    duration_seconds: float = 0.0
    antivirus_blocked_files: List[str] = field(default_factory=list)
    # Performance metrics
    extraction_duration_seconds: float = 0.0
    extracted_bytes: int = 0

    @property
    def extraction_speed_mbs(self) -> float:
        """Extraction speed in MB/s."""
        if self.extraction_duration_seconds > 0 and self.extracted_bytes > 0:
            return (self.extracted_bytes / (1024 * 1024)) / self.extraction_duration_seconds
        return 0.0


@dataclass
class NZBMetadata:
    """Metadata extracted from NZB file."""
    title: str = ""
    password: str = ""
    category: str = ""
    group: str = ""
    filenames: List[str] = field(default_factory=list)  # List of files in NZB


class MoveResult:
    """Result of a file move operation."""
    __slots__ = ('success', 'copied', 'av_blocked', 'error')

    def __init__(self, success: bool = False, copied: bool = False,
                 av_blocked: bool = False, error: str = ""):
        self.success = success
        self.copied = copied
        self.av_blocked = av_blocked
        self.error = error


def robust_move_file(src: Path, dest: Path, max_retries: int = 3, delay: float = 0.5) -> MoveResult:
    """
    Robustly move a file with retry logic for antivirus locks.

    Uses copy + delete approach which is more reliable than move
    when Windows Defender or other AV is scanning the file.

    Args:
        src: Source file path
        dest: Destination file path
        max_retries: Number of retry attempts
        delay: Delay between retries (seconds)

    Returns:
        MoveResult with details about what happened
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = MoveResult()

    for attempt in range(max_retries):
        try:
            # Try copy first (works even if source is locked for delete)
            shutil.copy2(str(src), str(dest))
            result.copied = True

            # Try to delete source (may fail if AV is holding it)
            try:
                src.unlink()
            except PermissionError:
                # File copied but couldn't delete source - AV lock
                result.av_blocked = True
                logger.debug(f"Copied {src.name} but couldn't delete source (AV lock?)")

            result.success = True
            return result

        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.debug(f"Retry {attempt + 1}/{max_retries} for {src.name}: {e}")
                time.sleep(delay)
            else:
                result.av_blocked = True
                result.error = str(e)
                logger.warning(f"Failed to move {src.name} after {max_retries} attempts: {e}")
                return result
        except Exception as e:
            result.error = str(e)
            logger.warning(f"Error moving {src.name}: {e}")
            return result

    return result


@dataclass
class BatchMoveResult:
    """Result of batch file move operation."""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    av_blocked: int = 0
    copied_only: int = 0  # Copied but source not deleted (AV lock)
    results: List[MoveResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Percentage of successful moves."""
        return self.successful / max(1, self.total_files)


def batch_move_files(
    file_pairs: List[Tuple[Path, Path]],
    max_workers: int = 8,
    max_retries: int = 3,
    delay: float = 0.5,
    on_progress: Optional[Callable[[int, int], None]] = None
) -> BatchMoveResult:
    """
    Move multiple files in parallel using ThreadPoolExecutor.

    Significantly faster than sequential moves for large file batches.
    Maintains AV lock retry logic per file.

    Args:
        file_pairs: List of (source, destination) path tuples
        max_workers: Number of parallel workers (default: 8)
        max_retries: Retries per file for AV locks
        delay: Delay between retries (seconds)
        on_progress: Callback (completed, total) for progress updates

    Returns:
        BatchMoveResult with aggregated statistics
    """
    if not file_pairs:
        return BatchMoveResult()

    start_time = time.time()
    num_workers = min(max_workers, len(file_pairs))
    results: List[Optional[MoveResult]] = [None] * len(file_pairs)
    completed_count = [0]  # Mutable for closure
    lock = threading.Lock()

    def move_with_index(args: Tuple[int, Tuple[Path, Path]]) -> Tuple[int, MoveResult]:
        idx, (src, dest) = args
        result = robust_move_file(src, dest, max_retries, delay)

        with lock:
            completed_count[0] += 1
            if on_progress:
                try:
                    on_progress(completed_count[0], len(file_pairs))
                except Exception:
                    pass

        return idx, result

    # Parallel execution
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(move_with_index, (i, pair))
            for i, pair in enumerate(file_pairs)
        ]

        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                logger.error(f"Batch move worker error: {e}")

    # Aggregate results
    batch_result = BatchMoveResult(
        total_files=len(file_pairs),
        duration_seconds=time.time() - start_time
    )

    for r in results:
        if r is None:
            batch_result.failed += 1
            continue

        batch_result.results.append(r)

        if r.success:
            batch_result.successful += 1
            if r.av_blocked:
                batch_result.copied_only += 1
        elif r.av_blocked:
            batch_result.av_blocked += 1
            batch_result.failed += 1
        else:
            batch_result.failed += 1

    logger.info(
        f"Batch move: {batch_result.successful}/{batch_result.total_files} files "
        f"in {batch_result.duration_seconds:.1f}s "
        f"({batch_result.av_blocked} AV-blocked)"
    )

    return batch_result


@dataclass
class ExtractionResult:
    """Result of a single archive extraction."""
    archive: Path
    success: bool
    files_extracted: int
    message: str
    duration: float


class ParallelExtractor:
    """
    Parallel archive extraction with resource balancing.

    Extracts multiple archives concurrently using ThreadPoolExecutor,
    while balancing 7z thread usage to avoid resource contention.
    """

    def __init__(
        self,
        sevenzip_path: str,
        max_parallel: int = 3,
        threads_per_extraction: int = 4,
        on_progress: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize parallel extractor.

        Args:
            sevenzip_path: Path to 7z executable
            max_parallel: Max concurrent extraction processes (2-4 recommended)
            threads_per_extraction: Threads per 7z instance (-mmt flag)
            on_progress: Progress callback (message, percent)
        """
        self.sevenzip_path = sevenzip_path
        self.max_parallel = max_parallel
        self.threads_per_extraction = threads_per_extraction
        self.on_progress = on_progress
        self._lock = threading.Lock()
        self._completed = 0
        self._total = 0

    def _report_progress(self, message: str, percent: float) -> None:
        """Report progress to callback."""
        if self.on_progress:
            try:
                self.on_progress(message, percent)
            except Exception:
                pass

    def _extract_single(
        self,
        archive: Path,
        output_dir: Path,
        password: Optional[str],
        index: int
    ) -> ExtractionResult:
        """
        Extract single archive (worker function).

        Args:
            archive: Archive file to extract
            output_dir: Output directory
            password: Optional password
            index: Archive index for progress

        Returns:
            ExtractionResult with details
        """
        start_time = time.time()

        try:
            # Build 7z command with reduced thread count for parallel execution
            cmd = [
                self.sevenzip_path,
                'x',  # Extract with full paths
                '-y',  # Yes to all prompts
                '-bb0',  # Less output
                '-bd',  # No progress indicator
                f'-o{output_dir}',  # Output directory
                f'-mmt={self.threads_per_extraction}',  # Reduced threads
            ]

            # Add password
            if password:
                cmd.append(f'-p{password}')
            else:
                cmd.append('-p')  # Empty password (skip prompts)

            cmd.append(str(archive))

            logger.debug(f"Parallel extract [{index}]: {archive.name}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(archive.parent),
                timeout=7200,  # 2 hours max
                creationflags=SUBPROCESS_FLAGS
            )

            elapsed = time.time() - start_time

            # Update progress
            with self._lock:
                self._completed += 1
                pct = (self._completed / max(1, self._total)) * 100
                self._report_progress(
                    f"Extracting: {self._completed}/{self._total}",
                    pct
                )

            if result.returncode == 0:
                # Count extracted files (estimate based on output)
                files_count = result.stdout.count('Extracting') if result.stdout else 1
                return ExtractionResult(
                    archive=archive,
                    success=True,
                    files_extracted=max(1, files_count),
                    message=f"OK ({elapsed:.1f}s)",
                    duration=elapsed
                )
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                if 'Wrong password' in error_msg or 'password' in error_msg.lower():
                    return ExtractionResult(
                        archive=archive,
                        success=False,
                        files_extracted=0,
                        message="Wrong password",
                        duration=elapsed
                    )
                return ExtractionResult(
                    archive=archive,
                    success=False,
                    files_extracted=0,
                    message=f"Failed: {error_msg[:100]}",
                    duration=elapsed
                )

        except subprocess.TimeoutExpired:
            return ExtractionResult(
                archive=archive,
                success=False,
                files_extracted=0,
                message="Timeout",
                duration=time.time() - start_time
            )
        except Exception as e:
            return ExtractionResult(
                archive=archive,
                success=False,
                files_extracted=0,
                message=str(e),
                duration=time.time() - start_time
            )

    def extract_parallel(
        self,
        archives: List[Path],
        output_dir: Path,
        password: Optional[str] = None
    ) -> List[ExtractionResult]:
        """
        Extract multiple archives in parallel.

        Args:
            archives: List of archive files
            output_dir: Output directory (shared for all)
            password: Optional password (same for all)

        Returns:
            List of ExtractionResult for each archive
        """
        if not archives:
            return []

        self._completed = 0
        self._total = len(archives)

        output_dir.mkdir(parents=True, exist_ok=True)

        num_workers = min(self.max_parallel, len(archives))
        results: List[Optional[ExtractionResult]] = [None] * len(archives)

        logger.info(
            f"Parallel extraction: {len(archives)} archives, "
            f"{num_workers} workers, {self.threads_per_extraction} threads each"
        )
        self._report_progress(f"Extracting {len(archives)} archives...", 0)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_single, archive, output_dir, password, i
                ): i
                for i, archive in enumerate(archives)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Parallel extract worker error: {e}")
                    results[idx] = ExtractionResult(
                        archive=archives[idx],
                        success=False,
                        files_extracted=0,
                        message=str(e),
                        duration=0
                    )

        # Log summary
        successful = sum(1 for r in results if r and r.success)
        total_time = sum(r.duration for r in results if r)
        logger.info(
            f"Parallel extraction complete: {successful}/{len(archives)} OK, "
            f"total time {total_time:.1f}s"
        )

        return [r for r in results if r is not None]

    @staticmethod
    def calculate_optimal_params(ram_gb: float = 0) -> Tuple[int, int]:
        """
        Calculate optimal parallel extraction parameters.

        Args:
            ram_gb: System RAM in GB (0 = auto-detect)

        Returns:
            (max_parallel, threads_per_extraction)
        """
        cpu_count = os.cpu_count() or 8

        # Auto-detect RAM
        if ram_gb <= 0:
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            except ImportError:
                ram_gb = 16  # Conservative default

        # Scale based on RAM and CPU
        if ram_gb >= 48:
            return 4, max(4, cpu_count // 4)
        elif ram_gb >= 24:
            return 3, max(4, cpu_count // 3)
        elif ram_gb >= 12:
            return 2, max(2, cpu_count // 4)
        else:
            return 1, max(2, cpu_count // 2)  # Sequential fallback


def detect_hardware_config() -> Dict[str, any]:
    """
    Auto-detect hardware configuration for optimal settings.

    Returns:
        Dict with cpu_count, ram_gb, recommended settings
    """
    cpu_count = os.cpu_count() or 8

    # Detect RAM
    ram_gb = 16  # Default
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    # Calculate optimal settings based on hardware
    if ram_gb >= 48 and cpu_count >= 16:
        # High-end system (like user's 64GB + many cores)
        config = {
            'parallel_extractions': 4,
            'threads_per_extraction': max(6, cpu_count // 4),
            'parallel_move_threads': min(16, cpu_count),
            'par2_threads': cpu_count,
            'profile': 'high_end'
        }
    elif ram_gb >= 24 and cpu_count >= 8:
        # Mid-high system
        config = {
            'parallel_extractions': 3,
            'threads_per_extraction': max(4, cpu_count // 3),
            'parallel_move_threads': min(12, cpu_count),
            'par2_threads': cpu_count,
            'profile': 'mid_high'
        }
    elif ram_gb >= 12 and cpu_count >= 4:
        # Mid system
        config = {
            'parallel_extractions': 2,
            'threads_per_extraction': max(2, cpu_count // 4),
            'parallel_move_threads': 8,
            'par2_threads': min(8, cpu_count),
            'profile': 'mid'
        }
    else:
        # Low-end system
        config = {
            'parallel_extractions': 1,
            'threads_per_extraction': max(2, cpu_count // 2),
            'parallel_move_threads': 4,
            'par2_threads': min(4, cpu_count),
            'profile': 'low_end'
        }

    config['cpu_count'] = cpu_count
    config['ram_gb'] = ram_gb

    logger.info(f"Hardware detected: {cpu_count} cores, {ram_gb:.1f} GB RAM → profile: {config['profile']}")

    return config


def are_on_same_drive(path1: Path, path2: Path) -> bool:
    """Check if two paths are on the same drive (Windows) or mount point."""
    try:
        if sys.platform == 'win32':
            # Windows: compare drive letters
            drive1 = Path(path1).resolve().drive.upper()
            drive2 = Path(path2).resolve().drive.upper()
            return drive1 == drive2
        else:
            # Unix: compare mount points using os.stat
            import os
            return os.stat(path1).st_dev == os.stat(path2).st_dev
    except Exception:
        return True  # Assume same drive if can't determine


class StreamingPostProcessor:
    """
    Post-processor that starts PAR2 verification during download.

    Monitors file completion events from the download engine and
    starts PAR2 verification as soon as PAR2 files are complete,
    enabling early detection of missing or corrupted segments.

    Usage:
        streaming_pp = StreamingPostProcessor(
            download_dir=Path("downloads"),
            on_early_issue=lambda msg: print(f"Warning: {msg}")
        )

        # Register with engine
        engine.on_file_complete = streaming_pp.on_file_complete

        # After download, check for early result
        early_result = streaming_pp.get_early_result()
    """

    def __init__(
        self,
        download_dir: Path,
        par2_path: Optional[str] = None,
        threads: int = 0,
        on_early_issue: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize streaming post-processor.

        Args:
            download_dir: Directory where files are being downloaded
            par2_path: Path to PAR2 executable (auto-detect if None)
            threads: PAR2 threads (0 = auto)
            on_early_issue: Callback when PAR2 detects issues early
            on_progress: Progress callback (message, percent)
        """
        self.download_dir = Path(download_dir)
        self.par2_path = par2_path or self._find_par2()
        self.threads = threads or os.cpu_count() or 8
        self.on_early_issue = on_early_issue
        self.on_progress = on_progress

        # State tracking
        self._completed_files: Set[Path] = set()
        self._par2_main_file: Optional[Path] = None
        self._par2_started = False
        self._par2_result: Optional[Tuple[bool, bool, str]] = None
        self._par2_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        logger.info(f"StreamingPostProcessor initialized: PAR2={self.par2_path or 'NOT FOUND'}")

    def _find_par2(self) -> Optional[str]:
        """Find PAR2 executable."""
        # Check bundled tools first
        if hasattr(sys, '_MEIPASS'):
            bundled = Path(sys._MEIPASS) / 'tools' / 'par2j64.exe'
        else:
            bundled = Path(__file__).parent.parent.parent / 'tools' / 'par2j64.exe'

        if bundled.exists():
            return str(bundled)

        # Check common paths
        paths = [
            r"par2.exe",
            r"C:\Program Files\MultiPar\par2j64.exe",
            r"C:\Program Files (x86)\MultiPar\par2j64.exe",
        ]

        for path in paths:
            if not os.path.isabs(path):
                result = shutil.which(path)
                if result:
                    return result
            elif Path(path).exists():
                return path

        return None

    def _is_main_par2(self, file_path: Path) -> bool:
        """Check if file is main PAR2 (not volume)."""
        name = file_path.name.lower()
        return name.endswith('.par2') and '.vol' not in name

    def on_file_complete(self, file_path: Path, filename: str) -> None:
        """
        Called when a file finishes downloading.

        This is the callback registered with TurboEngineV2.on_file_complete.

        Args:
            file_path: Path to completed file
            filename: Original filename from NZB
        """
        with self._lock:
            self._completed_files.add(file_path)

        # Check if this is a main PAR2 file
        if self._is_main_par2(file_path) and not self._par2_started:
            logger.info(f"Main PAR2 file complete: {file_path.name}")
            self._start_early_par2(file_path)

    def _start_early_par2(self, par2_file: Path) -> None:
        """Start PAR2 verification in background thread."""
        if not self.par2_path:
            logger.warning("PAR2 not available for streaming verification")
            return

        with self._lock:
            if self._par2_started:
                return
            self._par2_started = True
            self._par2_main_file = par2_file

        def verify_thread():
            try:
                logger.info(f"Starting early PAR2 verification: {par2_file.name}")

                if self.on_progress:
                    self.on_progress("Early PAR2 verification...", -1)

                # Run PAR2 verify (NOT repair - just check)
                result = self._run_par2_verify(par2_file)

                with self._lock:
                    self._par2_result = result

                verified, _, message = result

                if verified:
                    logger.info(f"Early PAR2 OK: {message}")
                else:
                    logger.warning(f"Early PAR2 issue: {message}")
                    if self.on_early_issue:
                        try:
                            self.on_early_issue(f"PAR2 issue detected: {message}")
                        except Exception:
                            pass

            except Exception as e:
                logger.error(f"Early PAR2 error: {e}")
                with self._lock:
                    self._par2_result = (False, False, str(e))

        self._par2_thread = threading.Thread(
            target=verify_thread,
            name="StreamingPAR2",
            daemon=True
        )
        self._par2_thread.start()

    def _run_par2_verify(self, par2_file: Path) -> Tuple[bool, bool, str]:
        """
        Run PAR2 verification only (no repair).

        Returns:
            (verified_ok, repaired, message)
        """
        try:
            cmd = [
                self.par2_path,
                'verify',  # Verify only, no repair
                '-q',  # Quiet
                f'-t{self.threads}',
                str(par2_file)
            ]

            logger.debug(f"Running early PAR2: {' '.join(cmd)}")
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(par2_file.parent),
                timeout=1800,  # 30 min max for early check
                creationflags=SUBPROCESS_FLAGS
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                return True, False, f"Verified OK ({elapsed:.1f}s)"
            else:
                # Check output for details
                output = result.stderr or result.stdout or ""

                # Missing blocks can be recovered with more PAR2 volumes
                if 'repair is required' in output.lower() or result.returncode == 1:
                    return False, False, f"Repair needed ({elapsed:.1f}s)"
                elif 'repair is not possible' in output.lower():
                    return False, False, f"Repair not possible - missing data"
                else:
                    return False, False, f"Verify failed: {output[:100]}"

        except subprocess.TimeoutExpired:
            return False, False, "Verification timeout"
        except Exception as e:
            return False, False, str(e)

    def get_early_result(self) -> Optional[Tuple[bool, bool, str]]:
        """
        Get result of early PAR2 verification if available.

        Returns:
            (verified_ok, repaired, message) or None if not yet complete
        """
        with self._lock:
            return self._par2_result

    def wait_for_par2(self, timeout: float = 60) -> bool:
        """
        Wait for early PAR2 verification to complete.

        Args:
            timeout: Max seconds to wait

        Returns:
            True if PAR2 completed (check get_early_result for status)
        """
        if self._par2_thread:
            self._par2_thread.join(timeout=timeout)
            return not self._par2_thread.is_alive()
        return True

    def is_par2_complete(self) -> bool:
        """Check if early PAR2 verification has finished."""
        with self._lock:
            return self._par2_result is not None

    def get_completed_files(self) -> Set[Path]:
        """Get set of completed file paths."""
        with self._lock:
            return self._completed_files.copy()


class PostProcessor:
    """
    Ultra-fast post-processor for downloaded NZB content.

    Handles:
    - PAR2 verification and repair
    - Archive extraction (RAR, ZIP, 7z)
    - Password handling from NZB metadata

    Usage:
        processor = PostProcessor(
            download_dir=Path("downloads"),
            extract_dir=Path("extracted"),
            par2_path="par2.exe",
            sevenzip_path="7z.exe"
        )
        result = processor.process(nzb_path)
    """

    @staticmethod
    def _get_bundled_tools_dir() -> Path:
        """Get path to bundled tools directory (for PyInstaller builds)."""
        # PyInstaller sets sys._MEIPASS for bundled apps
        if hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS) / 'tools'
        # Development: check relative to this file
        return Path(__file__).parent.parent.parent / 'tools'

    @classmethod
    def _get_tool_paths(cls) -> tuple:
        """Get search paths for tools, bundled first."""
        bundled = cls._get_bundled_tools_dir()

        sevenzip_paths = [
            str(bundled / '7z.exe'),  # Bundled (priority!)
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
            r"7z.exe",  # In PATH
        ]

        par2_paths = [
            str(bundled / 'par2j64.exe'),  # Bundled (priority!)
            r"par2.exe",  # In PATH
            r"C:\Users\Symphoenix\AppData\Local\MultiPar\par2j64.exe",
            r"C:\Program Files\Spotnet\SABnzbd\win\par2\par2.exe",
            r"C:\Program Files\MultiPar\par2j64.exe",
            r"C:\Program Files (x86)\MultiPar\par2j64.exe",
            r"C:\Program Files\par2cmdline\par2.exe",
        ]

        return sevenzip_paths, par2_paths

    # Legacy class attributes (now generated dynamically)
    SEVENZIP_PATHS = []  # Will use _get_tool_paths()
    PAR2_PATHS = []  # Will use _get_tool_paths()

    # Archive extensions
    ARCHIVE_EXTENSIONS = {'.rar', '.zip', '.7z', '.tar', '.gz', '.bz2', '.xz'}

    def __init__(
        self,
        download_dir: Path,
        extract_dir: Optional[Path] = None,
        par2_path: Optional[str] = None,
        sevenzip_path: Optional[str] = None,
        threads: int = 0,  # 0 = auto
        cleanup_after_extract: bool = True,
        on_progress: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize post-processor.

        Args:
            download_dir: Directory containing downloaded files
            extract_dir: Directory for extracted files (default: download_dir/extracted)
            par2_path: Path to par2 executable (auto-detect if None)
            sevenzip_path: Path to 7z executable (auto-detect if None)
            threads: Number of threads (0 = auto = CPU count)
            cleanup_after_extract: Delete archives after successful extraction
            on_progress: Callback for progress updates (message, percent)
        """
        self.download_dir = Path(download_dir)
        self.extract_dir = Path(extract_dir) if extract_dir else self.download_dir / "extracted"
        self.threads = threads or os.cpu_count() or 8
        self.cleanup_after_extract = cleanup_after_extract
        self.on_progress = on_progress

        # Find executables (bundled tools have priority)
        sevenzip_paths, par2_paths = self._get_tool_paths()
        self.par2_path = self._find_executable(par2_path, par2_paths)
        self.sevenzip_path = self._find_executable(sevenzip_path, sevenzip_paths)

        # Ensure extract directory exists
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._stop_requested = False
        self._current_status = PostProcessStatus.PENDING
        self._av_blocked_files: Set[str] = set()  # Track AV-blocked files

        logger.info(f"PostProcessor initialized:")
        logger.info(f"  PAR2: {self.par2_path or 'NOT FOUND'}")
        logger.info(f"  7-Zip: {self.sevenzip_path or 'NOT FOUND'}")
        logger.info(f"  Threads: {self.threads}")

    def _find_executable(self, user_path: Optional[str], default_paths: List[str]) -> Optional[str]:
        """Find executable in user path or default locations."""
        if user_path and Path(user_path).exists():
            return user_path

        for path in default_paths:
            # Check if in PATH
            if not os.path.isabs(path):
                result = shutil.which(path)
                if result:
                    return result
            # Check absolute path
            elif Path(path).exists():
                return path

        return None

    def _report_progress(self, message: str, percent: float = -1) -> None:
        """Report progress to callback."""
        if self.on_progress:
            try:
                self.on_progress(message, percent)
            except:
                pass

    @staticmethod
    def parse_nzb_metadata(nzb_path: Path) -> NZBMetadata:
        """
        Parse NZB file for metadata including password.

        Looks for:
        - <meta type="password">...</meta>
        - <meta type="name">...</meta>
        - <meta type="category">...</meta>
        """
        metadata = NZBMetadata()

        try:
            tree = ET.parse(nzb_path)
            root = tree.getroot()

            # Handle namespace
            ns = {'nzb': 'http://www.newzbin.com/DTD/2003/nzb'}

            # Try with namespace
            for meta in root.findall('.//nzb:meta', ns):
                meta_type = meta.get('type', '').lower()
                value = meta.text.strip() if meta.text else ''

                if meta_type == 'password':
                    metadata.password = value
                elif meta_type == 'name':
                    metadata.title = value
                elif meta_type == 'category':
                    metadata.category = value

            # Try without namespace (some NZBs don't use it)
            for meta in root.findall('.//meta'):
                meta_type = meta.get('type', '').lower()
                value = meta.text.strip() if meta.text else ''

                if meta_type == 'password' and not metadata.password:
                    metadata.password = value
                elif meta_type == 'name' and not metadata.title:
                    metadata.title = value
                elif meta_type == 'category' and not metadata.category:
                    metadata.category = value

            # Extract title from filename if not in metadata
            if not metadata.title:
                metadata.title = nzb_path.stem

            # Extract filenames from NZB <file> elements
            # Try with namespace first
            file_elements_ns = root.findall('.//nzb:file', ns)
            logger.debug(f"Found {len(file_elements_ns)} <file> elements with namespace")

            for file_elem in file_elements_ns:
                subject = file_elem.get('subject', '')
                # Extract filename from subject (usually in quotes or after yEnc)
                filename = _extract_filename_from_subject(subject)
                if filename:
                    metadata.filenames.append(filename)
                elif subject:
                    logger.debug(f"Could not extract filename from subject: {subject[:80]}...")

            # Try without namespace
            if not metadata.filenames:
                file_elements_no_ns = root.findall('.//file')
                logger.debug(f"Found {len(file_elements_no_ns)} <file> elements without namespace")

                for file_elem in file_elements_no_ns:
                    subject = file_elem.get('subject', '')
                    filename = _extract_filename_from_subject(subject)
                    if filename:
                        metadata.filenames.append(filename)
                    elif subject:
                        logger.debug(f"Could not extract filename from subject: {subject[:80]}...")

            logger.info(f"NZB Metadata: title={metadata.title}, password={'***' if metadata.password else 'None'}, files={len(metadata.filenames)}")

            # Debug: show first few filenames extracted
            if metadata.filenames:
                sample = metadata.filenames[:5]
                logger.info(f"NZB filenames sample: {sample}")

        except Exception as e:
            logger.error(f"Failed to parse NZB metadata: {e}")
            metadata.title = nzb_path.stem

        return metadata

    def find_par2_files(self, directory: Optional[Path] = None) -> List[Path]:
        """Find all PAR2 files in directory."""
        search_dir = directory or self.download_dir
        par2_files = []

        for ext in ['.par2', '.PAR2']:
            par2_files.extend(search_dir.glob(f'*{ext}'))

        # Sort: main par2 first, then volumes
        def sort_key(p: Path) -> tuple:
            name = p.name.lower()
            # Main par2 file (no vol prefix)
            if '.vol' not in name:
                return (0, name)
            return (1, name)

        return sorted(par2_files, key=sort_key)

    def find_archives(self, directory: Optional[Path] = None) -> List[Path]:
        """Find all archive files in directory."""
        search_dir = directory or self.download_dir
        archives = []

        for file in search_dir.iterdir():
            if file.is_file() and file.suffix.lower() in self.ARCHIVE_EXTENSIONS:
                archives.append(file)
            # RAR split archives (.r00, .r01, etc.)
            elif file.is_file() and re.match(r'\.(r|z)\d{2,}$', file.suffix.lower()):
                archives.append(file)

        # Filter to only first part of split archives
        first_parts = []
        seen_bases = set()

        for archive in sorted(archives):
            name = archive.name.lower()

            # RAR multi-part: .part01.rar or .rar + .r00
            if '.part' in name:
                # Get base name before .part
                base = re.sub(r'\.part\d+\.rar$', '', name, flags=re.IGNORECASE)
                if base not in seen_bases:
                    seen_bases.add(base)
                    first_parts.append(archive)
            elif archive.suffix.lower() == '.rar':
                # Check if this is the main .rar file
                base = archive.stem.lower()
                if base not in seen_bases:
                    seen_bases.add(base)
                    first_parts.append(archive)
            elif archive.suffix.lower() in {'.zip', '.7z', '.tar', '.gz'}:
                first_parts.append(archive)

        return first_parts

    def verify_par2(self, par2_file: Path, repair: bool = True) -> Tuple[bool, bool, str]:
        """
        Verify files using PAR2.

        Args:
            par2_file: Path to main .par2 file
            repair: Attempt repair if verification fails

        Returns:
            Tuple of (verified_ok, repaired, message)
        """
        if not self.par2_path:
            logger.warning("PAR2 not available, skipping verification")
            return True, False, "PAR2 not available"

        self._current_status = PostProcessStatus.VERIFYING
        self._report_progress(f"Verifying: {par2_file.name}", 0)

        try:
            # Build command
            cmd = [
                self.par2_path,
                'verify' if not repair else 'repair',
                '-q',  # Quiet (less output)
                f'-t{self.threads}',  # Threads (if supported)
                str(par2_file)
            ]

            logger.info(f"Running PAR2: {' '.join(cmd)}")
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(par2_file.parent),
                timeout=3600,  # 1 hour max
                creationflags=SUBPROCESS_FLAGS  # Hide console window
            )

            elapsed = time.time() - start_time

            # Parse result
            if result.returncode == 0:
                logger.info(f"PAR2 verification OK ({elapsed:.1f}s)")
                return True, False, f"Verified OK ({elapsed:.1f}s)"
            elif result.returncode == 1 and repair:
                # Repaired
                self._current_status = PostProcessStatus.REPAIRING
                logger.info(f"PAR2 repair completed ({elapsed:.1f}s)")
                return True, True, f"Repaired ({elapsed:.1f}s)"
            else:
                # Failed
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"PAR2 failed: {error_msg}")
                return False, False, f"Failed: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            return False, False, "PAR2 verification timed out"
        except Exception as e:
            logger.error(f"PAR2 error: {e}")
            return False, False, f"Error: {e}"

    def extract_archive(
        self,
        archive: Path,
        password: Optional[str] = None,
        output_dir: Optional[Path] = None,
        base_progress: float = 30,
        progress_range: float = 60
    ) -> Tuple[bool, int, str]:
        """
        Extract archive using 7-Zip with real-time progress tracking.

        Args:
            archive: Path to archive file
            password: Archive password (optional)
            output_dir: Output directory (default: self.extract_dir)
            base_progress: Starting progress percentage for reporting
            progress_range: Progress range to use (e.g., 30-90 = range of 60)

        Returns:
            Tuple of (success, files_extracted, message)
        """
        if not self.sevenzip_path:
            return False, 0, "7-Zip not available"

        self._current_status = PostProcessStatus.EXTRACTING
        self._report_progress(f"Extracting: {archive.name}", base_progress)

        out_dir = output_dir or self.extract_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Count existing files BEFORE extraction to compute delta
        existing_files = set(f for f in out_dir.rglob('*') if f.is_file())

        try:
            # Build 7z command with progress output
            cmd = [
                self.sevenzip_path,
                'x',  # Extract with full paths
                '-y',  # Yes to all prompts
                '-bsp1',  # Show progress on stdout
                '-bb1',  # Show file names
                f'-o{out_dir}',  # Output directory
            ]

            # Add password if provided
            if password:
                cmd.append(f'-p{password}')
            else:
                cmd.append('-p')  # Empty password (skip prompts)

            # Multi-threaded extraction
            cmd.append(f'-mmt={self.threads}')

            # Archive path
            cmd.append(str(archive))

            logger.info(f"Running 7z: {cmd[0]} x -y ... {archive.name}")
            start_time = time.time()
            last_progress_time = start_time
            last_percent = 0

            # Use Popen for real-time progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(archive.parent),
                creationflags=SUBPROCESS_FLAGS,
                bufsize=1  # Line buffered
            )

            # Read output in real-time
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    output_lines.append(line)

                    # Parse progress percentage from 7z output
                    # Format: " 45% - filename" or just "45%"
                    match = re.search(r'(\d+)%', line)
                    if match:
                        pct = int(match.group(1))
                        now = time.time()

                        # Update progress at most every 0.3s to avoid flooding
                        if pct != last_percent and (now - last_progress_time) > 0.3:
                            # Map 0-100% from 7z to base_progress to base_progress+progress_range
                            mapped_pct = base_progress + (progress_range * pct / 100)
                            self._report_progress(f"Extracting: {pct}%", mapped_pct)
                            last_percent = pct
                            last_progress_time = now

            process.wait()
            elapsed = time.time() - start_time

            if process.returncode == 0:
                # Count NEW extracted files only
                current_files = set(f for f in out_dir.rglob('*') if f.is_file())
                new_files = current_files - existing_files
                files_count = len(new_files)

                logger.info(f"Extraction OK: {files_count} files ({elapsed:.1f}s)")
                self._report_progress(f"Extracted: {files_count} files", base_progress + progress_range)
                return True, files_count, f"Extracted {files_count} files ({elapsed:.1f}s)"
            else:
                error_msg = ''.join(output_lines[-10:]) if output_lines else "Unknown error"
                # Check for password error
                if 'Wrong password' in error_msg or 'password' in error_msg.lower():
                    return False, 0, "Wrong password"
                logger.error(f"7z failed: {error_msg[:200]}")
                return False, 0, f"Failed: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            return False, 0, "Extraction timed out"
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return False, 0, f"Error: {e}"

    def cleanup_archives(
        self,
        directory: Optional[Path] = None,
        allowed_filenames: Optional[Set[str]] = None
    ) -> int:
        """
        Remove archive files, PAR2 files, and SFV files after successful extraction.

        Args:
            directory: Directory to clean (default: download_dir)
            allowed_filenames: If provided, only delete files whose names are in this set.
                              This ensures cleanup is specific to the current NZB.
        """
        search_dir = directory or self.download_dir
        removed = 0

        # Convert to normalized set for case-insensitive matching
        # Use _normalize_filename to handle spaces, underscores, parentheses
        allowed_normalized = set()
        if allowed_filenames:
            for f in allowed_filenames:
                allowed_normalized.add(_normalize_filename(f))

        # Extract common prefix from allowed filenames for fallback matching
        # e.g., "farmers_life_v1.0.43" from "Farmers_Life_v1.0.43.part001.rar"
        allowed_prefixes = set()
        if allowed_filenames:
            for fname in allowed_filenames:
                # Extract base name before .part, .vol, or extension
                base = re.sub(r'\.(part\d+|vol\d+[+\d]*|r\d+|par2|rar|zip|7z|sfv|nfo|flac|mp3|jpg|png).*$', '', fname.lower())
                if base and len(base) > 5:  # Minimum prefix length
                    # Normalize for consistent matching
                    base_normalized = _normalize_filename(base)
                    allowed_prefixes.add(base_normalized)

            # Debug logging
            if allowed_prefixes:
                logger.info(f"Cleanup prefixes ({len(allowed_prefixes)}): {list(allowed_prefixes)[:3]}")
            else:
                logger.warning(f"No valid prefixes extracted from {len(allowed_filenames)} NZB filenames")
                # Show sample of what was in allowed_filenames
                sample = list(allowed_filenames)[:5]
                logger.warning(f"NZB filenames sample: {sample}")

        try:
            patterns = ['*.rar', '*.r[0-9][0-9]', '*.zip', '*.7z', '*.par2',
                       '*.PAR2', '*.part*.rar', '*.sfv', '*.SFV',
                       # PAR2 backup files (created during repair): *.flac.1, *.mp3.1, etc.
                       '*.[0-9]', '*.[0-9][0-9]']

            files_found = []
            for pattern in patterns:
                files_found.extend(search_dir.glob(pattern))

            if files_found:
                sample_files = [f.name for f in files_found[:5]]
                logger.info(f"Files on disk ({len(files_found)}): {sample_files}")

            for file in files_found:
                file_normalized = _normalize_filename(file.name)

                # If allowed_filenames is set, only delete files from this NZB
                if allowed_normalized:
                    # Try normalized match first
                    if file_normalized in allowed_normalized:
                        pass  # Match found, proceed to delete
                    # Fallback: match by prefix (fully normalized)
                    elif allowed_prefixes:
                        file_base = re.sub(r'\.(part\d+|vol\d+[+\d]*|r\d+|par2|rar|zip|7z|sfv|nfo|flac|mp3|jpg|png).*$', '', file_normalized)
                        if not any(file_base.startswith(prefix) or prefix.startswith(file_base) for prefix in allowed_prefixes):
                            logger.debug(f"Skipping cleanup of {file.name} (not matching NZB)")
                            continue
                    else:
                        logger.debug(f"Skipping cleanup of {file.name} (not in NZB)")
                        continue

                try:
                    file.unlink()
                    removed += 1
                except Exception as e:
                    logger.debug(f"Could not delete {file.name}: {e}")

            logger.info(f"Cleaned up {removed} files")

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

        return removed

    # Media file extensions to move to final destination (video, audio, subtitles, images)
    MEDIA_EXTENSIONS = {
        # Video
        '.mkv', '.avi', '.mp4', '.m4v', '.mov', '.wmv', '.flv',
        '.webm', '.mpg', '.mpeg', '.m2ts', '.ts', '.vob',
        '.iso', '.img',
        # Audio
        '.flac', '.mp3', '.wav', '.aac', '.m4a', '.ogg', '.opus',
        '.wma', '.alac', '.ape', '.aiff', '.dsd', '.dsf', '.dff',
        '.mqa', '.cue',  # Lossless/Hi-Res audio formats
        # Playlists
        '.m3u', '.m3u8', '.pls', '.xspf',
        # Subtitles
        '.srt', '.sub', '.idx', '.ass', '.ssa', '.vtt',
        # Info/metadata
        '.nfo', '.txt', '.log',
        # Cover images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
    }

    def _is_archive(self, file: Path) -> bool:
        """Check if file is an archive."""
        suffix = file.suffix.lower()
        if suffix in self.ARCHIVE_EXTENSIONS:
            return True
        # RAR split: .r00, .r01, etc.
        if re.match(r'\.[r]\d{2,}$', suffix):
            return True
        return False

    def _is_media_file(self, file: Path) -> bool:
        """Check if file is a media file (video, subtitle, etc.)."""
        return file.suffix.lower() in self.MEDIA_EXTENSIONS

    def _move_media_files(self, src_dir: Path, release_name: Optional[str] = None) -> int:
        """
        Move media files (MKV, AVI, etc.) to the final destination.
        Uses parallel batch move for performance.

        Args:
            src_dir: Source directory containing downloaded files
            release_name: Name for destination subfolder

        Returns:
            Number of files moved
        """
        # Create destination subfolder
        dest_folder = self.extract_dir / (release_name or "media")
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Collect all files to move
        file_pairs: List[Tuple[Path, Path]] = []

        for file in src_dir.iterdir():
            if not file.is_file():
                continue

            # Skip PAR2 and archive files
            suffix = file.suffix.lower()
            if suffix == '.par2' or self._is_archive(file):
                continue

            # Collect media files
            if self._is_media_file(file):
                target = dest_folder / file.name

                # Handle conflicts
                if target.exists():
                    base, ext = target.stem, target.suffix
                    counter = 1
                    while target.exists():
                        target = dest_folder / f"{base}_{counter}{ext}"
                        counter += 1

                file_pairs.append((file, target))

        if not file_pairs:
            return 0

        # Batch move with progress
        def move_progress(done: int, total: int) -> None:
            self._report_progress(f"Moving media: {done}/{total}", 80 + (15 * done / total))

        batch_result = batch_move_files(
            file_pairs,
            max_workers=min(8, len(file_pairs)),
            on_progress=move_progress
        )

        # Track AV-blocked files
        for r in batch_result.results:
            if r.av_blocked and r.error:
                self._av_blocked_files.add(r.error)

        logger.info(f"Moved {batch_result.successful} media file(s) to {dest_folder.name}/")
        return batch_result.successful

    def _derive_filenames_from_disk(
        self,
        directory: Path,
        release_title: str
    ) -> Set[str]:
        """
        Derive cleanup filenames from files on disk when NZB has obfuscated subjects.

        This scans the download directory for archive/par2 files and matches them
        against the release title to find files belonging to this download.

        Args:
            directory: Download directory to scan
            release_title: Release title from NZB metadata (e.g., "Release.Name-GROUP")

        Returns:
            Set of filenames that match the release title
        """
        if not release_title:
            return set()

        # Normalize release title for matching
        title_normalized = _normalize_filename(release_title)
        # Also create a version without group suffix for broader matching
        title_base = re.sub(r'[-_](gog|codex|plaza|skidrow|fitgirl|dodi|elamigos|rune|razor1911|p2p)$',
                           '', title_normalized, flags=re.IGNORECASE)

        logger.debug(f"Disk cleanup: looking for files matching '{title_normalized}' or '{title_base}'")

        matching_files = set()

        # Scan for archive patterns
        patterns = ['*.rar', '*.r[0-9][0-9]', '*.zip', '*.7z', '*.par2',
                   '*.PAR2', '*.part*.rar', '*.sfv', '*.SFV',
                   '*.[0-9]', '*.[0-9][0-9]']

        for pattern in patterns:
            for file in directory.glob(pattern):
                file_normalized = _normalize_filename(file.stem)

                # Check if file matches release title
                # Use contains check for flexibility with different naming conventions
                if (title_normalized in file_normalized or
                    title_base in file_normalized or
                    file_normalized.startswith(title_normalized[:20]) or  # First 20 chars
                    file_normalized.startswith(title_base[:20])):
                    matching_files.add(file.name)

        if matching_files:
            sample = list(matching_files)[:5]
            logger.info(f"Disk-based cleanup found {len(matching_files)} files: {sample}")

        return matching_files

    def _copy_nfo_files(
        self,
        source_dir: Path,
        dest_dir: Path,
        allowed_filenames: Optional[Set[str]] = None
    ) -> int:
        """
        Copy NFO files from source directory to destination if not already present.

        Args:
            source_dir: Directory containing NFO files (download dir)
            dest_dir: Destination directory (extracted content)
            allowed_filenames: If provided, only copy NFOs whose names are in this set.

        Returns:
            Number of NFO files copied
        """
        if not dest_dir or not dest_dir.exists():
            return 0

        # Find NFO files in source directory (non-recursive)
        nfo_files = list(source_dir.glob('*.nfo'))

        # Filter to only NFOs from this NZB if allowed_filenames is set
        if allowed_filenames:
            # Build allowed set with normalized names (handles spaces, underscores, parentheses)
            allowed_normalized = set()
            for f in allowed_filenames:
                allowed_normalized.add(_normalize_filename(f))

            # Also extract prefixes for fallback matching (normalized)
            allowed_prefixes = set()
            for fname in allowed_filenames:
                base = re.sub(r'\.(part\d+|vol\d+[+\d]*|r\d+|par2|rar|zip|7z|sfv|nfo|flac|mp3|jpg|png).*$', '', fname.lower())
                if base and len(base) > 5:
                    allowed_prefixes.add(_normalize_filename(base))

            def matches_nzb(nfo_name: str) -> bool:
                nfo_normalized = _normalize_filename(nfo_name)
                # Exact match (normalized)
                if nfo_normalized in allowed_normalized:
                    return True
                # Prefix match (normalized)
                nfo_base = _normalize_filename(nfo_name.replace('.nfo', ''))
                return any(nfo_base.startswith(p) or p.startswith(nfo_base) for p in allowed_prefixes)

            nfo_files = [f for f in nfo_files if matches_nzb(f.name)]

        if not nfo_files:
            return 0

        # Check existing NFO files in destination
        existing_nfos = {f.name.lower() for f in dest_dir.rglob('*.nfo')}

        copied = 0
        for nfo in nfo_files:
            if nfo.name.lower() not in existing_nfos:
                try:
                    dest_path = dest_dir / nfo.name
                    shutil.copy2(nfo, dest_path)
                    logger.info(f"Copied NFO: {nfo.name} → {dest_dir.name}/")
                    copied += 1

                    # Delete NFO from source after successful copy
                    try:
                        nfo.unlink()
                        logger.debug(f"Deleted source NFO: {nfo.name}")
                    except Exception as e:
                        logger.debug(f"Could not delete source NFO {nfo.name}: {e}")

                except Exception as e:
                    logger.warning(f"Failed to copy NFO {nfo.name}: {e}")

        if copied > 0:
            logger.info(f"Copied {copied} NFO file(s) to extracted content")

        return copied

    def _find_nested_archives(self, directory: Path) -> List[Path]:
        """Find archives in extracted content (for nested extraction)."""
        archives = []
        seen_bases = set()

        for file in directory.rglob('*'):
            if not file.is_file():
                continue

            name_lower = file.name.lower()
            suffix = file.suffix.lower()

            # Skip PAR2 volumes
            if suffix == '.par2' or '.vol' in name_lower:
                continue

            # RAR multi-part: prioritize .part01.rar or just .rar
            if '.part' in name_lower and suffix == '.rar':
                base = re.sub(r'\.part\d+\.rar$', '', name_lower)
                if base not in seen_bases:
                    seen_bases.add(base)
                    # Find the .part1 or .part01 file
                    for p in directory.rglob(f'{file.stem.rsplit(".part", 1)[0]}.part1.rar'):
                        archives.append(p)
                        break
                    else:
                        for p in directory.rglob(f'{file.stem.rsplit(".part", 1)[0]}.part01.rar'):
                            archives.append(p)
                            break
            elif suffix == '.rar' and '.part' not in name_lower:
                base = file.stem.lower()
                if base not in seen_bases:
                    seen_bases.add(base)
                    archives.append(file)
            elif suffix in {'.zip', '.7z'}:
                archives.append(file)

        return archives

    def extract_all_to_temp(
        self,
        archives: List[Path],
        password: Optional[str] = None,
        parallel: bool = True,
        max_parallel: int = 3
    ) -> Tuple[Path, bool]:
        """
        Extract ALL archives to a single temp directory.
        Uses parallel extraction for multiple archives.

        Args:
            archives: List of archive files
            password: Optional archive password
            parallel: Use parallel extraction (default: True)
            max_parallel: Max concurrent extractions (default: 3)

        Returns:
            (temp_dir, success)
        """
        temp_dir = self.download_dir / "_temp_extract_combined"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Use parallel extraction for multiple archives
        if parallel and len(archives) > 1 and self.sevenzip_path:
            # Calculate optimal parameters
            max_par, threads_per = ParallelExtractor.calculate_optimal_params()
            max_par = min(max_par, max_parallel, len(archives))

            extractor = ParallelExtractor(
                sevenzip_path=self.sevenzip_path,
                max_parallel=max_par,
                threads_per_extraction=threads_per,
                on_progress=self.on_progress
            )

            results = extractor.extract_parallel(archives, temp_dir, password)

            # Log failures
            for r in results:
                if not r.success:
                    logger.warning(f"Failed to extract {r.archive.name}: {r.message}")

            # Success if at least one archive extracted
            any_success = any(r.success for r in results)
            return temp_dir, any_success

        else:
            # Sequential fallback for single archive or no 7z
            for archive in archives:
                self._report_progress(f"Extracting: {archive.name}", 50)
                success, _, msg = self.extract_archive(archive, password, temp_dir)
                if not success:
                    logger.warning(f"Failed to extract {archive.name}: {msg}")

            return temp_dir, True

    def _is_potentially_nested(self, archives: List[Path]) -> bool:
        """
        Check if archives might contain nested archives (ZIP→RAR scenario).

        Returns True if we should use temp extraction to handle nesting.
        """
        # ZIPs often contain nested RAR parts that need combining
        has_zips = any(a.suffix.lower() == '.zip' for a in archives)

        # Multiple archives could have cross-archive dependencies
        multiple = len(archives) > 3

        return has_zips or multiple

    def _extract_direct_to_destination(
        self,
        archives: List[Path],
        dest_dir: Path,
        password: Optional[str] = None
    ) -> Tuple[bool, int, str]:
        """
        Extract archives directly to destination directory (no temp, no move).

        This is the optimized path when:
        - Download and extract dirs are on different drives
        - Archives are simple (no nesting expected)

        Args:
            archives: List of archive files
            dest_dir: Final destination directory
            password: Optional archive password

        Returns:
            (success, files_count, message)
        """
        total_files = 0
        failed_archives = []

        # Use parallel extraction if multiple archives
        if len(archives) > 1 and self.sevenzip_path:
            max_par, threads_per = ParallelExtractor.calculate_optimal_params()
            max_par = min(max_par, len(archives))

            extractor = ParallelExtractor(
                sevenzip_path=self.sevenzip_path,
                max_parallel=max_par,
                threads_per_extraction=threads_per,
                on_progress=self.on_progress
            )

            results = extractor.extract_parallel(archives, dest_dir, password)

            for r in results:
                if r.success:
                    total_files += r.files_extracted
                else:
                    failed_archives.append(r.archive.name)
                    logger.warning(f"Direct extraction failed: {r.archive.name}: {r.message}")

        else:
            # Sequential extraction for single archive
            for i, archive in enumerate(archives):
                progress = 30 + (60 * i / max(1, len(archives)))
                self._report_progress(f"Extracting: {archive.name}", progress)

                success, count, msg = self.extract_archive(archive, password, dest_dir)
                if success:
                    total_files += count
                else:
                    failed_archives.append(archive.name)
                    logger.warning(f"Direct extraction failed: {archive.name}: {msg}")

        if total_files > 0:
            logger.info(f"Direct extraction complete: {total_files} files to {dest_dir}")
            return True, total_files, f"Extracted {total_files} files directly"
        elif failed_archives:
            return False, 0, f"All extractions failed: {', '.join(failed_archives)}"
        else:
            return True, 0, "No files extracted"

    def smart_extract(
        self,
        archives: List[Path],
        password: Optional[str] = None,
        release_name: Optional[str] = None
    ) -> Tuple[bool, int, str]:
        """
        Smart extraction handling nested archives (ZIP→RAR→content).

        Strategy:
        - FAST PATH: If archives are simple (RAR/7z only) and destination is on
          different drive, extract DIRECTLY to destination (no temp, no move)
        - NESTED PATH: If archives might contain nested content (ZIPs), use temp
          extraction to handle ZIP→RAR scenarios

        Args:
            archives: List of archive files
            password: Archive password
            release_name: Name for destination subfolder

        Returns:
            (success, files_count, message)
        """
        if not archives:
            return True, 0, "No archives"

        # Create destination subfolder with release name
        dest_subfolder = self.extract_dir / (release_name or "extracted")
        dest_subfolder.mkdir(parents=True, exist_ok=True)

        # OPTIMIZATION: Check if we can extract directly to destination
        # This avoids temp extraction + move when destination is on different drive
        different_drives = not are_on_same_drive(self.download_dir, self.extract_dir)
        potentially_nested = self._is_potentially_nested(archives)

        # FAST PATH: Direct extraction to destination (no temp, no move)
        if different_drives and not potentially_nested:
            logger.info(f"FAST PATH: Direct extraction to {dest_subfolder} (different drives, no nesting)")
            self._report_progress(f"Direct extraction ({len(archives)} archives)...", 30)

            return self._extract_direct_to_destination(archives, dest_subfolder, password)

        # NESTED PATH: Use temp extraction for complex scenarios
        try:
            # Phase 1: Extract all outer archives to combined temp
            logger.info(f"NESTED PATH: Extracting {len(archives)} archives to temp...")
            self._report_progress(f"Extracting {len(archives)} archives...", 30)

            temp_dir, _ = self.extract_all_to_temp(archives, password)

            # Phase 2: Find nested archives
            nested = self._find_nested_archives(temp_dir)

            if nested:
                # Phase 3: Extract nested archives to final destination
                logger.info(f"Phase 2: Found {len(nested)} nested archive(s), extracting...")
                self._report_progress(f"Found {len(nested)} nested archives", 60)

                total_files = 0
                for i, nested_archive in enumerate(nested):
                    progress = 60 + (30 * i / len(nested))
                    self._report_progress(f"Extracting: {nested_archive.name}", progress)

                    success, count, msg = self.extract_archive(
                        nested_archive, password, dest_subfolder
                    )
                    if success:
                        total_files += count
                        logger.info(f"Nested extraction OK: {nested_archive.name} → {count} files")
                    else:
                        logger.warning(f"Nested extraction failed: {nested_archive.name}: {msg}")

                # Cleanup temp
                shutil.rmtree(temp_dir, ignore_errors=True)

                if total_files > 0:
                    logger.info(f"Smart extract complete: {total_files} files to {dest_subfolder}")
                    return True, total_files, f"Extracted {total_files} files"
                else:
                    return False, 0, "No files extracted from nested archives"

            else:
                # No nested archives - move temp content to destination using batch move
                logger.info("No nested archives found, moving content directly...")
                self._report_progress("Collecting files to move...", 80)

                # Collect all files to move
                file_pairs: List[Tuple[Path, Path]] = []
                used_targets: Set[Path] = set()

                for item in temp_dir.iterdir():
                    # Skip archive files themselves if they ended up in temp
                    if self._is_archive(item):
                        continue

                    if item.is_dir():
                        # Collect all files in directory
                        base_target = dest_subfolder / item.name
                        for sub in item.rglob('*'):
                            if sub.is_file():
                                rel = sub.relative_to(item)
                                sub_target = base_target / rel
                                # Handle conflicts
                                if sub_target in used_targets or sub_target.exists():
                                    base, ext = sub_target.stem, sub_target.suffix
                                    counter = 1
                                    while sub_target in used_targets or sub_target.exists():
                                        sub_target = sub_target.parent / f"{base}_{counter}{ext}"
                                        counter += 1
                                sub_target.parent.mkdir(parents=True, exist_ok=True)
                                file_pairs.append((sub, sub_target))
                                used_targets.add(sub_target)
                    else:
                        target = dest_subfolder / item.name
                        # Handle conflicts
                        if target in used_targets or target.exists():
                            base, ext = target.stem, target.suffix
                            counter = 1
                            while target in used_targets or target.exists():
                                target = dest_subfolder / f"{base}_{counter}{ext}"
                                counter += 1
                        file_pairs.append((item, target))
                        used_targets.add(target)

                if not file_pairs:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return True, 0, "No files to move"

                # Batch move with progress
                def move_progress(done: int, total: int) -> None:
                    pct = 80 + (15 * done / max(1, total))
                    self._report_progress(f"Moving files: {done}/{total}", pct)

                batch_result = batch_move_files(
                    file_pairs,
                    max_workers=min(12, os.cpu_count() or 8),
                    on_progress=move_progress
                )

                # Track AV-blocked files
                for i, r in enumerate(batch_result.results):
                    if r.av_blocked:
                        src_file = file_pairs[i][0] if i < len(file_pairs) else None
                        if src_file:
                            self._av_blocked_files.add(src_file.name)

                # Cleanup temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)

                files_moved = batch_result.successful
                logger.info(f"Direct extract complete: {files_moved} files to {dest_subfolder}")
                return True, files_moved, f"Moved {files_moved} files"

        except Exception as e:
            logger.error(f"Smart extract error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0, str(e)

    def process(
        self,
        nzb_path: Optional[Path] = None,
        source_dir: Optional[Path] = None,
        password: Optional[str] = None,
        early_par2_result: Optional[Tuple[bool, bool, str]] = None
    ) -> PostProcessResult:
        """
        Full post-processing pipeline.

        1. Parse NZB for password (if provided)
        2. Find and verify PAR2 files (or use early result from streaming PAR2)
        3. Find and extract archives
        4. Optional cleanup

        Args:
            nzb_path: Path to NZB file (for password extraction)
            source_dir: Directory containing downloaded files (default: download_dir)
            password: Override password (if not from NZB)
            early_par2_result: Optional (verified, repaired, message) from streaming PAR2

        Returns:
            PostProcessResult with details
        """
        start_time = time.time()
        result = PostProcessResult(
            success=False,
            status=PostProcessStatus.PENDING,
            message=""
        )

        src_dir = source_dir or self.download_dir

        try:
            self._stop_requested = False
            metadata = NZBMetadata()  # Default empty metadata

            # Step 1: Parse NZB metadata for password
            if nzb_path and nzb_path.exists():
                metadata = self.parse_nzb_metadata(nzb_path)
                if metadata.password and not password:
                    password = metadata.password
                    logger.info(f"Using password from NZB: ***")

            # Step 2: PAR2 verification (use early result if available)
            par2_files = self.find_par2_files(src_dir)

            if early_par2_result:
                # Use streaming PAR2 result - skip re-verification
                verified, repaired, msg = early_par2_result
                logger.info(f"Using early PAR2 result: verified={verified}, repaired={repaired}")

                if verified:
                    result.par2_verified = True
                    result.par2_repaired = repaired
                    self._report_progress(f"PAR2 (early): {msg}", 40)
                else:
                    # Early verification failed - try repair
                    if par2_files and self.par2_path:
                        logger.info("Early PAR2 failed, attempting repair...")
                        self._report_progress("PAR2 repair needed...", 10)
                        verified, repaired, msg = self.verify_par2(par2_files[0], repair=True)
                        result.par2_verified = verified
                        result.par2_repaired = repaired

                        if not verified:
                            result.status = PostProcessStatus.FAILED
                            result.message = f"PAR2 repair failed: {msg}"
                            result.errors.append(msg)
                            return result

                        self._report_progress(f"PAR2: {msg}", 40)
                    else:
                        result.status = PostProcessStatus.FAILED
                        result.message = f"PAR2 verification failed: {msg}"
                        result.errors.append(msg)
                        return result

            elif par2_files:
                # No early result - run full verification
                self._report_progress("Starting PAR2 verification...", 10)
                main_par2 = par2_files[0]  # Main par2 file

                verified, repaired, msg = self.verify_par2(main_par2)
                result.par2_verified = verified
                result.par2_repaired = repaired

                if not verified:
                    result.status = PostProcessStatus.FAILED
                    result.message = f"PAR2 verification failed: {msg}"
                    result.errors.append(msg)
                    return result

                self._report_progress(f"PAR2: {msg}", 40)
            else:
                logger.info("No PAR2 files found, skipping verification")
                result.par2_verified = True  # Assume OK

            if self._stop_requested:
                result.status = PostProcessStatus.FAILED
                result.message = "Cancelled"
                return result

            # Step 3: Smart extraction (handles ZIP→RAR→content)
            archives = self.find_archives(src_dir)
            if archives:
                # Get release name for subfolder
                release_name = metadata.title if nzb_path else None
                if not release_name and archives:
                    # Fallback to first archive name
                    release_name = archives[0].stem

                logger.info(f"Smart extracting {len(archives)} archives → {release_name}/")
                self._report_progress(f"Smart extracting {len(archives)} archives...", 50)

                # Track extraction time for metrics
                extraction_start = time.time()

                # Use smart_extract: extracts ALL to temp, finds nested RARs, extracts to subfolder
                success, total_files, msg = self.smart_extract(
                    archives, password, release_name
                )

                result.extraction_duration_seconds = time.time() - extraction_start

                if success:
                    result.files_extracted = total_files
                    result.extract_path = self.extract_dir / release_name
                    self._report_progress(f"Extracted {total_files} files", 90)

                    # Calculate extracted bytes for metrics
                    try:
                        result.extracted_bytes = sum(
                            f.stat().st_size for f in result.extract_path.rglob('*') if f.is_file()
                        )
                    except Exception:
                        pass

                    # Copy NFO files if not already in extracted content
                    # Pass NZB filenames to only copy NFOs from this specific download
                    nzb_filenames = set(metadata.filenames) if metadata.filenames else None
                    nfo_copied = self._copy_nfo_files(src_dir, result.extract_path, nzb_filenames)
                    if nfo_copied > 0:
                        result.files_extracted += nfo_copied
                else:
                    result.errors.append(msg)
                    if total_files == 0:
                        result.status = PostProcessStatus.FAILED
                        result.message = f"Extraction failed: {msg}"
                        return result
            else:
                # No archives - check for media files to move (MKV, AVI, MP4, etc.)
                logger.info("No archives found, checking for media files...")
                moved_files = self._move_media_files(src_dir, metadata.title)
                if moved_files > 0:
                    result.files_extracted = moved_files
                    result.extract_path = self.extract_dir / metadata.title if metadata.title else self.extract_dir
                    logger.info(f"Moved {moved_files} media file(s) to destination")

            # Step 4: Cleanup (optional) - ONLY files from this NZB
            if self.cleanup_after_extract and result.files_extracted > 0:
                self._report_progress("Cleaning up...", 95)

                if metadata.filenames:
                    # Normal path: use NZB filenames
                    nzb_filenames = set(metadata.filenames)
                    self.cleanup_archives(src_dir, nzb_filenames)
                else:
                    # Fallback: derive filenames from disk using release title
                    # This handles obfuscated NZB subjects where filenames couldn't be parsed
                    logger.info("No NZB filenames - using disk-based cleanup with title matching")
                    disk_filenames = self._derive_filenames_from_disk(src_dir, metadata.title)
                    if disk_filenames:
                        logger.info(f"Found {len(disk_filenames)} matching files on disk for cleanup")
                        self.cleanup_archives(src_dir, disk_filenames)
                    else:
                        logger.warning("Skipping cleanup: could not match files on disk to release title")
                        result.warnings.append((
                            WarningType.CLEANUP_SKIPPED,
                            "Cleanup skipped - manual cleanup required"
                        ))

            # Success!
            result.success = True
            result.status = PostProcessStatus.COMPLETED
            result.duration_seconds = time.time() - start_time
            result.message = f"Completed: {result.files_extracted} files extracted"

            if result.par2_repaired:
                result.message += " (repaired)"

            # Add warnings for user attention
            if result.par2_repaired:
                result.warnings.append((
                    WarningType.PAR2_REPAIR_NEEDED,
                    "Files were corrupted and required PAR2 repair"
                ))

            if self._av_blocked_files:
                result.antivirus_blocked_files = list(self._av_blocked_files)
                blocked_list = ", ".join(list(self._av_blocked_files)[:3])
                if len(self._av_blocked_files) > 3:
                    blocked_list += f" (+{len(self._av_blocked_files) - 3} more)"
                result.warnings.append((
                    WarningType.ANTIVIRUS_BLOCK,
                    f"Antivirus may have blocked: {blocked_list}. Check Windows Defender exclusions."
                ))

            self._report_progress(result.message, 100)

        except Exception as e:
            logger.exception(f"Post-processing error: {e}")
            result.status = PostProcessStatus.FAILED
            result.message = f"Error: {e}"
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start_time
        return result

    def stop(self) -> None:
        """Request stop of current processing."""
        self._stop_requested = True


class ParallelPostProcessor:
    """
    Parallel post-processor for multiple downloads.

    Processes multiple downloads concurrently with shared resources.
    """

    def __init__(
        self,
        extract_dir: Path,
        max_concurrent: int = 2,
        par2_path: Optional[str] = None,
        sevenzip_path: Optional[str] = None,
        cleanup: bool = True
    ):
        self.extract_dir = Path(extract_dir)
        self.max_concurrent = max_concurrent
        self.par2_path = par2_path
        self.sevenzip_path = sevenzip_path
        self.cleanup = cleanup

        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._pending: Dict[str, 'Future'] = {}
        self._lock = threading.Lock()

    def queue_process(
        self,
        download_dir: Path,
        nzb_path: Optional[Path] = None,
        callback: Optional[Callable[[PostProcessResult], None]] = None
    ) -> str:
        """
        Queue a download for post-processing.

        Returns job ID.
        """
        job_id = f"{download_dir.name}_{time.time()}"

        def run_job():
            processor = PostProcessor(
                download_dir=download_dir,
                extract_dir=self.extract_dir / download_dir.name,
                par2_path=self.par2_path,
                sevenzip_path=self.sevenzip_path,
                cleanup_after_extract=self.cleanup
            )
            result = processor.process(nzb_path)
            if callback:
                callback(result)
            return result

        with self._lock:
            future = self._executor.submit(run_job)
            self._pending[job_id] = future

        return job_id

    def shutdown(self) -> None:
        """Shutdown executor."""
        self._executor.shutdown(wait=False)


# Utility function for quick processing
def quick_process(
    download_dir: Path,
    nzb_path: Optional[Path] = None,
    extract_dir: Optional[Path] = None,
    password: Optional[str] = None,
    cleanup: bool = True
) -> PostProcessResult:
    """
    Quick post-processing function.

    Args:
        download_dir: Directory containing downloaded files
        nzb_path: Path to NZB file (for password)
        extract_dir: Output directory (default: download_dir/extracted)
        password: Archive password (overrides NZB)
        cleanup: Delete archives after extraction

    Returns:
        PostProcessResult
    """
    processor = PostProcessor(
        download_dir=download_dir,
        extract_dir=extract_dir,
        cleanup_after_extract=cleanup
    )
    return processor.process(nzb_path, password=password)


# Test/demo
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python post_processor.py <download_dir> [nzb_file]")
        sys.exit(1)

    download_path = Path(sys.argv[1])
    nzb_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    def progress_callback(msg: str, percent: float):
        print(f"[{percent:5.1f}%] {msg}")

    processor = PostProcessor(
        download_dir=download_path,
        on_progress=progress_callback
    )

    result = processor.process(nzb_file)

    print(f"\n{'='*50}")
    print(f"Status: {result.status.value}")
    print(f"Message: {result.message}")
    print(f"PAR2 Verified: {result.par2_verified}")
    print(f"PAR2 Repaired: {result.par2_repaired}")
    print(f"Files Extracted: {result.files_extracted}")
    print(f"Extract Path: {result.extract_path}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    if result.errors:
        print(f"Errors: {result.errors}")
