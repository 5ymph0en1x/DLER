"""
File Verification Queue Component
==================================

Provides parallel MD5 verification of downloaded files against PAR2 metadata.
Files are queued for verification and processed by a thread pool, with results
delivered via callback for real-time progress tracking.

Key features:
- Thread-pool based parallel MD5 computation
- 1MB chunk reads for efficient large file processing
- Thread-safe queue management with result tracking
- Graceful shutdown support
- Support for files not in PAR2 (marked as SKIPPED)

Usage:
    def on_result(result: FileVerificationResult):
        print(f"{result.filename}: {result.status.name}")

    queue = FileVerificationQueue(
        output_dir=Path("/downloads"),
        on_result=on_result,
        threads=4
    )

    # Queue files for verification
    queue.queue_file_for_verification(
        file_path=Path("/downloads/archive.part01.rar"),
        par2_entry=db.get_entry_by_filename("archive.part01.rar")
    )

    # Wait for all verifications to complete
    results = queue.wait_for_all(timeout=300)

    # Clean shutdown
    queue.shutdown()
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .par2_database import FileVerificationStatus, Par2FileEntry

logger = logging.getLogger(__name__)

# Constants for MD5 computation
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for efficient large file processing


@dataclass
class FileVerificationResult:
    """
    Result of a file verification operation.

    Contains all information about a single file verification, including
    the computed MD5 hash, comparison result, timing, and any errors.

    Attributes:
        filename: Display name of the file (may differ from path basename for obfuscated files)
        file_path: Absolute path to the verified file
        status: Verification result status (VERIFIED, DAMAGED, SKIPPED, ERROR)
        md5_expected: Expected MD5 hash from PAR2 metadata (hex string), or None if not in PAR2
        md5_computed: Computed MD5 hash of the file (hex string), or None on error
        duration_seconds: Time taken to verify the file
        error: Error message if verification failed, None otherwise
        timestamp: Unix timestamp when verification completed
    """
    filename: str
    file_path: Path
    status: FileVerificationStatus
    md5_expected: Optional[str]
    md5_computed: Optional[str]
    duration_seconds: float
    error: Optional[str]
    timestamp: float

    @property
    def is_verified(self) -> bool:
        """Return True if file verified successfully (MD5 matches)."""
        return self.status == FileVerificationStatus.VERIFIED

    @property
    def is_damaged(self) -> bool:
        """Return True if file is damaged (MD5 mismatch)."""
        return self.status == FileVerificationStatus.DAMAGED

    @property
    def is_skipped(self) -> bool:
        """Return True if file was skipped (not in PAR2)."""
        return self.status == FileVerificationStatus.SKIPPED

    @property
    def has_error(self) -> bool:
        """Return True if verification encountered an error."""
        return self.status == FileVerificationStatus.ERROR

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"FileVerificationResult("
            f"filename={self.filename!r}, "
            f"status={self.status.name}, "
            f"duration={self.duration_seconds:.2f}s)"
        )


class FileVerificationQueue:
    """
    Thread-safe queue for parallel file MD5 verification.

    Manages a pool of worker threads that compute MD5 hashes and compare
    them against expected values from PAR2 metadata. Results are delivered
    via callback for real-time progress tracking.

    Thread Safety:
        All public methods are thread-safe. Internal state is protected
        by a reentrant lock.

    Example:
        queue = FileVerificationQueue(
            output_dir=Path("/downloads"),
            on_result=lambda r: print(f"{r.filename}: {r.status.name}"),
            threads=4
        )

        for file_path in files_to_verify:
            entry = db.get_entry_by_filename(file_path.name)
            queue.queue_file_for_verification(file_path, entry)

        results = queue.wait_for_all()
        queue.shutdown()
    """

    def __init__(
        self,
        output_dir: Path,
        on_result: Optional[Callable[[FileVerificationResult], None]] = None,
        threads: int = 4
    ) -> None:
        """
        Initialize the file verification queue.

        Args:
            output_dir: Base directory for file operations (used for logging context)
            on_result: Optional callback invoked when each file verification completes.
                       The callback receives a FileVerificationResult and is called
                       from the worker thread.
            threads: Number of worker threads for parallel MD5 computation.
                     Recommended: 2-4 for HDD, 4-8 for SSD. Default is 4.
        """
        self._output_dir = Path(output_dir)
        self._on_result = on_result
        self._threads = max(1, threads)  # Ensure at least 1 thread

        # Thread pool for parallel verification
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=self._threads,
            thread_name_prefix="FileVerifier"
        )

        # Track pending futures and results
        self._futures: Dict[Future, Path] = {}  # Future -> file_path
        self._results: List[FileVerificationResult] = []
        self._pending_paths: Set[Path] = set()  # Files currently being verified

        # Thread safety
        self._lock = threading.RLock()
        self._shutdown = False

        logger.info(
            f"FileVerificationQueue initialized: "
            f"output_dir={output_dir}, threads={self._threads}"
        )

    def queue_file_for_verification(
        self,
        file_path: Path,
        par2_entry: Optional[Par2FileEntry],
        original_filename: Optional[str] = None
    ) -> bool:
        """
        Add a file to the verification queue.

        The file will be verified asynchronously by a worker thread. If par2_entry
        is provided, the computed MD5 will be compared against the expected value.
        If par2_entry is None, the file will be marked as SKIPPED.

        Args:
            file_path: Path to the file to verify (must exist)
            par2_entry: PAR2 metadata entry for the file, or None if not in PAR2
            original_filename: Optional display name for the file (used when filename
                              differs from the PAR2 entry, e.g., for obfuscated files)

        Returns:
            True if file was queued successfully, False if queue is shut down
            or file is already being verified.

        Raises:
            ValueError: If file_path is None
        """
        if file_path is None:
            raise ValueError("file_path cannot be None")

        file_path = Path(file_path).resolve()

        with self._lock:
            # Check if shutdown
            if self._shutdown:
                logger.warning(f"Cannot queue {file_path.name}: queue is shut down")
                return False

            # Check if executor is available
            if self._executor is None:
                logger.warning(f"Cannot queue {file_path.name}: executor not available")
                return False

            # Avoid duplicate verification
            if file_path in self._pending_paths:
                logger.debug(f"File already pending: {file_path.name}")
                return False

            # Mark as pending
            self._pending_paths.add(file_path)

            # Determine display filename
            display_name = original_filename or (
                par2_entry.filename if par2_entry else file_path.name
            )

            # Submit verification task
            future = self._executor.submit(
                self._verify_file_worker,
                file_path,
                par2_entry,
                display_name
            )

            # Track the future
            self._futures[future] = file_path

            # Add completion callback
            future.add_done_callback(self._on_future_done)

            logger.debug(f"Queued for verification: {display_name}")
            return True

    def _verify_file_worker(
        self,
        file_path: Path,
        par2_entry: Optional[Par2FileEntry],
        display_name: str
    ) -> FileVerificationResult:
        """
        Worker method to compute MD5 and compare against PAR2 expected value.

        This method runs in a worker thread. It reads the file in chunks,
        computes the MD5 hash, and compares it against the expected value
        from the PAR2 entry.

        Args:
            file_path: Path to the file to verify
            par2_entry: PAR2 metadata entry, or None if not in PAR2
            display_name: Display name for the file

        Returns:
            FileVerificationResult with verification outcome
        """
        start_time = time.time()
        md5_computed: Optional[str] = None
        md5_expected: Optional[str] = None
        status: FileVerificationStatus
        error: Optional[str] = None

        logger.debug(f"Starting verification: {display_name}")

        try:
            # Check if file exists
            if not file_path.exists():
                error = f"File not found: {file_path}"
                status = FileVerificationStatus.ERROR
                logger.warning(error)

            # Handle files not in PAR2
            elif par2_entry is None:
                status = FileVerificationStatus.SKIPPED
                logger.debug(f"Skipped (not in PAR2): {display_name}")

            else:
                # Get expected MD5 from PAR2 entry
                md5_expected = par2_entry.md5_full_hex

                # Compute MD5 hash
                md5_computed = self._compute_md5(file_path)

                # Compare hashes
                if md5_computed == md5_expected:
                    status = FileVerificationStatus.VERIFIED
                    logger.info(f"Verified OK: {display_name}")
                else:
                    status = FileVerificationStatus.DAMAGED
                    logger.warning(
                        f"MD5 mismatch for {display_name}: "
                        f"expected={md5_expected}, computed={md5_computed}"
                    )

        except PermissionError as e:
            error = f"Permission denied reading file: {e}"
            status = FileVerificationStatus.ERROR
            logger.error(f"Permission error verifying {display_name}: {e}")

        except OSError as e:
            error = f"I/O error reading file: {e}"
            status = FileVerificationStatus.ERROR
            logger.error(f"I/O error verifying {display_name}: {e}")

        except Exception as e:
            error = f"Unexpected error during verification: {e}"
            status = FileVerificationStatus.ERROR
            logger.exception(f"Unexpected error verifying {display_name}")

        # Calculate duration
        duration = time.time() - start_time

        # Create result
        result = FileVerificationResult(
            filename=display_name,
            file_path=file_path,
            status=status,
            md5_expected=md5_expected,
            md5_computed=md5_computed,
            duration_seconds=duration,
            error=error,
            timestamp=time.time()
        )

        return result

    def _compute_md5(self, file_path: Path) -> str:
        """
        Compute MD5 hash of a file using chunked reads.

        Uses 1MB chunk reads for efficient processing of large files.
        This balances memory usage with read performance.

        Args:
            file_path: Path to the file to hash

        Returns:
            MD5 hash as lowercase hexadecimal string

        Raises:
            OSError: If file cannot be read
            PermissionError: If file access is denied
        """
        md5_hash = hashlib.md5()
        bytes_read = 0
        file_size = file_path.stat().st_size

        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                md5_hash.update(chunk)
                bytes_read += len(chunk)

                # Log progress for large files (every 100MB)
                if file_size > 100 * 1024 * 1024:  # 100MB threshold
                    if bytes_read % (100 * 1024 * 1024) == 0:
                        progress = (bytes_read / file_size) * 100
                        logger.debug(
                            f"MD5 progress {file_path.name}: "
                            f"{bytes_read / (1024*1024):.0f}MB / "
                            f"{file_size / (1024*1024):.0f}MB ({progress:.1f}%)"
                        )

        return md5_hash.hexdigest()

    def _on_future_done(self, future: Future) -> None:
        """
        Callback invoked when a verification future completes.

        Updates internal tracking state and invokes the user's result callback.

        Args:
            future: The completed Future object
        """
        with self._lock:
            # Get the file path associated with this future
            file_path = self._futures.pop(future, None)

            # Remove from pending set
            if file_path:
                self._pending_paths.discard(file_path)

            # Get the result
            try:
                result = future.result()
            except Exception as e:
                # This shouldn't happen since worker catches exceptions,
                # but handle it just in case
                logger.exception(f"Future raised unexpected exception")
                if file_path:
                    result = FileVerificationResult(
                        filename=file_path.name,
                        file_path=file_path,
                        status=FileVerificationStatus.ERROR,
                        md5_expected=None,
                        md5_computed=None,
                        duration_seconds=0.0,
                        error=f"Internal error: {e}",
                        timestamp=time.time()
                    )
                else:
                    return

            # Store result
            self._results.append(result)

        # Invoke user callback (outside lock to avoid deadlocks)
        if self._on_result:
            try:
                self._on_result(result)
            except Exception as e:
                logger.exception(f"Error in on_result callback: {e}")

    def wait_for_all(self, timeout: float = 300.0) -> List[FileVerificationResult]:
        """
        Wait for all pending verifications to complete.

        Blocks until all queued files have been verified or the timeout
        is reached. Returns all results collected so far.

        Args:
            timeout: Maximum time to wait in seconds. Default is 300 (5 minutes).

        Returns:
            List of all FileVerificationResult objects collected.

        Note:
            If timeout is reached before all verifications complete, the
            returned list will contain only results that completed in time.
            Pending verifications will continue in the background.
        """
        logger.info(f"Waiting for all verifications (timeout={timeout}s)")
        start_time = time.time()

        with self._lock:
            futures_to_wait = list(self._futures.keys())

        if not futures_to_wait:
            logger.debug("No pending verifications")
            with self._lock:
                return list(self._results)

        # Wait for futures to complete
        try:
            for future in as_completed(futures_to_wait, timeout=timeout):
                # Results are processed by _on_future_done callback
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    logger.warning("Timeout reached while waiting for verifications")
                    break
        except TimeoutError:
            logger.warning(f"Timeout ({timeout}s) reached waiting for verifications")

        elapsed = time.time() - start_time
        logger.info(f"Wait completed in {elapsed:.2f}s")

        with self._lock:
            return list(self._results)

    def get_all_results(self) -> List[FileVerificationResult]:
        """
        Get all verification results collected so far.

        Returns a copy of the results list. Results are added as verifications
        complete, so this may be called while verifications are still pending.

        Returns:
            List of all FileVerificationResult objects collected so far.
        """
        with self._lock:
            return list(self._results)

    def get_pending_count(self) -> int:
        """
        Get the number of pending verifications.

        Returns:
            Number of files currently queued or being verified.
        """
        with self._lock:
            return len(self._pending_paths)

    def get_completed_count(self) -> int:
        """
        Get the number of completed verifications.

        Returns:
            Number of files that have completed verification.
        """
        with self._lock:
            return len(self._results)

    def get_results_by_status(
        self,
        status: FileVerificationStatus
    ) -> List[FileVerificationResult]:
        """
        Get verification results filtered by status.

        Args:
            status: FileVerificationStatus to filter by

        Returns:
            List of FileVerificationResult with the specified status.
        """
        with self._lock:
            return [r for r in self._results if r.status == status]

    def get_verified_files(self) -> List[FileVerificationResult]:
        """
        Get all successfully verified files.

        Returns:
            List of FileVerificationResult with VERIFIED status.
        """
        return self.get_results_by_status(FileVerificationStatus.VERIFIED)

    def get_damaged_files(self) -> List[FileVerificationResult]:
        """
        Get all files that failed verification (MD5 mismatch).

        Returns:
            List of FileVerificationResult with DAMAGED status.
        """
        return self.get_results_by_status(FileVerificationStatus.DAMAGED)

    def get_error_files(self) -> List[FileVerificationResult]:
        """
        Get all files that encountered errors during verification.

        Returns:
            List of FileVerificationResult with ERROR status.
        """
        return self.get_results_by_status(FileVerificationStatus.ERROR)

    def clear_results(self) -> None:
        """
        Clear all collected results.

        This does not affect pending verifications. New results will
        continue to be collected.
        """
        with self._lock:
            self._results.clear()
            logger.debug("Verification results cleared")

    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Shut down the verification queue.

        Stops accepting new files and optionally waits for pending
        verifications to complete.

        Args:
            wait: If True, wait for pending verifications to complete.
                  If False, cancel pending work immediately.
            timeout: Maximum time to wait for pending work if wait=True.

        Note:
            After shutdown, the queue cannot be restarted. Create a new
            FileVerificationQueue instance if needed.
        """
        logger.info(f"Shutting down FileVerificationQueue (wait={wait})")

        with self._lock:
            if self._shutdown:
                logger.debug("Already shut down")
                return

            self._shutdown = True

            if self._executor is None:
                return

            executor = self._executor
            self._executor = None

        # Shutdown executor
        if wait:
            # Wait for pending work
            executor.shutdown(wait=True)
        else:
            # Cancel pending work (Python 3.9+)
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 doesn't have cancel_futures
                executor.shutdown(wait=False)

        logger.info("FileVerificationQueue shut down")

    def __enter__(self) -> "FileVerificationQueue":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures shutdown is called."""
        self.shutdown(wait=True)

    def __repr__(self) -> str:
        """Return string representation."""
        with self._lock:
            return (
                f"FileVerificationQueue("
                f"threads={self._threads}, "
                f"pending={len(self._pending_paths)}, "
                f"completed={len(self._results)}, "
                f"shutdown={self._shutdown})"
            )
