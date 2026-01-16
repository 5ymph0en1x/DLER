"""
RAM-Based Post-Processor Module
================================

Ultra-fast post-processing pipeline that keeps all data in RAM:
1. Stores downloaded/decoded files in memory (BytesIO)
2. PAR2 verification using pure Python (MD5 checksums)
3. PAR2 repair using GPU-accelerated Reed-Solomon (CUDA/CuPy)
4. Archive extraction from RAM (py7zr for 7z)
5. Final flush to destination disk

Performance targets (RTX 4090 + 96GB RAM):
- PAR2 verification: ~5 GB/s (parallel MD5)
- PAR2 repair: ~15 GB/s (CUDA Reed-Solomon)
- Extraction: ~2 GB/s (from RAM)

Requirements:
- cupy (for GPU acceleration)
- py7zr (for 7z extraction from memory)
"""

from __future__ import annotations

import io
import os
import re
import sys
import struct
import hashlib
import logging
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, BinaryIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

# Import RAM RAR extractor for TRUE RAM extraction
RAM_RAR_AVAILABLE = False
try:
    from src.core.ram_rar import RamRarExtractor, extract_multipart_rar_to_disk
    RAM_RAR_AVAILABLE = RamRarExtractor.is_available()
    if RAM_RAR_AVAILABLE:
        logger.info("[RAM] UnRAR DLL loaded - TRUE RAM extraction enabled")
    else:
        logger.info("[RAM] UnRAR DLL not found - will fall back to 7z")
except ImportError as e:
    logger.debug(f"[RAM] ram_rar module not available: {e}")

# Import Adaptive Extractor for intelligent extraction
try:
    from src.core.adaptive_extractor import (
        ReleaseClassifier, ExtractionStrategy, AdaptiveExtractor,
        ExtractionPlan, ReleaseType
    )
    ADAPTIVE_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"[RAM] adaptive_extractor module not available: {e}")
    ADAPTIVE_EXTRACTOR_AVAILABLE = False

# Try to import GPU libraries
CUPY_AVAILABLE = False
cp = None
try:
    import cupy as cp

    # Log CuPy version for diagnostics
    cupy_version = getattr(cp, '__version__', 'unknown')
    logger.info(f"CuPy version: {cupy_version}")

    # CuPy v14+ API: cupy.cuda.is_available() is the correct check
    # But we also try direct device enumeration as a more reliable fallback
    cuda_available = False
    detection_method = "none"

    # Method 1: Try direct device count via runtime API (most reliable)
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            cuda_available = True
            detection_method = f"runtime.getDeviceCount={device_count}"
    except Exception as e:
        logger.debug(f"getDeviceCount failed: {e}")

    # Method 2: Try cp.cuda.is_available() (may return False even when CUDA works)
    if not cuda_available:
        try:
            if hasattr(cp, 'cuda') and hasattr(cp.cuda, 'is_available'):
                if cp.cuda.is_available():
                    cuda_available = True
                    detection_method = "cuda.is_available()"
        except Exception as e:
            logger.debug(f"cuda.is_available() failed: {e}")

    # Method 3: Try to actually create a CuPy array (definitive test)
    if not cuda_available:
        try:
            test_array = cp.zeros(10, dtype=cp.float32)
            _ = test_array.sum()  # Force computation
            del test_array
            cuda_available = True
            detection_method = "array_test"
            # Get device count after successful array test
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
            except Exception:
                device_count = 1  # At least one device works
        except Exception as e:
            logger.debug(f"CuPy array test failed: {e}")

    if cuda_available:
        CUPY_AVAILABLE = True
        logger.info(f"CuPy + CUDA available via {detection_method}: {device_count} GPU(s)")
        # Log device info
        try:
            for i in range(device_count):
                device = cp.cuda.Device(i)
                props = device.attributes
                # Get device name via runtime
                name = cp.cuda.runtime.getDeviceProperties(i).get('name', b'Unknown')
                if isinstance(name, bytes):
                    name = name.decode('utf-8', errors='ignore').rstrip('\x00')
                mem_total = device.mem_info[1] / (1024**3)  # GB
                logger.info(f"  GPU {i}: {name} ({mem_total:.1f} GB)")
        except Exception as e:
            logger.debug(f"Could not get device info: {e}")
    else:
        logger.warning("CuPy imported but CUDA not available (GPU acceleration disabled)")
        # Additional diagnostics
        try:
            logger.info(f"  cp.cuda module present: {hasattr(cp, 'cuda')}")
            if hasattr(cp, 'cuda'):
                logger.info(f"  cp.cuda.is_available exists: {hasattr(cp.cuda, 'is_available')}")
                if hasattr(cp.cuda, 'is_available'):
                    logger.info(f"  cp.cuda.is_available() = {cp.cuda.is_available()}")
        except Exception as e:
            logger.debug(f"Diagnostics error: {e}")

except ImportError as e:
    logger.warning(f"CuPy not available: {e}")
except Exception as e:
    logger.warning(f"CuPy initialization error: {e}")
    import traceback
    logger.debug(traceback.format_exc())

# Try to import py7zr for in-memory 7z extraction
try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False
    logger.warning("py7zr not available: RAM-based 7z extraction disabled")


@dataclass
class RamFile:
    """A file stored in RAM."""
    filename: str
    data: io.BytesIO
    size: int  # Pre-allocated size (may include yEnc overhead)
    actual_size: int = 0  # Actual bytes written (max position reached)
    md5_expected: Optional[str] = None  # From PAR2
    md5_computed: Optional[str] = None
    is_verified: bool = False
    is_damaged: bool = False

    @property
    def current_size(self) -> int:
        """Get current data size."""
        return self.data.getbuffer().nbytes

    def update_actual_size(self, position: int, length: int) -> None:
        """Track the maximum position written to determine actual file size."""
        end_pos = position + length
        if end_pos > self.actual_size:
            self.actual_size = end_pos


@dataclass
class Par2SliceChecksum:
    """Checksum for a single slice/block of a file."""
    md5: bytes      # 16-byte MD5 of the slice
    crc32: int      # 32-bit CRC of the slice


@dataclass
class Par2FileInfo:
    """Information about a file from PAR2."""
    filename: str
    file_id: bytes  # 16-byte MD5 hash
    md5_hash: bytes  # 16-byte MD5 of file
    md5_16k: bytes   # 16-byte MD5 of first 16KB
    size: int
    slice_checksums: List[Par2SliceChecksum] = field(default_factory=list)  # Per-block checksums


@dataclass
class Par2RecoveryBlock:
    """A PAR2 recovery block for Reed-Solomon repair."""
    exponent: int
    data: bytes
    length: int


class RamBuffer:
    """
    Thread-safe RAM buffer for downloaded files.

    Stores all files in memory as BytesIO objects,
    tracking total memory usage against a threshold.
    """

    def __init__(self, max_size_mb: int = 32768):
        """
        Args:
            max_size_mb: Maximum total size in MB (default 32 GB)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._files: Dict[str, RamFile] = {}
        self._lock = threading.Lock()
        self._total_size = 0

    @property
    def total_size(self) -> int:
        """Total bytes stored in RAM."""
        return self._total_size

    @property
    def total_size_mb(self) -> float:
        """Total MB stored in RAM."""
        return self._total_size / (1024 * 1024)

    @property
    def usage_percent(self) -> float:
        """Percentage of max size used."""
        return (self._total_size / self.max_size_bytes) * 100

    def can_fit(self, size: int) -> bool:
        """Check if we can fit more data."""
        return self._total_size + size <= self.max_size_bytes

    def create_file(self, key: str, size: int, display_name: Optional[str] = None) -> Optional[RamFile]:
        """
        Create a new file buffer in RAM.

        Args:
            key: Unique key for the dict (may include __idx suffix for uniqueness)
            size: Expected size in bytes
            display_name: Actual filename to use for flush (default: same as key)

        Returns:
            RamFile if successful, None if would exceed max size
        """
        with self._lock:
            if not self.can_fit(size):
                logger.warning(f"Cannot fit {key} ({size/1024/1024:.1f} MB) in RAM buffer")
                return None

            # Pre-allocate BytesIO with expected size
            buffer = io.BytesIO()
            buffer.write(b'\x00' * size)  # Pre-fill with zeros
            buffer.seek(0)

            # Use display_name for actual filename, key for dict lookup
            actual_filename = display_name if display_name else key
            ram_file = RamFile(
                filename=actual_filename,
                data=buffer,
                size=size
            )

            self._files[key] = ram_file
            self._total_size += size

            logger.info(f"[RAM] Created buffer: {key} ({size} bytes)")
            return ram_file

    def write_at(self, filename: str, position: int, data: bytes) -> bool:
        """
        Write data at a specific position in a file.

        Args:
            filename: Target file
            position: Byte offset
            data: Data to write

        Returns:
            True if successful
        """
        with self._lock:
            if filename not in self._files:
                return False

            ram_file = self._files[filename]
            if position + len(data) > ram_file.size:
                logger.warning(f"Write exceeds file size: {filename}")
                return False

            ram_file.data.seek(position)
            ram_file.data.write(data)
            return True

    def get_file(self, filename: str) -> Optional[RamFile]:
        """Get a file from the buffer."""
        return self._files.get(filename)

    def get_all_files(self) -> Dict[str, RamFile]:
        """Get all files in the buffer."""
        return self._files.copy()

    def remove_file(self, filename: str) -> bool:
        """
        Remove a single file from the RAM buffer.

        Args:
            filename: The file key to remove

        Returns:
            True if file was removed, False if not found
        """
        with self._lock:
            if filename not in self._files:
                return False
            ram_file = self._files[filename]
            self._total_size -= ram_file.size
            ram_file.data.close()
            del self._files[filename]
            return True

    def get_file_data(self, filename: str) -> Optional[bytes]:
        """Get the raw bytes of a file (using actual_size to avoid trailing zeros)."""
        ram_file = self._files.get(filename)
        if ram_file:
            ram_file.data.seek(0)
            # Use actual_size to avoid returning trailing zeros from pre-allocation
            read_size = ram_file.actual_size if ram_file.actual_size > 0 else ram_file.size
            return ram_file.data.read(read_size)
        return None

    def get_diagnostics(self) -> Dict[str, dict]:
        """
        Get diagnostic info about all files in buffer.

        Returns dict of filename -> {size, actual_size, non_zero_bytes, first_16_hex}
        """
        result = {}
        with self._lock:
            for filename, ram_file in self._files.items():
                ram_file.data.seek(0)
                first_16 = ram_file.data.read(16)
                ram_file.data.seek(0)
                # Count non-zero bytes in first 4KB
                first_4k = ram_file.data.read(4096)
                non_zero = sum(1 for b in first_4k if b != 0)
                ram_file.data.seek(0)

                result[filename] = {
                    'size': ram_file.size,
                    'actual_size': ram_file.actual_size,
                    'non_zero_4k': non_zero,
                    'first_16_hex': first_16.hex()
                }
        return result

    def log_diagnostics(self) -> None:
        """Log diagnostic info about buffer contents."""
        diag = self.get_diagnostics()
        logger.info(f"[RAM DIAG] {len(diag)} files in buffer, total {self.total_size_mb:.1f} MB")
        for filename, info in list(diag.items())[:5]:  # First 5 files
            logger.info(f"[RAM DIAG]   {filename}: pre_alloc={info['size']}, actual={info['actual_size']}, "
                       f"non_zero_4k={info['non_zero_4k']}, first_16={info['first_16_hex']}")
        if len(diag) > 5:
            logger.info(f"[RAM DIAG]   ... and {len(diag) - 5} more files")

    def clear(self) -> None:
        """Clear all files from RAM."""
        with self._lock:
            for ram_file in self._files.values():
                ram_file.data.close()
            self._files.clear()
            self._total_size = 0
            logger.info("RAM buffer cleared")

    def flush_to_disk(self, output_dir: Path) -> int:
        """
        Write all RAM files to disk.

        Args:
            output_dir: Directory to write files to

        Returns:
            Number of files written
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = 0

        with self._lock:
            for filename, ram_file in self._files.items():
                try:
                    file_path = output_dir / filename
                    ram_file.data.seek(0)
                    # Use actual_size to avoid writing trailing zeros from pre-allocation
                    read_size = ram_file.actual_size if ram_file.actual_size > 0 else ram_file.size
                    with open(file_path, 'wb') as f:
                        f.write(ram_file.data.read(read_size))
                    written += 1
                except Exception as e:
                    logger.error(f"Failed to flush {filename} to disk: {e}")

        logger.info(f"Flushed {written} files to {output_dir}")
        return written


class Par2Parser:
    """
    Pure Python PAR2 file parser.

    Parses PAR2 files to extract:
    - File descriptions (names, sizes, MD5 hashes)
    - Recovery blocks (for Reed-Solomon repair)
    """

    # PAR2 packet signatures
    PACKET_HEADER = b'PAR2\x00PKT'
    PACKET_MAIN = b'PAR 2.0\x00Main\x00\x00\x00\x00'
    PACKET_FILE_DESC = b'PAR 2.0\x00FileDesc'
    PACKET_IFSC = b'PAR 2.0\x00IFSC\x00\x00\x00\x00'
    PACKET_RECOVERY = b'PAR 2.0\x00RecvSlic'
    PACKET_CREATOR = b'PAR 2.0\x00Creator\x00'

    def __init__(self):
        self.files: Dict[bytes, Par2FileInfo] = {}  # file_id -> info
        self.recovery_blocks: List[Par2RecoveryBlock] = []
        self.block_size: int = 0
        self.recovery_set_id: bytes = b''

    def parse(self, data: bytes) -> bool:
        """
        Parse a PAR2 file from bytes.

        Args:
            data: Raw PAR2 file content

        Returns:
            True if parsing succeeded
        """
        pos = 0
        data_len = len(data)

        while pos < data_len - 8:
            # Look for packet header
            if data[pos:pos+8] != self.PACKET_HEADER:
                pos += 1
                continue

            # Parse packet header
            # 8 bytes: magic
            # 8 bytes: length (including header)
            # 16 bytes: packet hash (MD5)
            # 16 bytes: recovery set ID
            # 16 bytes: packet type

            if pos + 64 > data_len:
                break

            length = struct.unpack('<Q', data[pos+8:pos+16])[0]
            if pos + length > data_len:
                break

            packet_type = data[pos+48:pos+64]
            packet_data = data[pos+64:pos+length]

            if packet_type == self.PACKET_MAIN:
                self._parse_main(packet_data)
            elif packet_type == self.PACKET_FILE_DESC:
                self._parse_file_desc(packet_data)
            elif packet_type == self.PACKET_IFSC:
                self._parse_ifsc(packet_data)
            elif packet_type == self.PACKET_RECOVERY:
                self._parse_recovery(packet_data, length - 64)

            pos += length

        # Count total slice checksums
        total_slices = sum(len(f.slice_checksums) for f in self.files.values())
        logger.info(f"PAR2 parsed: {len(self.files)} files, {len(self.recovery_blocks)} recovery blocks, "
                   f"{total_slices} slice checksums")
        return len(self.files) > 0

    def _parse_main(self, data: bytes) -> None:
        """Parse main packet to get block size."""
        if len(data) < 8:
            return
        # Slice size (block size) at offset 0
        self.block_size = struct.unpack('<Q', data[0:8])[0]
        logger.debug(f"PAR2 block size: {self.block_size}")

    def _parse_file_desc(self, data: bytes) -> None:
        """Parse file description packet."""
        if len(data) < 56:
            return

        # 16 bytes: file ID (MD5 of file info)
        # 16 bytes: MD5 hash of file
        # 16 bytes: MD5 hash of first 16KB
        # 8 bytes: file size
        # Remaining: filename (null-terminated, padded to 4 bytes)

        file_id = data[0:16]
        md5_hash = data[16:32]
        md5_16k = data[32:48]
        file_size = struct.unpack('<Q', data[48:56])[0]

        # Extract filename (null-terminated UTF-8)
        filename_bytes = data[56:]
        null_pos = filename_bytes.find(b'\x00')
        if null_pos >= 0:
            filename = filename_bytes[:null_pos].decode('utf-8', errors='replace')
        else:
            filename = filename_bytes.decode('utf-8', errors='replace')

        self.files[file_id] = Par2FileInfo(
            filename=filename,
            file_id=file_id,
            md5_hash=md5_hash,
            md5_16k=md5_16k,
            size=file_size
        )
        logger.debug(f"PAR2 file: {filename} ({file_size} bytes)")

    def _parse_ifsc(self, data: bytes) -> None:
        """
        Parse Input File Slice Checksum (IFSC) packet.

        IFSC packets contain per-block MD5 and CRC32 checksums for a file.
        This allows us to identify exactly which blocks are damaged.

        Structure:
        - 16 bytes: file ID
        - For each slice:
          - 16 bytes: MD5 hash
          - 4 bytes: CRC32
        """
        if len(data) < 16:
            return

        file_id = data[0:16]

        # Find the file this IFSC belongs to
        if file_id not in self.files:
            logger.debug(f"IFSC for unknown file ID: {file_id.hex()[:16]}")
            return

        file_info = self.files[file_id]

        # Parse slice checksums (20 bytes each: 16 MD5 + 4 CRC32)
        slice_data = data[16:]
        num_slices = len(slice_data) // 20

        for i in range(num_slices):
            offset = i * 20
            slice_md5 = slice_data[offset:offset+16]
            slice_crc = struct.unpack('<I', slice_data[offset+16:offset+20])[0]

            file_info.slice_checksums.append(Par2SliceChecksum(
                md5=slice_md5,
                crc32=slice_crc
            ))

        logger.debug(f"IFSC: {file_info.filename} - {num_slices} slice checksums")

    def _parse_recovery(self, data: bytes, length: int) -> None:
        """Parse recovery slice packet."""
        if len(data) < 4:
            return

        # 4 bytes: exponent
        exponent = struct.unpack('<I', data[0:4])[0]
        recovery_data = data[4:]

        self.recovery_blocks.append(Par2RecoveryBlock(
            exponent=exponent,
            data=recovery_data,
            length=len(recovery_data)
        ))


class RamVerifier:
    """
    Parallel MD5 verification for files in RAM.

    Uses ThreadPoolExecutor for parallel hash computation.
    """

    def __init__(self, threads: int = 0):
        """
        Args:
            threads: Number of threads (0 = auto)
        """
        self.threads = threads or min(os.cpu_count() or 4, 16)

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        """
        Normalize filename for matching PAR2 entries.

        Handles common Usenet variations:
        - Spaces ↔ underscores (NZB indexers often convert spaces)
        - Case differences
        """
        normalized = filename.lower()
        # Normalize underscores to spaces for comparison
        normalized = normalized.replace('_', ' ')
        return normalized

    def verify_files(
        self,
        ram_buffer: RamBuffer,
        par2_info: Dict[bytes, Par2FileInfo],
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Verify all files against PAR2 checksums.

        Args:
            ram_buffer: Buffer containing files
            par2_info: File info from PAR2 (file_id -> Par2FileInfo)
            on_progress: Callback(filename, percent)

        Returns:
            (verified_files, damaged_files) tuple
        """
        verified = []
        damaged = []

        # Build normalized filename -> expected MD5 mapping
        # PAR2 stores original filenames (often with spaces)
        expected_md5: Dict[str, bytes] = {}
        for file_info in par2_info.values():
            normalized = self._normalize_filename(file_info.filename)
            expected_md5[normalized] = file_info.md5_hash

        files_to_verify = []
        for dict_key, ram_file in ram_buffer.get_all_files().items():
            # Use ram_file.filename (original name) not dict_key (has __idxXXXX suffix)
            original_filename = ram_file.filename
            fname_normalized = self._normalize_filename(original_filename)
            if fname_normalized in expected_md5:
                files_to_verify.append((dict_key, ram_file, expected_md5[fname_normalized]))

        if not files_to_verify:
            # Debug: show what we have vs what PAR2 expects
            ram_files = [rf.filename for rf in ram_buffer.get_all_files().values()]
            par2_files = [f.filename for f in par2_info.values()]
            logger.warning(f"No files to verify - possible filename mismatch")
            logger.debug(f"RAM files (first 5): {ram_files[:5]}")
            logger.debug(f"PAR2 expects (first 5): {par2_files[:5]}")
            logger.debug(f"RAM normalized: {[self._normalize_filename(f) for f in ram_files[:3]]}")
            logger.debug(f"PAR2 normalized: {[self._normalize_filename(f) for f in par2_files[:3]]}")
            return verified, damaged

        logger.info(f"Verifying {len(files_to_verify)} files with {self.threads} threads")
        start_time = time.time()

        def verify_one(args) -> Tuple[str, bool, str]:
            filename, ram_file, expected = args
            ram_file.data.seek(0)

            # Use actual_size if tracked, otherwise use buffer size
            # This handles the case where NZB bytes attribute includes yEnc overhead
            read_size = ram_file.actual_size if ram_file.actual_size > 0 else ram_file.size
            data = ram_file.data.read(read_size)

            computed = hashlib.md5(data).digest()
            ram_file.md5_computed = computed.hex()
            ram_file.md5_expected = expected.hex()
            ram_file.is_verified = (computed == expected)
            ram_file.is_damaged = not ram_file.is_verified

            # Debug info for damaged files
            if ram_file.is_damaged:
                # Check if data is all zeros (not written)
                non_zero = sum(1 for b in data[:4096] if b != 0)
                logger.debug(f"[VERIFY] {filename}: expected={expected.hex()[:16]}... computed={computed.hex()[:16]}... "
                            f"size={len(data)} actual_size={ram_file.actual_size} "
                            f"pre_alloc={ram_file.size} non_zero_first_4k={non_zero}")

            return filename, ram_file.is_verified, ram_file.md5_computed

        completed = 0
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {executor.submit(verify_one, args): args[0] for args in files_to_verify}

            for future in as_completed(futures):
                filename, is_ok, computed_md5 = future.result()
                completed += 1

                if is_ok:
                    verified.append(filename)
                else:
                    damaged.append(filename)
                    ram_file = ram_buffer.get_file(filename)
                    expected_hex = ram_file.md5_expected if ram_file else "?"
                    logger.warning(f"Damaged: {filename} (expected={expected_hex[:16]}... got={computed_md5[:16]}...)")

                if on_progress:
                    on_progress(filename, completed / len(files_to_verify) * 100)

        elapsed = time.time() - start_time
        total_size = sum(f[1].size for f in files_to_verify)
        speed = total_size / elapsed / (1024 * 1024)

        logger.info(f"Verification complete: {len(verified)} OK, {len(damaged)} damaged "
                   f"({speed:.1f} MB/s)")

        return verified, damaged


class GpuReedSolomon:
    """
    GPU-accelerated Reed-Solomon repair using CuPy/CUDA.

    Implements GF(2^8) arithmetic on GPU for massive parallelism.
    PAR2 uses Reed-Solomon over GF(2^8) with primitive polynomial 0x11d.

    Performance: ~10-15 GB/s on RTX 4090 for repair operations.
    """

    # GF(2^8) primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 = 0x11d
    GF_PRIMITIVE = 0x11d

    # CUDA kernel for GF(2^8) matrix-vector multiplication
    _GF_MATMUL_KERNEL = """
    extern "C" __global__
    void gf_matmul(
        const unsigned char* matrix,    // [rows x cols] matrix
        const unsigned char* vector,    // [cols x block_size] input blocks
        unsigned char* result,          // [rows x block_size] output blocks
        const unsigned char* gf_log,    // GF log table
        const unsigned char* gf_exp,    // GF exp table
        int rows, int cols, int block_size
    ) {
        // Each thread handles one byte position across all rows
        int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (byte_idx >= block_size) return;

        for (int row = 0; row < rows; row++) {
            unsigned char acc = 0;
            for (int col = 0; col < cols; col++) {
                unsigned char m = matrix[row * cols + col];
                unsigned char v = vector[col * block_size + byte_idx];

                // GF multiplication: a * b = exp[log[a] + log[b]]
                if (m != 0 && v != 0) {
                    int log_sum = gf_log[m] + gf_log[v];
                    acc ^= gf_exp[log_sum];  // XOR for GF addition
                }
            }
            result[row * block_size + byte_idx] = acc;
        }
    }
    """

    # CUDA kernel for parallel XOR of blocks
    _XOR_KERNEL = """
    extern "C" __global__
    void xor_blocks(
        unsigned char* target,
        const unsigned char* source,
        int size
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            target[idx] ^= source[idx];
        }
    }
    """

    # CUDA kernel for recovery adjustment (subtract present block contributions)
    # This is the CRITICAL optimization - moves O(recovery * present * block_size) to GPU
    _RECOVERY_ADJUST_KERNEL = """
    extern "C" __global__
    void recovery_adjust(
        unsigned char* recovery_data,       // [num_recovery x block_size] recovery blocks to adjust
        const unsigned char* present_data,  // [num_present x block_size] present data blocks
        const unsigned char* coefficients,  // [num_recovery x num_present] GF coefficients
        const unsigned char* gf_log,        // GF log table
        const unsigned char* gf_exp,        // GF exp table
        int num_recovery, int num_present, int block_size
    ) {
        // Each thread handles one byte position
        int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (byte_idx >= block_size) return;

        // For each recovery block
        for (int rec = 0; rec < num_recovery; rec++) {
            unsigned char adjustment = 0;

            // XOR contribution from all present blocks
            for (int pres = 0; pres < num_present; pres++) {
                unsigned char coeff = coefficients[rec * num_present + pres];
                unsigned char data = present_data[pres * block_size + byte_idx];

                // GF multiplication: coeff * data
                if (coeff != 0 && data != 0) {
                    int log_sum = gf_log[coeff] + gf_log[data];
                    adjustment ^= gf_exp[log_sum];
                }
            }

            // XOR adjustment into recovery block
            recovery_data[rec * block_size + byte_idx] ^= adjustment;
        }
    }
    """

    def __init__(self, device_id: int = 0):
        """
        Args:
            device_id: CUDA device ID
        """
        self.device_id = device_id
        self._initialized = False
        self._gf_log = None
        self._gf_exp = None
        self._gf_log_cpu = None
        self._gf_exp_cpu = None
        self._matmul_kernel = None
        self._xor_kernel = None
        self._recovery_adjust_kernel = None

        if CUPY_AVAILABLE:
            self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize GPU, GF(2^8) lookup tables, and CUDA kernels."""
        try:
            # CuPy v14+ API: Select device using .use() method
            cp.cuda.Device(self.device_id).use()
            logger.debug(f"Selected CUDA device {self.device_id}")

            # Get GPU info using runtime API
            try:
                props = cp.cuda.runtime.getDeviceProperties(self.device_id)
                # Handle both bytes and string for GPU name
                gpu_name = props['name']
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8', errors='replace')
                gpu_mem = props['totalGlobalMem'] / (1024**3)
                logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            except Exception as e:
                logger.info(f"GPU: Device {self.device_id} (info unavailable: {e})")

            # Generate GF(2^8) log and exp tables
            log_table = [0] * 256
            exp_table = [0] * 512

            x = 1
            for i in range(255):
                exp_table[i] = x
                exp_table[i + 255] = x  # Duplicate for mod-free access
                log_table[x] = i
                x <<= 1
                if x & 0x100:
                    x ^= self.GF_PRIMITIVE

            # Store CPU copies for matrix operations
            self._gf_log_cpu = log_table
            self._gf_exp_cpu = exp_table

            # Transfer to GPU
            self._gf_log = cp.array(log_table, dtype=cp.uint8)
            self._gf_exp = cp.array(exp_table, dtype=cp.uint8)

            # Compile CUDA kernels
            self._matmul_kernel = cp.RawKernel(self._GF_MATMUL_KERNEL, 'gf_matmul')
            self._xor_kernel = cp.RawKernel(self._XOR_KERNEL, 'xor_blocks')
            self._recovery_adjust_kernel = cp.RawKernel(self._RECOVERY_ADJUST_KERNEL, 'recovery_adjust')

            self._initialized = True
            logger.info(f"GPU Reed-Solomon initialized on device {self.device_id}")

            # Warmup kernels to avoid JIT latency on first PAR2 repair
            self._warmup_kernels()

        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._initialized = False

    def _warmup_kernels(self) -> None:
        """
        Precompile CUDA kernels to avoid JIT latency on first use.

        CuPy compiles kernels Just-In-Time on first call, adding 10-50ms latency.
        Warming up during init ensures smooth operation during actual PAR2 repair.
        """
        try:
            warmup_start = time.time()

            # Small test data for warmup (16x16 = 256 bytes)
            test_size = 16
            dummy_matrix = cp.zeros(test_size * test_size, dtype=cp.uint8)
            dummy_data = cp.zeros((test_size, test_size), dtype=cp.uint8)
            dummy_result = cp.zeros((test_size, test_size), dtype=cp.uint8)

            # Warmup matmul kernel
            threads_per_block = 16
            blocks_per_grid = 1
            self._matmul_kernel(
                (blocks_per_grid,), (threads_per_block,),
                (dummy_matrix, dummy_data.ravel(), dummy_result.ravel(),
                 self._gf_log, self._gf_exp,
                 test_size, test_size, test_size)
            )

            # Warmup XOR kernel
            self._xor_kernel(
                (blocks_per_grid,), (threads_per_block,),
                (dummy_result.ravel(), dummy_data.ravel(), test_size * test_size)
            )

            # Warmup recovery adjustment kernel
            dummy_recovery = cp.zeros(test_size * test_size, dtype=cp.uint8)
            dummy_present = cp.zeros(test_size * test_size, dtype=cp.uint8)
            dummy_coeffs = cp.zeros(test_size * test_size, dtype=cp.uint8)
            self._recovery_adjust_kernel(
                (blocks_per_grid,), (threads_per_block,),
                (dummy_recovery, dummy_present, dummy_coeffs,
                 self._gf_log, self._gf_exp,
                 test_size, test_size, test_size)
            )
            del dummy_recovery, dummy_present, dummy_coeffs

            # Synchronize to ensure compilation is complete
            cp.cuda.Stream.null.synchronize()

            # Cleanup warmup data
            del dummy_matrix, dummy_data, dummy_result
            self._free_vram()

            warmup_elapsed = (time.time() - warmup_start) * 1000
            logger.info(f"CUDA kernels warmed up in {warmup_elapsed:.1f}ms")

        except Exception as e:
            logger.warning(f"Kernel warmup failed (non-critical): {e}")

    def _log_vram_usage(self, context: str = "") -> None:
        """Log current VRAM usage for debugging."""
        if not CUPY_AVAILABLE:
            return
        try:
            mempool = cp.get_default_memory_pool()
            used_mb = mempool.used_bytes() / (1024 * 1024)
            total_mb = mempool.total_bytes() / (1024 * 1024)

            # Also get device memory info
            device = cp.cuda.Device(self.device_id)
            free_bytes, total_bytes = device.mem_info
            device_used_mb = (total_bytes - free_bytes) / (1024 * 1024)
            device_total_mb = total_bytes / (1024 * 1024)

            prefix = f"[{context}] " if context else ""
            logger.debug(f"{prefix}VRAM: pool={used_mb:.1f}/{total_mb:.1f} MB, "
                        f"device={device_used_mb:.1f}/{device_total_mb:.1f} MB")
        except Exception as e:
            logger.debug(f"Could not get VRAM usage: {e}")

    def _free_vram(self) -> None:
        """
        Release unused VRAM back to the system.

        CuPy uses a memory pool to avoid repeated allocations.
        Calling this between operations releases cached memory.
        """
        if not CUPY_AVAILABLE:
            return
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

            # Free all unused blocks
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

            logger.debug("VRAM memory pool cleared")
        except Exception as e:
            logger.debug(f"Could not free VRAM: {e}")

    @property
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return CUPY_AVAILABLE and self._initialized

    def _gf_mul(self, a: int, b: int) -> int:
        """GF(2^8) multiplication (CPU, for matrix building)."""
        if a == 0 or b == 0:
            return 0
        return self._gf_exp_cpu[self._gf_log_cpu[a] + self._gf_log_cpu[b]]

    def _gf_div(self, a: int, b: int) -> int:
        """GF(2^8) division (CPU, for matrix inversion)."""
        if b == 0:
            raise ValueError("Division by zero in GF(2^8)")
        if a == 0:
            return 0
        return self._gf_exp_cpu[(self._gf_log_cpu[a] - self._gf_log_cpu[b]) % 255]

    def _gf_pow(self, base: int, exp: int) -> int:
        """GF(2^8) exponentiation (CPU)."""
        if exp == 0:
            return 1
        if base == 0:
            return 0
        return self._gf_exp_cpu[(self._gf_log_cpu[base] * exp) % 255]

    def _build_vandermonde_row(self, exponent: int, num_cols: int) -> List[int]:
        """Build a row of the Vandermonde matrix for given exponent."""
        # Row i: [1, g^i, g^(2i), g^(3i), ...]
        # Where g = 2 is the generator in GF(2^8)
        row = []
        for col in range(num_cols):
            row.append(self._gf_pow(2, exponent * col))
        return row

    def _invert_matrix_gf(self, matrix: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Invert a matrix over GF(2^8) using Gaussian elimination.

        Args:
            matrix: Square matrix as list of lists

        Returns:
            Inverted matrix or None if singular
        """
        n = len(matrix)
        if n == 0 or len(matrix[0]) != n:
            return None

        # Create augmented matrix [A | I]
        aug = [row[:] + [1 if i == j else 0 for j in range(n)]
               for i, row in enumerate(matrix)]

        # Forward elimination
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if aug[row][col] != 0:
                    pivot_row = row
                    break

            if pivot_row is None:
                logger.error(f"Matrix is singular at column {col}")
                return None

            # Swap rows if needed
            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            # Scale pivot row
            pivot = aug[col][col]
            pivot_inv = self._gf_div(1, pivot)
            for j in range(2 * n):
                aug[col][j] = self._gf_mul(aug[col][j], pivot_inv)

            # Eliminate column
            for row in range(n):
                if row != col and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] ^= self._gf_mul(factor, aug[col][j])

        # Extract inverse from augmented matrix
        inverse = [row[n:] for row in aug]
        return inverse

    def repair(
        self,
        data_blocks: List[Optional[bytes]],
        recovery_blocks: List[Par2RecoveryBlock],
        block_size: int,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> List[bytes]:
        """
        Repair damaged/missing blocks using GPU-accelerated Reed-Solomon.

        Args:
            data_blocks: List of data blocks (None for missing/damaged)
            recovery_blocks: Recovery blocks from PAR2
            block_size: Size of each block in bytes
            on_progress: Progress callback(percent)

        Returns:
            List of repaired blocks in order of missing_indices
        """
        if not self.is_available:
            logger.error("GPU not available for repair")
            return []

        # Identify missing blocks
        missing_indices = [i for i, b in enumerate(data_blocks) if b is None]
        present_indices = [i for i, b in enumerate(data_blocks) if b is not None]

        if not missing_indices:
            logger.info("No blocks to repair")
            return []

        num_missing = len(missing_indices)
        num_recovery = len(recovery_blocks)

        if num_missing > num_recovery:
            logger.error(f"Not enough recovery blocks: need {num_missing}, have {num_recovery}")
            return []

        logger.info(f"GPU Repair: {num_missing} missing blocks, using {num_missing} recovery blocks")
        start_time = time.time()

        try:
            # Step 1: Build the decoding matrix (CPU)
            # We need to solve: D * x = r
            # Where D is built from Vandermonde rows for missing positions
            # x is the missing data, r is derived from recovery blocks

            if on_progress:
                on_progress(10)

            # Select recovery blocks to use (first num_missing)
            used_recovery = recovery_blocks[:num_missing]

            # Build decoding matrix
            # Each row corresponds to a recovery block
            # Each column corresponds to a missing data block position
            decode_matrix = []
            for rec in used_recovery:
                row = []
                for missing_idx in missing_indices:
                    # Vandermonde coefficient: 2^(exponent * missing_idx)
                    coeff = self._gf_pow(2, rec.exponent * missing_idx)
                    row.append(coeff)
                decode_matrix.append(row)

            if on_progress:
                on_progress(20)

            # Step 2: Invert the decoding matrix (CPU)
            inv_matrix = self._invert_matrix_gf(decode_matrix)
            if inv_matrix is None:
                logger.error("Failed to invert decoding matrix")
                return []

            if on_progress:
                on_progress(30)

            # Step 3: Prepare data for GPU
            # We need to compute: repaired = inv_matrix * (recovery - contribution_from_present)

            # First, compute contribution from present blocks to each recovery
            # recovery_adjusted[i] = recovery[i] XOR sum(present[j] * coeff[i,j])
            # ===== GPU-ACCELERATED RECOVERY ADJUSTMENT =====
            # This replaces the O(recovery * present * block_size) CPU loop with GPU kernel
            num_present = len(present_indices)

            if num_present > 0:
                # Build coefficient matrix [num_missing x num_present]
                coeff_matrix = []
                for rec in used_recovery:
                    row = []
                    for present_idx in present_indices:
                        coeff = self._gf_pow(2, rec.exponent * present_idx)
                        row.append(coeff)
                    coeff_matrix.extend(row)

                # Prepare recovery block data
                recovery_flat = b''.join(rec.data[:block_size] for rec in used_recovery)

                # Prepare present block data
                present_flat = b''.join(
                    data_blocks[idx] if data_blocks[idx] else b'\x00' * block_size
                    for idx in present_indices
                )

                # Transfer to GPU
                recovery_gpu = cp.array(list(recovery_flat), dtype=cp.uint8)
                present_gpu = cp.array(list(present_flat), dtype=cp.uint8)
                coeff_gpu = cp.array(coeff_matrix, dtype=cp.uint8)

                # Launch recovery adjustment kernel
                threads_per_block = 256
                blocks_per_grid = (block_size + threads_per_block - 1) // threads_per_block

                self._recovery_adjust_kernel(
                    (blocks_per_grid,), (threads_per_block,),
                    (recovery_gpu, present_gpu, coeff_gpu,
                     self._gf_log, self._gf_exp,
                     num_missing, num_present, block_size)
                )
                cp.cuda.Stream.null.synchronize()

                # Reshape back to list of blocks
                recovery_adjusted_array = cp.asnumpy(recovery_gpu).reshape(num_missing, block_size)
                recovery_adjusted = [bytes(recovery_adjusted_array[i]) for i in range(num_missing)]

                # Cleanup intermediate GPU arrays
                del recovery_gpu, present_gpu, coeff_gpu
            else:
                # No present blocks - just use recovery blocks directly
                recovery_adjusted = [rec.data[:block_size] for rec in used_recovery]

            if on_progress:
                on_progress(50)

            # Step 4: Transfer to GPU and compute
            # Matrix multiplication: result = inv_matrix * recovery_adjusted
            self._log_vram_usage("before GPU transfer")

            # Convert to numpy/cupy arrays
            inv_matrix_flat = []
            for row in inv_matrix:
                inv_matrix_flat.extend(row)

            matrix_gpu = cp.array(inv_matrix_flat, dtype=cp.uint8)

            # Stack adjusted recovery blocks
            recovery_flat = b''.join(recovery_adjusted)
            recovery_gpu = cp.array(list(recovery_flat), dtype=cp.uint8).reshape(num_missing, block_size)

            result_gpu = cp.zeros((num_missing, block_size), dtype=cp.uint8)

            if on_progress:
                on_progress(60)

            # Launch kernel
            threads_per_block = 256
            blocks_per_grid = (block_size + threads_per_block - 1) // threads_per_block

            self._matmul_kernel(
                (blocks_per_grid,), (threads_per_block,),
                (matrix_gpu, recovery_gpu.ravel(), result_gpu.ravel(),
                 self._gf_log, self._gf_exp,
                 num_missing, num_missing, block_size)
            )

            # Synchronize
            cp.cuda.Stream.null.synchronize()

            if on_progress:
                on_progress(90)

            # Step 5: Transfer results back to CPU
            result_cpu = cp.asnumpy(result_gpu)
            repaired_blocks = [bytes(result_cpu[i]) for i in range(num_missing)]

            # Log VRAM usage after computation
            self._log_vram_usage("after GPU compute")

            # Cleanup GPU memory (release VRAM for other operations)
            del matrix_gpu, recovery_gpu, result_gpu
            self._free_vram()

            elapsed = time.time() - start_time
            speed = (num_missing * block_size) / elapsed / (1024 * 1024)

            logger.info(f"GPU Repair complete: {num_missing} blocks in {elapsed:.2f}s ({speed:.1f} MB/s)")

            if on_progress:
                on_progress(100)

            return repaired_blocks

        except Exception as e:
            logger.error(f"GPU repair failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Cleanup VRAM even on error
            self._free_vram()
            return []

    def repair_file_blocks(
        self,
        file_data: bytes,
        block_size: int,
        damaged_block_indices: List[int],
        recovery_blocks: List[Par2RecoveryBlock],
        on_progress: Optional[Callable[[float], None]] = None
    ) -> Optional[bytes]:
        """
        High-level API to repair a file given its data and damaged block info.

        Args:
            file_data: The file data (with damaged/zeroed blocks)
            block_size: PAR2 block size
            damaged_block_indices: Which blocks are damaged
            recovery_blocks: Recovery blocks from PAR2
            on_progress: Progress callback

        Returns:
            Repaired file data or None on failure
        """
        if not damaged_block_indices:
            return file_data

        # Split file into blocks
        num_blocks = (len(file_data) + block_size - 1) // block_size
        data_blocks: List[Optional[bytes]] = []

        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, len(file_data))
            block = file_data[start:end]

            # Pad last block if needed
            if len(block) < block_size:
                block = block + b'\x00' * (block_size - len(block))

            if i in damaged_block_indices:
                data_blocks.append(None)
            else:
                data_blocks.append(block)

        # Repair
        repaired = self.repair(data_blocks, recovery_blocks, block_size, on_progress)

        if len(repaired) != len(damaged_block_indices):
            logger.error(f"Repair returned wrong number of blocks: {len(repaired)} vs {len(damaged_block_indices)}")
            return None

        # Reconstruct file
        result = bytearray(file_data)
        for i, block_idx in enumerate(damaged_block_indices):
            start = block_idx * block_size
            end = min(start + block_size, len(result))
            result[start:end] = repaired[i][:end - start]

        return bytes(result)


class RamPostProcessor:
    """
    Complete RAM-based post-processing pipeline.

    Orchestrates:
    1. PAR2 parsing
    2. File verification
    3. GPU repair (if needed)
    4. Archive extraction
    5. Final disk flush
    """

    def __init__(
        self,
        ram_buffer: RamBuffer,
        extract_dir: Path,
        gpu_device_id: int = 0,
        verify_threads: int = 0,
        password: Optional[str] = None,
        on_progress: Optional[Callable[[str, float], None]] = None
    ):
        """
        Args:
            ram_buffer: Buffer containing downloaded files
            extract_dir: Final destination directory
            gpu_device_id: CUDA device for repairs
            verify_threads: Threads for MD5 verification
            password: Archive password for encrypted RARs
            on_progress: Progress callback(stage, percent)
        """
        self.ram_buffer = ram_buffer
        self.extract_dir = Path(extract_dir)
        self.password = password
        self.on_progress = on_progress

        if password:
            logger.info(f"[RAM] Archive password configured: {'*' * len(password)}")

        self.par2_parser = Par2Parser()
        self.verifier = RamVerifier(threads=verify_threads)
        self.gpu_rs = GpuReedSolomon(device_id=gpu_device_id)

        # Stats for summary popup
        self.stats_files_verified = 0
        self.stats_files_damaged = 0
        self.stats_files_repaired = 0
        self.stats_files_extracted = 0
        self.stats_files_flushed = 0
        self.stats_flush_bytes = 0
        self.stats_verify_speed_mbs = 0.0
        self.stats_extract_duration = 0.0
        self.stats_flush_duration = 0.0
        self.stats_flush_speed_mbs = 0.0
        self.stats_total_duration = 0.0
        self.stats_total_size_bytes = 0
        self.stats_extract_path: Optional[Path] = None
        self.stats_archives_found = 0  # Number of archives detected
        self.stats_extraction_failed = False  # True if archives existed but extraction failed

    def process(self, release_name: str, extraction_plan: 'ExtractionPlan' = None) -> Tuple[bool, str]:
        """
        Run the complete post-processing pipeline.

        Args:
            release_name: Name for the release subfolder
            extraction_plan: Optional pre-computed ExtractionPlan from classification

        Returns:
            (success, message) tuple
        """
        start_time = time.time()

        try:
            # Step 1: Find and parse PAR2 files
            self._report("Parsing PAR2...", 5)

            # Parse main PAR2 file (contains file descriptions)
            # Use ram_file.filename (original name) not dict_key (has __idxXXXX suffix)
            main_par2_files = [(k, rf) for k, rf in self.ram_buffer.get_all_files().items()
                              if rf.filename.lower().endswith('.par2') and '.vol' not in rf.filename.lower()]

            if main_par2_files:
                dict_key, ram_file = main_par2_files[0]
                par2_data = self.ram_buffer.get_file_data(dict_key)
                if par2_data:
                    self.par2_parser.parse(par2_data)
                    logger.info(f"[PAR2] Parsed main file: {ram_file.filename}")

            # Parse volume PAR2 files (contain recovery blocks)
            vol_par2_files = [(k, rf) for k, rf in self.ram_buffer.get_all_files().items()
                             if rf.filename.lower().endswith('.par2') and '.vol' in rf.filename.lower()]

            if vol_par2_files:
                logger.info(f"[GPU REPAIR] Loading recovery blocks from {len(vol_par2_files)} volume files...")
                for dict_key, ram_file in vol_par2_files:
                    vol_data = self.ram_buffer.get_file_data(dict_key)
                    if vol_data:
                        self.par2_parser.parse(vol_data)
                logger.info(f"[GPU REPAIR] Loaded {len(self.par2_parser.recovery_blocks)} recovery blocks "
                           f"(block_size={self.par2_parser.block_size})")

            # Step 2: Verify files
            self._report("Verifying MD5...", 20)

            # Debug: Log buffer contents before verification
            self.ram_buffer.log_diagnostics()
            logger.info(f"[RAM PP] PAR2 has info for {len(self.par2_parser.files)} files")

            verify_start = time.time()
            verified, damaged = self.verifier.verify_files(
                self.ram_buffer,
                self.par2_parser.files,
                on_progress=lambda f, p: self._report(f"Verifying: {f}", 20 + p * 0.3)
            )
            verify_elapsed = time.time() - verify_start

            # Store verification stats
            self.stats_files_verified = len(verified)
            self.stats_files_damaged = len(damaged)
            total_verified_size = sum(
                len(self.ram_buffer.get_file_data(f) or b'')
                for f in verified
            )
            self.stats_verify_speed_mbs = (total_verified_size / (1024*1024)) / verify_elapsed if verify_elapsed > 0 else 0

            # Step 3: Repair if needed
            if damaged and self.gpu_rs.is_available:
                self._report("GPU repair...", 50)
                repaired_count = self._repair_damaged_files(damaged)
                self.stats_files_repaired = repaired_count
                if repaired_count > 0:
                    logger.info(f"[GPU REPAIR] Successfully repaired {repaired_count}/{len(damaged)} files")
                elif damaged:
                    logger.warning(f"[GPU REPAIR] Could not repair {len(damaged)} files")

            # Step 4: Extract archives (using adaptive extraction if plan provided)
            self._report("Extracting...", 60)
            extract_start = time.time()
            extracted = self._extract_archives(release_name, extraction_plan)
            self.stats_extract_duration = time.time() - extract_start
            self.stats_files_extracted = extracted

            # Step 5: Flush non-archive files
            self._report("Flushing to disk...", 90)
            flush_start = time.time()
            flushed, flushed_bytes = self._flush_remaining(release_name)
            self.stats_flush_duration = time.time() - flush_start
            self.stats_files_flushed = flushed
            self.stats_flush_bytes = flushed_bytes
            if self.stats_flush_duration > 0:
                self.stats_flush_speed_mbs = (flushed_bytes / (1024 * 1024)) / self.stats_flush_duration

            elapsed = time.time() - start_time
            total_files = extracted + flushed

            # Store final stats
            self.stats_total_duration = elapsed
            self.stats_total_size_bytes = self.ram_buffer.total_size
            self.stats_extract_path = self.extract_dir / release_name

            # Check if extraction failed (archives existed but couldn't be extracted)
            if self.stats_extraction_failed:
                self._report("Failed", 100)
                return False, f"Extraction failed: {self.stats_archives_found} archives found but could not be extracted (corrupted/incomplete download)"

            self._report("Complete", 100)
            return True, f"Processed {total_files} files in {elapsed:.1f}s"

        except Exception as e:
            logger.error(f"RAM post-processing failed: {e}")
            return False, str(e)

    def _report(self, stage: str, percent: float) -> None:
        """Report progress."""
        if self.on_progress:
            self.on_progress(stage, percent)
        logger.info(f"RAM PostProcess: {stage} ({percent:.0f}%)")

    def _repair_damaged_files(self, damaged_files: List[str]) -> int:
        """
        Repair damaged files using GPU-accelerated Reed-Solomon.

        OPTIMIZED VERSION:
        - Phase 1: Parallel damage detection (ThreadPoolExecutor for MD5)
        - Phase 2: Batched GPU repair (single kernel call for all blocks)
        - Phase 3: Parallel file reconstruction

        Args:
            damaged_files: List of damaged file names (keys in RAM buffer)

        Returns:
            Number of files successfully repaired
        """
        if not self.par2_parser.recovery_blocks:
            logger.warning("[GPU REPAIR] No recovery blocks available - cannot repair")
            return 0

        if not self.par2_parser.block_size:
            logger.warning("[GPU REPAIR] Block size unknown - cannot repair")
            return 0

        block_size = self.par2_parser.block_size
        recovery_blocks = self.par2_parser.recovery_blocks
        num_recovery = len(recovery_blocks)

        logger.info(f"[GPU REPAIR] Starting OPTIMIZED repair: {len(damaged_files)} damaged files, "
                   f"{num_recovery} recovery blocks, block_size={block_size}")

        # Build mapping: normalized filename -> Par2FileInfo
        par2_file_map: Dict[str, Par2FileInfo] = {}
        for file_info in self.par2_parser.files.values():
            normalized = RamVerifier._normalize_filename(file_info.filename)
            par2_file_map[normalized] = file_info

        # ===== PHASE 1: Parallel damage detection =====
        self._report("Detecting damaged blocks", 10)
        phase1_start = time.time()

        # Prepare file info for parallel processing
        file_repair_info = []  # List of (damaged_key, ram_file, file_info, file_data)

        for damaged_key in damaged_files:
            ram_file = self.ram_buffer.get_file(damaged_key)
            if not ram_file:
                logger.warning(f"[GPU REPAIR] File not in RAM buffer: {damaged_key}")
                continue

            original_filename = ram_file.filename
            normalized_key = RamVerifier._normalize_filename(original_filename)
            file_info = par2_file_map.get(normalized_key)

            if not file_info:
                logger.warning(f"[GPU REPAIR] No PAR2 info for: {original_filename}")
                continue

            # Read file data
            ram_file.data.seek(0)
            read_size = ram_file.actual_size if ram_file.actual_size > 0 else ram_file.size
            file_data = ram_file.data.read(read_size)

            file_repair_info.append((damaged_key, ram_file, file_info, file_data))

        if not file_repair_info:
            logger.warning("[GPU REPAIR] No files to repair")
            return 0

        logger.info(f"[GPU REPAIR] Phase 1: Scanning {len(file_repair_info)} files for damaged blocks...")

        # Parallel block verification using ThreadPoolExecutor
        def detect_damaged_blocks(args):
            """Detect damaged blocks in a single file (runs in thread)."""
            damaged_key, ram_file, file_info, file_data = args
            expected_size = file_info.size
            num_blocks = (expected_size + block_size - 1) // block_size
            has_ifsc = len(file_info.slice_checksums) > 0

            data_blocks = []
            damaged_indices = []

            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, expected_size)
                block = bytes(file_data[start:end])

                # Pad last block if necessary
                if len(block) < block_size:
                    block = block + b'\x00' * (block_size - len(block))

                is_damaged = False
                if has_ifsc and i < len(file_info.slice_checksums):
                    expected_md5 = file_info.slice_checksums[i].md5
                    actual_md5 = hashlib.md5(block).digest()
                    if actual_md5 != expected_md5:
                        is_damaged = True
                        damaged_indices.append(i)
                else:
                    # No IFSC - quick zero check
                    if len(block) >= 1024 and all(b == 0 for b in block[:1024]):
                        is_damaged = True
                        damaged_indices.append(i)

                data_blocks.append(None if is_damaged else block)

            return {
                'damaged_key': damaged_key,
                'ram_file': ram_file,
                'file_info': file_info,
                'data_blocks': data_blocks,
                'damaged_indices': damaged_indices,
                'expected_size': expected_size
            }

        # Run parallel damage detection
        detection_results = []
        max_workers = min(8, len(file_repair_info))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(detect_damaged_blocks, info) for info in file_repair_info]
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    if result['damaged_indices']:
                        detection_results.append(result)
                except Exception as e:
                    logger.error(f"[GPU REPAIR] Detection error: {e}")

        phase1_elapsed = time.time() - phase1_start
        total_damaged_blocks = sum(len(r['damaged_indices']) for r in detection_results)
        logger.info(f"[GPU REPAIR] Phase 1 complete: {total_damaged_blocks} damaged blocks in "
                   f"{len(detection_results)} files ({phase1_elapsed:.2f}s)")

        if not detection_results:
            logger.info("[GPU REPAIR] No damaged blocks found")
            return 0

        if total_damaged_blocks > num_recovery:
            logger.error(f"[GPU REPAIR] Not enough recovery blocks: need {total_damaged_blocks}, have {num_recovery}")
            return 0

        # ===== PHASE 2: GPU repair (per-file for correctness) =====
        self._report("GPU repair", 30)
        phase2_start = time.time()

        repaired_count = 0
        total_files = len(detection_results)

        for file_idx, result in enumerate(detection_results):
            damaged_key = result['damaged_key']
            ram_file = result['ram_file']
            file_info = result['file_info']
            data_blocks = result['data_blocks']
            damaged_indices = result['damaged_indices']
            expected_size = result['expected_size']

            progress = 30 + (60 * file_idx / total_files)
            self._report(f"Repairing {file_idx + 1}/{total_files}", progress)

            logger.info(f"[GPU REPAIR] Repairing {ram_file.filename}: {len(damaged_indices)} blocks")

            try:
                repaired_blocks = self.gpu_rs.repair(
                    data_blocks=data_blocks,
                    recovery_blocks=recovery_blocks,
                    block_size=block_size,
                    on_progress=None  # Skip per-block progress for speed
                )

                if repaired_blocks:
                    # Replace damaged blocks
                    for i, repaired_idx in enumerate(damaged_indices):
                        if i < len(repaired_blocks):
                            data_blocks[repaired_idx] = repaired_blocks[i]

                    # Reconstruct file
                    repaired_data = bytearray()
                    for block in data_blocks:
                        if block:
                            repaired_data.extend(block)
                    repaired_data = repaired_data[:expected_size]

                    # Verify final MD5
                    repaired_md5 = hashlib.md5(repaired_data).digest()
                    if repaired_md5 == file_info.md5_hash:
                        # Write back to RAM
                        ram_file.data.seek(0)
                        ram_file.data.write(repaired_data)
                        ram_file.actual_size = len(repaired_data)
                        ram_file.is_damaged = False
                        ram_file.is_verified = True
                        ram_file.md5_computed = repaired_md5.hex()

                        logger.info(f"[GPU REPAIR] ✓ Repaired: {damaged_key} ({len(damaged_indices)} blocks)")
                        repaired_count += 1
                    else:
                        logger.warning(f"[GPU REPAIR] MD5 mismatch after repair: {damaged_key}")
                else:
                    logger.warning(f"[GPU REPAIR] No blocks returned: {damaged_key}")

            except Exception as e:
                logger.error(f"[GPU REPAIR] Error: {damaged_key}: {e}")

        phase2_elapsed = time.time() - phase2_start
        total_elapsed = phase1_elapsed + phase2_elapsed

        logger.info(f"[GPU REPAIR] Complete: {repaired_count}/{total_files} files repaired in {total_elapsed:.2f}s "
                   f"(detect: {phase1_elapsed:.2f}s, repair: {phase2_elapsed:.2f}s)")

        return repaired_count

    def _is_archive_file(self, filename: str) -> bool:
        """Check if file is an archive (RAR, ZIP, 7z) by extension. PAR2 files are NOT archives."""
        name_lower = filename.lower()

        # Exclude PAR2 files - they are parity files, not archives
        if name_lower.endswith('.par2') or '.par2' in name_lower:
            return False

        # Direct extensions
        if name_lower.endswith(('.rar', '.zip', '.7z')):
            return True

        # RAR split files: .r00, .r01, ..., .r99, .s00, etc.
        if re.match(r'.*\.[rs]\d{2,}$', name_lower):
            return True

        return False

    def _is_archive_by_magic(self, ram_file: 'RamFile') -> Optional[str]:
        """
        Detect archive type by magic bytes (file signature).

        Used for obfuscated NZBs where filenames don't have extensions.

        Returns:
            'rar', 'zip', '7z', or None if not an archive
        """
        ram_file.data.seek(0)
        header = ram_file.data.read(8)
        ram_file.data.seek(0)

        if len(header) < 4:
            return None

        # RAR: Rar!\x1a\x07\x00 (RAR5) or Rar!\x1a\x07\x01 (RAR4)
        if header[:4] == b'Rar!':
            return 'rar'

        # ZIP: PK\x03\x04 (local file header) or PK\x05\x06 (empty archive)
        if header[:2] == b'PK' and header[2:4] in (b'\x03\x04', b'\x05\x06'):
            return 'zip'

        # 7z: 7z\xbc\xaf\x27\x1c
        if header[:6] == b'7z\xbc\xaf\x27\x1c':
            return '7z'

        # PAR2: PAR2\x00PKT - NOT an archive!
        if header[:4] == b'PAR2':
            return None

        return None

    def _is_par2_by_magic(self, ram_file: 'RamFile') -> bool:
        """
        Detect PAR2 file by magic bytes.

        PAR2 files start with: PAR2\x00PKT
        Used to exclude obfuscated PAR2 files from final flush.
        """
        ram_file.data.seek(0)
        header = ram_file.data.read(8)
        ram_file.data.seek(0)

        if len(header) < 4:
            return False

        return header[:4] == b'PAR2'

    def _find_sevenzip(self) -> Optional[str]:
        """Find 7z executable."""
        import sys
        import shutil

        # Check bundled tools first
        if hasattr(sys, '_MEIPASS'):
            bundled = Path(sys._MEIPASS) / 'tools' / '7z.exe'
        else:
            bundled = Path(__file__).parent.parent.parent / 'tools' / '7z.exe'

        if bundled.exists():
            return str(bundled)

        # Check common paths
        paths = [
            r"7z.exe",
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ]

        for path in paths:
            if not os.path.isabs(path):
                result = shutil.which(path)
                if result:
                    return result
            elif Path(path).exists():
                return path

        return None

    def _extract_archives(self, release_name: str, extraction_plan: 'ExtractionPlan' = None) -> int:
        """
        Extract archives from RAM.

        If an extraction_plan is provided, uses AdaptiveExtractor for intelligent
        multi-stage extraction. Otherwise falls back to legacy reactive extraction.

        Args:
            release_name: Name of the release being processed
            extraction_plan: Optional pre-computed ExtractionPlan from classification

        Returns:
            Number of files extracted
        """
        import subprocess
        import tempfile
        import shutil

        extract_path = self.extract_dir / release_name

        # Use AdaptiveExtractor only for complex extractions (ZIP→RAR, obfuscated, etc.)
        # For simple MULTIPART_RAR, the FAST path (flush + 7z) is ~3x faster than native RAR module
        use_adaptive = (extraction_plan and ADAPTIVE_EXTRACTOR_AVAILABLE and
                        extraction_plan.release_type not in (ReleaseType.MULTIPART_RAR, ReleaseType.SINGLE_RAR))

        if use_adaptive:
            logger.info(f"[ADAPTIVE] Using intelligent extraction: {extraction_plan.release_type.value}")

            extractor = AdaptiveExtractor(
                ram_buffer=self.ram_buffer,
                extract_dir=extract_path,
                password=self.password or "",
                on_progress=None  # Could wire up progress callback here
            )

            result = extractor.execute(extraction_plan)

            # Success if we extracted files, even if there were minor errors
            # Don't fall back to legacy just because of flush errors on existing files
            if result.files_extracted > 0:
                if result.errors:
                    logger.warning(f"[ADAPTIVE] Extraction completed with minor errors: {result.errors}")
                logger.info(f"[ADAPTIVE] Extraction complete: {result.files_extracted} files, "
                           f"{result.stages_executed} stages, {result.adaptations_made} adaptations")

                # Handle any remaining nested archives
                if extraction_plan.expects_nested:
                    nested_count = self._extract_nested_archives(extract_path, release_name)
                    if nested_count > 0:
                        result.files_extracted += nested_count

                return result.files_extracted
            else:
                # Only fall back if NO files were extracted at all
                logger.warning(f"[ADAPTIVE] Extraction failed (0 files): {result.errors}")
                logger.info("[ADAPTIVE] Falling back to legacy extraction...")
                # Fall through to legacy extraction

        # FAST extraction path: flush to temp + 7z (3x faster than native RAR module)
        if extraction_plan:
            logger.info(f"[FAST] Using optimized extraction for {extraction_plan.release_type.value}")
        else:
            logger.info("[FAST] Using optimized extraction")

        # Windows long path support (>260 chars)
        extract_path_str = str(extract_path.resolve())
        if os.name == 'nt' and len(extract_path_str) > 200:
            if not extract_path_str.startswith('\\\\?\\'):
                extract_path_str = '\\\\?\\' + extract_path_str
            os.makedirs(extract_path_str, exist_ok=True)
        else:
            extract_path.mkdir(parents=True, exist_ok=True)

        # Find 7z executable
        sevenzip_path = self._find_sevenzip()
        if not sevenzip_path:
            logger.warning("7z not found, skipping archive extraction")
            return 0

        # Collect archive files (use ram_file.filename for detection, not dict key)
        # Also check magic bytes for obfuscated NZBs where filenames lack extensions
        archive_files = []
        obfuscated_archives = []  # Archives detected by magic bytes only

        for dict_key, ram_file in self.ram_buffer.get_all_files().items():
            if self._is_archive_file(ram_file.filename):
                archive_files.append((dict_key, ram_file))
            else:
                # Check magic bytes for obfuscated files without extensions
                archive_type = self._is_archive_by_magic(ram_file)
                if archive_type:
                    obfuscated_archives.append((dict_key, ram_file, archive_type))

        if obfuscated_archives:
            logger.info(f"[OBFUSCATED] Detected {len(obfuscated_archives)} archives by magic bytes (no extension)")
            for dict_key, ram_file, archive_type in obfuscated_archives:
                archive_files.append((dict_key, ram_file))
                logger.debug(f"[OBFUSCATED] {ram_file.filename} -> {archive_type}")

        if not archive_files:
            logger.info("No archives to extract")
            self.stats_archives_found = 0
            return 0

        # Track archives found for failure detection
        self.stats_archives_found = len(archive_files)
        self.stats_extraction_failed = False  # Reset before extraction attempt

        # Create temp directory for archive flush
        temp_dir = Path(tempfile.mkdtemp(prefix="dler_ram_"))

        # OPTIMIZATION: Flush ALL archives to temp for fastest extraction
        # NVMe flush is ~3000 MB/s, much faster than RAM->dict copy + extraction
        # This matches the fast path from StreamingExtractor approach
        files_to_flush = archive_files

        try:
            # Flush ALL archives to temp in parallel for maximum speed
            # This is the fastest path: NVMe write ~3000 MB/s + 7z extraction
            flush_start = time.time()
            flushed_filenames = []

            def flush_one(item):
                dict_key, ram_file = item
                # Use ram_file.filename (proper name) not dict_key (has __idx suffix)
                actual_filename = ram_file.filename
                dest = temp_dir / actual_filename
                ram_file.data.seek(0)
                read_size = ram_file.actual_size if ram_file.actual_size > 0 else ram_file.size
                with open(dest, 'wb') as f:
                    f.write(ram_file.data.read(read_size))
                return actual_filename

            if files_to_flush:
                logger.info(f"[FAST] Flushing {len(files_to_flush)} archive files to temp for extraction...")
                with ThreadPoolExecutor(max_workers=16) as executor:
                    flushed_filenames = list(executor.map(flush_one, files_to_flush))

                flush_elapsed = time.time() - flush_start
                total_size = sum(rf.actual_size or rf.size for _, rf in files_to_flush)
                flush_speed = (total_size / (1024**2)) / flush_elapsed if flush_elapsed > 0 else 0
                logger.info(f"[FAST] Flushed archives to temp in {flush_elapsed:.1f}s ({flush_speed:.0f} MB/s)")

                # Free RAM buffers immediately after flush to reduce memory pressure
                for dict_key, _ in files_to_flush:
                    self.ram_buffer.remove_file(dict_key)
                logger.info(f"[FAST] Freed {len(files_to_flush)} archive buffers from RAM")

            # Find first RAR file to extract (use ram_file.filename, not dict key)
            first_rar = None
            for _, ram_file in archive_files:
                actual_name = ram_file.filename
                name_lower = actual_name.lower()
                # Main RAR file (not split)
                if name_lower.endswith('.rar'):
                    # Prefer .part1.rar or .part01.rar
                    if '.part1.' in name_lower or '.part01.' in name_lower or '.part001.' in name_lower:
                        first_rar = temp_dir / actual_name
                        break
                    elif first_rar is None:
                        first_rar = temp_dir / actual_name

            # Also check for .7z files
            first_7z = None
            for _, ram_file in archive_files:
                actual_name = ram_file.filename
                if actual_name.lower().endswith('.7z'):
                    first_7z = temp_dir / actual_name
                    break

            # Also check for .zip files (split or single)
            first_zip = None
            for _, ram_file in archive_files:
                actual_name = ram_file.filename
                name_lower = actual_name.lower()
                if name_lower.endswith('.zip'):
                    # Prefer .zip.001 or numbered zips
                    if '.zip.001' in name_lower or '.z01' in name_lower:
                        first_zip = temp_dir / actual_name
                        break
                    elif first_zip is None:
                        first_zip = temp_dir / actual_name

            extracted = 0

            # FAST PATH: Extract directly from temp files using 7z
            # All archives are now flushed to temp_dir, use 7z for fastest extraction
            if first_rar and first_rar.exists():
                logger.info(f"[FAST] RAR extraction via 7z: {first_rar.name}")
                extract_start = time.time()

                # Windows: hide console window
                creationflags = 0
                if sys.platform == 'win32':
                    creationflags = subprocess.CREATE_NO_WINDOW

                cmd = [
                    sevenzip_path,
                    'x',  # Extract with full paths
                    '-y',  # Yes to all prompts
                    '-bb0',  # Less output
                    '-bd',  # No progress indicator
                    '-mmt=on',  # Multi-threading for maximum speed
                    f'-o{extract_path}',
                ]

                # Add password if provided
                if self.password:
                    cmd.append(f'-p{self.password}')
                else:
                    cmd.append('-p')  # Empty password (skip prompts)

                cmd.append(str(first_rar))

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(temp_dir),
                    timeout=3600,
                    creationflags=creationflags
                )

                extract_elapsed = time.time() - extract_start

                if result.returncode == 0:
                    # Count extracted files
                    extracted_files = list(extract_path.rglob('*'))
                    extracted_files = [f for f in extracted_files if f.is_file()]
                    extracted = len(extracted_files)
                    total_extracted_size = sum(f.stat().st_size for f in extracted_files)
                    extract_speed = (total_extracted_size / (1024**2)) / extract_elapsed if extract_elapsed > 0 else 0
                    logger.info(f"[FAST] RAR extraction complete: {extracted} files "
                               f"({total_extracted_size/(1024**3):.1f} GB) in {extract_elapsed:.1f}s "
                               f"({extract_speed:.0f} MB/s)")

                    # Fix obfuscated filenames (files without extensions)
                    self._fix_obfuscated_files(extracted_files, release_name)

                    # Extract any nested archives (RAR in RAR)
                    nested_count = self._extract_nested_archives(extract_path, release_name)
                    if nested_count > 0:
                        extracted_files = list(extract_path.rglob('*'))
                        extracted_files = [f for f in extracted_files if f.is_file()]
                        extracted = len(extracted_files)
                else:
                    logger.error(f"RAR extraction failed: {result.stderr or result.stdout}")

            # Extract 7z if found
            if first_7z and first_7z.exists() and extracted == 0:
                logger.info(f"[FAST] 7z extraction: {first_7z.name}")
                extract_start = time.time()

                creationflags = 0
                if sys.platform == 'win32':
                    creationflags = subprocess.CREATE_NO_WINDOW

                cmd = [
                    sevenzip_path,
                    'x',  # Extract with full paths
                    '-y',  # Yes to all prompts
                    '-bb0',  # Less output
                    '-bd',  # No progress indicator
                    '-mmt=on',  # Multi-threading for maximum speed
                    f'-o{extract_path}',
                ]

                if self.password:
                    cmd.append(f'-p{self.password}')
                else:
                    cmd.append('-p')

                cmd.append(str(first_7z))

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(temp_dir),
                    timeout=3600,
                    creationflags=creationflags
                )

                extract_elapsed = time.time() - extract_start

                if result.returncode == 0:
                    extracted_files = list(extract_path.rglob('*'))
                    extracted_files = [f for f in extracted_files if f.is_file()]
                    extracted = len(extracted_files)
                    logger.info(f"7z extraction complete: {extracted} files in {extract_elapsed:.1f}s")
                    self._fix_obfuscated_files(extracted_files, release_name)

                    # Extract any nested archives
                    nested_count = self._extract_nested_archives(extract_path, release_name)
                    if nested_count > 0:
                        extracted_files = list(extract_path.rglob('*'))
                        extracted_files = [f for f in extracted_files if f.is_file()]
                        extracted = len(extracted_files)
                else:
                    logger.error(f"7z extraction failed: {result.stderr or result.stdout}")

            # Extract ZIP if found (and nothing else was extracted)
            # Use 7z to extract ZIPs from temp files (fast path)
            if first_zip and first_zip.exists() and extracted == 0:
                logger.info(f"[FAST] ZIP extraction via 7z: {first_zip.name}")
                extract_start = time.time()

                creationflags = 0
                if sys.platform == 'win32':
                    creationflags = subprocess.CREATE_NO_WINDOW

                cmd = [
                    sevenzip_path,
                    'x',  # Extract with full paths
                    '-y',  # Yes to all prompts
                    '-bb0',  # Less output
                    '-bd',  # No progress indicator
                    '-mmt=on',  # Multi-threading for maximum speed
                    f'-o{extract_path}',
                ]

                if self.password:
                    cmd.append(f'-p{self.password}')
                else:
                    cmd.append('-p')

                cmd.append(str(first_zip))

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(temp_dir),
                    timeout=3600,
                    creationflags=creationflags
                )

                extract_elapsed = time.time() - extract_start

                if result.returncode == 0:
                    extracted_files = list(extract_path.rglob('*'))
                    extracted_files = [f for f in extracted_files if f.is_file()]
                    extracted = len(extracted_files)
                    total_extracted_size = sum(f.stat().st_size for f in extracted_files)
                    extract_speed = (total_extracted_size / (1024**2)) / extract_elapsed if extract_elapsed > 0 else 0
                    logger.info(f"[FAST] ZIP extraction complete: {extracted} files "
                               f"({total_extracted_size/(1024**3):.1f} GB) in {extract_elapsed:.1f}s "
                               f"({extract_speed:.0f} MB/s)")
                    self._fix_obfuscated_files(extracted_files, release_name)

                    # ZIP may contain RAR archives (scene releases: ZIP→RAR→content)
                    # Extract any nested archives
                    nested_count = self._extract_nested_archives(extract_path, release_name)
                    if nested_count > 0:
                        extracted_files = list(extract_path.rglob('*'))
                        extracted_files = [f for f in extracted_files if f.is_file()]
                        extracted = len(extracted_files)
                else:
                    logger.error(f"ZIP extraction failed: {result.stderr or result.stdout}")

            # OBFUSCATED: If we have obfuscated archives (no extension) and nothing extracted yet
            # Each outer RAR contains ONE inner volume - extract ALL to collect inner volumes
            if obfuscated_archives and extracted == 0:
                logger.info(f"[OBFUSCATED] Extracting {len(obfuscated_archives)} outer archives to collect inner volumes...")
                extract_start = time.time()

                creationflags = 0
                if sys.platform == 'win32':
                    creationflags = subprocess.CREATE_NO_WINDOW

                # Create inner temp directory for collecting inner volumes
                inner_temp = temp_dir / "_inner_volumes"
                inner_temp.mkdir(exist_ok=True)

                # Extract each outer RAR individually to collect inner files
                outer_ok = 0
                for dict_key, ram_file, archive_type in obfuscated_archives:
                    outer_path = temp_dir / ram_file.filename
                    if not outer_path.exists():
                        continue

                    cmd = [
                        sevenzip_path,
                        'x',  # Extract with full paths
                        '-y',  # Yes to all prompts
                        '-bb0',  # Less output
                        '-bd',  # No progress indicator
                        f'-o{inner_temp}',  # Extract to inner temp
                        '-aoa',  # Overwrite all
                    ]

                    if self.password:
                        cmd.append(f'-p{self.password}')
                    else:
                        cmd.append('-p')

                    cmd.append(str(outer_path))

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        creationflags=creationflags
                    )

                    if result.returncode == 0:
                        outer_ok += 1

                logger.info(f"[OBFUSCATED] Extracted {outer_ok}/{len(obfuscated_archives)} outer archives")

                # Check what we collected in inner_temp
                inner_files = list(inner_temp.rglob('*'))
                inner_files = [f for f in inner_files if f.is_file()]

                if inner_files:
                    # Check if inner files are RAR volumes (.rar, .r00, .r01, etc.)
                    import re
                    inner_rars = [f for f in inner_files if f.suffix.lower() == '.rar' or
                                  re.match(r'\.[rs]\d{2,}$', f.suffix.lower())]

                    if inner_rars:
                        # Find first RAR volume
                        first_inner_rar = None
                        for f in sorted(inner_rars, key=lambda x: x.name.lower()):
                            if f.suffix.lower() == '.rar':
                                first_inner_rar = f
                                break
                        if not first_inner_rar:
                            first_inner_rar = inner_rars[0]

                        logger.info(f"[OBFUSCATED] Found {len(inner_rars)} inner RAR volumes, extracting from {first_inner_rar.name}...")

                        cmd = [
                            sevenzip_path,
                            'x',  # Extract with full paths
                            '-y',  # Yes to all prompts
                            '-bb0',  # Less output
                            '-bd',  # No progress indicator
                            '-mmt=on',  # Multi-threading
                            f'-o{extract_path}',
                        ]

                        if self.password:
                            cmd.append(f'-p{self.password}')
                        else:
                            cmd.append('-p')

                        cmd.append(str(first_inner_rar))

                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            cwd=str(inner_temp),  # Run from inner_temp so 7z finds all volumes
                            timeout=3600,
                            creationflags=creationflags
                        )

                        extract_elapsed = time.time() - extract_start

                        if result.returncode == 0:
                            extracted_files = list(extract_path.rglob('*'))
                            extracted_files = [f for f in extracted_files if f.is_file()]
                            extracted = len(extracted_files)
                            total_extracted_size = sum(f.stat().st_size for f in extracted_files)
                            extract_speed = (total_extracted_size / (1024**2)) / extract_elapsed if extract_elapsed > 0 else 0
                            logger.info(f"[OBFUSCATED] Inner RAR extraction complete: {extracted} files "
                                       f"({total_extracted_size/(1024**3):.1f} GB) in {extract_elapsed:.1f}s "
                                       f"({extract_speed:.0f} MB/s)")
                            self._fix_obfuscated_files(extracted_files, release_name)

                            # Extract any nested archives
                            nested_count = self._extract_nested_archives(extract_path, release_name)
                            if nested_count > 0:
                                extracted_files = list(extract_path.rglob('*'))
                                extracted_files = [f for f in extracted_files if f.is_file()]
                                extracted = len(extracted_files)
                        else:
                            logger.error(f"[OBFUSCATED] Inner RAR extraction failed: {result.stderr or result.stdout}")
                    else:
                        # Inner files are not RARs - move them to extract_path
                        for f in inner_files:
                            dest = extract_path / f.name
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            import shutil
                            shutil.move(str(f), str(dest))
                            extracted += 1
                        logger.info(f"[OBFUSCATED] Moved {extracted} inner files to destination")

            # Mark extraction as failed if archives existed but nothing was extracted
            if extracted == 0:
                self.stats_extraction_failed = True
                logger.error(f"[EXTRACTION] Failed: {self.stats_archives_found} archives found but 0 files extracted")

            return extracted

        finally:
            # Cleanup temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp dir: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")

    # File magic signatures for type detection
    FILE_SIGNATURES = {
        b'\x1a\x45\xdf\xa3': '.mkv',           # Matroska/WebM
        b'\x00\x00\x00\x1c\x66\x74\x79\x70': '.mp4',  # MP4/M4V (ftyp)
        b'\x00\x00\x00\x20\x66\x74\x79\x70': '.mp4',  # MP4 variant
        b'\x00\x00\x00\x18\x66\x74\x79\x70': '.mp4',  # MP4 variant
        b'\x00\x00\x00\x14\x66\x74\x79\x70': '.mp4',  # MP4 variant
        b'\x52\x49\x46\x46': '.avi',           # AVI (RIFF)
        b'\x30\x26\xb2\x75': '.wmv',           # WMV/ASF
        b'\x47': '.ts',                         # MPEG-TS (single byte)
        b'\x00\x00\x01\xba': '.mpg',           # MPEG-PS
        b'\x00\x00\x01\xb3': '.mpg',           # MPEG video
        b'\x49\x44\x33': '.mp3',               # MP3 with ID3
        b'\xff\xfb': '.mp3',                   # MP3 without ID3
        b'\xff\xfa': '.mp3',                   # MP3 variant
        b'\x66\x4c\x61\x43': '.flac',          # FLAC
        b'\x4f\x67\x67\x53': '.ogg',           # OGG
        b'\x50\x4b\x03\x04': '.zip',           # ZIP
        b'\x52\x61\x72\x21': '.rar',           # RAR
        b'\x25\x50\x44\x46': '.pdf',           # PDF
        b'\x89\x50\x4e\x47': '.png',           # PNG
        b'\xff\xd8\xff': '.jpg',               # JPEG
        b'\x47\x49\x46\x38': '.gif',           # GIF
    }

    def _detect_file_type(self, file_path: Path) -> Optional[str]:
        """Detect file type by reading magic bytes."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)

            if not header:
                return None

            # Check against signatures (longest match first)
            for signature, extension in sorted(self.FILE_SIGNATURES.items(),
                                               key=lambda x: len(x[0]), reverse=True):
                if header.startswith(signature):
                    return extension

            # Special case: MP4 - check for 'ftyp' at offset 4
            if len(header) >= 8 and header[4:8] == b'ftyp':
                return '.mp4'

            # Special case: MKV - can start with different EBML headers
            if len(header) >= 4 and header[0:4] == b'\x1a\x45\xdf\xa3':
                return '.mkv'

            return None

        except Exception as e:
            logger.debug(f"Error detecting file type for {file_path}: {e}")
            return None

    def _fix_obfuscated_files(self, files: list, release_name: str) -> None:
        """
        Rename obfuscated files (no extension) to proper names.

        Args:
            files: List of extracted file paths
            release_name: Release name to use for renaming main content
        """
        for file_path in files:
            # Skip files that already have a recognized extension
            ext = file_path.suffix.lower()
            if ext and len(ext) > 1 and len(ext) <= 5:
                # Has extension, skip
                continue

            # Detect file type
            detected_ext = self._detect_file_type(file_path)
            if not detected_ext:
                logger.warning(f"Could not detect type for obfuscated file: {file_path.name}")
                continue

            # Build new filename
            # For video files, use release name; for others, keep original name + extension
            if detected_ext in ('.mkv', '.mp4', '.avi', '.wmv', '.mpg', '.ts'):
                # Main video file - use release name
                new_name = f"{release_name}{detected_ext}"
            else:
                # Other files - just add extension
                new_name = f"{file_path.name}{detected_ext}"

            new_path = file_path.parent / new_name

            # Avoid overwriting existing files
            if new_path.exists():
                # Add number suffix
                counter = 1
                while new_path.exists():
                    stem = release_name if detected_ext in ('.mkv', '.mp4', '.avi') else file_path.name
                    new_name = f"{stem}_{counter}{detected_ext}"
                    new_path = file_path.parent / new_name
                    counter += 1

            try:
                file_path.rename(new_path)
                logger.info(f"Renamed obfuscated file: {file_path.name} -> {new_name}")
            except Exception as e:
                logger.error(f"Failed to rename {file_path.name}: {e}")

    def _extract_nested_archives(self, extract_path: Path, release_name: str, max_depth: int = 3) -> int:
        """
        Extract nested archives (RAR/7z inside extracted content).

        Uses DLER's custom UnRAR engine for RAR files, 7z fallback for others.

        Handles scene releases where:
        - ZIP contains RAR which contains more RAR
        - RAR contains nested RAR archives

        Args:
            extract_path: Directory where files were extracted
            release_name: Release name for context
            max_depth: Maximum nesting depth to prevent infinite loops

        Returns:
            Number of files extracted from nested archives
        """
        if max_depth <= 0:
            return 0

        total_extracted = 0

        # Find RAR archives - collect all parts for each archive set
        rar_archive_sets = {}  # base_name -> list of (path, sort_key)
        for f in extract_path.rglob('*.rar'):
            name_lower = f.name.lower()

            # Determine base name and sort key
            # Modern: name.part001.rar
            part_match = re.search(r'(.+)\.part(\d+)\.rar$', name_lower)
            if part_match:
                base = part_match.group(1)
                sort_key = int(part_match.group(2))
                if base not in rar_archive_sets:
                    rar_archive_sets[base] = []
                rar_archive_sets[base].append((f, sort_key))
                continue

            # Single RAR (no .partXXX)
            if not re.search(r'\.part\d+\.rar$', name_lower):
                base = f.stem.lower()
                if base not in rar_archive_sets:
                    rar_archive_sets[base] = []
                rar_archive_sets[base].append((f, 0))

        # Also find old-style .rXX parts
        for f in extract_path.rglob('*.r[0-9][0-9]'):
            base = f.stem.lower()
            ext_match = re.search(r'\.r(\d{2})$', f.name.lower())
            if ext_match:
                sort_key = int(ext_match.group(1)) + 1  # .r00 = part 2
                if base not in rar_archive_sets:
                    rar_archive_sets[base] = []
                rar_archive_sets[base].append((f, sort_key))

        # Find 7z archives
        sevenz_files = list(extract_path.rglob('*.7z'))

        if not rar_archive_sets and not sevenz_files:
            return 0

        logger.info(f"[Nested] Found {len(rar_archive_sets)} RAR archive sets and {len(sevenz_files)} 7z archives")

        archives_to_cleanup = []

        # Extract RAR archives using DLER's custom UnRAR engine
        if rar_archive_sets and RAM_RAR_AVAILABLE:
            for base_name, parts_list in rar_archive_sets.items():
                # Sort parts by their sort key
                parts_list.sort(key=lambda x: x[1])

                logger.info(f"[Nested] Extracting RAR set '{base_name}' ({len(parts_list)} parts) via UnRAR engine")

                # Read all parts into memory
                rar_parts: Dict[str, bytes] = {}
                output_dir = parts_list[0][0].parent

                for part_path, _ in parts_list:
                    try:
                        rar_parts[part_path.name] = part_path.read_bytes()
                        archives_to_cleanup.append(part_path)
                    except Exception as e:
                        logger.warning(f"[Nested] Failed to read {part_path.name}: {e}")

                if rar_parts:
                    try:
                        count = extract_multipart_rar_to_disk(
                            rar_parts,
                            output_dir,
                            password=self.password or ""
                        )
                        if count > 0:
                            logger.info(f"[Nested] Extracted {count} files from '{base_name}'")
                            total_extracted += 1
                        else:
                            logger.warning(f"[Nested] UnRAR returned 0 files for '{base_name}'")
                            # Remove from cleanup if extraction failed
                            for part_path, _ in parts_list:
                                if part_path in archives_to_cleanup:
                                    archives_to_cleanup.remove(part_path)
                    except Exception as e:
                        logger.warning(f"[Nested] UnRAR failed for '{base_name}': {e}")
                        # Remove from cleanup if extraction failed
                        for part_path, _ in parts_list:
                            if part_path in archives_to_cleanup:
                                archives_to_cleanup.remove(part_path)

        # Fallback: use 7z for RAR if UnRAR not available, and for 7z archives
        sevenzip_path = self._find_sevenzip()
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

        # Extract RAR via 7z if UnRAR not available
        if rar_archive_sets and not RAM_RAR_AVAILABLE and sevenzip_path:
            for base_name, parts_list in rar_archive_sets.items():
                # Only extract from first part
                first_part = min(parts_list, key=lambda x: x[1])[0]
                logger.info(f"[Nested] Extracting RAR '{first_part.name}' via 7z fallback")

                output_dir = first_part.parent
                cmd = [sevenzip_path, 'x', '-y', '-bb0', '-bd', f'-o{output_dir}']
                if self.password:
                    cmd.append(f'-p{self.password}')
                else:
                    cmd.append('-p')
                cmd.append(str(first_part))

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                          timeout=1800, creationflags=creationflags)
                    if result.returncode == 0:
                        logger.info(f"[Nested] Successfully extracted: {first_part.name}")
                        for part_path, _ in parts_list:
                            archives_to_cleanup.append(part_path)
                        total_extracted += 1
                except Exception as e:
                    logger.warning(f"[Nested] 7z failed for {first_part.name}: {e}")

        # Extract 7z archives (always via 7z)
        if sevenz_files and sevenzip_path:
            for archive_path in sevenz_files:
                logger.info(f"[Nested] Extracting 7z: {archive_path.name}")
                output_dir = archive_path.parent
                cmd = [sevenzip_path, 'x', '-y', '-bb0', '-bd', f'-o{output_dir}']
                if self.password:
                    cmd.append(f'-p{self.password}')
                else:
                    cmd.append('-p')
                cmd.append(str(archive_path))

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                          timeout=1800, creationflags=creationflags)
                    if result.returncode == 0:
                        logger.info(f"[Nested] Successfully extracted: {archive_path.name}")
                        archives_to_cleanup.append(archive_path)
                        total_extracted += 1
                except Exception as e:
                    logger.warning(f"[Nested] 7z failed for {archive_path.name}: {e}")

        # Cleanup extracted archives
        cleanup_count = 0
        cleanup_size = 0
        for archive_path in archives_to_cleanup:
            try:
                if archive_path.exists():
                    size = archive_path.stat().st_size
                    archive_path.unlink()
                    cleanup_count += 1
                    cleanup_size += size
                    logger.debug(f"[Nested] Removed: {archive_path.name}")
            except Exception as e:
                logger.warning(f"[Nested] Failed to remove {archive_path.name}: {e}")

        if cleanup_count > 0:
            logger.info(f"[Nested] Cleaned up {cleanup_count} archive files ({cleanup_size / (1024*1024):.1f} MB)")

        if total_extracted > 0:
            logger.info(f"[Nested] Extraction complete: {total_extracted} archive sets processed")

            # Recursive check for more nested archives
            deeper_extracted = self._extract_nested_archives(extract_path, release_name, max_depth - 1)
            total_extracted += deeper_extracted

            # Fix any obfuscated files from nested extraction
            if total_extracted > 0:
                extracted_files = [f for f in extract_path.rglob('*') if f.is_file()]
                self._fix_obfuscated_files(extracted_files, release_name)

        return total_extracted

    def _flush_remaining(self, release_name: str) -> int:
        """Flush non-archive files to disk (parallel for speed)."""
        extract_path = self.extract_dir / release_name

        # Windows long path support (>260 chars)
        extract_path_str = str(extract_path.resolve())
        if os.name == 'nt' and len(extract_path_str) > 200:
            # Use \\?\ prefix for long paths on Windows
            if not extract_path_str.startswith('\\\\?\\'):
                extract_path_str = '\\\\?\\' + extract_path_str
            os.makedirs(extract_path_str, exist_ok=True)
        else:
            extract_path.mkdir(parents=True, exist_ok=True)

        # Collect non-archive files (use ram_file.filename for detection, not dict key)
        # Also exclude PAR2 files and archives (both by extension and magic bytes for obfuscated NZBs)
        files_to_flush = []
        par2_excluded = 0
        par2_excluded_size = 0
        archive_excluded = 0
        archive_excluded_size = 0
        for dict_key, ram_file in self.ram_buffer.get_all_files().items():
            # Skip archives by extension
            if self._is_archive_file(ram_file.filename):
                continue
            # Skip archives by magic bytes (for obfuscated files without extensions)
            archive_type = self._is_archive_by_magic(ram_file)
            if archive_type:
                archive_excluded += 1
                archive_excluded_size += ram_file.size
                logger.debug(f"Excluded obfuscated {archive_type} archive: {ram_file.filename}")
                continue
            # Skip PAR2 files by extension
            if '.par2' in ram_file.filename.lower():
                par2_excluded += 1
                par2_excluded_size += ram_file.size
                continue
            # Skip PAR2 files by magic bytes (for obfuscated files without extensions)
            if self._is_par2_by_magic(ram_file):
                par2_excluded += 1
                par2_excluded_size += ram_file.size
                logger.debug(f"Excluded obfuscated PAR2 file: {ram_file.filename}")
                continue
            files_to_flush.append((dict_key, ram_file))

        if archive_excluded > 0:
            logger.info(f"Excluded {archive_excluded} obfuscated archives ({archive_excluded_size / (1024*1024):.1f} MB) from flush")

        if par2_excluded > 0:
            logger.info(f"Excluded {par2_excluded} PAR2 files ({par2_excluded_size / (1024*1024):.1f} MB) from flush")

        if not files_to_flush:
            return 0, 0

        def flush_one(item):
            dict_key, ram_file = item
            try:
                # Use ram_file.filename (proper name) not dict_key (may have __idx suffix)
                actual_filename = ram_file.filename
                dest_path = extract_path / actual_filename

                # Windows long path support
                dest_path_str = str(dest_path.resolve())
                if os.name == 'nt' and len(dest_path_str) > 250:
                    if not dest_path_str.startswith('\\\\?\\'):
                        dest_path_str = '\\\\?\\' + dest_path_str

                ram_file.data.seek(0)
                read_size = ram_file.actual_size if ram_file.actual_size > 0 else ram_file.size
                with open(dest_path_str, 'wb') as f:
                    f.write(ram_file.data.read(read_size))
                return (actual_filename, read_size)
            except Exception as e:
                logger.error(f"Failed to flush {ram_file.filename}: {e}")
                return None

        # Parallel flush
        flushed = 0
        flushed_bytes = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(flush_one, files_to_flush))
            for r in results:
                if r is not None:
                    flushed += 1
                    flushed_bytes += r[1]

        logger.info(f"Flushed {flushed} non-archive files ({flushed_bytes / (1024*1024):.1f} MB)")
        return flushed, flushed_bytes
