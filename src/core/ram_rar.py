"""
DLER RAM RAR Extractor
======================

True RAM-to-RAM RAR extraction using UnRAR library via ctypes.

Architecture:
  1. RAR data in RAM → Memory-mapped file (pagefile-backed = stays in RAM)
  2. UnRAR reads from mapped "file" (RAM speed, no disk I/O)
  3. UCM_PROCESSDATA callback captures output → RAM buffer

This module provides maximum performance by:
  - Using Windows memory-mapped files backed by pagefile (pure RAM)
  - Capturing decompressed data via callback (no disk writes)
  - Processing multi-part RAR archives from memory buffers

Copyright (c) 2025 DLER Project
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import logging
import mmap
import os
import sys
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# UnRAR constants
ERAR_SUCCESS = 0
ERAR_END_ARCHIVE = 10
ERAR_NO_MEMORY = 11
ERAR_BAD_DATA = 12
ERAR_BAD_ARCHIVE = 13
ERAR_UNKNOWN_FORMAT = 14
ERAR_EOPEN = 15
ERAR_ECREATE = 16
ERAR_ECLOSE = 17
ERAR_EREAD = 18
ERAR_EWRITE = 19
ERAR_SMALL_BUF = 20
ERAR_UNKNOWN = 21
ERAR_MISSING_PASSWORD = 22
ERAR_BAD_PASSWORD = 24

RAR_OM_LIST = 0
RAR_OM_EXTRACT = 1

RAR_SKIP = 0
RAR_TEST = 1
RAR_EXTRACT = 2

# Callback messages
UCM_CHANGEVOLUME = 0
UCM_PROCESSDATA = 1
UCM_NEEDPASSWORD = 2
UCM_CHANGEVOLUMEW = 3
UCM_NEEDPASSWORDW = 4

# Header flags
RHDF_SPLITBEFORE = 0x01
RHDF_SPLITAFTER = 0x02
RHDF_ENCRYPTED = 0x04
RHDF_SOLID = 0x10
RHDF_DIRECTORY = 0x20


@dataclass
class RarEntry:
    """Information about a file in a RAR archive."""
    filename: str
    compressed_size: int
    uncompressed_size: int
    crc32: int
    is_directory: bool
    is_encrypted: bool


@dataclass
class ExtractedFile:
    """Extracted file data."""
    filename: str
    data: bytes
    original_size: int
    crc32: int
    is_directory: bool = False


# LPARAM type for 64-bit compatibility
_LPARAM = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long


class RAROpenArchiveDataEx(ctypes.Structure):
    """UnRAR archive open structure."""
    _fields_ = [
        ("ArcName", ctypes.c_char_p),
        ("ArcNameW", ctypes.c_wchar_p),
        ("OpenMode", ctypes.c_uint),
        ("OpenResult", ctypes.c_uint),
        ("CmtBuf", ctypes.c_char_p),
        ("CmtBufSize", ctypes.c_uint),
        ("CmtSize", ctypes.c_uint),
        ("CmtState", ctypes.c_uint),
        ("Flags", ctypes.c_uint),
        ("Callback", ctypes.c_void_p),
        ("UserData", _LPARAM),  # LPARAM is 64-bit on x64
        ("OpFlags", ctypes.c_uint),
        ("CmtBufW", ctypes.c_wchar_p),
        ("Reserved", ctypes.c_uint * 25),
    ]


class RARHeaderDataEx(ctypes.Structure):
    """UnRAR header data structure."""
    _fields_ = [
        ("ArcName", ctypes.c_char * 1024),
        ("ArcNameW", ctypes.c_wchar * 1024),
        ("FileName", ctypes.c_char * 1024),
        ("FileNameW", ctypes.c_wchar * 1024),
        ("Flags", ctypes.c_uint),
        ("PackSize", ctypes.c_uint),
        ("PackSizeHigh", ctypes.c_uint),
        ("UnpSize", ctypes.c_uint),
        ("UnpSizeHigh", ctypes.c_uint),
        ("HostOS", ctypes.c_uint),
        ("FileCRC", ctypes.c_uint),
        ("FileTime", ctypes.c_uint),
        ("UnpVer", ctypes.c_uint),
        ("Method", ctypes.c_uint),
        ("FileAttr", ctypes.c_uint),
        ("CmtBuf", ctypes.c_char_p),
        ("CmtBufSize", ctypes.c_uint),
        ("CmtSize", ctypes.c_uint),
        ("CmtState", ctypes.c_uint),
        ("DictSize", ctypes.c_uint),
        ("HashType", ctypes.c_uint),
        ("Hash", ctypes.c_char * 32),
        ("RedirType", ctypes.c_uint),
        ("RedirName", ctypes.c_wchar_p),
        ("RedirNameSize", ctypes.c_uint),
        ("DirTarget", ctypes.c_uint),
        ("MtimeLow", ctypes.c_uint),
        ("MtimeHigh", ctypes.c_uint),
        ("CtimeLow", ctypes.c_uint),
        ("CtimeHigh", ctypes.c_uint),
        ("AtimeLow", ctypes.c_uint),
        ("AtimeHigh", ctypes.c_uint),
        ("Reserved", ctypes.c_uint * 988),
    ]


# Callback function type
# IMPORTANT: On 64-bit Windows, LPARAM is 64-bit (c_longlong), not c_long (32-bit)!
if sys.platform == 'win32':
    UNRARCALLBACK = ctypes.WINFUNCTYPE(
        ctypes.c_int,     # return
        ctypes.c_uint,    # msg
        _LPARAM,          # UserData (LPARAM = pointer-sized)
        _LPARAM,          # P1 (pointer to data)
        _LPARAM           # P2 (size)
    )
else:
    UNRARCALLBACK = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_uint, _LPARAM, _LPARAM, _LPARAM
    )


class _CallbackData:
    """Shared data for UnRAR callback."""
    def __init__(self):
        self.buffer = BytesIO()
        self.password: str = ""
        self.current_file: str = ""
        self.cancel: bool = False
        self.temp_dir: Optional[Path] = None  # Directory containing volume files
        self.volume_change_count: int = 0  # Counter to detect infinite loops
        self.max_volume_changes: int = 200  # Safety limit


def _create_callback(data: _CallbackData) -> UNRARCALLBACK:
    """Create UnRAR callback function."""

    def callback(msg: int, user_data: int, p1: int, p2: int) -> int:
        if msg == UCM_PROCESSDATA:
            # p1 = pointer to data, p2 = size
            if p1 and p2 > 0:
                chunk = ctypes.string_at(p1, p2)
                data.buffer.write(chunk)
            return 0 if data.cancel else 1

        elif msg == UCM_NEEDPASSWORD:
            # p1 = buffer, p2 = size
            if data.password and p1 and p2 > 0:
                pwd_bytes = data.password.encode('utf-8')[:p2-1]
                ctypes.memmove(p1, pwd_bytes, len(pwd_bytes))
                ctypes.memset(p1 + len(pwd_bytes), 0, 1)
                return 1
            return -1

        elif msg == UCM_NEEDPASSWORDW:
            # Wide char password
            if data.password and p1 and p2 > 0:
                pwd = data.password[:p2-1]
                pwd_array = (ctypes.c_wchar * p2).from_address(p1)
                for i, c in enumerate(pwd):
                    pwd_array[i] = c
                pwd_array[len(pwd)] = '\0'
                return 1
            return -1

        elif msg in (UCM_CHANGEVOLUME, UCM_CHANGEVOLUMEW):
            # Multi-volume: P1 = volume name buffer, P2 = mode
            # RAR_VOL_ASK (0) = volume not found, need path
            # RAR_VOL_NOTIFY (1) = volume found, just notification
            # Return 1 to continue, -1 to abort

            data.volume_change_count += 1

            # Safety: prevent infinite loop
            if data.volume_change_count > data.max_volume_changes:
                logger.error(f"[RAM RAR] Too many volume change requests ({data.volume_change_count}), aborting!")
                return -1

            if p2 == 1:  # RAR_VOL_NOTIFY - volume exists, continue
                # Log which volume was found
                if msg == UCM_CHANGEVOLUMEW and p1:
                    try:
                        vol_name = ctypes.wstring_at(p1)
                        logger.debug(f"[RAM RAR] Volume found: {Path(vol_name).name}")
                    except:
                        pass
                return 1

            elif p2 == 0:  # RAR_VOL_ASK - volume not found
                # UnRAR is asking for a volume it can't find
                # Try to help by checking if it exists in temp_dir
                requested_name = None
                try:
                    if msg == UCM_CHANGEVOLUMEW and p1:
                        requested_name = ctypes.wstring_at(p1)
                    elif msg == UCM_CHANGEVOLUME and p1:
                        requested_name = ctypes.string_at(p1).decode('utf-8', errors='replace')
                except:
                    pass

                if requested_name:
                    logger.warning(f"[RAM RAR] Volume NOT FOUND: {Path(requested_name).name}")

                    # If we have a temp_dir, check if the volume exists there
                    if data.temp_dir and data.temp_dir.exists():
                        req_basename = Path(requested_name).name
                        possible_path = data.temp_dir / req_basename

                        if possible_path.exists():
                            # Volume exists! Update the path buffer
                            new_path = str(possible_path)
                            logger.info(f"[RAM RAR] Found volume at: {new_path}")

                            try:
                                if msg == UCM_CHANGEVOLUMEW:
                                    # Wide string buffer
                                    buf_array = (ctypes.c_wchar * 1024).from_address(p1)
                                    for i, c in enumerate(new_path[:1023]):
                                        buf_array[i] = c
                                    buf_array[min(len(new_path), 1023)] = '\0'
                                else:
                                    # ANSI buffer
                                    new_path_bytes = new_path.encode('utf-8')[:1023]
                                    ctypes.memmove(p1, new_path_bytes, len(new_path_bytes))
                                    ctypes.memset(p1 + len(new_path_bytes), 0, 1)
                                return 1  # Path updated, retry
                            except Exception as e:
                                logger.error(f"[RAM RAR] Failed to update volume path: {e}")
                        else:
                            # Volume doesn't exist - list what we have
                            available = list(data.temp_dir.glob('*'))
                            logger.warning(f"[RAM RAR] Volume not found in temp_dir. Available: "
                                         f"{[f.name for f in available[:5]]}...")

                # Can't find volume - abort to prevent infinite loop
                logger.error("[RAM RAR] Volume not found and cannot be located - aborting extraction")
                return -1

            return 1  # Default: continue

        return 1

    return UNRARCALLBACK(callback)


class RamRarExtractor:
    """
    Extract RAR archives from memory to memory.

    Uses Windows memory-mapped files for input (pagefile-backed = RAM)
    and UCM_PROCESSDATA callback for output (direct to RAM).

    Example:
        extractor = RamRarExtractor()
        extractor.set_archive_data(rar_bytes)
        files = extractor.extract_all()
        for name, data in files.items():
            print(f"{name}: {len(data)} bytes")
    """

    _unrar_dll = None
    _dll_loaded = False

    def __init__(self):
        self._archive_data: Optional[bytes] = None
        self._temp_path: Optional[Path] = None
        self._temp_dir_for_volumes: Optional[Path] = None  # For multi-volume support
        self._password: str = ""
        self._last_error: str = ""

        # Try to load UnRAR DLL
        self._load_dll()

    @classmethod
    def _load_dll(cls) -> bool:
        """Load UnRAR DLL."""
        if cls._dll_loaded:
            return cls._unrar_dll is not None

        cls._dll_loaded = True

        if sys.platform != 'win32':
            logger.warning("RamRarExtractor only supported on Windows")
            return False

        # Try to find UnRAR.dll (64-bit first for 64-bit Python)
        dll_paths = [
            Path(__file__).parent.parent.parent / "tools" / "UnRAR64.dll",
            Path(__file__).parent.parent.parent / "tools" / "UnRAR.dll",
            "UnRAR64.dll",
            "UnRAR.dll",
            "unrar64.dll",
            "unrar.dll",
        ]

        for dll_path in dll_paths:
            try:
                cls._unrar_dll = ctypes.WinDLL(str(dll_path))
                logger.info(f"Loaded UnRAR DLL: {dll_path}")

                # Setup function prototypes
                cls._unrar_dll.RAROpenArchiveEx.argtypes = [ctypes.POINTER(RAROpenArchiveDataEx)]
                cls._unrar_dll.RAROpenArchiveEx.restype = ctypes.c_void_p

                cls._unrar_dll.RARCloseArchive.argtypes = [ctypes.c_void_p]
                cls._unrar_dll.RARCloseArchive.restype = ctypes.c_int

                cls._unrar_dll.RARReadHeaderEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(RARHeaderDataEx)]
                cls._unrar_dll.RARReadHeaderEx.restype = ctypes.c_int

                cls._unrar_dll.RARProcessFile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
                cls._unrar_dll.RARProcessFile.restype = ctypes.c_int

                # UserData is LPARAM (64-bit on x64)
                cls._unrar_dll.RARSetCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _LPARAM]
                cls._unrar_dll.RARSetCallback.restype = None

                cls._unrar_dll.RARSetPassword.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                cls._unrar_dll.RARSetPassword.restype = None

                return True

            except OSError:
                continue

        logger.warning("UnRAR DLL not found")
        return False

    @classmethod
    def is_available(cls) -> bool:
        """Check if UnRAR DLL is available."""
        cls._load_dll()
        return cls._unrar_dll is not None

    def set_archive_data(self, data: bytes) -> bool:
        """
        Load RAR archive from bytes.

        Uses a temp file with FILE_ATTRIBUTE_TEMPORARY flag,
        which tells Windows to keep it in RAM cache.
        """
        self._cleanup()
        self._archive_data = data

        try:
            # Create temp file with TEMPORARY attribute (stays in RAM cache)
            fd, temp_path = tempfile.mkstemp(suffix='.rar', prefix='dler_ram_')
            self._temp_path = Path(temp_path)

            # Write data
            os.write(fd, data)
            os.close(fd)

            # On Windows, set TEMPORARY attribute for RAM caching
            if sys.platform == 'win32':
                import ctypes.wintypes
                kernel32 = ctypes.windll.kernel32
                FILE_ATTRIBUTE_TEMPORARY = 0x100
                kernel32.SetFileAttributesW(str(self._temp_path), FILE_ATTRIBUTE_TEMPORARY)

            logger.debug(f"[RAM RAR] Created temp file: {self._temp_path} ({len(data)} bytes)")
            return True

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[RAM RAR] Failed to create temp file: {e}")
            return False

    def set_password(self, password: str) -> None:
        """Set password for encrypted archives."""
        self._password = password

    def list_files(self) -> List[RarEntry]:
        """List all files in the archive."""
        if not self._temp_path or not self._unrar_dll:
            return []

        entries = []

        arc_data = RAROpenArchiveDataEx()
        arc_data.ArcName = str(self._temp_path).encode('utf-8')
        arc_data.OpenMode = RAR_OM_LIST

        handle = self._unrar_dll.RAROpenArchiveEx(ctypes.byref(arc_data))
        if not handle or arc_data.OpenResult != ERAR_SUCCESS:
            self._last_error = f"Failed to open archive: {arc_data.OpenResult}"
            return entries

        try:
            header = RARHeaderDataEx()
            while self._unrar_dll.RARReadHeaderEx(handle, ctypes.byref(header)) == ERAR_SUCCESS:
                entry = RarEntry(
                    filename=header.FileNameW if header.FileNameW else header.FileName.decode('utf-8', errors='replace'),
                    compressed_size=(header.PackSizeHigh << 32) | header.PackSize,
                    uncompressed_size=(header.UnpSizeHigh << 32) | header.UnpSize,
                    crc32=header.FileCRC,
                    is_directory=bool(header.Flags & RHDF_DIRECTORY),
                    is_encrypted=bool(header.Flags & RHDF_ENCRYPTED),
                )
                entries.append(entry)
                self._unrar_dll.RARProcessFile(handle, RAR_SKIP, None, None)

        finally:
            self._unrar_dll.RARCloseArchive(handle)

        return entries

    def extract_all(self) -> Dict[str, bytes]:
        """
        Extract all files to memory.

        Returns:
            Dictionary mapping filename to file content bytes.
        """
        if not self._temp_path or not self._unrar_dll:
            return {}

        result = {}

        arc_data = RAROpenArchiveDataEx()
        arc_data.ArcName = str(self._temp_path).encode('utf-8')
        arc_data.OpenMode = RAR_OM_EXTRACT

        handle = self._unrar_dll.RAROpenArchiveEx(ctypes.byref(arc_data))
        if not handle or arc_data.OpenResult != ERAR_SUCCESS:
            self._last_error = f"Failed to open archive: {arc_data.OpenResult}"
            logger.debug(f"[RAM RAR] Failed to open archive: error {arc_data.OpenResult}")
            return result

        try:
            # Set password if provided
            if self._password:
                self._unrar_dll.RARSetPassword(handle, self._password.encode('utf-8'))

            # Setup callback for data capture
            cb_data = _CallbackData()
            cb_data.password = self._password
            # Pass temp_dir for multi-volume support
            if self._temp_dir_for_volumes:
                cb_data.temp_dir = self._temp_dir_for_volumes
            elif self._temp_path:
                cb_data.temp_dir = self._temp_path.parent
            callback = _create_callback(cb_data)
            self._unrar_dll.RARSetCallback(handle, callback, 0)

            header = RARHeaderDataEx()
            file_count = 0
            total_bytes = 0

            logger.info(f"[RAM RAR] Starting file extraction loop...")

            while self._unrar_dll.RARReadHeaderEx(handle, ctypes.byref(header)) == ERAR_SUCCESS:
                filename = header.FileNameW if header.FileNameW else header.FileName.decode('utf-8', errors='replace')
                is_dir = bool(header.Flags & RHDF_DIRECTORY)
                unp_size = (header.UnpSizeHigh << 32) | header.UnpSize

                cb_data.buffer = BytesIO()
                cb_data.current_file = filename

                # Log progress for each file
                if not is_dir:
                    logger.debug(f"[RAM RAR] Extracting: {filename} ({unp_size / (1024*1024):.1f} MB)")

                # RAR_TEST extracts to callback without disk write
                proc_result = self._unrar_dll.RARProcessFile(handle, RAR_TEST, None, None)

                if proc_result == ERAR_SUCCESS and not is_dir:
                    data = cb_data.buffer.getvalue()
                    result[filename] = data
                    file_count += 1
                    total_bytes += len(data)

                    # Log progress every 10 files
                    if file_count % 10 == 0:
                        logger.info(f"[RAM RAR] Progress: {file_count} files, {total_bytes / (1024*1024):.1f} MB")

                elif proc_result != ERAR_SUCCESS and not is_dir:
                    # Log extraction errors for debugging
                    error_names = {
                        10: "END_ARCHIVE", 11: "NO_MEMORY", 12: "BAD_DATA",
                        13: "BAD_ARCHIVE", 14: "UNKNOWN_FORMAT", 15: "EOPEN",
                        16: "ECREATE", 17: "ECLOSE", 18: "EREAD", 19: "EWRITE",
                        20: "SMALL_BUF", 21: "UNKNOWN", 22: "MISSING_PASSWORD",
                        24: "BAD_PASSWORD"
                    }
                    error_name = error_names.get(proc_result, f"ERROR_{proc_result}")
                    logger.warning(f"[RAM RAR] Failed to extract {filename}: {error_name}")

                    # Check if we should abort
                    if proc_result in (ERAR_MISSING_PASSWORD, ERAR_BAD_PASSWORD):
                        logger.error(f"[RAM RAR] Password error - aborting extraction")
                        break

        finally:
            self._unrar_dll.RARCloseArchive(handle)

        logger.info(f"[RAM RAR] Extracted {len(result)} files ({total_bytes / (1024*1024):.1f} MB) from RAM")
        return result

    def extract_file(self, filename: str) -> Optional[bytes]:
        """Extract a single file to memory."""
        if not self._temp_path or not self._unrar_dll:
            return None

        arc_data = RAROpenArchiveDataEx()
        arc_data.ArcName = str(self._temp_path).encode('utf-8')
        arc_data.OpenMode = RAR_OM_EXTRACT

        handle = self._unrar_dll.RAROpenArchiveEx(ctypes.byref(arc_data))
        if not handle or arc_data.OpenResult != ERAR_SUCCESS:
            return None

        result = None
        try:
            if self._password:
                self._unrar_dll.RARSetPassword(handle, self._password.encode('utf-8'))

            cb_data = _CallbackData()
            cb_data.password = self._password
            callback = _create_callback(cb_data)
            self._unrar_dll.RARSetCallback(handle, callback, 0)

            header = RARHeaderDataEx()
            while self._unrar_dll.RARReadHeaderEx(handle, ctypes.byref(header)) == ERAR_SUCCESS:
                current = header.FileNameW if header.FileNameW else header.FileName.decode('utf-8', errors='replace')

                if current == filename:
                    cb_data.buffer = BytesIO()
                    proc_result = self._unrar_dll.RARProcessFile(handle, RAR_TEST, None, None)
                    if proc_result == ERAR_SUCCESS:
                        result = cb_data.buffer.getvalue()
                    break
                else:
                    self._unrar_dll.RARProcessFile(handle, RAR_SKIP, None, None)

        finally:
            self._unrar_dll.RARCloseArchive(handle)

        return result

    def extract_to_disk(self, extract_path: Path) -> int:
        """
        Extract all files directly to disk.

        This is the memory-efficient extraction method - files are written
        directly to disk by UnRAR without loading into memory first.

        Args:
            extract_path: Destination directory

        Returns:
            Number of files extracted
        """
        if not self._temp_path or not self._unrar_dll:
            return 0

        extract_path.mkdir(parents=True, exist_ok=True)
        extract_path_str = str(extract_path)
        count = 0

        arc_data = RAROpenArchiveDataEx()
        arc_data.ArcName = str(self._temp_path).encode('utf-8')
        arc_data.OpenMode = RAR_OM_EXTRACT

        handle = self._unrar_dll.RAROpenArchiveEx(ctypes.byref(arc_data))
        if not handle or arc_data.OpenResult != ERAR_SUCCESS:
            self._last_error = f"Failed to open archive: {arc_data.OpenResult}"
            return 0

        try:
            # Set password if provided
            if self._password:
                self._unrar_dll.RARSetPassword(handle, self._password.encode('utf-8'))

            # Setup callback for volume switching and password
            cb_data = _CallbackData()
            cb_data.password = self._password
            callback = _create_callback(cb_data)
            self._unrar_dll.RARSetCallback(handle, callback, 0)

            header = RARHeaderDataEx()
            while self._unrar_dll.RARReadHeaderEx(handle, ctypes.byref(header)) == ERAR_SUCCESS:
                filename = header.FileNameW if header.FileNameW else header.FileName.decode('utf-8', errors='replace')
                is_dir = bool(header.Flags & RHDF_DIRECTORY)

                # RAR_EXTRACT mode: extracts to DestPath (2nd arg) or DestName (3rd arg)
                # Using DestPath extracts with original path structure
                proc_result = self._unrar_dll.RARProcessFile(
                    handle,
                    RAR_EXTRACT,
                    extract_path_str.encode('utf-8'),  # DestPath
                    None  # DestName (not used)
                )

                if proc_result == ERAR_SUCCESS and not is_dir:
                    count += 1
                elif proc_result != ERAR_SUCCESS:
                    logger.warning(f"[RAM RAR] Failed to extract {filename}: error {proc_result}")

        finally:
            self._unrar_dll.RARCloseArchive(handle)

        logger.info(f"[RAM RAR] Extracted {count} files directly to disk")
        return count

    def get_last_error(self) -> str:
        """Get the last error message."""
        return self._last_error

    def _cleanup(self) -> None:
        """Clean up temp file."""
        if self._temp_path and self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except Exception:
                pass
        self._temp_path = None
        self._archive_data = None

    def __del__(self):
        self._cleanup()


def extract_rar_from_memory(
    data: bytes,
    password: str = ""
) -> Dict[str, bytes]:
    """
    Convenience function to extract RAR archive from memory.

    Args:
        data: RAR archive data as bytes
        password: Optional password for encrypted archives

    Returns:
        Dictionary mapping filename to file content bytes
    """
    extractor = RamRarExtractor()
    if not extractor.set_archive_data(data):
        raise RuntimeError(f"Failed to load RAR: {extractor.get_last_error()}")

    if password:
        extractor.set_password(password)

    return extractor.extract_all()


def extract_multipart_rar_from_memory(
    parts: Dict[str, bytes],
    password: str = "",
    presorted: bool = False
) -> Dict[str, bytes]:
    """
    Extract multi-part RAR archive 100% in RAM.

    Uses native C++ MultiVolumeRamExtractor for TRUE intra-RAM extraction:
    - Only ONE volume is written to temp file at a time
    - UCM_CHANGEVOLUME callback loads next volume on-demand from RAM
    - Minimal I/O compared to writing ALL temp files upfront

    Falls back to Python implementation if native module unavailable.

    Args:
        parts: Dictionary mapping part filename to data
        password: Optional password
        presorted: If True, skip sorting and use dict order (caller already sorted)

    Returns:
        Dictionary mapping extracted filename to content bytes
    """
    import re

    if not parts:
        return {}

    # === STEP 1: DETECT AND SORT VOLUMES ===
    # This must happen BEFORE trying native extractor
    filenames = [Path(name).name.lower() for name in parts.keys()]
    is_modern = any(re.search(r'\.part\d+\.rar$', f) for f in filenames)
    is_oldstyle = any(re.search(r'\.[rs]\d{2}$', f) for f in filenames)
    all_rar = all(f.endswith('.rar') for f in filenames)
    is_obfuscated = not is_modern and not is_oldstyle and len(parts) > 1

    # If caller already sorted (e.g., by buffer index for encrypted headers), use that order
    if presorted:
        sorted_parts = list(parts.items())
        logger.info(f"[RAM RAR] Using presorted order ({len(parts)} parts)")
    else:
        logger.info(f"[RAM RAR] {len(parts)} parts - modern={is_modern}, oldstyle={is_oldstyle}, "
                    f"all_rar={all_rar}, obfuscated={is_obfuscated}")

        # For obfuscated archives, try to sort by RAR volume number from headers
        if is_obfuscated and all_rar:
            logger.info("[RAM RAR] Obfuscated archive detected - sorting by RAR header volume numbers...")
            sorted_parts = sort_rar_volumes_by_header(parts)
        else:
            # Sort parts by filename pattern
            def sort_key(item):
                name = item[0].lower()
                basename = Path(name).name

                # Modern: .part001.rar, .part002.rar, etc.
                part_match = re.search(r'\.part(\d+)\.rar$', basename)
                if part_match:
                    return (0, int(part_match.group(1)), basename)

                # Old-style split: base .rar file is FIRST volume, .r00 is second, .r01 is third, etc.
                # For old-style splits, the base .rar MUST come before .rXX files
                if is_oldstyle and basename.endswith('.rar') and not re.search(r'\.[rs]\d{2}\.rar$', basename):
                    # This is the base .rar in an old-style split - it's the FIRST volume
                    return (1, -1, basename)

                # .rXX or .sXX extension (old-style continuation volumes)
                ext_match = re.search(r'\.([rs])(\d{2})$', basename)
                if ext_match:
                    letter = ext_match.group(1)
                    num = int(ext_match.group(2))
                    # .r00 = 0, .r01 = 1, ..., .s00 = 100, .s01 = 101, ...
                    offset = 0 if letter == 'r' else 100
                    return (1, offset + num, basename)

                # Generic .rar file (not part of old-style split)
                if basename.endswith('.rar'):
                    num_match = re.search(r'(\d+)\.rar$', basename)
                    if num_match:
                        return (2, int(num_match.group(1)), basename)
                    return (2, 0, basename)

                # Any other file with numbers
                num_match = re.search(r'(\d+)', basename)
                if num_match:
                    return (3, int(num_match.group(1)), basename)
                return (4, 0, basename)

            sorted_parts = sorted(parts.items(), key=sort_key)

    logger.info(f"[RAM RAR] First: {Path(sorted_parts[0][0]).name}, Last: {Path(sorted_parts[-1][0]).name}")

    # === STEP 2: TRY NATIVE MULTIVOLUME EXTRACTOR ===
    # This is TRUE intra-RAM: only one volume in temp file at a time
    try:
        # Try to import native module (compiled C++ with pybind11)
        # Must use importlib to avoid importing ourselves (this file is also ram_rar.py!)
        import importlib.util
        import sys
        native_path = Path(__file__).parent.parent / "native"
        # Find the .pyd matching current Python version (e.g., cp314 for Python 3.14)
        py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
        pyd_files = list(native_path.glob(f"ram_rar.{py_ver}*.pyd"))
        if not pyd_files:
            # Fallback: try any version
            pyd_files = list(native_path.glob("ram_rar.*.pyd"))
        if not pyd_files:
            raise ImportError("Native ram_rar.pyd not found")
        spec = importlib.util.spec_from_file_location("ram_rar", pyd_files[0])
        native_rar = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native_rar)

        if hasattr(native_rar, 'MultiVolumeRamExtractor'):
            logger.info(f"[RAM RAR] Using NATIVE MultiVolumeRamExtractor ({len(sorted_parts)} volumes)")

            # Create extractor and add volumes IN SORTED ORDER
            extractor = native_rar.MultiVolumeRamExtractor()
            total_bytes = 0
            for name, data in sorted_parts:
                data_bytes = data if isinstance(data, bytes) else bytes(data)
                total_bytes += len(data_bytes)
                extractor.add_volume(name, data_bytes)

            logger.info(f"[RAM RAR] NATIVE: {len(sorted_parts)} volumes, {total_bytes / 1024 / 1024:.1f} MB")

            if password:
                extractor.set_password(password)

            # Extract - TRUE 100% RAM (only 1 temp file at a time)
            result = extractor.extract_all()

            if result:
                logger.info(f"[RAM RAR] NATIVE extraction complete: {len(result)} files")
                return result
            else:
                error = extractor.get_last_error()
                logger.warning(f"[RAM RAR] Native extraction failed: {error}, falling back to Python impl")
    except ImportError:
        logger.debug("[RAM RAR] Native ram_rar module not available, using Python implementation")
    except Exception as e:
        logger.warning(f"[RAM RAR] Native extraction error: {e}, falling back to Python impl")

    # === STEP 3: FALLBACK TO PYTHON IMPLEMENTATION ===

    # Write all volumes to temp files with TEMPORARY attribute (stays in RAM cache)
    temp_dir = Path(tempfile.mkdtemp(prefix='dler_rar_'))

    try:
        # Detect if we need to rename obfuscated files
        need_rename = False
        if all_rar and not is_modern and not is_oldstyle and len(parts) > 1:
            first_data = sorted_parts[0][1]
            if len(first_data) >= 14:
                if first_data[:7] == b'Rar!\x1a\x07\x00' or first_data[:7] == b'Rar!\x1a\x07\x01':
                    need_rename = True
                    logger.info("[RAM RAR] Detected obfuscated multi-volume RAR, renaming to standard format")

        # Write volumes with FILE_ATTRIBUTE_TEMPORARY
        if need_rename:
            for idx, (name, data) in enumerate(sorted_parts):
                new_name = f"archive.part{idx+1:03d}.rar"
                part_path = temp_dir / new_name
                part_path.write_bytes(data)
                _set_temp_attribute(part_path)
            first_part = temp_dir / "archive.part001.rar"
        else:
            for name, data in sorted_parts:
                part_path = temp_dir / Path(name).name
                part_path.write_bytes(data)
                _set_temp_attribute(part_path)

            # Find first part
            first_part = None
            for name, _ in sorted_parts:
                lower_name = Path(name).name.lower()
                if re.search(r'\.part0*1\.rar$', lower_name):
                    first_part = temp_dir / Path(name).name
                    break

            if not first_part:
                for name, _ in sorted_parts:
                    lower_name = Path(name).name.lower()
                    if lower_name.endswith('.rar') and not re.search(r'\.part\d+\.rar$', lower_name):
                        first_part = temp_dir / Path(name).name
                        break

            if not first_part:
                first_part = temp_dir / Path(sorted_parts[0][0]).name

        logger.info(f"[RAM RAR] Extracting from: {first_part.name} (100% RAM output)")

        # Use RamRarExtractor with extract_all (RAM output via callback)
        extractor = RamRarExtractor()
        extractor._temp_path = first_part
        extractor._temp_dir_for_volumes = temp_dir  # Pass temp_dir for volume switching

        if password:
            extractor.set_password(password)

        logger.info(f"[RAM RAR] Starting extraction (temp_dir: {temp_dir})...")
        # extract_all uses RAR_TEST mode with callback - output goes to RAM!
        result = extractor.extract_all()

        if not result and len(sorted_parts) > 1:
            # Try individual extraction as fallback
            logger.warning("[RAM RAR] Multi-volume extraction returned 0 files, trying individual...")
            for f in temp_dir.iterdir():
                if f.suffix.lower() == '.rar':
                    ext = RamRarExtractor()
                    ext._temp_path = f
                    if password:
                        ext.set_password(password)
                    individual_result = ext.extract_all()
                    result.update(individual_result)
                    if individual_result:
                        logger.debug(f"[RAM RAR] {f.name}: {len(individual_result)} files to RAM")

        logger.info(f"[RAM RAR] Total extracted to RAM: {len(result)} files")
        return result

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def _set_temp_attribute(path: Path) -> None:
    """Set FILE_ATTRIBUTE_TEMPORARY on Windows to keep file in RAM cache."""
    if sys.platform == 'win32':
        try:
            kernel32 = ctypes.windll.kernel32
            FILE_ATTRIBUTE_TEMPORARY = 0x100
            kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_TEMPORARY)
        except Exception:
            pass


@dataclass
class RarVolumeInfo:
    """
    Complete metadata about a RAR volume extracted from headers.

    Based on official RAR 5.0 specification from rarlab.com/technote.htm
    """
    # Basic info
    is_valid: bool = False
    rar_version: int = 0  # 4 or 5

    # Volume information (from Main Archive Header)
    is_multivolume: bool = False
    volume_number: int = -1  # 0-based: 0=first, 1=second, etc.
    is_first_volume: bool = False
    has_volume_number_field: bool = False

    # Archive state (from End Archive Header)
    is_last_volume: bool = False
    has_end_archive: bool = False

    # File continuity (from first File Header)
    first_file_split_before: bool = False  # Data continues from previous volume
    first_file_split_after: bool = False   # Data continues to next volume

    # Additional flags
    is_solid: bool = False
    is_encrypted: bool = False
    has_recovery: bool = False


def parse_rar_volume_info(data: bytes) -> RarVolumeInfo:
    """
    Parse complete volume metadata from RAR archive data.

    This is the professional, specification-compliant implementation based on:
    - RAR 5.0 archive format: https://www.rarlab.com/technote.htm
    - RAR 4.x format documentation

    Args:
        data: RAR archive data bytes

    Returns:
        RarVolumeInfo with all available metadata
    """
    info = RarVolumeInfo()

    if len(data) < 12:
        return info

    # Check RAR signature
    if data[:4] != b'Rar!':
        return info

    # Detect RAR version
    if data[4:8] == b'\x1a\x07\x01\x00':
        info.rar_version = 5
        info.is_valid = True
        _parse_rar5_volume_info(data, info)
    elif data[4:7] == b'\x1a\x07\x00':
        info.rar_version = 4
        info.is_valid = True
        _parse_rar4_volume_info(data, info)

    return info


def _parse_rar5_volume_info(data: bytes, info: RarVolumeInfo) -> None:
    """
    Parse RAR5 archive headers to extract volume information.

    RAR5 Header Structure:
    - Signature: 8 bytes (Rar!\x1a\x07\x01\x00)
    - Headers: sequence of [CRC32(4) + Size(vint) + Type(vint) + Flags(vint) + Data...]

    Header Types:
    - 1: Main Archive Header
    - 2: File Header
    - 3: Service Header
    - 4: Encryption Header
    - 5: End of Archive Header
    """
    pos = 8  # After signature

    try:
        while pos < len(data) - 7:
            # Header CRC32 (4 bytes)
            pos += 4

            # Header size (vint)
            header_size, vlen = _read_vint(data, pos)
            if header_size <= 0 or vlen <= 0:
                break
            pos += vlen
            header_start = pos

            # Header type (vint)
            header_type, vlen = _read_vint(data, pos)
            if vlen <= 0:
                break
            pos += vlen

            # Header flags (vint)
            header_flags, vlen = _read_vint(data, pos)
            if vlen <= 0:
                break
            pos += vlen

            # === Main Archive Header (type 1) ===
            if header_type == 1:
                # Skip extra area size if present (header_flags & 0x0001)
                if header_flags & 0x0001:
                    _, vlen = _read_vint(data, pos)
                    if vlen <= 0:
                        break
                    pos += vlen

                # Archive flags (vint)
                archive_flags, vlen = _read_vint(data, pos)
                if vlen <= 0:
                    break
                pos += vlen

                # Parse archive flags (per RAR5 spec):
                # 0x0001: Volume (part of multivolume set)
                # 0x0002: Volume number field is present
                # 0x0004: Solid archive
                # 0x0008: Recovery record present
                # 0x0010: Locked archive

                info.is_multivolume = bool(archive_flags & 0x0001)
                info.has_volume_number_field = bool(archive_flags & 0x0002)
                info.is_solid = bool(archive_flags & 0x0004)
                info.has_recovery = bool(archive_flags & 0x0008)

                # Volume number field (vint, only if flag 0x0002 is set)
                # Per spec: "Not present for first volume, 1 for second volume, 2 for third..."
                if info.has_volume_number_field:
                    vol_num, vlen = _read_vint(data, pos)
                    if vlen > 0:
                        # Spec says 1=second, 2=third, so we store as-is (1-based for non-first)
                        info.volume_number = vol_num
                        info.is_first_volume = False
                else:
                    # No volume number field = first volume (or single-volume archive)
                    if info.is_multivolume:
                        info.is_first_volume = True
                        info.volume_number = 0
                    else:
                        info.is_first_volume = True
                        info.volume_number = 0

            # === File Header (type 2) - check first file for SPLIT flags ===
            elif header_type == 2:
                # General header flags for file:
                # 0x0008: Data area continuing from previous volume (SPLIT_BEFORE)
                # 0x0010: Data area continuing in next volume (SPLIT_AFTER)
                info.first_file_split_before = bool(header_flags & 0x0008)
                info.first_file_split_after = bool(header_flags & 0x0010)

                # First volume won't have SPLIT_BEFORE on first file
                if not info.first_file_split_before and info.is_multivolume:
                    info.is_first_volume = True
                    if info.volume_number < 0:
                        info.volume_number = 0

            # === End of Archive Header (type 5) ===
            elif header_type == 5:
                info.has_end_archive = True

                # End archive flags (vint)
                # 0x0001: Archive is volume and NOT the last in the set
                if pos < header_start + header_size:
                    end_flags, _ = _read_vint(data, pos)
                    info.is_last_volume = not bool(end_flags & 0x0001)
                else:
                    info.is_last_volume = True
                break  # End of archive reached

            # Skip to next header
            pos = header_start + header_size

    except Exception as e:
        logger.debug(f"[RAR5] Parse error: {e}")


def _parse_rar4_volume_info(data: bytes, info: RarVolumeInfo) -> None:
    """
    Parse RAR4 (RAR 2.x-4.x) archive headers to extract volume information.

    RAR4 Header Structure:
    - Signature: 7 bytes (Rar!\x1a\x07\x00)
    - Headers: sequence of [CRC16(2) + Type(1) + Flags(2) + Size(2) + Data...]

    Header Types:
    - 0x72: MARK_HEAD (marker block)
    - 0x73: MAIN_HEAD (main archive header)
    - 0x74: FILE_HEAD (file header)
    - 0x7b: ENDARC (end of archive)
    """
    pos = 7  # After signature

    try:
        while pos < len(data) - 7:
            # Header structure: CRC(2) + Type(1) + Flags(2) + Size(2)
            head_crc = int.from_bytes(data[pos:pos+2], 'little')
            head_type = data[pos+2]
            head_flags = int.from_bytes(data[pos+3:pos+5], 'little')
            head_size = int.from_bytes(data[pos+5:pos+7], 'little')

            if head_size < 7:
                break

            # === Main Archive Header (0x73) ===
            if head_type == 0x73:
                # Main header flags:
                # 0x0001: MHD_VOLUME - archive is part of multivolume set
                # 0x0002: MHD_COMMENT - comment present
                # 0x0004: MHD_LOCK - archive is locked
                # 0x0008: MHD_SOLID - solid archive
                # 0x0010: MHD_NEWNUMBERING - new volume naming (partN.rar)
                # 0x0020: MHD_AV - authenticity information present
                # 0x0040: MHD_PROTECT - recovery record present
                # 0x0080: MHD_PASSWORD - encrypted archive
                # 0x0100: MHD_FIRSTVOLUME - first volume (RAR 3.0+)
                # 0x0200: MHD_ENCRYPTVER - encryption version present

                info.is_multivolume = bool(head_flags & 0x0001)
                info.is_solid = bool(head_flags & 0x0008)
                info.has_recovery = bool(head_flags & 0x0040)
                info.is_encrypted = bool(head_flags & 0x0080)
                info.is_first_volume = bool(head_flags & 0x0100)

                if info.is_first_volume:
                    info.volume_number = 0

            # === File Header (0x74) - check first file for SPLIT flags ===
            elif head_type == 0x74:
                # File header flags:
                # 0x0001: LHD_SPLIT_BEFORE - file continued from previous volume
                # 0x0002: LHD_SPLIT_AFTER - file continued in next volume
                # 0x0004: LHD_PASSWORD - file encrypted

                info.first_file_split_before = bool(head_flags & 0x0001)
                info.first_file_split_after = bool(head_flags & 0x0002)

                # First volume won't have SPLIT_BEFORE on first file
                if not info.first_file_split_before and info.is_multivolume:
                    if not info.is_first_volume:
                        info.is_first_volume = True
                        if info.volume_number < 0:
                            info.volume_number = 0

            # === End of Archive (0x7b) ===
            elif head_type == 0x7b:
                info.has_end_archive = True

                # ENDARC flags at offset 7 (after base header)
                if head_size >= 9:
                    endarc_flags = int.from_bytes(data[pos+7:pos+9], 'little')

                    # ENDARC flags:
                    # 0x0001: NEXTVOL - next volume exists
                    # 0x0002: DATACRC - data CRC present (4 bytes)
                    # 0x0004: REVSPACE - rev space present (7 bytes)
                    # 0x0008: VOLNUMBER - volume number present (4 bytes)

                    info.is_last_volume = not bool(endarc_flags & 0x0001)

                    # Extract volume number if present
                    data_offset = pos + 9

                    if endarc_flags & 0x0002:  # DATACRC (4 bytes)
                        data_offset += 4

                    if endarc_flags & 0x0004:  # REVSPACE (7 bytes)
                        data_offset += 7

                    if endarc_flags & 0x0008:  # VOLNUMBER (4 bytes)
                        if data_offset + 4 <= pos + head_size:
                            info.volume_number = int.from_bytes(
                                data[data_offset:data_offset+4], 'little'
                            )
                            info.has_volume_number_field = True

                break  # End of archive reached

            # Calculate ADD_SIZE if flag set
            add_size = 0
            if head_flags & 0x8000:
                if pos + 11 <= len(data):
                    add_size = int.from_bytes(data[pos+7:pos+11], 'little')

            pos += head_size + add_size

    except Exception as e:
        logger.debug(f"[RAR4] Parse error: {e}")


def get_rar_volume_number(data: bytes) -> int:
    """
    Extract volume number from RAR archive data.

    Parses RAR4 and RAR5 headers to extract the volume number for
    multi-volume archives. This is essential for sorting obfuscated
    RAR volumes that don't have meaningful filenames.

    Args:
        data: RAR archive data bytes

    Returns:
        Volume number (0-based), or -1 if cannot determine
    """
    info = parse_rar_volume_info(data)

    if not info.is_valid:
        return -1

    return info.volume_number


def is_first_rar_volume(data: bytes) -> bool:
    """
    Check if this RAR file is the first volume of a multi-volume archive.

    Uses complete metadata parsing for accurate detection.

    Args:
        data: RAR archive data bytes

    Returns:
        True if this is the first volume, False otherwise
    """
    info = parse_rar_volume_info(data)

    if not info.is_valid:
        return False

    # If not multi-volume, treat as "first" (only volume)
    if not info.is_multivolume:
        return True

    return info.is_first_volume


def is_last_rar_volume(data: bytes) -> bool:
    """
    Check if this RAR file is the last volume of a multi-volume archive.

    Args:
        data: RAR archive data bytes

    Returns:
        True if this is the last volume, False otherwise
    """
    info = parse_rar_volume_info(data)

    if not info.is_valid:
        return False

    # If not multi-volume, treat as "last" (only volume)
    if not info.is_multivolume:
        return True

    return info.is_last_volume


def _read_vint(data: bytes, offset: int) -> Tuple[int, int]:
    """
    Read RAR5 variable-length integer.

    RAR5 uses vint format: 7 bits per byte, MSB indicates continuation.

    Returns:
        (value, bytes_read) or (-1, 0) on error
    """
    value = 0
    shift = 0
    bytes_read = 0

    while offset + bytes_read < len(data):
        byte = data[offset + bytes_read]
        bytes_read += 1
        value |= (byte & 0x7F) << shift
        shift += 7

        if (byte & 0x80) == 0:  # No continuation bit
            break

        if bytes_read > 10:  # Safety limit
            return -1, 0

    return value, bytes_read


def sort_rar_volumes_by_header(parts: Dict[str, bytes]) -> List[Tuple[str, bytes]]:
    """
    Sort RAR volumes using complete metadata from archive headers.

    Professional auto-adaptive solution based on RAR specification.
    Uses multiple detection strategies in priority order:
    1. Volume numbers from Main Archive Header (RAR5) or ENDARC (RAR4)
    2. First volume detection (no volume_number field in RAR5, MHD_FIRSTVOLUME in RAR4)
    3. SPLIT_BEFORE flag on first file header (first volume won't have this)
    4. Last volume detection via End Archive Header

    Args:
        parts: Dictionary mapping filename to RAR data

    Returns:
        List of (filename, data) tuples sorted by volume number
    """
    if not parts:
        return []

    # Parse complete metadata for each volume
    volume_data: List[Tuple[str, bytes, RarVolumeInfo]] = []
    first_volume_name = None
    last_volume_name = None

    # Detect RAR format from first file
    first_data = next(iter(parts.values()))
    rar_format = "unknown"
    if len(first_data) >= 8:
        if first_data[4:8] == b'\x1a\x07\x01\x00':
            rar_format = "RAR5"
        elif first_data[4:7] == b'\x1a\x07\x00':
            rar_format = "RAR4"

    logger.info(f"[RAR SORT] Analyzing {len(parts)} volumes (format: {rar_format})...")

    for name, data in parts.items():
        info = parse_rar_volume_info(data)
        volume_data.append((name, data, info))

        # Track first and last volumes
        if info.is_first_volume:
            if first_volume_name is None:
                first_volume_name = name
                logger.info(f"[RAR SORT] FIRST volume: {Path(name).name} "
                           f"(vol_num={info.volume_number}, split_before={info.first_file_split_before})")
            else:
                logger.warning(f"[RAR SORT] Multiple first volumes! Also: {Path(name).name}")

        if info.is_last_volume and info.has_end_archive:
            if last_volume_name is None:
                last_volume_name = name
                logger.info(f"[RAR SORT] LAST volume: {Path(name).name}")

        # Log details for debugging
        if info.volume_number >= 0:
            logger.debug(f"[RAR SORT] {Path(name).name}: vol={info.volume_number}, "
                        f"first={info.is_first_volume}, last={info.is_last_volume}, "
                        f"split_before={info.first_file_split_before}")

    # Statistics
    valid_vol_nums = sum(1 for _, _, i in volume_data if i.volume_number >= 0)
    first_count = sum(1 for _, _, i in volume_data if i.is_first_volume)
    encrypted_count = sum(1 for _, _, i in volume_data if i.is_encrypted)

    vol_nums = sorted([i.volume_number for _, _, i in volume_data if i.volume_number >= 0])
    if vol_nums:
        logger.info(f"[RAR SORT] Volume numbers: {vol_nums[:10]}{'...' if len(vol_nums) > 10 else ''}")

    logger.info(f"[RAR SORT] Stats: {valid_vol_nums}/{len(parts)} have vol_num, "
               f"{first_count} first, {encrypted_count} encrypted")

    # === SORTING STRATEGY ===

    # Strategy 1: All volumes have valid numbers
    if valid_vol_nums == len(parts):
        volume_data.sort(key=lambda x: x[2].volume_number)
        logger.info(f"[RAR SORT] SUCCESS: Sorted all {len(parts)} volumes by metadata volume numbers")
        return [(name, data) for name, data, _ in volume_data]

    # Strategy 2: We found first volume + some have numbers
    if first_volume_name and valid_vol_nums > 0:
        sorted_result = []

        # First volume goes first
        for name, data, info in volume_data:
            if info.is_first_volume:
                sorted_result.append((name, data))
                break

        # Sort remaining by volume number, then by name for unknowns
        remaining = [(n, d, i) for n, d, i in volume_data if not i.is_first_volume]
        remaining.sort(key=lambda x: (
            x[2].volume_number < 0,  # Unknown numbers last
            x[2].volume_number,       # Then by volume number
            x[0]                      # Then by name
        ))

        for name, data, _ in remaining:
            sorted_result.append((name, data))

        logger.info(f"[RAR SORT] SUCCESS: First volume + {valid_vol_nums - 1} sorted by number")
        return sorted_result

    # Strategy 3: First volume detected but no numbers (encrypted headers)
    if first_volume_name:
        sorted_result = []

        # First volume goes first
        for name, data, info in volume_data:
            if info.is_first_volume:
                sorted_result.append((name, data))
                break

        # Rest in original order (or by filename)
        remaining = [(n, d) for n, d, i in volume_data if not i.is_first_volume]
        remaining.sort(key=lambda x: x[0])

        for name, data in remaining:
            sorted_result.append((name, data))

        logger.info(f"[RAR SORT] PARTIAL: First volume found, rest sorted by filename")
        return sorted_result

    # Strategy 4: Partial volume numbers available
    if valid_vol_nums > 0:
        volume_data.sort(key=lambda x: (
            x[2].volume_number < 0,  # Unknown last
            x[2].volume_number,
            x[0]
        ))
        logger.info(f"[RAR SORT] PARTIAL: {valid_vol_nums}/{len(parts)} sorted by volume number")
        return [(name, data) for name, data, _ in volume_data]

    # Strategy 5: No metadata available - return original order
    logger.warning("[RAR SORT] FALLBACK: No volume metadata found, using original order")
    return list(parts.items())


def extract_multipart_rar_to_disk(
    parts: Dict[str, bytes],
    extract_path: Path,
    password: str = "",
    presorted: bool = False
) -> int:
    """
    Extract multi-part RAR archive directly to disk.

    Uses native C++ MultiVolumeRamExtractor for TRUE intra-RAM extraction:
    - Only ONE volume is written to temp file at a time
    - UCM_CHANGEVOLUME callback loads next volume on-demand from RAM
    - Much faster than writing ALL temp files upfront

    Falls back to Python implementation if native module unavailable.

    Args:
        parts: Dictionary mapping part filename to data
        extract_path: Destination directory for extracted files
        password: Optional password
        presorted: If True, skip sorting and use provided order (for encrypted headers)

    Returns:
        Number of extracted files
    """
    import tempfile
    import re

    if not parts:
        return 0

    # Ensure extract path exists
    extract_path.mkdir(parents=True, exist_ok=True)

    # === STEP 1: DETECT AND SORT VOLUMES ===
    # Skip if caller indicates data is already sorted (encrypted headers case)
    if presorted:
        logger.info(f"[RAM RAR] {len(parts)} parts - using pre-sorted order (encrypted headers)")
        sorted_parts = list(parts.items())
    else:
        # This must happen BEFORE trying native extractor
        filenames = [Path(name).name.lower() for name in parts.keys()]

        is_modern = any(re.search(r'\.part\d+\.rar$', f) for f in filenames)
        is_oldstyle = any(re.search(r'\.[rs]\d{2}$', f) for f in filenames)
        all_rar = all(f.endswith('.rar') for f in filenames)
        is_obfuscated = not is_modern and not is_oldstyle and len(parts) > 1

        logger.info(f"[RAM RAR] {len(parts)} parts - modern={is_modern}, oldstyle={is_oldstyle}, "
                    f"all_rar={all_rar}, obfuscated={is_obfuscated}")

        # For obfuscated archives, try to sort by RAR volume number from headers
        if is_obfuscated and all_rar:
            logger.info("[RAM RAR] Obfuscated archive detected - sorting by RAR header volume numbers...")
            sorted_parts = sort_rar_volumes_by_header(parts)
        else:
            # Sort parts based on detected convention
            def sort_key(item):
                name = item[0].lower()
                basename = Path(name).name

                # Modern naming: name.part001.rar -> extract number
                part_match = re.search(r'\.part(\d+)\.rar$', basename)
                if part_match:
                    return (0, int(part_match.group(1)), basename)

                # Old-style split: base .rar file is FIRST volume, .r00 is second, .r01 is third, etc.
                # CRITICAL: When is_oldstyle=True, the base .rar MUST come before .rXX files
                if is_oldstyle and basename.endswith('.rar') and not re.search(r'\.[rs]\d{2}\.rar$', basename):
                    # This is the base .rar in an old-style split - it's the FIRST volume
                    return (1, -1, basename)

                # Old-style extensions: .r00-.r99, .s00-.s99
                ext_match = re.search(r'\.([rs])(\d{2})$', basename)
                if ext_match:
                    letter = ext_match.group(1)
                    num = int(ext_match.group(2))
                    # .r00 = 0, .r01 = 1, ..., .s00 = 100, .s01 = 101, ...
                    offset = 0 if letter == 'r' else 100
                    return (1, offset + num, basename)

                # Generic .rar file (not part of old-style split)
                if basename.endswith('.rar'):
                    num_match = re.search(r'(\d+)\.rar$', basename)
                    if num_match:
                        return (2, int(num_match.group(1)), basename)
                    return (2, 0, basename)

                # Fallback: natural sort by extracting any number
                num_match = re.search(r'(\d+)', basename)
                if num_match:
                    return (3, int(num_match.group(1)), basename)
                return (4, 0, basename)

            sorted_parts = sorted(parts.items(), key=sort_key)

    logger.info(f"[RAM RAR] First: {Path(sorted_parts[0][0]).name}, Last: {Path(sorted_parts[-1][0]).name}")

    # === STEP 2: TRY NATIVE MULTIVOLUME EXTRACTOR ===
    try:
        # Try to import native module (compiled C++ with pybind11)
        # Must use importlib to avoid importing ourselves (this file is also ram_rar.py!)
        import importlib.util
        import sys
        native_path = Path(__file__).parent.parent / "native"
        # Find the .pyd matching current Python version (e.g., cp314 for Python 3.14)
        py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
        pyd_files = list(native_path.glob(f"ram_rar.{py_ver}*.pyd"))
        if not pyd_files:
            # Fallback: try any version
            pyd_files = list(native_path.glob("ram_rar.*.pyd"))
        if not pyd_files:
            raise ImportError("Native ram_rar.pyd not found")
        spec = importlib.util.spec_from_file_location("ram_rar", pyd_files[0])
        native_rar = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native_rar)

        if hasattr(native_rar, 'MultiVolumeRamExtractor'):
            logger.info(f"[RAM RAR] Using NATIVE MultiVolumeRamExtractor ({len(sorted_parts)} volumes) -> disk")

            # Create extractor and add volumes IN SORTED ORDER
            extractor = native_rar.MultiVolumeRamExtractor()
            for name, data in sorted_parts:
                extractor.add_volume(name, data)

            if password:
                extractor.set_password(password)

            # Extract directly to disk (memory efficient)
            count = extractor.extract_to_disk(str(extract_path))

            if count > 0:
                logger.info(f"[RAM RAR] NATIVE disk extraction complete: {count} files")
                return count
            else:
                error = extractor.get_last_error()
                logger.warning(f"[RAM RAR] Native disk extraction failed: {error}, falling back")
    except ImportError:
        logger.debug("[RAM RAR] Native ram_rar module not available")
    except Exception as e:
        logger.warning(f"[RAM RAR] Native extraction error: {e}, falling back")

    # === STEP 3: FALLBACK TO PYTHON IMPLEMENTATION ===

    # For multi-part RAR, UnRAR needs all volumes as files
    temp_dir = Path(tempfile.mkdtemp(prefix='dler_rar_'))

    try:
        # Detect if we need to rename obfuscated files to standard naming
        need_rename = False

        if all_rar and not is_modern and not is_oldstyle and len(parts) > 1:
            first_data = sorted_parts[0][1]
            if len(first_data) >= 14:
                if first_data[:7] == b'Rar!\x1a\x07\x00' or first_data[:7] == b'Rar!\x1a\x07\x01':
                    need_rename = True
                    logger.info("[RAM RAR] Detected obfuscated multi-volume RAR, will rename to standard format")

        if need_rename:
            logger.info(f"[RAM RAR] Renaming {len(sorted_parts)} obfuscated parts to standard naming...")
            for idx, (name, data) in enumerate(sorted_parts):
                new_name = f"archive.part{idx+1:03d}.rar"
                part_path = temp_dir / new_name
                part_path.write_bytes(data)
                logger.debug(f"[RAM RAR] {Path(name).name} -> {new_name}")

            first_part = temp_dir / "archive.part001.rar"
        else:
            # Write with original names
            for name, data in sorted_parts:
                part_path = temp_dir / Path(name).name
                part_path.write_bytes(data)

            # Find first part
            first_part = None

            # Modern naming (.part001.rar)
            for name, _ in sorted_parts:
                lower_name = Path(name).name.lower()
                if re.search(r'\.part0*1\.rar$', lower_name):
                    first_part = temp_dir / Path(name).name
                    break

            # Old-style (.rar without .partXXX)
            if not first_part:
                for name, _ in sorted_parts:
                    lower_name = Path(name).name.lower()
                    if lower_name.endswith('.rar') and not re.search(r'\.part\d+\.rar$', lower_name):
                        first_part = temp_dir / Path(name).name
                        break

            # Fallback
            if not first_part:
                first_part = temp_dir / Path(sorted_parts[0][0]).name

        logger.info(f"[RAM RAR] Extracting from: {first_part.name} -> {extract_path}")

        # Use UnRAR to extract directly to disk (memory-efficient)
        extractor = RamRarExtractor()
        extractor._temp_path = first_part

        if password:
            extractor.set_password(password)

        # Extract directly to destination using RAR_EXTRACT mode
        count = extractor.extract_to_disk(extract_path)

        if count == 0 and len(sorted_parts) > 1:
            # Try individual extraction if multi-volume failed
            logger.warning("[RAM RAR] Multi-volume extraction returned 0 files, trying individual extraction...")
            for f in temp_dir.iterdir():
                if f.suffix.lower() == '.rar':
                    extractor = RamRarExtractor()
                    extractor._temp_path = f
                    if password:
                        extractor.set_password(password)
                    individual_count = extractor.extract_to_disk(extract_path)
                    count += individual_count
                    logger.info(f"[RAM RAR] {f.name}: {individual_count} files")

        logger.info(f"[RAM RAR] Total extracted: {count} files to disk")
        return count

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
