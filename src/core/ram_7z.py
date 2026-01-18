"""
DLER RAM 7z Extractor
=====================

Fast 7z extraction using 7z.dll via COM interfaces.

Architecture:
  1. 7z data in RAM -> IInStream implementation (reads from memory buffer)
  2. 7z.dll extracts via IInArchive interface
  3. IArchiveExtractCallback captures output -> RAM buffers

This module provides high performance by:
  - Using 7z.dll directly (no subprocess overhead)
  - Streaming data from memory buffers
  - Capturing decompressed data via COM callbacks

Copyright (c) 2025 DLER Project
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Only works on Windows
if sys.platform != 'win32':
    logger.warning("ram_7z module only supported on Windows")

# ============================================================================
# 7-Zip GUIDs (from 7z SDK)
# ============================================================================

class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, data1=0, data2=0, data3=0, data4=None):
        super().__init__()
        self.Data1 = data1
        self.Data2 = data2
        self.Data3 = data3
        if data4:
            for i, b in enumerate(data4):
                self.Data4[i] = b


# Archive format GUIDs
CLSID_CFormat7z = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x01, 0x10, 0x07, 0x00, 0x00, 0x00))
CLSID_CFormatZip = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x01, 0x10, 0x01, 0x00, 0x00, 0x00))
CLSID_CFormatRar = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x01, 0x10, 0x03, 0x00, 0x00, 0x00))
CLSID_CFormatRar5 = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x01, 0x10, 0xCC, 0x00, 0x00, 0x00))

# Interface GUIDs
IID_IInArchive = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x00, 0x06, 0x00, 0x60, 0x00, 0x00))
IID_IInStream = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00))
IID_ISequentialInStream = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00))
IID_ISequentialOutStream = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00))
IID_IArchiveExtractCallback = GUID(0x23170F69, 0x40C1, 0x278A, (0x00, 0x00, 0x00, 0x06, 0x00, 0x20, 0x00, 0x00))

# Property IDs
kpidPath = 3
kpidIsDir = 6
kpidSize = 7
kpidPackSize = 8
kpidAttrib = 9
kpidCRC = 10
kpidEncrypted = 16

# Extract operation results
NArchive_NExtract_NAskMode_kExtract = 0
NArchive_NExtract_NAskMode_kTest = 1
NArchive_NExtract_NAskMode_kSkip = 2

NArchive_NExtract_NOperationResult_kOK = 0
NArchive_NExtract_NOperationResult_kUnsupportedMethod = 1
NArchive_NExtract_NOperationResult_kDataError = 2
NArchive_NExtract_NOperationResult_kCRCError = 3

# HRESULT values
S_OK = 0
E_ABORT = 0x80004004
E_NOTIMPL = 0x80004001


# ============================================================================
# PROPVARIANT structure for property access
# ============================================================================

class PROPVARIANT(ctypes.Structure):
    """Simplified PROPVARIANT for 7z properties."""
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("data", ctypes.c_ulonglong),
        ("data2", ctypes.c_ulonglong),
    ]

VT_EMPTY = 0
VT_BOOL = 11
VT_BSTR = 8
VT_UI4 = 19
VT_UI8 = 21
VT_FILETIME = 64


# ============================================================================
# COM Interface Helpers
# ============================================================================

def make_vtable(methods):
    """Create a ctypes structure for a COM vtable."""
    fields = []
    for name, restype, argtypes in methods:
        functype = ctypes.WINFUNCTYPE(restype, *argtypes)
        fields.append((name, functype))
    return type('VTable', (ctypes.Structure,), {'_fields_': fields})


# ============================================================================
# ISequentialOutStream Implementation (for receiving extracted data)
# ============================================================================

@dataclass
class ExtractedFile:
    """Extracted file data."""
    filename: str
    data: bytes
    size: int
    is_directory: bool = False


class SequentialOutStreamVTable(ctypes.Structure):
    """VTable for ISequentialOutStream."""
    _fields_ = [
        ("QueryInterface", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)),
        ("Write", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))),
    ]


class SequentialOutStream(ctypes.Structure):
    """ISequentialOutStream COM object implementation."""
    _fields_ = [
        ("lpVtbl", ctypes.POINTER(SequentialOutStreamVTable)),
    ]

    _instances = {}  # Track instances to prevent GC

    def __init__(self):
        super().__init__()
        self._buffer = BytesIO()
        self._ref_count = 1

        # Store reference to prevent GC
        self._id = id(self)
        SequentialOutStream._instances[self._id] = self

        # Create VTable
        self._vtable = SequentialOutStreamVTable()
        self._vtable.QueryInterface = self._make_query_interface()
        self._vtable.AddRef = self._make_add_ref()
        self._vtable.Release = self._make_release()
        self._vtable.Write = self._make_write()
        self.lpVtbl = ctypes.pointer(self._vtable)

    def _make_query_interface(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))
        def QueryInterface(this, riid, ppvObject):
            ppvObject[0] = this
            return S_OK
        return QueryInterface

    def _make_add_ref(self):
        @ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        def AddRef(this):
            self._ref_count += 1
            return self._ref_count
        return AddRef

    def _make_release(self):
        @ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        def Release(this):
            self._ref_count -= 1
            if self._ref_count == 0:
                SequentialOutStream._instances.pop(self._id, None)
            return self._ref_count
        return Release

    def _make_write(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
        def Write(this, data, size, processedSize):
            if data and size > 0:
                chunk = ctypes.string_at(data, size)
                self._buffer.write(chunk)
            if processedSize:
                processedSize[0] = size
            return S_OK
        return Write

    def get_data(self) -> bytes:
        """Get the accumulated data."""
        return self._buffer.getvalue()

    def reset(self):
        """Reset the buffer."""
        self._buffer = BytesIO()


# ============================================================================
# IArchiveExtractCallback Implementation
# ============================================================================

class ArchiveExtractCallbackVTable(ctypes.Structure):
    """VTable for IArchiveExtractCallback."""
    _fields_ = [
        # IUnknown
        ("QueryInterface", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)),
        # IProgress
        ("SetTotal", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint64)),
        ("SetCompleted", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64))),
        # IArchiveExtractCallback
        ("GetStream", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32)),
        ("PrepareOperation", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int32)),
        ("SetOperationResult", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int32)),
    ]


class ArchiveExtractCallback(ctypes.Structure):
    """IArchiveExtractCallback COM object implementation."""
    _fields_ = [
        ("lpVtbl", ctypes.POINTER(ArchiveExtractCallbackVTable)),
    ]

    _instances = {}  # Track instances to prevent GC

    def __init__(self, file_info_callback: Callable[[int], Tuple[str, bool]] = None):
        super().__init__()
        self._ref_count = 1
        self._current_stream: Optional[SequentialOutStream] = None
        self._extracted_files: Dict[str, bytes] = {}
        self._current_index = 0
        self._current_filename = ""
        self._current_is_dir = False
        self._file_info_callback = file_info_callback
        self._password = ""

        # Store reference to prevent GC
        self._id = id(self)
        ArchiveExtractCallback._instances[self._id] = self

        # Create VTable
        self._vtable = ArchiveExtractCallbackVTable()
        self._vtable.QueryInterface = self._make_query_interface()
        self._vtable.AddRef = self._make_add_ref()
        self._vtable.Release = self._make_release()
        self._vtable.SetTotal = self._make_set_total()
        self._vtable.SetCompleted = self._make_set_completed()
        self._vtable.GetStream = self._make_get_stream()
        self._vtable.PrepareOperation = self._make_prepare_operation()
        self._vtable.SetOperationResult = self._make_set_operation_result()
        self.lpVtbl = ctypes.pointer(self._vtable)

    def set_password(self, password: str):
        """Set password for encrypted archives."""
        self._password = password

    def _make_query_interface(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))
        def QueryInterface(this, riid, ppvObject):
            ppvObject[0] = this
            return S_OK
        return QueryInterface

    def _make_add_ref(self):
        @ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        def AddRef(this):
            self._ref_count += 1
            return self._ref_count
        return AddRef

    def _make_release(self):
        @ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        def Release(this):
            self._ref_count -= 1
            if self._ref_count == 0:
                ArchiveExtractCallback._instances.pop(self._id, None)
            return self._ref_count
        return Release

    def _make_set_total(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint64)
        def SetTotal(this, total):
            return S_OK
        return SetTotal

    def _make_set_completed(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64))
        def SetCompleted(this, completeValue):
            return S_OK
        return SetCompleted

    def _make_get_stream(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32)
        def GetStream(this, index, outStream, askExtractMode):
            self._current_index = index

            # Get file info via callback
            if self._file_info_callback:
                filename, is_dir = self._file_info_callback(index)
                self._current_filename = filename
                self._current_is_dir = is_dir
            else:
                self._current_filename = f"file_{index}"
                self._current_is_dir = False

            if askExtractMode != NArchive_NExtract_NAskMode_kExtract:
                outStream[0] = None
                return S_OK

            if self._current_is_dir:
                outStream[0] = None
                return S_OK

            # Create output stream
            self._current_stream = SequentialOutStream()
            outStream[0] = ctypes.addressof(self._current_stream)
            return S_OK
        return GetStream

    def _make_prepare_operation(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int32)
        def PrepareOperation(this, askExtractMode):
            return S_OK
        return PrepareOperation

    def _make_set_operation_result(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int32)
        def SetOperationResult(this, resultEOperationResult):
            if resultEOperationResult == NArchive_NExtract_NOperationResult_kOK:
                if self._current_stream and self._current_filename:
                    data = self._current_stream.get_data()
                    if data:
                        self._extracted_files[self._current_filename] = data
            self._current_stream = None
            return S_OK
        return SetOperationResult

    def get_extracted_files(self) -> Dict[str, bytes]:
        """Get all extracted files."""
        return self._extracted_files


# ============================================================================
# IInStream Implementation (for reading archive from memory)
# ============================================================================

class InStreamVTable(ctypes.Structure):
    """VTable for IInStream."""
    _fields_ = [
        # IUnknown
        ("QueryInterface", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)),
        # ISequentialInStream
        ("Read", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))),
        # IInStream
        ("Seek", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))),
    ]


class InStream(ctypes.Structure):
    """IInStream COM object implementation for reading from memory."""
    _fields_ = [
        ("lpVtbl", ctypes.POINTER(InStreamVTable)),
    ]

    _instances = {}

    def __init__(self, data: bytes):
        super().__init__()
        self._data = data
        self._position = 0
        self._ref_count = 1

        self._id = id(self)
        InStream._instances[self._id] = self

        self._vtable = InStreamVTable()
        self._vtable.QueryInterface = self._make_query_interface()
        self._vtable.AddRef = self._make_add_ref()
        self._vtable.Release = self._make_release()
        self._vtable.Read = self._make_read()
        self._vtable.Seek = self._make_seek()
        self.lpVtbl = ctypes.pointer(self._vtable)

    def _make_query_interface(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))
        def QueryInterface(this, riid, ppvObject):
            ppvObject[0] = this
            return S_OK
        return QueryInterface

    def _make_add_ref(self):
        @ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        def AddRef(this):
            self._ref_count += 1
            return self._ref_count
        return AddRef

    def _make_release(self):
        @ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        def Release(this):
            self._ref_count -= 1
            if self._ref_count == 0:
                InStream._instances.pop(self._id, None)
            return self._ref_count
        return Release

    def _make_read(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
        def Read(this, data, size, processedSize):
            if self._position >= len(self._data):
                if processedSize:
                    processedSize[0] = 0
                return S_OK

            bytes_to_read = min(size, len(self._data) - self._position)
            chunk = self._data[self._position:self._position + bytes_to_read]
            ctypes.memmove(data, chunk, bytes_to_read)
            self._position += bytes_to_read

            if processedSize:
                processedSize[0] = bytes_to_read
            return S_OK
        return Read

    def _make_seek(self):
        @ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))
        def Seek(this, offset, seekOrigin, newPosition):
            # seekOrigin: 0=SET, 1=CUR, 2=END
            if seekOrigin == 0:  # SEEK_SET
                self._position = offset
            elif seekOrigin == 1:  # SEEK_CUR
                self._position += offset
            elif seekOrigin == 2:  # SEEK_END
                self._position = len(self._data) + offset

            self._position = max(0, min(self._position, len(self._data)))

            if newPosition:
                newPosition[0] = self._position
            return S_OK
        return Seek


# ============================================================================
# ICryptoGetTextPassword Implementation (for password-protected archives)
# ============================================================================

class CryptoGetTextPasswordVTable(ctypes.Structure):
    """VTable for ICryptoGetTextPassword."""
    _fields_ = [
        ("QueryInterface", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))),
        ("AddRef", ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)),
        ("Release", ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)),
        ("CryptoGetTextPassword", ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p))),
    ]


# ============================================================================
# Main Extractor Class
# ============================================================================

class Ram7zExtractor:
    """
    Extract 7z/ZIP archives from memory to memory using 7z.dll.

    Uses COM interfaces to interact with 7z.dll directly,
    avoiding subprocess overhead for maximum performance.

    Example:
        extractor = Ram7zExtractor()
        extractor.set_archive_data(archive_bytes)
        files = extractor.extract_all()
        for name, data in files.items():
            print(f"{name}: {len(data)} bytes")
    """

    _sevenzip_dll = None
    _dll_loaded = False
    _create_object = None

    def __init__(self):
        self._archive_data: Optional[bytes] = None
        self._password: str = ""
        self._last_error: str = ""
        self._temp_path: Optional[Path] = None

        self._load_dll()

    @classmethod
    def _load_dll(cls) -> bool:
        """Load 7z.dll."""
        if cls._dll_loaded:
            return cls._sevenzip_dll is not None

        cls._dll_loaded = True

        if sys.platform != 'win32':
            logger.warning("Ram7zExtractor only supported on Windows")
            return False

        # Try to find 7z.dll
        dll_paths = [
            Path(__file__).parent.parent.parent / "tools" / "7z.dll",
            Path(os.environ.get('PROGRAMFILES', '')) / "7-Zip" / "7z.dll",
            Path(os.environ.get('PROGRAMFILES(X86)', '')) / "7-Zip" / "7z.dll",
            "7z.dll",
        ]

        for dll_path in dll_paths:
            try:
                dll_path = Path(dll_path)
                if not dll_path.exists():
                    continue

                cls._sevenzip_dll = ctypes.WinDLL(str(dll_path))
                logger.info(f"Loaded 7z.dll: {dll_path}")

                # Get CreateObject function
                cls._create_object = cls._sevenzip_dll.CreateObject
                cls._create_object.argtypes = [
                    ctypes.POINTER(GUID),  # clsid
                    ctypes.POINTER(GUID),  # iid
                    ctypes.POINTER(ctypes.c_void_p)  # outObject
                ]
                cls._create_object.restype = ctypes.c_long

                return True

            except OSError as e:
                logger.debug(f"Failed to load {dll_path}: {e}")
                continue
            except AttributeError as e:
                logger.debug(f"CreateObject not found in {dll_path}: {e}")
                cls._sevenzip_dll = None
                continue

        logger.warning("7z.dll not found or CreateObject not available")
        return False

    @classmethod
    def is_available(cls) -> bool:
        """Check if 7z.dll is available."""
        cls._load_dll()
        return cls._sevenzip_dll is not None and cls._create_object is not None

    def set_archive_data(self, data: bytes) -> bool:
        """Load archive from bytes."""
        self._cleanup()
        self._archive_data = data
        return True

    def set_password(self, password: str) -> None:
        """Set password for encrypted archives."""
        self._password = password

    def _cleanup(self):
        """Clean up temporary resources."""
        if self._temp_path and self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except:
                pass
        self._temp_path = None

    def _detect_format(self, data: bytes) -> GUID:
        """Detect archive format from magic bytes."""
        if len(data) < 6:
            return CLSID_CFormat7z

        # 7z: 37 7A BC AF 27 1C
        if data[:6] == b'7z\xbc\xaf\x27\x1c':
            return CLSID_CFormat7z

        # ZIP: 50 4B 03 04 or 50 4B 05 06
        if data[:2] == b'PK':
            return CLSID_CFormatZip

        # RAR5: 52 61 72 21 1A 07 01 00
        if data[:8] == b'Rar!\x1a\x07\x01\x00':
            return CLSID_CFormatRar5

        # RAR: 52 61 72 21 1A 07 00
        if data[:7] == b'Rar!\x1a\x07\x00':
            return CLSID_CFormatRar

        # Default to 7z
        return CLSID_CFormat7z

    def extract_all(self) -> Dict[str, bytes]:
        """
        Extract all files from the archive.

        Returns:
            Dictionary mapping filename to file data.
        """
        if not self._archive_data:
            self._last_error = "No archive data set"
            return {}

        if not self.is_available():
            self._last_error = "7z.dll not available"
            return {}

        # For now, use temp file approach (simpler and more reliable)
        # Full COM streaming implementation can be added later
        return self._extract_via_temp_file()

    def _extract_via_temp_file(self) -> Dict[str, bytes]:
        """Extract using temp file (fallback method)."""
        try:
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix='.7z', prefix='dler_7z_')
            self._temp_path = Path(temp_path)
            os.write(fd, self._archive_data)
            os.close(fd)

            # Create IInArchive
            archive = ctypes.c_void_p()
            format_guid = self._detect_format(self._archive_data)

            hr = self._create_object(
                ctypes.byref(format_guid),
                ctypes.byref(IID_IInArchive),
                ctypes.byref(archive)
            )

            if hr != S_OK:
                self._last_error = f"CreateObject failed: 0x{hr:08X}"
                return {}

            try:
                return self._extract_from_archive(archive, str(self._temp_path))
            finally:
                # Release archive
                if archive:
                    try:
                        # Call Release on IUnknown
                        vtable = ctypes.cast(archive, ctypes.POINTER(ctypes.c_void_p))[0]
                        release_fn = ctypes.cast(
                            ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[2],
                            ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
                        )
                        release_fn(archive)
                    except:
                        pass

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[RAM 7z] Extraction failed: {e}")
            return {}
        finally:
            self._cleanup()

    def _extract_from_archive(self, archive: ctypes.c_void_p, archive_path: str) -> Dict[str, bytes]:
        """Extract files from opened archive."""
        # IInArchive vtable offsets (after IUnknown: QueryInterface, AddRef, Release)
        # 3: Open
        # 4: Close
        # 5: GetNumberOfItems
        # 6: GetProperty
        # 7: Extract
        # ...

        vtable_ptr = ctypes.cast(archive, ctypes.POINTER(ctypes.c_void_p))[0]
        vtable = ctypes.cast(vtable_ptr, ctypes.POINTER(ctypes.c_void_p))

        # Define function types
        OpenFunc = ctypes.WINFUNCTYPE(
            ctypes.c_long,  # HRESULT
            ctypes.c_void_p,  # this
            ctypes.c_void_p,  # IInStream
            ctypes.POINTER(ctypes.c_uint64),  # maxCheckStartPosition
            ctypes.c_void_p  # IArchiveOpenCallback
        )

        CloseFunc = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)

        GetNumberOfItemsFunc = ctypes.WINFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32)
        )

        GetPropertyFunc = ctypes.WINFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_uint32,  # index
            ctypes.c_uint32,  # propID
            ctypes.POINTER(PROPVARIANT)
        )

        ExtractFunc = ctypes.WINFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),  # indices
            ctypes.c_uint32,  # numItems
            ctypes.c_int32,  # testMode
            ctypes.c_void_p  # IArchiveExtractCallback
        )

        # Get function pointers
        open_fn = ctypes.cast(vtable[3], OpenFunc)
        close_fn = ctypes.cast(vtable[4], CloseFunc)
        get_num_items_fn = ctypes.cast(vtable[5], GetNumberOfItemsFunc)
        get_property_fn = ctypes.cast(vtable[6], GetPropertyFunc)
        extract_fn = ctypes.cast(vtable[7], ExtractFunc)

        # Create input stream from file
        in_stream = InStream(self._archive_data)

        # Open archive
        max_check = ctypes.c_uint64(1 << 20)  # 1MB
        hr = open_fn(archive, ctypes.addressof(in_stream), ctypes.byref(max_check), None)
        if hr != S_OK:
            self._last_error = f"Open failed: 0x{hr:08X}"
            return {}

        try:
            # Get number of items
            num_items = ctypes.c_uint32()
            hr = get_num_items_fn(archive, ctypes.byref(num_items))
            if hr != S_OK:
                self._last_error = f"GetNumberOfItems failed: 0x{hr:08X}"
                return {}

            logger.debug(f"[RAM 7z] Archive has {num_items.value} items")

            # Get file info for each item
            file_info: Dict[int, Tuple[str, bool]] = {}

            for i in range(num_items.value):
                prop = PROPVARIANT()

                # Get path
                hr = get_property_fn(archive, i, kpidPath, ctypes.byref(prop))
                if hr == S_OK and prop.vt == VT_BSTR:
                    path_ptr = ctypes.cast(prop.data, ctypes.c_wchar_p)
                    path = path_ptr.value if path_ptr.value else f"file_{i}"
                else:
                    path = f"file_{i}"

                # Get IsDir
                prop2 = PROPVARIANT()
                hr = get_property_fn(archive, i, kpidIsDir, ctypes.byref(prop2))
                is_dir = prop2.vt == VT_BOOL and prop2.data != 0

                file_info[i] = (path, is_dir)

            # Create callback with file info
            def get_file_info(index: int) -> Tuple[str, bool]:
                return file_info.get(index, (f"file_{index}", False))

            callback = ArchiveExtractCallback(get_file_info)
            if self._password:
                callback.set_password(self._password)

            # Extract all files (indices=NULL means all)
            hr = extract_fn(
                archive,
                None,  # NULL = extract all
                0xFFFFFFFF,  # -1 = all items
                0,  # testMode = 0 (extract)
                ctypes.addressof(callback)
            )

            if hr != S_OK:
                self._last_error = f"Extract failed: 0x{hr:08X}"
                logger.error(f"[RAM 7z] Extract returned: 0x{hr:08X}")
                # Still return any files that were extracted
                return callback.get_extracted_files()

            return callback.get_extracted_files()

        finally:
            close_fn(archive)

    def get_last_error(self) -> str:
        """Get last error message."""
        return self._last_error


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_7z_from_memory(data: bytes, password: str = "") -> Dict[str, bytes]:
    """
    Extract 7z archive from memory.

    Args:
        data: Archive data as bytes
        password: Optional password for encrypted archives

    Returns:
        Dictionary mapping filename to file data
    """
    extractor = Ram7zExtractor()
    extractor.set_archive_data(data)
    if password:
        extractor.set_password(password)
    return extractor.extract_all()


def extract_7z_to_disk(
    data: bytes,
    output_dir: Path,
    password: str = "",
    progress_callback: Callable[[int, int], None] = None
) -> int:
    """
    Extract 7z archive from memory to disk.

    Args:
        data: Archive data as bytes
        output_dir: Directory to extract to
        password: Optional password
        progress_callback: Optional callback(current, total)

    Returns:
        Number of files extracted
    """
    files = extract_7z_from_memory(data, password)

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    total = len(files)

    for filename, file_data in files.items():
        dest_path = output_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle Windows long paths
        dest_str = str(dest_path.resolve())
        if len(dest_str) > 200 and not dest_str.startswith('\\\\?\\'):
            dest_str = '\\\\?\\' + dest_str

        Path(dest_str).write_bytes(file_data)
        count += 1

        if progress_callback:
            progress_callback(count, total)

    return count


# ============================================================================
# Module initialization
# ============================================================================

RAM_7Z_AVAILABLE = False
_init_error = None

try:
    if sys.platform == 'win32':
        RAM_7Z_AVAILABLE = Ram7zExtractor.is_available()
        if RAM_7Z_AVAILABLE:
            logger.info("[RAM 7z] 7z.dll loaded - fast extraction enabled")
        else:
            logger.info("[RAM 7z] 7z.dll not found or CreateObject not available")
except Exception as e:
    _init_error = str(e)
    logger.warning(f"[RAM 7z] Initialization error: {e}")
