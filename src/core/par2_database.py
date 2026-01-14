"""
PAR2 File Database Component
=============================

Provides structured storage and lookup for PAR2 file metadata, enabling:
- Parsing of main .par2 files and volume .vol*.par2 files
- Efficient lookup of file entries by filename (with obfuscation support)
- Thread-safe registration and access to PAR2 metadata
- Tracking of file verification status through the repair pipeline

PAR2 packet structure reference (from PAR2 specification):
- Packet header: 64 bytes
  - 8 bytes: Magic sequence (PAR2\x00PKT)
  - 8 bytes: Packet length (little-endian uint64)
  - 16 bytes: MD5 hash of packet body
  - 16 bytes: Recovery set ID
  - 16 bytes: Packet type signature
- Packet body: Variable length (packet_length - 64 bytes)

Supported packet types:
- Main packet: Block size, file count, file IDs
- FileDesc packet: File metadata (ID, MD5 hashes, size, filename)
- IFSC packet: Input File Slice Checksums (block-level MD5)
- RecvSlic packet: Recovery slice data (Reed-Solomon blocks)
- Creator packet: PAR2 client identification string

Usage:
    db = Par2FileDatabase()
    db.register_par2_file(Path("archive.par2"))
    metadata = db.get_metadata_for_set(set_id)
    entry = db.get_entry_by_filename("file.rar")
"""

from __future__ import annotations

import struct
import logging
import threading
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum, auto

logger = logging.getLogger(__name__)


class FileVerificationStatus(Enum):
    """
    Status of a file during PAR2 verification and repair.

    States progress through the verification pipeline:
    PENDING -> VERIFYING -> VERIFIED|DAMAGED -> REPAIRED|ERROR

    SKIPPED is used for files that don't need processing (e.g., already verified).
    """
    PENDING = auto()     # Not yet processed
    VERIFYING = auto()   # Currently being verified
    VERIFIED = auto()    # MD5 hash matches - file is intact
    DAMAGED = auto()     # MD5 hash mismatch - file needs repair
    REPAIRED = auto()    # Successfully repaired using recovery data
    SKIPPED = auto()     # Skipped (e.g., file not found, not needed)
    ERROR = auto()       # Error during verification or repair


@dataclass
class Par2FileEntry:
    """
    Metadata for a single file within a PAR2 set.

    Each file in a PAR2 set has a unique file_id (MD5 hash of file info)
    that serves as the primary identifier. The MD5 hashes are used for
    verification: md5_first_16k for quick checks, md5_full for complete
    verification.

    Attributes:
        file_id: 16-byte MD5 hash identifier (unique per file in set)
        filename: Original filename from PAR2 FileDesc packet
        size: File size in bytes
        md5_full: 16-byte MD5 hash of the complete file
        md5_first_16k: 16-byte MD5 hash of first 16KB (quick verification)
        status: Current verification status
        block_offsets: Optional list of (block_index, offset) for IFSC data
    """
    file_id: bytes
    filename: str
    size: int
    md5_full: bytes
    md5_first_16k: bytes
    status: FileVerificationStatus = FileVerificationStatus.PENDING
    block_offsets: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate field sizes."""
        if len(self.file_id) != 16:
            raise ValueError(f"file_id must be 16 bytes, got {len(self.file_id)}")
        if len(self.md5_full) != 16:
            raise ValueError(f"md5_full must be 16 bytes, got {len(self.md5_full)}")
        if len(self.md5_first_16k) != 16:
            raise ValueError(f"md5_first_16k must be 16 bytes, got {len(self.md5_first_16k)}")

    @property
    def md5_full_hex(self) -> str:
        """Return md5_full as lowercase hex string."""
        return self.md5_full.hex()

    @property
    def md5_first_16k_hex(self) -> str:
        """Return md5_first_16k as lowercase hex string."""
        return self.md5_first_16k.hex()

    @property
    def file_id_hex(self) -> str:
        """Return file_id as lowercase hex string."""
        return self.file_id.hex()


@dataclass
class Par2RecoveryBlock:
    """
    A PAR2 recovery block for Reed-Solomon repair.

    Recovery blocks contain parity data that can be used to reconstruct
    damaged or missing source blocks. Each block has an exponent that
    determines its position in the Reed-Solomon matrix.

    Attributes:
        exponent: Reed-Solomon exponent (determines block's role in repair)
        data: Raw recovery data bytes
        length: Length of recovery data
    """
    exponent: int
    data: bytes
    length: int


@dataclass
class Par2Metadata:
    """
    Complete metadata for a PAR2 recovery set.

    A recovery set is defined by its set_id and contains information about
    all protected files and available recovery blocks. Multiple .par2 files
    (main + volumes) may contribute to the same set.

    Attributes:
        files: Dict mapping file_id (bytes) to Par2FileEntry
        block_size: Size of each recovery block in bytes (slice size)
        total_blocks: Total number of source data blocks across all files
        recovery_count: Number of recovery blocks available
        set_id: 16-byte recovery set identifier (MD5)
        recovery_blocks: List of recovery blocks from volume files
        creator: PAR2 client that created this set (from Creator packet)
        source_files: Set of par2 file paths that contributed to this metadata
    """
    files: Dict[bytes, Par2FileEntry] = field(default_factory=dict)
    block_size: int = 0
    total_blocks: int = 0
    recovery_count: int = 0
    set_id: bytes = b''
    recovery_blocks: List[Par2RecoveryBlock] = field(default_factory=list)
    creator: str = ''
    source_files: Set[Path] = field(default_factory=set)

    def get_file_by_name(self, filename: str) -> Optional[Par2FileEntry]:
        """
        Find a file entry by filename.

        Args:
            filename: Filename to search for (case-sensitive)

        Returns:
            Par2FileEntry if found, None otherwise
        """
        for entry in self.files.values():
            if entry.filename == filename:
                return entry
        return None

    def get_file_by_id(self, file_id: bytes) -> Optional[Par2FileEntry]:
        """
        Find a file entry by file ID.

        Args:
            file_id: 16-byte file identifier

        Returns:
            Par2FileEntry if found, None otherwise
        """
        return self.files.get(file_id)

    @property
    def file_count(self) -> int:
        """Number of files in this recovery set."""
        return len(self.files)

    @property
    def total_size(self) -> int:
        """Total size of all protected files in bytes."""
        return sum(f.size for f in self.files.values())

    @property
    def can_repair(self) -> bool:
        """Whether recovery blocks are available for repair."""
        return self.recovery_count > 0


class Par2FileDatabase:
    """
    Thread-safe database for PAR2 file metadata.

    Manages parsing, storage, and lookup of PAR2 metadata across multiple
    recovery sets. Supports:
    - Parsing main .par2 files for file descriptions
    - Parsing volume .vol*.par2 files for recovery blocks
    - Filename lookup with normalization for obfuscation support
    - Thread-safe access for concurrent verification

    Example:
        db = Par2FileDatabase()

        # Register PAR2 files
        db.register_par2_file(Path("download/archive.par2"))
        db.register_par2_file(Path("download/archive.vol00+01.par2"))

        # Look up files
        entry = db.get_entry_by_filename("archive.part01.rar")
        if entry:
            print(f"Expected MD5: {entry.md5_full_hex}")
    """

    # PAR2 packet signatures (16 bytes each)
    PACKET_HEADER = b'PAR2\x00PKT'
    PACKET_MAIN = b'PAR 2.0\x00Main\x00\x00\x00\x00'
    PACKET_FILE_DESC = b'PAR 2.0\x00FileDesc'
    PACKET_IFSC = b'PAR 2.0\x00IFSC\x00\x00\x00\x00'
    PACKET_RECOVERY = b'PAR 2.0\x00RecvSlic'
    PACKET_CREATOR = b'PAR 2.0\x00Creator\x00'

    def __init__(self) -> None:
        """Initialize the PAR2 file database."""
        # Map set_id -> Par2Metadata
        self._sets: Dict[bytes, Par2Metadata] = {}

        # Map normalized_filename -> (set_id, file_id) for fast lookup
        self._filename_index: Dict[str, Tuple[bytes, bytes]] = {}

        # Thread safety
        self._lock = threading.RLock()

        logger.debug("Par2FileDatabase initialized")

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        """
        Normalize a filename for matching.

        Handles common Usenet variations:
        - Case differences
        - Underscores vs spaces
        - Leading/trailing whitespace

        Args:
            filename: Original filename

        Returns:
            Normalized filename for comparison
        """
        normalized = filename.lower().strip()
        # Normalize underscores to spaces (NZB indexers often convert)
        normalized = normalized.replace('_', ' ')
        return normalized

    def _update_filename_index(self, metadata: Par2Metadata) -> None:
        """
        Update the filename index with entries from a metadata set.

        Args:
            metadata: Par2Metadata to index
        """
        for file_id, entry in metadata.files.items():
            normalized = self._normalize_filename(entry.filename)
            self._filename_index[normalized] = (metadata.set_id, file_id)
            logger.debug(f"Indexed file: {entry.filename} -> {normalized}")

    def parse_par2_file(self, par2_path: Path) -> Par2Metadata:
        """
        Parse a PAR2 file and return its metadata.

        Parses both main .par2 files (containing file descriptions) and
        volume .vol*.par2 files (containing recovery blocks). All files
        in the same recovery set share the same set_id.

        Args:
            par2_path: Path to the .par2 file

        Returns:
            Par2Metadata containing parsed file entries and recovery blocks

        Raises:
            FileNotFoundError: If par2_path doesn't exist
            ValueError: If file is not a valid PAR2 file
        """
        par2_path = Path(par2_path)

        if not par2_path.exists():
            raise FileNotFoundError(f"PAR2 file not found: {par2_path}")

        logger.info(f"Parsing PAR2 file: {par2_path}")

        with open(par2_path, 'rb') as f:
            data = f.read()

        return self._parse_par2_data(data, par2_path)

    def _parse_par2_data(self, data: bytes, source_path: Optional[Path] = None) -> Par2Metadata:
        """
        Parse PAR2 data from bytes.

        Args:
            data: Raw PAR2 file content
            source_path: Optional path for logging and tracking

        Returns:
            Par2Metadata containing parsed data

        Raises:
            ValueError: If data is not a valid PAR2 file
        """
        metadata = Par2Metadata()
        if source_path:
            metadata.source_files.add(source_path)

        pos = 0
        data_len = len(data)
        packets_parsed = 0

        # Scan for packets
        while pos < data_len - 8:
            # Look for packet header magic
            if data[pos:pos + 8] != self.PACKET_HEADER:
                pos += 1
                continue

            # Validate we have enough data for header (64 bytes)
            if pos + 64 > data_len:
                logger.warning(f"Truncated packet header at position {pos}")
                break

            # Parse packet header
            # 8 bytes: magic (already validated)
            # 8 bytes: length (little-endian uint64, includes header)
            # 16 bytes: packet hash (MD5)
            # 16 bytes: recovery set ID
            # 16 bytes: packet type signature

            packet_length = struct.unpack('<Q', data[pos + 8:pos + 16])[0]

            # Validate packet length
            if packet_length < 64:
                logger.warning(f"Invalid packet length {packet_length} at position {pos}")
                pos += 1
                continue

            if pos + packet_length > data_len:
                logger.warning(f"Truncated packet at position {pos}, need {packet_length} bytes")
                break

            # Extract set ID (first time we see it)
            set_id = data[pos + 32:pos + 48]
            if not metadata.set_id:
                metadata.set_id = set_id
                logger.debug(f"Recovery set ID: {set_id.hex()}")
            elif metadata.set_id != set_id:
                logger.warning(f"Set ID mismatch: expected {metadata.set_id.hex()}, got {set_id.hex()}")

            # Extract packet type and body
            packet_type = data[pos + 48:pos + 64]
            packet_body = data[pos + 64:pos + packet_length]

            # Parse based on packet type
            if packet_type == self.PACKET_MAIN:
                self._parse_main_packet(packet_body, metadata)
                packets_parsed += 1
            elif packet_type == self.PACKET_FILE_DESC:
                self._parse_file_desc_packet(packet_body, metadata)
                packets_parsed += 1
            elif packet_type == self.PACKET_IFSC:
                self._parse_ifsc_packet(packet_body, metadata)
                packets_parsed += 1
            elif packet_type == self.PACKET_RECOVERY:
                self._parse_recovery_packet(packet_body, packet_length - 64, metadata)
                packets_parsed += 1
            elif packet_type == self.PACKET_CREATOR:
                self._parse_creator_packet(packet_body, metadata)
                packets_parsed += 1
            else:
                logger.debug(f"Unknown packet type: {packet_type.hex()}")

            # Move to next packet
            pos += packet_length

        if packets_parsed == 0:
            raise ValueError("No valid PAR2 packets found in file")

        logger.info(
            f"PAR2 parsed: {len(metadata.files)} files, "
            f"{metadata.recovery_count} recovery blocks, "
            f"block size {metadata.block_size}"
        )

        return metadata

    def _parse_main_packet(self, data: bytes, metadata: Par2Metadata) -> None:
        """
        Parse a Main packet to extract block size and file count.

        Main packet body format:
        - 8 bytes: Slice size (block size)
        - 4 bytes: Number of files (recovery set count)
        - Remaining: File IDs (16 bytes each)

        Args:
            data: Packet body bytes
            metadata: Par2Metadata to update
        """
        if len(data) < 8:
            logger.warning("Main packet too short")
            return

        # Block size (slice size)
        metadata.block_size = struct.unpack('<Q', data[0:8])[0]
        logger.debug(f"Block size: {metadata.block_size} bytes")

        # Count file IDs in the packet (each is 16 bytes)
        if len(data) >= 12:
            # Remaining bytes after slice size contain file IDs
            file_id_data = data[8:]
            num_files = len(file_id_data) // 16
            logger.debug(f"Main packet references {num_files} files")

    def _parse_file_desc_packet(self, data: bytes, metadata: Par2Metadata) -> None:
        """
        Parse a FileDesc packet to extract file metadata.

        FileDesc packet body format:
        - 16 bytes: File ID (MD5 of file info)
        - 16 bytes: MD5 hash of complete file
        - 16 bytes: MD5 hash of first 16KB
        - 8 bytes: File size
        - Remaining: Filename (null-terminated, padded to 4 bytes)

        Args:
            data: Packet body bytes
            metadata: Par2Metadata to update
        """
        if len(data) < 56:
            logger.warning(f"FileDesc packet too short: {len(data)} bytes")
            return

        # Extract fields
        file_id = data[0:16]
        md5_full = data[16:32]
        md5_first_16k = data[32:48]
        file_size = struct.unpack('<Q', data[48:56])[0]

        # Extract filename (null-terminated UTF-8, padded to 4-byte boundary)
        filename_bytes = data[56:]
        null_pos = filename_bytes.find(b'\x00')
        if null_pos >= 0:
            filename = filename_bytes[:null_pos].decode('utf-8', errors='replace')
        else:
            filename = filename_bytes.decode('utf-8', errors='replace')

        # Create entry
        entry = Par2FileEntry(
            file_id=file_id,
            filename=filename,
            size=file_size,
            md5_full=md5_full,
            md5_first_16k=md5_first_16k,
        )

        metadata.files[file_id] = entry
        logger.debug(f"File: {filename} ({file_size} bytes, MD5: {md5_full.hex()[:16]}...)")

    def _parse_ifsc_packet(self, data: bytes, metadata: Par2Metadata) -> None:
        """
        Parse an IFSC (Input File Slice Checksum) packet.

        IFSC packet body format:
        - 16 bytes: File ID
        - Remaining: Pairs of (MD5 hash, CRC32) for each block

        This information enables block-level verification and repair.

        Args:
            data: Packet body bytes
            metadata: Par2Metadata to update
        """
        if len(data) < 16:
            logger.warning("IFSC packet too short")
            return

        file_id = data[0:16]

        # Each slice checksum is 20 bytes (16 bytes MD5 + 4 bytes CRC32)
        checksum_data = data[16:]
        num_blocks = len(checksum_data) // 20

        if file_id in metadata.files:
            # Store block count for total calculation
            entry = metadata.files[file_id]
            metadata.total_blocks += num_blocks
            logger.debug(f"IFSC: {entry.filename} has {num_blocks} blocks")

    def _parse_recovery_packet(self, data: bytes, length: int, metadata: Par2Metadata) -> None:
        """
        Parse a RecvSlic (Recovery Slice) packet.

        RecvSlic packet body format:
        - 4 bytes: Exponent (Reed-Solomon matrix position)
        - Remaining: Recovery data (block_size bytes)

        Args:
            data: Packet body bytes
            length: Total length of packet body
            metadata: Par2Metadata to update
        """
        if len(data) < 4:
            logger.warning("Recovery packet too short")
            return

        exponent = struct.unpack('<I', data[0:4])[0]
        recovery_data = data[4:]

        block = Par2RecoveryBlock(
            exponent=exponent,
            data=recovery_data,
            length=len(recovery_data)
        )

        metadata.recovery_blocks.append(block)
        metadata.recovery_count += 1

        logger.debug(f"Recovery block: exponent={exponent}, size={len(recovery_data)}")

    def _parse_creator_packet(self, data: bytes, metadata: Par2Metadata) -> None:
        """
        Parse a Creator packet to extract PAR2 client info.

        Creator packet body format:
        - Variable: Null-terminated UTF-8 string identifying the PAR2 client

        Args:
            data: Packet body bytes
            metadata: Par2Metadata to update
        """
        null_pos = data.find(b'\x00')
        if null_pos >= 0:
            creator = data[:null_pos].decode('utf-8', errors='replace')
        else:
            creator = data.decode('utf-8', errors='replace')

        metadata.creator = creator
        logger.debug(f"Creator: {creator}")

    def register_par2_file(self, par2_path: Path) -> Optional[Par2Metadata]:
        """
        Parse and register a PAR2 file in the database.

        If the recovery set already exists, merges new file entries and
        recovery blocks into the existing metadata.

        Args:
            par2_path: Path to the .par2 file

        Returns:
            Par2Metadata for the recovery set, or None on error
        """
        try:
            new_metadata = self.parse_par2_file(par2_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to parse PAR2 file {par2_path}: {e}")
            return None

        with self._lock:
            set_id = new_metadata.set_id

            if set_id in self._sets:
                # Merge with existing metadata
                existing = self._sets[set_id]
                existing.source_files.add(par2_path)

                # Merge files (shouldn't conflict, but update if missing)
                for file_id, entry in new_metadata.files.items():
                    if file_id not in existing.files:
                        existing.files[file_id] = entry
                        logger.debug(f"Added file to set: {entry.filename}")

                # Add recovery blocks
                existing.recovery_blocks.extend(new_metadata.recovery_blocks)
                existing.recovery_count += len(new_metadata.recovery_blocks)

                # Update block size if not set
                if existing.block_size == 0 and new_metadata.block_size > 0:
                    existing.block_size = new_metadata.block_size

                # Update creator if not set
                if not existing.creator and new_metadata.creator:
                    existing.creator = new_metadata.creator

                # Update filename index
                self._update_filename_index(new_metadata)

                logger.info(
                    f"Merged PAR2 {par2_path.name}: "
                    f"set now has {len(existing.files)} files, "
                    f"{existing.recovery_count} recovery blocks"
                )
                return existing
            else:
                # New recovery set
                self._sets[set_id] = new_metadata
                self._update_filename_index(new_metadata)

                logger.info(
                    f"Registered new PAR2 set {set_id.hex()[:8]}...: "
                    f"{len(new_metadata.files)} files, "
                    f"{new_metadata.recovery_count} recovery blocks"
                )
                return new_metadata

    def get_entry_by_filename(
        self,
        filename: str,
        actual_filename: Optional[str] = None
    ) -> Optional[Par2FileEntry]:
        """
        Look up a file entry by filename with obfuscation support.

        Supports matching obfuscated filenames (e.g., random hashes) to
        PAR2 entries by trying multiple matching strategies:
        1. Exact match on filename
        2. Normalized match (case-insensitive, underscore/space normalization)
        3. Match on actual_filename if provided (for renamed files)

        Args:
            filename: Filename to look up (may be obfuscated)
            actual_filename: Optional actual/original filename for matching

        Returns:
            Par2FileEntry if found, None otherwise
        """
        with self._lock:
            # Strategy 1: Exact match via filename index
            normalized = self._normalize_filename(filename)
            if normalized in self._filename_index:
                set_id, file_id = self._filename_index[normalized]
                if set_id in self._sets:
                    entry = self._sets[set_id].files.get(file_id)
                    if entry:
                        logger.debug(f"Found exact match: {filename} -> {entry.filename}")
                        return entry

            # Strategy 2: Try actual_filename if provided
            if actual_filename:
                normalized_actual = self._normalize_filename(actual_filename)
                if normalized_actual in self._filename_index:
                    set_id, file_id = self._filename_index[normalized_actual]
                    if set_id in self._sets:
                        entry = self._sets[set_id].files.get(file_id)
                        if entry:
                            logger.debug(
                                f"Found via actual filename: {actual_filename} -> {entry.filename}"
                            )
                            return entry

            # Strategy 3: Scan all entries for partial matches
            # (handles obfuscated filenames that partially match)
            filename_lower = filename.lower()
            for metadata in self._sets.values():
                for entry in metadata.files.values():
                    entry_lower = entry.filename.lower()

                    # Check if one contains the other (for obfuscated prefixes/suffixes)
                    if filename_lower in entry_lower or entry_lower in filename_lower:
                        logger.debug(f"Found partial match: {filename} ~ {entry.filename}")
                        return entry

            logger.debug(f"No match found for filename: {filename}")
            return None

    def get_entry_by_file_id(self, file_id: bytes) -> Optional[Par2FileEntry]:
        """
        Look up a file entry by its 16-byte file ID.

        Args:
            file_id: 16-byte file identifier

        Returns:
            Par2FileEntry if found, None otherwise
        """
        with self._lock:
            for metadata in self._sets.values():
                if file_id in metadata.files:
                    return metadata.files[file_id]
            return None

    def get_metadata_for_set(self, set_id: bytes) -> Optional[Par2Metadata]:
        """
        Get metadata for a specific recovery set.

        Args:
            set_id: 16-byte recovery set identifier

        Returns:
            Par2Metadata if found, None otherwise
        """
        with self._lock:
            return self._sets.get(set_id)

    def get_all_sets(self) -> List[Par2Metadata]:
        """
        Get all registered recovery sets.

        Returns:
            List of Par2Metadata for all registered sets
        """
        with self._lock:
            return list(self._sets.values())

    def get_files_by_status(self, status: FileVerificationStatus) -> List[Par2FileEntry]:
        """
        Get all file entries with a specific status.

        Args:
            status: FileVerificationStatus to filter by

        Returns:
            List of Par2FileEntry with the specified status
        """
        with self._lock:
            results = []
            for metadata in self._sets.values():
                for entry in metadata.files.values():
                    if entry.status == status:
                        results.append(entry)
            return results

    def update_file_status(
        self,
        file_id: bytes,
        status: FileVerificationStatus
    ) -> bool:
        """
        Update the verification status of a file.

        Args:
            file_id: 16-byte file identifier
            status: New FileVerificationStatus

        Returns:
            True if file was found and updated, False otherwise
        """
        with self._lock:
            for metadata in self._sets.values():
                if file_id in metadata.files:
                    old_status = metadata.files[file_id].status
                    metadata.files[file_id].status = status
                    logger.debug(
                        f"File {file_id.hex()[:8]}... status: "
                        f"{old_status.name} -> {status.name}"
                    )
                    return True
            return False

    def clear(self) -> None:
        """Clear all registered PAR2 metadata."""
        with self._lock:
            self._sets.clear()
            self._filename_index.clear()
            logger.debug("Par2FileDatabase cleared")

    def __len__(self) -> int:
        """Return total number of registered recovery sets."""
        with self._lock:
            return len(self._sets)

    def __repr__(self) -> str:
        """Return string representation."""
        with self._lock:
            total_files = sum(len(m.files) for m in self._sets.values())
            total_recovery = sum(m.recovery_count for m in self._sets.values())
            return (
                f"Par2FileDatabase("
                f"sets={len(self._sets)}, "
                f"files={total_files}, "
                f"recovery_blocks={total_recovery})"
            )
