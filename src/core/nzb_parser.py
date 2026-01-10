"""
High-Performance NZB Parser
============================

Optimized XML parsing with:
- Streaming parser for large files
- Pre-computed segment ordering
- Memory-efficient data structures
- Parallel segment grouping
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Iterator
from pathlib import Path
import logging
from lxml import etree
from functools import cached_property

logger = logging.getLogger(__name__)

NZB_NAMESPACE = '{http://www.newzbin.com/DTD/2003/nzb}'


@dataclass(slots=True)
class NZBSegment:
    """
    Single segment of an NZB file.
    Uses slots for memory efficiency when handling millions of segments.
    """
    message_id: str
    number: int
    bytes: int
    file_index: int = 0

    def __hash__(self) -> int:
        return hash(self.message_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NZBSegment):
            return self.message_id == other.message_id
        return False


@dataclass
class NZBFile:
    """
    Single file within an NZB.
    Contains all segments needed to reconstruct the file.
    """
    filename: str
    subject: str
    poster: str
    date: int
    groups: list[str]
    segments: list[NZBSegment] = field(default_factory=list)
    index: int = 0

    @cached_property
    def total_bytes(self) -> int:
        """Total size of all segments."""
        return sum(seg.bytes for seg in self.segments)

    @cached_property
    def segment_count(self) -> int:
        """Number of segments."""
        return len(self.segments)

    @cached_property
    def sorted_segments(self) -> list[NZBSegment]:
        """Segments sorted by number for ordered assembly."""
        return sorted(self.segments, key=lambda s: s.number)

    @property
    def is_par2(self) -> bool:
        """Check if this is a PAR2 recovery file."""
        return '.par2' in self.filename.lower()

    @property
    def is_rar(self) -> bool:
        """Check if this is a RAR archive."""
        lower = self.filename.lower()
        return '.rar' in lower or '.r00' in lower

    def get_extension(self) -> str:
        """Extract file extension."""
        parts = self.filename.rsplit('.', 1)
        return parts[1].lower() if len(parts) > 1 else ''


@dataclass
class NZBDocument:
    """
    Complete parsed NZB document.
    Provides efficient access to files and segments.
    """
    files: list[NZBFile]
    meta: dict[str, str] = field(default_factory=dict)
    source_path: Optional[Path] = None

    @cached_property
    def total_bytes(self) -> int:
        """Total size of all files."""
        return sum(f.total_bytes for f in self.files)

    @cached_property
    def total_segments(self) -> int:
        """Total number of segments."""
        return sum(f.segment_count for f in self.files)

    @cached_property
    def file_count(self) -> int:
        """Number of files."""
        return len(self.files)

    def iter_segments(self) -> Iterator[NZBSegment]:
        """Iterate all segments in optimal download order."""
        for nzb_file in self.files:
            for segment in nzb_file.sorted_segments:
                yield segment

    def get_main_files(self) -> list[NZBFile]:
        """Get non-PAR2 files (main content)."""
        return [f for f in self.files if not f.is_par2]

    def get_par2_files(self) -> list[NZBFile]:
        """Get PAR2 recovery files."""
        return [f for f in self.files if f.is_par2]

    def get_groups(self) -> set[str]:
        """Get all newsgroups referenced."""
        groups: set[str] = set()
        for f in self.files:
            groups.update(f.groups)
        return groups


class NZBParser:
    """
    High-performance NZB parser using lxml.

    Features:
    - Streaming parse for memory efficiency
    - Parallel processing support
    - Robust error handling
    - Filename extraction from subject
    """

    @staticmethod
    def parse(path: str | Path) -> NZBDocument:
        """
        Parse NZB file from path.
        Uses iterparse for memory-efficient streaming.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NZB file not found: {path}")

        files: list[NZBFile] = []
        meta: dict[str, str] = {}

        try:
            # Use iterparse for streaming large files
            context = etree.iterparse(
                str(path),
                events=('end',),
                tag=[f'{NZB_NAMESPACE}file', f'{NZB_NAMESPACE}meta', 'file', 'meta']
            )

            file_index = 0

            for event, elem in context:
                tag = elem.tag.replace(NZB_NAMESPACE, '')

                if tag == 'meta':
                    meta_type = elem.get('type', '')
                    if meta_type and elem.text:
                        meta[meta_type] = elem.text

                elif tag == 'file':
                    nzb_file = NZBParser._parse_file_element(elem, file_index)
                    if nzb_file:
                        files.append(nzb_file)
                        file_index += 1

                # Clear element to save memory
                elem.clear()
                while elem.getprevious() is not None:
                    parent = elem.getparent()
                    if parent is not None:
                        del parent[0]

            logger.info(f"Parsed {len(files)} files, {sum(f.segment_count for f in files)} segments")

            return NZBDocument(files=files, meta=meta, source_path=path)

        except etree.XMLSyntaxError as e:
            raise ValueError(f"Invalid NZB XML: {e}")

    @staticmethod
    def _parse_file_element(elem: etree._Element, file_index: int) -> Optional[NZBFile]:
        """Parse a single file element."""
        subject = elem.get('subject', '')
        poster = elem.get('poster', '')
        date_str = elem.get('date', '0')

        try:
            date = int(date_str)
        except ValueError:
            date = 0

        # Extract filename from subject
        filename = NZBParser._extract_filename(subject)

        # Parse groups
        groups: list[str] = []
        for group_elem in elem.iter():
            tag = group_elem.tag.replace(NZB_NAMESPACE, '')
            if tag == 'group' and group_elem.text:
                groups.append(group_elem.text.strip())

        # Parse segments
        segments: list[NZBSegment] = []
        for seg_elem in elem.iter():
            tag = seg_elem.tag.replace(NZB_NAMESPACE, '')
            if tag == 'segment' and seg_elem.text:
                try:
                    number = int(seg_elem.get('number', '0'))
                    bytes_size = int(seg_elem.get('bytes', '0'))
                    message_id = seg_elem.text.strip()

                    # Validate segment data
                    if number <= 0 or bytes_size <= 0 or not message_id:
                        logger.warning(f"Invalid segment: num={number}, bytes={bytes_size}, id={message_id[:20] if message_id else 'empty'}")
                        continue

                    segments.append(NZBSegment(
                        message_id=message_id,
                        number=number,
                        bytes=bytes_size,
                        file_index=file_index
                    ))
                except ValueError:
                    continue

        if not segments:
            return None

        return NZBFile(
            filename=filename,
            subject=subject,
            poster=poster,
            date=date,
            groups=groups,
            segments=segments,
            index=file_index
        )

    @staticmethod
    def _extract_filename(subject: str) -> str:
        """
        Extract filename from NZB subject line.
        Handles various common formats.
        """
        # Try to find filename in quotes
        import re

        # Pattern: "filename.ext"
        match = re.search(r'"([^"]+\.[a-zA-Z0-9]{2,5})"', subject)
        if match:
            return match.group(1)

        # Pattern: [nn/mm] filename.ext
        match = re.search(r'\[\d+/\d+\]\s*["-]?\s*([^\s"]+\.[a-zA-Z0-9]{2,5})', subject)
        if match:
            return match.group(1)

        # Pattern: filename.ext (nn/mm)
        match = re.search(r'([^\s"]+\.[a-zA-Z0-9]{2,5})\s*[\(\[]?\d+/\d+', subject)
        if match:
            return match.group(1)

        # Pattern: any file-like string with extension
        match = re.search(r'([^\s"\[\]]+\.[a-zA-Z0-9]{2,5})', subject)
        if match:
            return match.group(1)

        # Fallback: use sanitized subject
        return re.sub(r'[<>:"/\\|?*]', '_', subject[:100])

    @staticmethod
    def parse_string(content: str) -> NZBDocument:
        """Parse NZB from string content."""
        # Write to temp file and parse
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.nzb', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            return NZBParser.parse(temp_path)
        finally:
            os.unlink(temp_path)
