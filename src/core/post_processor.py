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


@dataclass
class NZBMetadata:
    """Metadata extracted from NZB file."""
    title: str = ""
    password: str = ""
    category: str = ""
    group: str = ""


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

            logger.info(f"NZB Metadata: title={metadata.title}, password={'***' if metadata.password else 'None'}")

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
        output_dir: Optional[Path] = None
    ) -> Tuple[bool, int, str]:
        """
        Extract archive using 7-Zip.

        Args:
            archive: Path to archive file
            password: Archive password (optional)
            output_dir: Output directory (default: self.extract_dir)

        Returns:
            Tuple of (success, files_extracted, message)
        """
        if not self.sevenzip_path:
            return False, 0, "7-Zip not available"

        self._current_status = PostProcessStatus.EXTRACTING
        self._report_progress(f"Extracting: {archive.name}", 0)

        out_dir = output_dir or self.extract_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build 7z command
            cmd = [
                self.sevenzip_path,
                'x',  # Extract with full paths
                '-y',  # Yes to all prompts
                '-bb0',  # Less output
                '-bd',  # No progress indicator
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

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(archive.parent),
                timeout=7200,  # 2 hours max
                creationflags=SUBPROCESS_FLAGS  # Hide console window
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                # Count extracted files
                files_count = sum(1 for _ in out_dir.rglob('*') if _.is_file())
                logger.info(f"Extraction OK: {files_count} files ({elapsed:.1f}s)")
                return True, files_count, f"Extracted {files_count} files ({elapsed:.1f}s)"
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                # Check for password error
                if 'Wrong password' in error_msg or 'password' in error_msg.lower():
                    return False, 0, "Wrong password"
                logger.error(f"7z failed: {error_msg}")
                return False, 0, f"Failed: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            return False, 0, "Extraction timed out"
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return False, 0, f"Error: {e}"

    def cleanup_archives(self, directory: Optional[Path] = None) -> int:
        """Remove archive files and PAR2 files after successful extraction."""
        search_dir = directory or self.download_dir
        removed = 0

        try:
            patterns = ['*.rar', '*.r[0-9][0-9]', '*.zip', '*.7z', '*.par2',
                       '*.PAR2', '*.part*.rar']

            for pattern in patterns:
                for file in search_dir.glob(pattern):
                    try:
                        file.unlink()
                        removed += 1
                    except:
                        pass

            logger.info(f"Cleaned up {removed} files")

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

        return removed

    # Media file extensions to move to final destination
    MEDIA_EXTENSIONS = {'.mkv', '.avi', '.mp4', '.m4v', '.mov', '.wmv', '.flv',
                        '.webm', '.mpg', '.mpeg', '.m2ts', '.ts', '.vob',
                        '.iso', '.img', '.nfo', '.srt', '.sub', '.idx', '.ass'}

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
        Used when there are no archives to extract.

        Args:
            src_dir: Source directory containing downloaded files
            release_name: Name for destination subfolder

        Returns:
            Number of files moved
        """
        # Create destination subfolder
        dest_folder = self.extract_dir / (release_name or "media")
        dest_folder.mkdir(parents=True, exist_ok=True)

        moved = 0
        for file in src_dir.iterdir():
            if not file.is_file():
                continue

            # Skip PAR2 and archive files
            suffix = file.suffix.lower()
            if suffix == '.par2' or self._is_archive(file):
                continue

            # Move media files
            if self._is_media_file(file):
                target = dest_folder / file.name

                # Handle conflicts
                if target.exists():
                    base, ext = target.stem, target.suffix
                    counter = 1
                    while target.exists():
                        target = dest_folder / f"{base}_{counter}{ext}"
                        counter += 1

                move_result = robust_move_file(file, target)
                if move_result.success or move_result.copied:
                    logger.info(f"Moved: {file.name} → {dest_folder.name}/")
                    moved += 1

        return moved

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
        password: Optional[str] = None
    ) -> Tuple[Path, bool]:
        """
        Extract ALL archives to a single temp directory.
        This allows multi-part RARs split across ZIPs to be combined.

        Returns:
            (temp_dir, success)
        """
        temp_dir = self.download_dir / "_temp_extract_combined"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for archive in archives:
            self._report_progress(f"Extracting: {archive.name}", 50)
            success, _, msg = self.extract_archive(archive, password, temp_dir)
            if not success:
                logger.warning(f"Failed to extract {archive.name}: {msg}")

        return temp_dir, True

    def smart_extract(
        self,
        archives: List[Path],
        password: Optional[str] = None,
        release_name: Optional[str] = None
    ) -> Tuple[bool, int, str]:
        """
        Smart extraction handling nested archives (ZIP→RAR→content).

        Strategy:
        1. Extract ALL outer archives to single temp folder
        2. Find nested archives (RAR parts now combined)
        3. Extract nested to final destination subfolder
        4. Cleanup

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

        try:
            # Phase 1: Extract all outer archives to combined temp
            logger.info(f"Phase 1: Extracting {len(archives)} archives to temp...")
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
                # No nested archives - move temp content to destination
                logger.info("No nested archives found, moving content directly...")
                self._report_progress("Moving files...", 80)

                files_moved = 0
                for item in temp_dir.iterdir():
                    # Skip archive files themselves if they ended up in temp
                    if self._is_archive(item):
                        continue

                    target = dest_subfolder / item.name

                    # Handle conflicts
                    if target.exists():
                        if item.is_file():
                            base, ext = target.stem, target.suffix
                            counter = 1
                            while target.exists():
                                target = dest_subfolder / f"{base}_{counter}{ext}"
                                counter += 1

                    # Use robust move with retry for antivirus locks
                    if item.is_dir():
                        # Move all files in directory
                        for sub in item.rglob('*'):
                            if sub.is_file():
                                rel = sub.relative_to(item)
                                sub_target = target / rel
                                move_result = robust_move_file(sub, sub_target)
                                if move_result.success or move_result.copied:
                                    files_moved += 1
                                if move_result.av_blocked:
                                    self._av_blocked_files.add(sub.name)
                        # Cleanup empty source dir
                        shutil.rmtree(str(item), ignore_errors=True)
                    else:
                        move_result = robust_move_file(item, target)
                        if move_result.success or move_result.copied:
                            files_moved += 1
                        if move_result.av_blocked:
                            self._av_blocked_files.add(item.name)

                # Cleanup temp
                shutil.rmtree(temp_dir, ignore_errors=True)

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
        password: Optional[str] = None
    ) -> PostProcessResult:
        """
        Full post-processing pipeline.

        1. Parse NZB for password (if provided)
        2. Find and verify PAR2 files
        3. Find and extract archives
        4. Optional cleanup

        Args:
            nzb_path: Path to NZB file (for password extraction)
            source_dir: Directory containing downloaded files (default: download_dir)
            password: Override password (if not from NZB)

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

            # Step 2: PAR2 verification
            par2_files = self.find_par2_files(src_dir)
            if par2_files:
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

                # Use smart_extract: extracts ALL to temp, finds nested RARs, extracts to subfolder
                success, total_files, msg = self.smart_extract(
                    archives, password, release_name
                )

                if success:
                    result.files_extracted = total_files
                    result.extract_path = self.extract_dir / release_name
                    self._report_progress(f"Extracted {total_files} files", 90)
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

            # Step 4: Cleanup (optional)
            if self.cleanup_after_extract and result.files_extracted > 0:
                self._report_progress("Cleaning up...", 95)
                self.cleanup_archives(src_dir)

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
