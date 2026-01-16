"""
DLER Adaptive Extraction System
================================

Intelligent, proactive extraction system that:
1. Analyzes NZB content BEFORE download to classify release type
2. Creates optimal extraction strategy based on detected pattern
3. Executes adaptively, reacting to actual content in real-time
4. Keeps all operations in RAM until final flush

Supported Release Types:
- DIRECT_MEDIA: MKV/MP4/ISO without archives -> direct flush
- MULTIPART_RAR: .part001.rar series -> UnRAR to disk
- ZIP_CONTAINING_RAR: ZIP->RAR->content (scene standard) -> full RAM pipeline
- NESTED_RAR: RAR->RAR->content -> cascaded extraction
- OBFUSCATED: Random names -> magic byte detection + adaptive strategy
- SEVENZ: 7z archives -> 7z extraction

Copyright (c) 2025 DLER Project
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.nzb_parser import NZBFile, NZBDocument
    from src.core.ram_processor import RamBuffer, RamFile

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ReleaseType(Enum):
    """Classification of release archive organization."""
    DIRECT_MEDIA = "direct_media"           # MKV/MP4/ISO without archives
    SINGLE_RAR = "single_rar"               # Single RAR file
    MULTIPART_RAR = "multipart_rar"         # .part001.rar series
    ZIP_CONTAINING_RAR = "zip_rar"          # ZIP -> RAR -> content (scene)
    NESTED_RAR = "nested_rar"               # RAR -> RAR -> content
    OBFUSCATED = "obfuscated"               # Random names, magic detection
    SEVENZ = "sevenz"                       # 7z archives
    MIXED = "mixed"                         # Multiple archive types
    UNKNOWN = "unknown"                     # Cannot determine


class ExtractionTool(Enum):
    """Available extraction tools."""
    UNRAR_DLL = "unrar"                     # RAM RAR via UnRAR DLL
    ZIPFILE = "zipfile"                     # Python zipfile (RAM)
    SEVENZIP_CLI = "7z"                     # 7z.exe fallback (disk)
    DIRECT_FLUSH = "flush"                  # No extraction needed


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ClassificationResult:
    """Result of NZB content classification."""
    release_type: ReleaseType
    confidence: float                       # 0.0-1.0
    detected_patterns: List[str] = field(default_factory=list)
    archive_count: int = 0
    media_count: int = 0
    par2_count: int = 0
    estimated_nesting_depth: int = 1
    is_obfuscated: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractionStage:
    """Single stage in extraction pipeline."""
    stage_number: int
    tool: ExtractionTool
    input_pattern: str                      # "*.rar", "*.zip", "*" (all)
    output_type: str                        # "rar", "7z", "content", "media"
    keep_in_ram: bool = True                # Keep output in RAM for next stage
    description: str = ""


@dataclass
class ExtractionPlan:
    """Complete extraction plan for a release."""
    release_type: ReleaseType
    stages: List[ExtractionStage] = field(default_factory=list)
    expects_nested: bool = False
    max_nesting_depth: int = 3
    estimated_memory_mb: int = 0


@dataclass
class StageResult:
    """Result of executing a single extraction stage."""
    files_extracted: int = 0
    bytes_extracted: int = 0
    found_types: Set[str] = field(default_factory=set)
    needs_adaptation: bool = False
    output_files: Dict[str, bytes] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Final result of adaptive extraction."""
    success: bool
    files_extracted: int
    bytes_written: int = 0
    stages_executed: int = 0
    adaptations_made: int = 0
    errors: List[str] = field(default_factory=list)


# =============================================================================
# RELEASE CLASSIFIER
# =============================================================================

class ReleaseClassifier:
    """
    Analyzes NZB content BEFORE download to determine archive organization.

    Uses filename patterns, extension analysis, and heuristics to classify
    the release type with confidence scoring.
    """

    # Extension categories
    MEDIA_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.iso', '.m2ts', '.ts', '.wmv', '.mov', '.webm'}
    RAR_EXTENSIONS = {'.rar'}
    RAR_SPLIT_EXTENSIONS = {'.r00', '.r01', '.r02'}  # Sample of old-style
    ZIP_EXTENSIONS = {'.zip'}
    SEVENZ_EXTENSIONS = {'.7z'}
    PAR2_EXTENSIONS = {'.par2'}

    # Patterns
    MULTIPART_RAR_PATTERN = re.compile(r'\.part(\d+)\.rar$', re.I)
    OLD_RAR_SPLIT_PATTERN = re.compile(r'\.[rs](\d{2})$', re.I)
    OBFUSCATED_PATTERN = re.compile(r'^[a-f0-9]{20,}$', re.I)  # Long hex strings

    def __init__(self):
        self._file_stats: Dict[str, int] = {}
        self._patterns_detected: List[str] = []
        self._warnings: List[str] = []

    def classify(self, nzb_files: List['NZBFile']) -> ClassificationResult:
        """
        Classify release type based on NZB file list.

        Args:
            nzb_files: List of NZBFile objects from parsed NZB

        Returns:
            ClassificationResult with type, confidence, and metadata
        """
        # Reset state
        self._file_stats = {
            'media': 0, 'rar': 0, 'rar_split': 0, 'zip': 0,
            '7z': 0, 'par2': 0, 'other': 0, 'no_ext': 0
        }
        self._patterns_detected = []
        self._warnings = []

        # Skip PAR2 files for classification
        main_files = [f for f in nzb_files if not self._is_par2(f)]
        par2_files = [f for f in nzb_files if self._is_par2(f)]

        if not main_files:
            return ClassificationResult(
                release_type=ReleaseType.UNKNOWN,
                confidence=0.0,
                warnings=["No main files found in NZB"]
            )

        # Step 1: Analyze extensions
        self._analyze_extensions(main_files)

        # Step 2: Detect patterns
        self._detect_archive_patterns(main_files)
        self._detect_obfuscation(main_files)

        # Step 3: Calculate scores and select best type
        release_type, confidence = self._calculate_best_type()

        # Step 4: Estimate nesting depth
        nesting_depth = self._estimate_nesting_depth(release_type)

        logger.info(f"[ADAPTIVE] Classification: {release_type.value} "
                   f"(confidence: {confidence:.0%})")
        logger.debug(f"[ADAPTIVE] Stats: {self._file_stats}")
        logger.debug(f"[ADAPTIVE] Patterns: {self._patterns_detected}")

        return ClassificationResult(
            release_type=release_type,
            confidence=confidence,
            detected_patterns=self._patterns_detected.copy(),
            archive_count=self._file_stats['rar'] + self._file_stats['rar_split'] +
                         self._file_stats['zip'] + self._file_stats['7z'],
            media_count=self._file_stats['media'],
            par2_count=len(par2_files),
            estimated_nesting_depth=nesting_depth,
            is_obfuscated='OBFUSCATED' in self._patterns_detected,
            warnings=self._warnings.copy()
        )

    def _is_par2(self, nzb_file: 'NZBFile') -> bool:
        """Check if file is PAR2."""
        return '.par2' in nzb_file.filename.lower()

    def _get_extension(self, filename: str) -> str:
        """Get lowercase extension from filename."""
        if '.' not in filename:
            return ''
        return '.' + filename.rsplit('.', 1)[-1].lower()

    def _analyze_extensions(self, files: List['NZBFile']) -> None:
        """Count files by extension category."""
        for f in files:
            ext = self._get_extension(f.filename)

            if not ext:
                self._file_stats['no_ext'] += 1
            elif ext in self.MEDIA_EXTENSIONS:
                self._file_stats['media'] += 1
            elif ext in self.RAR_EXTENSIONS:
                self._file_stats['rar'] += 1
            elif self.OLD_RAR_SPLIT_PATTERN.match(ext):
                self._file_stats['rar_split'] += 1
            elif ext in self.ZIP_EXTENSIONS:
                self._file_stats['zip'] += 1
            elif ext in self.SEVENZ_EXTENSIONS:
                self._file_stats['7z'] += 1
            elif ext in self.PAR2_EXTENSIONS:
                self._file_stats['par2'] += 1
            else:
                self._file_stats['other'] += 1

    def _detect_archive_patterns(self, files: List['NZBFile']) -> None:
        """Detect multipart RAR, split archives, etc."""
        filenames = [f.filename for f in files]

        # Check for modern multipart RAR (.part001.rar)
        part_numbers = []
        for name in filenames:
            match = self.MULTIPART_RAR_PATTERN.search(name)
            if match:
                part_numbers.append(int(match.group(1)))

        if part_numbers:
            if len(part_numbers) > 1:
                # Check if sequence is complete
                expected = set(range(1, max(part_numbers) + 1))
                actual = set(part_numbers)
                if expected == actual:
                    self._patterns_detected.append("COMPLETE_MULTIPART_RAR")
                else:
                    self._patterns_detected.append("INCOMPLETE_MULTIPART_RAR")
                    missing = expected - actual
                    self._warnings.append(f"Missing RAR parts: {sorted(missing)}")
            else:
                self._patterns_detected.append("SINGLE_PART_RAR")

        # Check for old-style split RAR (.rar + .r00, .r01, ...)
        has_base_rar = any(name.lower().endswith('.rar') and
                          not self.MULTIPART_RAR_PATTERN.search(name.lower())
                          for name in filenames)
        has_r_parts = self._file_stats['rar_split'] > 0

        if has_base_rar and has_r_parts:
            self._patterns_detected.append("OLD_STYLE_SPLIT_RAR")
        elif has_base_rar and not has_r_parts and self._file_stats['rar'] == 1:
            self._patterns_detected.append("SINGLE_RAR_FILE")

        # Check for ZIP only (possible scene ZIP->RAR)
        if self._file_stats['zip'] > 0 and self._file_stats['rar'] == 0:
            self._patterns_detected.append("ZIP_ONLY")

        # Check for 7z
        if self._file_stats['7z'] > 0:
            self._patterns_detected.append("HAS_7Z")

    def _detect_obfuscation(self, files: List['NZBFile']) -> None:
        """Detect obfuscated filenames."""
        no_ext_count = self._file_stats['no_ext']
        total = len(files)

        if total == 0:
            return

        # High ratio of files without extensions = likely obfuscated
        if no_ext_count > total * 0.5:
            self._patterns_detected.append("OBFUSCATED")

        # Check for hex-like names
        hex_count = 0
        for f in files:
            stem = Path(f.filename).stem
            if self.OBFUSCATED_PATTERN.match(stem):
                hex_count += 1

        if hex_count > total * 0.3:
            if "OBFUSCATED" not in self._patterns_detected:
                self._patterns_detected.append("OBFUSCATED")

    def _calculate_best_type(self) -> Tuple[ReleaseType, float]:
        """Calculate scores for each release type and return best match."""
        scores: Dict[ReleaseType, float] = {}

        total_archives = (self._file_stats['rar'] + self._file_stats['rar_split'] +
                        self._file_stats['zip'] + self._file_stats['7z'])

        # DIRECT_MEDIA: Media files only, no archives
        if self._file_stats['media'] > 0 and total_archives == 0:
            if self._file_stats['no_ext'] == 0:
                scores[ReleaseType.DIRECT_MEDIA] = 0.95
            else:
                # Some files without extensions - might be obfuscated
                scores[ReleaseType.DIRECT_MEDIA] = 0.60

        # MULTIPART_RAR: Modern .partXXX.rar format
        if "COMPLETE_MULTIPART_RAR" in self._patterns_detected:
            scores[ReleaseType.MULTIPART_RAR] = 0.95
        elif "INCOMPLETE_MULTIPART_RAR" in self._patterns_detected:
            scores[ReleaseType.MULTIPART_RAR] = 0.80

        # SINGLE_RAR: Just one RAR file
        if "SINGLE_RAR_FILE" in self._patterns_detected:
            scores[ReleaseType.SINGLE_RAR] = 0.90

        # OLD_STYLE_SPLIT_RAR: .rar + .r00, .r01, ...
        if "OLD_STYLE_SPLIT_RAR" in self._patterns_detected:
            scores[ReleaseType.MULTIPART_RAR] = max(
                scores.get(ReleaseType.MULTIPART_RAR, 0), 0.90
            )

        # ZIP_CONTAINING_RAR: ZIP only, likely scene format
        if "ZIP_ONLY" in self._patterns_detected:
            scores[ReleaseType.ZIP_CONTAINING_RAR] = 0.80

        # SEVENZ: 7z archives present
        if "HAS_7Z" in self._patterns_detected:
            scores[ReleaseType.SEVENZ] = 0.85

        # OBFUSCATED: Random/hex filenames without extensions
        if "OBFUSCATED" in self._patterns_detected:
            # Obfuscated takes precedence if no clear archive pattern
            if not any(p in self._patterns_detected for p in
                      ["COMPLETE_MULTIPART_RAR", "OLD_STYLE_SPLIT_RAR", "ZIP_ONLY"]):
                scores[ReleaseType.OBFUSCATED] = 0.85

        # Select best match
        if not scores:
            return ReleaseType.UNKNOWN, 0.5

        best_type = max(scores, key=scores.get)
        return best_type, scores[best_type]

    def _estimate_nesting_depth(self, release_type: ReleaseType) -> int:
        """Estimate how many extraction levels are needed."""
        if release_type == ReleaseType.DIRECT_MEDIA:
            return 0
        elif release_type == ReleaseType.ZIP_CONTAINING_RAR:
            return 2  # ZIP -> RAR -> content
        elif release_type == ReleaseType.NESTED_RAR:
            return 2  # RAR -> RAR -> content
        elif release_type == ReleaseType.OBFUSCATED:
            return 2  # Assume nested for safety
        else:
            return 1  # Single level extraction


# =============================================================================
# EXTRACTION STRATEGY
# =============================================================================

class ExtractionStrategy:
    """
    Factory for creating extraction plans based on release classification.

    Each strategy defines which tools to use, in what order, and whether
    to keep intermediate results in RAM or flush to disk.
    """

    @staticmethod
    def create_plan(classification: ClassificationResult) -> ExtractionPlan:
        """
        Create extraction plan based on classification result.

        Args:
            classification: Result from ReleaseClassifier

        Returns:
            ExtractionPlan with stages and configuration
        """
        strategy_map = {
            ReleaseType.DIRECT_MEDIA: ExtractionStrategy._direct_media_plan,
            ReleaseType.SINGLE_RAR: ExtractionStrategy._single_rar_plan,
            ReleaseType.MULTIPART_RAR: ExtractionStrategy._multipart_rar_plan,
            ReleaseType.ZIP_CONTAINING_RAR: ExtractionStrategy._zip_rar_plan,
            ReleaseType.NESTED_RAR: ExtractionStrategy._nested_rar_plan,
            ReleaseType.OBFUSCATED: ExtractionStrategy._obfuscated_plan,
            ReleaseType.SEVENZ: ExtractionStrategy._sevenz_plan,
        }

        builder = strategy_map.get(
            classification.release_type,
            ExtractionStrategy._fallback_plan
        )

        plan = builder(classification)

        logger.info(f"[ADAPTIVE] Plan created: {len(plan.stages)} stages, "
                   f"expects_nested={plan.expects_nested}")
        for stage in plan.stages:
            logger.debug(f"[ADAPTIVE]   Stage {stage.stage_number}: "
                        f"{stage.tool.value} ({stage.input_pattern}) -> {stage.output_type}")

        return plan

    @staticmethod
    def _direct_media_plan(cls: ClassificationResult) -> ExtractionPlan:
        """Direct media: just flush to disk, no extraction."""
        return ExtractionPlan(
            release_type=ReleaseType.DIRECT_MEDIA,
            stages=[
                ExtractionStage(
                    stage_number=1,
                    tool=ExtractionTool.DIRECT_FLUSH,
                    input_pattern="*",
                    output_type="media",
                    keep_in_ram=False,
                    description="Flush media files directly to disk"
                )
            ],
            expects_nested=False,
            max_nesting_depth=0,
            estimated_memory_mb=0
        )

    @staticmethod
    def _single_rar_plan(cls: ClassificationResult) -> ExtractionPlan:
        """Single RAR file extraction."""
        return ExtractionPlan(
            release_type=ReleaseType.SINGLE_RAR,
            stages=[
                ExtractionStage(
                    stage_number=1,
                    tool=ExtractionTool.UNRAR_DLL,
                    input_pattern="*.rar",
                    output_type="content",
                    keep_in_ram=False,
                    description="Extract single RAR to disk"
                )
            ],
            expects_nested=False,
            max_nesting_depth=1
        )

    @staticmethod
    def _multipart_rar_plan(cls: ClassificationResult) -> ExtractionPlan:
        """Multipart RAR extraction (.partXXX.rar or .rXX)."""
        return ExtractionPlan(
            release_type=ReleaseType.MULTIPART_RAR,
            stages=[
                ExtractionStage(
                    stage_number=1,
                    tool=ExtractionTool.UNRAR_DLL,
                    input_pattern="*.rar",
                    output_type="content",
                    keep_in_ram=False,
                    description="Extract multipart RAR to disk"
                )
            ],
            expects_nested=True,  # Might contain nested archives
            max_nesting_depth=2,
            estimated_memory_mb=cls.archive_count * 50
        )

    @staticmethod
    def _zip_rar_plan(cls: ClassificationResult) -> ExtractionPlan:
        """ZIP containing RAR (scene standard): ZIP(RAM) -> RAR(RAM) -> disk."""
        return ExtractionPlan(
            release_type=ReleaseType.ZIP_CONTAINING_RAR,
            stages=[
                ExtractionStage(
                    stage_number=1,
                    tool=ExtractionTool.ZIPFILE,
                    input_pattern="*.zip",
                    output_type="rar",
                    keep_in_ram=True,
                    description="Extract ZIP to RAM (expect RAR parts)"
                ),
                ExtractionStage(
                    stage_number=2,
                    tool=ExtractionTool.UNRAR_DLL,
                    input_pattern="*.rar",
                    output_type="content",
                    keep_in_ram=False,
                    description="Extract RAR parts to disk"
                )
            ],
            expects_nested=True,
            max_nesting_depth=3,
            estimated_memory_mb=cls.archive_count * 100
        )

    @staticmethod
    def _nested_rar_plan(cls: ClassificationResult) -> ExtractionPlan:
        """RAR containing RAR: cascaded extraction."""
        return ExtractionPlan(
            release_type=ReleaseType.NESTED_RAR,
            stages=[
                ExtractionStage(
                    stage_number=1,
                    tool=ExtractionTool.UNRAR_DLL,
                    input_pattern="*.rar",
                    output_type="rar",  # Expect nested RAR
                    keep_in_ram=False,
                    description="Extract outer RAR to disk"
                ),
                ExtractionStage(
                    stage_number=2,
                    tool=ExtractionTool.UNRAR_DLL,
                    input_pattern="*.rar",
                    output_type="content",
                    keep_in_ram=False,
                    description="Extract nested RAR to disk"
                )
            ],
            expects_nested=True,
            max_nesting_depth=3
        )

    @staticmethod
    def _obfuscated_plan(cls: ClassificationResult) -> ExtractionPlan:
        """Obfuscated release: use magic byte detection."""
        return ExtractionPlan(
            release_type=ReleaseType.OBFUSCATED,
            stages=[
                ExtractionStage(
                    stage_number=1,
                    tool=ExtractionTool.UNRAR_DLL,  # Most common, adapt at runtime
                    input_pattern="*",  # Match all (no extensions)
                    output_type="unknown",
                    keep_in_ram=False,
                    description="Extract obfuscated archives (magic byte detection)"
                )
            ],
            expects_nested=True,
            max_nesting_depth=3,
            estimated_memory_mb=cls.archive_count * 100
        )

    @staticmethod
    def _sevenz_plan(cls: ClassificationResult) -> ExtractionPlan:
        """7z archive extraction."""
        return ExtractionPlan(
            release_type=ReleaseType.SEVENZ,
            stages=[
                ExtractionStage(
                    stage_number=1,
                    tool=ExtractionTool.SEVENZIP_CLI,
                    input_pattern="*.7z",
                    output_type="content",
                    keep_in_ram=False,
                    description="Extract 7z via 7-Zip CLI"
                )
            ],
            expects_nested=True,
            max_nesting_depth=2
        )

    @staticmethod
    def _fallback_plan(cls: ClassificationResult) -> ExtractionPlan:
        """Fallback plan when type cannot be determined."""
        return ExtractionPlan(
            release_type=ReleaseType.UNKNOWN,
            stages=[
                ExtractionStage(
                    stage_number=1,
                    tool=ExtractionTool.SEVENZIP_CLI,
                    input_pattern="*",
                    output_type="content",
                    keep_in_ram=False,
                    description="Fallback: extract all via 7-Zip"
                )
            ],
            expects_nested=True,
            max_nesting_depth=3
        )


# =============================================================================
# ADAPTIVE EXTRACTOR
# =============================================================================

class AdaptiveExtractor:
    """
    Runtime extraction engine that adapts based on actual content.

    Uses the pre-selected strategy as a baseline but can switch
    strategies if extracted content differs from expectation.

    All extraction stays in RAM until final flush to disk.
    """

    # Magic byte signatures
    RAR_MAGIC = b'Rar!'
    ZIP_MAGIC_1 = b'PK\x03\x04'
    ZIP_MAGIC_2 = b'PK\x05\x06'
    SEVENZ_MAGIC = b'7z\xbc\xaf\x27\x1c'
    PAR2_MAGIC = b'PAR2'

    def __init__(
        self,
        ram_buffer: 'RamBuffer',
        extract_dir: Path,
        password: str = "",
        on_progress: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize adaptive extractor.

        Args:
            ram_buffer: RamBuffer containing downloaded files
            extract_dir: Destination directory for extracted files
            password: Password for encrypted archives
            on_progress: Progress callback (message, percentage)
        """
        self.ram_buffer = ram_buffer
        self.extract_dir = extract_dir
        self.password = password
        self.on_progress = on_progress

        # Runtime state
        self._current_stage = 0
        self._intermediate_data: Dict[str, bytes] = {}
        self._adaptations_made = 0

        # Import RAM RAR extractor
        self._rar_available = False
        try:
            from src.core.ram_rar import extract_multipart_rar_to_disk, extract_multipart_rar_from_memory
            self._extract_rar_to_disk = extract_multipart_rar_to_disk
            self._extract_rar_to_memory = extract_multipart_rar_from_memory
            self._rar_available = True
        except ImportError:
            logger.warning("[ADAPTIVE] UnRAR DLL not available")

    def execute(self, plan: ExtractionPlan) -> ExtractionResult:
        """
        Execute extraction plan with real-time adaptation.

        Args:
            plan: ExtractionPlan from ExtractionStrategy

        Returns:
            ExtractionResult with statistics and any errors
        """
        total_extracted = 0
        total_bytes = 0
        errors = []

        # Ensure output directory exists
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[ADAPTIVE] Starting extraction: {plan.release_type.value}")
        logger.info(f"[ADAPTIVE] Output: {self.extract_dir}")

        try:
            stages_to_run = list(plan.stages)
            stage_idx = 0

            while stage_idx < len(stages_to_run):
                stage = stages_to_run[stage_idx]
                self._current_stage = stage.stage_number

                progress = (stage_idx / len(stages_to_run)) * 100
                self._report_progress(
                    f"Stage {stage.stage_number}: {stage.description}",
                    progress
                )

                logger.info(f"[ADAPTIVE] Stage {stage.stage_number}/{len(stages_to_run)}: "
                           f"{stage.tool.value}")

                # Execute stage
                result = self._execute_stage(stage)

                total_extracted += result.files_extracted
                total_bytes += result.bytes_extracted
                errors.extend(result.errors)

                # Check if adaptation is needed
                if result.needs_adaptation and plan.expects_nested:
                    new_stages = self._adapt_plan(result, stage_idx + 1)
                    if new_stages:
                        stages_to_run.extend(new_stages)
                        self._adaptations_made += len(new_stages)
                        logger.info(f"[ADAPTIVE] Added {len(new_stages)} new stages "
                                   f"for nested archives")

                stage_idx += 1

            # === FINAL FLUSH: Write intermediate data to disk ===
            if self._intermediate_data:
                flush_result = self._final_flush_to_disk()
                total_extracted += flush_result.files_extracted
                total_bytes += flush_result.bytes_extracted
                errors.extend(flush_result.errors)

            self._report_progress("Extraction complete", 100)

            logger.info(f"[ADAPTIVE] Complete: {total_extracted} files, "
                       f"{total_bytes / (1024*1024):.1f} MB")

            return ExtractionResult(
                success=len(errors) == 0,
                files_extracted=total_extracted,
                bytes_written=total_bytes,
                stages_executed=len(stages_to_run),
                adaptations_made=self._adaptations_made,
                errors=errors
            )

        except Exception as e:
            logger.error(f"[ADAPTIVE] Extraction failed: {e}")
            return ExtractionResult(
                success=False,
                files_extracted=total_extracted,
                bytes_written=total_bytes,
                stages_executed=self._current_stage,
                adaptations_made=self._adaptations_made,
                errors=[str(e)]
            )

    def _execute_stage(self, stage: ExtractionStage) -> StageResult:
        """Execute a single extraction stage."""
        if stage.tool == ExtractionTool.DIRECT_FLUSH:
            return self._execute_direct_flush(stage)
        elif stage.tool == ExtractionTool.UNRAR_DLL:
            return self._execute_rar_extraction(stage)
        elif stage.tool == ExtractionTool.ZIPFILE:
            return self._execute_zip_extraction(stage)
        elif stage.tool == ExtractionTool.SEVENZIP_CLI:
            return self._execute_7z_extraction(stage)
        else:
            return StageResult(errors=[f"Unknown tool: {stage.tool}"])

    def _execute_direct_flush(self, stage: ExtractionStage) -> StageResult:
        """Flush files directly to disk without extraction."""
        files_flushed = 0
        bytes_flushed = 0

        for key, ram_file in self.ram_buffer.get_all_files().items():
            # Skip PAR2 files
            if '.par2' in ram_file.filename.lower():
                continue
            if self._is_par2_by_magic(ram_file):
                continue

            # Skip archives (shouldn't be here for DIRECT_MEDIA but safety check)
            if self._is_archive_by_magic(ram_file):
                continue

            # Flush to disk
            dest_path = self.extract_dir / ram_file.filename
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            ram_file.data.seek(0)
            read_size = ram_file.actual_size if ram_file.actual_size > 0 else ram_file.size

            with open(dest_path, 'wb') as f:
                f.write(ram_file.data.read(read_size))

            files_flushed += 1
            bytes_flushed += read_size

        logger.info(f"[ADAPTIVE] Flushed {files_flushed} files "
                   f"({bytes_flushed / (1024*1024):.1f} MB)")

        return StageResult(
            files_extracted=files_flushed,
            bytes_extracted=bytes_flushed
        )

    def _execute_rar_extraction(self, stage: ExtractionStage) -> StageResult:
        """Extract RAR archives - DIRECT TO DISK for maximum speed."""
        if not self._rar_available:
            return StageResult(errors=["UnRAR DLL not available"])

        # For obfuscated releases, first extract any ZIPs to get inner RARs
        is_obfuscated_mode = stage.input_pattern == "*"
        if is_obfuscated_mode:
            zip_result = self._extract_obfuscated_zips()
            if zip_result.files_extracted > 0:
                logger.info(f"[ADAPTIVE] Pre-extracted {zip_result.files_extracted} "
                           f"files from ZIPs to intermediate RAM buffer")

        # Collect RAR files from RAM buffer or intermediate data
        rar_parts: Dict[str, bytes] = {}

        # First check intermediate data (from previous stage)
        if self._intermediate_data:
            for name, data in self._intermediate_data.items():
                if self._matches_rar(name, data):
                    rar_parts[name] = data
            # Clear processed archives from intermediate data
            for name in list(rar_parts.keys()):
                if name in self._intermediate_data:
                    del self._intermediate_data[name]

        # Also collect from RAM buffer if no intermediate data or if we need more
        if not rar_parts:
            collected_files = []

            for key, ram_file in self.ram_buffer.get_all_files().items():
                # Skip PAR2 files
                if self._is_par2_by_magic(ram_file):
                    continue

                if self._matches_pattern(ram_file.filename, stage.input_pattern):
                    if self._is_rar_file(ram_file):
                        ram_file.data.seek(0)
                        read_size = ram_file.actual_size or ram_file.size
                        data = ram_file.data.read(read_size)

                        # Check if filename looks obfuscated (no proper archive extension)
                        filename_lower = ram_file.filename.lower()
                        has_rar_ext = filename_lower.endswith('.rar') or \
                                     re.search(r'\.[rs]\d{2}$', filename_lower) or \
                                     re.search(r'\.part\d+\.rar$', filename_lower)

                        if is_obfuscated_mode and not has_rar_ext:
                            # Obfuscated filename - collect for sorting by RAR header
                            collected_files.append((key, data))
                        else:
                            # Normal filename - use as-is
                            rar_parts[ram_file.filename] = data

            # Sort obfuscated files by RAR header analysis (first volume + volume numbers)
            obfuscated_presorted = False  # Will be set True if we sort and rename obfuscated files
            if collected_files:
                from src.core.ram_rar import get_rar_volume_number, is_first_rar_volume

                # Helper to extract idx number from buffer key (e.g., "filename__idx0052" -> 52)
                def get_buffer_idx(key: str) -> int:
                    match = re.search(r'__idx(\d+)$', key)
                    return int(match.group(1)) if match else 9999

                # Parse volume info from RAR headers
                files_with_info = []
                first_volume_idx = None

                for idx, (key, data) in enumerate(collected_files):
                    vol_num = get_rar_volume_number(data)
                    is_first = is_first_rar_volume(data)
                    buf_idx = get_buffer_idx(key)  # Extract NZB order index
                    files_with_info.append((key, data, vol_num, is_first, buf_idx))

                    if is_first and first_volume_idx is None:
                        first_volume_idx = idx
                        logger.info(f"[ADAPTIVE] Found FIRST volume at index {idx} (buf_idx={buf_idx}): {key}")
                    if vol_num >= 0:
                        logger.debug(f"[ADAPTIVE] {key}: RAR volume {vol_num}")

                # Check how many have valid volume numbers
                valid_count = sum(1 for _, _, v, _, _ in files_with_info if v >= 0)
                first_count = sum(1 for _, _, _, f, _ in files_with_info if f)

                logger.info(f"[ADAPTIVE] Analysis: {valid_count}/{len(collected_files)} have volume numbers, "
                           f"{first_count} marked as first")

                if valid_count == len(collected_files):
                    # All files have volume numbers - sort by them
                    files_with_info.sort(key=lambda x: x[2])
                    logger.info(f"[ADAPTIVE] Sorted {len(collected_files)} obfuscated RAR files by volume number")
                elif valid_count > len(collected_files) * 0.8:
                    # Most files have volume numbers - use volume numbers, fallback to buffer idx
                    files_with_info.sort(key=lambda x: (x[2] if x[2] >= 0 else x[4]))
                    logger.info(f"[ADAPTIVE] Sorted {valid_count}/{len(collected_files)} by volume number, "
                               f"rest by buffer index")
                else:
                    # Most files don't have volume numbers - use buffer index (NZB order)
                    # This works for encrypted headers where we can't read volume numbers
                    files_with_info.sort(key=lambda x: x[4])  # Sort by buf_idx
                    logger.info(f"[ADAPTIVE] Sorted by NZB buffer index (encrypted headers likely)")

                # Build rar_parts dict with proper naming - already sorted, so set presorted flag
                for idx, (key, data, vol_num, is_first, buf_idx) in enumerate(files_with_info):
                    rar_parts[f"obfuscated_{idx:04d}.rar"] = data

                logger.info(f"[ADAPTIVE] Collected {len(collected_files)} obfuscated RAR files")
                # Mark as presorted so extract_multipart_rar_from_memory won't re-sort
                obfuscated_presorted = True

        if not rar_parts:
            logger.warning("[ADAPTIVE] No RAR files found for extraction")
            return StageResult()

        total_compressed = sum(len(d) for d in rar_parts.values())
        logger.info(f"[ADAPTIVE] Processing {len(rar_parts)} RAR files "
                   f"({total_compressed / (1024*1024):.1f} MB)")

        # === DIRECT TO DISK EXTRACTION ===
        # This is much faster than extracting to RAM then flushing
        # - Eliminates 5+ GB RAM usage for large files
        # - Eliminates the "FINAL FLUSH" step (saves ~8 seconds for 5GB)
        try:
            from src.core.ram_rar import extract_multipart_rar_to_disk

            # Ensure output directory exists
            self.extract_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"[ADAPTIVE] DIRECT DISK extraction to: {self.extract_dir}")

            # For obfuscated archives, try multi-volume extraction with password
            # Even with encrypted headers, if data is pre-sorted by NZB index, multi-volume should work
            if is_obfuscated_mode and obfuscated_presorted and len(rar_parts) > 1:
                # Check if headers are encrypted (most files lack volume numbers)
                from src.core.ram_rar import get_rar_volume_number
                sample_data = list(rar_parts.values())[:5]
                valid_vol_nums = sum(1 for d in sample_data if get_rar_volume_number(d) >= 0)

                if valid_vol_nums < len(sample_data) * 0.5:
                    # Encrypted headers detected - but STILL try multi-volume first with password
                    # The data is pre-sorted by NZB buffer index, so order should be correct
                    logger.info("[ADAPTIVE] Encrypted headers detected - trying multi-volume with password...")
                    files_extracted = extract_multipart_rar_to_disk(
                        rar_parts,
                        self.extract_dir,
                        password=self.password or "",
                        presorted=True  # Skip re-sorting, use NZB buffer order
                    )
                    # If multi-volume fails, fallback will be triggered below (line 954+)
                else:
                    # Volume numbers available - use multi-volume extraction
                    files_extracted = extract_multipart_rar_to_disk(
                        rar_parts,
                        self.extract_dir,
                        password=self.password or ""
                    )
            else:
                # Extract directly to disk - no RAM intermediate!
                files_extracted = extract_multipart_rar_to_disk(
                    rar_parts,
                    self.extract_dir,
                    password=self.password or ""
                )

            if files_extracted == 0:
                # Fallback: try individual extraction for obfuscated archives
                if is_obfuscated_mode and len(rar_parts) > 1:
                    logger.info("[ADAPTIVE] Multi-volume failed, trying individual extraction...")
                    files_extracted = self._extract_obfuscated_individual_to_disk(rar_parts)

                if files_extracted == 0:
                    logger.warning("[ADAPTIVE] RAR extraction returned no files")
                    return StageResult()

            # Scan extracted files to detect nested archives
            found_types: Set[str] = set()
            nested_archives: List[Path] = []
            final_files_count = 0
            total_bytes = 0

            for f in self.extract_dir.rglob('*'):
                if f.is_file():
                    ext = f.suffix.lower()
                    file_size = f.stat().st_size
                    total_bytes += file_size

                    # Check if it's a nested archive
                    is_archive = False
                    if ext in {'.rar', '.zip', '.7z'} or re.match(r'\.[rs]\d{2}$', ext):
                        is_archive = True
                        found_types.add(ext.lstrip('.') if ext != '.rar' else 'rar')
                    else:
                        # Check magic bytes
                        try:
                            with open(f, 'rb') as fh:
                                magic = fh.read(8)
                            if magic[:4] == self.RAR_MAGIC:
                                is_archive = True
                                found_types.add('rar')
                            elif magic[:4] in (self.ZIP_MAGIC_1, self.ZIP_MAGIC_2):
                                is_archive = True
                                found_types.add('zip')
                            elif magic[:6] == self.SEVENZ_MAGIC:
                                is_archive = True
                                found_types.add('7z')
                        except:
                            pass

                    if is_archive:
                        nested_archives.append(f)
                    else:
                        final_files_count += 1
                        if ext in {'.mkv', '.mp4', '.avi', '.iso', '.m2ts'}:
                            found_types.add('media')

            # If nested archives found, load them into intermediate data for next stage
            if nested_archives:
                logger.info(f"[ADAPTIVE] Found {len(nested_archives)} nested archives on disk, "
                           f"loading to RAM for next stage...")
                for archive_path in nested_archives:
                    try:
                        archive_data = archive_path.read_bytes()
                        self._intermediate_data[archive_path.name] = archive_data
                        # Delete the archive from disk since we'll re-extract it
                        archive_path.unlink()
                    except Exception as e:
                        logger.warning(f"[ADAPTIVE] Failed to load nested archive {archive_path}: {e}")

            # Fix duplicate folder issue: if archive contained a root folder matching extract_dir name
            # e.g., extract_dir/Release.Name/Release.Name/* -> extract_dir/Release.Name/*
            try:
                subdirs = [d for d in self.extract_dir.iterdir() if d.is_dir()]
                if len(subdirs) == 1:
                    subdir = subdirs[0]
                    # Check if subfolder name matches or is similar to extract_dir name
                    if subdir.name == self.extract_dir.name:
                        logger.info(f"[ADAPTIVE] Fixing duplicate folder: {subdir.name}")
                        import shutil
                        # Move all contents up one level
                        for item in subdir.iterdir():
                            dest = self.extract_dir / item.name
                            if dest.exists():
                                if dest.is_dir():
                                    shutil.rmtree(dest)
                                else:
                                    dest.unlink()
                            shutil.move(str(item), str(dest))
                        # Remove empty subfolder
                        subdir.rmdir()
                        logger.info(f"[ADAPTIVE] Moved contents up from duplicate folder")
            except Exception as e:
                logger.warning(f"[ADAPTIVE] Could not fix duplicate folder: {e}")

            logger.info(f"[ADAPTIVE] DIRECT extraction complete: {final_files_count} final files, "
                       f"{len(nested_archives)} nested archives, {total_bytes / (1024*1024):.1f} MB")

            needs_adaptation = 'rar' in found_types or '7z' in found_types or 'zip' in found_types

            return StageResult(
                files_extracted=final_files_count,
                bytes_extracted=total_bytes,
                found_types=found_types,
                needs_adaptation=needs_adaptation
            )

        except Exception as e:
            logger.error(f"[ADAPTIVE] RAR extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return StageResult(errors=[str(e)])

    def _final_flush_to_disk(self) -> StageResult:
        """
        Final flush: Write all intermediate data to disk.
        This is the ONLY place where content is written to disk.
        """
        files_written = 0
        bytes_written = 0
        errors = []

        if not self._intermediate_data:
            return StageResult()

        # Ensure output directory exists
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[ADAPTIVE] FINAL FLUSH: Writing {len(self._intermediate_data)} files to disk...")

        for filename, content in self._intermediate_data.items():
            try:
                # Skip any remaining archives (shouldn't happen, but safety check)
                ext = self._get_extension(filename)
                if ext in {'.rar', '.zip', '.7z'} or re.match(r'\.[rs]\d{2}$', ext):
                    logger.warning(f"[ADAPTIVE] Skipping unflushed archive: {filename}")
                    continue

                # Handle paths with subdirectories
                dest_path = self.extract_dir / filename
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Skip if file already exists with same size (already extracted)
                if dest_path.exists():
                    existing_size = dest_path.stat().st_size
                    if existing_size == len(content):
                        logger.debug(f"[ADAPTIVE] Skipping existing file: {filename} ({existing_size} bytes)")
                        files_written += 1  # Count as success
                        bytes_written += len(content)
                        continue
                    else:
                        logger.debug(f"[ADAPTIVE] Overwriting {filename} (size mismatch: {existing_size} vs {len(content)})")

                # Write to disk
                dest_path.write_bytes(content)
                files_written += 1
                bytes_written += len(content)

            except Exception as e:
                logger.error(f"[ADAPTIVE] Failed to flush {filename}: {e}")
                errors.append(f"Flush failed: {filename}: {e}")

        # Clear intermediate data
        self._intermediate_data.clear()

        logger.info(f"[ADAPTIVE] FINAL FLUSH complete: {files_written} files, "
                   f"{bytes_written / (1024*1024):.1f} MB written to disk")

        return StageResult(
            files_extracted=files_written,
            bytes_extracted=bytes_written,
            errors=errors
        )

    def _execute_zip_extraction(self, stage: ExtractionStage) -> StageResult:
        """Extract ZIP archives in RAM. Each ZIP is extracted individually."""
        files_extracted = 0
        bytes_extracted = 0
        found_types: Set[str] = set()
        failed_zips = 0
        total_zips = 0

        for key, ram_file in self.ram_buffer.get_all_files().items():
            if not self._matches_pattern(ram_file.filename, stage.input_pattern):
                if not self._is_zip_by_magic(ram_file):
                    continue

            total_zips += 1

            # CRITICAL: Truncate BytesIO to actual_size!
            # The buffer is pre-allocated larger than actual data.
            # zipfile looks for central directory at END of buffer,
            # so we must give it only the valid bytes.
            ram_file.data.seek(0)
            read_size = ram_file.actual_size if ram_file.actual_size > 0 else ram_file.size
            truncated_data = io.BytesIO(ram_file.data.read(read_size))

            try:
                with zipfile.ZipFile(truncated_data) as zf:
                    pwd = self.password.encode() if self.password else None

                    for info in zf.infolist():
                        if info.is_dir():
                            continue

                        # Extract to intermediate RAM storage
                        content = zf.read(info.filename, pwd=pwd)

                        if stage.keep_in_ram:
                            # Store in intermediate - will be counted in final flush
                            self._intermediate_data[info.filename] = content
                        else:
                            # Write directly to disk - count now
                            dest = self.extract_dir / info.filename
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(content)
                            files_extracted += 1
                            bytes_extracted += len(content)

                        # Track type
                        ext = self._get_extension(info.filename)
                        if ext == '.rar' or self._is_rar_data(content):
                            found_types.add('rar')
                        elif ext == '.7z':
                            found_types.add('7z')
                        elif ext in {'.mkv', '.mp4', '.avi', '.iso'}:
                            found_types.add('media')

            except zipfile.BadZipFile as e:
                logger.debug(f"[ADAPTIVE] Bad ZIP (may be split): {ram_file.filename}: {e}")
                failed_zips += 1
                continue

        # If most ZIPs failed, they're probably split archives - concatenate and extract in RAM
        if failed_zips > 0 and failed_zips >= total_zips * 0.5:
            logger.info(f"[ADAPTIVE] {failed_zips}/{total_zips} ZIPs failed - trying split ZIP extraction IN RAM...")
            split_result = self._extract_split_zip_in_ram(stage)
            if split_result.files_extracted > 0:
                files_extracted += split_result.files_extracted
                bytes_extracted += split_result.bytes_extracted
                found_types.update(split_result.found_types)

        logger.info(f"[ADAPTIVE] ZIP extracted: {files_extracted} files "
                   f"({bytes_extracted / (1024*1024):.1f} MB)")
        if found_types:
            logger.info(f"[ADAPTIVE] Found types: {found_types}")

        needs_adaptation = 'rar' in found_types or '7z' in found_types

        return StageResult(
            files_extracted=files_extracted,
            bytes_extracted=bytes_extracted,
            found_types=found_types,
            needs_adaptation=needs_adaptation
        )

    def _extract_split_zip_in_ram(self, stage: ExtractionStage) -> StageResult:
        """
        Extract split ZIP archives 100% in RAM.

        Split ZIP archives (spanned archives) have:
        - Multiple data files without central directory
        - ONE file containing the central directory (must be LAST in concatenation)

        We detect the central directory file by looking for PK\x05\x06 signature
        near the end of the file.

        NO DISK I/O - everything stays in RAM until final flush.
        """
        # Collect all ZIP parts from buffer
        zip_parts: List[Tuple[str, bytes]] = []
        central_dir_file: Optional[Tuple[str, bytes]] = None

        for key, ram_file in self.ram_buffer.get_all_files().items():
            filename_lower = ram_file.filename.lower()
            if self._is_zip_by_magic(ram_file) or filename_lower.endswith('.zip'):
                ram_file.data.seek(0)
                read_size = ram_file.actual_size or ram_file.size
                data = ram_file.data.read(read_size)

                # Check if this file contains the central directory (PK\x05\x06 or PK\x06\x06)
                # The End of Central Directory record is near the end of the file
                has_central_dir = False
                # Check last 64KB for EOCD signature (can have comment up to 64KB)
                search_start = max(0, len(data) - 65536)
                eocd_pos = data.find(b'PK\x05\x06', search_start)
                if eocd_pos != -1:
                    has_central_dir = True
                else:
                    # Check for ZIP64 EOCD
                    eocd64_pos = data.find(b'PK\x06\x06', search_start)
                    if eocd64_pos != -1:
                        has_central_dir = True

                if has_central_dir:
                    central_dir_file = (ram_file.filename, data)
                    logger.debug(f"[ADAPTIVE] Found central directory in: {ram_file.filename}")
                else:
                    zip_parts.append((ram_file.filename, data))

        if not zip_parts and not central_dir_file:
            return StageResult()

        # If we found a central directory file, it must be LAST
        if central_dir_file:
            logger.info(f"[ADAPTIVE] Central directory file: {central_dir_file[0]}")
        else:
            logger.warning("[ADAPTIVE] No central directory found - trying best-effort ordering")

        logger.info(f"[ADAPTIVE] Concatenating {len(zip_parts)} ZIP parts + 1 central dir in RAM...")

        # Sort data parts by filename (best effort for obfuscated names)
        # For standard naming: file.z01, file.z02, ..., file.zip (last)
        def sort_zip_parts(item):
            name = item[0].lower()
            # .z01, .z02, etc. - standard split naming
            if re.search(r'\.z(\d+)$', name):
                match = re.search(r'\.z(\d+)$', name)
                return (0, int(match.group(1)), name)
            # .zip.001, .zip.002, etc.
            if re.search(r'\.zip\.(\d+)$', name):
                match = re.search(r'\.zip\.(\d+)$', name)
                return (0, int(match.group(1)), name)
            # For obfuscated names, sort alphabetically (better than random)
            return (1, 0, name)

        zip_parts_sorted = sorted(zip_parts, key=sort_zip_parts)

        # Try to determine correct order using central directory info
        if central_dir_file and len(zip_parts) > 1:
            # Parse central directory to get disk mapping
            disk_order = self._parse_zip_central_dir_for_order(
                central_dir_file[1], len(zip_parts)
            )
            if disk_order:
                # disk_order maps disk_number -> expected size or offset
                # We can try to match files by their size
                logger.info(f"[ADAPTIVE] Parsed central dir: {len(disk_order)} disk entries")

                # Try to reorder using size matching
                reordered = self._reorder_zip_parts_by_size(zip_parts, disk_order)
                if reordered:
                    zip_parts_sorted = reordered
                    logger.info("[ADAPTIVE] Reordered ZIP parts using central directory info")

        # Add central directory file at the END (this is critical!)
        if central_dir_file:
            zip_parts_sorted.append(central_dir_file)

        # Log the order for debugging
        logger.info(f"[ADAPTIVE] ZIP concatenation order ({len(zip_parts_sorted)} parts):")
        for i, (name, data) in enumerate(zip_parts_sorted[:5]):
            logger.info(f"[ADAPTIVE]   {i}: {name} ({len(data)} bytes)")
        if len(zip_parts_sorted) > 5:
            logger.info(f"[ADAPTIVE]   ... and {len(zip_parts_sorted) - 5} more")

        # Concatenate all parts into one BytesIO buffer
        combined_buffer = io.BytesIO()
        total_size = 0
        for name, data in zip_parts_sorted:
            combined_buffer.write(data)
            total_size += len(data)

        logger.info(f"[ADAPTIVE] Combined ZIP: {total_size / (1024*1024):.1f} MB in RAM")

        # Now extract from the combined buffer
        combined_buffer.seek(0)
        files_extracted = 0
        bytes_extracted = 0
        found_types: Set[str] = set()

        try:
            with zipfile.ZipFile(combined_buffer) as zf:
                pwd = self.password.encode() if self.password else None

                for info in zf.infolist():
                    if info.is_dir():
                        continue

                    try:
                        content = zf.read(info.filename, pwd=pwd)
                    except Exception as e:
                        logger.debug(f"[ADAPTIVE] Failed to extract {info.filename} from combined ZIP: {e}")
                        continue

                    # Store in intermediate RAM buffer
                    if stage.keep_in_ram:
                        # Store in intermediate - will be counted in final flush
                        self._intermediate_data[info.filename] = content
                    else:
                        # Write directly to disk - count now
                        dest = self.extract_dir / info.filename
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(content)
                        files_extracted += 1
                        bytes_extracted += len(content)

                    # Track types
                    ext = self._get_extension(info.filename)
                    if ext == '.rar' or (len(content) >= 4 and content[:4] == self.RAR_MAGIC):
                        found_types.add('rar')
                    elif ext == '.7z':
                        found_types.add('7z')
                    elif ext in {'.mkv', '.mp4', '.avi', '.iso'}:
                        found_types.add('media')

            logger.info(f"[ADAPTIVE] Split ZIP extracted IN RAM: {files_extracted} files "
                       f"({bytes_extracted / (1024*1024):.1f} MB)")

            return StageResult(
                files_extracted=files_extracted,
                bytes_extracted=bytes_extracted,
                found_types=found_types,
                needs_adaptation='rar' in found_types or '7z' in found_types
            )

        except zipfile.BadZipFile as e:
            logger.error(f"[ADAPTIVE] Combined ZIP still invalid: {e}")
            return StageResult(errors=[f"Split ZIP extraction failed: {e}"])

    def _extract_obfuscated_zips(self) -> StageResult:
        """
        Extract ZIP files from obfuscated buffer (detected by magic bytes).
        Results are stored in intermediate_data for subsequent RAR extraction.
        """
        files_extracted = 0
        bytes_extracted = 0
        zips_found = 0

        for key, ram_file in self.ram_buffer.get_all_files().items():
            # Skip PAR2 files
            if self._is_par2_by_magic(ram_file):
                continue

            # Check if this is a ZIP by magic bytes
            if not self._is_zip_by_magic(ram_file):
                continue

            zips_found += 1
            ram_file.data.seek(0)

            try:
                with zipfile.ZipFile(ram_file.data) as zf:
                    pwd = self.password.encode() if self.password else None

                    for info in zf.infolist():
                        if info.is_dir():
                            continue

                        # Extract to intermediate RAM storage
                        try:
                            content = zf.read(info.filename, pwd=pwd)
                        except Exception as e:
                            logger.debug(f"[ADAPTIVE] Failed to read {info.filename} from ZIP: {e}")
                            continue

                        # Store in intermediate data for RAR extraction
                        self._intermediate_data[info.filename] = content
                        files_extracted += 1
                        bytes_extracted += len(content)

            except zipfile.BadZipFile as e:
                logger.debug(f"[ADAPTIVE] Not a valid ZIP despite magic: {ram_file.filename}: {e}")
                continue
            except Exception as e:
                logger.warning(f"[ADAPTIVE] ZIP extraction failed for {ram_file.filename}: {e}")
                continue

        if zips_found > 0:
            logger.info(f"[ADAPTIVE] Found {zips_found} obfuscated ZIPs, "
                       f"extracted {files_extracted} inner files "
                       f"({bytes_extracted / (1024*1024):.1f} MB) to RAM")

        return StageResult(
            files_extracted=files_extracted,
            bytes_extracted=bytes_extracted
        )

    def _extract_obfuscated_individual_rars(self, rar_parts: Dict[str, bytes]) -> Dict[str, bytes]:
        """
        Extract obfuscated releases where each file is an INDIVIDUAL RAR archive
        containing one part of an inner old-style RAR split.

        This handles scene releases like:
        - 61 obfuscated files, each is a separate RAR
        - Each outer RAR contains ONE inner file (.rar, .r00-.r99, .s00-.s99)
        - Inner files form a complete old-style RAR split

        Strategy:
        1. Extract each outer RAR individually to get inner files
        2. Collect all inner files in memory
        3. If inner files form a RAR split, extract them
        4. Return final extracted content

        Args:
            rar_parts: Dict of filename -> bytes (outer RAR data)

        Returns:
            Dict of extracted filename -> content bytes
        """
        from src.core.ram_rar import RamRarExtractor, extract_multipart_rar_from_memory
        import tempfile
        import shutil

        logger.info(f"[ADAPTIVE] Extracting {len(rar_parts)} individual outer RARs...")

        # Step 1: Extract each outer RAR individually to collect inner files
        inner_files: Dict[str, bytes] = {}
        outer_extracted = 0
        outer_failed = 0

        for outer_name, outer_data in rar_parts.items():
            try:
                extractor = RamRarExtractor()
                if not extractor.set_archive_data(outer_data):
                    logger.debug(f"[ADAPTIVE] Failed to load outer RAR: {outer_name}: {extractor.get_last_error()}")
                    outer_failed += 1
                    continue

                # IMPORTANT: Outer RARs are usually NOT password protected!
                # Try WITHOUT password first, then WITH password if that fails
                extracted = extractor.extract_all()  # No password

                # If no password didn't work, try WITH password
                if not extracted and self.password:
                    extractor2 = RamRarExtractor()
                    extractor2.set_archive_data(outer_data)
                    extractor2.set_password(self.password)
                    extracted = extractor2.extract_all()

                # If RAM extraction still failed, try disk-based extraction
                if not extracted:
                    temp_extract = Path(tempfile.mkdtemp(prefix='dler_outer_'))
                    try:
                        # Try without password first
                        extractor3 = RamRarExtractor()
                        extractor3.set_archive_data(outer_data)
                        count = extractor3.extract_to_disk(temp_extract)

                        # If that failed AND we have a password, try WITH password
                        if count == 0 and self.password:
                            extractor4 = RamRarExtractor()
                            extractor4.set_archive_data(outer_data)
                            extractor4.set_password(self.password)
                            count = extractor4.extract_to_disk(temp_extract)

                        if count > 0:
                            for f in temp_extract.rglob('*'):
                                if f.is_file():
                                    rel_path = str(f.relative_to(temp_extract))
                                    extracted[rel_path] = f.read_bytes()
                    finally:
                        shutil.rmtree(temp_extract, ignore_errors=True)

                if extracted:
                    outer_extracted += 1
                    for inner_name, inner_data in extracted.items():
                        # If same name exists, make it unique with counter
                        final_name = inner_name
                        if final_name in inner_files:
                            base = Path(inner_name).stem
                            ext = Path(inner_name).suffix
                            counter = 1
                            while final_name in inner_files:
                                final_name = f"{base}_{counter}{ext}"
                                counter += 1
                        inner_files[final_name] = inner_data
                        logger.debug(f"[ADAPTIVE] Collected inner file: {final_name} ({len(inner_data)} bytes)")
                else:
                    outer_failed += 1
                    logger.debug(f"[ADAPTIVE] No files extracted from {outer_name}")

            except Exception as e:
                outer_failed += 1
                logger.debug(f"[ADAPTIVE] Exception extracting {outer_name}: {e}")
                continue

        logger.info(f"[ADAPTIVE] Outer RAR results: {outer_extracted} succeeded, {outer_failed} failed, "
                   f"collected {len(inner_files)} inner files")

        if not inner_files:
            return {}

        # Step 2: Check if inner files form a RAR split (old-style .rar + .rXX)
        inner_names = list(inner_files.keys())
        inner_extensions = [self._get_extension(n) for n in inner_names]

        has_base_rar = any(ext == '.rar' for ext in inner_extensions)
        has_split_parts = any(re.match(r'\.[rs]\d{2}$', ext) for ext in inner_extensions)
        all_rar_split = all(ext == '.rar' or re.match(r'\.[rs]\d{2}$', ext)
                           for ext in inner_extensions if ext)

        logger.info(f"[ADAPTIVE] Inner files: has_base_rar={has_base_rar}, "
                   f"has_split_parts={has_split_parts}, all_rar_split={all_rar_split}")

        # If inner files are a RAR split, extract them
        if (has_base_rar or has_split_parts) and all_rar_split and len(inner_files) > 1:
            total_inner_size = sum(len(d) for d in inner_files.values())
            logger.info(f"[ADAPTIVE] Inner files form old-style RAR split ({len(inner_files)} parts, "
                       f"{total_inner_size / (1024*1024):.1f} MB), extracting...")

            try:
                # Extract the inner RAR split - 100% RAM
                final_extracted = extract_multipart_rar_from_memory(
                    inner_files,
                    password=self.password
                )

                if final_extracted:
                    logger.info(f"[ADAPTIVE] Inner RAR extraction complete: "
                               f"{len(final_extracted)} final files "
                               f"({sum(len(d) for d in final_extracted.values()) / (1024*1024):.1f} MB)")
                    return final_extracted
                else:
                    logger.warning("[ADAPTIVE] Inner RAR extraction returned no files")
                    # Fall through to return inner_files as-is

            except Exception as e:
                logger.error(f"[ADAPTIVE] Inner RAR extraction failed: {e}")
                # Fall through to return inner_files as-is

        # If not a RAR split or extraction failed, check for other archive types
        # or return inner files directly
        final_result = {}
        archive_files = {}

        for name, data in inner_files.items():
            ext = self._get_extension(name)
            is_archive = False

            # Check by extension
            if ext == '.rar' or re.match(r'\.[rs]\d{2}$', ext):
                is_archive = True
            elif ext in {'.zip', '.7z'}:
                is_archive = True
            # Check by magic
            elif len(data) >= 4:
                if data[:4] == self.RAR_MAGIC:
                    is_archive = True
                elif data[:4] in (self.ZIP_MAGIC_1, self.ZIP_MAGIC_2):
                    is_archive = True
                elif len(data) >= 6 and data[:6] == self.SEVENZ_MAGIC:
                    is_archive = True

            if is_archive:
                archive_files[name] = data
            else:
                final_result[name] = data

        # If we have archives but they're not a proper set, store for later
        if archive_files and not final_result:
            logger.info(f"[ADAPTIVE] {len(archive_files)} archive files not forming a set, "
                       f"returning as-is for further processing")
            return archive_files

        if final_result:
            return final_result

        return inner_files

    def _extract_obfuscated_individual_to_disk(self, rar_parts: Dict[str, bytes]) -> int:
        """
        Extract obfuscated releases where each file is an INDIVIDUAL RAR archive.
        Extracts directly to disk for maximum speed.

        Handles scene releases like:
        - 61 obfuscated files, each is a separate RAR
        - Each outer RAR contains ONE inner file (.rar, .r00-.r99, .s00-.s99)
        - Inner files form a complete old-style RAR split

        Args:
            rar_parts: Dict of filename -> bytes (outer RAR data)

        Returns:
            Number of files extracted to disk
        """
        from src.core.ram_rar import RamRarExtractor, extract_multipart_rar_to_disk
        import tempfile
        import shutil

        logger.info(f"[ADAPTIVE] Extracting {len(rar_parts)} individual outer RARs to disk...")

        # Step 1: Extract each outer RAR individually to collect inner files
        inner_files: Dict[str, bytes] = {}
        outer_extracted = 0
        outer_failed = 0

        for outer_name, outer_data in rar_parts.items():
            try:
                extractor = RamRarExtractor()
                if not extractor.set_archive_data(outer_data):
                    outer_failed += 1
                    continue

                # Try WITHOUT password first (outer RARs often unprotected)
                extracted = extractor.extract_all()

                if not extracted and self.password:
                    extractor2 = RamRarExtractor()
                    extractor2.set_archive_data(outer_data)
                    extractor2.set_password(self.password)
                    extracted = extractor2.extract_all()

                if extracted:
                    outer_extracted += 1
                    for inner_name, inner_data in extracted.items():
                        final_name = inner_name
                        if final_name in inner_files:
                            base = Path(inner_name).stem
                            ext = Path(inner_name).suffix
                            counter = 1
                            while final_name in inner_files:
                                final_name = f"{base}_{counter}{ext}"
                                counter += 1
                        inner_files[final_name] = inner_data
                else:
                    outer_failed += 1

            except Exception as e:
                outer_failed += 1
                logger.debug(f"[ADAPTIVE] Exception extracting {outer_name}: {e}")

        logger.info(f"[ADAPTIVE] Outer RAR results: {outer_extracted} OK, {outer_failed} failed, "
                   f"collected {len(inner_files)} inner files")

        if not inner_files:
            return 0

        # Step 2: Check if inner files form a RAR split
        inner_extensions = [self._get_extension(n) for n in inner_files.keys()]
        has_base_rar = any(ext == '.rar' for ext in inner_extensions)
        has_split_parts = any(re.match(r'\.[rs]\d{2}$', ext) for ext in inner_extensions)
        all_rar_split = all(ext == '.rar' or re.match(r'\.[rs]\d{2}$', ext)
                           for ext in inner_extensions if ext)

        # If inner files are a RAR split, extract them directly to disk
        if (has_base_rar or has_split_parts) and all_rar_split and len(inner_files) > 1:
            total_inner_size = sum(len(d) for d in inner_files.values())
            logger.info(f"[ADAPTIVE] Inner files form RAR split ({len(inner_files)} parts, "
                       f"{total_inner_size / (1024*1024):.1f} MB), extracting to disk...")

            try:
                files_extracted = extract_multipart_rar_to_disk(
                    inner_files,
                    self.extract_dir,
                    password=self.password or ""
                )
                if files_extracted > 0:
                    logger.info(f"[ADAPTIVE] Inner RAR extraction complete: {files_extracted} files to disk")
                    return files_extracted
            except Exception as e:
                logger.error(f"[ADAPTIVE] Inner RAR extraction failed: {e}")

        # Not a RAR split - write non-archive files directly to disk
        files_written = 0
        for name, data in inner_files.items():
            ext = self._get_extension(name)
            is_archive = (ext == '.rar' or re.match(r'\.[rs]\d{2}$', ext) or
                         ext in {'.zip', '.7z'} or
                         (len(data) >= 4 and data[:4] in (self.RAR_MAGIC, self.ZIP_MAGIC_1)))

            if not is_archive:
                dest_path = self.extract_dir / name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(data)
                files_written += 1

        return files_written

    def _execute_7z_extraction(self, stage: ExtractionStage) -> StageResult:
        """Extract using 7-Zip CLI."""
        import subprocess
        import shutil

        # Find 7z executable
        sevenzip_path = self._find_7z()
        if not sevenzip_path:
            return StageResult(errors=["7-Zip not found"])

        # Need to write archives to temp for 7z CLI
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="dler_7z_"))

        try:
            # Write archives to temp
            archives_written = 0
            for key, ram_file in self.ram_buffer.get_all_files().items():
                if self._matches_pattern(ram_file.filename, stage.input_pattern):
                    if self._is_archive_by_magic(ram_file):
                        ram_file.data.seek(0)
                        read_size = ram_file.actual_size or ram_file.size
                        temp_path = temp_dir / ram_file.filename
                        temp_path.write_bytes(ram_file.data.read(read_size))
                        archives_written += 1

            if archives_written == 0:
                return StageResult()

            # Find first archive to extract
            archives = list(temp_dir.glob('*'))
            first_archive = archives[0] if archives else None

            if not first_archive:
                return StageResult()

            # Run 7z
            cmd = [
                sevenzip_path, 'x', '-y', '-bb0', '-bd',
                f'-o{self.extract_dir}'
            ]
            if self.password:
                cmd.append(f'-p{self.password}')
            else:
                cmd.append('-p')
            cmd.append(str(first_archive))

            import sys
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=3600, creationflags=creationflags
            )

            if result.returncode == 0:
                # Count extracted files
                extracted = list(self.extract_dir.rglob('*'))
                extracted = [f for f in extracted if f.is_file()]

                found_types = self._analyze_directory(self.extract_dir)

                return StageResult(
                    files_extracted=len(extracted),
                    found_types=found_types,
                    needs_adaptation='rar' in found_types or '7z' in found_types
                )
            else:
                return StageResult(errors=[f"7z failed: {result.stderr}"])

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _adapt_plan(self, result: StageResult, next_stage_num: int) -> List[ExtractionStage]:
        """Create additional stages for nested archives found."""
        new_stages = []

        if 'rar' in result.found_types:
            new_stages.append(ExtractionStage(
                stage_number=next_stage_num,
                tool=ExtractionTool.UNRAR_DLL,
                input_pattern="*.rar",
                output_type="content",
                keep_in_ram=False,
                description="Extract nested RAR archives"
            ))

        if '7z' in result.found_types:
            new_stages.append(ExtractionStage(
                stage_number=next_stage_num + len(new_stages),
                tool=ExtractionTool.SEVENZIP_CLI,
                input_pattern="*.7z",
                output_type="content",
                keep_in_ram=False,
                description="Extract nested 7z archives"
            ))

        return new_stages

    def _parse_zip_central_dir_for_order(
        self, central_dir_data: bytes, num_parts: int
    ) -> Optional[Dict[int, int]]:
        """
        Parse ZIP central directory to extract disk number information.

        In split ZIP archives, each file entry in the central directory
        contains a "disk number start" field indicating which volume
        contains the start of that file.

        Returns:
            Dict mapping disk_number -> cumulative_offset_before_this_disk
            or None if parsing fails
        """
        import struct

        # Find End of Central Directory record
        eocd_sig = b'PK\x05\x06'
        eocd_pos = central_dir_data.rfind(eocd_sig)
        if eocd_pos == -1:
            return None

        try:
            # EOCD structure (22 bytes minimum):
            # 4 bytes: signature
            # 2 bytes: disk number
            # 2 bytes: disk with central directory start
            # 2 bytes: entries on this disk
            # 2 bytes: total entries
            # 4 bytes: central directory size
            # 4 bytes: offset of central directory
            # 2 bytes: comment length
            eocd = central_dir_data[eocd_pos:eocd_pos + 22]
            if len(eocd) < 22:
                return None

            (_, disk_num, cd_start_disk, entries_on_disk, total_entries,
             cd_size, cd_offset, comment_len) = struct.unpack('<IHHHHIIH', eocd)

            logger.debug(f"[ZIP CENTRAL] Total entries: {total_entries}, "
                        f"CD offset: {cd_offset}, CD size: {cd_size}, "
                        f"Total disks: {disk_num + 1}")

            # For split archives, cd_offset is relative to the start of the
            # archive (disk 0), not the current file

            # Find central directory entries
            cd_start = central_dir_data.find(b'PK\x01\x02')
            if cd_start == -1:
                return None

            # Parse entries to get disk numbers
            disk_info: Dict[int, List[int]] = {}  # disk -> list of file sizes
            pos = cd_start

            for _ in range(total_entries):
                if pos + 46 > len(central_dir_data):
                    break

                sig = central_dir_data[pos:pos + 4]
                if sig != b'PK\x01\x02':
                    break

                # Central directory file header structure
                # Bytes 34-35: disk number start (where file data begins)
                entry_header = central_dir_data[pos:pos + 46]
                (_, version_made, version_needed, flags, compression,
                 mod_time, mod_date, crc32, comp_size, uncomp_size,
                 name_len, extra_len, comment_len, disk_start,
                 internal_attr, external_attr, local_header_offset
                ) = struct.unpack('<IHHHHHHIIIHHHHHII', entry_header)

                if disk_start not in disk_info:
                    disk_info[disk_start] = []
                disk_info[disk_start].append(comp_size)

                # Move to next entry
                pos += 46 + name_len + extra_len + comment_len

            if not disk_info:
                return None

            logger.debug(f"[ZIP CENTRAL] Disk distribution: "
                        f"{dict((k, len(v)) for k, v in disk_info.items())}")

            # Calculate cumulative sizes per disk
            disk_sizes: Dict[int, int] = {}
            for disk_num in sorted(disk_info.keys()):
                disk_sizes[disk_num] = sum(disk_info[disk_num])

            return disk_sizes

        except Exception as e:
            logger.debug(f"[ZIP CENTRAL] Parse error: {e}")
            return None

    def _reorder_zip_parts_by_size(
        self,
        zip_parts: List[Tuple[str, bytes]],
        disk_sizes: Dict[int, int]
    ) -> Optional[List[Tuple[str, bytes]]]:
        """
        Try to reorder ZIP parts based on expected sizes from central directory.

        This is a heuristic approach - we try to match part files to disk numbers
        based on their sizes.
        """
        if not disk_sizes or not zip_parts:
            return None

        # Get unique disk numbers in order
        disk_nums = sorted(disk_sizes.keys())

        # If number of disks doesn't match parts, we can't reliably reorder
        if len(disk_nums) > len(zip_parts):
            logger.debug(f"[ZIP ORDER] Disk count mismatch: {len(disk_nums)} disks vs {len(zip_parts)} parts")
            return None

        # For obfuscated names, try to detect if there's a pattern
        # Check if names end with hex-like characters
        part_names = [name for name, _ in zip_parts]

        # Try to extract last character before .zip as order hint
        def extract_order_char(name: str) -> str:
            """Extract the ordering character from obfuscated name."""
            base = name.lower().replace('.zip', '')
            if base:
                return base[-1]
            return ''

        # Check if all order chars are unique (suggesting they encode order)
        order_chars = [extract_order_char(name) for name in part_names]
        unique_chars = set(order_chars)

        if len(unique_chars) == len(order_chars):
            # All unique - might be base36 encoding
            # Try sorting by this character (0-9, a-z order)
            def base36_value(c: str) -> int:
                if c.isdigit():
                    return int(c)
                elif c.isalpha():
                    return 10 + ord(c.lower()) - ord('a')
                return 999

            reordered = sorted(zip_parts, key=lambda x: base36_value(extract_order_char(x[0])))
            logger.info("[ZIP ORDER] Reordered using base36 character detection")
            return reordered

        # If sizes vary significantly, try to match by size
        part_sizes = [(name, len(data), data) for name, data in zip_parts]

        # Most split archives have equal-sized parts except possibly the last
        sizes = [s for _, s, _ in part_sizes]
        if len(set(sizes)) <= 2:  # All same size or one different
            # Can't distinguish by size, fall back to name sort
            return None

        logger.debug("[ZIP ORDER] Sizes vary too much for reliable size-based ordering")
        return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches glob pattern."""
        if pattern == "*":
            return True

        import fnmatch
        filename_lower = filename.lower()
        pattern_lower = pattern.lower()

        # Special case: *.rar also matches old-style RAR splits (.r00-.r99, .s00-.s99)
        if pattern_lower == "*.rar":
            if fnmatch.fnmatch(filename_lower, "*.rar"):
                return True
            # Check for old-style RAR splits
            if re.search(r'\.[rs]\d{2}$', filename_lower):
                return True
            return False

        return fnmatch.fnmatch(filename_lower, pattern_lower)

    def _matches_rar(self, name: str, data: bytes) -> bool:
        """Check if file is RAR by name or magic."""
        if name.lower().endswith('.rar'):
            return True
        if len(data) >= 4 and data[:4] == self.RAR_MAGIC:
            return True
        return False

    def _is_rar_file(self, ram_file: 'RamFile') -> bool:
        """Check if RamFile is RAR."""
        if ram_file.filename.lower().endswith('.rar'):
            return True
        return self._is_rar_by_magic(ram_file)

    def _is_rar_by_magic(self, ram_file: 'RamFile') -> bool:
        """Detect RAR by magic bytes."""
        ram_file.data.seek(0)
        header = ram_file.data.read(4)
        ram_file.data.seek(0)
        return header == self.RAR_MAGIC

    def _is_rar_data(self, data: bytes) -> bool:
        """Check if bytes are RAR data."""
        return len(data) >= 4 and data[:4] == self.RAR_MAGIC

    def _is_zip_by_magic(self, ram_file: 'RamFile') -> bool:
        """Detect ZIP by magic bytes."""
        ram_file.data.seek(0)
        header = ram_file.data.read(4)
        ram_file.data.seek(0)
        return header == self.ZIP_MAGIC_1 or header == self.ZIP_MAGIC_2

    def _is_par2_by_magic(self, ram_file: 'RamFile') -> bool:
        """Detect PAR2 by magic bytes."""
        ram_file.data.seek(0)
        header = ram_file.data.read(4)
        ram_file.data.seek(0)
        return header == self.PAR2_MAGIC

    def _is_archive_by_magic(self, ram_file: 'RamFile') -> bool:
        """Check if file is any archive type by magic."""
        ram_file.data.seek(0)
        header = ram_file.data.read(8)
        ram_file.data.seek(0)

        if header[:4] == self.RAR_MAGIC:
            return True
        if header[:4] == self.ZIP_MAGIC_1 or header[:4] == self.ZIP_MAGIC_2:
            return True
        if header[:6] == self.SEVENZ_MAGIC:
            return True
        return False

    def _get_extension(self, filename: str) -> str:
        """Get lowercase extension."""
        if '.' not in filename:
            return ''
        return '.' + filename.rsplit('.', 1)[-1].lower()

    def _analyze_directory(self, directory: Path) -> Set[str]:
        """Analyze directory contents to find archive types."""
        types: Set[str] = set()

        for f in directory.rglob('*'):
            if not f.is_file():
                continue

            ext = f.suffix.lower()
            # Match .rar and old-style splits: .r00-.r99, .s00-.s99
            if ext == '.rar' or re.match(r'\.[rs]\d{2}$', ext):
                types.add('rar')
            elif ext == '.7z':
                types.add('7z')
            elif ext == '.zip':
                types.add('zip')
            elif ext in {'.mkv', '.mp4', '.avi', '.iso', '.m2ts'}:
                types.add('media')
            else:
                # Check magic for extensionless files
                try:
                    with open(f, 'rb') as fp:
                        header = fp.read(8)
                    if header[:4] == self.RAR_MAGIC:
                        types.add('rar')
                    elif header[:6] == self.SEVENZ_MAGIC:
                        types.add('7z')
                except:
                    pass

        return types

    def _find_7z(self) -> Optional[str]:
        """Find 7-Zip executable."""
        import shutil

        # Check bundled tools
        bundled = Path(__file__).parent.parent.parent / 'tools' / '7z.exe'
        if bundled.exists():
            return str(bundled)

        # Check PATH
        found = shutil.which('7z')
        if found:
            return found

        # Check common locations
        locations = [
            Path(r'C:\Program Files\7-Zip\7z.exe'),
            Path(r'C:\Program Files (x86)\7-Zip\7z.exe'),
        ]
        for loc in locations:
            if loc.exists():
                return str(loc)

        return None

    def _report_progress(self, message: str, percentage: float) -> None:
        """Report progress if callback is set."""
        if self.on_progress:
            self.on_progress(message, percentage)
