"""Core engine components for DLER."""

# Lazy imports to avoid numpy dependency when using native C++ decoder

__all__ = [
    "DownloadEngine",
    "NNTPConnectionPool",
    "NNTPConnection",
    "NNTPConfig",
    "NZBParser",
    "NZBFile",
    "NZBSegment",
    "YEncDecoder",
    "FileAssembler",
    "TurboEngineV2",
]


def __getattr__(name):
    """Lazy import modules to avoid numpy dependency."""
    if name == 'DownloadEngine':
        from .engine import DownloadEngine
        return DownloadEngine
    elif name == 'NNTPConnectionPool':
        from .nntp import NNTPConnectionPool
        return NNTPConnectionPool
    elif name == 'NNTPConnection':
        from .nntp import NNTPConnection
        return NNTPConnection
    elif name == 'NNTPConfig':
        from .nntp import NNTPConfig
        return NNTPConfig
    elif name == 'NZBParser':
        from .nzb_parser import NZBParser
        return NZBParser
    elif name == 'NZBFile':
        from .nzb_parser import NZBFile
        return NZBFile
    elif name == 'NZBSegment':
        from .nzb_parser import NZBSegment
        return NZBSegment
    elif name == 'YEncDecoder':
        from .yenc import YEncDecoder
        return YEncDecoder
    elif name == 'FileAssembler':
        from .assembler import FileAssembler
        return FileAssembler
    elif name == 'TurboEngineV2':
        from .turbo_engine_v2 import TurboEngineV2
        return TurboEngineV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
