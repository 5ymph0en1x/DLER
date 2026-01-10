"""
DLER Configuration System
==========================

Persistent configuration with:
- JSON storage
- Environment variable overrides
- Secure credential handling
- Validation
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".dler"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class ServerConfig:
    """NNTP server configuration."""
    host: str = ""
    port: int = 563
    username: str = ""
    password: str = ""
    use_ssl: bool = True
    connections: int = 20
    timeout: int = 30


@dataclass
class DownloadConfig:
    """Download settings."""
    output_dir: str = field(
        default_factory=lambda: str(Path.home() / "Downloads" / "dler")
    )
    temp_dir: str = field(
        default_factory=lambda: str(Path.home() / ".dler" / "temp")
    )
    max_retries: int = 3
    segment_timeout: int = 60
    verify_crc: bool = True
    auto_extract: bool = True
    auto_repair: bool = True
    delete_after_extract: bool = False


@dataclass
class CacheSettings:
    """Cache configuration."""
    enabled: bool = True
    max_size_gb: float = 10.0
    segment_ttl_hours: int = 24
    memory_cache_mb: int = 512


@dataclass
class UIConfig:
    """UI settings."""
    dark_mode: bool = True
    refresh_rate_ms: int = 500
    show_speed_graph: bool = True
    show_connections: bool = True
    compact_mode: bool = False


@dataclass
class TurboConfig:
    """Turbo V2 engine settings."""
    decoder_threads: int = 0  # 0 = auto (CPU count)
    writer_threads: int = 8
    pipeline_depth: int = 20
    write_through: bool = False  # DyMaxIO compatibility mode


@dataclass
class PostProcessConfig:
    """Post-processing settings (PAR2 verification + extraction)."""
    enabled: bool = True
    extract_dir: str = field(
        default_factory=lambda: str(Path.home() / "Downloads" / "dler_extracted")
    )
    par2_verify: bool = True
    par2_repair: bool = True
    auto_extract: bool = True
    cleanup_after_extract: bool = True
    par2_path: str = ""  # Empty = auto-detect
    sevenzip_path: str = ""  # Empty = auto-detect


@dataclass
class Config:
    """Main configuration container."""
    server: ServerConfig = field(default_factory=ServerConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    cache: CacheSettings = field(default_factory=CacheSettings)
    ui: UIConfig = field(default_factory=UIConfig)
    turbo: TurboConfig = field(default_factory=TurboConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """Create from dictionary."""
        return cls(
            server=ServerConfig(**data.get("server", {})),
            download=DownloadConfig(**data.get("download", {})),
            cache=CacheSettings(**data.get("cache", {})),
            ui=UIConfig(**data.get("ui", {})),
            turbo=TurboConfig(**data.get("turbo", {})),
            postprocess=PostProcessConfig(**data.get("postprocess", {})),
            version=data.get("version", "1.0.0")
        )

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of errors."""
        errors = []

        if self.server.port < 1 or self.server.port > 65535:
            errors.append("Invalid port number")

        if self.server.connections < 1 or self.server.connections > 200:
            errors.append("Connections must be between 1 and 200")

        if self.cache.max_size_gb < 0.1:
            errors.append("Cache size must be at least 0.1 GB")

        if self.download.max_retries < 0:
            errors.append("Max retries must be at least 0")

        # Turbo V2 validation
        if self.turbo.writer_threads < 1 or self.turbo.writer_threads > 32:
            errors.append("Writer threads must be between 1 and 32")

        if self.turbo.pipeline_depth < 1 or self.turbo.pipeline_depth > 50:
            errors.append("Pipeline depth must be between 1 and 50")

        return errors


def load_config() -> Config:
    """
    Load configuration from file.
    Falls back to defaults if not found.
    Supports environment variable overrides.
    """
    config = Config()

    # Load from file if exists
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                config = Config.from_dict(data)
                logger.info(f"Loaded config from {CONFIG_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Environment variable overrides
    env_overrides = {
        "DLER_HOST": ("server", "host"),
        "DLER_PORT": ("server", "port", int),
        "DLER_USER": ("server", "username"),
        "DLER_PASS": ("server", "password"),
        "DLER_CONNECTIONS": ("server", "connections", int),
        "DLER_OUTPUT": ("download", "output_dir"),
        "DLER_CACHE_SIZE": ("cache", "max_size_gb", float),
    }

    for env_var, path in env_overrides.items():
        value = os.environ.get(env_var)
        if value:
            section, key = path[0], path[1]
            converter = path[2] if len(path) > 2 else str

            try:
                section_obj = getattr(config, section)
                setattr(section_obj, key, converter(value))
                logger.debug(f"Override from {env_var}: {section}.{key}")
            except Exception as e:
                logger.warning(f"Failed to apply {env_var}: {e}")

    return config


def save_config(config: Config) -> bool:
    """
    Save configuration to file.
    Creates config directory if needed.
    """
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Saved config to {CONFIG_FILE}")
        return True

    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def ensure_directories(config: Config) -> None:
    """Ensure all required directories exist."""
    dirs = [
        Path(config.download.output_dir),
        Path(config.download.temp_dir),
        Path(config.postprocess.extract_dir),
        CONFIG_DIR / "cache",
        CONFIG_DIR / "logs",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
