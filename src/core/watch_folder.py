"""Watchfolder: poll a directory for new *.nzb files and hand each STABLE one
to a callback, then move it into <folder>/_processed/.

Stability: a file is only handed over once its size is unchanged across two
consecutive scans (avoids grabbing a half-copied file). Polling (not the
`watchdog` library) keeps this dependency-free and the stability check trivial.
"""
from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Callable, Dict

logger = logging.getLogger(__name__)


class WatchFolder:
    def __init__(self, folder, on_nzb: Callable[[Path], None], poll_interval_s: int = 5):
        self.folder = Path(folder)
        self.on_nzb = on_nzb
        self.poll_interval_s = max(1, int(poll_interval_s))
        self._stop = threading.Event()
        self._thread = None
        self._sizes: Dict[str, int] = {}  # filename -> last-seen size

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="WatchFolder", daemon=True)
        self._thread.start()
        logger.info(f"[WATCH] started on {self.folder} every {self.poll_interval_s}s")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("[WATCH] stopped")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.scan_once()
            self._stop.wait(self.poll_interval_s)

    def scan_once(self) -> None:
        """One scan pass (no thread/sleep). Picks up files that were the same
        size on the previous pass."""
        try:
            if not self.folder.is_dir():
                return
            current: Dict[str, int] = {}
            for entry in self.folder.iterdir():
                if entry.is_file() and entry.suffix.lower() == ".nzb":
                    try:
                        current[entry.name] = entry.stat().st_size
                    except OSError:
                        continue
            for name, size in current.items():
                prev = self._sizes.get(name)
                if prev is not None and prev == size:
                    self._pickup(self.folder / name)
                    self._sizes.pop(name, None)
                else:
                    self._sizes[name] = size
            for gone in [n for n in self._sizes if n not in current]:
                self._sizes.pop(gone, None)
        except Exception as e:
            logger.warning(f"[WATCH] scan error: {e}")

    def _pickup(self, path: Path) -> None:
        try:
            dest_dir = self.folder / "_processed"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / path.name
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{path.stem}_{counter}{path.suffix}"
                counter += 1
            shutil.move(str(path), str(dest))
            logger.info(f"[WATCH] picked up {path.name} -> {dest.name}")
            self.on_nzb(dest)
        except Exception as e:
            logger.warning(f"[WATCH] pickup failed for {path.name}: {e}")
