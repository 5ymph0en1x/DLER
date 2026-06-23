"""
Multiprocess download engine
============================

Escapes the CPython GIL by sharding an NZB's *files* across worker processes.
Files in an NZB are independent, so each process downloads a disjoint subset and
writes its own complete output files to the shared output directory -- there is
NO cross-process transfer of segment bytes and NO shared mmap/RAM buffer, which
is what makes this both correct and actually faster (pickling 1.25 GB/s of
segment data across processes would defeat the purpose).

Constraints:
- Disk mode only. RAM mode keeps everything in one process's memory, so it is
  not shareable across processes; callers should fall back to the single-process
  engine for RAM mode.
- PAR2 verification / post-processing runs in the MAIN process after all shards
  finish (workers run with incremental_verify=False).
- The provider connection cap is split across processes (sum stays <= cap).

This module is self-contained and importable; throughput must be validated on a
live connection.
"""

from __future__ import annotations

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


def partition_files(items: List[Tuple[int, int]], n: int) -> List[List[int]]:
    """Greedy longest-processing-time bin-packing of files into ``n`` shards,
    balanced by byte size.

    Args:
        items: list of (file_index, size_bytes).
        n: number of shards.

    Returns:
        A list of ``n`` lists of file indices (some may be empty). Largest files
        are placed first onto the currently-least-loaded shard.
    """
    n = max(1, n)
    shards: List[List[int]] = [[] for _ in range(n)]
    loads = [0] * n
    for idx, size in sorted(items, key=lambda t: t[1], reverse=True):
        k = min(range(n), key=lambda j: loads[j])
        shards[k].append(idx)
        loads[k] += max(0, size)
    return shards


def _src_dir() -> str:
    """Absolute path of the project's ``src`` dir (parent of ``core``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_engine_classes():
    """Import the engine classes in a way that works BOTH from source
    (``core.*`` with ``src/`` on sys.path) AND from a frozen PyInstaller bundle
    (where the modules are packaged under ``src.core.*``). Without this, MP child
    processes in the exe crash with ``ModuleNotFoundError: No module named 'core'``.
    """
    try:
        from src.core.fast_nntp import ServerConfig
        from src.core.turbo_engine_v2 import TurboEngineV2
    except ImportError:
        from core.fast_nntp import ServerConfig
        from core.turbo_engine_v2 import TurboEngineV2
    return ServerConfig, TurboEngineV2


def _parse_files_meta(nzb_path: str, src_dir: str) -> List[Tuple[int, int]]:
    """Parse the NZB once (in the main process) to get (index, size) per file."""
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    ServerConfig, TurboEngineV2 = _load_engine_classes()
    eng = TurboEngineV2(ServerConfig(host="parse.local"), output_dir=Path(nzb_path).parent)
    files = eng._parse_nzb(Path(nzb_path))
    return [(f.index, f.size) for f in files]


# Per-shard row layout in the shared stats array:
#   0 bytes_downloaded  1 segments_downloaded  2 segments_total  3 errors
#   4 connections(active)  5 raw_queue depth  6 write_queue depth
_STATS_SLOTS = 7


def _aggregate(stats_array, n: int) -> Dict[str, int]:
    """Sum the per-shard rows into a single aggregate dict."""
    agg = {"bytes_downloaded": 0, "segments_downloaded": 0,
           "segments_total": 0, "errors": 0,
           "connections": 0, "raw_queue": 0, "write_queue": 0}
    for sid in range(n):
        base = sid * _STATS_SLOTS
        agg["bytes_downloaded"] += stats_array[base + 0]
        agg["segments_downloaded"] += stats_array[base + 1]
        agg["segments_total"] += stats_array[base + 2]
        agg["errors"] += stats_array[base + 3]
        agg["connections"] += stats_array[base + 4]
        agg["raw_queue"] += stats_array[base + 5]
        agg["write_queue"] += stats_array[base + 6]
    return agg


def _mp_worker(src_dir: str, nzb_path: str, server_dict: dict, output_dir: str,
               file_indices: List[int], connections: int,
               turbo_kwargs: Optional[dict], stats_array, shard_id: int,
               stop_event) -> None:
    """Child-process entry point: download this shard's files. Exit code 0 on
    success, 1 on failure (read by the dispatcher)."""
    import threading
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    ServerConfig, TurboEngineV2 = _load_engine_classes()

    sc = ServerConfig(**server_dict)
    sc.connections = connections
    base = shard_id * _STATS_SLOTS

    def on_progress(stats) -> None:
        try:
            stats_array[base + 0] = int(stats.bytes_downloaded)
            stats_array[base + 1] = int(stats.segments_downloaded)
            stats_array[base + 2] = int(stats.segments_total)
            stats_array[base + 3] = int(stats.errors)
            # Live connection + queue depth — the main process can't see these
            # (they live in this child), so report them through the shared array.
            try:
                stats_array[base + 4] = sum(1 for c in eng._connections if c and c.socket)
                stats_array[base + 5] = eng._raw_queue.qsize()
                stats_array[base + 6] = eng._write_queue.qsize()
            except Exception:
                pass
        except Exception:
            pass

    eng = TurboEngineV2(sc, output_dir=Path(output_dir), on_progress=on_progress,
                        incremental_verify=False, **(turbo_kwargs or {}))
    # Cross-process stop: when the dispatcher sets the event, stop this engine.
    threading.Thread(target=lambda: (stop_event.wait(), eng.stop()), daemon=True).start()

    ok = False
    try:
        ok = eng.download_nzb(Path(nzb_path), only_file_indices=set(file_indices))
    finally:
        try:
            eng.disconnect()
        except Exception:
            pass
    sys.exit(0 if ok else 1)


def download_multiprocess(
    nzb_path,
    server_config,
    output_dir,
    num_processes: int = 0,
    turbo_kwargs: Optional[dict] = None,
    progress_cb: Optional[Callable[[Dict[str, int]], None]] = None,
    poll_interval: float = 0.25,
    stop_event=None,
) -> bool:
    """Run a file-sharded multiprocess download.

    Args:
        nzb_path: path to the .nzb.
        server_config: a fast_nntp.ServerConfig (its .connections is the cap,
            split across processes).
        output_dir: shared output directory (each process writes disjoint files).
        num_processes: 0 = auto (cpu_count, capped by file count).
        turbo_kwargs: extra kwargs forwarded to each per-process TurboEngineV2
            (e.g. pipeline_depth, decoder_threads). Do NOT enable RAM mode.
        progress_cb: called ~every poll_interval with an aggregate stats dict.
        stop_event: optional pre-created multiprocessing Event to request stop;
            if None one is created internally.

    Returns:
        True if every shard exited successfully.
    """
    nzb_path = str(nzb_path)
    output_dir = str(output_dir)
    src_dir = _src_dir()

    files_meta = _parse_files_meta(nzb_path, src_dir)
    if not files_meta:
        return False

    auto = num_processes if num_processes and num_processes > 0 else (os.cpu_count() or 2)
    n = max(1, min(auto, len(files_meta)))
    shards = [s for s in partition_files(files_meta, n) if s]
    n = len(shards)

    total_conns = max(1, getattr(server_config, "connections", 1))
    per_proc = max(1, total_conns // n)

    server_dict = dict(
        host=server_config.host, port=server_config.port,
        username=server_config.username, password=server_config.password,
        use_ssl=server_config.use_ssl, connections=per_proc,
        timeout=server_config.timeout,
    )

    ctx = mp.get_context("spawn")
    stats_array = ctx.Array("q", n * _STATS_SLOTS)  # zero-initialised
    own_stop = stop_event is None
    if own_stop:
        stop_event = ctx.Event()

    procs = []
    for sid, idxs in enumerate(shards):
        p = ctx.Process(
            target=_mp_worker,
            args=(src_dir, nzb_path, server_dict, output_dir, idxs, per_proc,
                  turbo_kwargs, stats_array, sid, stop_event),
            name=f"DLER-MP-{sid}",
            daemon=False,
        )
        p.start()
        procs.append(p)

    stopped = False
    try:
        while any(p.is_alive() for p in procs):
            if stop_event.is_set():
                stopped = True
                for p in procs:
                    if p.is_alive():
                        p.terminate()      # hard-stop the shard immediately
                break
            time.sleep(poll_interval)
            if progress_cb:
                progress_cb(_aggregate(stats_array, n))
        if progress_cb:
            progress_cb(_aggregate(stats_array, n))
    except KeyboardInterrupt:
        stopped = True
        stop_event.set()
        for p in procs:
            if p.is_alive():
                p.terminate()

    for p in procs:
        p.join()

    return (not stopped) and all(p.exitcode == 0 for p in procs)
