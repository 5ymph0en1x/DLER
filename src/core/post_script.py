"""Run a user-configured command after a download completes (automation hook).

The command runs as an isolated subprocess with DLER_* context injected into
the environment. Failures are logged but NEVER affect the download result.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Tuple

logger = logging.getLogger(__name__)


def run_post_script(command: str, context: dict, timeout_s: int = 300) -> Tuple[bool, str]:
    """Run `command` (a full shell command line) with DLER_* env vars from
    `context`. Returns (ok, message). Never raises."""
    if not command or not command.strip():
        return False, "no command configured"

    env = os.environ.copy()
    env["DLER_RELEASE"] = str(context.get("release", ""))
    env["DLER_NZB"] = str(context.get("nzb", ""))
    env["DLER_OUTPUT_DIR"] = str(context.get("output_dir", ""))
    env["DLER_EXTRACT_DIR"] = str(context.get("extract_dir", ""))
    env["DLER_SUCCESS"] = "1" if context.get("success") else "0"
    env["DLER_REPAIRED"] = "1" if context.get("repaired") else "0"
    env["DLER_FILES_EXTRACTED"] = str(int(context.get("files_extracted", 0) or 0))
    env["DLER_DOWNLOAD_BYTES"] = str(int(context.get("download_bytes", 0) or 0))
    env["DLER_DURATION_S"] = str(int(context.get("duration_s", 0) or 0))
    env["DLER_AVG_SPEED_MBPS"] = str(context.get("avg_speed_mbps", 0) or 0)
    env["DLER_SEGMENTS_DONE"] = str(int(context.get("segments_done", 0) or 0))
    env["DLER_SEGMENTS_TOTAL"] = str(int(context.get("segments_total", 0) or 0))
    env["DLER_CONNECTIONS"] = str(int(context.get("connections", 0) or 0))
    env["DLER_NUM_PROCESSES"] = str(int(context.get("num_processes", 1) or 1))

    try:
        proc = subprocess.run(
            command, shell=True, env=env, timeout=timeout_s,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[POST-SCRIPT] timed out after {timeout_s}s: {command}")
        return False, f"timed out after {timeout_s}s"
    except Exception as e:  # launch failure (bad command, etc.)
        logger.warning(f"[POST-SCRIPT] failed to launch: {e}")
        return False, f"failed to launch: {e}"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if out:
        logger.info(f"[POST-SCRIPT] stdout: {out[:1000]}")
    if err:
        logger.info(f"[POST-SCRIPT] stderr: {err[:1000]}")
    if proc.returncode != 0:
        logger.warning(f"[POST-SCRIPT] exit code {proc.returncode}: {command}")
        return False, f"exit code {proc.returncode}"
    return True, "ok"
