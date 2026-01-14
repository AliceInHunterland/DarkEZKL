import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class _UTCFormatter(logging.Formatter):
    # Ensure our "...Z" timestamp is actually UTC.
    converter = time.gmtime


def setup_logging(level: Union[int, str] = logging.INFO) -> None:
    """
    Configure logging to stdout with UTC timestamps.
    """
    if isinstance(level, str):
        level = level.strip().upper()
        level = getattr(logging, level, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _UTCFormatter(
            fmt="%(asctime)sZ [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    # Remove existing handlers to avoid duplication
    if root.handlers:
        root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _meminfo_kib() -> Dict[str, int]:
    """
    Parse /proc/meminfo into KiB ints.
    """
    out: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # e.g. "MemAvailable:   123456 kB"
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    key = parts[0].rstrip(":")
                    out[key] = int(parts[1])
    except Exception:
        return {}
    return out


def get_process_memory_stats() -> Dict[str, Any]:
    """
    Parse /proc/self/status for process memory diagnostics (no psutil dependency).
    VmRSS: current resident set size.
    VmHWM: peak resident set size ("high water mark").
    Values are returned in MiB where possible.
    """
    stats: Dict[str, Any] = {"vmrss_mb": None, "vmhwm_mb": None}
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        stats["vmrss_mb"] = int(parts[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        stats["vmhwm_mb"] = int(parts[1]) / 1024.0
    except Exception:
        pass
    return stats


def get_system_stats(paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    """
    Lightweight system stats (no psutil dependency).
    Useful for debugging OOM (exit code 137) and disk pressure.
    """
    mi = _meminfo_kib()
    stats: Dict[str, Any] = {
        "mem_total_mb": (mi.get("MemTotal", 0) / 1024.0) if mi else None,
        "mem_available_mb": (mi.get("MemAvailable", 0) / 1024.0) if mi else None,
        "swap_total_mb": (mi.get("SwapTotal", 0) / 1024.0) if mi else None,
        "swap_free_mb": (mi.get("SwapFree", 0) / 1024.0) if mi else None,
        "process": get_process_memory_stats(),
    }

    paths = paths or [Path("/app")]
    disks: Dict[str, Any] = {}
    for p in paths:
        try:
            st = os.statvfs(str(p))
            free = (st.f_bavail * st.f_frsize) / (1024.0 * 1024.0)
            total = (st.f_blocks * st.f_frsize) / (1024.0 * 1024.0)
            disks[str(p)] = {"disk_free_mb": free, "disk_total_mb": total}
        except Exception:
            disks[str(p)] = {"disk_free_mb": None, "disk_total_mb": None}
    stats["disk"] = disks
    return stats


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    Writes JSON to file with explicit fsync to ensure data is on disk
    before subsequent Rust calls try to read it.
    """
    ensure_dir(path.parent)
    try:
        with path.open("w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        # Fallback logging if disk is full or path invalid
        logging.getLogger(__name__).error(f"Failed to write JSON to {path}: {e}")


def write_json_compact(path: Path, obj: Dict[str, Any]) -> None:
    """
    Compact JSON writer (smaller + faster for huge arrays like input.json).
    Still fsync()'d for safety with downstream Rust reads.
    """
    ensure_dir(path.parent)
    try:
        with path.open("w") as f:
            json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to write compact JSON to {path}: {e}")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def cat_file(path: Path, max_lines: int = 100, header: str = "") -> None:
    """
    Logs the content of a file to stdout for debugging purposes.
    Limits output to `max_lines`.
    """
    logger = logging.getLogger(__name__)
    if not path.exists():
        logger.warning(f"Cannot cat file (not found): {path}")
        return

    logger.info(f"--- BEGIN FILE CONTENT: {header} ({path}) ---")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    logger.info(f"... (truncated after {max_lines} lines) ...")
                    break
                logger.info(f"{i+1:04d}| {line.rstrip()}")
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
    logger.info(f"--- END FILE CONTENT: {path} ---")


@contextmanager
def timed(description: str = "Operation") -> Dict[str, Optional[float]]:
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    holder: Dict[str, Optional[float]] = {"elapsed": None}
    try:
        yield holder
    finally:
        elapsed = time.perf_counter() - start
        holder["elapsed"] = elapsed
        logger.info(f"{description} took {elapsed:.4f}s")


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def run_command(cmd: List[str], timeout_s: int = 15) -> Dict[str, Any]:
    """
    Run a subprocess and return structured output (useful for diagnostics in JSON).
    Never raises (returns a payload with error instead).
    """
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": (p.stdout or "").strip(),
            "stderr": (p.stderr or "").strip(),
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
        }


def run_subprocess_streaming(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
    tail_lines: int = 200,
    line_prefix: str = "",
) -> Dict[str, Any]:
    """
    Run a subprocess, stream combined stdout/stderr to our stdout, and keep the last N lines.
    Useful when a child is killed by OOM (exit 137) so you still get a tail for debugging.
    """
    start = time.perf_counter()
    tail: deque[str] = deque(maxlen=max(1, int(tail_lines)))

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    try:
        assert p.stdout is not None
        for line in p.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if line_prefix:
                sys.stdout.write(f"{line_prefix}{line}\n")
            else:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        rc = p.wait(timeout=timeout_s)
        elapsed = time.perf_counter() - start
        return {"returncode": rc, "elapsed_s": elapsed, "tail": list(tail), "cmd": cmd}
    except subprocess.TimeoutExpired:
        p.kill()
        elapsed = time.perf_counter() - start
        return {"returncode": None, "elapsed_s": elapsed, "tail": list(tail), "cmd": cmd, "timeout": True}

