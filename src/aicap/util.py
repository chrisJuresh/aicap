from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import CaptionItem


def die(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def clean_text(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip('"').strip()


def seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    millis = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    if millis == 1000:
        total += 1
        millis = 0
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    return seconds_to_srt_time(seconds).replace(",", ".")


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    except FileNotFoundError:
        die(f"Command not found: {cmd[0]}. Install it or add it to PATH.")
    except subprocess.CalledProcessError as e:
        print(e.stdout or "", file=sys.stderr)
        print(e.stderr or "", file=sys.stderr)
        die(f"Command failed: {' '.join(cmd)}")


def require_exe(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        die(f"{name} was not found on PATH. Run install_windows.ps1 or install {name} manually.")
    return exe


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def json_fingerprint(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except Exception:
        return {"path": str(path.resolve()), "size": None, "mtime_ns": None}


def items_hash(items: List[CaptionItem]) -> str:
    return json_fingerprint([asdict(i) for i in items])
