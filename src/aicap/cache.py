from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .models import VideoJob
from .util import clean_text, json_fingerprint, read_json, write_json


def resume_enabled(args: Any) -> bool:
    return bool(getattr(args, "resume", True)) and not bool(getattr(args, "force", False))


def resume_dir(job: VideoJob) -> Path:
    return job.out_dir / "_resume"


def checkpoint_path(job: VideoJob, stage: str) -> Path:
    return resume_dir(job) / f"{stage}.json"


def read_checkpoint(job: VideoJob, stage: str) -> Optional[Dict[str, Any]]:
    data = read_json(checkpoint_path(job, stage), None)
    return data if isinstance(data, dict) else None


def write_checkpoint(job: VideoJob, stage: str, signature: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "stage": stage,
        "signature": signature,
        "signature_hash": json_fingerprint(signature),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    write_json(checkpoint_path(job, stage), payload)


def checkpoint_matches(job: VideoJob, stage: str, signature: Dict[str, Any]) -> bool:
    checkpoint = read_checkpoint(job, stage)
    return bool(checkpoint and checkpoint.get("signature_hash") == json_fingerprint(signature))


def load_visual_cache(cache_path: Path) -> Dict[str, str]:
    captions: Dict[str, str] = {}
    if not cache_path.exists():
        return captions
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json

                    row = json.loads(line)
                except Exception:
                    print(f"Warning: skipping corrupt visual cache line in {cache_path}", file=sys.stderr)
                    continue
                frame = str(row.get("frame", ""))
                caption = clean_text(str(row.get("caption", "")))
                if frame and caption:
                    captions[frame] = caption
    except Exception as e:
        print(f"Warning: could not read visual cache {cache_path}: {e}", file=sys.stderr)
        return captions
    return captions
