from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import VideoJob
from .util import die


def parse_extensions(value: str) -> List[str]:
    extensions: List[str] = []
    for part in value.split(","):
        ext = part.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        extensions.append(ext)
    return sorted(set(extensions))


def safe_name(path: Path) -> str:
    name = path.stem.strip() or "video"
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" ._")
    return name or "video"


def discover_video_files(folder: Path, extensions: List[str], recursive: bool) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        die(f"Input folder does not exist: {folder}")
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    videos = [p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in extensions]
    videos = sorted(videos, key=lambda p: str(p).lower())
    if not videos:
        ext_list = ", ".join(extensions)
        die(f"No video files found in {folder}. Checked extensions: {ext_list}")
    return videos


def build_jobs(input_path: Path, out_dir: Path, extensions: List[str], recursive: bool) -> Tuple[List[VideoJob], bool]:
    input_path = input_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if input_path.is_dir():
        videos = discover_video_files(input_path, extensions, recursive)
        used_names: Dict[str, int] = {}
        jobs: List[VideoJob] = []
        for video in videos:
            base = safe_name(video)
            count = used_names.get(base.lower(), 0)
            used_names[base.lower()] = count + 1
            folder_name = base if count == 0 else f"{base}_{count + 1}"
            job_out = out_dir / folder_name
            jobs.append(VideoJob(video_path=video, out_dir=job_out, work_dir=job_out / "_work"))
        return jobs, True

    if not input_path.exists():
        die(f"Input video does not exist: {input_path}")
    return [VideoJob(video_path=input_path, out_dir=out_dir, work_dir=out_dir / "_work")], False


def captioned_video_output_path(job: VideoJob, args: Any, is_batch: bool) -> Optional[Path]:
    if not args.burn_in:
        return None
    if args.burn_output and not is_batch:
        return args.burn_output.expanduser().resolve()
    return job.out_dir / "captioned_video.mp4"


def outputs_are_complete(job: VideoJob, args: Any, is_batch: bool) -> bool:
    if not (job.out_dir / "captions.json").exists():
        return False
    if not (job.out_dir / "captions.srt").exists():
        return False
    if not (job.out_dir / "captions.vtt").exists():
        return False
    mp4 = captioned_video_output_path(job, args, is_batch)
    if mp4 is not None and not mp4.exists():
        return False
    return True


def manifest_row_for_job(job: VideoJob, args: Any, is_batch: bool) -> Dict[str, str]:
    row = {
        "input": str(job.video_path),
        "output_dir": str(job.out_dir),
        "srt": str(job.out_dir / "captions.srt"),
        "vtt": str(job.out_dir / "captions.vtt"),
        "json": str(job.out_dir / "captions.json"),
    }
    mp4 = captioned_video_output_path(job, args, is_batch)
    if mp4 is not None:
        row["mp4"] = str(mp4)
    return row
