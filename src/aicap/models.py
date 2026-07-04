from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class CaptionItem:
    index: int
    start: float
    end: float
    frame: str
    visual_caption: str
    speech: str
    final_caption: str


@dataclass
class VideoJob:
    video_path: Path
    out_dir: Path
    work_dir: Path
    duration: Optional[float] = None
    frames: Optional[List[Path]] = None
    transcript: Optional[List[TranscriptSegment]] = None
    items: Optional[List[CaptionItem]] = None
