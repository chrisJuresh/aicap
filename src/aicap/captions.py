from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import CaptionItem, TranscriptSegment
from .util import clean_text, read_json, seconds_to_srt_time, write_json


def load_transcript(path: Path) -> List[TranscriptSegment]:
    data = read_json(path, [])
    if not isinstance(data, list):
        return []
    segments: List[TranscriptSegment] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            segments.append(TranscriptSegment(float(row.get("start", 0)), float(row.get("end", 0)), clean_text(str(row.get("text", "")))))
        except Exception:
            continue
    return [s for s in segments if s.text]


def load_caption_items(path: Path) -> List[CaptionItem]:
    data = read_json(path, [])
    if not isinstance(data, list):
        return []
    items: List[CaptionItem] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            items.append(
                CaptionItem(
                    index=int(row.get("index", row.get("i", 0))),
                    start=float(row.get("start", 0)),
                    end=float(row.get("end", 0)),
                    frame=str(row.get("frame", "")),
                    visual_caption=clean_text(str(row.get("visual_caption", row.get("visual", "")))),
                    speech=clean_text(str(row.get("speech", ""))),
                    final_caption=clean_text(str(row.get("final_caption", row.get("caption", "")))),
                )
            )
        except Exception:
            continue
    return [i for i in items if i.index > 0]


def save_items_snapshot(path: Path, items: List[CaptionItem]) -> None:
    from dataclasses import asdict

    write_json(path, [asdict(i) for i in items])


def speech_for_interval(segments: List[TranscriptSegment], start: float, end: float) -> str:
    if not segments:
        return ""
    pieces: List[str] = []
    for seg in segments:
        if max(seg.start, start - 0.15) < min(seg.end, end + 0.15):
            pieces.append(seg.text)
    return clean_text(" ".join(pieces))


def fallback_caption(visual: str, speech: str) -> str:
    visual = clean_text(visual)
    speech = clean_text(speech)
    if speech and visual:
        return clean_text(f"{visual} Speech: {speech}")
    return visual or speech or "[No caption generated]"


def build_items(
    frames: List[Path],
    visual_captions: List[str],
    transcript: List[TranscriptSegment],
    sample_every: float,
    duration: Optional[float],
    max_frames: Optional[int],
) -> List[CaptionItem]:
    if max_frames is not None:
        frames = frames[:max_frames]
    items: List[CaptionItem] = []
    for idx, (frame, visual) in enumerate(zip(frames, visual_captions), start=1):
        start = (idx - 1) * sample_every
        end = idx * sample_every if idx < len(frames) else start + sample_every
        if duration is not None:
            end = min(end, duration)
        if end <= start:
            end = start + sample_every
        speech = speech_for_interval(transcript, start, end)
        items.append(
            CaptionItem(
                index=idx,
                start=float(start),
                end=float(end),
                frame=frame.name,
                visual_caption=visual,
                speech=speech,
                final_caption=fallback_caption(visual, speech),
            )
        )
    return items


def extract_json_array(text: str) -> Optional[List[Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, list):
                return obj
        except Exception:
            return None
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


CAPTION_KEYS = ("caption", "text", "final_caption", "subtitle", "line")
INDEX_KEYS = ("i", "index", "id", "caption_index")


def caption_rows_from_payload(payload: Any) -> Optional[List[Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None

    nested = payload.get("captions")
    if nested is not None:
        return caption_rows_from_payload(nested)

    if any(str(key).isdigit() for key in payload.keys()):
        return [{"i": key, "caption": value} for key, value in payload.items()]

    if any(key in payload for key in CAPTION_KEYS):
        return [payload]

    return None


def caption_text_from_row(row: Any) -> str:
    if isinstance(row, str):
        return clean_text(row)
    if not isinstance(row, dict):
        return ""
    for key in CAPTION_KEYS:
        if key in row:
            return clean_text(str(row[key]))
    return ""


def caption_index_from_row(row: Any) -> Optional[int]:
    if not isinstance(row, dict):
        return None
    for key in INDEX_KEYS:
        if key not in row:
            continue
        try:
            return int(row[key])
        except Exception:
            return None
    return None


def captions_by_index_from_rows(rows: Iterable[Any]) -> Dict[int, str]:
    by_index: Dict[int, str] = {}
    for row in rows:
        caption = caption_text_from_row(row)
        index = caption_index_from_row(row)
        if caption and index is not None:
            by_index[index] = caption
    return by_index


def captions_by_chunk_from_rows(rows: Iterable[Any], chunk: List[CaptionItem]) -> Dict[int, str]:
    requested = [item.index for item in chunk]
    requested_set = set(requested)
    by_index: Dict[int, str] = {}
    ordered_fallbacks: List[str] = []

    for row in rows:
        caption = caption_text_from_row(row)
        if not caption:
            continue
        index = caption_index_from_row(row)
        if index in requested_set and index not in by_index:
            by_index[index] = caption
        else:
            ordered_fallbacks.append(caption)

    if not by_index and len(ordered_fallbacks) >= len(chunk):
        return {item.index: ordered_fallbacks[position] for position, item in enumerate(chunk)}

    fallback_iter = iter(ordered_fallbacks)
    for item in chunk:
        if item.index in by_index:
            continue
        fallback_caption = next(fallback_iter, "")
        if fallback_caption:
            by_index[item.index] = fallback_caption

    return by_index


def missing_caption_indices(chunk: List[CaptionItem], by_index: Dict[int, str]) -> List[int]:
    return [item.index for item in chunk if not by_index.get(item.index)]


def clamp_words(text: str, max_words: int) -> str:
    text = clean_text(text)
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "..."


def format_previous_captions(items: List[CaptionItem], max_count: int, max_chars: int) -> str:
    if max_count <= 0 or not items:
        return ""
    rows: List[str] = []
    for item in items[-max_count:]:
        caption = clean_text(item.final_caption)
        if caption:
            rows.append(f"{seconds_to_srt_time(item.start)} - {caption}")
    text = "\n".join(rows)
    return text if len(text) <= max_chars else text[-max_chars:].lstrip()
