from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from .media import ffprobe_video_size
from .models import CaptionItem
from .util import atomic_write_text, clean_text, require_exe, run_cmd, seconds_to_srt_time, seconds_to_vtt_time


SUBTITLE_LAYOUT_VERSION = 2


def write_srt(path: Path, items: List[CaptionItem]) -> None:
    blocks: List[str] = []
    for n, item in enumerate(items, start=1):
        caption = clean_text(item.final_caption)
        wrapped = "\n".join(textwrap.wrap(caption, width=64)) if caption else "[No caption]"
        blocks.append(f"{n}\n{seconds_to_srt_time(item.start)} --> {seconds_to_srt_time(item.end)}\n{wrapped}")
    atomic_write_text(path, "\n\n".join(blocks) + "\n")


def write_vtt(path: Path, items: List[CaptionItem]) -> None:
    blocks = ["WEBVTT", ""]
    for item in items:
        caption = clean_text(item.final_caption)
        wrapped = "\n".join(textwrap.wrap(caption, width=64)) if caption else "[No caption]"
        blocks.append(f"{seconds_to_vtt_time(item.start)} --> {seconds_to_vtt_time(item.end)}\n{wrapped}\n")
    atomic_write_text(path, "\n".join(blocks))


def ffmpeg_filter_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", r"\'").replace(":", r"\:").replace(",", r"\,")


def ffmpeg_subtitle_path(path: Path) -> str:
    return ffmpeg_filter_escape(path.resolve().as_posix())


def build_ass_force_style(font: str, font_size: int, margin_v: int) -> str:
    safe_font = font.replace(",", " ").strip() or "Arial"
    return ",".join(
        [
            f"FontName={safe_font}",
            f"FontSize={max(8, font_size)}",
            "PrimaryColour=&H00FFFFFF",
            "OutlineColour=&H00000000",
            "BackColour=&H80000000",
            "BorderStyle=1",
            "Outline=2",
            "Shadow=1",
            "Alignment=2",
            f"MarginV={max(0, margin_v)}",
        ]
    )


def even_int(value: int, minimum: int = 2) -> int:
    value = max(minimum, int(round(value)))
    return value if value % 2 == 0 else value + 1


def percent_of_height(height: int, percent: float, minimum: int) -> int:
    return max(minimum, int(round(height * (percent / 100.0))))


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        maximum = minimum
    return max(minimum, min(maximum, int(round(value))))


def resolve_subtitle_layout(
    video_path: Path,
    placement: str,
    fixed_font_size: Optional[int],
    font_size_percent: float,
    fixed_margin_v: Optional[int],
    margin_v_percent: float,
    fixed_band_height: Optional[int],
    band_height_percent: float,
) -> Tuple[int, int, int, int]:
    size = ffprobe_video_size(video_path)
    input_width = 1920 if size is None else size[0]
    input_height = 1080 if size is None else size[1]
    readable_dimension = max(360, min(input_width, input_height))

    if placement == "below":
        requested_band_height = fixed_band_height if fixed_band_height is not None else percent_of_height(input_height, band_height_percent, 48)
    else:
        requested_band_height = 0

    font_reference_height = input_height + (requested_band_height if placement == "below" else 0)
    if fixed_font_size is not None:
        font_size = max(8, int(fixed_font_size))
    else:
        requested_font_size = percent_of_height(font_reference_height, font_size_percent, 18)
        adaptive_font_size = clamp_int(round(readable_dimension * 0.028), 16, 34)
        font_size = clamp_int(min(requested_font_size, adaptive_font_size), 14, 40)

    if fixed_margin_v is not None:
        margin_v = max(0, int(fixed_margin_v))
    else:
        requested_margin = percent_of_height(font_reference_height, margin_v_percent, 8)
        adaptive_margin = clamp_int(round(font_size * 0.45), 6, max(8, font_size))
        margin_v = clamp_int(min(requested_margin, adaptive_margin), 4, max(8, font_size))

    if placement == "below":
        if fixed_band_height is not None:
            band_height = even_int(fixed_band_height, minimum=24)
        else:
            natural_band = even_int(round(font_size * 2.8 + margin_v * 2), minimum=48)
            max_band = even_int(clamp_int(round(input_height * 0.10), 56, 132), minimum=24)
            min_band = even_int(round(font_size * 1.8 + margin_v * 2), minimum=48)
            band_height = even_int(clamp_int(min(requested_band_height, natural_band), min_band, max_band), minimum=24)
        reference_height = input_height + band_height
    else:
        band_height = 0
        reference_height = input_height

    return font_size, margin_v, band_height, reference_height


def burn_captions_to_video(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    placement: str,
    font: str,
    font_size: Optional[int],
    font_size_percent: float,
    margin_v: Optional[int],
    margin_v_percent: float,
    band_height: Optional[int],
    band_height_percent: float,
    crf: int,
    preset: str,
) -> None:
    ffmpeg = require_exe("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_font_size, resolved_margin_v, resolved_band_height, reference_height = resolve_subtitle_layout(
        video_path=video_path,
        placement=placement,
        fixed_font_size=font_size,
        font_size_percent=font_size_percent,
        fixed_margin_v=margin_v,
        margin_v_percent=margin_v_percent,
        fixed_band_height=band_height,
        band_height_percent=band_height_percent,
    )

    style = build_ass_force_style(font, resolved_font_size, resolved_margin_v)
    srt = ffmpeg_subtitle_path(srt_path)
    subtitle_filter = f"subtitles='{srt}':force_style='{style}'"
    video_filter = f"pad=iw:ih+{resolved_band_height}:0:0:color=black,{subtitle_filter}" if placement == "below" else subtitle_filter

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-filter_complex",
        f"[0:v]{video_filter}[v]",
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    print(
        f"Burning captions into video ({placement}): {output_path} "
        f"[font={resolved_font_size}px, margin={resolved_margin_v}px, "
        f"band={resolved_band_height}px, reference-height={reference_height}px]"
    )
    run_cmd(cmd)
