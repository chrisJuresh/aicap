from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import TranscriptSegment
from .util import clean_text, die, require_exe, run_cmd


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def ffprobe_duration(video_path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            check=True,
            text=True,
            capture_output=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def ffprobe_video_size(video_path: Path) -> Optional[Tuple[int, int]]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(video_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        raw = result.stdout.strip().splitlines()[0].strip()
        width_s, height_s = raw.split("x", 1)
        width, height = int(width_s), int(height_s)
        if width <= 0 or height <= 0:
            return None
        return width, height
    except Exception:
        return None


def ffprobe_audio_streams(video_path: Path) -> List[Dict[str, Any]]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index,codec_name,channels,sample_rate",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        return [s for s in streams if isinstance(s, dict)]
    except Exception as e:
        print(f"Warning: could not probe audio streams for {video_path.name}: {e}", file=sys.stderr)
        return []


def has_audio_stream(video_path: Path) -> bool:
    return bool(ffprobe_audio_streams(video_path))


def extract_frames(video_path: Path, work_dir: Path, sample_every: float, frame_width: int, jpeg_quality: int) -> List[Path]:
    ffmpeg = require_exe("ffmpeg")
    frames_dir = work_dir / "frames"
    if frames_dir.exists():
        for old in frames_dir.glob("frame_*.jpg"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    fps_value = 1.0 / sample_every
    filters = [f"fps={fps_value:.6f}"]
    if frame_width and frame_width > 0:
        filters.append(f"scale={frame_width}:-2")

    out_pattern = str(frames_dir / "frame_%06d.jpg")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        ",".join(filters),
        "-q:v",
        str(jpeg_quality),
        out_pattern,
    ]
    print(f"Extracting frames every {sample_every:g}s...")
    run_cmd(cmd)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        die("No frames were extracted. Check that the input is a valid video file.")
    return frames


def extract_audio(video_path: Path, work_dir: Path) -> Optional[Path]:
    ffmpeg = require_exe("ffmpeg")
    audio_path = work_dir / "audio_16k_mono.wav"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not has_audio_stream(video_path):
        print(f"No audio stream found in {video_path.name}; continuing with visual captions only.")
        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass
        return None

    print("Extracting audio...")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    try:
        run_cmd(cmd)
    except SystemExit:
        print(
            f"Warning: audio extraction failed for {video_path.name}; continuing with visual captions only. "
            "The video may have an unsupported/corrupt audio stream.",
            file=sys.stderr,
        )
        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass
        return None

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        print(f"Warning: extracted audio is empty for {video_path.name}; continuing with visual captions only.", file=sys.stderr)
        return None
    return audio_path


def load_whisper_model(whisper_model: str, whisper_device: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        die(f"Could not import faster-whisper. Run: pip install -r requirements.txt\nDetails: {e}")

    attempted: List[str] = []
    devices = ["cuda", "cpu"] if whisper_device == "auto" else [whisper_device]
    last_error: Optional[Exception] = None

    for device in devices:
        compute_type = "float16" if device == "cuda" else "int8"
        attempted.append(f"{device}/{compute_type}")
        try:
            print(f"Loading Whisper model {whisper_model} on {device}...")
            return WhisperModel(whisper_model, device=device, compute_type=compute_type)
        except Exception as e:
            last_error = e
            print(f"Whisper failed to load on {device}: {e}", file=sys.stderr)
            continue

    die(f"Whisper model loading failed. Attempted: {', '.join(attempted)}. Last error: {last_error}")


def transcribe_audio_with_model(
    video_path: Path,
    work_dir: Path,
    model: Any,
    language: Optional[str],
) -> List[TranscriptSegment]:
    audio_path = extract_audio(video_path, work_dir)
    if audio_path is None:
        return []
    kwargs: Dict[str, Any] = {
        "beam_size": 5,
        "vad_filter": True,
        "word_timestamps": False,
    }
    if language:
        kwargs["language"] = language
    try:
        segments_iter, info = model.transcribe(str(audio_path), **kwargs)
        print(f"Transcribing audio ({getattr(info, 'language', 'unknown')} detected)...")
        segments = [TranscriptSegment(float(s.start), float(s.end), clean_text(s.text)) for s in segments_iter]
        return [s for s in segments if s.text]
    except Exception as e:
        die(f"Whisper transcription failed for {video_path.name}: {e}")
