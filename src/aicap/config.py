from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .constants import DEFAULT_OLLAMA_HOST, DEFAULT_PROMPTS_PATH, DEFAULT_SETTINGS_PATH, DEFAULT_TEXT_MODEL, DEFAULT_VIDEO_EXTENSIONS, DEFAULT_VISION_MODEL
from .util import die


PATH_SETTING_KEYS = {"out_dir", "burn_output", "prompts_file"}

ARG_FLAG_NAMES: Dict[str, Tuple[str, ...]] = {
    "out_dir": ("--out-dir",),
    "recursive": ("--recursive", "--no-recursive"),
    "video_extensions": ("--video-extensions",),
    "ollama_host": ("--ollama-host",),
    "ollama_keep_alive": ("--ollama-keep-alive",),
    "ollama_num_ctx": ("--ollama-num-ctx",),
    "ollama_seed": ("--ollama-seed",),
    "vision_model": ("--vision-model",),
    "text_model": ("--text-model",),
    "sample_every": ("--sample-every",),
    "frame_width": ("--frame-width",),
    "jpeg_quality": ("--jpeg-quality",),
    "visual_temperature": ("--visual-temperature",),
    "caption_mode": ("--caption-mode",),
    "prompts_file": ("--prompts-file",),
    "refine": ("--refine", "--no-refine"),
    "refine_batch_size": ("--refine-batch-size",),
    "refine_temperature": ("--refine-temperature",),
    "llm_retries": ("--llm-retries",),
    "strict_refine": ("--strict-refine", "--no-strict-refine"),
    "story_context": ("--story-context", "--no-story-context"),
    "story_previous_captions": ("--story-previous-captions",),
    "story_context_max_chars": ("--story-context-max-chars",),
    "story_summary_max_words": ("--story-summary-max-words",),
    "no_transcribe": ("--no-transcribe", "--transcribe"),
    "whisper_model": ("--whisper-model",),
    "whisper_device": ("--whisper-device",),
    "language": ("--language",),
    "max_frames": ("--max-frames",),
    "burn_in": ("--burn-in", "--no-burn-in"),
    "caption_placement": ("--caption-placement",),
    "burn_output": ("--burn-output",),
    "subtitle_font": ("--subtitle-font",),
    "subtitle_font_size": ("--subtitle-font-size",),
    "subtitle_font_size_percent": ("--subtitle-font-size-percent",),
    "subtitle_margin_v": ("--subtitle-margin-v",),
    "subtitle_margin_v_percent": ("--subtitle-margin-v-percent",),
    "subtitle_band_height": ("--subtitle-band-height",),
    "subtitle_band_height_percent": ("--subtitle-band-height-percent",),
    "video_crf": ("--video-crf",),
    "video_preset": ("--video-preset",),
    "keep_temp": ("--keep-temp", "--no-keep-temp"),
    "resume": ("--resume", "--no-resume"),
    "force": ("--force", "--no-force"),
}


def flatten_settings_table(table: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in table.items():
        if key == "profiles":
            continue
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if not isinstance(child_value, dict):
                    flattened[child_key.replace("-", "_")] = child_value
        else:
            flattened[key.replace("-", "_")] = value
    return flattened


def load_settings_tables(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if not path.exists():
        print(f"Settings file not found, using built-in defaults: {path}", file=sys.stderr)
        return {}, {}
    try:
        with path.open("rb") as f:
            loaded = tomllib.load(f)
    except Exception as e:
        die(f"Could not load settings file: {path}\nDetails: {e}")

    base = flatten_settings_table(loaded)
    profiles: Dict[str, Dict[str, Any]] = {}
    raw_profiles = loaded.get("profiles", {})
    if isinstance(raw_profiles, dict):
        for profile_name, profile_table in raw_profiles.items():
            if isinstance(profile_table, dict):
                profiles[str(profile_name)] = flatten_settings_table(profile_table)
    print(f"Loaded settings from {path}")
    return base, profiles


def coerce_setting_value(key: str, value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if key in PATH_SETTING_KEYS:
        return Path(str(value)) if value != "" else None
    if default is None:
        return value
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)
        if isinstance(default, int) and not isinstance(default, bool):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, Path):
            return Path(str(value))
        if isinstance(default, str):
            return str(value)
    except Exception:
        return value
    return value


def setting_default(settings: Dict[str, Any], key: str, default: Any) -> Any:
    return coerce_setting_value(key, settings.get(key, default), default)


def cli_flag_present(argv: List[str], key: str) -> bool:
    flags = ARG_FLAG_NAMES.get(key, ())
    for arg in argv:
        for flag in flags:
            if arg == flag or arg.startswith(flag + "="):
                return True
    return False


def apply_profile_settings(args: argparse.Namespace, profile_settings: Dict[str, Any], argv: List[str]) -> None:
    for key, value in profile_settings.items():
        if not hasattr(args, key) or cli_flag_present(argv, key):
            continue
        current = getattr(args, key)
        setattr(args, key, coerce_setting_value(key, value, current))


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings-file", type=Path, default=DEFAULT_SETTINGS_PATH)
    pre.add_argument("--settings-profile", choices=["auto", "single", "batch"], default="auto")
    pre_args, _ = pre.parse_known_args(argv)

    settings_path = pre_args.settings_file.expanduser().resolve()
    base_settings, profile_settings_by_name = load_settings_tables(settings_path)

    def cfg(key: str, default: Any) -> Any:
        return setting_default(base_settings, key, default)

    p = argparse.ArgumentParser(description="Local Windows video-to-text captioner using FFmpeg, Ollama vision models, and faster-whisper.")
    p.add_argument("video", type=Path, help="Input video file or a folder of videos, e.g. C:\\Videos\\clip.mp4 or C:\\Videos")
    p.add_argument("--settings-file", type=Path, default=settings_path, help="TOML file containing default runtime settings")
    p.add_argument("--settings-profile", choices=["auto", "single", "batch"], default=cfg("settings_profile", "auto"), help="Settings profile to apply from settings.toml. auto uses batch for folders and single for files")
    p.add_argument("--out-dir", type=Path, default=cfg("out_dir", Path("output")), help="Output folder")
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=cfg("recursive", False), help="When input is a folder, also process videos in subfolders")
    p.add_argument("--video-extensions", default=cfg("video_extensions", DEFAULT_VIDEO_EXTENSIONS), help="Comma-separated video extensions used in folder batch mode")
    p.add_argument("--ollama-host", default=cfg("ollama_host", DEFAULT_OLLAMA_HOST), help="Ollama API host")
    p.add_argument("--ollama-keep-alive", default=cfg("ollama_keep_alive", "30m"), help="How long Ollama should keep the active model loaded between requests, e.g. 30m or 1h")
    p.add_argument("--ollama-num-ctx", type=int, default=cfg("ollama_num_ctx", 8192), help="Ollama context window for prompt-heavy captioning. Use 0 for the model default")
    p.add_argument("--ollama-seed", type=int, default=cfg("ollama_seed", 42), help="Fixed Ollama seed for repeatable generations. Use a negative value to disable")
    p.add_argument("--vision-model", default=cfg("vision_model", DEFAULT_VISION_MODEL), help="Ollama vision model")
    p.add_argument("--text-model", default=cfg("text_model", DEFAULT_TEXT_MODEL), help="Ollama text model used when --refine is set")
    p.add_argument("--sample-every", type=float, default=cfg("sample_every", 10.0), help="Seconds between visual samples. Default is 10.0 for story-style captions; use 1.0 or 2.0 for denser captions")
    p.add_argument("--frame-width", type=int, default=cfg("frame_width", 960), help="Resize extracted frames to this width. Use 0 to keep original size")
    p.add_argument("--jpeg-quality", type=int, default=cfg("jpeg_quality", 4), help="FFmpeg JPG quality, 2 is high, 31 is low")
    p.add_argument("--visual-temperature", type=float, default=cfg("visual_temperature", 0.0), help="Ollama sampling temperature for per-frame visual captions")
    p.add_argument("--caption-mode", choices=["neutral", "explicit"], default=cfg("caption_mode", "explicit"), help="Caption style")
    p.add_argument("--prompts-file", type=Path, default=cfg("prompts_file", DEFAULT_PROMPTS_PATH), help="Editable TOML file containing all model prompts")
    p.add_argument("--refine", action=argparse.BooleanOptionalAction, default=cfg("refine", False), help="Use the text LLM to merge visual captions and speech into cleaner subtitles")
    p.add_argument("--refine-batch-size", type=int, default=cfg("refine_batch_size", 8), help="Number of captions per LLM refinement request")
    p.add_argument("--refine-temperature", type=float, default=cfg("refine_temperature", 0.0), help="Ollama sampling temperature for text refinement")
    p.add_argument("--llm-retries", type=int, default=cfg("llm_retries", 2), help="Retry malformed or incomplete LLM JSON responses before falling back")
    p.add_argument("--strict-refine", action=argparse.BooleanOptionalAction, default=cfg("strict_refine", True), help="Stop instead of writing fallback captions when LLM refinement fails after retries")
    p.add_argument("--story-context", action=argparse.BooleanOptionalAction, default=cfg("story_context", True), help="Use rolling story memory between refinement batches. Enabled by default; use --no-story-context for independent batches")
    p.add_argument("--story-previous-captions", type=int, default=cfg("story_previous_captions", 18), help="How many recent final captions to pass into the next story-aware refinement batch")
    p.add_argument("--story-context-max-chars", type=int, default=cfg("story_context_max_chars", 4000), help="Approximate maximum characters of story memory/recent-caption context per refinement request")
    p.add_argument("--story-summary-max-words", type=int, default=cfg("story_summary_max_words", 140), help="Maximum words in the rolling story summary passed between batches")
    p.add_argument("--no-transcribe", dest="no_transcribe", action="store_true", default=cfg("no_transcribe", False), help="Skip Whisper speech transcription")
    p.add_argument("--transcribe", dest="no_transcribe", action="store_false", help="Enable Whisper speech transcription even if settings.toml disables it")
    p.add_argument("--whisper-model", default=cfg("whisper_model", "large-v3-turbo"), help="faster-whisper model size/name")
    p.add_argument("--whisper-device", choices=["auto", "cuda", "cpu"], default=cfg("whisper_device", "auto"), help="Whisper device")
    p.add_argument("--language", default=cfg("language", None), help="Optional speech language code, e.g. en, nl, de. Omit for auto-detect")
    p.add_argument("--max-frames", type=int, default=cfg("max_frames", None), help="Only process the first N sampled frames, useful for testing")
    p.add_argument("--burn-in", action=argparse.BooleanOptionalAction, default=cfg("burn_in", True), help="Create a captioned MP4 with subtitles burned into the video. Use --no-burn-in to disable")
    p.add_argument("--caption-placement", choices=["below", "bottom"], default=cfg("caption_placement", "below"), help="below adds a black caption band under the video; bottom overlays captions inside the video")
    p.add_argument("--burn-output", type=Path, default=cfg("burn_output", None), help="Output MP4 path for the captioned video. Defaults to <out-dir>/captioned_video.mp4")
    p.add_argument("--subtitle-font", default=cfg("subtitle_font", "Arial"), help="Subtitle font used when burning captions")
    p.add_argument("--subtitle-font-size", type=int, default=cfg("subtitle_font_size", None), help="Fixed subtitle font size in pixels. Overrides --subtitle-font-size-percent when set")
    p.add_argument("--subtitle-font-size-percent", type=float, default=cfg("subtitle_font_size_percent", 3.2), help="Subtitle font size as a percent of the final video height when --subtitle-font-size is not set")
    p.add_argument("--subtitle-margin-v", type=int, default=cfg("subtitle_margin_v", None), help="Fixed bottom subtitle margin in pixels. Overrides --subtitle-margin-v-percent when set")
    p.add_argument("--subtitle-margin-v-percent", type=float, default=cfg("subtitle_margin_v_percent", 1.8), help="Bottom subtitle margin as a percent of the final video height when --subtitle-margin-v is not set")
    p.add_argument("--subtitle-band-height", type=int, default=cfg("subtitle_band_height", None), help="Fixed extra black band height in pixels when --caption-placement below is used. Overrides --subtitle-band-height-percent when set")
    p.add_argument("--subtitle-band-height-percent", type=float, default=cfg("subtitle_band_height_percent", 16.0), help="Extra black band height as a percent of the original video height when --caption-placement below is used")
    p.add_argument("--video-crf", type=int, default=cfg("video_crf", 20), help="x264 CRF for captioned MP4. Lower is higher quality/larger file")
    p.add_argument("--video-preset", default=cfg("video_preset", "medium"), help="x264 preset for captioned MP4, e.g. ultrafast, veryfast, medium, slow")
    p.add_argument("--keep-temp", action=argparse.BooleanOptionalAction, default=cfg("keep_temp", False), help="Keep extracted frames and audio")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=cfg("resume", True), help="Resume from completed checkpoints and cached intermediate files. Enabled by default")
    p.add_argument("--force", action=argparse.BooleanOptionalAction, default=cfg("force", False), help="Ignore resume checkpoints and regenerate everything from scratch")

    args = p.parse_args(argv)
    profile_name = args.settings_profile
    if profile_name == "auto":
        try:
            profile_name = "batch" if args.video.expanduser().resolve().is_dir() else "single"
        except Exception:
            profile_name = "single"

    profile_settings = profile_settings_by_name.get(profile_name, {})
    if profile_settings:
        apply_profile_settings(args, profile_settings, argv)
        print(f"Applied settings profile: {profile_name}")
    elif args.settings_profile != "auto":
        print(f"Settings profile not found, using base settings only: {profile_name}", file=sys.stderr)

    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_every <= 0:
        die("--sample-every must be greater than 0")
    if args.subtitle_font_size_percent <= 0:
        die("--subtitle-font-size-percent must be greater than 0")
    if args.subtitle_margin_v_percent < 0:
        die("--subtitle-margin-v-percent must be 0 or greater")
    if args.subtitle_band_height_percent < 0:
        die("--subtitle-band-height-percent must be 0 or greater")
    if args.refine_batch_size <= 0:
        die("--refine-batch-size must be greater than 0")
    if args.visual_temperature < 0:
        die("--visual-temperature must be 0 or greater")
    if args.refine_temperature < 0:
        die("--refine-temperature must be 0 or greater")
    if args.llm_retries < 0:
        die("--llm-retries must be 0 or greater")
    if args.ollama_num_ctx < 0:
        die("--ollama-num-ctx must be 0 or greater")
    if args.story_previous_captions < 0:
        die("--story-previous-captions must be 0 or greater")
    if args.story_context_max_chars < 0:
        die("--story-context-max-chars must be 0 or greater")
    if args.story_summary_max_words < 0:
        die("--story-summary-max-words must be 0 or greater")
