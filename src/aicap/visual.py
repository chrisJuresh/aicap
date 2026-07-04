from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .cache import load_visual_cache
from .media import image_to_base64
from .ollama import check_ollama, ollama_generate
from .prompts import visual_prompt
from .util import append_jsonl, clean_text


def caption_frames_cached(
    frames: List[Path],
    host: str,
    vision_model: str,
    caption_mode: str,
    max_frames: Optional[int],
    prompts: Dict[str, Dict[str, str]],
    keep_alive: str,
    cache_path: Path,
    temperature: float,
    seed: Optional[int],
    num_ctx: Optional[int],
) -> List[str]:
    if max_frames is not None:
        frames = frames[:max_frames]
    prompt = visual_prompt(caption_mode, prompts)
    cached = load_visual_cache(cache_path)
    captions_by_frame: Dict[str, str] = dict(cached)
    missing = [frame for frame in frames if frame.name not in captions_by_frame]

    if cached:
        print(f"Resume: found {len(cached)} cached frame caption(s).")
    print(f"Captioning {len(missing)} missing frame(s) with {vision_model}...")

    if missing:
        check_ollama(host)
    for frame in tqdm(missing, unit="frame"):
        raw = ollama_generate(
            host,
            vision_model,
            prompt,
            images=[image_to_base64(frame)],
            temperature=temperature,
            num_predict=90,
            timeout=600,
            keep_alive=keep_alive,
            seed=seed,
            num_ctx=num_ctx,
        )
        caption = clean_text(raw)
        captions_by_frame[frame.name] = caption
        append_jsonl(cache_path, {"frame": frame.name, "caption": caption})
    return [captions_by_frame.get(frame.name, "") for frame in frames]
