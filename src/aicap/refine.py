from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .captions import (
    caption_rows_from_payload,
    caption_rows_from_text,
    captions_by_chunk_from_rows,
    clamp_words,
    extract_json_array,
    extract_json_object,
    format_previous_captions,
    missing_caption_indices,
)
from .models import CaptionItem
from .ollama import check_ollama, ollama_generate
from .prompts import format_prompt_template, get_prompt
from .util import clean_text, die, read_json, write_json


def fit_story_context(story_summary: str, previous_captions: str, max_chars: int) -> Tuple[str, str]:
    if max_chars <= 0:
        return "", ""
    story_summary = clean_text(story_summary)
    previous_captions = previous_captions.strip()
    combined_len = len(story_summary) + len(previous_captions)
    if combined_len <= max_chars:
        return story_summary, previous_captions

    max_prev = max(0, max_chars - len(story_summary) - 200)
    if max_prev <= 0:
        return story_summary[-max_chars:].lstrip(), ""
    return story_summary, previous_captions[-max_prev:].lstrip()


def load_story_refine_cache(cache_path: Optional[Path]) -> Dict[str, Any]:
    if cache_path is None:
        return {"captions": {}, "batches": {}, "final_story_summary": ""}
    loaded = read_json(cache_path, {})
    if not isinstance(loaded, dict):
        return {"captions": {}, "batches": {}, "final_story_summary": ""}
    if "captions" not in loaded and any(str(k).isdigit() for k in loaded.keys()):
        return {
            "captions": {str(k): clean_text(str(v)) for k, v in loaded.items() if str(k).isdigit() and str(v).strip()},
            "batches": {},
            "final_story_summary": "",
        }
    captions = loaded.get("captions", {})
    batches = loaded.get("batches", {})
    return {
        "captions": {str(k): clean_text(str(v)) for k, v in captions.items()} if isinstance(captions, dict) else {},
        "batches": batches if isinstance(batches, dict) else {},
        "final_story_summary": clean_text(str(loaded.get("final_story_summary", ""))),
    }


def write_story_refine_cache(cache_path: Optional[Path], cache: Dict[str, Any]) -> None:
    if cache_path is not None:
        write_json(cache_path, cache)


def _handle_refine_failure(message: str, strict_refine: bool, fallback_text: str) -> bool:
    if strict_refine:
        die(message + " Stopping because strict refinement is enabled.")
    print(message + f" {fallback_text}", file=sys.stderr)
    return False


def parse_caption_rows(raw: str) -> Tuple[Optional[List[Any]], str]:
    parsed_obj = extract_json_object(raw)
    if parsed_obj is not None:
        rows = caption_rows_from_payload(parsed_obj)
        summary = clean_text(str(parsed_obj.get("story_summary", "")))
        if rows is not None:
            return rows, summary

    rows = caption_rows_from_payload(extract_json_array(raw))
    if rows is not None:
        return rows, ""

    text_rows = caption_rows_from_text(raw)
    if text_rows:
        return text_rows, ""

    return None, ""


def simplified_refine_prompt(chunk: List[CaptionItem], input_rows: List[Dict[str, Any]], story_summary_max_words: int) -> str:
    ids = ", ".join(str(item.index) for item in chunk)
    return (
        "Return ONLY valid JSON. Do not include markdown or explanations.\n"
        f"Use exactly these i values, once each, in this order: {ids}.\n"
        "Write one concise subtitle caption for each input row.\n"
        "Use only the visible/speech evidence in the input. Do not invent unsupported events.\n"
        "The JSON must have this shape: "
        '{"captions":[{"i": number, "caption": "short caption"}], "story_summary": "brief memory"}.\n'
        f"Keep story_summary under {story_summary_max_words} words.\n\n"
        "Input JSON:\n"
        f"{json.dumps(input_rows, ensure_ascii=False)}"
    )


def repair_refine_batch(
    chunk: List[CaptionItem],
    input_rows: List[Dict[str, Any]],
    host: str,
    text_model: str,
    keep_alive: str,
    seed: Optional[int],
    num_ctx: Optional[int],
    story_summary_max_words: int,
) -> Tuple[Dict[int, str], str]:
    raw = ollama_generate(
        host,
        text_model,
        simplified_refine_prompt(chunk, input_rows, story_summary_max_words),
        temperature=0.0,
        num_predict=1400,
        timeout=900,
        keep_alive=keep_alive,
        seed=seed,
        num_ctx=num_ctx,
        response_format="json",
    )
    rows, summary = parse_caption_rows(raw)
    if rows is None:
        return {}, ""
    return captions_by_chunk_from_rows(rows, chunk), summary


def refine_with_llm(
    items: List[CaptionItem],
    host: str,
    text_model: str,
    caption_mode: str,
    batch_size: int,
    prompts: Dict[str, Dict[str, str]],
    keep_alive: str,
    temperature: float,
    seed: Optional[int],
    num_ctx: Optional[int],
    llm_retries: int,
    strict_refine: bool,
    cache_path: Optional[Path] = None,
    story_context: bool = True,
    story_previous_captions: int = 18,
    story_context_max_chars: int = 4000,
    story_summary_max_words: int = 140,
) -> List[CaptionItem]:
    if not text_model:
        return items

    if not story_context:
        return _refine_independent_batches(
            items,
            host,
            text_model,
            caption_mode,
            batch_size,
            prompts,
            keep_alive,
            temperature,
            seed,
            num_ctx,
            llm_retries,
            strict_refine,
            cache_path,
        )

    return _refine_story_batches(
        items,
        host,
        text_model,
        caption_mode,
        batch_size,
        prompts,
        keep_alive,
        temperature,
        seed,
        num_ctx,
        llm_retries,
        strict_refine,
        cache_path,
        story_previous_captions,
        story_context_max_chars,
        story_summary_max_words,
    )


def _refine_independent_batches(
    items: List[CaptionItem],
    host: str,
    text_model: str,
    caption_mode: str,
    batch_size: int,
    prompts: Dict[str, Dict[str, str]],
    keep_alive: str,
    temperature: float,
    seed: Optional[int],
    num_ctx: Optional[int],
    llm_retries: int,
    strict_refine: bool,
    cache_path: Optional[Path],
) -> List[CaptionItem]:
    cached_refined: Dict[str, str] = {}
    if cache_path is not None:
        loaded = read_json(cache_path, {})
        if isinstance(loaded, dict):
            if "captions" in loaded and isinstance(loaded.get("captions"), dict):
                loaded = loaded.get("captions", {})
            cached_refined = {str(k): clean_text(str(v)) for k, v in loaded.items() if str(v).strip()}
        if cached_refined:
            print(f"Resume: found {len(cached_refined)} cached refined caption(s).")
            for item in items:
                cached_value = cached_refined.get(str(item.index))
                if cached_value:
                    item.final_caption = cached_value

    print(f"Refining captions with {text_model} in independent batches of {batch_size}...")
    for start_idx in tqdm(range(0, len(items), batch_size), unit="batch"):
        chunk = items[start_idx : start_idx + batch_size]
        if cache_path is not None and all(str(item.index) in cached_refined for item in chunk):
            continue
        input_rows = [
            {"i": item.index, "start": round(item.start, 3), "end": round(item.end, 3), "visual": item.visual_caption, "speech": item.speech}
            for item in chunk
        ]
        mode_key = "explicit_mode_instruction" if caption_mode == "explicit" else "neutral_mode_instruction"
        prompt = format_prompt_template(
            get_prompt(prompts, "refine", "template"),
            mode_instruction=get_prompt(prompts, "refine", mode_key),
            input_json=json.dumps(input_rows, ensure_ascii=False),
        )
        check_ollama(host)
        by_index: Dict[int, str] = {}
        missing_indices = [item.index for item in chunk]
        for attempt in range(llm_retries + 1):
            raw = ollama_generate(
                host,
                text_model,
                prompt,
                temperature=temperature,
                num_predict=1600,
                timeout=900,
                keep_alive=keep_alive,
                seed=seed,
                num_ctx=num_ctx,
                response_format="json",
            )
            parsed_rows, _ = parse_caption_rows(raw)
            if parsed_rows is None:
                if attempt < llm_retries:
                    print("Could not parse LLM JSON for one batch; retrying.", file=sys.stderr)
                continue
            by_index = captions_by_chunk_from_rows(parsed_rows, chunk)
            missing_indices = missing_caption_indices(chunk, by_index)
            if not missing_indices:
                break
            if attempt < llm_retries:
                print(f"LLM omitted {len(missing_indices)} caption(s) in one batch; retrying.", file=sys.stderr)
        if missing_indices:
            print("Trying simplified refinement repair for one batch.", file=sys.stderr)
            repaired, _ = repair_refine_batch(chunk, input_rows, host, text_model, keep_alive, seed, num_ctx, 120)
            if repaired:
                by_index = repaired
                missing_indices = missing_caption_indices(chunk, by_index)
        if not by_index:
            if not _handle_refine_failure("Could not parse LLM JSON after retries.", strict_refine, "Keeping fallback captions for this batch."):
                continue
        if missing_indices:
            _handle_refine_failure(f"LLM still omitted {len(missing_indices)} caption(s) after retries.", strict_refine, "Keeping fallback for those item(s).")
        for item in chunk:
            if item.index in by_index and by_index[item.index]:
                item.final_caption = by_index[item.index]
            if cache_path is not None:
                cached_refined[str(item.index)] = item.final_caption
        if cache_path is not None:
            write_json(cache_path, cached_refined)
    return items


def _refine_story_batches(
    items: List[CaptionItem],
    host: str,
    text_model: str,
    caption_mode: str,
    batch_size: int,
    prompts: Dict[str, Dict[str, str]],
    keep_alive: str,
    temperature: float,
    seed: Optional[int],
    num_ctx: Optional[int],
    llm_retries: int,
    strict_refine: bool,
    cache_path: Optional[Path],
    story_previous_captions: int,
    story_context_max_chars: int,
    story_summary_max_words: int,
) -> List[CaptionItem]:
    cache = load_story_refine_cache(cache_path)
    cached_refined = cache["captions"]
    batches = cache["batches"]
    if cached_refined:
        print(f"Resume: found {len(cached_refined)} cached story-aware refined caption(s).")
        for item in items:
            cached_value = cached_refined.get(str(item.index))
            if cached_value:
                item.final_caption = cached_value

    print(f"Refining captions with {text_model} in story-aware batches of {batch_size}...")
    story_summary = get_prompt(prompts, "story", "empty_story_summary")
    completed_items: List[CaptionItem] = []
    mode_key = "explicit_mode_instruction" if caption_mode == "explicit" else "neutral_mode_instruction"
    mode_instruction = get_prompt(prompts, "refine", mode_key)

    for start_idx in tqdm(range(0, len(items), batch_size), unit="batch"):
        chunk = items[start_idx : start_idx + batch_size]
        batch_key = f"{chunk[0].index}-{chunk[-1].index}" if chunk else str(start_idx)
        batch_cache = batches.get(batch_key, {}) if isinstance(batches, dict) else {}
        if cache_path is not None and all(str(item.index) in cached_refined for item in chunk) and isinstance(batch_cache, dict):
            cached_summary = clean_text(str(batch_cache.get("story_summary_after", "")))
            if cached_summary:
                story_summary = cached_summary
            completed_items.extend(chunk)
            continue

        previous_captions = format_previous_captions(completed_items, story_previous_captions, max(500, story_context_max_chars))
        if not previous_captions:
            previous_captions = get_prompt(prompts, "story", "empty_previous_captions")
        fitted_story, fitted_previous = fit_story_context(story_summary, previous_captions, story_context_max_chars)
        input_rows = [
            {
                "i": item.index,
                "start": round(item.start, 3),
                "end": round(item.end, 3),
                "visual": item.visual_caption,
                "speech": item.speech,
                "fallback_caption": item.final_caption,
            }
            for item in chunk
        ]
        prompt = format_prompt_template(
            get_prompt(prompts, "story", "template"),
            mode_instruction=mode_instruction,
            story_summary=fitted_story or get_prompt(prompts, "story", "empty_story_summary"),
            previous_captions=fitted_previous or get_prompt(prompts, "story", "empty_previous_captions"),
            input_json=json.dumps(input_rows, ensure_ascii=False),
            story_summary_max_words=str(story_summary_max_words),
        )
        check_ollama(host)
        new_summary = ""
        by_index: Dict[int, str] = {}
        missing_indices = [item.index for item in chunk]

        for attempt in range(llm_retries + 1):
            raw = ollama_generate(
                host,
                text_model,
                prompt,
                temperature=temperature,
                num_predict=2200,
                timeout=900,
                keep_alive=keep_alive,
                seed=seed,
                num_ctx=num_ctx,
                response_format="json",
            )
            parsed_captions, new_summary = parse_caption_rows(raw)
            new_summary = clamp_words(new_summary, story_summary_max_words)

            if parsed_captions is None:
                if attempt < llm_retries:
                    print("Could not parse story-aware LLM JSON for one batch; retrying.", file=sys.stderr)
                continue
            by_index = captions_by_chunk_from_rows(parsed_captions, chunk)
            missing_indices = missing_caption_indices(chunk, by_index)
            if not missing_indices:
                break
            if attempt < llm_retries:
                print(f"Story-aware LLM omitted {len(missing_indices)} caption(s) in one batch; retrying.", file=sys.stderr)

        if missing_indices:
            print("Trying simplified story-aware refinement repair for one batch.", file=sys.stderr)
            repaired, repaired_summary = repair_refine_batch(
                chunk,
                input_rows,
                host,
                text_model,
                keep_alive,
                seed,
                num_ctx,
                story_summary_max_words,
            )
            if repaired:
                by_index = repaired
                missing_indices = missing_caption_indices(chunk, by_index)
                if repaired_summary:
                    new_summary = clamp_words(repaired_summary, story_summary_max_words)

        if not by_index:
            if not _handle_refine_failure("Could not parse story-aware LLM JSON after retries.", strict_refine, "Keeping fallback captions for this batch."):
                completed_items.extend(chunk)
                continue
        if missing_indices:
            _handle_refine_failure(f"Story-aware LLM still omitted {len(missing_indices)} caption(s) after retries.", strict_refine, "Keeping fallback for those item(s).")

        for item in chunk:
            if item.index in by_index and by_index[item.index]:
                item.final_caption = by_index[item.index]
            cached_refined[str(item.index)] = item.final_caption

        if not new_summary:
            recent = format_previous_captions(completed_items + chunk, story_previous_captions, story_context_max_chars)
            new_summary = clamp_words(recent.replace("\n", " "), story_summary_max_words) or story_summary
        story_summary = new_summary
        batches[batch_key] = {
            "caption_indices": [item.index for item in chunk],
            "story_summary_before": fitted_story,
            "story_summary_after": story_summary,
        }
        cache["captions"] = cached_refined
        cache["batches"] = batches
        cache["final_story_summary"] = story_summary
        write_story_refine_cache(cache_path, cache)
        completed_items.extend(chunk)

    return items
