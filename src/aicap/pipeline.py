from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .cache import checkpoint_matches, load_visual_cache, resume_dir, resume_enabled, write_checkpoint
from .captions import build_items, load_caption_items, load_transcript, save_items_snapshot
from .jobs import captioned_video_output_path, manifest_row_for_job, outputs_are_complete
from .media import ffprobe_duration, load_whisper_model, transcribe_audio_with_model, extract_frames
from .models import VideoJob
from .ollama import unload_ollama_model
from .prompts import get_prompt, visual_prompt
from .refine import load_story_refine_cache, refine_with_llm
from .subtitles import SUBTITLE_LAYOUT_VERSION, burn_captions_to_video, write_srt, write_vtt
from .util import die, file_fingerprint, items_hash, json_fingerprint, read_json, write_json
from .visual import caption_frames_cached


CAPTION_ITEM_VERSION = 2
STORY_REFINEMENT_RULES_VERSION = 2


def prepare_signature(job: VideoJob, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "source": file_fingerprint(job.video_path),
        "sample_every": args.sample_every,
        "frame_width": args.frame_width,
        "jpeg_quality": args.jpeg_quality,
        "no_transcribe": args.no_transcribe,
        "whisper_model": None if args.no_transcribe else args.whisper_model,
        "language": None if args.no_transcribe else args.language,
    }


def visual_cache_signature(job: VideoJob, args: argparse.Namespace, prompts: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    return {
        "source": file_fingerprint(job.video_path),
        "sample_every": args.sample_every,
        "frame_width": args.frame_width,
        "jpeg_quality": args.jpeg_quality,
        "max_frames": args.max_frames,
        "caption_mode": args.caption_mode,
        "vision_model": args.vision_model,
        "visual_temperature": args.visual_temperature,
        "ollama_seed": args.ollama_seed,
        "ollama_num_ctx": args.ollama_num_ctx,
        "visual_prompt_hash": hashlib.sha256(visual_prompt(args.caption_mode, prompts).encode("utf-8")).hexdigest(),
        "prepare_hash": json_fingerprint(prepare_signature(job, args)),
    }


def visual_signature(job: VideoJob, args: argparse.Namespace, prompts: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    signature = visual_cache_signature(job, args, prompts)
    signature.update(
        {
            "caption_item_version": CAPTION_ITEM_VERSION,
            "caption_window_seconds": args.caption_window_seconds,
        }
    )
    return signature


def refine_signature(job: VideoJob, args: argparse.Namespace, prompts: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    mode_key = "explicit_mode_instruction" if args.caption_mode == "explicit" else "neutral_mode_instruction"
    prompt_blob = get_prompt(prompts, "refine", mode_key) + "\n" + get_prompt(prompts, "refine", "template")
    story_prompt_blob = "\n".join(
        [
            get_prompt(prompts, "story", "template"),
            get_prompt(prompts, "story", "global_context"),
            get_prompt(prompts, "story", "empty_story_summary"),
            get_prompt(prompts, "story", "empty_previous_captions"),
        ]
    )
    return {
        "source": file_fingerprint(job.video_path),
        "caption_item_version": CAPTION_ITEM_VERSION,
        "caption_window_seconds": args.caption_window_seconds,
        "caption_mode": args.caption_mode,
        "text_model": args.text_model,
        "refine_batch_size": args.refine_batch_size,
        "refine_temperature": args.refine_temperature,
        "llm_retries": args.llm_retries,
        "strict_refine": args.strict_refine,
        "ollama_seed": args.ollama_seed,
        "ollama_num_ctx": args.ollama_num_ctx,
        "story_context": args.story_context,
        "story_previous_captions": args.story_previous_captions,
        "story_context_max_chars": args.story_context_max_chars,
        "story_summary_max_words": args.story_summary_max_words,
        "story_rules_version": STORY_REFINEMENT_RULES_VERSION,
        "story_prompt_hash": hashlib.sha256(story_prompt_blob.encode("utf-8")).hexdigest(),
        "refine_prompt_hash": hashlib.sha256(prompt_blob.encode("utf-8")).hexdigest(),
        "raw_items_hash": items_hash(job.items or []),
    }


def outputs_signature(job: VideoJob, args: argparse.Namespace, prompts: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
    signature: Dict[str, Any] = {
        "source": file_fingerprint(job.video_path),
        "items_hash": items_hash(job.items or []),
        "caption_item_version": CAPTION_ITEM_VERSION,
        "caption_window_seconds": args.caption_window_seconds,
        "burn_in": args.burn_in,
        "caption_placement": args.caption_placement,
        "subtitle_layout_version": SUBTITLE_LAYOUT_VERSION,
        "subtitle_font": args.subtitle_font,
        "subtitle_font_size": args.subtitle_font_size,
        "subtitle_font_size_percent": args.subtitle_font_size_percent,
        "subtitle_margin_v": args.subtitle_margin_v,
        "subtitle_margin_v_percent": args.subtitle_margin_v_percent,
        "subtitle_band_height": args.subtitle_band_height,
        "subtitle_band_height_percent": args.subtitle_band_height_percent,
        "video_crf": args.video_crf,
        "video_preset": args.video_preset,
    }
    if prompts is not None:
        mode_key = "explicit_mode_instruction" if args.caption_mode == "explicit" else "neutral_mode_instruction"
        signature.update(
            {
                "sample_every": args.sample_every,
                "caption_window_seconds": args.caption_window_seconds,
                "frame_width": args.frame_width,
                "jpeg_quality": args.jpeg_quality,
                "max_frames": args.max_frames,
                "no_transcribe": args.no_transcribe,
                "whisper_model": None if args.no_transcribe else args.whisper_model,
                "language": None if args.no_transcribe else args.language,
                "caption_mode": args.caption_mode,
                "refine": args.refine,
                "vision_model": args.vision_model,
                "text_model": args.text_model if args.refine else None,
                "refine_batch_size": args.refine_batch_size if args.refine else None,
                "story_context": args.story_context if args.refine else None,
                "story_previous_captions": args.story_previous_captions if args.refine else None,
                "story_context_max_chars": args.story_context_max_chars if args.refine else None,
                "story_summary_max_words": args.story_summary_max_words if args.refine else None,
                "visual_temperature": args.visual_temperature,
                "refine_temperature": args.refine_temperature if args.refine else None,
                "llm_retries": args.llm_retries if args.refine else None,
                "strict_refine": args.strict_refine if args.refine else None,
                "ollama_seed": args.ollama_seed,
                "ollama_num_ctx": args.ollama_num_ctx,
                "visual_prompt_hash": hashlib.sha256(visual_prompt(args.caption_mode, prompts).encode("utf-8")).hexdigest(),
                "refine_prompt_hash": hashlib.sha256((get_prompt(prompts, "refine", mode_key) + "\n" + get_prompt(prompts, "refine", "template")).encode("utf-8")).hexdigest()
                if args.refine
                else None,
                "story_rules_version": STORY_REFINEMENT_RULES_VERSION if args.refine and args.story_context else None,
                "story_prompt_hash": hashlib.sha256(
                    "\n".join(
                        [
                            get_prompt(prompts, "story", "template"),
                            get_prompt(prompts, "story", "global_context"),
                            get_prompt(prompts, "story", "empty_story_summary"),
                            get_prompt(prompts, "story", "empty_previous_captions"),
                        ]
                    ).encode("utf-8")
                ).hexdigest()
                if args.refine and args.story_context
                else None,
            }
        )
    return signature


def run_stage_prepare_and_transcribe(jobs: List[VideoJob], args: argparse.Namespace, prompts: Dict[str, Dict[str, str]], is_batch: bool) -> None:
    whisper_model_obj: Optional[Any] = None
    for job_index, job in enumerate(jobs, start=1):
        print(f"\n[{job_index}/{len(jobs)}] Preparing: {job.video_path.name}")
        job.out_dir.mkdir(parents=True, exist_ok=True)
        job.work_dir.mkdir(parents=True, exist_ok=True)

        if resume_enabled(args):
            final_items = load_caption_items(job.out_dir / "captions.json")
            if final_items:
                job.items = final_items
                out_sig = outputs_signature(job, args, prompts)
                if checkpoint_matches(job, "outputs", out_sig) and outputs_are_complete(job, args, is_batch):
                    job.duration = ffprobe_duration(job.video_path)
                    job.frames = []
                    job.transcript = []
                    print("Resume: final outputs are complete; skipping prepare/transcribe for this video.")
                    continue
                job.items = None

        sig = prepare_signature(job, args)
        frames_dir = job.work_dir / "frames"
        existing_frames = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []
        transcript_path = job.out_dir / "transcript.json"
        can_reuse_transcript = args.no_transcribe or transcript_path.exists()
        if resume_enabled(args) and checkpoint_matches(job, "prepare", sig) and existing_frames and can_reuse_transcript:
            job.duration = ffprobe_duration(job.video_path)
            job.frames = existing_frames
            job.transcript = [] if args.no_transcribe else load_transcript(transcript_path)
            print(f"Resume: skipping prepare/transcribe; using {len(job.frames)} existing frame(s).")
            continue

        job.duration = ffprobe_duration(job.video_path)
        if job.duration:
            print(f"Video duration: {job.duration:.1f}s")
        job.frames = extract_frames(job.video_path, job.work_dir, args.sample_every, args.frame_width, args.jpeg_quality)
        if args.max_frames is not None:
            print(f"Test mode: processing only first {args.max_frames} frame(s).")
        if args.no_transcribe:
            job.transcript = []
        else:
            if whisper_model_obj is None:
                whisper_model_obj = load_whisper_model(args.whisper_model, args.whisper_device)
            job.transcript = transcribe_audio_with_model(job.video_path, job.work_dir, whisper_model_obj, args.language)
            write_json(transcript_path, [asdict(s) for s in job.transcript])
            print(f"Wrote {transcript_path}")
        write_checkpoint(job, "prepare", sig, {"duration": job.duration, "frame_count": len(job.frames or []), "transcript_count": len(job.transcript or [])})


def run_stage_visual_captions(jobs: List[VideoJob], args: argparse.Namespace, prompts: Dict[str, Dict[str, str]], is_batch: bool) -> bool:
    print(f"\nStage: visual captions for all videos using {args.vision_model}")
    used_model = False
    for job_index, job in enumerate(jobs, start=1):
        print(f"\n[{job_index}/{len(jobs)}] Visual captioning: {job.video_path.name}")
        if job.items is not None and resume_enabled(args):
            out_sig = outputs_signature(job, args, prompts)
            if checkpoint_matches(job, "outputs", out_sig) and outputs_are_complete(job, args, is_batch):
                print("Resume: final outputs are complete; skipping visual captioning for this video.")
                continue
        frames = job.frames or []
        transcript = job.transcript or []
        sig = visual_signature(job, args, prompts)
        cache_sig = visual_cache_signature(job, args, prompts)
        raw_path = job.out_dir / "raw_captions.json"
        visual_cache_path = resume_dir(job) / "visual_frame_captions.jsonl"
        if resume_enabled(args) and checkpoint_matches(job, "visual", sig) and raw_path.exists():
            loaded_items = load_caption_items(raw_path)
            if loaded_items:
                job.items = loaded_items
                print(f"Resume: skipping visual captioning; loaded {len(job.items)} item(s) from {raw_path}.")
                continue
        if not resume_enabled(args) or not checkpoint_matches(job, "visual_cache", cache_sig):
            try:
                visual_cache_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            write_checkpoint(job, "visual_cache", cache_sig, {"cache": str(visual_cache_path)})
        frames_for_cache_check = frames[: args.max_frames] if args.max_frames is not None else frames
        cached_before = load_visual_cache(visual_cache_path)
        if any(frame.name not in cached_before for frame in frames_for_cache_check):
            used_model = True
        visual_captions = caption_frames_cached(
            frames,
            args.ollama_host,
            args.vision_model,
            args.caption_mode,
            args.max_frames,
            prompts,
            args.ollama_keep_alive,
            visual_cache_path,
            args.visual_temperature,
            args.ollama_seed,
            args.ollama_num_ctx,
        )
        job.items = build_items(
            frames,
            visual_captions,
            transcript,
            args.sample_every,
            job.duration,
            args.max_frames,
            args.caption_window_seconds,
        )
        save_items_snapshot(raw_path, job.items)
        write_checkpoint(job, "visual", sig, {"item_count": len(job.items), "raw_captions": str(raw_path)})
        print(f"Wrote {raw_path}")
    return used_model


def run_stage_refine(jobs: List[VideoJob], args: argparse.Namespace, prompts: Dict[str, Dict[str, str]], is_batch: bool) -> bool:
    if not args.refine:
        return False
    print(f"\nStage: LLM refinement for all videos using {args.text_model}")
    used_model = False
    for job_index, job in enumerate(jobs, start=1):
        print(f"\n[{job_index}/{len(jobs)}] Refining: {job.video_path.name}")
        if job.items is None:
            die(f"Internal error: no caption items for {job.video_path}")
        if resume_enabled(args):
            out_sig = outputs_signature(job, args, prompts)
            if checkpoint_matches(job, "outputs", out_sig) and outputs_are_complete(job, args, is_batch):
                print("Resume: final outputs are complete; skipping refinement for this video.")
                continue
        sig = refine_signature(job, args, prompts)
        refined_path = job.out_dir / "refined_captions.json"
        refine_cache_path = resume_dir(job) / "refined_caption_cache.json"
        if resume_enabled(args) and checkpoint_matches(job, "refine", sig) and refined_path.exists():
            loaded_items = load_caption_items(refined_path)
            if loaded_items:
                job.items = loaded_items
                print(f"Resume: skipping refinement; loaded {len(job.items)} item(s) from {refined_path}.")
                continue
        if not resume_enabled(args) or not checkpoint_matches(job, "refine_cache", sig):
            try:
                refine_cache_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            write_checkpoint(job, "refine_cache", sig, {"cache": str(refine_cache_path)})
        cached_refined_payload = load_story_refine_cache(refine_cache_path) if args.story_context else read_json(refine_cache_path, {})
        cached_refined_map = cached_refined_payload.get("captions", {}) if args.story_context and isinstance(cached_refined_payload, dict) else cached_refined_payload
        if not isinstance(cached_refined_map, dict) or any(str(item.index) not in cached_refined_map for item in job.items):
            used_model = True
        job.items = refine_with_llm(
            job.items,
            args.ollama_host,
            args.text_model,
            args.caption_mode,
            args.refine_batch_size,
            prompts,
            args.ollama_keep_alive,
            args.refine_temperature,
            args.ollama_seed,
            args.ollama_num_ctx,
            args.llm_retries,
            args.strict_refine,
            cache_path=refine_cache_path,
            story_context=args.story_context,
            story_previous_captions=args.story_previous_captions,
            story_context_max_chars=args.story_context_max_chars,
            story_summary_max_words=args.story_summary_max_words,
        )
        save_items_snapshot(refined_path, job.items)
        write_checkpoint(job, "refine", sig, {"item_count": len(job.items), "refined_captions": str(refined_path)})
        print(f"Wrote {refined_path}")
    return used_model


def write_outputs_and_burn(jobs: List[VideoJob], args: argparse.Namespace, prompts: Dict[str, Dict[str, str]], is_batch: bool) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    print("\nStage: writing subtitle files and captioned videos")
    for job_index, job in enumerate(jobs, start=1):
        print(f"\n[{job_index}/{len(jobs)}] Writing outputs: {job.video_path.name}")
        if job.items is None:
            die(f"Internal error: no caption items for {job.video_path}")
        sig = outputs_signature(job, args, prompts)
        captioned_video_path = captioned_video_output_path(job, args, is_batch)
        if resume_enabled(args) and checkpoint_matches(job, "outputs", sig) and outputs_are_complete(job, args, is_batch):
            print("Resume: skipping output writing/burn-in; final files already exist.")
            manifest.append(manifest_row_for_job(job, args, is_batch))
            if not args.keep_temp:
                try:
                    shutil.rmtree(job.work_dir)
                except Exception:
                    pass
            continue
        captions_json_path = job.out_dir / "captions.json"
        captions_srt_path = job.out_dir / "captions.srt"
        captions_vtt_path = job.out_dir / "captions.vtt"
        write_json(captions_json_path, [asdict(i) for i in job.items])
        write_srt(captions_srt_path, job.items)
        write_vtt(captions_vtt_path, job.items)
        if args.burn_in:
            assert captioned_video_path is not None
            burn_captions_to_video(
                video_path=job.video_path,
                srt_path=captions_srt_path,
                output_path=captioned_video_path,
                placement=args.caption_placement,
                font=args.subtitle_font,
                font_size=args.subtitle_font_size,
                font_size_percent=args.subtitle_font_size_percent,
                margin_v=args.subtitle_margin_v,
                margin_v_percent=args.subtitle_margin_v_percent,
                band_height=args.subtitle_band_height,
                band_height_percent=args.subtitle_band_height_percent,
                crf=args.video_crf,
                preset=args.video_preset,
            )
        write_checkpoint(job, "outputs", sig, {"outputs": manifest_row_for_job(job, args, is_batch)})
        if not args.keep_temp:
            try:
                shutil.rmtree(job.work_dir)
            except Exception:
                pass
        row = manifest_row_for_job(job, args, is_batch)
        manifest.append(row)
        print(f"SRT:  {captions_srt_path}")
        print(f"VTT:  {captions_vtt_path}")
        print(f"JSON: {captions_json_path}")
        if captioned_video_path:
            print(f"MP4:  {captioned_video_path}")
    return manifest
