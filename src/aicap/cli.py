from __future__ import annotations

from .config import parse_args, validate_args
from .jobs import build_jobs, parse_extensions
from .ollama import unload_ollama_model
from .pipeline import run_stage_prepare_and_transcribe, run_stage_refine, run_stage_visual_captions, write_outputs_and_burn
from .prompts import load_prompts
from .util import write_json


def main() -> None:
    args = parse_args()
    validate_args(args)

    extensions = parse_extensions(args.video_extensions)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, is_batch = build_jobs(args.video, out_dir, extensions, args.recursive)
    if is_batch:
        print(f"Batch mode: found {len(jobs)} video(s).")
        print("Model order: all visual-caption work first, then all text-refinement work.")
        if args.burn_output:
            print("Note: --burn-output is ignored in folder batch mode. Each video gets its own captioned_video.mp4.")
    else:
        print(f"Single-video mode: {jobs[0].video_path.name}")

    prompts_path = args.prompts_file.expanduser().resolve()
    prompts = load_prompts(prompts_path)

    run_stage_prepare_and_transcribe(jobs, args, prompts, is_batch)
    used_vision_model = run_stage_visual_captions(jobs, args, prompts, is_batch)

    if args.refine and used_vision_model:
        print(f"\nUnloading vision model before text refinement: {args.vision_model}")
        unload_ollama_model(args.ollama_host, args.vision_model)

    used_text_model = run_stage_refine(jobs, args, prompts, is_batch)
    if used_text_model:
        unload_ollama_model(args.ollama_host, args.text_model)

    manifest = write_outputs_and_burn(jobs, args, prompts, is_batch)
    if is_batch:
        manifest_path = out_dir / "batch_manifest.json"
        write_json(manifest_path, manifest)
        print(f"\nBatch manifest: {manifest_path}")

    print("\nDone.")
