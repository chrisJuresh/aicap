from __future__ import annotations

from .config import parse_args, validate_args
from .jobs import build_jobs, parse_extensions
from .ollama import unload_ollama_model
from .output_browser import open_in_browser, write_output_browser
from .pipeline import run_stage_prepare_and_transcribe, run_stage_refine, run_stage_visual_captions, write_outputs_and_burn
from .prompts import load_prompts
from .util import write_json


def chunk_jobs(jobs, chunk_size: int):
    if chunk_size <= 0 or chunk_size >= len(jobs):
        return [jobs]
    return [jobs[index : index + chunk_size] for index in range(0, len(jobs), chunk_size)]


def run_jobs(jobs, args, prompts, is_batch: bool):
    run_stage_prepare_and_transcribe(jobs, args, prompts, is_batch)
    used_vision_model = run_stage_visual_captions(jobs, args, prompts, is_batch)

    if args.refine and used_vision_model:
        print(f"\nUnloading vision model before text refinement: {args.vision_model}")
        unload_ollama_model(args.ollama_host, args.vision_model)

    used_text_model = run_stage_refine(jobs, args, prompts, is_batch)
    if used_text_model:
        unload_ollama_model(args.ollama_host, args.text_model)

    return write_outputs_and_burn(jobs, args, prompts, is_batch)


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

    chunks = chunk_jobs(jobs, args.batch_chunk_size if is_batch else 0)
    if is_batch and len(chunks) > 1:
        print(f"Chunked batch mode: writing outputs every {args.batch_chunk_size} video(s).")

    manifest = []
    completed_jobs = []
    browser_opened = False
    for chunk_index, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            start = len(completed_jobs) + 1
            end = len(completed_jobs) + len(chunk)
            print(f"\n=== Output batch {chunk_index}/{len(chunks)}: videos {start}-{end} of {len(jobs)} ===")

        manifest.extend(run_jobs(chunk, args, prompts, is_batch))
        completed_jobs.extend(chunk)

        if is_batch:
            manifest_path = out_dir / "batch_manifest.json"
            write_json(manifest_path, manifest)
            print(f"\nBatch manifest: {manifest_path}")

        if args.output_browser:
            browser_path = write_output_browser(
                out_dir,
                completed_jobs,
                args,
                is_batch,
                total_jobs=len(jobs),
                auto_refresh=is_batch and len(completed_jobs) < len(jobs),
            )
            print(f"Output browser: {browser_path}")
            if args.open_output_browser and not browser_opened:
                open_in_browser(browser_path)
                browser_opened = True

    print("\nDone.")
