# Local Video Captioner for Windows

This is a Windows-ready local video-to-text captioning pipeline.

It uses:

- **FFmpeg** to extract video frames and audio.
- **Ollama** to caption sampled video frames with a local vision model.
- **faster-whisper** to transcribe speech locally.
- An optional local text LLM to merge visual descriptions + speech into cleaner subtitles.
- **Editable local prompts in `prompts.toml`** so you can change caption behavior without editing Python.
- **Editable local runtime defaults in `settings.toml`** so you do not need long command lines.
- **FFmpeg subtitle burn-in** to automatically create a captioned MP4.
- **Adaptive burned-in caption sizing** so font size and caption band stay readable on landscape and vertical videos.
- **Folder batch mode** that processes every video in a folder without switching back and forth between the vision model and text model.
- **Resumable checkpoints** so interrupted runs continue from cached frames, transcripts, visual captions, story-aware refined captions, or finished outputs.
- **Story-aware refinement** so caption batches can use concise context from earlier parts of the video without overloading local model context windows.
- **Static output browser** so batch results can be reviewed from one generated `index.html`.

The implementation is split into small modules under `src/aicap`:

- `config.py` parses settings and CLI flags.
- `media.py` handles FFmpeg/ffprobe and Whisper transcription.
- `visual.py` captions frames with the vision model and a resumable JSONL cache.
- `refine.py` handles independent or story-aware text-model refinement.
- `subtitles.py` writes SRT/VTT and burns captions into MP4s.
- `pipeline.py` coordinates the stages, signatures, resume checks, and final outputs.

The default model choices are aimed at an RTX 3080 Ti 12 GB:

- Vision model: `huihui_ai/qwen2.5-vl-abliterated:latest`
- Text model: `richardyoung/qwen3-14b-abliterated:Q4_K_M`
- Whisper model: `large-v3-turbo`

Use only with lawful media you have the right to process. Do not process illegal, abusive, coercive, or exploitative material.

---

## 1. Install

Open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows.ps1
```

This script tries to install/check:

- Python 3.11+
- FFmpeg
- Ollama
- Python dependencies
- The default Ollama vision/text models

If you already have the models or want to pull them manually:

```powershell
.\install_windows.ps1 -SkipModelPull
```

Manual model pulls:

```powershell
ollama pull huihui_ai/qwen2.5-vl-abliterated:latest
ollama pull richardyoung/qwen3-14b-abliterated:Q4_K_M
```

---

## 2. Test Ollama

```powershell
.\.venv\Scripts\python.exe .\src\test_ollama.py
```

You should see `Ollama is running` and a list of installed models.

If it fails, open the Ollama app or run this in another terminal:

```powershell
ollama serve
```

---

## 2a. Test the pipeline code

Run the unit tests after editing the script:

```powershell
.\run_tests.bat
```

These tests do not call Ollama or process videos. They cover cache parsing and the refinement retry/strict-failure behavior.

---

## 2b. Keep private media out of Git

This repo is set up so local media, generated outputs, and local prompt/settings files stay out of source control:

- `input/`, `output/`, and `output_batch/` are ignored.
- `prompts.toml` and `settings.toml` are ignored.
- Commit `prompts.toml.example` and `settings.toml.example` only.

After initializing or cloning the Git repo, install the Git guards:

```powershell
.\scripts\install_git_hooks.ps1
```

The guards block commits/pushes that include local media/output paths, local config files, common media extensions, or sensitive prompt text. Files needed as templates should be committed as sanitized `*.example` files only.

---


## 3. Change defaults in `settings.toml`

Most local settings now live in:

```text
settings.toml
```

Copy `settings.toml.example` to `settings.toml`, then edit that local file when you want to change normal behavior. The local `settings.toml` is ignored by Git. The `.bat` files now only pass the input path plus a settings profile:

```text
run_captioner.bat        uses [profiles.single]
run_batch_captioner.bat  uses [profiles.batch]
run_batch_captioner_10.bat uses [profiles.batch] plus --batch-chunk-size 10
open_output_browser.bat  opens the latest generated output browser
```

Common settings to change:

```toml
[captioning]
sample_every = 2.0
caption_window_seconds = 6.0
caption_mode = "neutral"
frame_width = 960
visual_temperature = 0.0

[refinement]
refine = true
refine_batch_size = 8
refine_temperature = 0.0
llm_retries = 2
strict_refine = true
story_context = true

[models]
ollama_num_ctx = 8192
ollama_seed = 42

[burn_in]
burn_in = true
caption_placement = "below"
subtitle_font_size_percent = 2.4
subtitle_band_height_percent = 8.0

[runtime]
resume = true
force = false
recursive = false
output_browser = true
open_output_browser = false
batch_chunk_size = 0

[profiles.single]
out_dir = "output"

[profiles.batch]
out_dir = "output_batch"
```

Prompt/story wording belongs in your local ignored prompt file:

```text
prompts.toml
```

Copy `prompts.toml.example` to `prompts.toml` before customizing it. Keep personal or sensitive prompt wording out of committed files.

Command-line flags still work and override `settings.toml` for one run. For example:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\FolderOfVideos" `
  --settings-profile batch `
  --force
```

---

## 4. Caption a video

Easy mode:

```powershell
.\run_captioner.bat "C:\Path\To\video.mp4"
```

Or drag a video file onto `run_captioner.bat`. This automatically writes `output\captioned_video.mp4` with captions burned in below the video picture. Change the defaults in `settings.toml`.

Outputs will be written to `output`:

- `captioned_video.mp4` - the original video with captions burned in below the picture
- `captions.srt` - subtitles for video players/editors
- `captions.vtt` - web captions
- `captions.json` - structured final captions
- `raw_captions.json` - raw visual + speech captions before final LLM cleanup
- `transcript.json` - Whisper speech transcript


---

## 5. Batch caption a whole folder

Easy mode:

```powershell
.\run_batch_captioner.bat "C:\Path\To\FolderOfVideos"
```

Or drag a folder onto `run_batch_captioner.bat`. This is the most efficient default path for a 3080 Ti: it uses `settings.toml`, processes every video in full-folder model stages, and keeps story-aware refinement enabled. By default it samples visual evidence every 2 seconds, then merges those samples into 6-second caption beats so subtitles have time to be read. For denser visual evidence, lower `sample_every`; for slower/fewer final captions, raise `caption_window_seconds`; for faster runs, raise `sample_every`.

If you want finished outputs to appear sooner while a large folder is still running, use:

```powershell
.\run_batch_captioner_10.bat "C:\Path\To\FolderOfVideos"
```

That processes the folder in groups of 10 videos. After each group, it writes the MP4/SRT/VTT/JSON files, updates `output_batch\batch_manifest.json`, and refreshes `output_batch\index.html`. If the page is open while the run is still going, it reloads itself every 30 seconds. The captions use the same quality settings as the normal batch runner; the tradeoff is a little extra model load/unload overhead between groups.

The batch runner processes videos in this order:

1. Prepare every video, extract frames, and transcribe audio.
2. Run the **vision model** across all videos.
3. Unload the vision model.
4. Run the **text/refinement model** across all videos, using rolling story context between caption batches when `--refine` is enabled.
5. Write `.srt`, `.vtt`, `.json`, and burned-in `.mp4` files for each video.

This avoids loading and unloading the two Ollama models once per video.

Batch outputs are written under `output_batch`, one folder per source video:

```text
output_batch\Video One\captioned_video.mp4
output_batch\Video One\captions.srt
output_batch\Video One\captions.vtt
output_batch\Video One\captions.json

output_batch\Video Two\captioned_video.mp4
output_batch\Video Two\captions.srt
...
```

A manifest is also written here:

```text
output_batch\batch_manifest.json
```

The batch output browser is written here:

```text
output_batch\index.html
```

Open it in your browser to review all generated videos, jump through captions, search across outputs, and open the SRT/VTT/JSON files.

To open the latest generated browser page later, run:

```bat
open_output_browser.bat
```

It opens `output_batch\index.html` when it exists, otherwise `output\index.html`. You can also pass a specific output folder:

```bat
open_output_browser.bat "C:\Path\To\output_batch"
```

Set `batch_chunk_size = 10` in `settings.toml`, or pass `--batch-chunk-size 10`, if you want chunked output from a manual command. Use `0` to process the entire folder as one efficient batch.

Manual batch command using the settings file:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\FolderOfVideos" `
  --settings-profile batch
```

Include subfolders:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\FolderOfVideos" --recursive --refine
```

Change which extensions count as videos:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\FolderOfVideos" --video-extensions ".mp4,.mkv,.mov,.webm"
```

In batch mode, `--burn-output` is ignored because each video needs its own output file. Use `[profiles.batch].out_dir` in `settings.toml` to choose the batch output folder, or override once with `--out-dir`.


---

## 6. Resume interrupted runs

Resume is enabled by default. If the script stops, crashes, your PC sleeps, or you cancel with `Ctrl+C`, run the same command again:

```powershell
.\run_batch_captioner.bat "C:\Path\To\FolderOfVideos"
```

The script will reuse completed work instead of starting over. It stores resume data in each output folder under `_resume` and uses these files when they are still valid for the same input video, model, prompt, and settings.

It can resume/skip these stages:

- extracted frames
- Whisper transcript
- per-frame visual captions, including partially completed visual-caption runs
- refined LLM captions, including partially completed refinement batches
- final `.srt`, `.vtt`, `.json`, and burned-in `.mp4` outputs

Force a full rerun from scratch:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\FolderOfVideos" --force --refine
```

Disable resume for one run:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\FolderOfVideos" --no-resume --refine
```

Changing important settings, such as `--sample-every`, `--vision-model`, `--text-model`, `--caption-mode`, story-context settings, or prompt text in `prompts.toml`, automatically invalidates the affected cached stage and reruns it.


---

## 7. Story-aware refinement

When `--refine` is enabled, story-aware refinement is now enabled by default. Instead of refining each caption batch in isolation, the text model receives:

- a compact rolling story summary from previous batches
- a limited number of recent captions
- the current batch of visual captions and speech transcript text

This helps captions follow naturally from earlier parts of the video while keeping the prompt small enough for local 7B/14B models.
The refinement stage asks Ollama for JSON responses, uses a fixed seed by default, and retries malformed or incomplete batches. By default, it stops instead of writing fallback captions when refinement still fails after retries. If a rerun starts behaving differently after you change these settings, the affected resume cache is automatically invalidated.

Default command:

```powershell
.\run_batch_captioner.bat "C:\Path\To\FolderOfVideos"
```

Manual command with story controls:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" `
  --refine `
  --story-context `
  --story-previous-captions 18 `
  --story-context-max-chars 4000 `
  --story-summary-max-words 140
```

Useful options:

- `--story-context` - use rolling story memory. This is the default.
- `--no-story-context` - go back to independent refinement batches.
- `--story-previous-captions 18` - how many recent final captions are passed into the next batch.
- `--story-context-max-chars 4000` - approximate character budget for story memory + recent captions.
- `--story-summary-max-words 140` - maximum length of the rolling story summary.
- `--caption-window-seconds 6` - how many seconds each final caption covers. The default keeps captions readable and reduces repetitive one-frame subtitles.

- `--refine-temperature 0.0` - keep text refinement deterministic.
- `--llm-retries 2` - retry malformed or partial JSON before keeping fallback captions.
- `--strict-refine` - stop the run if refinement fails after retries instead of producing raw fallback captions.
- `--ollama-num-ctx 8192` - give long story prompts more context room. Use `0` for the model default.
- `--ollama-seed 42` - use a fixed seed for repeatable reruns. Use a negative value to disable.

For a 3080 Ti, keep `--refine-batch-size` around `8` to `16` and `--story-context-max-chars` around `3000` to `6000`. Larger values can improve continuity, but may make local models slower or less reliable.

## 8. Burned-in caption placement

By default, the script now creates:

```text
output\captioned_video.mp4
```

The default placement is `below`, which adds a black caption band underneath the original picture so captions do not cover the video content.

Use bottom overlay inside the video instead:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" --caption-placement bottom
```

Disable MP4 creation and only write `.srt/.vtt/.json`:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" --no-burn-in
```

Customize the burned-in captions proportionally:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" `
  --subtitle-font-size-percent 2.4 `
  --subtitle-band-height-percent 8 `
  --subtitle-margin-v-percent 1 `
  --video-crf 18
```

The default font size uses your percentage setting but is capped by the video's readable dimension, so vertical videos no longer get oversized captions just because they are tall. In `below` mode, the black caption band is also capped to a compact readable height.

Useful options:

- `--caption-placement below` - add a black caption band below the picture. This is the default.
- `--caption-placement bottom` - overlay captions at the bottom inside the picture.
- `--burn-output "C:\Path\To\captioned.mp4"` - choose the MP4 output path.
- `--subtitle-font-size-percent 2.4` - default proportional subtitle size. Increase for larger captions, decrease for smaller captions.
- `--subtitle-band-height-percent 8` - default black band height in `below` mode.
- `--subtitle-margin-v-percent 1` - default bottom margin for the burned-in subtitles.
- `--subtitle-font-size 36` - fixed pixel font size override. Use this only when you do not want proportional sizing.
- `--subtitle-band-height 200` - fixed pixel black band height override.
- `--subtitle-margin-v 30` - fixed pixel bottom margin override.
- `--video-crf 20` - video quality. Lower means better quality and larger files.

---

## 9. Edit prompts

Public-safe example prompt text is in:

```text
prompts.toml.example
```

Copy it to ignored `prompts.toml`, then edit the local file in Notepad, VS Code, or any text editor. The important sections are:

- `[visual].base` - the shared instruction sent to the vision model for every sampled frame.
- `[visual].explicit` - extra instruction used with `--caption-mode explicit`.
- `[visual].neutral` - extra instruction used with `--caption-mode neutral`.
- `[refine].template` - the prompt used by the text LLM when `--refine` is enabled.
- `[refine].explicit_mode_instruction` - inserted into the refine template in explicit mode.
- `[refine].neutral_mode_instruction` - inserted into the refine/story template in neutral mode.
- `[story].global_context` - optional persistent names, roles, or audience direction that should apply to every story batch.
- `[story].template` - the story-aware refinement prompt used by default when `--refine` is enabled.
- `[story].empty_story_summary` and `[story].empty_previous_captions` - fallback text for the first batch.

Run with a different prompt file:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" `
  --prompts-file .\my_prompts.toml
```

In `[refine].template`, keep these placeholders unless you know what you are changing:

```text
{mode_instruction}
{input_json}
```

In `[story].template`, keep these placeholders unless you know what you are changing:

```text
{mode_instruction}
{global_context}
{story_summary}
{previous_captions}
{input_json}
{story_summary_max_words}
```

If you write literal curly braces in prompt templates, double them like this:

```text
{{ and }}
```

---

## 10. Speed/quality settings

The default `settings.toml.example` is now quality-leaning for a 3080 Ti: `sample_every = 2.0`, `caption_window_seconds = 6.0`, story refinement on, Whisper on, deterministic temperatures, and readable burned-in captions. That is the setting used by `run_batch_captioner.bat` after you copy or install the local settings file.

For faster processing:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" --sample-every 5 --no-transcribe
```

For more detailed visual coverage:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" --sample-every 1 --refine
```

For a quick test on the first 20 sampled frames:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" --max-frames 20 --refine
```

To avoid the text LLM cleanup step:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" --sample-every 1
```

The 3080 Ti has 12 GB VRAM, so close games, browsers with video tabs, and other GPU-heavy apps before running this.

---

## 11. Change models

Use a different vision model:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" --vision-model qwen2.5vl:7b
```

Use a different text model:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\video.mp4" --text-model qwen3:14b --refine
```

---

## 12. Troubleshooting

### PowerShell blocks the installer

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows.ps1
```

This only affects the current PowerShell session.

### `ffmpeg was not found on PATH`

Open a new PowerShell window after installing FFmpeg, then re-run the installer. If needed, install FFmpeg manually and add its `bin` folder to PATH.

### `Could not connect to Ollama`

Open the Ollama app, or run:

```powershell
ollama serve
```

Then re-run the captioner.

### Whisper CUDA fails

The script automatically falls back to CPU when `--whisper-device auto` is used. It will be slower but should still work.

### The captions are too vague

Try:

```powershell
--sample-every 0.5 --refine
```

### The run is too slow

Try:

```powershell
--sample-every 2 --no-transcribe
```

or remove `--refine`.

### Audio extraction fails on one file

Some videos have no audio stream, a corrupt audio stream, or an audio codec FFmpeg cannot decode cleanly. The current script probes for audio first and will continue with visual-only captions if audio is missing or fails to extract.

To intentionally skip all Whisper/audio work:

```powershell
.\.venv\Scripts\python.exe .\src\video_captioner.py "C:\Path\To\Folder" --no-transcribe
```

Or set this in `settings.toml`:

```toml
[speech]
no_transcribe = true
```
