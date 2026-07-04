from __future__ import annotations

import json
import os
import webbrowser
from dataclasses import asdict
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .jobs import captioned_video_output_path
from .models import VideoJob
from .util import atomic_write_text


def relative_url(base_dir: Path, target: Path) -> str:
    relative = os.path.relpath(target.resolve(), base_dir.resolve())
    return quote(relative.replace(os.sep, "/"), safe="/._-~")


def existing_file(base_dir: Path, path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None or not path.exists():
        return None
    return {
        "name": path.name,
        "href": relative_url(base_dir, path),
        "path": str(path),
    }


def browser_job_payload(base_dir: Path, job: VideoJob, args: Any, is_batch: bool) -> Dict[str, Any]:
    files = {
        "captioned_video": existing_file(base_dir, captioned_video_output_path(job, args, is_batch)),
        "captions_srt": existing_file(base_dir, job.out_dir / "captions.srt"),
        "captions_vtt": existing_file(base_dir, job.out_dir / "captions.vtt"),
        "captions_json": existing_file(base_dir, job.out_dir / "captions.json"),
        "raw_captions_json": existing_file(base_dir, job.out_dir / "raw_captions.json"),
        "refined_captions_json": existing_file(base_dir, job.out_dir / "refined_captions.json"),
        "transcript_json": existing_file(base_dir, job.out_dir / "transcript.json"),
    }
    captions = [asdict(item) for item in (job.items or [])]
    caption_text = " ".join(
        " ".join(
            [
                str(item.get("final_caption", "")),
                str(item.get("visual_caption", "")),
                str(item.get("speech", "")),
            ]
        )
        for item in captions
    )
    return {
        "title": job.video_path.stem,
        "source": str(job.video_path),
        "output_dir": str(job.out_dir),
        "duration": job.duration,
        "caption_count": len(captions),
        "search_text": f"{job.video_path.stem} {job.video_path.name} {caption_text}".lower(),
        "files": files,
        "captions": captions,
    }


def safe_json_for_script(data: Any) -> str:
    return (
        json.dumps(data, ensure_ascii=False, indent=2)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def write_output_browser(out_dir: Path, jobs: List[VideoJob], args: Any, is_batch: bool) -> Path:
    out_dir = out_dir.expanduser().resolve()
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_batch": is_batch,
        "job_count": len(jobs),
        "caption_count": sum(len(job.items or []) for job in jobs),
        "jobs": [browser_job_payload(out_dir, job, args, is_batch) for job in jobs],
    }
    html = render_output_browser(payload)
    path = out_dir / "index.html"
    atomic_write_text(path, html)
    return path


def open_in_browser(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())


def render_output_browser(payload: Dict[str, Any]) -> str:
    title = "AICap Outputs"
    payload_json = safe_json_for_script(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f5ef;
      --panel: #ffffff;
      --ink: #1f2428;
      --muted: #667075;
      --line: #d9ddd7;
      --accent: #176b66;
      --accent-2: #b85032;
      --soft: #eef5f2;
      --warn: #f7eadf;
      --shadow: 0 10px 24px rgba(31, 36, 40, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 "Segoe UI", Arial, sans-serif;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(220px, 420px);
      gap: 18px;
      align-items: center;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 3;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 22px; font-weight: 700; letter-spacing: 0; }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary span {{
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fafafa;
    }}
    input[type="search"] {{
      width: 100%;
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      font: inherit;
      color: var(--ink);
      background: #fbfbfb;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(240px, 340px) minmax(0, 1fr);
      min-height: calc(100vh - 78px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: #fbfaf7;
      overflow: auto;
      max-height: calc(100vh - 78px);
    }}
    .video-list {{
      display: grid;
      gap: 8px;
      padding: 12px;
    }}
    .video-button {{
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px;
      cursor: pointer;
      font: inherit;
      color: var(--ink);
    }}
    .video-button:hover {{ border-color: #9bb9b3; }}
    .video-button.active {{
      border-color: var(--accent);
      background: var(--soft);
    }}
    .video-title {{
      display: block;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    .video-meta {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .workspace {{
      padding: 18px;
      overflow: auto;
      max-height: calc(100vh - 78px);
    }}
    .viewer {{
      display: grid;
      grid-template-columns: minmax(300px, 1.2fr) minmax(280px, 0.8fr);
      gap: 18px;
      align-items: start;
    }}
    .media-pane, .caption-pane {{
      min-width: 0;
    }}
    video {{
      display: block;
      width: 100%;
      max-height: 62vh;
      background: #111;
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .empty-media {{
      min-height: 220px;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--warn);
      color: var(--muted);
      text-align: center;
      padding: 18px;
    }}
    .details {{
      margin-top: 14px;
      display: grid;
      gap: 10px;
    }}
    .details h2 {{
      font-size: 19px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .source {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      border: 1px solid #b7d1cb;
      border-radius: 6px;
      padding: 6px 9px;
      background: #f8fcfb;
    }}
    .links a:hover {{ border-color: var(--accent); }}
    .caption-toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .caption-toolbar h3 {{
      font-size: 16px;
      line-height: 1.2;
    }}
    .toggle-row {{
      display: flex;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .caption-list {{
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 162px);
      overflow: auto;
      padding-right: 4px;
    }}
    .caption-row {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px;
      cursor: pointer;
    }}
    .caption-row:hover {{ border-color: #9bb9b3; }}
    .caption-row.active {{
      border-color: var(--accent-2);
      background: #fff7f2;
    }}
    .time {{
      display: inline-block;
      min-width: 88px;
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 650;
    }}
    .caption-text {{
      display: block;
      margin-top: 5px;
      overflow-wrap: anywhere;
    }}
    .secondary {{
      display: none;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      overflow-wrap: anywhere;
    }}
    body.show-raw .secondary {{ display: block; }}
    mark {{
      background: #ffe6a8;
      color: inherit;
      padding: 0 2px;
      border-radius: 3px;
    }}
    .no-results {{
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 920px) {{
      header {{ grid-template-columns: 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      aside {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
        max-height: 240px;
      }}
      .workspace {{ max-height: none; }}
      .viewer {{ grid-template-columns: 1fr; }}
      .caption-list {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AICap Outputs</h1>
      <div class="summary" id="summary"></div>
    </div>
    <input id="search" type="search" autocomplete="off" placeholder="Search videos and captions">
  </header>
  <main>
    <aside>
      <div class="video-list" id="videoList"></div>
    </aside>
    <section class="workspace">
      <div class="viewer">
        <div class="media-pane" id="mediaPane"></div>
        <div class="caption-pane">
          <div class="caption-toolbar">
            <h3 id="captionHeading">Captions</h3>
            <label class="toggle-row"><input id="rawToggle" type="checkbox"> Visual and speech</label>
          </div>
          <div class="caption-list" id="captionList"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const DATA = {payload_json};
    const state = {{ selected: 0, query: "", activeCaption: -1 }};
    const videoList = document.getElementById("videoList");
    const mediaPane = document.getElementById("mediaPane");
    const captionList = document.getElementById("captionList");
    const captionHeading = document.getElementById("captionHeading");
    const search = document.getElementById("search");
    const rawToggle = document.getElementById("rawToggle");
    const summary = document.getElementById("summary");

    function fmtTime(value) {{
      const total = Math.max(0, Math.floor(Number(value) || 0));
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      const base = `${{String(m).padStart(2, "0")}}:${{String(s).padStart(2, "0")}}`;
      return h ? `${{h}}:${{base}}` : base;
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function highlight(value) {{
      const text = escapeHtml(value);
      const query = state.query.trim();
      if (!query) return text;
      const safe = query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&");
      return text.replace(new RegExp(safe, "ig"), match => `<mark>${{match}}</mark>`);
    }}

    function filteredJobs() {{
      const q = state.query.trim().toLowerCase();
      if (!q) return DATA.jobs.map((job, index) => [job, index]);
      return DATA.jobs.map((job, index) => [job, index]).filter(([job]) => job.search_text.includes(q));
    }}

    function renderSummary() {{
      summary.innerHTML = `
        <span>${{DATA.job_count}} video${{DATA.job_count === 1 ? "" : "s"}}</span>
        <span>${{DATA.caption_count}} captions</span>
        <span>Generated ${{escapeHtml(DATA.generated_at)}}</span>
      `;
    }}

    function renderList() {{
      const rows = filteredJobs();
      if (!rows.length) {{
        videoList.innerHTML = `<div class="no-results">No matching outputs.</div>`;
        return;
      }}
      if (!rows.some(([, index]) => index === state.selected)) {{
        state.selected = rows[0][1];
      }}
      videoList.innerHTML = rows.map(([job, index]) => `
        <button class="video-button ${{index === state.selected ? "active" : ""}}" data-index="${{index}}">
          <span class="video-title">${{highlight(job.title)}}</span>
          <span class="video-meta">${{job.caption_count}} captions</span>
        </button>
      `).join("");
      for (const button of videoList.querySelectorAll("button")) {{
        button.addEventListener("click", () => {{
          state.selected = Number(button.dataset.index);
          state.activeCaption = -1;
          render();
        }});
      }}
    }}

    function fileLinks(job) {{
      const labels = [
        ["captioned_video", "MP4"],
        ["captions_srt", "SRT"],
        ["captions_vtt", "VTT"],
        ["captions_json", "Captions JSON"],
        ["raw_captions_json", "Raw JSON"],
        ["refined_captions_json", "Refined JSON"],
        ["transcript_json", "Transcript"]
      ];
      return labels
        .filter(([key]) => job.files[key])
        .map(([key, label]) => `<a href="${{job.files[key].href}}">${{label}}</a>`)
        .join("");
    }}

    function renderMedia() {{
      const job = DATA.jobs[state.selected];
      const video = job.files.captioned_video;
      const links = fileLinks(job);
      const media = video
        ? `<video id="player" controls preload="metadata" src="${{video.href}}"></video>`
        : `<div class="empty-media">No captioned MP4 was written for this output.</div>`;
      mediaPane.innerHTML = `
        ${{media}}
        <div class="details">
          <h2>${{highlight(job.title)}}</h2>
          <p class="source">${{escapeHtml(job.source)}}</p>
          <div class="links">${{links}}</div>
        </div>
      `;
      const player = document.getElementById("player");
      if (player) {{
        player.addEventListener("timeupdate", () => updateActiveCaption(player.currentTime));
      }}
    }}

    function filteredCaptions(job) {{
      const q = state.query.trim().toLowerCase();
      if (!q) return job.captions;
      return job.captions.filter(item =>
        String(item.final_caption || "").toLowerCase().includes(q) ||
        String(item.visual_caption || "").toLowerCase().includes(q) ||
        String(item.speech || "").toLowerCase().includes(q)
      );
    }}

    function renderCaptions() {{
      const job = DATA.jobs[state.selected];
      const captions = filteredCaptions(job);
      captionHeading.textContent = `Captions (${{captions.length}})`;
      if (!captions.length) {{
        captionList.innerHTML = `<div class="no-results">No matching captions.</div>`;
        return;
      }}
      captionList.innerHTML = captions.map(item => `
        <button class="caption-row ${{item.index === state.activeCaption ? "active" : ""}}" data-start="${{item.start}}" data-index="${{item.index}}">
          <span class="time">${{fmtTime(item.start)}} - ${{fmtTime(item.end)}}</span>
          <span class="caption-text">${{highlight(item.final_caption)}}</span>
          <span class="secondary">
            <strong>Visual:</strong> ${{highlight(item.visual_caption)}}<br>
            <strong>Speech:</strong> ${{highlight(item.speech || "")}}
          </span>
        </button>
      `).join("");
      for (const row of captionList.querySelectorAll(".caption-row")) {{
        row.addEventListener("click", () => {{
          const player = document.getElementById("player");
          if (player) {{
            player.currentTime = Number(row.dataset.start) || 0;
            player.play().catch(() => {{}});
          }}
          state.activeCaption = Number(row.dataset.index);
          renderCaptions();
        }});
      }}
    }}

    function updateActiveCaption(time) {{
      const job = DATA.jobs[state.selected];
      const hit = job.captions.find(item => time >= item.start && time < item.end);
      const next = hit ? hit.index : -1;
      if (next !== state.activeCaption) {{
        state.activeCaption = next;
        renderCaptions();
      }}
    }}

    function render() {{
      renderSummary();
      renderList();
      renderMedia();
      renderCaptions();
    }}

    search.addEventListener("input", () => {{
      state.query = search.value;
      state.activeCaption = -1;
      render();
    }});
    rawToggle.addEventListener("change", () => {{
      document.body.classList.toggle("show-raw", rawToggle.checked);
    }});

    render();
  </script>
</body>
</html>
"""
