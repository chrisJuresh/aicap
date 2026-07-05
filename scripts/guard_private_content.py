from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]

BLOCKED_DIRS = {
    ".venv",
    "input",
    "output",
    "output_batch",
}

BLOCKED_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".gif",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

LOCAL_ONLY_FILENAMES = {
    "model_io.jsonl",
    "prompts.toml",
    "settings.toml",
}

TEXT_SUFFIXES = {
    ".bat",
    ".css",
    ".example",
    ".gitignore",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".srt",
    ".toml",
    ".txt",
    ".vtt",
    ".yaml",
    ".yml",
}

# Stored encoded so the guard itself does not contain the content it blocks.
ENCODED_TERMS = (
    "cG9ybg==",
    "cmVkZ2lmcw==",
    "Z29vbmVy",
    "Y3Vjaw==",
    "ZXJvdGlj",
    "bnNmdw==",
    "c2V4dWFs",
    "bnVkZQ==",
)


def blocked_terms() -> List[str]:
    return [base64.b64decode(term).decode("utf-8") for term in ENCODED_TERMS]


def run_git(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, text=False, capture_output=True)


def git_paths(args: List[str]) -> List[Path]:
    result = run_git(args)
    if result.returncode != 0:
        print("Git file scan failed.", file=sys.stderr)
        return []
    names = [name.decode("utf-8", errors="replace") for name in result.stdout.split(b"\0") if name]
    return [ROOT / name for name in names]


def staged_files() -> List[Path]:
    result = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"])
    if result.returncode != 0:
        print("Not inside a Git repository; staged scan skipped.", file=sys.stderr)
        return []
    names = [name.decode("utf-8", errors="replace") for name in result.stdout.split(b"\0") if name]
    return [ROOT / name for name in names]


def tracked_files() -> List[Path]:
    return git_paths(["ls-files", "-z"])


def all_candidate_files() -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_parts = Path(dirpath).relative_to(ROOT).parts
        if any(part in BLOCKED_DIRS or part == ".git" for part in rel_parts):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in BLOCKED_DIRS and d != ".git" and d != "__pycache__"]
        for filename in filenames:
            files.append(Path(dirpath) / filename)
    return files


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def path_issues(path: Path) -> List[str]:
    issues: List[str] = []
    relative = path.resolve().relative_to(ROOT)
    parts = relative.parts
    lower_name = path.name.lower()
    if any(part in BLOCKED_DIRS for part in parts):
        issues.append("local input/output/runtime directory")
    if lower_name in LOCAL_ONLY_FILENAMES:
        issues.append("local-only config; commit the matching .example file instead")
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        issues.append("media/binary asset is blocked from this repo")
    return issues


def staged_blob(path: Path) -> Optional[bytes]:
    relative = rel(path)
    result = run_git(["show", f":{relative}"])
    if result.returncode != 0:
        return None
    return result.stdout


def text_issues(path: Path, data: Optional[bytes] = None) -> List[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    if data is None:
        try:
            data = path.read_bytes()
        except Exception:
            return []
    text = data.decode("utf-8", errors="ignore").lower()
    hits = [term for term in blocked_terms() if term in text]
    if not hits:
        return []
    return ["sensitive keyword(s) found; keep local wording in ignored files and commit only sanitized *.example templates"]


def scan(paths: Iterable[Path], *, staged: bool = False) -> int:
    failed = False
    for path in paths:
        if staged:
            blob = staged_blob(path)
            if blob is None:
                continue
            exists = True
        else:
            blob = None
            exists = path.exists() and path.is_file()
        if not exists:
            continue
        issues = path_issues(path) + text_issues(path, blob)
        if issues:
            failed = True
            print(f"BLOCKED {rel(path)}")
            for issue in issues:
                print(f"  - {issue}")
    if failed:
        print("\nCommit blocked: remove private/sensitive files from the index, or commit only sanitized *.example templates.")
        return 1
    print("Private-content guard passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Block local/private or sensitive material from Git commits.")
    parser.add_argument("--staged", action="store_true", help="Scan staged Git files.")
    parser.add_argument("--tracked", action="store_true", help="Scan tracked Git files.")
    parser.add_argument("--all", action="store_true", help="Scan non-ignored repository candidate files.")
    args = parser.parse_args()
    selected = sum([args.staged, args.tracked, args.all])
    if selected > 1:
        parser.error("choose only one scan mode")
    if args.staged:
        return scan(staged_files(), staged=True)
    if args.tracked:
        return scan(tracked_files())
    return scan(all_candidate_files())


if __name__ == "__main__":
    raise SystemExit(main())
