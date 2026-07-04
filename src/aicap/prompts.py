from __future__ import annotations

import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any, Dict

from .constants import DEFAULT_PROMPTS
from .util import die


def merge_prompt_dicts(defaults: Dict[str, Dict[str, str]], overrides: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {section: values.copy() for section, values in defaults.items()}
    for section, values in overrides.items():
        if not isinstance(values, dict):
            continue
        merged.setdefault(section, {})
        for key, value in values.items():
            if isinstance(value, str):
                merged[section][key] = textwrap.dedent(value).strip()
    return merged


def load_prompts(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        print(f"Prompt file not found, using built-in defaults: {path}", file=sys.stderr)
        return merge_prompt_dicts(DEFAULT_PROMPTS, {})
    try:
        with path.open("rb") as f:
            loaded = tomllib.load(f)
        prompts = merge_prompt_dicts(DEFAULT_PROMPTS, loaded)
        print(f"Loaded prompts from {path}")
        return prompts
    except Exception as e:
        die(f"Could not load prompt file: {path}\nDetails: {e}")


def get_prompt(prompts: Dict[str, Dict[str, str]], section: str, key: str) -> str:
    try:
        return prompts[section][key].strip()
    except KeyError:
        die(f"Missing prompt [{section}].{key} in prompts file.")


def format_prompt_template(template: str, **values: str) -> str:
    try:
        return template.format(**values).strip()
    except KeyError as e:
        die(f"Prompt template is missing or has an invalid placeholder: {e}")


def visual_prompt(caption_mode: str, prompts: Dict[str, Dict[str, str]]) -> str:
    base = get_prompt(prompts, "visual", "base")
    mode_key = "explicit" if caption_mode == "explicit" else "neutral"
    mode_prompt = get_prompt(prompts, "visual", mode_key)
    return f"{base}\n\n{mode_prompt}".strip()
