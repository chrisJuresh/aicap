from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import requests

from .util import die


def check_ollama(host: str) -> None:
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        die(
            "Could not connect to Ollama at "
            f"{host}. Start Ollama first. On Windows, open Ollama or run: ollama serve\nDetails: {e}"
        )


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    images: Optional[List[str]] = None,
    temperature: float = 0.2,
    num_predict: int = 160,
    timeout: int = 600,
    keep_alive: str = "30m",
    seed: Optional[int] = None,
    num_ctx: Optional[int] = None,
    response_format: Optional[str] = None,
) -> str:
    options: Dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
    if seed is not None and seed >= 0:
        options["seed"] = seed
    if num_ctx is not None and num_ctx > 0:
        options["num_ctx"] = num_ctx
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
        "keep_alive": keep_alive,
    }
    if response_format:
        payload["format"] = response_format
    if images:
        payload["images"] = images
    try:
        r = requests.post(f"{host.rstrip('/')}/api/generate", json=payload, timeout=timeout)
        if r.status_code == 404:
            die(f"Ollama model not found: {model}. Pull it first with: ollama pull {model}")
        r.raise_for_status()
        data = r.json()
        return str(data.get("response", ""))
    except requests.HTTPError as e:
        detail = e.response.text[:1000] if e.response is not None else str(e)
        die(f"Ollama request failed for model {model}: {detail}")
    except Exception as e:
        die(f"Ollama request failed for model {model}: {e}")


def unload_ollama_model(host: str, model: str) -> None:
    if not model:
        return
    try:
        requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=30,
        )
    except Exception as e:
        print(f"Warning: could not explicitly unload Ollama model {model}: {e}", file=sys.stderr)
