from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import requests

from .util import append_jsonl, die


def check_ollama(host: str) -> None:
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        die(
            "Could not connect to Ollama at "
            f"{host}. Start Ollama first. On Windows, open Ollama or run: ollama serve\nDetails: {e}"
        )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_model_io_log(log_path: Optional[Path], row: Dict[str, Any]) -> None:
    if log_path is None:
        return
    try:
        append_jsonl(log_path, row)
    except Exception as e:
        print(f"Warning: could not write model I/O log {log_path}: {e}", file=sys.stderr)


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
    log_path: Optional[Path] = None,
    log_context: Optional[Dict[str, Any]] = None,
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
    request_id = uuid.uuid4().hex
    started_at = utc_timestamp()
    started = time.perf_counter()
    base_log: Dict[str, Any] = {
        "timestamp": started_at,
        "request_id": request_id,
        "context": log_context or {},
        "host": host,
        "model": model,
        "options": options,
        "keep_alive": keep_alive,
        "response_format": response_format,
        "image_count": len(images or []),
        "prompt": prompt,
    }
    try:
        r = requests.post(f"{host.rstrip('/')}/api/generate", json=payload, timeout=timeout)
        if r.status_code == 404:
            write_model_io_log(
                log_path,
                {
                    **base_log,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "status": "error",
                    "error": f"Ollama model not found: {model}",
                },
            )
            die(f"Ollama model not found: {model}. Pull it first with: ollama pull {model}")
        r.raise_for_status()
        data = r.json()
        response = str(data.get("response", ""))
        write_model_io_log(
            log_path,
            {
                **base_log,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "status": "ok",
                "response": response,
            },
        )
        return response
    except requests.HTTPError as e:
        detail = e.response.text[:1000] if e.response is not None else str(e)
        write_model_io_log(
            log_path,
            {
                **base_log,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "status": "error",
                "error": detail,
            },
        )
        die(f"Ollama request failed for model {model}: {detail}")
    except Exception as e:
        write_model_io_log(
            log_path,
            {
                **base_log,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "status": "error",
                "error": str(e),
            },
        )
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
