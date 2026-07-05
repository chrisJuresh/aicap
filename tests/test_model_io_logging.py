from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicap import ollama as ollama_module  # noqa: E402


class ModelIoLoggingTests(unittest.TestCase):
    def test_ollama_generate_logs_exact_prompt_and_response(self) -> None:
        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {"response": "raw model response"}

        old_post = ollama_module.requests.post
        try:
            ollama_module.requests.post = lambda *args, **kwargs: FakeResponse()
            with tempfile.TemporaryDirectory() as tmp:
                log_path = Path(tmp) / "model_io.jsonl"
                response = ollama_module.ollama_generate(
                    "http://example.invalid",
                    "dummy-model",
                    "exact prompt text",
                    temperature=0.0,
                    log_path=log_path,
                    log_context={"stage": "test"},
                )
                rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        finally:
            ollama_module.requests.post = old_post

        self.assertEqual(response, "raw model response")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt"], "exact prompt text")
        self.assertEqual(rows[0]["response"], "raw model response")
        self.assertEqual(rows[0]["context"], {"stage": "test"})
        self.assertEqual(rows[0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
