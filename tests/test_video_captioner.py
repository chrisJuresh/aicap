from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicap import refine as refine_module  # noqa: E402
from aicap.cache import load_visual_cache  # noqa: E402
from aicap.constants import DEFAULT_PROMPTS  # noqa: E402
from aicap.models import CaptionItem  # noqa: E402


def make_item(index: int) -> CaptionItem:
    return CaptionItem(
        index=index,
        start=float(index - 1),
        end=float(index),
        frame=f"frame_{index:06d}.jpg",
        visual_caption=f"visual {index}",
        speech="",
        final_caption=f"visual {index}",
    )


class PipelineReliabilityTests(unittest.TestCase):
    def test_story_refine_retries_malformed_json(self) -> None:
        items = [make_item(1), make_item(2)]
        calls = []
        old_check_ollama = refine_module.check_ollama
        old_generate = refine_module.ollama_generate
        try:
            refine_module.check_ollama = lambda host: None

            def fake_generate(*args, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return "not json"
                return '{"captions":[{"i":1,"caption":"caption one"},{"i":2,"caption":"caption two"}],"story_summary":"two captions"}'

            refine_module.ollama_generate = fake_generate
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                refined = refine_module.refine_with_llm(
                    items,
                    host="http://example.invalid",
                    text_model="dummy",
                    caption_mode="neutral",
                    batch_size=2,
                    prompts=DEFAULT_PROMPTS,
                    keep_alive="0",
                    temperature=0.0,
                    seed=42,
                    num_ctx=8192,
                    llm_retries=1,
                    strict_refine=True,
                    cache_path=None,
                    story_context=True,
                )
        finally:
            refine_module.check_ollama = old_check_ollama
            refine_module.ollama_generate = old_generate

        self.assertEqual([item.final_caption for item in refined], ["caption one", "caption two"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["response_format"] == "json" for call in calls))
        self.assertTrue(all(call["seed"] == 42 for call in calls))
        self.assertTrue(all(call["num_ctx"] == 8192 for call in calls))

    def test_strict_refine_stops_after_retries(self) -> None:
        old_check_ollama = refine_module.check_ollama
        old_generate = refine_module.ollama_generate
        try:
            refine_module.check_ollama = lambda host: None
            refine_module.ollama_generate = lambda *args, **kwargs: "not json"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    refine_module.refine_with_llm(
                        [make_item(1)],
                        host="http://example.invalid",
                        text_model="dummy",
                        caption_mode="neutral",
                        batch_size=1,
                        prompts=DEFAULT_PROMPTS,
                        keep_alive="0",
                        temperature=0.0,
                        seed=42,
                        num_ctx=8192,
                        llm_retries=1,
                        strict_refine=True,
                        cache_path=None,
                        story_context=True,
                    )
        finally:
            refine_module.check_ollama = old_check_ollama
            refine_module.ollama_generate = old_generate

    def test_visual_cache_skips_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "visual_frame_captions.jsonl"
            cache_path.write_text(
                '{"frame":"frame_000001.jpg","caption":"first"}\n'
                'not json\n'
                '{"frame":"frame_000002.jpg","caption":"second"}\n',
                encoding="utf-8",
            )
            with redirect_stderr(StringIO()):
                self.assertEqual(
                    load_visual_cache(cache_path),
                    {"frame_000001.jpg": "first", "frame_000002.jpg": "second"},
                )


if __name__ == "__main__":
    unittest.main()
