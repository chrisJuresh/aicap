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
from aicap import subtitles as subtitles_module  # noqa: E402
from aicap.captions import build_items, caption_rows_from_text, captions_by_chunk_from_rows  # noqa: E402
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

    def test_story_refine_accepts_ordered_captions_when_model_renumbers_ids(self) -> None:
        items = [make_item(9), make_item(10)]
        old_check_ollama = refine_module.check_ollama
        old_generate = refine_module.ollama_generate
        try:
            refine_module.check_ollama = lambda host: None
            refine_module.ollama_generate = (
                lambda *args, **kwargs: '{"captions":[{"i":1,"caption":"caption nine"},{"i":2,"caption":"caption ten"}],"story_summary":"two captions"}'
            )
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
                    llm_retries=0,
                    strict_refine=True,
                    cache_path=None,
                    story_context=True,
                )
        finally:
            refine_module.check_ollama = old_check_ollama
            refine_module.ollama_generate = old_generate

        self.assertEqual([item.final_caption for item in refined], ["caption nine", "caption ten"])

    def test_story_refine_repairs_after_malformed_json_retries(self) -> None:
        items = [make_item(1), make_item(2)]
        calls = []
        old_check_ollama = refine_module.check_ollama
        old_generate = refine_module.ollama_generate
        try:
            refine_module.check_ollama = lambda host: None

            def fake_generate(*args, **kwargs):
                calls.append(kwargs)
                if len(calls) <= 2:
                    return "not json"
                return '{"captions":[{"i":1,"caption":"repaired one"},{"i":2,"caption":"repaired two"}],"story_summary":"repaired"}'

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

        self.assertEqual([item.final_caption for item in refined], ["repaired one", "repaired two"])
        self.assertEqual(len(calls), 3)

    def test_caption_rows_fall_back_to_chunk_order(self) -> None:
        chunk = [make_item(9), make_item(10)]

        by_index = captions_by_chunk_from_rows(
            [{"i": 1, "caption": "caption nine"}, {"i": 2, "caption": "caption ten"}],
            chunk,
        )

        self.assertEqual(by_index, {9: "caption nine", 10: "caption ten"})

    def test_caption_rows_from_text_requires_caption_like_lines(self) -> None:
        self.assertEqual(caption_rows_from_text("not json"), [])
        self.assertEqual(caption_rows_from_text("1. first caption\n2. second caption"), ["first caption", "second caption"])

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

    def test_build_items_groups_frames_into_readable_caption_windows(self) -> None:
        frames = [Path(f"frame_{index:06d}.jpg") for index in range(1, 7)]
        visual = [
            "A person enters the room.",
            "A person enters the room.",
            "The person turns toward the table.",
            "Another person raises a hand.",
            "Another person raises a hand.",
            "The first person answers.",
        ]

        items = build_items(frames, visual, [], sample_every=2.0, duration=12.0, max_frames=None, caption_window_seconds=6.0)

        self.assertEqual(len(items), 2)
        self.assertEqual((items[0].start, items[0].end), (0.0, 6.0))
        self.assertEqual((items[1].start, items[1].end), (6.0, 12.0))
        self.assertIn("+0s: A person enters the room.", items[0].visual_caption)
        self.assertEqual(items[0].visual_caption.count("A person enters the room."), 1)

    def test_build_items_merges_short_final_caption_sliver(self) -> None:
        frames = [Path(f"frame_{index:06d}.jpg") for index in range(1, 8)]
        visual = [f"visual {index}" for index in range(1, 8)]

        items = build_items(frames, visual, [], sample_every=2.0, duration=13.0, max_frames=None, caption_window_seconds=6.0)

        self.assertEqual(len(items), 2)
        self.assertEqual((items[-1].start, items[-1].end), (6.0, 13.0))
        self.assertIn("visual 7", items[-1].visual_caption)

    def test_adaptive_subtitle_layout_clamps_vertical_video_size(self) -> None:
        old_probe = subtitles_module.ffprobe_video_size
        try:
            subtitles_module.ffprobe_video_size = lambda _path: (1080, 1920)
            font_size, margin_v, band_height, reference_height = subtitles_module.resolve_subtitle_layout(
                Path("vertical.mp4"),
                placement="below",
                fixed_font_size=None,
                font_size_percent=3.2,
                fixed_margin_v=None,
                margin_v_percent=1.8,
                fixed_band_height=None,
                band_height_percent=16.0,
            )
        finally:
            subtitles_module.ffprobe_video_size = old_probe

        self.assertLessEqual(font_size, 34)
        self.assertLessEqual(margin_v, font_size)
        self.assertLessEqual(band_height, 132)
        self.assertEqual(reference_height, 1920 + band_height)


if __name__ == "__main__":
    unittest.main()
