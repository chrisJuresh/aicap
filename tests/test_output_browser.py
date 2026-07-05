from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicap.models import CaptionItem, VideoJob  # noqa: E402
from aicap.output_browser import relative_url, write_output_browser  # noqa: E402


class OutputBrowserTests(unittest.TestCase):
    def test_write_output_browser_embeds_jobs_and_captions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "output_batch"
            job_dir = out_dir / "A Video"
            job_dir.mkdir(parents=True)
            (job_dir / "captioned_video.mp4").write_bytes(b"")
            (job_dir / "captions.srt").write_text("1\n", encoding="utf-8")
            (job_dir / "model_io.jsonl").write_text('{"prompt":"p","response":"r"}\n', encoding="utf-8")
            job = VideoJob(
                video_path=root / "source video.mp4",
                out_dir=job_dir,
                work_dir=job_dir / "_work",
                duration=10.0,
                items=[
                    CaptionItem(
                        index=1,
                        start=0.0,
                        end=10.0,
                        frame="frame_000001.jpg",
                        visual_caption="visual caption",
                        speech="spoken words",
                        final_caption="final caption",
                    )
                ],
            )
            args = SimpleNamespace(burn_in=True, burn_output=None)

            path = write_output_browser(out_dir, [job], args, is_batch=True, total_jobs=3, auto_refresh=True)
            html = path.read_text(encoding="utf-8")

            self.assertTrue(path.exists())
            self.assertIn("AICap Outputs", html)
            self.assertIn('"total_job_count": 3', html)
            self.assertIn('"auto_refresh": true', html)
            self.assertIn("final caption", html)
            self.assertIn("A%20Video/captioned_video.mp4", html)
            self.assertIn("A%20Video/model_io.jsonl", html)

    def test_relative_url_quotes_spaces(self) -> None:
        base = Path("C:/tmp/out")
        target = Path("C:/tmp/out/Video One/captions.json")

        self.assertEqual("Video%20One/captions.json", relative_url(base, target))


if __name__ == "__main__":
    unittest.main()
