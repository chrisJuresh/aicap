from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicap.util import find_exe  # noqa: E402


class ExecutableResolutionTests(unittest.TestCase):
    def test_ffmpeg_dir_env_works_when_path_is_stale(self) -> None:
        old_path = os.environ.get("PATH")
        old_ffmpeg_dir = os.environ.get("AICAP_FFMPEG_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fake = Path(tmp) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
                fake.write_text("", encoding="utf-8")
                os.environ["PATH"] = ""
                os.environ["AICAP_FFMPEG_DIR"] = tmp

                self.assertEqual(str(fake), find_exe("ffmpeg"))
        finally:
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path
            if old_ffmpeg_dir is None:
                os.environ.pop("AICAP_FFMPEG_DIR", None)
            else:
                os.environ["AICAP_FFMPEG_DIR"] = old_ffmpeg_dir


if __name__ == "__main__":
    unittest.main()
