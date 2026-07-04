from __future__ import annotations

import base64
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import guard_private_content as guard  # noqa: E402


class PrivateContentGuardTests(unittest.TestCase):
    def test_safe_example_template_passes(self) -> None:
        path = guard.ROOT / "prompts.toml.example"

        self.assertEqual([], guard.path_issues(path))
        self.assertEqual([], guard.text_issues(path, b'caption_mode = "neutral"\n'))

    def test_local_config_is_blocked(self) -> None:
        issues = guard.path_issues(guard.ROOT / "prompts.toml")

        self.assertTrue(any("local-only config" in issue for issue in issues))

    def test_sensitive_text_is_blocked_in_examples_too(self) -> None:
        path = guard.ROOT / "prompts.toml.example"
        blocked_word = base64.b64decode("bnNmdw==")

        issues = guard.text_issues(path, b"describe " + blocked_word)

        self.assertTrue(issues)

    def test_staged_scan_uses_staged_content(self) -> None:
        path = guard.ROOT / "README.md"
        blocked_word = base64.b64decode("bnNmdw==")
        old_staged_blob = guard.staged_blob
        try:
            guard.staged_blob = lambda staged_path: b"describe " + blocked_word
            with redirect_stdout(StringIO()):
                status = guard.scan([path], staged=True)
        finally:
            guard.staged_blob = old_staged_blob

        self.assertEqual(1, status)


if __name__ == "__main__":
    unittest.main()
