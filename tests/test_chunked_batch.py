from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicap.cli import chunk_jobs  # noqa: E402


class ChunkedBatchTests(unittest.TestCase):
    def test_chunk_jobs_groups_by_requested_size(self) -> None:
        jobs = list(range(23))

        chunks = chunk_jobs(jobs, 10)

        self.assertEqual([len(chunk) for chunk in chunks], [10, 10, 3])
        self.assertEqual(chunks[0], list(range(10)))
        self.assertEqual(chunks[-1], [20, 21, 22])

    def test_chunk_jobs_zero_keeps_single_batch(self) -> None:
        jobs = list(range(3))

        self.assertEqual(chunk_jobs(jobs, 0), [jobs])


if __name__ == "__main__":
    unittest.main()
