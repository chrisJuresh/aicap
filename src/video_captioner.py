from __future__ import annotations

from aicap.cache import load_visual_cache
from aicap.cli import main
from aicap.constants import DEFAULT_PROMPTS
from aicap.models import CaptionItem, TranscriptSegment, VideoJob
from aicap.ollama import check_ollama, ollama_generate
from aicap.refine import refine_with_llm


if __name__ == "__main__":
    main()
