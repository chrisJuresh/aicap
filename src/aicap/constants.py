from __future__ import annotations

from pathlib import Path
from typing import Dict


DEFAULT_VISION_MODEL = "huihui_ai/qwen3.5-abliterated:9b-q4_K"
DEFAULT_TEXT_MODEL = "richardyoung/qwythos-9b-abliterated:Q8_0"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "prompts.toml"
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings.toml"
DEFAULT_VIDEO_EXTENSIONS = ".mp4,.mkv,.mov,.avi,.webm,.m4v,.wmv,.mpg,.mpeg"

DEFAULT_PROMPTS: Dict[str, Dict[str, str]] = {
    "visual": {
        "base": (
            "You create timestamped accessibility captions for a local video. "
            "Describe only what is visible in this single frame. "
            "Return one concise factual sentence, 8 to 25 words. "
            "Do not mention that this is a frame or image."
        ),
        "explicit": (
            "Use direct, concrete wording for visible actions. "
            "Stay factual, do not embellish, and do not invent details."
        ),
        "neutral": (
            "Keep the caption neutral and factual. "
            "If sensitive content is visible, describe it only at a high level."
        ),
    },
    "refine": {
        "explicit_mode_instruction": (
            "Use direct, concrete wording while staying factual. Do not invent details."
        ),
        "neutral_mode_instruction": "Keep captions neutral and factual.",
        "template": (
            "You combine visual frame descriptions and speech transcripts into subtitle captions.\n"
            "Return ONLY a valid JSON array. Do not include markdown.\n"
            "Each output item must have this exact shape: {{\"i\": number, \"caption\": \"short caption\"}}.\n"
            "Use the exact i values from the input JSON. Do not renumber, replace, or omit them.\n"
            "Write concise captions suitable for SRT subtitles. Preserve spoken words when useful, but do not hallucinate.\n"
            "{mode_instruction}\n\n"
            "Input JSON:\n"
            "{input_json}"
        ),
    },
    "story": {
        "global_context": "",
        "template": (
            "You combine visual frame descriptions and speech transcripts into subtitle captions that read like a coherent story.\n"
            "Use the story memory and recent captions only for continuity: names, roles, setting, actions already established, and pronoun clarity.\n"
            "Treat global names, roles, relationships, and audience directions as persistent facts for the whole video.\n"
            "Each input row is a multi-second story beat; write one readable caption for that whole beat.\n"
            "Make each caption advance from the previous caption instead of restating the same idea.\n"
            "Do not invent events that are not supported by the current input.\n"
            "Return ONLY a valid JSON object. Do not include markdown.\n"
            "The object must have this exact shape: {{\"captions\": [{{\"i\": number, \"caption\": \"short caption\"}}], \"story_summary\": \"brief updated memory for the next batch\"}}.\n"
            "Use the exact i values from the current input JSON. Do not renumber, replace, or omit them.\n"
            "Captions should be concise SRT subtitles, but they should follow naturally from the previous video context.\n"
            "Keep story_summary factual, compact, and under {story_summary_max_words} words.\n"
            "{mode_instruction}\n\n"
            "Global story direction:\n{global_context}\n\n"
            "Story memory so far:\n{story_summary}\n\n"
            "Recent captions:\n{previous_captions}\n\n"
            "Current input JSON:\n{input_json}"
        ),
        "empty_story_summary": "No previous context yet.",
        "empty_previous_captions": "No previous captions yet.",
    },
}
