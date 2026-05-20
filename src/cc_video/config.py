"""Runtime config — env vars, defaults, feature flags."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # API keys
    gemini_key: str | None = None
    groq_key: str | None = None
    openai_key: str | None = None
    hf_token: str | None = None  # for pyannote diarization

    # Feature toggles
    enable_vlm: bool = True
    enable_ocr: bool = True
    enable_diarization: bool = True
    deep_keyframes: bool = True
    prefer_whisper_over_captions: bool = True  # native YouTube captions often poor for non-English

    # Models
    whisper_model: str = "large-v3"  # tiny|base|small|medium|large-v3
    vlm_model: str = "gemini-2.5-flash"

    # Limits
    max_keyframes_per_shot: int = 3
    keyframe_resolution: int = 1080
    min_shot_seconds: float = 0.6
    adaptive_threshold: float = 3.0

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            gemini_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            groq_key=os.environ.get("GROQ_API_KEY"),
            openai_key=os.environ.get("OPENAI_API_KEY"),
            hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        )

    def has_any_asr(self) -> bool:
        return bool(self.groq_key or self.openai_key) or _have("faster_whisper")

    def has_vlm(self) -> bool:
        return bool(self.gemini_key) and _have("google.genai")

    def has_ocr(self) -> bool:
        return _have("paddleocr")


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False
