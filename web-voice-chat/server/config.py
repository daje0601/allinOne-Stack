"""환경변수 한 곳에서 읽고 기본값 부여. server.py와 클라이언트들이 공유."""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    stt_base: str
    stt_model: str
    stt_language: str

    llm_base: str
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    llm_system_prompt: str

    tts_base: str
    tts_voice: str
    tts_language: str

    app_host: str
    app_port: int


SETTINGS = Settings(
    stt_base=_get("STT_BASE", "http://localhost:11000/v1"),
    stt_model=_get("STT_MODEL", "openai/whisper-large-v3-turbo"),
    stt_language=_get("STT_LANGUAGE", "ko"),

    llm_base=_get("LLM_BASE", "http://localhost:13000/v1"),
    llm_model=_get("LLM_MODEL", "iamjoon/llama3-8b-persona-chatbot"),
    llm_max_tokens=_get_int("LLM_MAX_TOKENS", 256),
    llm_temperature=_get_float("LLM_TEMPERATURE", 0.7),
    llm_system_prompt=_get(
        "LLM_SYSTEM_PROMPT",
        "너는 친근한 한국어 음성 챗봇이야. 답변은 한 두 문장으로 짧고 자연스럽게 해줘.",
    ),

    tts_base=_get("TTS_BASE", "http://localhost:12000/v1"),
    tts_voice=_get("TTS_VOICE", "sohee"),
    tts_language=_get("TTS_LANGUAGE", "Korean"),

    app_host=_get("APP_HOST", "0.0.0.0"),
    app_port=_get_int("APP_PORT", 14000),
)
