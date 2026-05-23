"""환경변수 한 곳에서 읽고 기본값 부여. server.py와 클라이언트들이 공유."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 레포 루트 (allinOne-Stack/) 기준 경로 — config.py는 web-voice-chat/server/ 안에 있음
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

# 곰돌이 푸 페르소나 시스템 프롬프트는 llm-test에 있는 원본을 단일 진실 소스로 사용
_PERSONA_FILE = _REPO_ROOT / "llm-test" / "data" / "system_prompt.txt"

# 음성 출력용 추가 지시 — TTS로 듣기 좋게 짧고 자연스럽게.
# 페르소나 본문 뒤에 붙여서 톤은 유지하면서 길이만 제어.
_VOICE_SUFFIX = """

### 음성 채팅 추가 지시
- 답변은 **반드시 한두 문장**으로 짧게. 음성으로 듣기 좋도록 자연스러운 구어체로.
- 마크다운, 번호 매기기, 이모지, 줄바꿈 목록은 절대 쓰지 말 것 (음성으로 읽히지 않음).
- 다만 페르소나의 말투(쉼표, 말끝 흐림, 비유)는 유지.
"""


def _load_persona_prompt() -> str:
    """llm-test/data/system_prompt.txt 가 있으면 페르소나 + 음성 톤 지시를 반환.
    없으면 안전한 폴백 프롬프트.
    """
    if _PERSONA_FILE.exists():
        try:
            return _PERSONA_FILE.read_text(encoding="utf-8").rstrip() + _VOICE_SUFFIX
        except Exception:
            pass
    return "너는 친근한 한국어 음성 챗봇이야. 답변은 한 두 문장으로 짧고 자연스럽게 해줘."


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
    llm_system_prompt=_get("LLM_SYSTEM_PROMPT", _load_persona_prompt()),

    tts_base=_get("TTS_BASE", "http://localhost:12000/v1"),
    tts_voice=_get("TTS_VOICE", "sohee"),
    tts_language=_get("TTS_LANGUAGE", "Korean"),

    app_host=_get("APP_HOST", "0.0.0.0"),
    app_port=_get_int("APP_PORT", 14000),
)
