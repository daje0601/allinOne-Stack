"""WS 메시지 스키마. 클라이언트와 서버가 같은 규약으로 주고받음."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


# ── Client → Server ──────────────────────────────────────────────────────────
class SessionStart(BaseModel):
    type: Literal["session.start"] = "session.start"
    # 향후 화자/언어/시스템프롬프트 오버라이드 등 확장 자리


class AudioUtterance(BaseModel):
    """브라우저 VAD가 잘라낸 발화 단위 WAV (base64)."""
    type: Literal["audio.utterance"] = "audio.utterance"
    wav_base64: str
    utterance_idx: int


class ResetHistory(BaseModel):
    type: Literal["history.reset"] = "history.reset"


# ── Server → Client ──────────────────────────────────────────────────────────
class Ready(BaseModel):
    type: Literal["ready"] = "ready"
    voice: str
    model: str


class STTResult(BaseModel):
    type: Literal["stt.result"] = "stt.result"
    utterance_idx: int
    text: str
    latency_ms: int


class LLMResult(BaseModel):
    type: Literal["llm.result"] = "llm.result"
    utterance_idx: int
    text: str
    latency_ms: int


class TTSAudio(BaseModel):
    type: Literal["tts.audio"] = "tts.audio"
    utterance_idx: int
    wav_base64: str
    latency_ms: int


class Error(BaseModel):
    type: Literal["error"] = "error"
    utterance_idx: int | None = None
    stage: str  # "stt" | "llm" | "tts" | "session"
    message: str
