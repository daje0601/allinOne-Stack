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
    """[deprecated] Whole-WAV one-shot. Kept for backward compat / debug."""
    type: Literal["tts.audio"] = "tts.audio"
    utterance_idx: int
    wav_base64: str
    latency_ms: int


class TTSStart(BaseModel):
    """첫 PCM 청크 직전에 전송. 클라이언트가 AudioContext를 sample_rate에 맞춰 초기화."""
    type: Literal["tts.start"] = "tts.start"
    utterance_idx: int
    sample_rate: int
    channels: int = 1
    sample_format: Literal["int16le"] = "int16le"
    ttfb_ms: int  # request → first PCM byte 사이 시간 (서버 측정)


class TTSChunk(BaseModel):
    """Raw PCM 청크 (base64). seq는 0부터 증가."""
    type: Literal["tts.chunk"] = "tts.chunk"
    utterance_idx: int
    seq: int
    pcm_base64: str


class TTSEnd(BaseModel):
    """utterance 종료 마커. 클라이언트가 재생 큐 끝까지 흘리고 마이크 재개."""
    type: Literal["tts.end"] = "tts.end"
    utterance_idx: int
    total_chunks: int
    total_latency_ms: int


class Error(BaseModel):
    type: Literal["error"] = "error"
    utterance_idx: int | None = None
    stage: str  # "stt" | "llm" | "tts" | "session"
    message: str
