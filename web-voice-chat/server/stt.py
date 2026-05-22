"""Whisper(vLLM, OpenAI 호환) /v1/audio/transcriptions 클라이언트."""
from __future__ import annotations
import httpx
from .config import SETTINGS


class STTClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    async def transcribe(self, wav_bytes: bytes) -> str:
        url = f"{SETTINGS.stt_base}/audio/transcriptions"
        files = {"file": ("utterance.wav", wav_bytes, "audio/wav")}
        data = {
            "model": SETTINGS.stt_model,
            "language": SETTINGS.stt_language,
            "response_format": "json",
        }
        r = await self._client.post(url, files=files, data=data)
        r.raise_for_status()
        body = r.json()
        return (body.get("text") or "").strip()

    async def aclose(self) -> None:
        await self._client.aclose()
