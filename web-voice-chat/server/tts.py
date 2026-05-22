"""Qwen3-TTS(vllm-omni) /v1/audio/speech 클라이언트. 결과는 wav 바이너리."""
from __future__ import annotations
import httpx
from .config import SETTINGS


class TTSClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def synthesize(self, text: str) -> bytes:
        url = f"{SETTINGS.tts_base}/audio/speech"
        body = {
            "input": text,
            "voice": SETTINGS.tts_voice,
            "language": SETTINGS.tts_language,
            "response_format": "wav",
        }
        r = await self._client.post(url, json=body)
        r.raise_for_status()
        return r.content  # raw wav bytes

    async def aclose(self) -> None:
        await self._client.aclose()
