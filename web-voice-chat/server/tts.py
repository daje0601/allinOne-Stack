"""Qwen3-TTS(vllm-omni) /v1/audio/speech 클라이언트.

- synthesize(): 한 번에 wav 받기 (legacy, 더는 안 씀)
- synthesize_stream(): PCM raw int16 LE @ 16kHz mono 청크 스트림 (현재 사용)
  Qwen3-TTS는 stream=true 시 response_format='pcm' 강제 + Transfer-Encoding: chunked.
"""
from __future__ import annotations
from typing import AsyncIterator
import httpx
from .config import SETTINGS

# Qwen3-TTS PCM output sample rate.
# Authoritative source: vllm-omni serving_speech.py:345,549 → `sample_rate_val = 24000`.
TTS_PCM_SAMPLE_RATE = 24000


class TTSClient:
    def __init__(self) -> None:
        # No total timeout for streams (long utterances), but generous read timeout per chunk
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0, read=60.0))

    async def synthesize(self, text: str) -> bytes:
        """Non-streaming WAV fetch (kept for reference / debug)."""
        url = f"{SETTINGS.tts_base}/audio/speech"
        body = {
            "input": text,
            "voice": SETTINGS.tts_voice,
            "language": SETTINGS.tts_language,
            "response_format": "wav",
        }
        r = await self._client.post(url, json=body)
        r.raise_for_status()
        return r.content

    async def synthesize_stream(self, text: str, chunk_bytes: int = 8192) -> AsyncIterator[bytes]:
        """Stream raw int16 LE mono PCM @ TTS_PCM_SAMPLE_RATE in ~chunk_bytes chunks."""
        url = f"{SETTINGS.tts_base}/audio/speech"
        body = {
            "input": text,
            "voice": SETTINGS.tts_voice,
            "language": SETTINGS.tts_language,
            "response_format": "pcm",
            "stream": True,
        }
        async with self._client.stream("POST", url, json=body) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes(chunk_size=chunk_bytes):
                if chunk:
                    yield chunk

    async def aclose(self) -> None:
        await self._client.aclose()
