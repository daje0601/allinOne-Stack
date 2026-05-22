"""Llama 챗 서버(vLLM, OpenAI 호환) /v1/chat/completions 클라이언트."""
from __future__ import annotations
import httpx
from .config import SETTINGS


class LLMClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def chat(self, messages: list[dict]) -> str:
        url = f"{SETTINGS.llm_base}/chat/completions"
        body = {
            "model": SETTINGS.llm_model,
            "messages": messages,
            "max_tokens": SETTINGS.llm_max_tokens,
            "temperature": SETTINGS.llm_temperature,
        }
        r = await self._client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    async def aclose(self) -> None:
        await self._client.aclose()
