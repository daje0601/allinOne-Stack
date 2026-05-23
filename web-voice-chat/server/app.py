"""FastAPI 앱: 정적 파일 서빙 + WS /ws.

WS 흐름:
  client → audio.utterance(wav_base64)
  server → stt.result   (Whisper 결과)
  server → llm.result   (Llama 답변)
  server → tts.audio    (TTS wav)
"""
from __future__ import annotations
import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import SETTINGS
from .llm import LLMClient
from .protocol import (
    AudioUtterance,
    Error,
    LLMResult,
    Ready,
    ResetHistory,
    STTResult,
    SessionStart,
    TTSChunk,
    TTSEnd,
    TTSStart,
)
from .session import Session
from .stt import STTClient
from .tts import TTS_PCM_SAMPLE_RATE, TTSClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("web-voice-chat")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.stt = STTClient()
    app.state.llm = LLMClient()
    app.state.tts = TTSClient()
    log.info(
        "ready · STT=%s · LLM=%s · TTS=%s/%s",
        SETTINGS.stt_model, SETTINGS.llm_model, SETTINGS.tts_voice, SETTINGS.tts_language,
    )
    try:
        yield
    finally:
        await asyncio.gather(
            app.state.stt.aclose(),
            app.state.llm.aclose(),
            app.state.tts.aclose(),
            return_exceptions=True,
        )


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
async def health():
    return {"ok": True, "voice": SETTINGS.tts_voice, "model": SETTINGS.llm_model}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    await _send(ws, Ready(voice=SETTINGS.tts_voice, model=SETTINGS.llm_model))
    log.info("client connected")

    try:
        while True:
            raw = await ws.receive_json()
            msg_type = raw.get("type")

            if msg_type == "session.start":
                SessionStart.model_validate(raw)
                session.reset()

            elif msg_type == "history.reset":
                ResetHistory.model_validate(raw)
                session.reset()

            elif msg_type == "audio.utterance":
                msg = AudioUtterance.model_validate(raw)
                log.info("audio.utterance idx=%s bytes=%s", msg.utterance_idx, len(msg.wav_base64))
                await _handle_utterance(ws, session, msg, app.state)

            else:
                log.warning("unknown msg type: %s", msg_type)

    except WebSocketDisconnect:
        log.info("client disconnected")


async def _handle_utterance(
    ws: WebSocket,
    session: Session,
    msg: AudioUtterance,
    state,
) -> None:
    idx = msg.utterance_idx
    try:
        wav = base64.b64decode(msg.wav_base64)
    except Exception as e:
        await _send(ws, Error(utterance_idx=idx, stage="session", message=f"base64 decode: {e}"))
        return

    # 1) STT
    try:
        t0 = time.perf_counter()
        text = await state.stt.transcribe(wav)
        stt_ms = int((time.perf_counter() - t0) * 1000)
        if not text:
            await _send(ws, STTResult(utterance_idx=idx, text="", latency_ms=stt_ms))
            return  # 빈 발화는 LLM/TTS 스킵
        await _send(ws, STTResult(utterance_idx=idx, text=text, latency_ms=stt_ms))
        session.add_user(text)
    except Exception as e:
        log.exception("STT failed")
        await _send(ws, Error(utterance_idx=idx, stage="stt", message=str(e)))
        return

    # 2) LLM
    try:
        t0 = time.perf_counter()
        answer = await state.llm.chat(session.messages_for_llm())
        llm_ms = int((time.perf_counter() - t0) * 1000)
        if not answer:
            answer = "음, 다시 말씀해 주실 수 있을까요?"
        session.add_assistant(answer)
        await _send(ws, LLMResult(utterance_idx=idx, text=answer, latency_ms=llm_ms))
    except Exception as e:
        log.exception("LLM failed")
        await _send(ws, Error(utterance_idx=idx, stage="llm", message=str(e)))
        return

    # 3) TTS — streaming PCM chunks (TTFB ~100ms vs ~8s for full-synthesis-then-send)
    try:
        t0 = time.perf_counter()
        seq = 0
        start_sent = False
        async for chunk in state.tts.synthesize_stream(answer):
            if not start_sent:
                ttfb_ms = int((time.perf_counter() - t0) * 1000)
                await _send(ws, TTSStart(
                    utterance_idx=idx,
                    sample_rate=TTS_PCM_SAMPLE_RATE,
                    ttfb_ms=ttfb_ms,
                ))
                start_sent = True
            await _send(ws, TTSChunk(
                utterance_idx=idx,
                seq=seq,
                pcm_base64=base64.b64encode(chunk).decode("ascii"),
            ))
            seq += 1
        total_ms = int((time.perf_counter() - t0) * 1000)
        await _send(ws, TTSEnd(
            utterance_idx=idx,
            total_chunks=seq,
            total_latency_ms=total_ms,
        ))
        log.info("tts idx=%s chunks=%s total=%sms", idx, seq, total_ms)
    except Exception as e:
        log.exception("TTS failed")
        await _send(ws, Error(utterance_idx=idx, stage="tts", message=str(e)))


async def _send(ws: WebSocket, msg) -> None:
    await ws.send_json(msg.model_dump(by_alias=False))
