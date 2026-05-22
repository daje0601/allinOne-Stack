"""Gradio + FastRTC 실시간 스트리밍 음성 챗.

브라우저 마이크 → WebRTC 로 서버에 실시간 오디오 스트림.
서버측 VAD (`ReplyOnPause`, Silero VAD) 가 사용자 발화 끝남을 자동 감지하면
한 발화 통째로 핸들러 호출 → STT → LLM → TTS → 결과를 청크로 yield 해서
브라우저에서 실시간 재생. 사용자는 녹음 버튼 누를 필요 없이 자연스럽게 말하면 됨.

`share=True` 로 띄우므로 *.gradio.live 공개 URL 발급 → 포트 노출 불필요.
"""
from __future__ import annotations

import io
import logging
import time
import wave

import gradio as gr
import httpx
import numpy as np
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    WebRTC,
    get_cloudflare_turn_credentials,
    get_cloudflare_turn_credentials_async,
)

from server.config import SETTINGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("gradio-voice-chat")

_STT = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
_LLM = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
_TTS = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))


def _stt(wav_bytes: bytes) -> str:
    files = {"file": ("utterance.wav", wav_bytes, "audio/wav")}
    data = {
        "model": SETTINGS.stt_model,
        "language": SETTINGS.stt_language,
        "response_format": "json",
    }
    r = _STT.post(f"{SETTINGS.stt_base}/audio/transcriptions", files=files, data=data)
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def _llm(messages: list[dict]) -> str:
    body = {
        "model": SETTINGS.llm_model,
        "messages": messages,
        "max_tokens": SETTINGS.llm_max_tokens,
        "temperature": SETTINGS.llm_temperature,
    }
    r = _LLM.post(f"{SETTINGS.llm_base}/chat/completions", json=body)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _tts(text: str) -> bytes:
    body = {
        "input": text,
        "voice": SETTINGS.tts_voice,
        "language": SETTINGS.tts_language,
        "response_format": "wav",
    }
    r = _TTS.post(f"{SETTINGS.tts_base}/audio/speech", json=body)
    r.raise_for_status()
    return r.content


def _to_wav_bytes(sr: int, arr: np.ndarray) -> bytes:
    arr = np.asarray(arr).flatten()
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(arr.tobytes())
    return buf.getvalue()


def _parse_wav(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        arr = np.frombuffer(frames, dtype=np.int16)
    elif sw == 4:
        # 32-bit PCM → int16 로 축소
        arr32 = np.frombuffer(frames, dtype=np.int32)
        arr = (arr32 >> 16).astype(np.int16)
    else:
        # 8-bit unsigned → signed 16
        arr = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) << 8
    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    return sr, arr


def respond(audio: tuple[int, np.ndarray], chatbot: list[dict] | None):
    """ReplyOnPause 핸들러. audio = (sr, ndarray) 발화 한 덩어리."""
    chatbot = list(chatbot or [])
    sr, arr = audio
    wav_bytes = _to_wav_bytes(sr, arr)

    t0 = time.perf_counter()
    try:
        user_text = _stt(wav_bytes)
    except Exception as e:
        log.exception("STT failed")
        chatbot.append({"role": "assistant", "content": f"(STT 오류: {e})"})
        yield AdditionalOutputs(chatbot)
        return
    stt_ms = int((time.perf_counter() - t0) * 1000)
    if not user_text:
        log.info("empty STT (%dms) — skipping turn", stt_ms)
        return

    chatbot.append({"role": "user", "content": user_text})
    yield AdditionalOutputs(chatbot)

    messages = [{"role": "system", "content": SETTINGS.llm_system_prompt}, *chatbot]
    t0 = time.perf_counter()
    try:
        answer = _llm(messages) or "음, 다시 말씀해 주실 수 있을까요?"
    except Exception as e:
        log.exception("LLM failed")
        chatbot.append({"role": "assistant", "content": f"(LLM 오류: {e})"})
        yield AdditionalOutputs(chatbot)
        return
    llm_ms = int((time.perf_counter() - t0) * 1000)
    chatbot.append({"role": "assistant", "content": answer})
    yield AdditionalOutputs(chatbot)

    t0 = time.perf_counter()
    try:
        wav_out = _tts(answer)
    except Exception as e:
        log.exception("TTS failed")
        return
    tts_ms = int((time.perf_counter() - t0) * 1000)
    out_sr, out_arr = _parse_wav(wav_out)
    log.info(
        "turn · stt=%dms llm=%dms tts=%dms · user=%r · bot=%r",
        stt_ms, llm_ms, tts_ms, user_text, answer,
    )

    # ~200ms 청크로 잘라서 스트리밍 재생
    chunk_samples = max(1, out_sr // 5)
    for i in range(0, len(out_arr), chunk_samples):
        chunk = out_arr[i : i + chunk_samples]
        yield (out_sr, chunk.reshape(1, -1))


with gr.Blocks(title="Voice Chat (Streaming)") as demo:
    gr.Markdown(
        f"# 음성 챗봇 (실시간 스트리밍)\n"
        f"**[대화 시작]** 버튼을 한 번만 누르면 마이크가 계속 켜진 상태가 됩니다 "
        f"(녹음/정지 반복 아님). 자연스럽게 말하면 무음이 감지될 때마다 자동으로 응답이 흘러나와요. "
        f"끝낼 때만 **[대화 종료]** 를 누르세요.\n\n"
        f"- STT: `{SETTINGS.stt_model}` · LLM: `{SETTINGS.llm_model}` · "
        f"TTS: `{SETTINGS.tts_voice}` ({SETTINGS.tts_language})"
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio = WebRTC(
                modality="audio",
                mode="send-receive",
                label="대화 (마이크 스트리밍)",
                # "녹음" 으로 오해되지 않도록 라벨 재정의. 한 번 누르면 마이크 세션이
                # 계속 켜져있고 VAD 가 발화 끝남을 자동 감지함 (녹음/정지 반복 아님).
                button_labels={"start": "대화 시작", "stop": "대화 종료", "waiting": "연결 중..."},
                # 컨테이너 NAT 환경에서는 STUN만으로는 미디어 경로가 안 잡혀 connecting 에서
                # 실패함. HF 가 브로커하는 Cloudflare TURN 으로 양방향 모두 자격증명 부여.
                rtc_configuration=get_cloudflare_turn_credentials_async,
                server_rtc_configuration=get_cloudflare_turn_credentials(ttl=360_000),
            )
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="대화 기록", height=460, type="messages")

    audio.stream(
        ReplyOnPause(respond),
        inputs=[audio, chatbot],
        outputs=[audio],
    )
    audio.on_additional_outputs(
        lambda updated: updated,
        outputs=[chatbot],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=SETTINGS.app_host,
        server_port=SETTINGS.app_port,
        share=True,
    )
