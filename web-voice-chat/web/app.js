import { createVAD, VADController } from "/static/vad.js";
import { encodeWavPcm16, arrayBufferToBase64, base64ToBlob, base64ToBytes, PcmStreamPlayer } from "/static/audio-utils.js";

const statusEl = document.getElementById("status");
const messagesEl = document.getElementById("messages");
const micBtn = document.getElementById("mic-btn");
const resetBtn = document.getElementById("reset-btn");
const player = document.getElementById("player");
const micSelect = document.getElementById("mic-select");

let ws = null;
let vad = null;
let vadCtrl = null;
let listening = false;
let utteranceIdx = 0;
let utteranceCapTimer = null;
const pendingBubbles = new Map();   // idx → { userBubble, assistantBubble }
const pcmPlayers = new Map();       // idx → PcmStreamPlayer (TTS streaming)

function toast(msg) {
  console.warn(msg);
  setStatus(msg, "error");
}

// ── Status helpers ───────────────────────────────────────────────────────────
function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status " + (cls || "");
}

// ── WebSocket ────────────────────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    setStatus("connected", "connected");
    ws.send(JSON.stringify({ type: "session.start" }));
    micBtn.disabled = false;
    // 페이지 라이프타임당 1회만 mic 디바이스 enumerate.
    // WS 재연결 시 또 부르면 getUserMedia probe가 반복되어 macOS Chrome의
    // mic 디바이스 라우팅이 꼬일 수 있음 → 무음 디바이스 잡힘 재발.
    if (!window.__micSelectPopulated) {
      window.__micSelectPopulated = true;
      populateMicSelect().catch(e => {
        window.__micSelectPopulated = false;  // 실패 시 재시도 허용
        console.warn("populateMicSelect:", e);
      });
    }
  };
  ws.onclose = () => {
    setStatus("disconnected");
    micBtn.disabled = true;
    setTimeout(connect, 1500); // 자동 재접속
  };
  ws.onerror = () => setStatus("ws error", "error");
  ws.onmessage = (ev) => handleServerMessage(JSON.parse(ev.data));
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "ready":
      console.log("server ready", msg);
      break;

    case "stt.result": {
      const bubble = ensureUserBubble(msg.utterance_idx);
      if (msg.text) {
        bubble.classList.remove("pending");
        bubble.querySelector(".text").textContent = msg.text;
        bubble.querySelector(".meta").textContent = `STT ${msg.latency_ms} ms`;
      } else {
        // 빈 STT (무음 발화) → 버블 자체를 제거하고 다시 듣기 시작
        bubble.remove();
        pendingBubbles.delete(msg.utterance_idx);
        resumeListeningAfterTurn();
      }
      break;
    }

    case "llm.result": {
      const bubble = ensureAssistantBubble(msg.utterance_idx);
      bubble.classList.remove("pending");
      bubble.querySelector(".text").textContent = msg.text;
      bubble.querySelector(".meta").textContent = `LLM ${msg.latency_ms} ms · TTS 합성 중…`;
      break;
    }

    case "tts.audio": {
      // [legacy] one-shot WAV — 더는 서버가 보내지 않음. 호환용으로 남김.
      const bubble = ensureAssistantBubble(msg.utterance_idx);
      const meta = bubble.querySelector(".meta");
      meta.textContent = meta.textContent.replace("TTS 합성 중…", `TTS ${msg.latency_ms} ms`);
      playAudio(msg.wav_base64);
      break;
    }

    case "tts.start": {
      // TTS 스트리밍 시작 — 즉시 플레이어 초기화, ttfb 표시
      const bubble = ensureAssistantBubble(msg.utterance_idx);
      bubble.classList.add("streaming");
      const meta = bubble.querySelector(".meta");
      // sample rate override (드롭다운에서 잘못된 값 고르면 의도적으로 이상한 톤 재현)
      const srOverride = document.getElementById("sample-rate-select")?.value;
      const sampleRate = srOverride ? parseInt(srOverride, 10) : msg.sample_rate;
      const srNote = sampleRate === msg.sample_rate ? "" : ` ⚠sr=${sampleRate}(서버는 ${msg.sample_rate})`;
      meta.textContent = meta.textContent.replace("TTS 합성 중…", `TTS ttfb ${msg.ttfb_ms}ms · 스트리밍…${srNote}`);
      // 이미 있다면 정리
      pcmPlayers.get(msg.utterance_idx)?.abort();
      const player = new PcmStreamPlayer({
        sampleRate,
        sampleFormat: msg.sample_format,
        channels: msg.channels,
        onEnded: () => {
          bubble.classList.remove("streaming");
          resumeListeningAfterTurn();
        },
      });
      pcmPlayers.set(msg.utterance_idx, player);
      setStatus("speaking", "processing");
      break;
    }

    case "tts.chunk": {
      const player = pcmPlayers.get(msg.utterance_idx);
      if (!player) { console.warn("tts.chunk for unknown idx", msg.utterance_idx); break; }
      player.feed(base64ToBytes(msg.pcm_base64));
      break;
    }

    case "tts.end": {
      const player = pcmPlayers.get(msg.utterance_idx);
      if (!player) break;
      const bubble = ensureAssistantBubble(msg.utterance_idx);
      const meta = bubble.querySelector(".meta");
      meta.textContent = meta.textContent.replace(/스트리밍…?/, `${msg.total_chunks} chunks · ${msg.total_latency_ms}ms`);
      player.end();
      // onEnded 콜백에서 마이크 재개됨; 메모리 정리는 onEnded 이후
      setTimeout(() => pcmPlayers.delete(msg.utterance_idx), 5000);
      break;
    }

    case "error": {
      console.error("server error", msg);
      addSystemBubble(`[${msg.stage}] ${msg.message}`);
      setStatus("error", "error");
      // 에러여도 다음 턴을 위해 마이크 복귀
      resumeListeningAfterTurn();
      break;
    }
  }
}

// ── Bubble helpers ───────────────────────────────────────────────────────────
function addBubble(role, text, pending) {
  const div = document.createElement("div");
  div.className = `bubble ${role}` + (pending ? " pending" : "");
  div.innerHTML = `<div class="text"></div><div class="meta"></div>`;
  div.querySelector(".text").textContent = text;
  messagesEl.appendChild(div);
  messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
  return div;
}

function addSystemBubble(text) {
  const div = addBubble("assistant", text, false);
  div.style.background = "#3a1f1f";
  div.style.color = "#ffd2d2";
}

function ensureUserBubble(idx) {
  let entry = pendingBubbles.get(idx);
  if (!entry) {
    entry = {};
    pendingBubbles.set(idx, entry);
  }
  if (!entry.user) {
    entry.user = addBubble("user", "…", true);
  }
  return entry.user;
}

function ensureAssistantBubble(idx) {
  let entry = pendingBubbles.get(idx);
  if (!entry) {
    entry = {};
    pendingBubbles.set(idx, entry);
  }
  if (!entry.assistant) {
    entry.assistant = addBubble("assistant", "…", true);
  }
  return entry.assistant;
}

// ── Audio playback ───────────────────────────────────────────────────────────
function playAudio(b64) {
  const blob = base64ToBlob(b64, "audio/wav");
  const url = URL.createObjectURL(blob);
  player.src = url;
  setStatus("speaking", "processing");
  player.play().catch((e) => {
    console.warn("autoplay blocked", e);
    // 재생이 안 되면 곧장 마이크 복귀
    URL.revokeObjectURL(url);
    resumeListeningAfterTurn();
  });
  player.onended = () => {
    URL.revokeObjectURL(url);
    resumeListeningAfterTurn();
  };
}

// 한 턴(STT→LLM→TTS 재생)이 끝나면 mic_disabled 를 해제해 다시 듣기 시작.
function resumeListeningAfterTurn() {
  if (!listening) return;
  vadCtrl?.clearPriority("mic_disabled");
  setStatus("listening", "listening");
}

// ── Mic device picker ────────────────────────────────────────────────────────
async function populateMicSelect() {
  try {
    // Need to grant permission once to see device labels (Chrome hides them otherwise)
    const probeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    probeStream.getTracks().forEach(t => t.stop());
  } catch (e) {
    console.warn("mic probe permission denied:", e);
  }
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "audioinput");
  micSelect.innerHTML = "";
  const def = document.createElement("option");
  def.value = ""; def.textContent = "(default device)";
  micSelect.appendChild(def);
  for (const d of devs) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `(unnamed ${d.deviceId.slice(0, 6)})`;
    micSelect.appendChild(opt);
  }
  console.log(`[mic-select] ${devs.length} input(s) enumerated`);
}

// When user changes mic while listening, restart VAD with the new device
micSelect?.addEventListener("change", async () => {
  if (!listening) return;
  setStatus("switching mic…");
  if (vadCtrl) vadCtrl.destroy();
  vad = null; vadCtrl = null;
  try {
    await startListening();
  } catch (e) {
    console.error("mic switch failed", e);
    setStatus("mic error", "error");
  }
});

// ── VAD wiring ───────────────────────────────────────────────────────────────
async function ensureVAD() {
  if (vad) return vad;
  setStatus("loading VAD…");
  const deviceId = micSelect?.value || undefined;
  vad = await createVAD({
    deviceId,
    onSpeechStart: () => {
      setStatus("listening", "listening");
      clearTimeout(utteranceCapTimer);
      utteranceCapTimer = setTimeout(() => {
        toast("발화가 30초를 넘어 자동 커밋됩니다");
        // pause→start 로 강제 종료 신호. VADController 가 misfire 보다 안전한 경로.
        vadCtrl?.pause();
        vadCtrl?.start();
      }, 30000);
    },
    onSpeechEnd: (float32) => {
      clearTimeout(utteranceCapTimer);
      vadCtrl?.onSpeechCommitted();
      // 처리/재생이 끝날 때까지 VAD 일시정지 — 자기 응답을 다시 듣지 않도록.
      vadCtrl?.setPriority("mic_disabled");
      setStatus("processing", "processing");

      const wav = encodeWavPcm16(float32, 16000);
      const wavBase64 = arrayBufferToBase64(wav);
      const idx = ++utteranceIdx;
      pendingBubbles.set(idx, { user: null, assistant: null });
      ensureUserBubble(idx);
      ws.send(JSON.stringify({
        type: "audio.utterance",
        wav_base64: wavBase64,
        utterance_idx: idx,
      }));
    },
    onMisfire: () => {
      vadCtrl?.onMisfire(toast);
      setStatus("listening", "listening");
    },
  });
  vadCtrl = new VADController(vad);
  return vad;
}

function setMicBtnLabel(text) {
  // SVG 아이콘을 보존하고 <span>의 텍스트만 갱신
  const span = micBtn.querySelector("span");
  if (span) span.textContent = text;
  else micBtn.textContent = text;
}

async function startListening() {
  await ensureVAD();
  vadCtrl.start();
  listening = true;
  micBtn.classList.add("listening");
  setMicBtnLabel("마이크 정지");
  setStatus("listening", "listening");
}

function stopListening() {
  clearTimeout(utteranceCapTimer);
  if (vadCtrl) vadCtrl.setPriority("session_stopping");
  listening = false;
  micBtn.classList.remove("listening");
  setMicBtnLabel("마이크 시작");
  setStatus("connected", "connected");
}

// ── Buttons ──────────────────────────────────────────────────────────────────
micBtn.addEventListener("click", async () => {
  try {
    if (listening) stopListening();
    else await startListening();
  } catch (e) {
    console.error(e);
    setStatus("mic error", "error");
  }
});

resetBtn.addEventListener("click", () => {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "history.reset" }));
  }
  messagesEl.innerHTML = "";
  pendingBubbles.clear();
  utteranceIdx = 0;
});

connect();
