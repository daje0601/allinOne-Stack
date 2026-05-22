// Silero VAD integration via @ricky0123/vad-web (ESM from esm.sh).
// The library loads ONNX silero_vad model + onnxruntime-web from its CDN.

import { MicVAD } from "https://esm.sh/@ricky0123/vad-web@0.0.24";

export async function createVAD({ onSpeechStart, onSpeechEnd, onMisfire }) {
  const vad = await MicVAD.new({
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.35,
    preSpeechPadFrames: 30,   // ~960ms pre-pad
    redemptionFrames: 24,     // ~768ms hangover
    frameSamples: 512,
    onSpeechStart: () => { try { onSpeechStart?.(); } catch (e) { console.error(e); } },
    onSpeechEnd: (audio) => { try { onSpeechEnd?.(audio); } catch (e) { console.error(e); } },
    onVADMisfire: () => { try { onMisfire?.(); } catch (e) { console.error(e); } },
  });
  return vad;
}

// Thin state machine around the vad instance with misfire self-healing.
// 우선순위 상태머신: session_stopping > mic_disabled > voice_ready
// session_stopping 또는 mic_disabled 이면 paused, voice_ready 일 때만 listening
export class VADController {
  constructor(vad, opts = {}) {
    this._vad = vad;
    this._status = "paused"; // "paused" | "listening"
    this._misfireCount = 0;
    this._retryCount = 0;
    this._retryWindowStart = 0;
    this._maxRetries = opts.maxRetries ?? 3;
    this._retryWindowMs = opts.retryWindowMs ?? 10_000;
    this._misfireTrigger = opts.misfireTrigger ?? 10;
    this._disabledReasons = new Set(); // "session_stopping" | "mic_disabled"
  }
  get status() { return this._status; }

  /** 우선순위 기반 VAD 상태 전환.
   * @param {"session_stopping"|"mic_disabled"|"voice_ready"} priority
   */
  setPriority(priority) {
    if (priority === "voice_ready") {
      if (this._disabledReasons.size === 0) {
        this._doStart();
      }
    } else {
      this._disabledReasons.add(priority);
      this._doPause();
    }
  }

  /** 비활성 이유 해제. 모든 이유가 없어지면 listening 전환. */
  clearPriority(priority) {
    this._disabledReasons.delete(priority);
    if (this._disabledReasons.size === 0) {
      this._doStart();
    }
  }

  _doStart() {
    if (this._status === "listening") return;
    this._vad.start();
    this._status = "listening";
  }

  _doPause() {
    if (this._status === "paused") return;
    this._vad.pause();
    this._status = "paused";
  }

  start() {
    this._disabledReasons.clear();
    this._doStart();
  }
  pause() {
    this._doPause();
  }
  destroy() {
    try { this._vad.destroy?.(); } catch {}
    this._status = "paused";
  }

  onMisfire(notify) {
    this._misfireCount++;
    if (this._misfireCount < this._misfireTrigger) return;

    const now = Date.now();
    if (now - this._retryWindowStart > this._retryWindowMs) {
      this._retryWindowStart = now;
      this._retryCount = 0;
    }
    this._retryCount++;
    this._misfireCount = 0;

    if (this._retryCount > this._maxRetries) {
      notify?.("VAD 복구 실패 — 마이크 입력을 확인하세요");
      return;
    }
    this.pause();
    setTimeout(() => this.start(), 500);
  }

  onSpeechCommitted() { this._misfireCount = 0; }
}
