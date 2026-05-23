// Silero VAD integration via @ricky0123/vad-web (ESM from esm.sh).
// The library loads ONNX silero_vad model + onnxruntime-web from its CDN.

// === DIAGNOSTIC: intercept getUserMedia to surface mic device + enumerate available inputs ===
(() => {
  if (!navigator.mediaDevices || window.__gumPatched) return;
  window.__gumPatched = true;
  const orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => {
    const stream = await orig(c);
    const t = stream.getAudioTracks()[0];
    window.__micDevice = t?.label || '(no label)';
    console.log('[mic] active device =', window.__micDevice, '| settings:', t?.getSettings?.());
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      window.__micList = devs.filter(d => d.kind === 'audioinput').map(d => d.label || '(unnamed)');
      console.log('[mic] available audio inputs:', window.__micList);
      // banner — styled by #mic-info rule in styles.css (between header & main)
      let banner = document.getElementById('mic-info');
      if (!banner) {
        banner = document.createElement('div');
        banner.id = 'mic-info';
        const header = document.querySelector('header');
        if (header && header.nextSibling) header.parentNode.insertBefore(banner, header.nextSibling);
        else document.body.insertBefore(banner, document.body.firstChild);
      }
      banner.textContent =
        `MIC ▸ ${window.__micDevice}\n` +
        `INPUTS ▸ ${window.__micList.join(' · ') || '(none enumerated)'}`;
    } catch (e) { console.error('[mic] enumerate failed', e); }
    return stream;
  };
})();

import { MicVAD } from "https://esm.sh/@ricky0123/vad-web@0.0.24";

export async function createVAD({ onSpeechStart, onSpeechEnd, onMisfire, deviceId }) {
  const opts = {
    positiveSpeechThreshold: 0.5,
    // hangover 동안 음성을 더 너그럽게 유지하기 위해 negativeSpeechThreshold 낮춤
    negativeSpeechThreshold: 0.25,
    preSpeechPadFrames: 30,   // ~960ms pre-pad
    // 한국어 발화 중 호흡/조사 사이 멈춤은 보통 0.5~1.2초.
    // 24(768ms) 너무 짧아서 한 문장이 2~3조각으로 잘리던 현상 → 48(~1.54s)로 상향.
    redemptionFrames: 48,
    frameSamples: 512,
    onSpeechStart: () => { try { onSpeechStart?.(); } catch (e) { console.error(e); } },
    onSpeechEnd: (audio) => { try { onSpeechEnd?.(audio); } catch (e) { console.error(e); } },
    onVADMisfire: () => { try { onMisfire?.(); } catch (e) { console.error(e); } },
    // === DIAGNOSTIC (temporary): live Silero probability + raw mic RMS ===
    onFrameProcessed: (probs, frame) => {
      const s = document.getElementById('status');
      if (!s) return;
      const p = probs.isSpeech ?? 0;
      // raw audio RMS from the frame (Float32Array of 512 samples @16kHz)
      let rms = 0;
      if (frame && frame.length) {
        let sum = 0;
        for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
        rms = Math.sqrt(sum / frame.length);
      }
      const dbg = window.__vadDbg || (window.__vadDbg = { last: 0, maxP: 0, maxRms: 0, frames: 0 });
      dbg.frames++;
      dbg.maxP = Math.max(dbg.maxP, p);
      dbg.maxRms = Math.max(dbg.maxRms, rms);
      if (Date.now() - dbg.last > 300) {
        dbg.last = Date.now();
        console.log(`vad#${dbg.frames} p=${p.toFixed(3)} rms=${rms.toFixed(4)} (maxP=${dbg.maxP.toFixed(3)} maxRms=${dbg.maxRms.toFixed(4)})`);
      }
      if (s.classList.contains('listening')) {
        const pBar = '▰'.repeat(Math.min(6, Math.round(p * 6))).padEnd(6, '▱');
        const rBar = '▰'.repeat(Math.min(6, Math.round(rms * 60))).padEnd(6, '▱');
        s.textContent = `vad ${pBar} ${p.toFixed(2)} · rms ${rBar} ${rms.toFixed(3)}`;
      }
    },
  };
  if (deviceId) {
    opts.additionalAudioConstraints = { deviceId: { exact: deviceId } };
    console.log('[vad] using explicit deviceId =', deviceId);
  }
  const vad = await MicVAD.new(opts);
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
