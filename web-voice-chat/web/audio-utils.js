// Float32Array (16kHz mono PCM from Silero VAD) → WAV (RIFF) ArrayBuffer
// Returns a complete RIFF WAVE file (44-byte header + PCM16 LE data).

export function encodeWavPcm16(float32, sampleRate = 16000) {
  const numSamples = float32.length;
  const dataSize = numSamples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");

  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);

  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

export function arrayBufferToBase64(ab) {
  const bytes = new Uint8Array(ab);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

// base64 → Blob (for audio playback)
export function base64ToBlob(b64, mime = "audio/wav") {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

// base64 → Uint8Array (no Blob wrapper)
export function base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// ── PCM streaming player ─────────────────────────────────────────────────────
// 청크 단위로 들어오는 int16 LE mono PCM을 끊김 없이 chained playback.
// AudioBuffer를 매 청크마다 만들고 nextTime 에 정확히 schedule → gapless 재생.
export class PcmStreamPlayer {
  constructor({
    sampleRate,
    sampleFormat = "int16le",
    channels = 1,
    onEnded,
    preRollMs = 700,           // jitter 흡수 (톤 영향 없음)
    playbackRate = 1.0,        // 정상 톤. 클라이언트 측 끊김 해결 한계 인정 — 서버 RTF 개선이 본질.
  } = {}) {
    this.sampleRate = sampleRate;
    this.sampleFormat = sampleFormat;
    this.channels = channels;
    this.onEnded = onEnded;
    this.playbackRate = playbackRate;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate, latencyHint: 'playback' });
    this.nextTime = 0;
    this.scheduledSources = new Set();
    this.endRequested = false;
    this._leftover = null;
    this.preRollMs = preRollMs;
    this.playbackStarted = false;
    this.pendingBuffers = [];
    this.pendingDurationMs = 0;
    // ── 진단: RTF / underrun 추적 ────────────────────────────────────────
    this._t0 = performance.now();
    this._chunksReceived = 0;
    this._audioMsAccum = 0;       // 받은 PCM의 총 분량(ms)
    this._underrunCount = 0;      // nextTime이 currentTime에 너무 가까웠던 횟수 = 실질 갭
  }

  // 청크 1개 enqueue. ArrayBuffer 또는 Uint8Array 받음.
  feed(bytes) {
    if (this.endRequested) return;
    if (this.sampleFormat !== "int16le") throw new Error(`unsupported format ${this.sampleFormat}`);
    let u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    if (this._leftover) {
      const combined = new Uint8Array(this._leftover.length + u8.length);
      combined.set(this._leftover, 0);
      combined.set(u8, this._leftover.length);
      u8 = combined;
      this._leftover = null;
    }
    if (u8.length % 2 !== 0) {
      this._leftover = u8.subarray(u8.length - 1);
      u8 = u8.subarray(0, u8.length - 1);
    }
    if (u8.length === 0) return;

    // PCM → AudioBuffer 변환
    const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    const i16 = new Int16Array(ab);
    if (i16.length === 0) return;
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const buf = this.ctx.createBuffer(this.channels, f32.length / this.channels, this.sampleRate);
    if (this.channels === 1) buf.copyToChannel(f32, 0);
    else for (let c = 0; c < this.channels; c++) {
      const ch = new Float32Array(f32.length / this.channels);
      for (let i = 0; i < ch.length; i++) ch[i] = f32[i * this.channels + c];
      buf.copyToChannel(ch, c);
    }

    // 진단 누적
    this._chunksReceived++;
    this._audioMsAccum += buf.duration * 1000;

    if (!this.playbackStarted) {
      this.pendingBuffers.push(buf);
      this.pendingDurationMs += buf.duration * 1000;
      if (this.pendingDurationMs >= this.preRollMs) {
        this._flushPending();
      }
      return;
    }
    this._scheduleBuffer(buf);
  }

  _flushPending() {
    if (this.playbackStarted) return;
    this.playbackStarted = true;
    this.nextTime = this.ctx.currentTime + 0.05;
    for (const buf of this.pendingBuffers) this._scheduleBuffer(buf);
    this.pendingBuffers = [];
    this.pendingDurationMs = 0;
  }

  _scheduleBuffer(buf) {
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    src.playbackRate.value = this.playbackRate;

    // underrun 진단 (resolution: 톤 변경 없이 가시화만)
    const lookahead = this.nextTime - this.ctx.currentTime;
    if (lookahead < 0.05) {
      this._underrunCount++;
      console.warn(`[pcm] UNDERRUN #${this._underrunCount} lookahead=${(lookahead*1000).toFixed(0)}ms chunk=${this._chunksReceived}`);
    } else if (lookahead < 0.3) {
      console.log(`[pcm] low buffer lookahead=${(lookahead*1000).toFixed(0)}ms chunk=${this._chunksReceived}`);
    }
    if (this.nextTime < this.ctx.currentTime + 0.01) {
      this.nextTime = this.ctx.currentTime + 0.01;
    }
    src.start(this.nextTime);
    this.scheduledSources.add(src);
    src.onended = () => {
      this.scheduledSources.delete(src);
      if (this.endRequested && this.scheduledSources.size === 0) {
        const wall = performance.now() - this._t0;
        const audio = this._audioMsAccum;
        console.log(`[pcm] done · chunks=${this._chunksReceived} audio=${audio.toFixed(0)}ms wall=${wall.toFixed(0)}ms effRTF=${(wall/audio).toFixed(3)} underruns=${this._underrunCount}`);
        this.onEnded?.();
        this.ctx.close().catch(() => {});
      }
    };
    this.nextTime += buf.duration / this.playbackRate;
  }

  // 더 이상 청크 없음 신호. pre-roll 미달이면 모은 만큼 그대로 재생.
  end() {
    this.endRequested = true;
    if (!this.playbackStarted && this.pendingBuffers.length > 0) {
      this._flushPending();
    }
    if (this.scheduledSources.size === 0) {
      this.onEnded?.();
      this.ctx.close().catch(() => {});
    }
  }

  // 강제 종료 (turn 중단 등)
  abort() {
    this.endRequested = true;
    this.pendingBuffers = [];
    for (const src of this.scheduledSources) {
      try { src.onended = null; src.stop(); } catch {}
    }
    this.scheduledSources.clear();
    this.ctx.close().catch(() => {});
  }
}
