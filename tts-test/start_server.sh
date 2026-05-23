#!/usr/bin/env bash
# ==============================================================================
# Qwen3-TTS 서버 기동 스크립트 (vllm-omni)
# ==============================================================================
#
# 이 스크립트가 하는 일:
#   1. vllm-omni라는 확장 추론 엔진으로 Qwen3-TTS 모델을 띄움
#      (vllm-omni = vLLM의 멀티모달 확장판. TTS/이미지 생성 등을 지원)
#   2. 12000번 포트에 OpenAI 호환 HTTP API로 서빙
#      엔드포인트: POST /v1/audio/speech (텍스트 → 음성 wav)
#   3. 서버 준비 대기 → 워밍업 1발 → foreground 대기
#
# 비개발자용 용어:
#   - TTS (Text-To-Speech): 텍스트를 음성(wav)으로 변환
#   - STT (Speech-To-Text): 음성을 텍스트로 변환 (반대 방향, stt-test 폴더 참고)
#   - 2-stage 파이프라인: Qwen3-TTS는 내부에 두 단계를 가짐.
#       Stage 0 (Talker)    : 텍스트 → "스피치 토큰"
#       Stage 1 (Code2Wav)  : 스피치 토큰 → 실제 음파 wav
#
# 중요 함정 (vllm-omni 0.16.0):
#   `--omni` 플래그는 **반드시 `serve` 다음에** 와야 함.
#   `vllm-omni --omni serve ...` (X, argparse 에러)
#   `vllm-omni serve ... --omni` (O)
# ==============================================================================

set -euo pipefail                              # 엄격 모드 (오류 즉시 중단)
cd "$(dirname "$0")"                           # 스크립트가 놓인 폴더로 이동
unset LD_LIBRARY_PATH                          # 시스템 CUDA 라이브러리 가림 회피
export CUDA_VISIBLE_DEVICES=0                  # GPU 0번을 STT와 공유 (LLM은 GPU 1)

# ------------------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------------------
PORT=12000                                     # 서비스 포트
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"   # HuggingFace 모델 ID
READY_TIMEOUT=900                              # 2-stage 파이프라인이라 더 김 (15분)

# ------------------------------------------------------------------------------
# 1단계: vllm-omni 서버 백그라운드 기동
# ------------------------------------------------------------------------------
echo "[start_tts] launching vllm-omni serve on :$PORT (--omni mode)"
# 핵심 옵션:
#   serve <model>          모델을 OpenAI 호환 서버로 띄움
#   --omni                 vLLM-Omni 모드 활성화 (이게 없으면 일반 LLM 서버로 떨어짐)
#   --host 0.0.0.0         LAN 접근 허용
#   --port 12000           포트
#   --trust-remote-code    HF Hub 모델의 커스텀 코드 신뢰 (Qwen3-TTS는 필수)
#   --gpu-memory-utilization 0.85
#       STT(0.3)와 GPU 0 공유하지만 공격적으로 0.85 사용.
#       합계 0.3 + 0.85 = 1.15 > 1.0 → OOM 위험 감수하고 RTF 개선 시도.
#       OOM 나면 0.6 정도로 백오프 필요.
uv run vllm-omni serve "$MODEL" \
    --omni \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 &
SERVER_PID=$!                                  # 백그라운드 PID 기록

# 종료 시 정리 함수
cleanup() {
  echo
  echo "[start_tts] stopping server (pid $SERVER_PID)"
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  # vllm-omni는 worker 자식이 많으므로 이름 기반으로도 한 번 처리
  pkill -f "Qwen3-TTS\|run_omni_serve" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# ------------------------------------------------------------------------------
# 2단계: 서버 준비 대기 (2-stage이라 보통 2-3분, 첫 다운로드면 더 길게)
# ------------------------------------------------------------------------------
echo "[start_tts] waiting for server ready (timeout ${READY_TIMEOUT}s)..."
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[start_tts] ERROR: server process died before becoming ready" >&2
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "[start_tts] ERROR: readiness timeout" >&2
    exit 1
  fi
  sleep 3                                      # 3초에 한 번씩 두드림
done
echo "[start_tts] /v1/models OK"

# ------------------------------------------------------------------------------
# 3단계: 워밍업 — 짧은 영어 "Hello, warmup."을 음성으로 변환해 파일로 저장
# ------------------------------------------------------------------------------
echo "[start_tts] sending warmup speech request..."
t0=$(date +%s.%N)
mkdir -p samples
# 이번엔 multipart가 아니라 JSON body로 요청 (TTS는 JSON이 표준).
# -H "Content-Type: application/json" : 요청 본문 형식 명시
# -d '...' : 요청 본문 (JSON 문자열)
# -o samples/tts-warmup.wav : 응답 본문(바이너리 wav)을 이 파일에 저장
curl -fsS -o samples/tts-warmup.wav \
  -X POST "http://localhost:$PORT/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello, warmup.","voice":"vivian","language":"English","response_format":"wav"}' \
  || echo "[start_tts] warmup failed (continuing)"
t1=$(date +%s.%N)
printf "[start_tts] warmup done in %.2fs.\n" "$(echo "$t1 - $t0" | bc)"

# ------------------------------------------------------------------------------
# 4단계: foreground 대기. Ctrl+C로 정지.
# ------------------------------------------------------------------------------
echo "[start_tts] foreground attached. Ctrl+C to stop."
wait "$SERVER_PID"
