#!/usr/bin/env bash
# ==============================================================================
# LLaMA3 페르소나 챗봇 서버 기동 스크립트 (vLLM)
# ==============================================================================
#
# 이 스크립트가 하는 일:
#   1. vLLM으로 "iamjoon/llama3-8b-persona-chatbot"을 띄움
#      (LLaMA3 8B를 한국어 페르소나 챗봇용으로 파인튜닝한 HF 공개 모델)
#   2. 13000번 포트에 OpenAI 호환 HTTP API로 서빙
#      엔드포인트: POST /v1/chat/completions (system+user 메시지 → assistant 응답)
#   3. 워밍업 1발 → foreground 대기
#
# 이 서버를 직접 호출해도 챗봇으로 사용 가능하지만, 진짜 페르소나(곰돌이 푸)를
# 살리려면 system 메시지에 system_prompt.txt를 넣어줘야 함.
# RAG(에피소드 검색)까지 붙은 풀스택은 `app.py` 참고.
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")"
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3                  # GPU 3번 (GPU 0=whisper, 2=qwen3-tts)

# ------------------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------------------
PORT=13000
MODEL="iamjoon/llama3-8b-persona-chatbot"      # LLaMA3 8B 한국어 페르소나 파인튜닝
READY_TIMEOUT=600                              # 모델 다운로드(~16GB) 포함이면 길 수 있음

# ------------------------------------------------------------------------------
# 1단계: vLLM 서버 백그라운드 기동
# ------------------------------------------------------------------------------
echo "[start_llm] launching vllm serve on :$PORT (GPU 3)"
# 옵션 설명:
#   --max-model-len 8192                컨텍스트 길이 상한 (system prompt + 대화 + 응답)
#   --gpu-memory-utilization 0.85       GPU 메모리의 85%까지 사용 → 충분한 KV 캐시
#   --dtype bfloat16                    BF16으로 가중치 저장 (LLaMA3 표준)
uv run vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 &
SERVER_PID=$!

# 종료 시 정리
cleanup() {
  echo
  echo "[start_llm] stopping server (pid $SERVER_PID)"
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# ------------------------------------------------------------------------------
# 2단계: 서버 준비 대기
# ------------------------------------------------------------------------------
echo "[start_llm] waiting for server ready (timeout ${READY_TIMEOUT}s)..."
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[start_llm] ERROR: server process died before becoming ready" >&2
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "[start_llm] ERROR: readiness timeout" >&2
    exit 1
  fi
  sleep 3
done
echo "[start_llm] /v1/models OK"

# ------------------------------------------------------------------------------
# 3단계: 워밍업 — 짧은 "안녕"으로 첫 추론 한 번 돌려둠
# ------------------------------------------------------------------------------
echo "[start_llm] warmup completion..."
t0=$(date +%s.%N)
# /v1/chat/completions에 JSON body로 요청.
# 셸 안에서 JSON 문자열을 만들려니 따옴표 escape(\")가 많아 보임.
# {
#   "model": "iamjoon/llama3-8b-persona-chatbot",
#   "messages": [{"role": "user", "content": "안녕"}],
#   "max_tokens": 16
# }
curl -fsS -o /dev/null \
  -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"안녕\"}],\"max_tokens\":16}"
t1=$(date +%s.%N)
printf "[start_llm] warmup done in %.2fs.\n" "$(echo "$t1 - $t0" | bc)"

# ------------------------------------------------------------------------------
# 4단계: foreground 대기
# ------------------------------------------------------------------------------
echo "[start_llm] foreground attached. Ctrl+C to stop."
wait "$SERVER_PID"
