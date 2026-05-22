#!/usr/bin/env bash
# ==============================================================================
# web-voice-chat 서버 기동 (FastAPI + WS, 정적 파일 서빙).
#
# 사전 조건: stt(:11000), llm(:13000), tts(:12000) 모두 가동 중.
# 환경변수는 .env (필요 시) 또는 OS env로 오버라이드.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PORT="${APP_PORT:-14000}"
HOST="${APP_HOST:-0.0.0.0}"

echo "[web-voice-chat] launching on http://${HOST}:${PORT}"
# 1 worker로 충분 (WS + 세션 상태가 메모리에 있어서 다중 worker 안 어울림)
exec uv run uvicorn server.app:app --host "$HOST" --port "$PORT" --log-level info
