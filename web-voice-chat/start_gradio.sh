#!/usr/bin/env bash
# ==============================================================================
# Gradio 버전 음성 챗 기동 (share=True → *.gradio.live 공개 URL 발급).
# 사전 조건: stt(:11000), llm(:13000), tts(:12000) 모두 가동 중.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[gradio-voice-chat] launching (share=True)"
export PYTHONUNBUFFERED=1
exec uv run python -u gradio_app.py
