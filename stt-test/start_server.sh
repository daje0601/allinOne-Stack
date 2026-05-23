#!/usr/bin/env bash
# ==============================================================================
# Whisper STT 서버 기동 스크립트
# ==============================================================================
#
# 이 스크립트가 하는 일:
#   1. vLLM이라는 추론 엔진으로 "openai/whisper-large-v3-turbo" 모델을 띄움
#   2. 11000번 포트에 OpenAI 호환 HTTP API로 서빙
#   3. 서버가 응답하기 시작할 때까지 기다림 (보통 1-3분)
#   4. 첫 요청 지연을 없애기 위해 "워밍업" 더미 요청 1발 발사
#   5. 사용자가 Ctrl+C로 중단할 때까지 foreground로 대기
#
# 비개발자용 용어 설명:
#   - 모델(Model): AI의 "두뇌"가 담긴 파일 (수 GB 크기)
#   - 추론(Inference): 모델에게 입력을 주고 답을 받는 것
#   - 서버(Server): "들어오는 요청을 받아 응답하는 프로그램"
#   - 포트(Port): 한 컴퓨터 안의 "주소". 11000번을 통해 이 서버에 접근
#   - HTTP API: 웹사이트가 쓰는 것과 같은 표준 통신 방식
#
# 첫 줄의 #!/usr/bin/env bash는 "이 파일은 bash라는 셸로 실행하라"는 표시.
# ==============================================================================

# set -e   : 어떤 명령이든 실패하면 즉시 스크립트 중단 (오류 무시 방지)
# set -u   : 정의 안 한 변수 쓰면 에러 (오타 방지)
# set -o pipefail: 파이프(|)로 연결된 명령 중간이 실패해도 알아챔
set -euo pipefail

# 스크립트가 어디서 호출되든 항상 이 스크립트가 놓인 폴더로 이동.
# $(dirname "$0") = "이 스크립트의 경로에서 파일명 빼고 폴더만"
cd "$(dirname "$0")"

# 셸 환경에 LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64가 설정돼있으면
# 우리 venv 안의 NVIDIA 라이브러리(cu12.4용)가 가려져서 충돌이 남.
# 그래서 이 스크립트에서는 그 변수를 지운 채로 진행.
unset LD_LIBRARY_PATH

# 이 머신엔 GPU가 2개(0~1). vLLM에 "GPU 0번만 보여라"라고 알려줌.
# 배치: STT+TTS는 GPU 0 공유, LLM은 GPU 1 독점.
export CUDA_VISIBLE_DEVICES=0

# ------------------------------------------------------------------------------
# 설정 변수
# ------------------------------------------------------------------------------
PORT=11000                                     # 서비스 포트
MODEL=openai/whisper-large-v3-turbo            # HuggingFace 모델 ID (없으면 자동 다운로드)
WARMUP_WAV=samples/stt-warmup.wav              # 워밍업용 짧은 영어 음성 파일
READY_TIMEOUT=1800                             # 서버 준비 대기 최대 시간(초) — 첫 다운로드(~1.5GB) + CUDA graph capture 여유로 30분

# ------------------------------------------------------------------------------
# 0단계: 워밍업용 wav 파일이 없으면 만들어두기
# ------------------------------------------------------------------------------
# vLLM이 자체적으로 가지고 있는 "mary_had_lamb"(에디슨이 녹음한 동요 5초)을
# 가져와서 samples/stt-warmup.wav로 저장. 한 번만 만들면 다음 실행부터 재사용.
mkdir -p samples                               # samples 폴더가 없으면 만듦
# [[ ! -s "$WARMUP_WAV" ]] : "이 파일이 없거나 비어있다면"
if [[ ! -s "$WARMUP_WAV" ]]; then
  echo "[start_server] preparing warmup audio at $WARMUP_WAV"
  # uv run python -c "..." : venv 환경에서 Python 한 줄 실행
  uv run python -c "
import soundfile as sf
from vllm.assets.audio import AudioAsset
audio, sr = AudioAsset('mary_had_lamb').audio_and_sample_rate
sf.write('$WARMUP_WAV', audio, sr, format='WAV', subtype='PCM_16')
"
fi

# ------------------------------------------------------------------------------
# 1단계: vLLM 서버를 백그라운드에서 띄우고, 종료 시 정리 약속(trap) 등록
# ------------------------------------------------------------------------------
echo "[start_server] launching vllm serve on :$PORT"
# uv run vllm serve <모델> [옵션들] &
#   --host 0.0.0.0          어떤 네트워크 인터페이스든 받겠다 (LAN 접근 허용)
#   --port 11000            이 포트에서 listen
#   --max-model-len 448     한 요청의 디코더 컨텍스트 최대 길이 (Whisper 한계)
#   --max-num-seqs 400      동시에 처리할 수 있는 최대 요청 수
#   --kv-cache-dtype fp8    KV 캐시를 8비트로 저장해 메모리 절약
#   --enforce-eager         CUDA graph capture 비활성 → 워밍업 빠름 (~5분 → ~1분).
#                           추론은 미세하게 느려짐. 약한 GPU에서 600s timeout 회피용.
#   --gpu-memory-utilization 0.3  GPU 메모리의 30%만 사용 (TTS와 GPU 0 공유)
#   &                       백그라운드 실행
uv run vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 448 \
    --max-num-seqs 400 \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --gpu-memory-utilization 0.3 &
SERVER_PID=$!                                  # 방금 띄운 백그라운드 프로세스의 PID 저장

# cleanup(): 스크립트 종료 시 호출되는 정리 함수
cleanup() {
  echo
  echo "[start_server] stopping server (pid $SERVER_PID)"
  kill -TERM "$SERVER_PID" 2>/dev/null || true # 부드러운 종료 신호 (실패해도 무시)
  wait "$SERVER_PID" 2>/dev/null || true        # 완전히 종료될 때까지 기다림
}
# trap: "INT(Ctrl+C)/TERM(kill)/EXIT(스크립트 끝) 신호가 오면 cleanup 호출하라"
trap cleanup INT TERM EXIT

# ------------------------------------------------------------------------------
# 2단계: 서버가 준비될 때까지 폴링 (몇 초에 한 번씩 두드려보기)
# ------------------------------------------------------------------------------
echo "[start_server] waiting for server ready (timeout ${READY_TIMEOUT}s)..."
deadline=$((SECONDS + READY_TIMEOUT))          # 마감 시각 = 현재 + 300초

# until ...; do ...; done : "조건이 참이 될 때까지 반복"
# curl -fsS http://localhost:11000/v1/models > /dev/null 2>&1
#   서버에 GET 요청 보내서 응답을 받으면 성공(0), 실패면 비0.
#   응답 본문은 /dev/null로 버림 (어차피 성공 여부만 보면 됨)
until curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
  # 서버 프로세스가 죽었나? (kill -0은 신호 안 보내고 존재 여부만 확인)
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[start_server] ERROR: server process died before becoming ready" >&2
    exit 1
  fi
  # 마감 넘었나?
  if (( SECONDS > deadline )); then
    echo "[start_server] ERROR: readiness timeout after ${READY_TIMEOUT}s" >&2
    exit 1
  fi
  sleep 2                                      # 2초 쉬고 다시 시도
done
echo "[start_server] /v1/models OK"

# ------------------------------------------------------------------------------
# 3단계: 워밍업 — 첫 진짜 요청이 느리지 않도록 더미 1발 발사
# ------------------------------------------------------------------------------
echo "[start_server] sending warmup request..."
t0=$(date +%s.%N)                              # 시작 시각(나노초 포함)
# curl -F "file=@경로" : multipart/form-data 형식으로 파일 업로드
# (브라우저 폼에서 파일 첨부하는 것과 같은 형식)
curl -fsS -o /dev/null \
  -X POST "http://localhost:$PORT/v1/audio/transcriptions" \
  -F "file=@$WARMUP_WAV" \
  -F "model=$MODEL" \
  -F "language=en" \
  -F "response_format=json"
t1=$(date +%s.%N)                              # 종료 시각
# bc는 부동소수점 계산기 (셸의 산술은 정수만 가능)
printf "[start_server] warmup done in %.2fs — server warm and ready.\n" "$(echo "$t1 - $t0" | bc)"

# ------------------------------------------------------------------------------
# 4단계: foreground로 대기. Ctrl+C 누르면 trap이 cleanup 호출.
# ------------------------------------------------------------------------------
echo "[start_server] foreground attached. Ctrl+C to stop."
# wait $PID: 그 프로세스가 끝날 때까지 셸이 여기서 멈춤
wait "$SERVER_PID"
