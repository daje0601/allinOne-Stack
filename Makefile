# ==============================================================================
# voice-stack 모노레포 - 명령어 단축 모음 (Makefile)
# ==============================================================================
#
# 이 파일은 "Makefile"이라는 표준 빌드 도구의 설정 파일입니다.
# 원래는 C/C++ 같은 언어를 컴파일할 때 쓰는 도구지만, 우리는 그냥
# "긴 명령어들을 짧은 별칭으로 묶어두는 노트"로 사용합니다.
#
# 사용법:
#   make <별칭>     예) make start-stt, make health, make stop-all
#
# 각 별칭은 "target(타겟)"이라 부르고, 아래쪽 그 이름 다음 줄에 들여쓰기로
# 적힌 명령어들이 실제로 실행됩니다. (Makefile은 들여쓰기에 반드시 TAB을 써야 함)
# ==============================================================================

# .PHONY 선언: "이 이름들은 실제 파일이 아니라 그냥 명령어 별칭이다" 라고 알려줌.
# (안 적어도 동작하지만, 같은 이름의 파일이 우연히 생기면 헷갈리지 않도록 명시)
.PHONY: help sync-all start-stt start-tts start-llm start-app stop-all stop-stt stop-tts stop-llm stop-app _stop-port health logs

# ------------------------------------------------------------------------------
# help: 그냥 `make` 또는 `make help` 입력 시 사용 가능한 명령어 목록 출력
# ------------------------------------------------------------------------------
# @ 표시는 "이 명령 자체는 화면에 출력하지 말고 결과만 보여줘"라는 뜻
help:
	@echo "voice-stack 모노레포 명령어:"
	@echo "  make sync-all     — uv 설치 + 스크립트 실행권한 + 3개 서브 프로젝트 uv sync"
	@echo "  make start-stt    — Whisper STT 서버 (:11000, GPU 0 — TTS와 공유)"
	@echo "  make start-tts    — Qwen3-TTS 서버 (:12000, GPU 0 — STT와 공유)"
	@echo "  make start-llm    — LLaMA3 LLM 서버 (:13000, GPU 1 독점)"
	@echo "  make start-app    — 음성 챗 웹 UI (:14000, FastAPI+WS, STT+LLM+TTS 파이프라인)"
	@echo "  make stop-stt     — STT만 정지 (:11000)"
	@echo "  make stop-tts     — TTS만 정지 (:12000)"
	@echo "  make stop-llm     — LLM만 정지 (:13000)"
	@echo "  make stop-app     — 웹 UI만 정지 (:14000)"
	@echo "  make stop-all     — 4개 서비스 모두 정지"
	@echo "  make health       — 4개 endpoint 헬스체크 (HTTP 응답 코드 확인)"
	@echo "  make logs         — 마지막 30줄씩 로그 보기"

# ------------------------------------------------------------------------------
# sync-all: 부트스트랩(uv 설치 + 스크립트 권한) + 3개 서브 프로젝트 의존성 설치
# ------------------------------------------------------------------------------
# uv는 Python 패키지 매니저 (pip의 빠른 대체재). uv sync는 pyproject.toml에 적힌
# 의존성을 .venv 폴더 안에 설치합니다.
# UV_TORCH_BACKEND=cu126은 "PyTorch를 CUDA 12.6 호환 빌드로 가져와라"라는 힌트.
# (이 머신의 GPU 드라이버가 CUDA 12.2까지 native 지원이라 cu126이 안전선)
#
# 앞쪽 두 줄은 "처음 받아온 환경에서도 한 방에 돌아가게" 하기 위한 부트스트랩:
#   1) pip install -U uv         — uv가 없으면 설치, 있으면 최신으로 (idempotent)
#   2) chmod +x .../*.sh         — git clone 직후엔 실행권한이 빠져있어서
#                                   ./start_server.sh가 "Permission denied" 나는 것 방지
sync-all:
	@# 시스템 도구: stop-all에서 쓰는 fuser(=psmisc) 보장
	@# (apt-get update && apt-get install -y psmisc 가 필요하면 주석 풀어 사용 — recipe 라인이라 '#'을 라인 맨 앞에 두면 GNU make가 missing separator 에러 냄)
	pip install -U uv
	chmod +x stt-test/*.sh tts-test/*.sh llm-test/*.sh web-voice-chat/*.sh
	cd stt-test && uv sync
	cd tts-test && UV_TORCH_BACKEND=cu126 uv sync
	cd llm-test && UV_TORCH_BACKEND=cu126 uv sync
	cd web-voice-chat && uv sync
	@# RAG chroma_db 자동 빌드: OPENAI_API_KEY가 llm-test/.env 에 채워져 있을 때만.
	@# 이미 chroma_db가 있으면 스킵. 키가 없으면 안내만 출력하고 통과 (sync-all은 항상 성공).
	@if [ -d llm-test/chroma_db ]; then \
	    echo "[sync-all] llm-test/chroma_db already exists → skipping ingest"; \
	elif [ -f llm-test/.env ] && grep -qE "^OPENAI_API_KEY=sk-" llm-test/.env; then \
	    echo "[sync-all] building chroma_db (RAG index, OpenAI embedding)..."; \
	    cd llm-test && unset LD_LIBRARY_PATH && uv run python ingest.py; \
	else \
	    echo "[sync-all] OPENAI_API_KEY not set in llm-test/.env → skipping chroma_db build."; \
	    echo "[sync-all]   to enable RAG later:"; \
	    echo "[sync-all]     cp llm-test/.env.example llm-test/.env"; \
	    echo "[sync-all]     # llm-test/.env 열어 OPENAI_API_KEY=sk-... 채우기"; \
	    echo "[sync-all]     cd llm-test && uv run python ingest.py"; \
	fi

# ------------------------------------------------------------------------------
# start-XXX: 각 서비스를 백그라운드에서 실행 (foreground 안 잡고 바로 셸 복귀)
# ------------------------------------------------------------------------------
# 명령어 해석:
#   cd <폴더>              그 폴더로 이동
#   nohup ./start_server.sh   "터미널 끊겨도 죽지 마라" + 스크립트 실행
#   > server.log 2>&1      표준 출력(1) + 오류(2) 모두 server.log 파일로 저장
#   &                      "백그라운드에서 돌려라" (셸이 안 기다림)
start-stt:
	cd stt-test && nohup ./start_server.sh > server.log 2>&1 &
	@echo "stt-test launched, tail -f stt-test/server.log"

start-tts:
	cd tts-test && nohup ./start_server.sh > server.log 2>&1 &
	@echo "tts-test launched, tail -f tts-test/server.log"

start-llm:
	cd llm-test && nohup ./start_server.sh > server.log 2>&1 &
	@echo "llm-test launched, tail -f llm-test/server.log"

# 음성 챗 웹 UI (FastAPI + WS). STT/LLM/TTS 셋 다 떠있어야 정상 동작.
start-app:
	cd web-voice-chat && nohup ./start_server.sh > server.log 2>&1 &
	@echo "web-voice-chat launched, tail -f web-voice-chat/server.log"

# ------------------------------------------------------------------------------
# stop-{stt,tts,llm,app}: 개별 서비스만 깔끔히 종료
# ------------------------------------------------------------------------------
# stop-all과 달리 "다른 서비스의 worker는 건드리면 안 되므로" pkill 패턴 매칭은
# 못 씁니다. 대신 process group(PGID) 단위로 죽입니다.
#
# 동작 원리:
#   1) fuser로 해당 포트를 잡은 메인 PID를 찾음
#   2) ps로 그 PID의 PGID 조회 (vLLM 부모 + worker 자식들이 같은 PGID 공유)
#   3) kill -9 -<PGID> 로 그룹 전체를 한 번에 보냄 → worker도 같이 정리
#   4) sleep 후 포트가 비었는지 한 번 더 검증
#
# 모두 _stop-port 헬퍼를 재호출하는 방식으로 공통 로직을 한 곳에 둠.
stop-stt:
	@$(MAKE) --no-print-directory _stop-port PORT=11000 NAME=stt
stop-tts:
	@$(MAKE) --no-print-directory _stop-port PORT=12000 NAME=tts
stop-llm:
	@$(MAKE) --no-print-directory _stop-port PORT=13000 NAME=llm
stop-app:
	@$(MAKE) --no-print-directory _stop-port PORT=14000 NAME=app

# 내부 헬퍼 (사용자가 직접 부르지 않음). PORT, NAME 변수를 받아 동작.
_stop-port:
	@PID=$$(fuser $(PORT)/tcp 2>/dev/null | tr -d ' '); \
	if [ -z "$$PID" ]; then \
	    echo "[$(NAME)] not running (port $(PORT) is free)"; \
	else \
	    PGID=$$(ps -o pgid= -p $$PID 2>/dev/null | tr -d ' '); \
	    echo "[$(NAME)] killing pgrp $$PGID (port $(PORT), main pid $$PID)"; \
	    kill -9 -$$PGID 2>/dev/null || kill -9 $$PID 2>/dev/null || true; \
	    sleep 2; \
	    if fuser $(PORT)/tcp >/dev/null 2>&1; then \
	        echo "[$(NAME)] WARNING: still alive on port $(PORT)"; \
	    else \
	        echo "[$(NAME)] stopped."; \
	    fi; \
	fi

# ------------------------------------------------------------------------------
# stop-all: 4개 서비스를 한 번에 깔끔히 종료
# ------------------------------------------------------------------------------
# 두 단계로 죽임:
#   1) fuser -k PORT/tcp  → 그 포트를 잡고있는 프로세스(부모 vllm)를 종료
#   2) pkill -9 -f "..."  → 부모를 죽여도 살아남는 worker/EngineCore 자식들 처치
#
# 왜 2단계가 필요한가:
# vLLM은 여러 프로세스로 동작합니다. 메인 API 서버 1개 + 모델 추론 worker
# 여러 개. fuser는 "포트를 점유한 프로세스"만 죽이므로 worker는 살아남고
# GPU 메모리를 계속 잡습니다. 그래서 이름 패턴으로 한 번 더 정리.
stop-all:
	@# 1) 포트 잡고있는 부모 죽이기
	-fuser -k 11000/tcp 12000/tcp 13000/tcp 14000/tcp 2>/dev/null
	@sleep 3
	@# 2) 잔존 vLLM worker / EngineCore 자식 처치
	-pkill -9 -f "VLLM::Worker\|VLLM::EngineCore\|EngineCore_DP" 2>/dev/null
	@sleep 2
	@echo "ports + vllm workers released."

# ------------------------------------------------------------------------------
# health: 4개 endpoint의 HTTP 응답 코드 확인 (200이면 정상)
# ------------------------------------------------------------------------------
# curl은 HTTP 요청을 보내는 도구.
#   -s  silent (진행 막대 숨김)
#   -o /dev/null  응답 본문 버림
#   -w '%{http_code}'  응답 헤더의 상태 코드만 출력 (200/404/000 등)
# $$(...)는 Makefile에서 "이 명령을 실행해서 그 결과를 여기 넣어라"라는 뜻.
# (셸에서는 $(...)지만 Makefile은 $를 변수로 쓰니까 $$로 escape)
health:
	@printf "  STT  :11000 (whisper)    → HTTP %s\n" \
	    $$(curl -s -o /dev/null -w '%{http_code}' http://localhost:11000/v1/models)
	@printf "  TTS  :12000 (qwen3-tts)  → HTTP %s\n" \
	    $$(curl -s -o /dev/null -w '%{http_code}' http://localhost:12000/v1/models)
	@printf "  LLM  :13000 (llama3)     → HTTP %s\n" \
	    $$(curl -s -o /dev/null -w '%{http_code}' http://localhost:13000/v1/models)
	@printf "  APP  :14000 (voice-chat) → HTTP %s\n" \
	    $$(curl -s -o /dev/null -w '%{http_code}' http://localhost:14000/)

# ------------------------------------------------------------------------------
# logs: 각 서비스의 최근 로그 30줄씩 출력 (트러블슈팅용)
# ------------------------------------------------------------------------------
# for ... in ...; do ...; done은 반복문. 여기선 3개 폴더 돌면서 server.log를 tail.
# 백슬래시(\)는 "다음 줄로 명령이 이어진다"는 뜻.
logs:
	@for sub in stt-test tts-test llm-test; do \
	    echo "=== $$sub/server.log (tail) ==="; \
	    tail -30 $$sub/server.log 2>/dev/null || echo "(no log)"; \
	    echo; \
	done
	@echo "=== llm-test/app.log (tail) ==="
	@tail -30 llm-test/app.log 2>/dev/null || echo "(no log)"
