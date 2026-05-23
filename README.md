# voice-stack

음성·LLM 3종 마이크로서비스 모노레포. 각 서브 프로젝트는 독립된 venv를 가지며,
서로 OpenAI-호환 HTTP API로 느슨하게 결합됩니다.

```
voice-stack/
├─ stt-test/   # vLLM × Whisper-large-v3-turbo (STT, port 11000)
├─ tts-test/   # vllm-omni × Qwen3-TTS (TTS, port 12000)
└─ llm-test/   # vLLM × LLaMA3-8B 페르소나 챗봇 + RAG (port 13000) + Gradio UI (14000)
```

## 서비스 매트릭스

| 서비스 | 포트 | 모델 | GPU | 메모리 | 엔드포인트 |
|---|---|---|---|---|---|
| **STT** | `11000` | `openai/whisper-large-v3-turbo` | 0 | ~26GB | `POST /v1/audio/transcriptions` |
| **TTS** | `12000` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 2 | ~25GB | `POST /v1/audio/speech` |
| **LLM** | `13000` | `iamjoon/llama3-8b-persona-chatbot` | 3 | ~50GB | `POST /v1/chat/completions` |
| **Chat UI** | `14000` | (RAG, LLM 백엔드 호출) | – | – | `http://<host>:14000` |

모두 OpenAI-호환 API → `from openai import OpenAI; OpenAI(base_url=...)` 그대로 사용 가능.

## 환경 (공통 함정)

- GPU: H100 80GB × 2
- Python: 3.12.12 (uv venv 각자)
- 실행 시 셸 환경의 `LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64`가 venv NVIDIA 라이브러리를
  가리므로 모든 `start_server.sh`에 `unset LD_LIBRARY_PATH` 박혀있음.

## Quick start

### 0. 셋업 (각 서브 프로젝트 한 번씩)

```bash
cd voice-stack
make sync-all
```

```bash
cd voice-stack/stt-test  && uv sync
cd voice-stack/tts-test  && uv sync
cd voice-stack/llm-test  && uv sync
```

`llm-test`는 추가로:
```bash
apt-get update && apt-get install -y vim
cd voice-stack/llm-test
vi .env       # OPENAI_API_KEY 채우기
uv run python ingest.py    # Chroma 벡터DB 1회 빌드
```

### 1. 서버 띄우기 (각각)

```bash
make start-stt    # whisper :11000 (GPU 0)
make start-tts    # qwen3-tts :12000 (GPU 2)
make start-llm    # llama3 :13000 (GPU 3)
make start-app    # gradio :14000 (LLM 백엔드 호출)
```

또는 직접:
```bash
./stt-test/start_server.sh
./tts-test/start_server.sh
./llm-test/start_server.sh
cd llm-test && uv run python app.py
```

### 2. 검증

```bash
make health       # 4개 endpoint 다 200인지 확인
```

### 3. 정지

```bash
make stop-all
```

## 디렉토리별 상세

각 서브폴더의 `README.md` + `API_USAGE.md` 참고:

- [stt-test/README.md](./stt-test/README.md) — 셋업 함정 3가지 (vllm 0.7.3, transformers<5, LD_LIBRARY_PATH)
- [stt-test/API_USAGE.md](./stt-test/API_USAGE.md) — STT 호출 가이드
- [tts-test/README.md](./tts-test/README.md) — 셋업 함정 4가지 (vllm-omni `--omni` 플래그 위치 등)
- [tts-test/API_USAGE.md](./tts-test/API_USAGE.md) — TTS 호출 가이드 (9개 화자 포함)
- [llm-test/README.md](./llm-test/README.md) — RAG 파이프라인 + 벡터DB 빌드
- [llm-test/API_USAGE.md](./llm-test/API_USAGE.md) — chat completions 호출 + 스트리밍

## 사용된 vLLM 버전 (의도적으로 다름)

| 서비스 | vllm | 이유 |
|---|---|---|
| stt-test | 0.7.3 | Whisper STT는 0.7.3에 정착, transformers 4.x 페어링 필요 |
| tts-test | 0.16.0 (+ vllm-omni 0.16.0) | Qwen3-TTS day-0 지원 vllm-omni의 가장 안정적 페어 |
| llm-test | 0.16.0 | tts와 동일 — LLaMA3 8B는 vllm 0.7+에서 어디든 OK |

세 서비스가 같은 venv를 공유하지 못하는 근본 원인이 vllm 0.7.3 vs 0.16.0 버전 충돌.
모노레포 안에서도 각자 venv를 두는 이유.

## 라이선스 / 출처

- Whisper: OpenAI Apache-2.0
- Qwen3-TTS: Alibaba Cloud (Qwen license, 일부 제약)
- iamjoon/llama3-8b-persona-chatbot: 원본 LLaMA3 Meta 라이선스 + 파인튜닝 작가의 HF 모델카드 참고
