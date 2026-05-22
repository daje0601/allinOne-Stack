# llm-test — 곰돌이 푸 페르소나 RAG 챗봇

`5. serving.ipynb`를 풀어서 vLLM 서버 + Gradio + Chroma RAG 형태로 재구성한 프로젝트.

## 구성

```
llm-test/
├─ pyproject.toml         # vllm 0.16 + langchain + chromadb + gradio
├─ uv.lock                # 잠금
├─ .env.example           # OPENAI_API_KEY 채우는 템플릿
├─ data/
│   ├─ characters.json    # 10개 (곰돌이푸/피글렛/이요르 등 캐릭터 소개)
│   ├─ episodes.json      # 100개 에피소드
│   ├─ episodes2.json     # 추가 100개 에피소드
│   └─ system_prompt.txt  # 푸 페르소나 시스템프롬프트
├─ ingest.py              # Chroma 벡터DB 구축 (1회 실행)
├─ start_server.sh        # vLLM 서버 (LLaMA3 8B persona, port 13000)
└─ app.py                 # Gradio UI + RAG (port 7860)
```

## 셋업

```bash
cd voice-stack/llm-test
cp .env.example .env
# .env 열어서 OPENAI_API_KEY=sk-... 채우기

UV_TORCH_BACKEND=cu126 uv sync
```

## 1) 벡터 DB 구축 (최초 1회)

```bash
unset LD_LIBRARY_PATH
uv run python ingest.py
# → ./chroma_db/  (210 docs 임베딩됨, OpenAI text-embedding-3-small)
```

## 2) vLLM 서버 띄우기

```bash
./start_server.sh
# → http://0.0.0.0:13000/v1
# 모델 다운로드(~16GB) + 로드 + CUDA graph capture까지 ~2-3분
```

GPU 3 사용 (다른 STT/TTS 서버와 충돌 없음). 8B bf16 ≈ 16GB 가중치 + KV cache.

## 3) Gradio UI 실행

```bash
unset LD_LIBRARY_PATH
uv run python app.py
# → http://<server-host>:14000
```

브라우저에서 푸에게 한국어로 질문하면:
1. 질문이 임베딩됨
2. Chroma에서 top-5 관련 에피소드 검색
3. 시스템프롬프트 + 대화 이력 + 사용자 질문 + `<context>...</context>` 검색 결과를 LLaMA3에 전달
4. 응답 스트리밍 표시

## 환경

- GPU: H100 80GB ×8 (사용 = GPU 3)
- Driver: 535.183.06 (CUDA 12.2 native)
- Python: 3.12.12 (uv venv)
- vLLM: **0.16.0** + torch **2.9.1+cu128** (tts-test와 동일 스택)
- LangChain: 0.3+

## 비고

- **OpenAI 임베딩 사용**: `OPENAI_API_KEY` 필수. 210 docs × ~50 토큰 ≈ embedding 비용 거의 0원.
- **모델**: `iamjoon/llama3-8b-persona-chatbot` — LLaMA3 8B를 한국어 페르소나 챗봇으로 fine-tune한 HF 공개 모델.
- **메모리**: 8B bf16 + max_model_len 8192 + KV cache 0.85 활용 → ~65GiB. 여유 충분.
- **RAG 검색 안 쓸 거면**: app.py의 `chat_fn`에서 `get_formatted_context` 호출 제거하면 일반 챗봇처럼 작동.

## 트러블슈팅

| 증상 | 조치 |
|---|---|
| `OPENAI_API_KEY not set` | `.env`에 키 채우고 ingest.py / app.py 둘 다 재실행 |
| ingest.py 빠른 실패 | langchain-openai 버전 / API 키 권한 확인 |
| vllm 서버 시작 시 OOM | `start_server.sh`의 `--gpu-memory-utilization`을 0.7로 낮춤 |
| Gradio 7860 접속 안 됨 | 방화벽 확인 / `server_name="0.0.0.0"` 인지 / 보안그룹 |
| 응답이 영어로 나옴 | system_prompt가 손상됐는지 확인. 한국어 prompt 그대로 들어가야 함 |
