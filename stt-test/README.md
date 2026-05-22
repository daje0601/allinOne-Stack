# vLLM × whisper-large-v3-turbo 테스트

H100 80GB · CUDA driver 12.2 환경에서 `openai/whisper-large-v3-turbo`를 vLLM으로
1) **오프라인 추론**과 2) **OpenAI-호환 서버**로 검증한 결과.

## 환경

- GPU: NVIDIA H100 80GB ×8 (테스트는 GPU 0 단일 사용)
- Driver: 535.183.06 (CUDA 12.2)
- Python: 3.12.12 (uv venv)
- vLLM: **0.7.3** (cu12.4 torch 2.5.1 wheel — driver 12.2와 minor-version compat)
- transformers: **4.57.6** (5.x는 vLLM 0.7.3과 비호환)

## 셋업 (uv 단일 명령)

```bash
cd voice-stack/stt-test
uv sync           # pyproject.toml + uv.lock → .venv 자동 생성
```

이후 모든 실행은 `uv run …` 형태 (venv 활성화 불필요).

핀 셋은 `pyproject.toml`에 박혀있음:
- `vllm==0.7.3` — cu12.4 torch 2.5.1 휠. driver 12.2 minor-version compat.
- `transformers<5` — vLLM 0.7.3은 `tokenizer.all_special_tokens_extended`(transformers 5에서 제거) 사용.
- `[tool.uv].override-dependencies`로 torch 2.5.1/torchvision 0.20.1/torchaudio 2.5.1 고정.

### 셋업 함정 1가지 (pyproject로 못 막는 것)

**LD_LIBRARY_PATH 충돌**: 셸 환경의 `/usr/local/cuda-12.2/lib64`가 venv의
`libnvJitLink.so.12`(12.4)보다 우선이라 `__nvJitLinkComplete_12_4` symbol
not found 발생. → 실행 시 `unset LD_LIBRARY_PATH` (`start_server.sh`에 포함됨,
스크립트 밖에서 `uv run` 직접 호출 시에도 필요).

## 1) 오프라인 추론 — `test_offline.py`

```bash
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 uv run python test_offline.py
```

**결과** (8 prompts, 4×{Mary had lamb, Edgar Martinez winning call}):

- 모델 로드 + 프로파일 + CUDA graph: ~100s 1회성
- 추론: **1.66s · 4.81 RPS** (max_num_seqs=400, KV cache fp8, 53.46GiB KV cache)
- 두 샘플 모두 정확히 전사됨

```
[0] The first words I spoke in the original phonograph, a little piece of practical
    poetry. Mary had a little lamb, ...
[1] And the 0-1 pitch on the way to Edgar Martinez. Swung on the line down the left
    field line for a base hit. ...
```

## 2) OpenAI-호환 서버 — `start_server.sh` + `test_server.py`

서버 기동:

```bash
./start_server.sh   # 백그라운드/foreground 자유. 11000 포트.
```

클라이언트 테스트:

```bash
unset LD_LIBRARY_PATH
uv run python test_server.py
```

엔드포인트:

- `GET /v1/models` — 등록된 모델 목록
- `POST /v1/audio/transcriptions` — OpenAI 호환. multipart/form-data로 `file`, `model`,
  `language`, `response_format` 전달
- (참고) `POST /v1/audio/translations` — 비영어→영어 번역

순수 curl 예:

```bash
curl -s -X POST http://localhost:11000/v1/audio/transcriptions \
  -F "file=@samples/stt-warmup.wav" \
  -F "model=openai/whisper-large-v3-turbo" \
  -F "language=en" \
  -F "response_format=json"
```

**실측 응답 시간** (단일 요청, ~12초 오디오):
- `requests` 직접: 2.54s (Mary had lamb)
- OpenAI SDK: 3.62s (winning call)
- curl: ~1s
- `response_format=verbose_json` → **400 BadRequest** ("Currently only support `text` or `json`")
  — 0.7.3 한계, PR #24209 이후 버전 필요

## vLLM 서빙 옵션

`start_server.sh` 안의 옵션 의미:

| 옵션 | 값 | 이유 |
|---|---|---|
| `--max-model-len` | 448 | Whisper 디코더 컨텍스트 한계 |
| `--max-num-seqs` | 400 | 동시 배치 (오프라인 코드와 동일) |
| `--kv-cache-dtype` | fp8 | 메모리 절감, turbo는 디코더 4층뿐이라 안전 |
| `--gpu-memory-utilization` | 0.5 | 동일 노드 다른 작업과 공유 가능하게 |

## 서버 정지

```bash
# 백그라운드로 띄운 경우
pkill -f "vllm serve" || true
```

## 알려진 한계 (조사 결과 + 실측)

- **언어 자동 감지 불가** (PR #34342 진행 중) — 비영어는 `language="ko"` 등 명시 필요.
- **타임스탬프 / verbose_json**: PR #24209 부분 지원.
- **V1 엔진 회귀** (#24946): 0.10.2 기준 turbo가 v0 대비 5~6배 느렸음.
  vLLM 0.7.3은 V0 엔진 사용이므로 영향 없음.
