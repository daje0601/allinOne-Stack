# vllm-omni × Qwen3-TTS 테스트

H100 80GB · CUDA driver 12.2 환경에서 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`를
vllm-omni의 OpenAI-호환 `/v1/audio/speech` 엔드포인트로 검증.

## 환경

- GPU: NVIDIA H100 80GB ×8 (테스트는 GPU 2)
- Driver: 535.183.06 (CUDA 12.2 native, 12.6/12.8 minor-version compat)
- Python: 3.12.12 (uv venv)
- vLLM: **0.16.0** (2026-02-26 PyPI, torch 2.9.1)
- vllm-omni: **0.16.0** (2026-03-01 PyPI — Qwen3-TTS day-0 지원)
- torch: **2.9.1+cu128** (cu126 요청했지만 cu128로 충족, 둘 다 driver 535 호환)

## 셋업 (uv 단일 명령)

```bash
cd voice-stack/tts-test
UV_TORCH_BACKEND=cu126 uv sync
```

`pyproject.toml`에 핀이 박혀있어 `uv sync` 한 번이면 끝.

## 셋업에서 부딪혔던 함정 4가지

| # | 함정 | 해결 |
|---|---|---|
| 1 | **vLLM 0.21.0은 torch 2.11+cu130을 가져옴 → driver 12.2 incompatible** | vllm-omni가 Qwen3-TTS 지원 시작한 가장 오래된 페어 = vllm 0.16.0 / vllm-omni 0.16.0 (torch 2.9.1) 사용. cu126 wheel이 driver 535에서 작동. |
| 2 | **LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64**가 venv 라이브러리 가림 | 실행 시 `unset LD_LIBRARY_PATH` (start_server.sh에 포함). |
| 3 | **`--omni` 플래그 위치** | `vllm-omni --omni serve <model>` (X) — argparse 거부. `vllm-omni serve <model> --omni` (O) — serve 서브파서가 등록함. |
| 4 | **`trust_remote_code`만으로는 `qwen3_tts` model_type 미인식** (transformers 5.9.0까지도 release X) | vllm-omni의 자체 등록 메커니즘 활용: `vllm-omni serve --omni`로 호출하면 `register_omni_models_to_vllm()`이 `AutoConfig.register("qwen3_tts", Qwen3TTSConfig)` 실행. |

## 1) 서버 띄우기

```bash
./start_server.sh
```

내부 동작:
1. GPU 2에서 백그라운드 기동 (Stage-0 Talker + Stage-1 Code2Wav 2-stage 파이프라인)
2. `/v1/models` 폴링 (모델 로드 + KV cache + 2-stage init = ~2-3분)
3. 워밍업 POST `/v1/audio/speech` 1회 → `samples/tts-warmup.wav`
4. foreground attach (Ctrl+C로 정지, trap이 vllm 정리)

엔드포인트:

```
POST /v1/audio/speech         ← 메인 TTS
GET  /v1/audio/voices         ← 보이스 목록
GET  /v1/models               ← 헬스체크용
```

## 2) 클라이언트 테스트

```bash
unset LD_LIBRARY_PATH
uv run python test_server.py
```

curl 한 줄:

```bash
curl -X POST http://localhost:12000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"안녕하세요","voice":"vivian","language":"Korean","response_format":"wav"}' \
  --output out.wav
```

## 3) 실측 (1.7B-CustomVoice, GPU 2 단일, 비-스트리밍)

| 입력 | wall(s) | audio(s) | RTF (audio/wall) |
|---|---|---|---|
| `"Hello, warmup."` | 7.19 | 1.04 | 0.14× (콜드 포함) |
| `"Hello world. This is Qwen 3 TTS..."` | 10.49 | 8.88 | **0.85× realtime** |
| `"안녕하세요. 큐웬 쓰리 티티에스가..."` | 6.58 | 5.60 | **0.85× realtime** |
| OpenAI SDK 영문 | 9.10 | 5.04 | 0.55× |

**해석**: 비-스트리밍 모드에서는 실시간보다 살짝 느림. 단,
- Qwen3-TTS의 광고된 **97ms latency는 "first-packet" 스트리밍 지표** — 전체 합성 시간이 아님. 스트리밍 모드(`response_format=pcm`, `stream=true`)에서 측정해야 그 숫자가 나옴.
- Stage-1 (Code2Wav)이 `enforce_eager: true`로 CUDA graph 비활성. FlashAttn2 설치 시 가속 가능.

## 4) 알아둘 점

- **언어 명시 필수**: `language` 인자에 `Korean`, `English`, `Chinese`, `Japanese`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian` 중 하나. 생략하면 `Auto`이지만 cross-language 환각 위험.
- **voice 인자**: CustomVoice 모델은 preset 보이스 (`vivian`, `ryan`, `aiden`...). 전체 목록은 `GET /v1/audio/voices`.
- **task_type**: `CustomVoice` (default, preset speaker) / `VoiceDesign` (natural-language 지시) / `Base` (reference audio voice cloning). 모델 변형마다 지원 task가 다름 — 본 셋업은 CustomVoice 모델이라 그것만 권장.
- **output 포맷**: `wav` (default), `mp3`, `flac`, `pcm`, `aac`, `opus`. 24kHz mono 출력.
- **2-stage 파이프라인**: 단일 GPU에서도 동작하지만 stage 간 ZMQ + shared memory 통신 오버헤드가 있음. multi-GPU split도 가능 (yaml의 `devices: "0,1"` 등).

## 5) 산출물

| 파일 | 역할 |
|---|---|
| `pyproject.toml` | vllm 0.16 + vllm-omni 0.16 + 의존성 핀 |
| `uv.lock` | 재현 가능한 잠금파일 |
| `start_server.sh` | 서버 기동 + 워밍업 자동 |
| `test_server.py` | requests + OpenAI SDK 양쪽 검증 |
| `server.log` | vllm-omni 로그 |
| `samples/*.wav` | 실측 결과물 (start_server 워밍업 + test_server.py + 보이스 비교) |

## 6) 차후 (스트리밍 활성화하면 진짜 97ms?)

```python
# 클라이언트 (스트리밍)
import requests
r = requests.post(
    "http://localhost:12000/v1/audio/speech",
    json={"input":"...","voice":"vivian","language":"Korean",
          "response_format":"pcm","stream":True},
    stream=True,
)
for chunk in r.iter_content(chunk_size=4096):
    # 첫 chunk까지의 시간을 측정 → 97ms 검증
    play(chunk)
```

추가 검증 대상.
