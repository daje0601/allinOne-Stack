# STT API 사용 가이드

vLLM이 `openai/whisper-large-v3-turbo`를 OpenAI-호환 형식으로 서빙합니다.
이 문서는 **API 사용법**에만 집중합니다. 환경/설치 함정은 [`README.md`](./README.md) 참고.

- 베이스 URL: `http://localhost:11000/v1`
- 모델 ID: `openai/whisper-large-v3-turbo`
- 인증: 없음 (`api_key="EMPTY"`로 둠)

## 0. 서버 띄우기

```bash
cd voice-stack/stt-test
./start_server.sh   # foreground로 띄움. 백그라운드 원하면 `./start_server.sh > server.log 2>&1 &`
```

서버 준비 확인:

```bash
curl http://localhost:11000/v1/models
```

응답:

```json
{"object":"list","data":[{"id":"openai/whisper-large-v3-turbo","object":"model","max_model_len":448,...}]}
```

## 1. 엔드포인트 한눈에

| Method | Path | 용도 |
|---|---|---|
| GET | `/v1/models` | 등록된 모델 목록 |
| POST | `/v1/audio/transcriptions` | 음성 → 동일 언어 텍스트 (STT) |
| GET | `/health` | 헬스체크 |

> `/v1/audio/translations`는 **vLLM 0.7.3에서 미라우팅** (`404 Not Found`).
> 한→영 등 번역이 필요하면 transcribe 후 별도 번역 모델로 보내거나 더 최신 vLLM 필요.

## 2. 요청 파라미터

`multipart/form-data` 로 전송.

| 필드 | 필수 | 값 / 기본 | 비고 |
|---|---|---|---|
| `file` | ✅ | wav / mp3 / m4a / flac / ogg / webm | Whisper 자체는 99개 언어 |
| `model` | ✅ | `openai/whisper-large-v3-turbo` | `/v1/models`에서 본 ID |
| `language` | ✖ | ISO 639-1 (`en`, `ko`, `ja`, `zh`, ...) | **생략 시 영어로 폴백** (자동 감지 미지원) |
| `response_format` | ✖ | `json` (기본) / `text` | ⚠️ `verbose_json`/`srt`/`vtt` 미지원 (400) |
| `temperature` | ✖ | `0.0` (기본) | 결정론적 디코딩 |
| `prompt` | ✖ | "" | 도메인/이름/전문용어 힌트 (≤224토큰) |

## 3. 호출 방법

### 3-1. curl (가장 기본)

```bash
curl -X POST http://localhost:11000/v1/audio/transcriptions \
  -F "file=@samples/stt-warmup.wav" \
  -F "model=openai/whisper-large-v3-turbo" \
  -F "language=en" \
  -F "response_format=json"
```

응답:

```json
{"text": " And the 0-1 pitch on the way to Edgar Martinez. ..."}
```

### 3-2. Python — OpenAI SDK (권장)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11000/v1", api_key="EMPTY")

with open("/path/to/audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=f,
        language="ko",          # 한국어면 "ko", 영어면 "en"
        response_format="json", # 또는 "text"
        temperature=0.0,
    )

print(result.text)
```

### 3-3. Python — requests (SDK 없이)

```python
import requests

with open("/path/to/audio.wav", "rb") as f:
    r = requests.post(
        "http://localhost:11000/v1/audio/transcriptions",
        files={"file": ("audio.wav", f, "audio/wav")},
        data={
            "model": "openai/whisper-large-v3-turbo",
            "language": "ko",
            "response_format": "json",
        },
        timeout=60,
    )
r.raise_for_status()
print(r.json()["text"])
```

### 3-4. 한국어 STT 실측 예

```bash
# UTF-8 한글 파일명은 셸 변수 전달 시 깨질 수 있으니 ASCII 경로 권장
# (samples/kr_07_jambatchim.wav가 미리 ASCII 이름으로 준비돼있음)

curl -X POST http://localhost:11000/v1/audio/transcriptions \
  -F "file=@samples/kr_07_jambatchim.wav" \
  -F "model=openai/whisper-large-v3-turbo" \
  -F "language=ko" \
  -F "response_format=json"
```

응답 예 (1.98초 한국어 음성):

```json
{"text": " 값이 없어서 삿도 못 받았어요."}
```

⚠️ `language=ko` **반드시 명시**. 생략하면 영어 폴백되어 `"I'm not paying for it."`
같이 엉뚱한 영어 문장을 만들어냄.

## 4. 메모리에서 바로 보내기 (디스크 거치지 않고)

녹음 직후·합성 직후 바이트를 그대로 업로드:

```python
import io
import soundfile as sf
from openai import OpenAI

# 예: numpy array → wav 바이트
buf = io.BytesIO()
sf.write(buf, audio_np, 16000, format="WAV", subtype="PCM_16")
buf.seek(0)

client = OpenAI(base_url="http://localhost:11000/v1", api_key="EMPTY")
result = client.audio.transcriptions.create(
    model="openai/whisper-large-v3-turbo",
    file=("recording.wav", buf, "audio/wav"),
    language="ko",
)
print(result.text)
```

## 5. 동시/배치 요청

vLLM은 자체적으로 continuous batching을 합니다. 클라이언트는 동시 요청만 보내면 됨.

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:11000/v1", api_key="EMPTY")

async def transcribe(path):
    with open(path, "rb") as f:
        r = await client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=f,
            language="ko",
        )
    return r.text

async def main():
    paths = ["a.wav", "b.wav", "c.wav", "d.wav"]
    results = await asyncio.gather(*(transcribe(p) for p in paths))
    for p, t in zip(paths, results):
        print(p, "→", t)

asyncio.run(main())
```

서버 측 동시성 상한은 `--max-num-seqs`로 조절 (현재 `start_server.sh`에서 `32`).

## 6. 오디오 형식 / 길이 제한

- **포맷**: ffmpeg가 디코드할 수 있는 거면 모두 OK (wav/mp3/m4a/flac/ogg/webm/...)
- **샘플링레이트**: 자동 16kHz로 리샘플
- **길이 한 청크**: Whisper는 **30초 단위로 처리**. 30초 넘는 파일도 받지만,
  vLLM 0.7.3의 단발 요청은 첫 30초만 전사함.
- **긴 오디오**: 클라이언트에서 30초 청크 분할 후 순차 호출 권장
  (vLLM 자동 chunking은 PR #34628 진행 중, 0.7.3 미포함).

긴 오디오 청크 분할 예:

```python
import soundfile as sf
import numpy as np

def chunk_30s(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # 모노로
    chunk_samples = 30 * sr
    for i in range(0, len(audio), chunk_samples):
        yield audio[i:i + chunk_samples], sr
```

## 7. 응답 형식

### `response_format=json` (기본)

```json
{"text": " 안녕하세요. 오늘 날씨가 좋네요."}
```

### `response_format=text`

```
 안녕하세요. 오늘 날씨가 좋네요.
```

> `verbose_json` (타임스탬프 포함), `srt`, `vtt`는 vLLM 0.7.3에서 미지원.
> 요청 시 `400 BadRequest: "Currently only support response_format text or json"`.

## 8. 알려진 에러 / 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| `Connection refused` | 서버 안 떠있음 / 포트 다름 | `curl http://localhost:11000/v1/models` 로 확인 |
| `400 Currently only support text or json` | `response_format=verbose_json` 등 사용 | `json` 또는 `text`로 변경 |
| `No available memory for the cache blocks` | `gpu_memory_utilization` 낮은데 `max_num_seqs` 큼 | `max_num_seqs` 줄이거나 GPU 메모리 비율 올리기 |
| 결과가 영어로 나옴 (입력은 한국어) | `language` 미지정 → 영어 폴백 | `language="ko"` 명시 |
| 30초 이후가 잘림 | 한 요청은 첫 30초만 처리 | 클라이언트에서 30초 청크 분할 |
| 응답에 "Mmm, mm-hmm" 같은 환각 | 무음/노이즈 구간 | `prompt` 힌트 주거나 VAD로 사전 컷 |
| `/v1/audio/translations` 호출 시 `404 Not Found` | vLLM 0.7.3 미라우팅 | transcribe 후 별도 번역 모델로 보냄 |
| 한글 파일명에 curl이 `(26) Failed to open` | 셸 UTF-8 변수 전달 이슈 | ASCII 이름으로 복사 (예: `samples/kr_07_jambatchim.wav`) |

## 9. 빠른 검증 스크립트

이미 있는 [`test_server.py`](./test_server.py)를 다른 포트로 재사용:

```bash
VLLM_URL=http://localhost:11000/v1 uv run python test_server.py
```

세 가지 호출(`/v1/models`, requests, OpenAI SDK)을 한 번에 검증.
