# Qwen3-TTS API 사용 가이드

vllm-omni가 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`를 OpenAI-호환 `/v1/audio/speech`
엔드포인트로 서빙합니다. 이 문서는 **API 호출 방법**만 다룹니다. 환경/설치 함정은
[`README.md`](./README.md) 참고.

- 베이스 URL: `http://<server-host>:12000/v1`
- 모델 ID: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- 인증: 없음 (`api_key="EMPTY"`로 둠)

## 0. 다른 서버에서 호출하려면

vllm-omni는 기본적으로 `0.0.0.0:12000`에 바인딩되어 LAN의 어디서든 호출 가능합니다.
단,

1. **방화벽**: 호스트에서 12000 포트가 열려있는지 확인.
   ```bash
   sudo ufw allow 12000     # Ubuntu
   # 또는 iptables / 보안그룹(클라우드) 룰 추가
   ```

2. **접근 확인** (클라이언트 머신에서):
   ```bash
   curl http://usr039-gpuvm01:12000/v1/models     # 호스트네임으로
   curl http://10.x.y.z:12000/v1/models           # IP로
   ```

3. **운영 시 권장**: 외부 노출 시 nginx/caddy 같은 reverse proxy로 HTTPS + 토큰 인증
   래핑. vllm-omni 자체는 인증 미구현.

이하 예시는 편의상 `localhost`를 쓰지만, 모두 `<server-host>` 치환하면 외부 호출.

## 1. 엔드포인트

| Method | Path | 용도 |
|---|---|---|
| GET | `/v1/models` | 등록된 모델 목록 (헬스체크 겸용) |
| POST | `/v1/audio/speech` | 메인 TTS — 텍스트 → 음성 |
| GET | `/v1/audio/voices` | 사용 가능한 화자 목록 |
| GET | `/health` | 헬스체크 |

## 2. 사용 가능한 화자 (CustomVoice 모델 기준)

```bash
curl http://localhost:12000/v1/audio/voices
```

```json
{"voices":["aiden","dylan","eric","ono_anna","ryan","serena","sohee","uncle_fu","vivian"]}
```

추정 (음향상):

| voice | 추정 화자 |
|---|---|
| `sohee` | 한국어 (한국 화자) ✅ |
| `ono_anna` | 일본어 |
| `uncle_fu` | 중국어 |
| `vivian`, `serena` | 영어 여성 |
| `aiden`, `dylan`, `eric`, `ryan` | 영어 남성 |

> 모든 화자는 모든 언어를 합성 가능 (cross-lingual). 다만 화자의 모국어와 입력 언어를
> 맞추는 게 발음 자연스러움. 한국어는 **`sohee`** 권장.

## 3. 요청 파라미터

`Content-Type: application/json`으로 전송.

### OpenAI 표준

| 필드 | 필수 | 값 / 기본 | 비고 |
|---|---|---|---|
| `input` | ✅ | 합성할 텍스트 (string) | UTF-8 |
| `voice` | ✖ | `"vivian"` | 위 9개 중 하나 |
| `model` | ✖ | 서버 모델 ID | 생략 가능 |
| `response_format` | ✖ | `"wav"` | `wav`/`mp3`/`flac`/`pcm`/`aac`/`opus` |
| `speed` | ✖ | `1.0` | 0.25~4.0 |

### vLLM-Omni 확장

| 필드 | 필수 | 값 / 기본 | 비고 |
|---|---|---|---|
| `language` | ✖ | `"Auto"` | `Auto`/`Korean`/`English`/`Chinese`/`Japanese`/`German`/`French`/`Russian`/`Portuguese`/`Spanish`/`Italian` |
| `task_type` | ✖ | `"CustomVoice"` | CustomVoice 모델은 이거 고정 |
| `instructions` | ✖ | `""` | VoiceDesign 모델에서 음성 스타일 자연어 지시 |
| `max_new_tokens` | ✖ | `2048` | 생성 토큰 상한 |
| `stream` | ✖ | `false` | `true` 시 `response_format="pcm"` 필수 |

## 4. 호출 방법

### 4-1. curl (가장 기본)

```bash
curl -X POST http://localhost:12000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요. 반갑습니다.",
    "voice": "sohee",
    "language": "Korean",
    "response_format": "wav"
  }' \
  --output korean.wav
```

응답: 바이너리 wav (24kHz mono PCM_16). 헤더 `Content-Type: audio/wav`.

### 4-2. Python — OpenAI SDK (권장)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:12000/v1", api_key="EMPTY")

response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    voice="sohee",
    input="안녕하세요. 반갑습니다.",
    response_format="wav",
    extra_body={"language": "Korean"},   # OpenAI 표준 외 인자
)

# stream_to_file는 deprecated. with_streaming_response 권장 (스트리밍 섹션 참고)
response.write_to_file("korean.wav")
```

### 4-3. Python — requests (SDK 없이)

```python
import requests

r = requests.post(
    "http://localhost:12000/v1/audio/speech",
    json={
        "input": "안녕하세요. 반갑습니다.",
        "voice": "sohee",
        "language": "Korean",
        "response_format": "wav",
    },
    timeout=120,
)
r.raise_for_status()
with open("korean.wav", "wb") as f:
    f.write(r.content)
```

### 4-4. 메모리에서 바로 받기 (디스크 안 쓰고)

```python
import io
import requests
import soundfile as sf

r = requests.post(
    "http://localhost:12000/v1/audio/speech",
    json={"input":"안녕","voice":"sohee","language":"Korean","response_format":"wav"},
)
audio, sr = sf.read(io.BytesIO(r.content))
# audio: numpy float32 array, sr=24000
```

### 4-5. 동시/배치 요청 (asyncio)

```python
import asyncio
import aiohttp

async def synth(session, text, voice="sohee", language="Korean"):
    async with session.post(
        "http://localhost:12000/v1/audio/speech",
        json={"input": text, "voice": voice, "language": language, "response_format": "wav"},
    ) as r:
        return await r.read()

async def main():
    texts = ["첫 번째 문장.", "두 번째 문장.", "세 번째 문장."]
    async with aiohttp.ClientSession() as s:
        results = await asyncio.gather(*(synth(s, t) for t in texts))
    for i, buf in enumerate(results):
        with open(f"out_{i}.wav", "wb") as f:
            f.write(buf)

asyncio.run(main())
```

vllm-omni가 자체적으로 continuous batching을 하므로 클라이언트는 동시 요청만 보내면 됨.

### 4-6. 스트리밍 (97ms first-packet latency 검증용)

```python
import requests, time

t0 = time.time()
first_chunk_time = None

with requests.post(
    "http://localhost:12000/v1/audio/speech",
    json={
        "input": "스트리밍 모드 테스트입니다.",
        "voice": "sohee",
        "language": "Korean",
        "response_format": "pcm",   # 스트리밍은 PCM 필수
        "stream": True,
    },
    stream=True,
) as r:
    with open("stream_out.pcm", "wb") as f:
        for chunk in r.iter_content(chunk_size=4096):
            if first_chunk_time is None:
                first_chunk_time = time.time() - t0
                print(f"first packet latency: {first_chunk_time*1000:.0f}ms")
            f.write(chunk)

# PCM → wav 변환
import subprocess
subprocess.run([
    "ffmpeg","-y","-f","s16le","-ar","24000","-ac","1",
    "-i","stream_out.pcm","stream_out.wav"
], check=True)
```

> ⚠️ 우리 실측은 아직 비-스트리밍 모드만 — 0.85× realtime. 스트리밍 모드 first-packet
> latency가 진짜 97ms인지는 위 코드로 직접 검증 필요.

## 5. 응답 형식

### `response_format="wav"` (기본)

- Content-Type: `audio/wav`
- 24kHz mono, 16-bit PCM
- Body: 바이너리 wav 파일

### `response_format="pcm"` (스트리밍용)

- Content-Type: `audio/L16` 또는 유사
- 24kHz mono, 16-bit signed little-endian raw PCM (헤더 없음)
- Body: 청크 단위 raw PCM

### `response_format="mp3"` / `flac` / `aac` / `opus`

서버가 즉석 인코딩. 네트워크 절감 시 유용. mp3/aac는 약간의 음질 손실 있음.

## 6. 트러블슈팅 / 함정

| 증상 | 원인 | 조치 |
|---|---|---|
| `Connection refused` | 서버 안 떠있거나 포트 닫힘 | `curl http://<host>:12000/v1/models` 로 확인, 방화벽 점검 |
| `400 BadRequest: Unknown voice` | voice 오타 | `GET /v1/audio/voices`로 정확한 이름 확인 |
| 결과가 영어 톤으로 한국어 발음 | `language` 미지정 → Auto → 영어로 판단 | `language="Korean"` 명시 |
| `400: ... task_type ...` | CustomVoice 모델에 VoiceDesign/Base task | 모델 변형이 맞는지 확인 (이 셋업은 CustomVoice 전용) |
| 합성 결과가 잘림 | 텍스트가 max_new_tokens(2048) 한계 초과 | 긴 텍스트는 문장 단위로 분할 호출 |
| 응답이 너무 오래 걸림 (RTF ~0.85) | 비-스트리밍 + enforce_eager | 스트리밍 모드 (`pcm` + `stream:true`)로 first-packet만 빨리 받기 |
| 외부에서 접근 안 됨 | host 바인딩 또는 방화벽 | start_server.sh의 `--host 0.0.0.0` 확인 + 보안그룹 |

## 7. 빠른 검증 스크립트

```bash
unset LD_LIBRARY_PATH
VLLM_URL=http://<server-host>:12000/v1 uv run python test_server.py
```

`test_server.py`는 `/v1/models` + 영문 + 한글 + OpenAI SDK 4가지 경로를 한 번에 검증.
결과 wav는 `samples/`에 떨어집니다.

## 8. 다른 모델 변형으로 전환

같은 셋업에서 `start_server.sh`의 `MODEL` 변수만 교체:

```bash
# 음성 자연어 지시 (예: "20대 차분한 여성")
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
# → API: task_type="VoiceDesign", instructions="..."

# 본인 목소리 복제 (3-10초 ref audio 필요)
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# → API: task_type="Base", ref_audio="<base64 or url>", ref_text="..."

# 경량 (메모리 4GB+ 환경)
MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
```

모델 변경 후 재기동: `pkill -f Qwen3-TTS && ./start_server.sh`.
