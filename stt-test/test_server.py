# SPDX-License-Identifier: Apache-2.0
"""
============================================================================
Whisper STT 서버 (OpenAI 호환 HTTP API) 클라이언트 테스트
============================================================================

이 스크립트는 이미 떠있는 서버(`./start_server.sh`로 띄운)를 향해
HTTP 요청을 보내서 정상 동작하는지 확인합니다.

세 가지 호출 방식을 모두 검증:
  1. GET /v1/models           — 어떤 모델이 등록돼있는지 목록 받기
  2. POST /v1/audio/transcriptions (requests 라이브러리로 직접)
  3. POST /v1/audio/transcriptions (OpenAI SDK 통해)

비개발자용 용어:
  - HTTP GET    : "정보를 달라"는 요청 (서버 상태 안 바뀜)
  - HTTP POST   : "이 데이터로 작업 좀 해줘"라는 요청 (파일 업로드 등)
  - multipart/form-data : 파일을 HTTP로 올릴 때 쓰는 표준 인코딩
  - SDK         : Software Development Kit. 누군가 만들어둔 편한 라이브러리.

실행:
  unset LD_LIBRARY_PATH
  uv run python test_server.py
  # 다른 서버에 연결하려면:
  VLLM_URL=http://other-host:11000/v1 uv run python test_server.py
============================================================================
"""
import io                                      # 메모리 안의 가짜 파일 만들기용
import os                                      # 환경변수 읽기용
import time                                    # 시간 측정용

import requests                                # HTTP 클라이언트 라이브러리 (간편)
import soundfile as sf                         # wav 파일 읽기/쓰기 라이브러리
from openai import OpenAI                      # OpenAI 공식 Python SDK
from vllm.assets.audio import AudioAsset       # 테스트용 번들 오디오


# ----------------------------------------------------------------------------
# 설정 — 환경변수가 있으면 그 값, 없으면 기본값
# ----------------------------------------------------------------------------
# os.environ.get("KEY", default): 환경변수 KEY가 정의돼있으면 그 값, 아니면 default
BASE_URL = os.environ.get("VLLM_URL", "http://localhost:11000/v1")
MODEL = "openai/whisper-large-v3-turbo"


# ----------------------------------------------------------------------------
# 헬퍼: 번들 오디오를 wav 바이트로 변환
# ----------------------------------------------------------------------------
def fetch_audio_bytes(name: str) -> tuple[bytes, str]:
    """vLLM 내장 테스트 오디오를 받아 wav 바이트로 변환.

    name 예시: "mary_had_lamb" 또는 "winning_call"
    반환: (wav바이트, "이름.wav" 파일이름 추천)
    """
    audio, sr = AudioAsset(name).audio_and_sample_rate  # (오디오 array, 샘플레이트)
    buf = io.BytesIO()                          # 메모리 안의 "가짜 파일" 객체
    # 오디오를 16비트 PCM wav 형식으로 buf에 씀
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)                                 # 읽기 위해 커서를 맨 앞으로
    return buf.read(), f"{name}.wav"


# ----------------------------------------------------------------------------
# 테스트 1: 서버에 어떤 모델이 등록돼있는지 조회
# ----------------------------------------------------------------------------
def test_list_models() -> None:
    # GET 요청 — 서버에 "/v1/models 정보 줘"
    r = requests.get(f"{BASE_URL}/models", timeout=10)
    r.raise_for_status()                        # 응답이 4xx/5xx면 예외 발생
    print("[/v1/models]")
    # 응답 본문은 JSON 형식이고 data 필드에 모델 목록이 들어있음
    for m in r.json().get("data", []):
        print(f"  - {m['id']}")


# ----------------------------------------------------------------------------
# 테스트 2: requests 라이브러리로 직접 multipart 업로드 (curl과 동일한 방식)
# ----------------------------------------------------------------------------
def test_transcription_curl_like() -> None:
    """multipart/form-data로 wav 파일 업로드 — curl -F와 같은 형식."""
    audio_bytes, fname = fetch_audio_bytes("mary_had_lamb")

    # files = 파일 첨부, data = 일반 폼 필드
    files = {"file": (fname, audio_bytes, "audio/wav")}
    data = {"model": MODEL, "language": "en", "response_format": "json"}

    t0 = time.time()
    r = requests.post(f"{BASE_URL}/audio/transcriptions",
                      files=files, data=data, timeout=60)
    dt = time.time() - t0                       # 응답까지 걸린 시간

    r.raise_for_status()
    print(f"[transcription:requests] ({dt:.2f}s)")
    print(f"  {r.json()}")                      # 응답 JSON 통째로 출력


# ----------------------------------------------------------------------------
# 테스트 3: OpenAI 공식 SDK 사용 (다른 OpenAI 서비스 코드 그대로 재활용 가능)
# ----------------------------------------------------------------------------
def test_transcription_openai_sdk() -> None:
    """OpenAI SDK — base_url만 우리 서버로 바꿔주면 동일 코드 사용 가능."""
    audio_bytes, fname = fetch_audio_bytes("winning_call")

    # OpenAI SDK 클라이언트. api_key는 형식상 필요 (vLLM 서버는 키 검증 안 함)
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

    t0 = time.time()
    # client.audio.transcriptions.create는 OpenAI 공식 인터페이스
    result = client.audio.transcriptions.create(
        model=MODEL,
        file=(fname, audio_bytes, "audio/wav"),
        language="en",
        response_format="json",
    )
    dt = time.time() - t0

    print(f"[transcription:openai-sdk] ({dt:.2f}s)")
    print(f"  text = {result.text!r}")


# ----------------------------------------------------------------------------
# 스크립트로 직접 실행됐을 때만 main 호출
# ----------------------------------------------------------------------------
# 이 if 블록은 "이 파일이 import 되는 게 아니라 직접 실행됐을 때만 동작하라"는
# Python의 관례적 가드. 다른 코드가 from test_server import 했을 때
# 테스트가 자동 실행되지 않게 막아줌.
if __name__ == "__main__":
    test_list_models()
    print()
    test_transcription_curl_like()
    print()
    test_transcription_openai_sdk()
