# SPDX-License-Identifier: Apache-2.0
"""
============================================================================
Qwen3-TTS 서버 클라이언트 테스트 (텍스트 → 음성 wav)
============================================================================

이미 떠있는 TTS 서버(`./start_server.sh`)에 요청을 보내 영어/한국어 음성을
합성하고 wav 파일로 저장합니다.

테스트하는 시나리오:
  1. GET /v1/models                 — 서버 헬스체크
  2. POST /v1/audio/speech (영어)   — requests로 합성, samples/tts_test_en.wav 저장
  3. POST /v1/audio/speech (한국어) — 같은 방식, samples/tts_test_ko.wav 저장
  4. OpenAI SDK 호출                 — samples/tts_test_sdk.wav 저장

비개발자용 용어:
  - voice: 화자 이름. vivian/ryan/sohee 등 9개 preset 중 선택.
  - language: 발음 결정에 영향. 한국어면 "Korean" 명시 필수
              (생략하면 영어로 폴백돼 발음이 이상해짐).
  - response_format: 출력 오디오 포맷. wav/mp3/flac/pcm/aac/opus.

실행:
  unset LD_LIBRARY_PATH
  uv run python test_server.py
============================================================================
"""
import os
import time

import requests
from openai import OpenAI


# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
BASE_URL = os.environ.get("VLLM_URL", "http://localhost:12000/v1")
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


# ----------------------------------------------------------------------------
# 테스트 1: /v1/models 조회 (서버 살아있나)
# ----------------------------------------------------------------------------
def test_list_models() -> None:
    r = requests.get(f"{BASE_URL}/models", timeout=10)
    r.raise_for_status()
    print("[/v1/models]")
    for m in r.json().get("data", []):
        print(f"  - {m['id']}")


# ----------------------------------------------------------------------------
# 공통 헬퍼: TTS 요청 1회 발사 + 결과를 파일로 저장 + 걸린 시간 반환
# ----------------------------------------------------------------------------
def speech_request(text: str, language: str, out_path: str,
                   voice: str = "vivian") -> float:
    """텍스트 → 음성 wav 합성 요청.

    인자:
      text:     합성할 텍스트
      language: "English"/"Korean"/... (Auto는 비추천)
      out_path: 응답 wav를 저장할 경로
      voice:    화자 이름 (기본 vivian)

    반환: 응답까지 걸린 시간(초)
    """
    t0 = time.time()
    # POST /v1/audio/speech 본문은 JSON.
    # requests의 json=... 인자는 자동으로 JSON 직렬화 + Content-Type 설정.
    r = requests.post(
        f"{BASE_URL}/audio/speech",
        json={
            "model": MODEL,
            "input": text,
            "voice": voice,
            "language": language,
            "response_format": "wav",
            "task_type": "CustomVoice",        # CustomVoice 모델 전용 태스크
        },
        timeout=120,                            # TTS는 STT보다 오래 걸림
    )
    dt = time.time() - t0
    r.raise_for_status()
    # 응답 본문(r.content)은 wav 바이너리. "wb"는 write-binary 모드.
    with open(out_path, "wb") as f:
        f.write(r.content)
    return dt


# ----------------------------------------------------------------------------
# 테스트 2: 영어 합성
# ----------------------------------------------------------------------------
def test_english() -> None:
    dt = speech_request(
        "Hello world. This is Qwen 3 TTS speaking through vLLM Omni.",
        "English",
        "samples/tts_test_en.wav",
    )
    size = os.path.getsize("samples/tts_test_en.wav")  # 파일 크기(바이트)
    print(f"[english] {dt:.2f}s → samples/tts_test_en.wav ({size} bytes)")


# ----------------------------------------------------------------------------
# 테스트 3: 한국어 합성 (language="Korean" 명시!)
# ----------------------------------------------------------------------------
def test_korean() -> None:
    dt = speech_request(
        "안녕하세요. 큐웬 쓰리 티티에스가 한국어로 말하고 있습니다.",
        "Korean",                              # ← 이거 빼면 영어로 폴백돼 망함
        "samples/tts_test_ko.wav",
    )
    size = os.path.getsize("samples/tts_test_ko.wav")
    print(f"[korean]  {dt:.2f}s → samples/tts_test_ko.wav ({size} bytes)")


# ----------------------------------------------------------------------------
# 테스트 4: OpenAI 공식 SDK 사용 (다른 OpenAI 코드 그대로 재활용 가능)
# ----------------------------------------------------------------------------
def test_openai_sdk() -> None:
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    t0 = time.time()
    response = client.audio.speech.create(
        model=MODEL,
        voice="vivian",
        input="Testing the OpenAI SDK path for Qwen3 TTS.",
    )
    # SDK가 제공하는 헬퍼: 응답을 파일로 바로 저장
    response.stream_to_file("samples/tts_test_sdk.wav")
    dt = time.time() - t0
    size = os.path.getsize("samples/tts_test_sdk.wav")
    print(f"[sdk]     {dt:.2f}s → samples/tts_test_sdk.wav ({size} bytes)")


# ----------------------------------------------------------------------------
# 직접 실행 시 4개 테스트 순차 수행
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    test_list_models()
    print()
    test_english()
    test_korean()
    test_openai_sdk()
