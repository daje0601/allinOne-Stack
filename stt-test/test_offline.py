# SPDX-License-Identifier: Apache-2.0
"""
============================================================================
Whisper STT 오프라인 추론 테스트
============================================================================

"오프라인 추론"이란 무엇인가?
  - 서버를 띄우지 않고, 이 파이썬 스크립트 자체가 모델을 메모리에 올려서
    음성→텍스트 변환을 직접 수행하는 것.
  - 서버 모드와 비교: 서버는 외부에서 HTTP로 요청을 받지만, 오프라인은
    같은 프로세스 안에서 직접 vLLM API를 호출.
  - 용도: "이 모델이 우리 GPU에서 잘 도는지" 가장 빠르게 확인하는 smoke test.

이 스크립트는 vLLM 공식 Whisper 예제를 가져와서 turbo 모델용으로
파라미터만 살짝 조정한 것입니다.

실행 방법:
  unset LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES=0 uv run python test_offline.py

기대 결과:
  - 모델 로드 + CUDA graph 캡처 ~100초 (첫 실행)
  - 추론 8개 prompt 1~2초 (RPS ~5)
  - "Mary had a little lamb..."과 야구 winning call 텍스트가 출력됨
============================================================================
"""
import time                                    # 시간 측정용 표준 라이브러리

# vLLM 패키지에서 두 가지 클래스 import:
#   LLM            : 모델을 메모리에 올려두고 추론 요청을 받는 객체
#   SamplingParams : "어떻게 생성할지"를 적은 설정 묶음 (온도, 최대 토큰 등)
from vllm import LLM, SamplingParams

# vLLM이 기본 제공하는 테스트용 오디오 (Edison의 첫 녹음, 야구 중계)
from vllm.assets.audio import AudioAsset


# ----------------------------------------------------------------------------
# 1) 모델 인스턴스 생성 (시간 가장 오래 걸리는 단계)
# ----------------------------------------------------------------------------
# 이 줄을 만나면 vLLM은:
#   a. HuggingFace에서 모델 가중치 다운로드 (캐시되어있으면 스킵)
#   b. GPU에 가중치 로드 (~1.5GB)
#   c. 가짜 prompt로 메모리 사용량 프로파일링
#   d. CUDA Graph 캡처 (반복 추론 가속을 위한 최적화)
# 전체 1~3분 소요.
llm = LLM(
    model="openai/whisper-large-v3-turbo",     # HuggingFace 모델 이름
    max_model_len=448,                         # Whisper 디코더 컨텍스트 한계 (고정값)
    max_num_seqs=400,                          # 동시에 처리 가능한 최대 prompt 수
    limit_mm_per_prompt={"audio": 1},          # 한 prompt당 오디오 1개 첨부 가능
    kv_cache_dtype="fp8",                      # KV 캐시를 8비트로 저장 (메모리 절약)
)


# ----------------------------------------------------------------------------
# 2) 입력 prompt 8개 준비 (4번 반복 × 2가지 형식)
# ----------------------------------------------------------------------------
# Whisper는 encoder-decoder 모델입니다:
#   - encoder가 오디오를 "내부 표현"으로 인코딩
#   - decoder가 그 표현을 보고 텍스트를 생성
#
# vLLM에서 두 가지 입력 형식 모두 지원:
#   (1) 간단형: {"prompt": ..., "multi_modal_data": {...}}
#   (2) 명시형: {"encoder_prompt": {...}, "decoder_prompt": ...}
# 두 형식이 같은 의미인지 검증하려고 일부러 둘 다 섞어둠.
prompts = [
    {
        # 간단형: prompt에는 디코더 시작 토큰만, audio는 multi_modal_data로
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
    {
        # 명시형: encoder/decoder를 명확히 분리
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
        "decoder_prompt": "<|startoftranscript|>",
    },
] * 4  # [A, B] * 4 = [A, B, A, B, A, B, A, B] — 총 8개


# ----------------------------------------------------------------------------
# 3) 생성 파라미터 (어떻게 텍스트를 만들지)
# ----------------------------------------------------------------------------
sampling_params = SamplingParams(
    temperature=0,                             # 0 = 결정론적 (같은 입력엔 같은 출력)
    top_p=1.0,                                 # nucleus sampling 비활성
    max_tokens=200,                            # 최대 200토큰까지 생성
)


# ----------------------------------------------------------------------------
# 4) 추론 실행 — 8개 prompt를 한 번에 배치로 처리
# ----------------------------------------------------------------------------
start = time.time()                            # 시작 시각 기록
outputs = llm.generate(prompts, sampling_params)  # ← 실제 추론. 보통 1~2초.
duration = time.time() - start                 # 걸린 시간


# ----------------------------------------------------------------------------
# 5) 결과 출력
# ----------------------------------------------------------------------------
# enumerate는 (0, 첫번째), (1, 두번째), ... 로 인덱스를 함께 반환
for i, output in enumerate(outputs):
    prompt = output.prompt                     # 디코더에 들어간 prompt
    encoder_prompt = output.encoder_prompt     # 인코더에 들어간 prompt
    generated_text = output.outputs[0].text    # 생성된 텍스트 (첫 후보만)

    # f-string: f"..."안에 {변수}를 쓰면 그 값이 박힘. !r은 repr() — 따옴표 포함 표시.
    print(f"[{i}] encoder_prompt={encoder_prompt!r}")
    print(f"    decoder_prompt={prompt!r}")
    print(f"    text={generated_text!r}")

print("---")
print(f"Duration: {duration:.2f}s")            # .2f = 소수점 둘째 자리까지
print(f"RPS: {len(prompts) / duration:.2f}")   # Requests Per Second = 처리량
