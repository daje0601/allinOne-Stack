"""
============================================================================
[역사적 잔재] vllm-omni 디버깅용 Python 래퍼 — 현재 미사용
============================================================================

이 파일은 개발 초기에 vllm-omni 0.16.0의 `--omni` 플래그 위치 문제를
우회하려고 만든 임시 래퍼입니다. 지금은 `start_server.sh`가 `vllm-omni serve
<모델> --omni` 형태로 직접 호출하므로 **이 래퍼는 더 이상 쓰이지 않습니다**.

왜 만들었었나 (참고용 기록):
  - vllm-omni 0.16.0의 `vllm_omni.entrypoints.cli.main:main`은 sys.argv에
    `--omni`가 있는지로 분기. 그런데 `--omni` 플래그는 `serve` 서브명령의
    하위 옵션이라 위치를 잘못 두면 argparse가 거부함.
  - 디버깅 도중 "위치는 신경 쓰지 말고 Python에서 직접 등록 함수만 호출하자"는
    아이디어로 이 래퍼 제작. → 모델 로드는 됐지만 /v1/audio/speech 라우트는
    여전히 없어서 막힘.
  - 결국 `serve` 뒤에 `--omni`를 두면 CLI도 잘 동작한다는 걸 발견. 이 래퍼는
    필요 없어졌지만, 같은 함정에 빠지는 사람을 위한 학습 자료로 보존.

지우셔도 무방합니다.
============================================================================
"""
import sys

# vllm_omni를 import만 해도 patch.py가 실행되며 vLLM의 일부 클래스가 패치됨.
# multiprocessing worker가 이 모듈을 재-import할 때도 같은 patch가 적용되도록
# import 시점에 실행.
import vllm_omni  # noqa: F401  -- runs vllm_omni.patch
from vllm_omni.engine.arg_utils import register_omni_models_to_vllm

# qwen3_tts model_type을 transformers의 AutoConfig에 등록.
# (transformers 4.57.6은 qwen3_tts를 모르므로 이 줄 없이는 vLLM이 거부)
register_omni_models_to_vllm()


# __main__ 가드: multiprocessing이 워커에서 이 모듈을 import할 때
# vllm_main()이 한 번 더 실행돼 무한 재귀처럼 되는 걸 방지
if __name__ == "__main__":
    # vllm 일반 CLI에 sys.argv 그대로 넘김
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.exit(vllm_main())
