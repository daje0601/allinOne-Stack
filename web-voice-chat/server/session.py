"""연결 1개 당 대화 히스토리 + 시스템 프롬프트를 유지하는 객체."""
from __future__ import annotations
from dataclasses import dataclass, field

from .config import SETTINGS

# LLM 컨텍스트(max_model_len=8192)를 보호하기 위해 최근 N턴만 보관.
MAX_TURNS_KEPT = 10  # user/assistant 메시지 쌍 기준


@dataclass
class Session:
    system_prompt: str = field(default_factory=lambda: SETTINGS.llm_system_prompt)
    history: list[dict] = field(default_factory=list)  # [{"role": ..., "content": ...}, ...]

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        # 너무 길어지면 앞쪽부터 잘라냄 (시스템 프롬프트는 별도라 history에 안 들어있음)
        max_msgs = MAX_TURNS_KEPT * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def messages_for_llm(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}, *self.history]

    def reset(self) -> None:
        self.history.clear()
