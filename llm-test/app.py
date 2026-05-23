"""
============================================================================
곰돌이 푸 페르소나 챗봇 — CLI 테스트 도구
============================================================================

이 파일의 역할:
  1. 곰돌이 푸 페르소나의 원본 정의(SYSTEM_PROMPT)와 chat_fn 로직을 보관.
  2. 명령줄에서 ad-hoc으로 페르소나 응답을 빠르게 검증.

실제 사용자가 마주하는 챗 UI는 `web-voice-chat/`(FastAPI + WS) 입니다.
이 파일의 SYSTEM_PROMPT(`data/system_prompt.txt`)를 web-voice-chat이
자동으로 읽어서 동일한 페르소나로 음성 대화합니다.

실행:
  unset LD_LIBRARY_PATH
  uv run python app.py
  → "you> " 프롬프트가 뜸. 빈 줄 또는 'exit' 입력 시 종료.

전체 흐름 (CLI 한 턴):
  CLI (이 파일)
       ↓ 사용자 입력
       ↓ (옵션) Chroma에서 비슷한 에피소드 5개 검색 — chroma_db 있을 때만
       ↓ 시스템프롬프트 + 대화이력 + 질문 + 검색결과를 묶어
  vLLM 서버 (LLaMA3 8B, port 13000)
       ↓ 페르소나 톤으로 응답 토큰 단위 스트리밍
  CLI (이 파일)
       ↓ 토큰 받을 때마다 stdout으로 출력
============================================================================
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------------------------------------------------------
# 시작 시 1회 실행: .env 로드 + 환경변수 설정
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent
load_dotenv(HERE / ".env")


# ----------------------------------------------------------------------------
# 설정 (환경변수 우선, 없으면 기본값)
# ----------------------------------------------------------------------------
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:13000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "iamjoon/llama3-8b-persona-chatbot")
PERSIST_DIR = HERE / "chroma_db"               # ingest.py가 만든 벡터DB 폴더
SYSTEM_PROMPT_FILE = HERE / "data" / "system_prompt.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text()


# ----------------------------------------------------------------------------
# 시작 시 1회 실행: 클라이언트 + (옵션) 벡터DB 핸들 준비
# ----------------------------------------------------------------------------
client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")

# RAG는 선택. chroma_db 폴더가 있을 때만 활성화.
# `ingest.py`로 데이터 색인을 만든 적이 없으면 자동으로 비활성화되고,
# 페르소나만 적용된 일반 챗으로 동작.
vectordb = None
if PERSIST_DIR.exists():
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embedding)
    except Exception as e:
        print(f"[warn] Chroma 로드 실패 → RAG OFF: {e}")


# ----------------------------------------------------------------------------
# RAG 헬퍼 1: 질문에 관련된 에피소드 top-k 검색해서 컨텍스트 문자열로 포맷
# ----------------------------------------------------------------------------
def get_formatted_context(question: str, k: int = 5) -> str:
    """질문 → Chroma 검색 → <context>...</context> XML-스타일 문자열로 묶기.

    vectordb이 없으면 빈 문자열 반환 (= RAG 비활성화).
    """
    if vectordb is None:
        return ""
    docs = vectordb.similarity_search(question, k=k)
    body = "\n".join(f"<doc{i}>{d.page_content}</doc{i}>" for i, d in enumerate(docs, 1))
    return f"<context>\n{body}\n</context>"


# ----------------------------------------------------------------------------
# RAG 헬퍼 2: 이전 대화 이력에서 마지막에 끼워넣었던 context를 제거
# ----------------------------------------------------------------------------
def remove_last_context(messages: list[dict]) -> list[dict]:
    """messages 리스트에서 가장 최근의 <context>로 시작하는 user 메시지를 1개 제거."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user" and messages[i]["content"].startswith("<context>"):
            return messages[:i] + messages[i + 1:]
    return messages


# ----------------------------------------------------------------------------
# 핵심: 한 턴의 대화 흐름 (스트리밍 generator)
# ----------------------------------------------------------------------------
def chat_fn(user_input: str, history: list[dict]):
    """사용자 입력 → vLLM 호출 → 토큰 단위 스트리밍.

    yield (assistant_full_text, full_messages_history) — 매 토큰마다.
    """
    # 1) 옛 context를 정리하고
    messages = remove_last_context(history.copy())

    # 2) 새 사용자 질문 + (있다면) 새로 검색한 컨텍스트를 추가
    messages.append({"role": "user", "content": user_input})
    ctx = get_formatted_context(user_input)
    if ctx:
        messages.append({"role": "user", "content": ctx})

    # 3) 응답이 들어올 자리(빈 assistant turn)를 미리 마련
    messages.append({"role": "assistant", "content": ""})

    # 4) vLLM에 스트리밍 요청
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages[:-1]],
        temperature=0,
        max_tokens=2048,
        stream=True,
        extra_body={"stop": ["<|eot_id|>"]},   # LLaMA3 인스트럭트 종료 토큰
    )

    # 5) 스트림에서 토큰이 도착할 때마다 누적 + yield
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        messages[-1]["content"] += delta
        yield messages[-1]["content"], messages


# ----------------------------------------------------------------------------
# CLI 진입점: 간단한 대화 루프
# ----------------------------------------------------------------------------
def _cli_loop() -> None:
    """터미널에서 곰돌이 푸와 대화. Ctrl+D 또는 'exit' 입력 시 종료."""
    print("=" * 60)
    print(" 곰돌이 푸 CLI 챗봇")
    print(f"   model : {LLM_MODEL}")
    print(f"   vLLM  : {VLLM_URL}")
    print(f"   RAG   : {'ON (chroma_db 로드됨)' if vectordb else 'OFF (chroma_db 없음)'}")
    print(" 종료: Ctrl+D 또는 'exit'")
    print("=" * 60)
    history: list[dict] = []
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in {"exit", "quit", "q"}:
            break

        print("푸> ", end="", flush=True)
        last_text = ""
        new_history: list[dict] = []
        for text, msgs in chat_fn(user_input, history):
            # 토큰 단위 prefix delta만 출력
            delta = text[len(last_text):]
            print(delta, end="", flush=True)
            last_text = text
            new_history = msgs
        print()
        history = new_history


if __name__ == "__main__":
    _cli_loop()
