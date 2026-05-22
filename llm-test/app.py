"""
============================================================================
곰돌이 푸 페르소나 RAG 챗봇 (Gradio 웹 UI)
============================================================================

이 파일은 사용자가 브라우저로 접속해 챗봇과 대화하는 웹 앱입니다.
한 줄 요약: "곰돌이 푸"라는 페르소나로 한국어 질문에 답하는 챗봇 +
관련 에피소드를 자동 검색해 답변에 활용 (RAG).

전체 흐름:
   사용자 (브라우저)
        ↓ 질문 입력
   Gradio (이 파일, port 14000)
        ↓ 1. Chroma에서 비슷한 에피소드 5개 검색
        ↓ 2. 시스템프롬프트 + 대화이력 + 질문 + 검색결과를 묶어
   vLLM 서버 (LLaMA3 8B, port 13000)
        ↓ 3. 페르소나 톤으로 응답 토큰 단위 스트리밍
   Gradio (이 파일)
        ↓ 4. 토큰 받을 때마다 채팅창 갱신
   사용자 (브라우저)

구성 요소:
  - Gradio       : 웹 UI 프레임워크 (HTML/JS 직접 안 짜도 챗UI 자동 생성)
  - LangChain    : Chroma 벡터DB를 다루는 도구
  - OpenAI SDK   : vLLM 서버(13000)에 HTTP 호출
  - python-dotenv: .env 파일에서 OPENAI_API_KEY 읽기

실행:
  unset LD_LIBRARY_PATH
  uv run python app.py
  # → http://<host>:14000 접속
============================================================================
"""
import os
from pathlib import Path

import gradio as gr                            # 웹 UI 프레임워크
from dotenv import load_dotenv                 # .env 파일 로더
from langchain_community.vectorstores import Chroma     # 벡터DB
from langchain_openai import OpenAIEmbeddings           # 임베딩 모델
from openai import OpenAI                      # OpenAI 호환 HTTP 클라이언트


# ----------------------------------------------------------------------------
# 시작 시 1회 실행: .env 로드 + 환경변수 설정
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent                   # 이 스크립트의 경로
load_dotenv(HERE / ".env")                     # OPENAI_API_KEY 등을 환경변수에 등록


# ----------------------------------------------------------------------------
# 설정 (환경변수 우선, 없으면 기본값)
# ----------------------------------------------------------------------------
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:13000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "iamjoon/llama3-8b-persona-chatbot")
PERSIST_DIR = HERE / "chroma_db"               # ingest.py가 만든 벡터DB 폴더
SYSTEM_PROMPT = (HERE / "data" / "system_prompt.txt").read_text()  # 페르소나 정의


# ----------------------------------------------------------------------------
# 시작 시 1회 실행: 클라이언트 + 벡터DB 핸들 준비
# ----------------------------------------------------------------------------
# LLM 호출 클라이언트 (vLLM 서버 13000 향함)
client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")

# 임베딩 모델 (ingest.py와 반드시 같은 모델 사용 — 안 그러면 벡터가 호환 안 됨)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 디스크에 저장된 Chroma 색인을 메모리에 로드
vectordb = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=embedding,
)


# ----------------------------------------------------------------------------
# RAG 헬퍼 1: 질문에 관련된 에피소드 top-5 검색해서 컨텍스트 문자열로 포맷
# ----------------------------------------------------------------------------
def get_formatted_context(question: str, k: int = 5) -> str:
    """질문 → Chroma 검색 → <context>...</context> XML-스타일 문자열로 묶기.

    LLM이 "어디부터 어디까지가 검색 결과인지" 알아보기 쉽게 태그로 감쌈.
    """
    # similarity_search: 질문을 임베딩한 뒤 가장 가까운 k개 문서 반환
    docs = vectordb.similarity_search(question, k=k)
    # join: 여러 줄을 줄바꿈으로 이어 붙임
    body = "\n".join(f"<doc{i}>{d.page_content}</doc{i}>" for i, d in enumerate(docs, 1))
    return f"<context>\n{body}\n</context>"


# ----------------------------------------------------------------------------
# RAG 헬퍼 2: 이전 대화 이력에서 마지막에 끼워넣었던 context를 제거
# ----------------------------------------------------------------------------
# 매 턴마다 새 검색 결과를 끼우므로, 옛 검색 결과는 빼야 컨텍스트 폭발 안 함.
def remove_last_context(messages: list[dict]) -> list[dict]:
    """messages 리스트에서 가장 최근의 <context>로 시작하는 user 메시지를 1개 제거."""
    # 뒤에서부터 훑어 처음 만나는 context user 메시지를 빼고 반환
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user" and messages[i]["content"].startswith("<context>"):
            return messages[:i] + messages[i + 1:]
    return messages                            # 없으면 그대로


# ----------------------------------------------------------------------------
# 핵심: 한 턴의 대화 흐름 (스트리밍 generator)
# ----------------------------------------------------------------------------
# Python의 "generator" 개념:
#   - `return` 대신 `yield`를 쓰는 함수
#   - 호출하면 즉시 결과를 반환하지 않고, 호출 측이 for 루프 등으로 끌어쓸 때
#     yield 직전까지 실행되고 멈춤. 다시 끌면 그 다음부터 진행.
#   - 여기서는 vLLM이 토큰을 받을 때마다 yield → Gradio가 UI 갱신.
def chat_fn(user_input: str, history: list[dict]):
    """사용자 입력 → vLLM 호출 → 토큰 단위 스트리밍.

    각 yield는 (chatbot에 보일 메시지 리스트, state에 저장할 메시지 리스트) 튜플.
    Gradio가 매 yield마다 chatbot 영역을 다시 그림.
    """
    # 1) 옛 context를 정리하고
    messages = remove_last_context(history.copy())

    # 2) 새 사용자 질문 + 새로 검색한 컨텍스트를 추가
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "user", "content": get_formatted_context(user_input)})

    # 3) 응답이 들어올 자리(빈 assistant turn)를 미리 마련
    #    스트리밍 도중 이 자리에 토큰을 누적
    messages.append({"role": "assistant", "content": ""})

    # 4) vLLM에 스트리밍 요청
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        # 실제 prompt = [system 페르소나] + 대화 이력 + 질문 + 컨텍스트
        #              (마지막 빈 assistant turn은 빼고 보냄)
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages[:-1]],
        temperature=0,                         # 결정론적 (재현 가능)
        max_tokens=2048,
        stream=True,                           # 토큰 단위로 받기
        extra_body={"stop": ["<|eot_id|>"]},   # LLaMA3 인스트럭트 형식의 종료 토큰
    )

    # 5) 스트림에서 토큰이 도착할 때마다 누적 + yield
    for chunk in stream:
        delta = chunk.choices[0].delta.content  # 이번 청크에 새로 들어온 텍스트
        if not delta:
            continue                            # 빈 청크는 스킵 (서버 키핑 등)
        messages[-1]["content"] += delta        # 마지막 assistant turn에 누적
        yield messages, messages                # Gradio: "다시 그려"


# ----------------------------------------------------------------------------
# Gradio UI 정의
# ----------------------------------------------------------------------------
def make_ui() -> gr.Blocks:
    """챗봇 웹 페이지를 만들어 반환."""
    # gr.Blocks: 자유롭게 컴포넌트를 배치할 수 있는 컨테이너
    with gr.Blocks(title="곰돌이 푸 페르소나 챗봇") as demo:
        # 페이지 상단 제목 (markdown 형식)
        gr.Markdown("## 🐻 곰돌이 푸 멀티턴 RAG 챗봇 (LLaMA3 + vLLM + Chroma)")

        # 채팅 메시지가 표시되는 영역
        chatbot = gr.Chatbot(height=480)

        # 대화 이력 저장용 상태 변수 (사용자에게 안 보임)
        state = gr.State([])                    # 초기값: 빈 리스트

        # 입력창 + 전송 버튼을 가로로 배치
        with gr.Row():
            user_input = gr.Textbox(placeholder="푸에게 질문해보세요", label="", lines=1, scale=9)
            send_btn = gr.Button("전송", scale=1)

        # 전송 핸들러 — 스트리밍 generator
        def respond(text: str, history: list[dict]):
            # 첫 yield: 입력창을 즉시 비우고, 응답이 차오를 빈 영역을 준비
            yield history, history, ""
            # 이후 토큰이 들어올 때마다 chatbot + state 갱신, 입력창은 빈 채로
            for new_history, updated in chat_fn(text, history):
                yield new_history, updated, ""

        # 이벤트 바인딩:
        #   전송 버튼 클릭 → respond 호출
        #   엔터키 입력    → respond 호출
        # inputs/outputs는 함수의 인자/반환을 매핑하는 Gradio 약속
        send_btn.click(respond, [user_input, state], [chatbot, state, user_input])
        user_input.submit(respond, [user_input, state], [chatbot, state, user_input])

    return demo


# ----------------------------------------------------------------------------
# 스크립트로 직접 실행됐을 때 웹 서버 띄우기
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    demo = make_ui()
    demo.launch(
        server_name="0.0.0.0",                 # 모든 네트워크 인터페이스에서 접근 허용
        server_port=int(os.environ.get("GRADIO_PORT", "14000")),
        share=False,                           # Gradio의 공개 터널 기능은 끔
    )
