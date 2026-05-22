"""
============================================================================
Chroma 벡터 데이터베이스 빌드 스크립트 (RAG의 "검색용 색인" 만들기)
============================================================================

이 스크립트는 한 번만 실행합니다. 챗봇이 동작할 때마다 매번 돌리는 게 아니라
"미리 색인을 만들어 디스크에 저장"하는 단계.

RAG (Retrieval-Augmented Generation)란?
  1. 사용자 질문이 들어오면
  2. "관련된 자료"를 데이터베이스에서 검색하고 (Retrieval)
  3. 그 자료를 LLM 프롬프트에 끼워넣어
  4. LLM이 자료를 참고해 답변 생성 (Generation)

검색을 잘 하려면 "의미"로 검색해야 합니다. "꿀"이라고 물으면 "벌" 관련
내용도 찾아야 좋음. 이걸 가능하게 하는 게 **임베딩**:
  - 텍스트를 1536차원 같은 숫자 벡터로 변환
  - 의미가 비슷한 문장은 벡터 공간에서 가까이 위치
  - "코사인 유사도"로 가까운 벡터 찾기 = 의미 검색

이 스크립트는 OpenAI의 text-embedding-3-small 모델을 사용해
data/{characters,episodes,episodes2}.json 안의 210개 문서를 모두
임베딩한 뒤 Chroma라는 로컬 벡터DB에 저장합니다.

실행 방법:
  cp .env.example .env       # OPENAI_API_KEY=sk-... 채우기
  unset LD_LIBRARY_PATH
  uv run python ingest.py

결과:
  chroma_db/ 폴더에 sqlite + 임베딩 파일 생성 (~수 MB)
  → 이후 app.py가 그 폴더를 읽어 검색에 사용
============================================================================
"""
import json
from pathlib import Path                       # 경로를 객체로 다루는 표준 라이브러리

# .env 파일 자동 로드 (OPENAI_API_KEY를 환경변수로 등록)
from dotenv import load_dotenv

# LangChain — RAG 파이프라인 구성 도구
from langchain_community.vectorstores import Chroma       # Chroma 벡터DB 래퍼
from langchain_core.documents import Document             # 문서 객체 (text + 메타데이터)
from langchain_openai import OpenAIEmbeddings             # OpenAI 임베딩 모델 래퍼


# ----------------------------------------------------------------------------
# 경로 상수
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent                   # 이 스크립트가 놓인 폴더
DATA = HERE / "data"                           # data/  (입력 json)
PERSIST_DIR = HERE / "chroma_db"               # chroma_db/  (출력 색인)


# ----------------------------------------------------------------------------
# 1) 입력 텍스트 로드
# ----------------------------------------------------------------------------
def load_corpus() -> list[str]:
    """data/ 안의 3개 json에서 텍스트만 모아 평탄한 리스트로 반환.

    구성:
      characters.json — 10개 캐릭터 소개 (푸/피글렛/티거 등)
      episodes.json   — 100개 에피소드
      episodes2.json  — 추가 100개 에피소드
      합계 = 210개 문서
    """
    docs: list[str] = []
    for fname in ("characters.json", "episodes.json", "episodes2.json"):
        # Path / "..." 는 경로 연결. read_text()는 파일 내용 통째로 읽기.
        # json.loads는 JSON 문자열을 Python 리스트/dict로 변환.
        docs.extend(json.loads((DATA / fname).read_text()))
    return docs


# ----------------------------------------------------------------------------
# 2) 메인 처리
# ----------------------------------------------------------------------------
def main() -> None:
    # .env에서 OPENAI_API_KEY를 환경변수로 로드. 없으면 OpenAIEmbeddings가 실패.
    load_dotenv(HERE / ".env")

    corpus = load_corpus()
    print(f"loaded {len(corpus)} documents from {DATA}")

    # 텍스트 문자열 리스트 → LangChain Document 객체 리스트로 변환
    # (LangChain은 메타데이터를 같이 다루기 위해 Document 객체를 요구)
    documents = [Document(page_content=t) for t in corpus]

    # 임베딩 모델 선택. text-embedding-3-small은 OpenAI의 저렴+우수한 임베딩
    # (1536차원, $0.02 per 1M tokens — 210 docs 임베딩 비용은 거의 0)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"embedding + persisting to {PERSIST_DIR} ...")
    # Chroma.from_documents는 다음을 한 번에 처리:
    #   1. 각 문서를 OpenAI API로 임베딩 (네트워크 호출, ~수 초)
    #   2. 임베딩 벡터를 Chroma DB에 저장
    #   3. persist_directory에 sqlite + parquet 형식으로 디스크에 영속화
    Chroma.from_documents(
        documents,
        embedding,
        persist_directory=str(PERSIST_DIR),
    )
    print("done.")


if __name__ == "__main__":
    main()
