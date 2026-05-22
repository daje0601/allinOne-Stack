# LLM API 사용 가이드

vLLM이 `iamjoon/llama3-8b-persona-chatbot`를 OpenAI-호환 chat completions 엔드포인트로
서빙합니다. 이 문서는 **vLLM 서버 API 호출 방법**에 집중. RAG/Gradio 통합은
[`README.md`](./README.md)와 `app.py` 참고.

- 베이스 URL: `http://<server-host>:13000/v1`
- 모델 ID: `iamjoon/llama3-8b-persona-chatbot`
- 인증: 없음 (`api_key="EMPTY"`)

## 0. 다른 서버에서 호출하려면

서버는 `0.0.0.0:13000`에 바인딩 → LAN 어디서든 접근 가능. 단,

```bash
# 호스트 방화벽 13000 포트 개방
sudo ufw allow 13000
# 클라이언트에서 접근 확인
curl http://<server-host>:13000/v1/models
```

운영 시엔 reverse proxy + HTTPS + 인증 토큰 래핑 권장 (vLLM 자체엔 인증 없음).

## 1. 엔드포인트

| Method | Path | 용도 |
|---|---|---|
| GET | `/v1/models` | 등록된 모델 목록 (헬스체크) |
| POST | `/v1/chat/completions` | 메인 챗 (멀티턴 지원) |
| POST | `/v1/completions` | 단발 텍스트 컴플리션 |
| GET | `/health` | 헬스체크 |

## 2. 요청 형식 (`/v1/chat/completions`)

### 필수

| 필드 | 값 |
|---|---|
| `model` | `iamjoon/llama3-8b-persona-chatbot` |
| `messages` | `[{role, content}, ...]` (role: `system`/`user`/`assistant`) |

### 권장

| 필드 | 기본/권장 | 비고 |
|---|---|---|
| `temperature` | `0` | 결정론적. 다양성 원하면 0.7~1.0 |
| `max_tokens` | `2048` | LLaMA3 컨텍스트 한계 내에서 |
| `stop` | `["<|eot_id|>"]` | LLaMA3 instruct format의 종결 토큰. 누락 시 생성 멈춤 늦어질 수 있음 |
| `stream` | `false` (기본) | `true`로 SSE 스트리밍 가능 |

## 3. 호출 예시

### 3-1. curl

```bash
curl -X POST http://localhost:13000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "iamjoon/llama3-8b-persona-chatbot",
    "messages": [
      {"role": "system", "content": "당신의 이름은 푸입니다."},
      {"role": "user", "content": "기분이 울적할 땐 어떻게 해?"}
    ],
    "temperature": 0,
    "max_tokens": 512,
    "stop": ["<|eot_id|>"]
  }'
```

### 3-2. Python — OpenAI SDK (권장)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:13000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="iamjoon/llama3-8b-persona-chatbot",
    messages=[
        {"role": "system", "content": "당신은 곰돌이 푸입니다."},
        {"role": "user",   "content": "오늘은 뭐가 좋아?"},
    ],
    temperature=0,
    max_tokens=512,
    extra_body={"stop": ["<|eot_id|>"]},
)
print(resp.choices[0].message.content)
```

### 3-3. 스트리밍

```python
stream = client.chat.completions.create(
    model="iamjoon/llama3-8b-persona-chatbot",
    messages=[...],
    stream=True,
    extra_body={"stop": ["<|eot_id|>"]},
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### 3-4. 멀티턴 (history 직접 관리)

```python
history = []
def turn(user_msg: str) -> str:
    history.append({"role": "user", "content": user_msg})
    r = client.chat.completions.create(
        model="iamjoon/llama3-8b-persona-chatbot",
        messages=[{"role":"system","content":SYS}, *history],
        temperature=0, max_tokens=512,
        extra_body={"stop":["<|eot_id|>"]},
    )
    answer = r.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    return answer

print(turn("기분 안 좋아"))
print(turn("그래서 뭘 하면 좋을까?"))   # 이전 맥락 유지됨
```

### 3-5. RAG 패턴 (app.py의 축약본)

```python
# 1. 검색
docs = vectordb.similarity_search(user_query, k=5)
context = "<context>\n" + "\n".join(
    f"<doc{i}>{d.page_content}</doc{i}>" for i, d in enumerate(docs, 1)
) + "\n</context>"

# 2. messages에 context를 user role로 삽입
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *history,
    {"role": "user", "content": user_query},
    {"role": "user", "content": context},   # ← 검색 결과
]

# 3. 호출
resp = client.chat.completions.create(model=..., messages=messages, ...)
```

## 4. 시스템 프롬프트 구조 (`data/system_prompt.txt`)

페르소나 + 답변 형식 + 힌트 사용법까지 1222자로 압축돼있습니다. 핵심:

```
- 정체성: 이름=푸, 곰, 100에이커 숲 거주, 꿀 좋아함
- 답변 형식: 단순/순수 말투, 느린 속도, 공감 중심, 감각적 비유, '너' 중심
- <context>...</context>가 messages에 있으면 그걸 힌트로 활용
```

본인 페르소나 챗봇으로 바꾸려면 이 텍스트 파일만 교체하면 됩니다 (모델은 그대로).

## 5. 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| `Connection refused` | 서버 안 떠있음 | `curl http://<host>:13000/v1/models` 로 확인 |
| 응답이 영어로 나옴 | system prompt 누락 | system role 메시지 반드시 포함 |
| 응답이 비어있음 | `stop`에 잘못된 토큰, 또는 모델 다른 종결 토큰 | `extra_body={"stop":["<|eot_id|>"]}` 확인. 또는 `stop` 제거하고 max_tokens만 사용 |
| 응답이 끊김 | `max_tokens` 부족 | 2048→4096 등 상향, 또는 stream=True로 받아 클라이언트에서 처리 |
| 같은 답만 반복 | `temperature=0` | 0.7~1.0로 올리거나 `top_p=0.9`, `presence_penalty=0.5` |
| GPU OOM | KV cache 부족 | start_server.sh의 `--gpu-memory-utilization`을 0.7로 낮추거나, `--max-model-len`을 4096으로 |

## 6. 빠른 검증

```bash
# 헬스
curl http://localhost:13000/v1/models

# 단발 호출 (system 없이 — 페르소나 살리려면 system 필수)
curl -X POST http://localhost:13000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"iamjoon/llama3-8b-persona-chatbot","messages":[{"role":"user","content":"안녕"}],"max_tokens":64}'
```
