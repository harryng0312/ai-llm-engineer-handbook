# Dự án 1: Trợ lý RAG + Agent chạy local

> Dự án này ráp toàn bộ kiến thức từ [05-LLM](../05-LLM/README.md), [07-Vector-Database](../07-Vector-Database/README.md), [06-RAG](../06-RAG/README.md) và [08-AI-Agent](../08-AI-Agent/README.md) thành một ứng dụng chạy được từ đầu đến cuối: pipeline `Qwen -> BGE -> Qdrant -> RAG -> LangGraph -> Backend -> Web UI`.

> 📓 Toàn bộ code ví dụ trong dự án này cũng có ở dạng notebook chạy được: [`01-rag-agent-assistant.ipynb`](./01-rag-agent-assistant.ipynb).

## Mục lục

1. [Mục tiêu và kiến trúc](#1-mục-tiêu-và-kiến-trúc)
2. [Chuẩn bị môi trường](#2-chuẩn-bị-môi-trường)
3. [Bước 1 — Ingest tài liệu vào Qdrant](#3-bước-1--ingest-tài-liệu-vào-qdrant)
4. [Bước 2 — Agent kết hợp RAG + tool](#4-bước-2--agent-kết-hợp-rag--tool)
5. [Bước 3 — Backend API (FastAPI)](#5-bước-3--backend-api-fastapi)
6. [Bước 4 — Web UI](#6-bước-4--web-ui)
7. [Mở rộng: Rust Backend](#7-mở-rộng-rust-backend)
8. [Checklist triển khai](#8-checklist-triển-khai)
9. [Bài tập mở rộng](#9-bài-tập-mở-rộng)

---

## 1. Mục tiêu và kiến trúc

Xây một trợ lý hỏi-đáp trên tài liệu riêng (ví dụ: chính các chương của handbook này), chạy hoàn toàn local, có khả năng vừa trả lời dựa trên tài liệu (RAG) vừa dùng tool khi cần (tính toán, tra cứu bổ sung) — tức là **Agent có RAG như một trong các tool**, không phải RAG và Agent là hai hệ thống tách biệt.

```
                        ┌─────────────────────────────┐
   Tài liệu (.md/.txt)  │   Ingest pipeline (1 lần)    │
   ───────────────────▶ │  chunk -> BGE embed -> Qdrant│
                        └─────────────────────────────┘

┌──────────┐   HTTP   ┌────────────────┐   LangGraph   ┌───────────────────┐
│  Web UI  │ ───────▶ │  Backend API   │ ────────────▶ │  Agent (Qwen)      │
│ (HTML/JS)│ ◀─────── │  (FastAPI)     │ ◀──────────── │  tools: rag_search,│
└──────────┘  answer  └────────────────┘   answer      │  calculator        │
                                                         └─────────┬─────────┘
                                                                   │ rag_search tool
                                                                   ▼
                                                          Qdrant (chunks đã index)
```

Các thành phần đã học ở chương trước được tái sử dụng nguyên vẹn:

| Thành phần | Vai trò trong dự án | Xem lại ở |
|---|---|---|
| Qwen2.5 (qua Ollama) | LLM chính, vừa sinh câu trả lời vừa quyết định gọi tool | [05-LLM §7](../05-LLM/README.md#7-serving-ollama-vllm-llamacpp) |
| BGE (`bge-small`) | Embedding model để chunk hóa tài liệu và query | [07-Vector-Database §4](../07-Vector-Database/README.md#4-các-model-embedding-bgee5gte) |
| Qdrant | Lưu trữ vector, hỗ trợ filter theo nguồn tài liệu | [07-Vector-Database §7](../07-Vector-Database/README.md#7-qdrant) |
| Chunking + retrieval | Chuẩn bị dữ liệu và truy xuất ngữ cảnh | [06-RAG](../06-RAG/README.md) |
| LangGraph agent | Vòng lặp ReAct, gọi `rag_search` như một tool | [08-AI-Agent §5](../08-AI-Agent/README.md#5-langgraph) |

## 2. Chuẩn bị môi trường

```bash
# 1. Ollama + Qwen
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# 2. Qdrant (Docker)
docker run -d -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

# 3. Python deps
pip install fastapi uvicorn ollama qdrant-client sentence-transformers \
            langchain-ollama langgraph langchain-core \
            langchain-text-splitters
```

Cấu trúc thư mục đề xuất cho dự án:

```
handbook-assistant/
├── ingest.py        # Bước 1: nạp tài liệu vào Qdrant
├── agent.py         # Bước 2: định nghĩa agent + tool
├── server.py        # Bước 3: FastAPI backend
├── static/
│   └── index.html   # Bước 4: Web UI
└── docs/             # tài liệu nguồn, ví dụ copy các README.md ở chương trước vào đây
```

## 3. Bước 1 — Ingest tài liệu vào Qdrant

`ingest.py` — đọc mọi file `.md` trong thư mục `docs/`, chunk, embed bằng BGE, lưu vào Qdrant (áp dụng trực tiếp kỹ thuật ở [06-RAG §2-3](../06-RAG/README.md#2-chunking)):

```python
# ingest.py
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "handbook"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url="http://localhost:6333")
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    points = []
    point_id = 0
    for path in glob.glob("docs/**/*.md", recursive=True):
        text = open(path, encoding="utf-8").read()
        for chunk in splitter.split_text(text):
            vector = model.encode(chunk, normalize_embeddings=True)
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={"text": chunk, "source": path},
            ))
            point_id += 1

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Đã index {len(points)} chunks từ {len(glob.glob('docs/**/*.md', recursive=True))} file.")

if __name__ == "__main__":
    main()
```

```bash
python ingest.py
# Đã index 187 chunks từ 5 file.
```

## 4. Bước 2 — Agent kết hợp RAG + tool

`agent.py` — biến retrieval thành một **tool** (`rag_search`) để agent tự quyết định khi nào cần tra cứu tài liệu, kết hợp thêm một tool tiện ích (`calculate`), dùng LangGraph ReAct agent như ở [08-AI-Agent §3](../08-AI-Agent/README.md#3-langchain):

```python
# agent.py
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

_embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
_qdrant = QdrantClient(url="http://localhost:6333")

@tool
def rag_search(query: str) -> str:
    """Tìm kiếm trong tài liệu handbook (LLM, RAG, Vector DB, Agent) để trả lời câu hỏi kỹ thuật.
    Luôn dùng tool này trước khi trả lời câu hỏi liên quan đến nội dung handbook."""
    vector = _embed_model.encode(query, normalize_embeddings=True)
    hits = _qdrant.search(collection_name="handbook", query_vector=vector.tolist(), limit=3)
    if not hits:
        return "Không tìm thấy thông tin liên quan trong tài liệu."
    return "\n---\n".join(f"(nguồn: {h.payload['source']})\n{h.payload['text']}" for h in hits)

@tool
def calculate(expression: str) -> str:
    """Tính giá trị một biểu thức toán học, ví dụ '384 * 1024'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Lỗi: {e}"

SYSTEM_PROMPT = (
    "Bạn là trợ lý kỹ thuật của cuốn AI/LLM Engineer Handbook. "
    "Luôn dùng tool rag_search khi câu hỏi liên quan đến nội dung trong handbook, "
    "trích dẫn nguồn (source) khi trả lời. Nếu không tìm thấy thông tin, nói rõ thay vì bịa."
)

llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)
agent = create_react_agent(llm, tools=[rag_search, calculate], prompt=SYSTEM_PROMPT)

def ask(question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
```

```python
from agent import ask
print(ask("KV cache hoạt động như thế nào?"))
```

## 5. Bước 3 — Backend API (FastAPI)

`server.py` — expose agent qua HTTP, phục vụ Web UI:

```python
# server.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import ask

app = FastAPI(title="Handbook Assistant")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return ChatResponse(answer=ask(req.question))

app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

```bash
uvicorn server:app --reload --port 8080
```

## 6. Bước 4 — Web UI

Giao diện chat tối giản, thuần HTML/JS (không cần framework frontend) — đủ để thử nghiệm và demo:

```html
<!-- static/index.html -->
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>Handbook Assistant</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; }
    #log { border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 300px; margin-bottom: 12px; }
    .msg { margin-bottom: 12px; white-space: pre-wrap; }
    .user { font-weight: 600; }
    #form { display: flex; gap: 8px; }
    #q { flex: 1; padding: 8px; }
  </style>
</head>
<body>
  <h2>Handbook Assistant (Qwen + RAG + Agent, chạy local)</h2>
  <div id="log"></div>
  <form id="form">
    <input id="q" placeholder="Hỏi về LLM, RAG, Vector DB, Agent..." autocomplete="off" />
    <button type="submit">Gửi</button>
  </form>

  <script>
    const log = document.getElementById("log");
    const form = document.getElementById("form");
    const input = document.getElementById("q");

    function append(role, text) {
      const div = document.createElement("div");
      div.className = "msg";
      div.innerHTML = `<span class="${role}">${role === "user" ? "Bạn" : "Assistant"}:</span> ${text}`;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      append("user", question);
      input.value = "";
      append("assistant", "Đang suy nghĩ...");

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      log.lastChild.innerHTML = `<span class="assistant">Assistant:</span> ${data.answer}`;
    });
  </script>
</body>
</html>
```

Mở `http://localhost:8080` — gõ câu hỏi, backend gọi agent (agent tự quyết định có cần `rag_search` không), trả lời hiện ra trên UI.

## 7. Mở rộng: Rust Backend

Giáo trình gốc đề xuất pipeline `... -> LangGraph -> Rust Backend -> Web UI`. Lý do cân nhắc Rust cho backend production: hiệu năng I/O cao hơn (nhiều client đồng thời), binary gọn nhẹ khi deploy, và tận dụng hệ sinh thái Rust nếu phần còn lại của hệ thống (ví dụ ở [13-Rust](../13-Rust/README.md)) đã dùng Rust.

Việc cần làm khi chuyển sang Rust: **giữ nguyên phần Python** (ingest + agent LangGraph) làm một service riêng expose qua HTTP nội bộ (giống `server.py` ở mục 5), rồi thêm một lớp Rust (`axum`) đứng trước làm reverse proxy / gateway kiêm xử lý các phần cần hiệu năng cao (rate limiting, auth, streaming response):

```rust
// Bộ khung tối giản minh họa ý tưởng — chi tiết đầy đủ xem chương 13-Rust
use axum::{routing::post, Json, Router};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct ChatRequest { question: String }

#[derive(Serialize)]
struct ChatResponse { answer: String }

async fn chat(Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    // Gọi sang Python agent service (server.py) qua HTTP nội bộ
    let client = reqwest::Client::new();
    let resp: ChatResponse = client
        .post("http://localhost:8080/api/chat")
        .json(&serde_json::json!({ "question": req.question }))
        .send().await.unwrap()
        .json().await.unwrap();
    Json(resp)
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/api/chat", post(chat));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:9090").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

Đây là kiến trúc thực dụng: không cần viết lại LangGraph/model bằng Rust (hệ sinh thái AI vẫn mạnh nhất ở Python), chỉ cần đặt Rust ở lớp gateway để hưởng lợi về hiệu năng mà không đánh đổi tốc độ phát triển của phần AI. Đi sâu về Rust cho AI (candle, ONNX Runtime bindings...) xem thêm ở [13-Rust](../13-Rust/README.md).

## 8. Checklist triển khai

- [ ] Ollama chạy, đã `pull` model Qwen2.5
- [ ] Qdrant chạy (Docker), collection `handbook` đã được tạo và index dữ liệu (`python ingest.py`)
- [ ] `agent.py` trả lời đúng khi hỏi trực tiếp qua Python REPL (chưa cần qua backend)
- [ ] `server.py` chạy, `POST /api/chat` trả JSON hợp lệ (test bằng `curl`)
- [ ] Web UI load được, gửi/nhận câu hỏi-trả lời qua trình duyệt
- [ ] (tuỳ chọn) Đo retrieval quality bằng bộ eval ở [06-RAG §7](../06-RAG/README.md#7-evaluation) trên chính bộ tài liệu đã ingest

## 9. Bài tập mở rộng

1. Thêm streaming response (`stream=True` ở Ollama) để câu trả lời hiện dần trên UI thay vì chờ xong toàn bộ mới hiển thị.
2. Thêm memory ngắn hạn ([08-AI-Agent §4](../08-AI-Agent/README.md#4-memory)) để agent nhớ được ngữ cảnh hội thoại nhiều lượt trong cùng một phiên chat trên UI.
3. Thêm endpoint `POST /api/ingest` cho phép upload file `.md` mới và tự động chunk/index vào Qdrant mà không cần chạy lại `ingest.py` thủ công.
4. Viết bộ test tự động (10-15 câu hỏi mẫu về nội dung handbook), chạy qua `ask()`, so sánh câu trả lời với đáp án mẫu bằng RAGAS ([06-RAG §7](../06-RAG/README.md#7-evaluation)) — dùng làm regression test mỗi khi đổi prompt hoặc chunk size.
5. (Nâng cao) Cài đặt bản Rust gateway ở [mục 7](#7-mở-rộng-rust-backend) đầy đủ, thêm rate limiting theo IP bằng middleware của `tower`.
