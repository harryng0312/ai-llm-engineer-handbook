# 16 — Projects: Dự án tổng hợp

Chương này tập hợp các dự án capstone, mỗi dự án ráp nhiều chương lý thuyết trước đó thành một ứng dụng chạy được từ đầu đến cuối. Hiện có 1 dự án; các dự án tương lai sẽ được thêm dưới dạng "Chương 16.2", "Chương 16.3"... ngay trong chính file này khi các chương nền tảng liên quan được viết xong.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`16-Projects.ipynb`](./16-Projects.ipynb).

## Mục lục

1. [Chương 16.1 — Dự án 1: Trợ lý RAG + Agent chạy local](#chương-161--dự-án-1-trợ-lý-rag--agent-chạy-local)
   - [16.1.1 — Mục tiêu và kiến trúc](#1611--mục-tiêu-và-kiến-trúc)
   - [16.1.2 — Chuẩn bị môi trường](#1612--chuẩn-bị-môi-trường)
   - [16.1.3 — Bước 1 — Ingest tài liệu vào Qdrant](#1613--bước-1--ingest-tài-liệu-vào-qdrant)
   - [16.1.4 — Bước 2 — Agent kết hợp RAG + tool](#1614--bước-2--agent-kết-hợp-rag--tool)
   - [16.1.5 — Bước 3 — Backend API (FastAPI)](#1615--bước-3--backend-api-fastapi)
   - [16.1.6 — Bước 4 — Web UI](#1616--bước-4--web-ui)
   - [16.1.7 — Mở rộng: Rust Backend](#1617--mở-rộng-rust-backend)
   - [16.1.8 — Checklist triển khai](#1618--checklist-triển-khai)
   - [16.1.9 — Bài tập mở rộng](#1619--bài-tập-mở-rộng)
2. [Chương 16.2 — Dự án sắp tới](#chương-162--dự-án-sắp-tới)

---

## Chương 16.1 — Dự án 1: Trợ lý RAG + Agent chạy local

> Dự án này ráp toàn bộ kiến thức từ [14-Deployment.md](../14-Deployment/14-Deployment.md), [08-Embedding.md](../08-Embedding/08-Embedding.md), [09-RAG.md](../09-RAG/09-RAG.md) và [13-Agent.md](../13-Agent/13-Agent.md) thành một ứng dụng chạy được từ đầu đến cuối: pipeline `Qwen -> BGE -> Qdrant -> RAG -> LangGraph -> Backend -> Web UI`.

> 📓 Toàn bộ code ví dụ trong dự án này cũng có ở dạng notebook chạy được: [`16-Projects.ipynb`](./16-Projects.ipynb).

---

### 16.1.1 — Mục tiêu và kiến trúc

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
| Qwen2.5 (qua Ollama) | LLM chính, vừa sinh câu trả lời vừa quyết định gọi tool | [14-Deployment.md](../14-Deployment/14-Deployment.md) |
| BGE (`bge-small`) | Embedding model để chunk hóa tài liệu và query | [08-Embedding.md](../08-Embedding/08-Embedding.md) |
| Qdrant | Lưu trữ vector, hỗ trợ filter theo nguồn tài liệu | [08-Embedding.md](../08-Embedding/08-Embedding.md) |
| Chunking + retrieval | Chuẩn bị dữ liệu và truy xuất ngữ cảnh | [09-RAG.md](../09-RAG/09-RAG.md) |
| LangGraph agent | Vòng lặp ReAct, gọi `rag_search` như một tool | [12-LangGraph.md](../12-LangGraph/12-LangGraph.md) |

### 16.1.2 — Chuẩn bị môi trường

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

### 16.1.3 — Bước 1 — Ingest tài liệu vào Qdrant

`ingest.py` — đọc mọi file `.md` trong thư mục `docs/`, chunk, embed bằng BGE, lưu vào Qdrant (áp dụng trực tiếp kỹ thuật ở [09-RAG.md](../09-RAG/09-RAG.md)):

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

### 16.1.4 — Bước 2 — Agent kết hợp RAG + tool

`agent.py` — biến retrieval thành một **tool** (`rag_search`) để agent tự quyết định khi nào cần tra cứu tài liệu, kết hợp thêm một tool tiện ích (`calculate`), dùng LangGraph ReAct agent như ở [11-LangChain.md](../11-LangChain/11-LangChain.md):

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

### 16.1.5 — Bước 3 — Backend API (FastAPI)

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

### 16.1.6 — Bước 4 — Web UI

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

### 16.1.7 — Mở rộng: Rust Backend

Giáo trình gốc đề xuất pipeline `... -> LangGraph -> Rust Backend -> Web UI`. Lý do cân nhắc Rust cho backend production: hiệu năng I/O cao hơn (nhiều client đồng thời), binary gọn nhẹ khi deploy, và tận dụng hệ sinh thái Rust nếu phần còn lại của hệ thống (ví dụ ở [15-Rust-AI.md](../15-Rust-AI/15-Rust-AI.md)) đã dùng Rust.

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

Đây là kiến trúc thực dụng: không cần viết lại LangGraph/model bằng Rust (hệ sinh thái AI vẫn mạnh nhất ở Python), chỉ cần đặt Rust ở lớp gateway để hưởng lợi về hiệu năng mà không đánh đổi tốc độ phát triển của phần AI. Đi sâu về Rust cho AI (candle, ONNX Runtime bindings...) xem thêm ở [15-Rust-AI.md](../15-Rust-AI/15-Rust-AI.md).

### 16.1.8 — Checklist triển khai

- [ ] Ollama chạy, đã `pull` model Qwen2.5
- [ ] Qdrant chạy (Docker), collection `handbook` đã được tạo và index dữ liệu (`python ingest.py`)
- [ ] `agent.py` trả lời đúng khi hỏi trực tiếp qua Python REPL (chưa cần qua backend)
- [ ] `server.py` chạy, `POST /api/chat` trả JSON hợp lệ (test bằng `curl`)
- [ ] Web UI load được, gửi/nhận câu hỏi-trả lời qua trình duyệt
- [ ] (tuỳ chọn) Đo retrieval quality bằng bộ eval ở [09-RAG.md](../09-RAG/09-RAG.md) trên chính bộ tài liệu đã ingest

### 16.1.9 — Bài tập mở rộng

1. Thêm streaming response (`stream=True` ở Ollama) để câu trả lời hiện dần trên UI thay vì chờ xong toàn bộ mới hiển thị.
2. Thêm memory ngắn hạn ([13-Agent.md](../13-Agent/13-Agent.md)) để agent nhớ được ngữ cảnh hội thoại nhiều lượt trong cùng một phiên chat trên UI.
3. Thêm endpoint `POST /api/ingest` cho phép upload file `.md` mới và tự động chunk/index vào Qdrant mà không cần chạy lại `ingest.py` thủ công.
4. Viết bộ test tự động (10-15 câu hỏi mẫu về nội dung handbook), chạy qua `ask()`, so sánh câu trả lời với đáp án mẫu bằng RAGAS ([09-RAG.md](../09-RAG/09-RAG.md)) — dùng làm regression test mỗi khi đổi prompt hoặc chunk size.
5. (Nâng cao) Cài đặt bản Rust gateway ở [16.1.7](#1617--mở-rộng-rust-backend) đầy đủ, thêm rate limiting theo IP bằng middleware của `tower`.

---

## Chương 16.2 — Dự án sắp tới

Danh sách này sẽ có thêm dự án mới khi các chương nền tảng liên quan được viết xong, ví dụ:

- **Dự án multimodal** (ráp thêm [18-Diffusion.md](../18-Diffusion/18-Diffusion.md)): agent có thể vừa trả lời bằng text vừa gọi tool sinh ảnh minh họa.
- **Dự án fine-tune bằng RL** (ráp thêm [04-Reinforcement-Learning.md](../04-Reinforcement-Learning/04-Reinforcement-Learning.md)): áp dụng RLHF/DPO lên chính model dùng trong Dự án 1.
- **Dự án production-ready** (ráp thêm [19-MLOps.md](../19-MLOps/19-MLOps.md), [20-System-Design.md](../20-System-Design/20-System-Design.md), [15-Rust-AI.md](../15-Rust-AI/15-Rust-AI.md)): triển khai Dự án 1 với CI/CD, observability, và gateway viết bằng Rust thay vì chỉ phác thảo như ở [16.1.7](#1617--mở-rộng-rust-backend).

**Quy ước đặt tên:** vì toàn bộ chương này giờ nằm trong 1 file phẳng duy nhất (`16-Projects.md`), mỗi dự án mới **không còn tạo file `.md` riêng** như trước nữa. Thay vào đó, thêm một heading mới `## Chương 16.N — Tên dự án` (N tăng dần: 14.3, 14.4...) vào cuối chính file `16-Projects.md` này, cùng các mục con `### 14.N.1`, `### 14.N.2`... theo đúng quy ước đánh số đã dùng ở Chương 16.1. Đừng quên cập nhật lại mục "Mục lục" ở đầu file để trỏ tới các mục con mới. Code ví dụ đi kèm vẫn có thể được thêm vào chung notebook [`16-Projects.ipynb`](./16-Projects.ipynb).
