# 14 — Deployment: Serving, Observability, Platform

> Các chương trước xây model ([06-Transformer.md](../06-Transformer/06-Transformer.md), [10-Fine-Tuning.md](../10-Fine-Tuning/10-Fine-Tuning.md)) và agent ([13-Agent.md](../13-Agent/13-Agent.md)) — nhưng chạy được trên máy dev là một chuyện, đưa vào production là chuyện khác. Chương này bàn về ba mối quan tâm khi đưa hệ thống LLM/agent ra thực tế: **serving** (chạy model hiệu quả, phục vụ nhiều request), **observability** (biết được hệ thống đang làm gì, debug khi có sự cố), và các mối quan tâm ở tầng **platform** khi hệ thống phục vụ nhiều team/tenant thay vì một ứng dụng đơn lẻ.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`14-Deployment.ipynb`](./14-Deployment.ipynb).

## Mục lục

1. [Chương 14.1 — Serving: Ollama, vLLM, llama.cpp](#chương-141--serving-ollama-vllm-llamacpp)
2. [Chương 14.2 — Observability](#chương-142--observability)
3. [Chương 14.3 — Agent Platform và Governance](#chương-143--agent-platform-và-governance)
4. [Chương 14.4 — Bài tập](#chương-144--bài-tập)
5. [Chương 14.5 — Tài liệu tham khảo](#chương-145--tài-liệu-tham-khảo)

---

## Chương 14.1 — Serving: Ollama, vLLM, llama.cpp

Ba công cụ phổ biến để chạy LLM, mỗi cái phù hợp một use case:

| Công cụ | Phù hợp khi | Đặc điểm |
|---|---|---|
| **Ollama** | Chạy local để dev/thử nghiệm nhanh | Đóng gói model + runtime (dựa trên llama.cpp), API REST đơn giản, quản lý model như Docker image |
| **llama.cpp** | Chạy trên CPU/máy yếu, cần kiểm soát chi tiết | C/C++ thuần, hỗ trợ quantization GGUF (Q4_K_M, Q5_K_M...), chạy được trên điện thoại/Raspberry Pi |
| **vLLM** | Serving production, nhiều user đồng thời | PagedAttention (quản lý KV cache hiệu quả như virtual memory của OS), continuous batching, throughput cao |

### Cài Ollama và chạy Qwen local

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Tải model Qwen2.5 7B (khoảng 4-5GB, dùng cho các ví dụ trong sách này)
ollama pull qwen2.5:7b

# Chat thử qua CLI
ollama run qwen2.5:7b "Giải thích KV cache trong 3 câu"

# Ollama tự chạy server tại http://localhost:11434
```

Gọi bằng Python với thư viện chính thức `ollama`:

```python
import ollama

response = ollama.chat(model="qwen2.5:7b", messages=[
    {"role": "user", "content": "Tóm tắt sự khác biệt giữa MHA, GQA, MQA trong 3 gạch đầu dòng."}
])
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(model="qwen2.5:7b", messages=[
    {"role": "user", "content": "Viết một đoạn code Python đọc file CSV."}
], stream=True):
    print(chunk["message"]["content"], end="", flush=True)
```

### vLLM (tham khảo, cần GPU)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_compatible_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

vLLM expose một API tương thích OpenAI, nên có thể dùng lại SDK `openai` trỏ `base_url` về server local — hữu ích khi muốn tái sử dụng code viết cho OpenAI API nhưng chạy trên model self-host.

## Chương 14.2 — Observability

Khi agent chạy nhiều bước qua nhiều tool, một câu trả lời sai có thể bắt nguồn từ bất kỳ bước nào trong chuỗi — không có observability, debug gần như phải đoán. **Tracing** ghi lại toàn bộ chuỗi: prompt đã gửi, tool đã gọi cùng tham số, kết quả trả về, thời gian mỗi bước.

Cách nhanh nhất với hệ sinh thái LangChain/LangGraph: bật [LangSmith](https://smith.langchain.com) qua biến môi trường, không cần sửa code:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-key>
export LANGCHAIN_PROJECT=handbook-agent
```

Sau khi bật, mọi lệnh gọi `agent.invoke(...)` tự động gửi trace lên dashboard LangSmith — xem được cây gọi đầy đủ (agent → tool → agent → ...), token dùng ở mỗi bước, và độ trễ từng node trong graph.

Muốn tự host, không phụ thuộc dịch vụ ngoài, dùng chuẩn mở **OpenTelemetry**:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
tracer = trace.get_tracer(__name__)

def ask_with_tracing(question: str) -> str:
    with tracer.start_as_current_span("agent_invoke") as span:
        span.set_attribute("question", question)
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer = result["messages"][-1].content
        span.set_attribute("answer_length", len(answer))
        return answer
```

Ngoài trace, nên theo dõi thêm **cost/token usage** theo thời gian (số token input/output mỗi request, quy đổi chi phí nếu dùng model trả phí) — đây là cách duy nhất phát hiện sớm khi một thay đổi prompt (ví dụ nhét thêm quá nhiều context) âm thầm làm tăng chi phí vận hành gấp nhiều lần.

## Chương 14.3 — Agent Platform và Governance

Khi một agent phục vụ nhiều team/khách hàng thay vì 1 ứng dụng đơn lẻ, xuất hiện thêm các mối quan tâm ở tầng hạ tầng — đây là các khái niệm cần biết khi hệ thống lớn dần, không phải điểm bắt đầu cho một dự án cá nhân:

**Model Routing** — không phải mọi câu hỏi đều cần model mạnh nhất (và đắt nhất); một router nhẹ quyết định dùng model local rẻ hay model API mạnh hơn tùy độ phức tạp:

```python
def route_model(question: str) -> str:
    """Heuristic đơn giản: câu hỏi ngắn/tra cứu fact dùng model local miễn phí,
    câu hỏi cần suy luận sâu mới trả thêm chi phí cho model mạnh hơn."""
    needs_deep_reasoning = len(question.split()) > 40 or any(
        kw in question.lower() for kw in ["phân tích", "so sánh chi tiết", "thiết kế"]
    )
    return "gpt-4o" if needs_deep_reasoning else "qwen2.5:7b"
```

**AI Gateway** — một lớp đứng trước mọi request tới LLM/agent, đảm nhiệm auth, rate limiting, logging tập trung — giống vai trò của lớp Rust gateway đã phác thảo ở [15-Rust-AI.md](../15-Rust-AI/15-Rust-AI.md), chỉ khác là ở quy mô nhiều team/nhiều model hơn.

**Enterprise RAG đa tenant** — nhiều khách hàng/phòng ban dùng chung hạ tầng vector DB nhưng dữ liệu phải cách ly tuyệt đối. Cách làm phổ biến: gắn `tenant_id` vào payload mọi điểm dữ liệu, **bắt buộc** filter theo `tenant_id` trong mọi câu query, và `tenant_id` phải lấy từ thông tin xác thực (token đã verify) — **không bao giờ** tin `tenant_id` do client tự gửi lên:

```python
def search_for_tenant(query_vector: list[float], tenant_id: str, client: QdrantClient):
    """tenant_id PHẢI đến từ session/token đã xác thực ở tầng gateway,
    không được nhận trực tiếp từ request body -- nếu không, 1 tenant có thể
    giả mạo tenant_id để đọc dữ liệu của tenant khác."""
    return client.search(
        collection_name="shared_knowledge_base",
        query_vector=query_vector,
        query_filter=Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]),
        limit=5,
    )
```

**Governance** — ở quy mô doanh nghiệp, cần trả lời được các câu hỏi tuân thủ (GDPR, SOC2...): "dữ liệu người dùng X được lưu ở đâu, xóa được không (right to be forgotten)?", "ai đã gọi tool nào, khi nào (audit log)?". Checklist tối thiểu đáng áp dụng ngay cả ở dự án nhỏ, không cần chờ đến quy mô doanh nghiệp:

- Ghi audit log mọi lời gọi tool có tác dụng phụ (ai, khi nào, tham số gì, kết quả gì).
- Có cách xóa toàn bộ dữ liệu (vector + log) gắn với 1 user cụ thể khi được yêu cầu.
- Định kỳ review log để phát hiện dấu hiệu prompt injection đã lọt qua guardrail ở [13-Agent.md](../13-Agent/13-Agent.md).

Việc triển khai đầy đủ Model Routing/AI Gateway/Governance cho một nền tảng nhiều tenant là chủ đề riêng của kiến trúc hệ thống doanh nghiệp — mục này chỉ nhằm giúp nhận diện đúng vấn đề khi dự án cá nhân ở [16-Projects.md](../16-Projects/16-Projects.md) phát triển vượt quy mô một ứng dụng đơn lẻ.

## Chương 14.4 — Bài tập

1. Sửa `generate_with_cache` ở [06-Transformer.md](../06-Transformer/06-Transformer.md) để in ra số token/giây, so sánh với `generate_naive` khi `n_new_tokens` = 20, 50, 100 — vẽ biểu đồ tốc độ theo độ dài chuỗi.
2. Cài Ollama, pull 2 model khác nhau (ví dụ `qwen2.5:7b` và `llama3.1:8b`), viết script so sánh câu trả lời cho cùng 5 câu hỏi kỹ thuật.
3. Bật LangSmith tracing (xem [Chương 14.2](#chương-142--observability)) cho một agent đã xây ở [13-Agent.md](../13-Agent/13-Agent.md), chạy vài câu hỏi cần gọi 2-3 tool, quan sát trace trên dashboard: đếm số bước, so sánh độ trễ giữa các node, xác định node nào chậm nhất.
4. Viết một model router đơn giản (mở rộng `route_model` ở [Chương 14.3](#chương-143--agent-platform-và-governance)) chọn giữa 2 model dựa trên độ dài/độ phức tạp câu hỏi, thử với 10 câu hỏi khác nhau và tự đánh giá router có chọn hợp lý không.

## Chương 14.5 — Tài liệu tham khảo

> 📖 Xem chú giải chi tiết ở [17-Papers.md](../17-Papers/17-Papers.md).

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023)
- [Ollama docs](https://github.com/ollama/ollama) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [vLLM docs](https://docs.vllm.ai)
