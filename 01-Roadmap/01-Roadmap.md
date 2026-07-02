# 01 — Roadmap

> Chương này trả lời câu hỏi: "Bắt đầu từ đâu, học theo thứ tự nào, và môi trường cần chuẩn bị những gì trước khi đụng vào code LLM thật?" Kết thúc chương, bạn sẽ có môi trường Python + Ollama sẵn sàng và đã tự chạy được một cuộc hội thoại với model Qwen chạy hoàn toàn local.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`01-Roadmap.ipynb`](./01-Roadmap.ipynb).

## Chương 1.1 — Giới thiệu LLM, Roadmap và môi trường

### 01.1.1 — Giới thiệu AI Engineer & LLM Engineer

Hai vai trò này thường bị dùng lẫn lộn nhưng phạm vi công việc khác nhau khá rõ:

| | AI Engineer | LLM Engineer |
|---|---|---|
| Phạm vi | Rộng: ML cổ điển, computer vision, recommendation, cả LLM | Hẹp và sâu hơn: chủ yếu xoay quanh LLM đã có sẵn (không tự train from scratch) |
| Công việc chính | Chọn thuật toán phù hợp bài toán, có thể tự train model từ đầu | Fine-tune (LoRA/QLoRA), xây RAG, xây Agent, tối ưu prompt, serving |
| Kiến thức nền cần sâu | Toán ML, thống kê, nhiều họ thuật toán khác nhau | Kiến trúc Transformer, vector DB, framework agent, hạ tầng serving |
| Ví dụ task điển hình | Xây model dự đoán churn khách hàng bằng gradient boosting | Xây trợ lý hỏi-đáp nội bộ công ty dựa trên Qwen + RAG |

Trong thực tế, ranh giới này rất mờ — nhiều vị trí tuyển dụng ghi "AI Engineer" nhưng công việc thực chất 90% là LLM Engineering. Cuốn sách này tập trung vào nhánh **LLM Engineer**: giả định model nền (Qwen, Llama, GPT...) đã có sẵn, và dạy cách biến nó thành sản phẩm — qua fine-tuning, RAG, agent, và serving production. Các chương ML cổ điển (16-Machine-Learning.md), Diffusion (17-Diffusion.md), Reinforcement Learning (18-Reinforcement-Learning.md) được giữ lại như phần mở rộng cho ai muốn phủ luôn phạm vi AI Engineer đầy đủ hơn.

Công việc hàng ngày của một LLM Engineer thường xoay quanh:
- Thiết kế và tinh chỉnh pipeline RAG (chunking, retrieval, re-ranking, đánh giá chất lượng).
- Xây dựng agent: định nghĩa tool, quản lý luồng ReAct/LangGraph, xử lý lỗi khi model gọi sai tool.
- Đánh giá và so sánh model (benchmark nội bộ, A/B test prompt, đo latency/chi phí).
- Làm việc với hạ tầng serving (Ollama khi dev, vLLM khi production) và theo dõi observability (tracing, cost, chất lượng câu trả lời theo thời gian).

Kỹ năng cần có: Python vững, hiểu cơ chế hoạt động của Transformer ở mức đủ để debug (không cần tự cài đặt training loop từ đầu), quen với ít nhất một vector DB (Qdrant/FAISS), và một framework agent (LangChain/LangGraph).

### 01.1.2 — Bức tranh toàn cảnh của một hệ thống LLM

Một hệ thống LLM production thường tổ chức theo 4 tầng, xếp chồng lên nhau:

```
┌──────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                         │
│  FastAPI / Rust gateway, Web UI, xử lý auth, rate limit    │
└───────────────────────────┬──────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  RAG / AGENT LAYER                                          │
│  Qdrant (retrieval) · LangGraph (điều phối, tool calling)   │
│  ReAct loop, memory, multi-agent khi cần                    │
└───────────────────────────┬──────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  SERVING LAYER                                              │
│  Ollama (dev/local) · vLLM (production, PagedAttention)     │
│  llama.cpp (CPU/máy yếu)                                     │
└───────────────────────────┬──────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  MODEL LAYER                                                │
│  Qwen2.5 / Llama 3 / model đã fine-tune bằng LoRA            │
└──────────────────────────────────────────────────────────┘
```

Đọc từ dưới lên: **Model layer** là bộ trọng số đã train sẵn (hoặc đã fine-tune thêm bằng LoRA — xem 08-Fine-Tuning.md). **Serving layer** biến bộ trọng số đó thành một service có thể gọi qua API (Ollama lúc phát triển, vLLM lúc cần phục vụ nhiều user đồng thời). **RAG/Agent layer** là nơi thêm "trí nhớ" (retrieval tài liệu riêng) và "khả năng hành động" (gọi tool, tra cứu, tính toán) cho model. **Application layer** là lớp người dùng thực sự chạm vào — API, giao diện web, cơ chế xác thực.

Mỗi tầng ứng với nhóm chương tương ứng trong sách: Model layer ↔ 04-Transformer.md, 05-BERT-T5-GPT.md, 08-Fine-Tuning.md; Serving layer ↔ 12-Deployment.md, 13-Rust-AI.md; RAG/Agent layer ↔ 06-Embedding.md, 07-RAG.md, 09-LangChain.md, 10-LangGraph.md, 11-Agent.md; Application layer ↔ 14-Projects.md.

### 01.1.3 — Roadmap học tập

Gợi ý lộ trình theo giai đoạn, tổng thời gian tham khảo 3–6 tháng tuỳ tốc độ và nền tảng sẵn có:

| Giai đoạn | Chương | Thời lượng gợi ý | Mục tiêu đạt được |
|---|---|---|---|
| 1. Nhập môn | 01-Roadmap.md | 2–3 ngày | Môi trường sẵn sàng, chat được với Qwen local |
| 2. Nền tảng toán & DL | 02-Math-for-LLM.md, 03-Deep-Learning.md | 2–3 tuần | Đủ nền để đọc hiểu paper và code training |
| 3. Kiến trúc LLM | 04-Transformer.md, 05-BERT-T5-GPT.md | 2–3 tuần | Hiểu RoPE, KV Cache, GQA/MQA, phân biệt BERT/GPT/T5 |
| 4. Retrieval | 06-Embedding.md, 07-RAG.md | 2 tuần | Xây được pipeline RAG end-to-end với Qdrant |
| 5. Fine-tuning | 08-Fine-Tuning.md | 1–2 tuần | Tự fine-tune model nhỏ bằng LoRA/QLoRA |
| 6. Agent | 09-LangChain.md, 10-LangGraph.md, 11-Agent.md | 3–4 tuần | Xây agent có tool calling, memory, multi-agent |
| 7. Production | 12-Deployment.md, 13-Rust-AI.md | 2 tuần | Serving bằng vLLM, gateway hiệu năng cao bằng Rust |
| 8. Tổng hợp | 14-Projects.md | 1–2 tuần | Ráp toàn bộ thành 1 sản phẩm chạy được từ đầu đến cuối |
| Song song | 15-Papers.md | Xuyên suốt | Đọc bổ sung khi cần hiểu sâu hơn một kỹ thuật cụ thể |
| Mở rộng (tuỳ chọn) | 16 → 21 | Sau khi xong mạch chính | Mở rộng sang ML cổ điển, Diffusion, RL, MLOps, System Design, Python nâng cao |

Nguyên tắc thực dụng: nếu mục tiêu là có sản phẩm chạy được sớm, có thể đọc lướt giai đoạn 2 (Toán/DL) trong lần đầu, quay lại tra cứu khi gặp khái niệm chưa rõ ở giai đoạn 3 trở đi — không nhất thiết phải "học xong toán mới được đụng vào LLM".

### 01.1.4 — Chuẩn bị môi trường phát triển

**1. Cài Python** — khuyến nghị dùng [`uv`](https://docs.astral.sh/uv/) (nhanh hơn pip/venv truyền thống) hoặc `pyenv` nếu cần quản lý nhiều phiên bản Python:

```bash
# Cài uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Tạo virtualenv Python 3.11 và kích hoạt
uv venv --python 3.11
source .venv/bin/activate

# Cài các thư viện dùng xuyên suốt sách
uv pip install ollama huggingface_hub transformers
```

Nếu không dùng `uv`, cách truyền thống với `venv` chuẩn của Python vẫn hoàn toàn tương đương:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ollama huggingface_hub transformers
```

**2. Cài Git** (nếu chưa có) — dùng để clone repo bài tập và theo dõi thay đổi code của chính bạn:

```bash
git --version   # nếu chưa có: https://git-scm.com/downloads
```

**3. Cài Ollama và tải model Qwen2.5**:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: tải installer tại https://ollama.com/download

# Tải model Qwen2.5 7B (khoảng 4-5GB) -- model chính dùng xuyên suốt sách
ollama pull qwen2.5:7b
```

**4. IDE gợi ý**: [Visual Studio Code](https://code.visualstudio.com) với extension Python + Jupyter — đủ để chạy cả file `.py` lẫn mở trực tiếp các notebook `.ipynb` đi kèm mỗi chương.

**5. Kiểm tra môi trường đã sẵn sàng** bằng một script Python nhỏ gọi Ollama:

```python
import ollama

def check_environment() -> None:
    """Gọi thử Ollama -- nếu in ra được câu trả lời, môi trường đã sẵn sàng
    cho toàn bộ code trong sách này."""
    response = ollama.chat(model="qwen2.5:7b", messages=[
        {"role": "user", "content": "Trả lời đúng 1 câu: môi trường đã sẵn sàng chưa?"}
    ])
    print(response["message"]["content"])

if __name__ == "__main__":
    check_environment()
```

Chạy `python check_env.py` — nếu thấy Qwen trả lời (bằng tiếng Việt hoặc tiếng Anh tuỳ prompt), môi trường đã sẵn sàng cho các chương tiếp theo.

## Chương 1.2 — Python, Rust và Hugging Face

Ba mảnh ghép nền tảng của hệ sinh thái AI/LLM hiện nay, mỗi thứ đóng một vai trò riêng, không thay thế nhau:

- **Python** là ngôn ngữ chính cho prototyping, training, và phần lớn thư viện AI (`transformers`, `torch`, `langchain`...). Gần như mọi ví dụ trong sách này viết bằng Python vì hệ sinh thái thư viện phong phú nhất và tốc độ phát triển nhanh nhất.
- **Rust** đóng vai trò ở tầng hạ tầng hiệu năng cao: gateway phục vụ nhiều request đồng thời, xử lý dữ liệu tốc độ cao, nơi độ trễ và an toàn bộ nhớ quan trọng hơn tốc độ viết code. Sách dành hẳn [13-Rust-AI.md](../13-Rust-AI/13-Rust-AI.md) để đi sâu vào việc dùng Rust xây AI Gateway đứng trước các model server.
- **Hugging Face** là "kho chung" của cộng đồng AI: [Hugging Face Hub](https://huggingface.co) lưu trữ hàng trăm nghìn model và dataset mở, còn các thư viện `transformers` (load/chạy model), `datasets` (tải và xử lý dữ liệu), `tokenizers` (tokenize hiệu năng cao) là bộ công cụ chuẩn gần như mọi dự án LLM đều dùng tới ở một mức độ nào đó — kể cả khi serving cuối cùng bằng Ollama/vLLM.

Ví dụ dùng `huggingface_hub` để xem thông tin một model nhỏ, và dùng `AutoTokenizer` để load tokenizer tương ứng (không cần tải toàn bộ trọng số model, tokenizer nhẹ hơn nhiều):

```python
from huggingface_hub import model_info
from transformers import AutoTokenizer

# Xem thông tin model trên Hub mà không cần tải về
info = model_info("Qwen/Qwen2.5-0.5B-Instruct")
print(info.modelId, "-", info.pipeline_tag)

# Tải tokenizer (nhẹ, vài trăm KB -- vài MB) để xem cách model tách token
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokens = tokenizer.tokenize("Xin chào, tôi đang học LLM Engineering")
print(tokens)
print(f"Số token: {len(tokens)}")
```

Điểm cần nhớ: Ollama đóng gói sẵn cả model lẫn tokenizer bên trong file GGUF, nên khi chỉ cần chat qua Ollama (như ở mục 1.5) bạn không cần chạm tới `transformers` — nhưng khi cần tuỳ biến sâu hơn (đếm token chính xác để cắt chunk cho RAG ở [07-RAG.md](../07-RAG/07-RAG.md), hoặc fine-tune ở [08-Fine-Tuning.md](../08-Fine-Tuning/08-Fine-Tuning.md)), `transformers` và Hugging Face Hub là công cụ bắt buộc phải quen.

## Chương 1.3 — CUDA, ONNX, llama.cpp, Ollama

Bốn khái niệm hay xuất hiện khi tìm hiểu cách chạy một LLM, dễ gây nhầm lẫn vì chúng nằm ở các lớp khác nhau của bài toán:

- **CUDA**: nền tảng tính toán song song của NVIDIA cho GPU. Hầu hết training/inference LLM tốc độ cao đều dựa trên CUDA (thông qua PyTorch/`bitsandbytes`...) — nếu không có GPU NVIDIA, các thao tác vẫn chạy được trên CPU nhưng chậm hơn nhiều lần.
- **ONNX** (Open Neural Network Exchange): định dạng model trung gian, cho phép export model đã train (từ PyTorch, TensorFlow...) sang một format chung, rồi chạy trên nhiều runtime khác nhau (ONNX Runtime, trên nhiều loại phần cứng) mà không cần giữ nguyên framework gốc — hữu ích khi cần triển khai model nhỏ (ví dụ embedding model) lên môi trường không có PyTorch.
- **llama.cpp**: engine inference viết bằng C/C++ thuần, tối ưu để chạy LLM trên CPU (và cả GPU) với định dạng quantization riêng gọi là GGUF — nhờ đó chạy được cả trên máy yếu, thậm chí điện thoại.
- **Ollama**: lớp đóng gói thân thiện xây trên nền llama.cpp — quản lý model như Docker image (`ollama pull`, `ollama run`), tự expose REST API tại `localhost:11434`, là cách nhanh nhất để có một LLM chạy local phục vụ phát triển.

Ở giai đoạn này, chỉ cần nắm được **CUDA là hạ tầng phần cứng/driver**, **ONNX là định dạng trung gian giữa các framework**, còn **llama.cpp/Ollama là 2 tầng serving cụ thể** (Ollama nằm trên llama.cpp). Chi tiết kiến trúc bên trong model (RoPE, KV Cache, GQA/MQA — những thứ khiến llama.cpp/vLLM tối ưu được tốc độ) được trình bày sâu ở [04-Transformer.md](../04-Transformer/04-Transformer.md); chi tiết vận hành serving production (so sánh Ollama/vLLM/llama.cpp, observability, governance) được trình bày ở [12-Deployment.md](../12-Deployment/12-Deployment.md).

## Chương 1.4 — Kiến trúc tổng thể của một hệ thống LLM

Chi tiết hoá sơ đồ ở mục 01.1.2 thành luồng dữ liệu cụ thể cho một request hỏi-đáp thực tế, đi qua từng thành phần kỹ thuật:

```
User gõ câu hỏi
      │
      ▼
┌───────────────────────┐
│  Web UI / API client    │
└───────────┬────────────┘
            │ HTTP request
            ▼
┌───────────────────────────────────────────┐
│  Backend (FastAPI hoặc Rust Gateway)        │
│  - xác thực, rate limit, logging             │
└───────────┬─────────────────────────────────┘
            │ gọi agent
            ▼
┌───────────────────────────────────────────┐
│  Agent (LangGraph, ReAct loop)               │
│  - quyết định: trả lời thẳng, hay gọi tool?  │
└───────┬─────────────────────┬────────────────┘
        │ cần tra cứu tài liệu │ cần tính toán/API khác
        ▼                     ▼
┌───────────────────┐   ┌────────────────────┐
│  RAG: embed query   │   │  Tool khác (calc,   │
│  -> Qdrant search   │   │  API nội bộ, MCP)   │
└─────────┬──────────┘   └─────────┬──────────┘
          │ context liên quan       │ kết quả tool
          └───────────┬────────────┘
                       ▼
            ┌───────────────────────┐
            │  LLM server (Ollama/   │
            │  vLLM) sinh câu trả lời │
            └───────────┬────────────┘
                        ▼
                 Trả lời về Backend
                        ▼
                 Hiển thị lên Web UI
```

Điểm mấu chốt: RAG không phải một hệ thống tách biệt với Agent — trong kiến trúc hiện đại, **retrieval chỉ là một trong các tool mà agent có thể chọn gọi**, y hệt như một tool tính toán hay gọi API khác. Toàn bộ kiến trúc này được hiện thực hoá đầy đủ, có code chạy được từng bước, ở [14-Projects.md](../14-Projects/14-Projects.md) — nơi bạn sẽ ráp chính xác các thành phần trong sơ đồ trên thành một ứng dụng hoàn chỉnh.

## Chương 1.5 — Dự án đầu tiên (Chat với Qwen local)

Áp dụng ngay phần môi trường đã chuẩn bị ở mục 01.1.4 để có trải nghiệm đầu tiên: một script Python chat được với Qwen2.5 chạy hoàn toàn trên máy, không gọi API nào ra ngoài.

**Bước 1 — đảm bảo Ollama đang chạy và đã có model** (đã làm ở mục 01.1.4, kiểm tra lại):

```bash
ollama list          # phải thấy qwen2.5:7b trong danh sách
ollama pull qwen2.5:7b   # nếu chưa có
```

**Bước 2 — viết script chat**, lưu thành `chat_qwen.py`:

```python
import ollama

MODEL = "qwen2.5:7b"

def chat_once(prompt: str) -> str:
    """Gửi 1 câu hỏi tới Qwen chạy local qua Ollama, trả về câu trả lời."""
    response = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": "Bạn là trợ lý kỹ thuật, trả lời ngắn gọn bằng tiếng Việt."},
        {"role": "user", "content": prompt},
    ])
    return response["message"]["content"]

def main() -> None:
    print(f"Đang chat với {MODEL} (gõ 'exit' để thoát)")
    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() == "exit":
            break
        answer = chat_once(user_input)
        print(f"Qwen: {answer}\n")

if __name__ == "__main__":
    main()
```

**Bước 3 — chạy thử**:

```bash
python chat_qwen.py
```

```
Đang chat với qwen2.5:7b (gõ 'exit' để thoát)
Bạn: KV Cache trong LLM là gì?
Qwen: KV Cache lưu lại Key/Value đã tính của các token trước đó, giúp sinh
token mới không cần tính lại attention cho toàn bộ chuỗi, tăng tốc đáng kể
quá trình sinh văn bản autoregressive.
```

Chỉ với ~15 dòng code, bạn đã có một chatbot chạy hoàn toàn local — không cần API key, không tốn phí theo token, dữ liệu không rời khỏi máy. Đây chính là nền móng sẽ được mở rộng dần qua các chương sau: thêm RAG để trả lời dựa trên tài liệu riêng (07-RAG.md), thêm khả năng gọi tool (11-Agent.md), rồi đóng gói thành API production (12-Deployment.md, 14-Projects.md).

## Bài tập

1. Tự cài đặt môi trường theo mục 01.1.4 và chạy thành công dự án ở Chương 1.5 trên máy của bạn.
2. Sửa `chat_qwen.py` để dùng một model khác đã pull về (ví dụ `llama3.1:8b` hoặc `qwen2.5:0.5b` nếu máy yếu), so sánh tốc độ trả lời và chất lượng câu trả lời cho cùng 3 câu hỏi kỹ thuật.
3. Dùng `AutoTokenizer` ở Chương 1.2 đếm số token của 5 câu tiếng Việt khác nhau, nhận xét xem tiếng Việt có tokenize "tốn" token hơn tiếng Anh không.

## Tài liệu tham khảo

- [Ollama docs](https://ollama.com) · [Ollama GitHub](https://github.com/ollama/ollama)
- [Hugging Face Hub docs](https://huggingface.co/docs/hub/index) · [`transformers` docs](https://huggingface.co/docs/transformers/index) · [`huggingface_hub` docs](https://huggingface.co/docs/huggingface_hub/index)
- [uv docs](https://docs.astral.sh/uv/) — công cụ quản lý môi trường Python dùng trong chương này
- [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX](https://onnx.ai)
