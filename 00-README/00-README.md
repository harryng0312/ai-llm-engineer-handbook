# AI/LLM Engineer Handbook

> Cuốn sách này trả lời một câu hỏi thực dụng: "Tôi biết lập trình, muốn trở thành AI/LLM Engineer — nên học gì, theo thứ tự nào, và làm sao để tự tay chạy được một hệ thống LLM thật (không chỉ gọi API)?" Toàn bộ nội dung được viết bằng tiếng Việt, ưu tiên code chạy được thật hơn là lý thuyết suông, và xoay quanh việc tự host model bằng Ollama/vLLM/llama.cpp thay vì chỉ phụ thuộc API trả phí.

## Cuốn sách này dành cho ai

- Người đã có nền tảng lập trình (biết Python cơ bản, dùng được terminal/git), muốn chuyển hướng hoặc đào sâu sang AI/LLM Engineering.
- Người đang làm ML/Backend/Data muốn hiểu "bên trong" LLM hoạt động thế nào, thay vì chỉ gọi `openai.chat.completions.create(...)`.
- Người muốn tự xây một sản phẩm có AI (trợ lý RAG, agent tự động hoá) chạy được trên máy cá nhân, không phụ thuộc hạ tầng cloud đắt đỏ.

Sách **không** dạy lại từ số 0 kiến thức lập trình hay xác suất thống kê — các chương nền tảng (Toán, Machine Learning, Reinforcement Learning, Deep Learning) được đưa vào để tra cứu khi cần, không phải điều kiện bắt buộc phải học kỹ trước.

## Bức tranh tổng quan: một hệ thống LLM hiện đại

```
┌───────────────────┐
│   DATA             │   text, code, tài liệu nội bộ, hội thoại...
└─────────┬──────────┘
          ▼
┌───────────────────┐
│   TOKENIZE / EMBED │   BPE tokenizer (Hugging Face `tokenizers`),
└─────────┬──────────┘   embedding model (BGE/E5/GTE) cho retrieval
          ▼
┌───────────────────┐
│   MODEL            │   Decoder-only Transformer: Qwen, Llama, GPT...
└─────────┬──────────┘   (kiến trúc ở 06-Transformer.md, 07-BERT-T5-GPT.md)
          ▼
┌───────────────────┐
│   FINE-TUNE        │   LoRA/QLoRA — "nướng" domain/style riêng vào model
└─────────┬──────────┘   khi prompt engineering không đủ (10-Fine-Tuning.md)
          ▼
┌───────────────────┐
│   SERVE             │   Ollama (dev/local) · vLLM (production, PagedAttention)
└─────────┬──────────┘   llama.cpp (CPU/máy yếu) — 14-Deployment.md
          ▼
┌───────────────────┐
│   RAG / AGENT       │   Qdrant + retriever + re-ranking (09-RAG.md)
└─────────┬──────────┘   LangChain/LangGraph, tool calling, multi-agent, MCP
          ▼               (11-LangChain.md, 12-LangGraph.md, 13-Agent.md)
┌───────────────────┐
│   APPLICATION       │   Backend API (FastAPI/Rust gateway), Web UI
└─────────┬──────────┘   (15-Rust-AI.md, 16-Projects.md)
          ▼
       Người dùng
```

Mỗi tầng trong sơ đồ trên tương ứng với một (hoặc vài) chương trong sách — bạn sẽ đi từ trên xuống dưới, xây dần từng lớp cho tới khi ráp được toàn bộ pipeline trong chương Dự án.

## Mục lục

| # | Chương | Nội dung |
|---|---|---|
| 00 | [00-README.md](../00-README/00-README.md) | Trang này — tổng quan và mục lục toàn sách |
| 01 | [01-Roadmap.md](../01-Roadmap/01-Roadmap.md) | Lộ trình học, chuẩn bị môi trường, dự án chat local đầu tiên |
| 02 | [02-Math-for-LLM.md](../02-Math-for-LLM/02-Math-for-LLM.md) | Toán nền tảng cho LLM: đại số tuyến tính, xác suất, tối ưu hoá *(placeholder)* |
| 03 | [03-Machine-Learning.md](../03-Machine-Learning/03-Machine-Learning.md) | Machine Learning cổ điển: hồi quy, cây quyết định, ensemble *(placeholder)* |
| 04 | [04-Reinforcement-Learning.md](../04-Reinforcement-Learning/04-Reinforcement-Learning.md) | Reinforcement Learning, RLHF *(placeholder)* |
| 05 | [05-Deep-Learning.md](../05-Deep-Learning/05-Deep-Learning.md) | Deep Learning nền tảng: mạng nơ-ron, backprop, CNN, RNN/LSTM, word embeddings, seq2seq & attention |
| 06 | [06-Transformer.md](../06-Transformer/06-Transformer.md) | Kiến trúc Transformer, Decoder-only LLM: RoPE, KV Cache, GQA/MQA, Prompt Engineering |
| 07 | [07-BERT-T5-GPT.md](../07-BERT-T5-GPT/07-BERT-T5-GPT.md) | Ba dòng kiến trúc nền tảng: BERT (encoder), GPT (decoder), T5 (encoder-decoder) |
| 08 | [08-Embedding.md](../08-Embedding/08-Embedding.md) | Contrastive Learning, Triplet Loss, BGE/E5/GTE, FAISS, Qdrant |
| 09 | [09-RAG.md](../09-RAG/09-RAG.md) | Chunking, Retriever, Re-ranking, Evaluation cho Retrieval-Augmented Generation |
| 10 | [10-Fine-Tuning.md](../10-Fine-Tuning/10-Fine-Tuning.md) | Fine-tune LLM tiết kiệm tài nguyên: LoRA, QLoRA |
| 11 | [11-LangChain.md](../11-LangChain/11-LangChain.md) | LangChain: định nghĩa tool, structured output, tích hợp LLM/vector DB |
| 12 | [12-LangGraph.md](../12-LangGraph/12-LangGraph.md) | LangGraph: StateGraph, Checkpoint, Human-in-the-loop |
| 13 | [13-Agent.md](../13-Agent/13-Agent.md) | ReAct, Tool Calling, Memory, Multi-Agent, MCP, Evaluation, Security |
| 14 | [14-Deployment.md](../14-Deployment/14-Deployment.md) | Serving (Ollama/vLLM/llama.cpp), Observability, Platform & Governance |
| 15 | [15-Rust-AI.md](../15-Rust-AI/15-Rust-AI.md) | Rust cho AI Gateway — serving hiệu năng cao |
| 16 | [16-Projects.md](../16-Projects/16-Projects.md) | Dự án tổng hợp: Trợ lý RAG + Agent chạy local, ráp toàn bộ pipeline |
| 17 | [17-Papers.md](../17-Papers/17-Papers.md) | Danh sách paper nên đọc, có chú giải vì sao mỗi paper quan trọng |
| 18 | [18-Diffusion.md](../18-Diffusion/18-Diffusion.md) | Diffusion Models — nền tảng sinh ảnh/video *(placeholder)* |
| 19 | [19-MLOps.md](../19-MLOps/19-MLOps.md) | MLOps: pipeline train/deploy/monitor model *(placeholder)* |
| 20 | [20-System-Design.md](../20-System-Design/20-System-Design.md) | System Design cho hệ thống AI quy mô lớn *(placeholder)* |
| 21 | [21-Python.md](../21-Python/21-Python.md) | Kỹ năng Python cho AI Engineer: async, typing, packaging *(placeholder)* |

Các chương đánh dấu *(placeholder)* hiện chỉ có khung sườn, sẽ được bổ sung nội dung đầy đủ ở các đợt cập nhật sau — không ảnh hưởng tới mạch chính 06→16 (Transformer → Dự án) vốn đã đầy đủ.

## Cách dùng sách

- **Đọc theo thứ tự số**: các chương được sắp để chương sau dùng lại khái niệm/code của chương trước (ví dụ 09-RAG.md dùng embedding đã học ở 08-Embedding.md, 13-Agent.md dùng LangGraph đã học ở 12-LangGraph.md). Nếu đã có nền tảng, có thể nhảy thẳng tới chương cần thiết — mỗi chương đều ghi rõ chương nào cần đọc trước.
- **Mỗi file `.md` có một file `.ipynb` cùng tên đi kèm** (trừ 15-Rust-AI.md, vì Rust không chạy trong notebook Python) — mở notebook để chạy thử trực tiếp mọi đoạn code trong chương thay vì copy tay từng đoạn.
- **Các chương placeholder (02, 03, 04, 18–21)** hiện là khung mục lục, sẽ được viết đầy đủ sau; có thể bỏ qua trong lần đọc đầu nếu mục tiêu là làm được sản phẩm LLM nhanh nhất (bám theo mạch 01 → 06 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 16).
- **Chương 17-Papers.md** không phải để đọc tuần tự mà để tra cứu — mỗi chương kỹ thuật đều trỏ tới phần liên quan trong 17-Papers.md ở mục "Tài liệu tham khảo".

## Yêu cầu tiên quyết

- Biết lập trình Python ở mức cơ bản (hàm, class, list/dict comprehension, `pip`/`venv`).
- Biết dùng terminal (di chuyển thư mục, chạy script, biến môi trường) và git ở mức cơ bản (clone, commit, branch).
- Có một máy tính chạy được [Ollama](https://ollama.com) — CPU vẫn chạy được model nhỏ (7B lượng tử hoá), có GPU (kể cả GPU laptop 8GB VRAM trở lên) sẽ nhanh hơn đáng kể; các ví dụ nặng hơn (fine-tune, vLLM) sẽ ghi rõ yêu cầu GPU riêng.
- Không bắt buộc biết Rust trước khi đọc 15-Rust-AI.md — chương đó tự giới thiệu từ đầu trong phạm vi cần cho AI Gateway.

Chương tiếp theo: [01-Roadmap.md](../01-Roadmap/01-Roadmap.md) — lộ trình học chi tiết và dự án chat local đầu tiên.
