# 16 — Projects: Dự án tổng hợp

> Chương này là nơi tập hợp các dự án capstone, mỗi dự án ráp nhiều chương lý thuyết trước đó lại thành một ứng dụng chạy được từ đầu đến cuối. Mỗi dự án có file `.md` + notebook `.ipynb` riêng — không gộp chung vào 1 file để dễ mở rộng thêm dự án mới khi các chương khác của handbook được viết tiếp.

## Danh sách dự án

| # | Dự án | Ráp từ các chương | Xem chi tiết |
|---|---|---|---|
| 1 | **Trợ lý RAG + Agent chạy local** — Qwen (Ollama) → BGE → Qdrant → RAG → LangGraph agent → FastAPI → Web UI | [05-LLM](../05-LLM/README.md), [07-Vector-Database](../07-Vector-Database/README.md), [06-RAG](../06-RAG/README.md), [08-AI-Agent](../08-AI-Agent/README.md) | [01-rag-agent-assistant.md](01-rag-agent-assistant.md) · [notebook](01-rag-agent-assistant.ipynb) |

## Dự án sắp tới

Danh sách này sẽ có thêm dự án mới khi các chương nền tảng liên quan được viết xong, ví dụ:

- **Dự án multimodal** (ráp thêm [09-Diffusion](../09-Diffusion/README.md)): agent có thể vừa trả lời bằng text vừa gọi tool sinh ảnh minh họa.
- **Dự án fine-tune bằng RL** (ráp thêm [10-Reinforcement-Learning](../10-Reinforcement-Learning/README.md)): áp dụng RLHF/DPO lên chính model dùng trong Dự án 1.
- **Dự án production-ready** (ráp thêm [11-MLOps](../11-MLOps/README.md), [12-System-Design](../12-System-Design/README.md), [13-Rust](../13-Rust/README.md)): triển khai Dự án 1 với CI/CD, observability, và gateway viết bằng Rust thay vì chỉ phác thảo như ở [01-rag-agent-assistant.md §7](01-rag-agent-assistant.md#7-mở-rộng-rust-backend).

## Quy ước đặt tên

Mỗi dự án mới thêm vào chương này theo mẫu:

```
16-Projects/
├── README.md                          # trang mục lục này
├── 01-rag-agent-assistant.md          # nội dung dự án 1
├── 01-rag-agent-assistant.ipynb       # notebook đi kèm dự án 1
├── 02-<ten-du-an-2>.md
├── 02-<ten-du-an-2>.ipynb
└── ...
```
