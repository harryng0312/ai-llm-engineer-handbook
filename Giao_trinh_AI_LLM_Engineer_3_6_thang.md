# AI/LLM Engineer Roadmap (3--6 tháng)

Đây là giáo trình tóm tắt.

## 1. Decoder-only LLM

-   Kiến trúc Transformer Decoder
-   RoPE
-   KV Cache
-   GQA/MQA
-   Prompt Engineering
-   LoRA/QLoRA
-   Ollama, vLLM, llama.cpp

## 2. Embedding

-   Contrastive Learning
-   Triplet Loss
-   BGE/E5/GTE
-   FAISS/Qdrant

## 3. RAG

-   Chunking
-   Embedding
-   Retriever
-   Re-ranking
-   Evaluation

## 4. Agent

-   LangChain
-   LangGraph
-   Tool Calling
-   Memory
-   Multi-Agent

## Dự án

Qwen -\> BGE -\> Qdrant -\> RAG -\> LangGraph -\> Rust Backend -\> Web
UI

## Structure:
AI-LLM-Engineer/
│
├── 00-README
├── 01-Roadmap
├── 02-Math-for-LLM
├── 03-Deep-Learning
├── 04-Transformer
├── 05-BERT-T5-GPT
├── 06-Embedding
├── 07-RAG
├── 08-Fine-Tuning
├── 09-LangChain
├── 10-LangGraph
├── 11-Agent
├── 12-Deployment
├── 13-Rust-AI
├── 14-Projects
└── assets/

* Chương 1.1: Giới thiệu LLM, Roadmap và môi trường
* 01.1.1 - Giới thiệu AI Engineer & LLM Engineer
* 01.1.2 - Bức tranh toàn cảnh của một hệ thống LLM
* 01.1.3 - Roadmap học tập
* 01.1.4 - Chuẩn bị môi trường phát triển

* Chương 1.2: Python, Rust và Hugging Face
* Chương 1.3: CUDA, ONNX, llama.cpp, Ollama
* Chương 1.4: Kiến trúc tổng thể của một hệ thống LLM
* Chương 1.5: Dự án đầu tiên (Chat với Qwen local)