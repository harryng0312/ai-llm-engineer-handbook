# AI Agent / Harness Engineer Roadmap (Chi tiết)

## Tổng thời lượng
- 16–24 tuần
- 10–15 giờ/tuần

---

# Phase 0 - Foundation LLM

## Mục tiêu
Hiểu cách LLM hoạt động trước khi học Agent.

### Module 0.1 Transformer
Kiến thức:
- Tokenization
- Embedding
- Positional Encoding
- Self Attention
- Multi Head Attention
- Decoder-only Transformer
- Encoder-Decoder

Bài tập:
- Tự code Scaled Dot Product Attention bằng TensorFlow/PyTorch.
- Minh họa Q, K, V.

Tiêu chí:
- Giải thích được GPT, BERT khác nhau thế nào.
- Giải thích KV Cache.

### Module 0.2 Embedding
- Word2Vec
- FastText
- SBERT
- E5
- BGE

Project:
- Semantic Search.

Tiêu chí:
- Xây được cosine similarity search.

---

# Phase 1 - Tool Calling & Harness

## Mục tiêu
Hiểu Agent = LLM + Harness.

### Module 1.1 Function Calling

Học:
- JSON Schema
- Structured Output
- Tool Calling

Project:
- Calculator Tool
- Weather Tool

Tiêu chí:
- Tool Registry
- Tool Dispatcher
- Tool Executor

### Module 1.2 ReAct

Chu trình:
Thought -> Action -> Observation

Project:
- Search Agent

Tiêu chí:
- Không dùng framework.

---

# Phase 2 - RAG

## Module 2.1 Vector Database

Học:
- pgvector
- Qdrant
- Milvus

So sánh:
- Recall
- Latency
- HNSW
- IVF

Project:
- PDF QA Bot

Tiêu chí:
- Search được hàng nghìn tài liệu.

## Module 2.2 Chunking

- Fixed
- Recursive
- Semantic

Tiêu chí:
- Benchmark Recall@K.

## Module 2.3 Hybrid Search

- BM25
- Dense Retrieval
- Hybrid

Project:
- Enterprise Search.

---

# Phase 3 - LangChain

## Module 3.1 LCEL

- Runnable
- Chain Composition

Tiêu chí:
- Không dùng legacy chain.

## Module 3.2 Retriever

- Vector Store Retriever
- Multi Query Retriever

## Module 3.3 Structured Output

- Pydantic
- Typed Output

Project:
- RAG Bot bằng LangChain.

---

# Phase 4 - LangGraph

## Mục tiêu
Harness Production.

### Module 4.1 StateGraph

- State
- Node
- Edge

Project:
- Research Agent

### Module 4.2 Conditional Routing

- Retry
- Branching

### Module 4.3 Checkpoint

- Persistence
- Resume

### Module 4.4 Human In The Loop

Project:
- Approval Workflow

Tiêu chí:
- Pause/Resume workflow.

---

# Phase 5 - MCP

## Module 5.1 MCP Client

- Discovery
- Invocation

## Module 5.2 MCP Server

Project:
- Filesystem MCP
- PostgreSQL MCP

Tiêu chí:
- Claude Desktop gọi được tool.

---

# Phase 6 - Dify

## Module 6.1 Workflow

- Condition
- Loop
- LLM Node
- Tool Node

## Module 6.2 Knowledge Base

- Embedding
- Retrieval

Project:
- Customer Support Bot

---

# Phase 7 - Multi Agent

## CrewAI

Vai trò:
- Researcher
- Writer
- Reviewer

## AutoGen

- Debate
- Collaboration

Project:
- Team nghiên cứu tự động.

---

# Phase 8 - Evaluation

## LLM Evaluation

- LM Evaluation Harness

## RAG Evaluation

- RAGAS
- DeepEval

Metrics:
- Faithfulness
- Recall
- Relevancy

---

# Phase 9 - Production

## Observability

- LangSmith
- OpenTelemetry

## Security

- Prompt Injection
- Tool Injection
- Data Leakage

## Deployment

- Docker
- Kubernetes
- Ollama
- vLLM

Project:
- Production Agent.

---

# Tài liệu chính thức

## LangChain
https://python.langchain.com/docs/

## LangGraph
https://langchain-ai.github.io/langgraph/

## Dify
https://docs.dify.ai/

## MCP
https://modelcontextprotocol.io/

## OpenAI Agents
https://platform.openai.com/docs/guides/agents

## Anthropic Agents
https://www.anthropic.com/engineering/building-effective-agents

---

# Khóa học đề xuất

## DeepLearning.AI

- LangChain for LLM Application Development
- AI Agents in LangGraph
- Multi AI Agent Systems with CrewAI

## Coursera

- Generative AI with Large Language Models
- AI Engineering Specialization

---

# Capstone

Xây hệ thống:

User
-> Planner
-> Tool Calling
-> Memory
-> RAG
-> LangGraph
-> MCP
-> Evaluation
-> Production

Tiêu chí tốt nghiệp:

Có thể tự xây Agent hoàn chỉnh không phụ thuộc framework.
