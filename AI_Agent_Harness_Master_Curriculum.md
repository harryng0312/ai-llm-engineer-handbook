Bạn có thể copy trực tiếp nội dung dưới đây vào file:

AI_Agent_Harness_Master_Curriculum.md

AI Agent & Harness Engineering Master Curriculum

Mục tiêu

Lộ trình đào tạo từ:

ML Engineer
    ↓
RAG Engineer
    ↓
Agent Engineer
    ↓
AI Platform Architect

⸻

Tổng quan chương trình

Phase	Chủ đề	Thời lượng
0	Python & AI System Foundation	1 tuần
1	LLM Internals	2 tuần
2	Embedding & Retrieval	2 tuần
3	RAG Engineering	3 tuần
4	Harness Fundamentals	3 tuần
5	LangChain	2 tuần
6	LangGraph	3 tuần
7	MCP	2 tuần
8	Dify	1 tuần
9	Multi-Agent	2 tuần
10	Evaluation	1 tuần
11	Production	2 tuần
12	Capstone	2 tuần

⸻

PHASE 0 – PYTHON & AI SYSTEM FOUNDATION

Mục tiêu

Xây dựng nền tảng backend và hệ thống.

Học

* Python Typing
* AsyncIO
* Dataclass
* Pydantic
* FastAPI
* PostgreSQL
* Redis
* Docker

Assignment

Xây API:

POST /chat
POST /embed
POST /search

Tiêu chí hoàn thành

* Docker hóa được hệ thống
* Triển khai được bằng Docker Compose

⸻

PHASE 1 – LLM INTERNALS

Chương 1 – Transformer

Học

* Tokenization
* BPE
* SentencePiece
* Embedding
* Self Attention
* Multi-head Attention
* Decoder Only
* Encoder Decoder

Assignment

Tự cài đặt:

scaled_dot_product_attention()

Tiêu chí

Giải thích được:

* Q
* K
* V
* Attention Score

Reading

* Attention Is All You Need
* The Illustrated Transformer

⸻

Chương 2 – Inference

Học

* KV Cache
* Quantization
* GGUF
* Ollama
* vLLM

Assignment

Deploy local:

* Qwen
* Llama
* Gemma

Tiêu chí

Benchmark:

* Throughput
* Latency
* Memory

⸻

PHASE 2 – EMBEDDING

Chương 3 – Embedding Models

Học

* Word2Vec
* FastText
* SBERT
* E5
* BGE

Assignment

Xây Semantic Search:

Query
 ↓
Embedding
 ↓
Cosine Similarity
 ↓
Top-K

Tiêu chí

So sánh:

* SBERT
* E5
* BGE

⸻

PHASE 3 – RAG ENGINEERING

Chương 4 – Vector Database

Học

* pgvector
* Qdrant
* Milvus

Assignment

PDF QA System:

PDF
 ↓
Chunk
 ↓
Embedding
 ↓
Qdrant
 ↓
Retriever

Deliverable

Web UI tìm kiếm tài liệu.

⸻

Chương 5 – Chunking

Học

* Fixed Chunk
* Recursive Chunk
* Semantic Chunk

Assignment

Benchmark:

* Recall@5
* Recall@10

⸻

Chương 6 – Retrieval

Học

* Dense Retrieval
* BM25
* Hybrid Search

Assignment

Enterprise Search.

⸻

Chương 7 – Reranker

Học

* Cross Encoder
* BGE Reranker
* Cohere Rerank

Assignment

Pipeline:

Retriever
 ↓
Reranker
 ↓
LLM

⸻

PHASE 4 – HARNESS FUNDAMENTALS

Chương 8 – Function Calling

Học

* Tool Calling
* Structured Output
* JSON Schema

Assignment

Viết:

ToolRegistry
ToolDispatcher
ToolExecutor

Tiêu chí

Agent gọi được:

* Search
* Calculator
* Weather

⸻

Chương 9 – ReAct

Học

Thought → Action → Observation

Assignment

Search Agent:

Thought
 ↓
Search
 ↓
Observe
 ↓
Reason
 ↓
Answer

Tiêu chí

Không dùng framework.

⸻

Chương 10 – Memory

Học

* Short Term Memory
* Long Term Memory
* Episodic Memory
* Semantic Memory

Assignment

Chatbot có Memory.

⸻

PHASE 5 – LANGCHAIN

Chương 11 – LCEL

Học

prompt | llm
retriever | prompt | llm

Assignment

RAG bằng LCEL.

⸻

Chương 12 – Structured Output

Học

* Pydantic
* JSON Schema

Assignment

JSON Output Agent.

Tiêu chí

Output hợp lệ 100%.

⸻

PHASE 6 – LANGGRAPH

Chương 13 – StateGraph

Học

* Node
* Edge
* State

Assignment

Research Agent:

Search
 ↓
Read
 ↓
Summarize
 ↓
Report

⸻

Chương 14 – Conditional Routing

Assignment

Retry Workflow.

⸻

Chương 15 – Checkpoint

Assignment

Pause / Resume.

⸻

Chương 16 – Human In The Loop

Assignment

Approval Workflow.

⸻

PHASE 7 – MCP

Chương 17 – MCP Client

Học

* Discovery
* Tool Invocation

⸻

Chương 18 – MCP Server

Assignment

Filesystem MCP Server.

Assignment nâng cao

PostgreSQL MCP Server.

Tiêu chí

Claude Desktop gọi được tool.

⸻

PHASE 8 – DIFY

Chương 19 – Workflow

Học

* Condition
* Loop
* Tool
* Knowledge Base

Assignment

Customer Support Bot.

⸻

Chương 20 – Agent

Assignment

Knowledge Assistant.

⸻

PHASE 9 – MULTI AGENT

Chương 21 – CrewAI

Assignment

Research Team:

* Researcher
* Writer
* Reviewer

⸻

Chương 22 – AutoGen

Assignment

Agent Debate.

⸻

PHASE 10 – EVALUATION

Chương 23

Học

* RAGAS
* DeepEval
* LM Evaluation Harness

Assignment

Đo:

* Faithfulness
* Context Recall
* Answer Relevancy
* Precision

⸻

PHASE 11 – PRODUCTION

Chương 24 – Observability

Học

* LangSmith
* OpenTelemetry

⸻

Chương 25 – Security

Học

* Prompt Injection
* Tool Injection
* Data Leakage

⸻

Chương 26 – Deployment

Học

* Docker
* Kubernetes
* vLLM
* Nginx

Assignment

Deploy:

LangGraph
+
Qdrant
+
vLLM
+
FastAPI

⸻

PHASE 12 – CAPSTONE PROJECTS

Project 1 – Research Agent

Search
 ↓
Read
 ↓
Summarize
 ↓
Report

⸻

Project 2 – Code Review Agent

GitHub
 ↓
Analyze
 ↓
Review
 ↓
Fix Suggestion

⸻

Project 3 – Enterprise Knowledge Assistant

User
 ↓
API Gateway
 ↓
LangGraph
 ↓
MCP
 ↓
Qdrant
 ↓
vLLM

⸻

OFFICIAL REFERENCES

OpenAI

https://platform.openai.com/docs

Anthropic

https://www.anthropic.com/engineering/building-effective-agents

LangChain

https://python.langchain.com/docs/

LangGraph

https://langchain-ai.github.io/langgraph/

MCP

https://modelcontextprotocol.io/

Dify

https://docs.dify.ai/

Qdrant

https://qdrant.tech/documentation/

Milvus

https://milvus.io/docs

pgvector

https://github.com/pgvector/pgvector

⸻

COURSES

DeepLearning.AI

* LangChain for LLM Application Development
* AI Agents in LangGraph
* Multi AI Agent Systems with CrewAI

Coursera

* Generative AI with Large Language Models
* AI Engineering Specialization

⸻

TIÊU CHÍ TỐT NGHIỆP

Có khả năng tự xây dựng:

User Query
 ↓
Planner
 ↓
Memory
 ↓
Tool Calling
 ↓
Retriever
 ↓
Reasoning Loop
 ↓
Answer

không phụ thuộc LangChain, Dify hoặc framework tương tự.

Sau khi lưu file này, mình khuyên nên tiếp tục xây dựng Volume 2 riêng cho:

* LangGraph chuyên sâu
* MCP chuyên sâu
* Dify Production
* AI Agent System Design
* Rust + Agent + MCP
* Enterprise RAG

vì phần trên mới tương đương khoảng Level 1 → Level 2 Agent Engineer, còn để đạt mức Senior/Architect sẽ cần thêm khoảng 200–300 trang tài liệu nâng cao nữa.


Mình không thể tạo file để tải về trực tiếp trong lượt này, nhưng có thể cung cấp nội dung Markdown hoàn chỉnh để bạn lưu thành:

AI_Agent_Harness_Architect_Curriculum.md

AI Agent & Harness Architect Curriculum

Senior → Staff → Architect Track

⸻

MỤC TIÊU

Sau chương trình này học viên có thể:

* Thiết kế Agent Platform quy mô doanh nghiệp
* Thiết kế RAG Platform đa tenant
* Xây Agent Runtime
* Xây MCP Ecosystem
* Thiết kế Multi-Agent System
* Thiết kế AI Gateway
* Xây Evaluation Platform
* Thiết kế AI Governance

⸻

LEVEL 3 – SENIOR AGENT ENGINEER

Module S1 - Advanced Retrieval

Mục tiêu

Hiểu sâu retrieval architecture.

Học

Dense Retrieval

* SBERT
* E5
* BGE
* GTE

Sparse Retrieval

* BM25
* SPLADE

Hybrid Retrieval

Dense
+
Sparse
+
Reranker

Assignment

Xây:

Enterprise Search Engine

Hỗ trợ:

* PDF
* DOCX
* HTML
* Wiki

Tiêu chí

* Recall@10 > 90%
* MRR Benchmark

⸻

Module S2 - Reranking

Học

Cross Encoder

* BGE Reranker
* Cohere Rerank
* Jina Reranker

Assignment

Benchmark:

No Rerank
vs
Rerank

⸻

Module S3 - Long Context

Học

* Context Compression
* Context Ranking
* Context Distillation

Assignment

Xây:

1000+
documents QA

⸻

LEVEL 4 – AGENT ENGINEER

Module A1 - Planning

Học

ReAct

Plan-and-Execute

Tree-of-Thought

Graph-of-Thought

Reflexion

Assignment

Planner Agent

Goal
 ↓
Subtasks
 ↓
Execution

⸻

Module A2 - Agent Memory

Học

Episodic Memory

User actions

Semantic Memory

Facts

Procedural Memory

Skills

Assignment

Chat Agent có:

* User Profile
* Long Term Memory
* Skill Memory

⸻

Module A3 - Tool Ecosystem

Học

Tool Registry

Tool Discovery

Tool Routing

Tool Authorization

Assignment

Xây:

Tool Hub

cho 50+ tools.

⸻

LEVEL 5 – MULTI AGENT SYSTEMS

Module M1

Học

CrewAI

AutoGen

Swarm

Hierarchical Agents

Kiến trúc

Manager
 ↓
Researcher
 ↓
Writer
 ↓
Reviewer

Assignment

Research Team Agent.

⸻

Module M2

Debate Systems

Agent A

vs

Agent B

Assignment

Code Review Debate.

⸻

Module M3

Market Simulation

Multiple Agents

Assignment

AI Trading Simulation.

⸻

LEVEL 6 – MCP ARCHITECT

Module MCP1

Học

Protocol Design

Transport

Capability Discovery

Assignment

Xây MCP Server:

Filesystem

Git

PostgreSQL

Qdrant

⸻

Module MCP2

MCP Gateway

Kiến trúc:

Agent
 ↓
Gateway
 ↓
MCP Servers

Assignment

MCP Router.

⸻

Module MCP3

Security

Authentication

Authorization

Audit

Assignment

Enterprise MCP Platform.

⸻

LEVEL 7 – EVALUATION ENGINEERING

Module E1

Học

LM Evaluation Harness

RAGAS

DeepEval

OpenAI Evals

Assignment

Evaluation Pipeline.

⸻

Module E2

Metrics

Faithfulness

Answer Relevancy

Context Precision

Context Recall

Latency

Cost

Assignment

Dashboard.

⸻

LEVEL 8 – AGENT OBSERVABILITY

Module O1

Học

LangSmith

OpenTelemetry

Tracing

Assignment

Trace toàn bộ workflow.

⸻

Module O2

Cost Analytics

Token Usage

Cost Prediction

Cost Optimization

Assignment

AI Cost Dashboard.

⸻

LEVEL 9 – AGENT SECURITY

Module SEC1

Học

Prompt Injection

Indirect Prompt Injection

Jailbreak

Data Leakage

Assignment

Attack & Defense Lab.

⸻

Module SEC2

Guardrails

NeMo Guardrails

PydanticAI

Policy Engine

Assignment

Safe Enterprise Agent.

⸻

LEVEL 10 – AGENT PLATFORM ARCHITECTURE

Module P1

Agent Runtime

Thiết kế:

Scheduler
Memory
Tool Engine
Reasoning Engine

Assignment

Mini Agent Runtime.

⸻

Module P2

Agent Gateway

Kiến trúc:

Users
 ↓
API Gateway
 ↓
Agent Gateway
 ↓
Model Router

Assignment

Multi-Model Router.

⸻

Module P3

Model Routing

GPT

Claude

Gemini

Llama

Assignment

Dynamic Routing.

⸻

Module P4

Cost Optimization

Routing theo:

* Cost
* Latency
* Quality

Assignment

AI Load Balancer.

⸻

LEVEL 11 – ENTERPRISE RAG PLATFORM

Module ERP1

Multi Tenant

Tenant Isolation

Metadata Filtering

ACL

Assignment

Enterprise Knowledge Base.

⸻

Module ERP2

Data Ingestion

PDF

DOCX

Confluence

SharePoint

GitHub

Assignment

Universal Ingestion Service.

⸻

Module ERP3

Knowledge Governance

Retention

Versioning

Auditing

Assignment

Governance Layer.

⸻

LEVEL 12 – AI GOVERNANCE

Module G1

Compliance

GDPR

SOC2

ISO27001

Assignment

Compliance Checklist.

⸻

Module G2

Risk Management

Model Risk

Prompt Risk

Data Risk

Assignment

Risk Assessment.

⸻

ARCHITECT CAPSTONE PROJECTS

Project 1

Enterprise AI Assistant

Users
 ↓
Gateway
 ↓
LangGraph
 ↓
MCP
 ↓
Qdrant
 ↓
vLLM

⸻

Project 2

AI Operating System

Components

Memory

Planner

Tool Hub

Evaluation

Observability

Governance

⸻

Project 3

Agent Platform

Features

* Multi Tenant
* Multi Model
* Multi Agent
* Evaluation
* Cost Analytics
* Governance

⸻

READING LIST

Papers

Attention Is All You Need

ReAct

Tree of Thoughts

Reflexion

Self-RAG

CRAG

AgentBench

GAIA

SWE-Bench

⸻

OFFICIAL DOCS

OpenAI

https://platform.openai.com/docs

Anthropic

https://www.anthropic.com/engineering/building-effective-agents

LangChain

https://python.langchain.com/docs/

LangGraph

https://langchain-ai.github.io/langgraph/

MCP

https://modelcontextprotocol.io/

CrewAI

https://docs.crewai.com/

AutoGen

https://microsoft.github.io/autogen/

Dify

https://docs.dify.ai/

Qdrant

https://qdrant.tech/documentation/

Milvus

https://milvus.io/docs

vLLM

https://docs.vllm.ai/

⸻

KỸ NĂNG KHI TỐT NGHIỆP

Senior Agent Engineer

* RAG Production
* LangGraph
* MCP
* Evaluation

Staff Engineer

* Agent Platform
* Multi Agent
* Observability
* Governance

Architect

* AI Infrastructure
* AI Gateway
* AI Governance
* Enterprise Platform Design

Đây là phần Senior → Architect. Nếu học nghiêm túc toàn bộ Volume 1 (Foundation → Agent Engineer) và Volume 2 (Senior → Architect), khối lượng tương đương khoảng:

~ 500-800 giờ học
~ 20-30 project
~ 6-12 tháng part-time

và đã bao phủ gần như toàn bộ các mảng đang được dùng trong hệ sinh thái Agent AI hiện nay: LangGraph, MCP, Dify, CrewAI, AutoGen, Evaluation, Observability, Security và Platform Architecture.