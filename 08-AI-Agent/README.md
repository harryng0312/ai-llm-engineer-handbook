# 08 — AI Agent

> RAG ([chương 06](../06-RAG/README.md)) cho LLM "đọc thêm tài liệu" trước khi trả lời — một bước retrieve cố định. **Agent** đi xa hơn: LLM tự quyết định cần làm gì tiếp theo (gọi tool nào, tìm gì, khi nào dừng) qua nhiều bước, dựa trên kết quả của chính các bước trước đó.

## Mục lục

1. [Agent là gì — vòng lặp ReAct](#1-agent-là-gì--vòng-lặp-react)
2. [Tool Calling](#2-tool-calling)
3. [LangChain](#3-langchain)
4. [Memory](#4-memory)
5. [LangGraph](#5-langgraph)
6. [Multi-Agent](#6-multi-agent)
7. [Bài tập](#7-bài-tập)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Agent là gì — vòng lặp ReAct

Một LLM thông thường: `input -> output`, một lần duy nhất. Một **Agent**: LLM chạy trong một **vòng lặp**, ở mỗi bước có thể chọn hành động (gọi tool, tìm kiếm, tính toán) thay vì trả lời ngay, quan sát kết quả hành động đó, rồi quyết định bước tiếp theo — cho đến khi đủ thông tin để trả lời.

Pattern phổ biến nhất là **ReAct** (Reasoning + Acting, Yao et al. 2022):

```
Thought: mình cần biết giá cổ phiếu ABC hôm nay để trả lời
Action: call_tool("get_stock_price", {"ticker": "ABC"})
Observation: {"price": 152.3, "currency": "USD"}
Thought: đã có đủ thông tin, có thể trả lời
Final Answer: Giá cổ phiếu ABC hôm nay là 152.3 USD
```

So với RAG (1 bước retrieve cố định), Agent linh hoạt hơn: có thể gọi 0, 1, hoặc nhiều tool tùy câu hỏi, có thể tự sửa sai (nếu tool trả lỗi, thử lại với tham số khác), và có thể kết hợp nhiều loại tool khác nhau (tìm kiếm web, truy vấn DB, gọi API nội bộ, thậm chí retrieval RAG như một tool).

## 2. Tool Calling

**Tool calling** (hay function calling) là cơ chế nền tảng cho agent: LLM được cung cấp một danh sách "tool" (tên, mô tả, schema tham số), và thay vì chỉ sinh text, model có thể sinh ra một **lời gọi hàm có cấu trúc** (tên tool + tham số dạng JSON) — code của bạn thực thi hàm đó thật, rồi đưa kết quả trở lại cho model.

Quan trọng: **model không tự chạy code** — nó chỉ quyết định "nên gọi tool nào với tham số gì"; việc thực thi tool là trách nhiệm của ứng dụng.

Ví dụ với Qwen2.5 qua Ollama (hỗ trợ tool calling native từ Qwen2.5 trở lên):

```python
import ollama
import json

def get_weather(city: str) -> dict:
    """Hàm thật sự thực thi khi model quyết định gọi tool này (giả lập)."""
    fake_db = {"Hà Nội": {"temp_c": 28, "condition": "Nắng"}, "Đà Lạt": {"temp_c": 18, "condition": "Mát"}}
    return fake_db.get(city, {"error": "không có dữ liệu"})

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Lấy thông tin thời tiết hiện tại của một thành phố",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "Tên thành phố"}},
            "required": ["city"],
        },
    },
}]

messages = [{"role": "user", "content": "Thời tiết ở Đà Lạt hôm nay thế nào?"}]
response = ollama.chat(model="qwen2.5:7b", messages=messages, tools=tools)

if response["message"].get("tool_calls"):
    for call in response["message"]["tool_calls"]:
        fn_name = call["function"]["name"]
        fn_args = call["function"]["arguments"]  # đã là dict, Ollama tự parse JSON
        result = get_weather(**fn_args) if fn_name == "get_weather" else {"error": "unknown tool"}

        # Đưa kết quả tool trở lại hội thoại để model tổng hợp câu trả lời cuối
        messages.append(response["message"])
        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    final = ollama.chat(model="qwen2.5:7b", messages=messages)
    print(final["message"]["content"])
else:
    print(response["message"]["content"])
```

Nguyên tắc thiết kế tool tốt: **tên và mô tả rõ ràng** (model chọn tool dựa trên mô tả, mô tả mơ hồ → model chọn sai hoặc gọi thừa), **schema tham số chặt chẽ** (dùng `enum` nếu tham số chỉ nhận vài giá trị cố định), và **trả về lỗi có cấu trúc** thay vì raise exception (để model có thể tự sửa ở vòng lặp tiếp theo thay vì crash cả pipeline).

## 3. LangChain

[LangChain](https://python.langchain.com) là framework cung cấp các abstraction tái sử dụng cho ứng dụng LLM: chuẩn hóa cách gọi nhiều provider LLM khác nhau (Ollama, OpenAI, Anthropic...) qua cùng 1 interface, cách định nghĩa tool, cách quản lý prompt template, và tích hợp sẵn với các vector DB (Qdrant, FAISS...).

Định nghĩa tool bằng decorator `@tool` (gọn hơn viết JSON schema tay như ở mục 2 — LangChain tự sinh schema từ type hint và docstring):

```python
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str:
    """Lấy thông tin thời tiết hiện tại của một thành phố."""
    fake_db = {"Hà Nội": "28°C, Nắng", "Đà Lạt": "18°C, Mát"}
    return fake_db.get(city, "Không có dữ liệu cho thành phố này")

@tool
def calculate(expression: str) -> str:
    """Tính giá trị một biểu thức toán học đơn giản, ví dụ '12 * (3 + 4)'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Lỗi: {e}"

llm = ChatOllama(model="qwen2.5:7b")
agent = create_react_agent(llm, tools=[get_weather, calculate])

result = agent.invoke({"messages": [{"role": "user", "content": "Đà Lạt bao nhiêu độ, và 18 * 2 bằng bao nhiêu?"}]})
print(result["messages"][-1].content)
```

`create_react_agent` (từ `langgraph.prebuilt`) đã đóng gói sẵn vòng lặp ReAct hoàn chỉnh: model quyết định gọi tool nào, LangGraph tự chạy tool, đưa kết quả trở lại, lặp lại cho tới khi model trả lời cuối cùng — tương đương với đoạn code thủ công ở [mục 2](#2-tool-calling) nhưng không phải tự viết vòng lặp.

## 4. Memory

Agent xử lý hội thoại nhiều lượt cần "nhớ" — có 2 loại memory với mục đích khác nhau:

**Short-term memory (conversation memory)**: lịch sử hội thoại trong 1 phiên làm việc, thường chỉ là danh sách message truyền thẳng vào context mỗi lượt gọi:

```python
class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.messages: list[dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # giữ N lượt gần nhất để tránh context quá dài (tốn KV cache, xem chương 05)
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]

    def get(self) -> list[dict]:
        return self.messages
```

**Long-term memory**: thông tin cần nhớ **xuyên suốt nhiều phiên** (sở thích người dùng, sự kiện quan trọng đã xảy ra), thường lưu vào vector DB và retrieve lại như RAG khi cần — bản chất là áp dụng kỹ thuật ở [chương 06](../06-RAG/README.md) cho chính lịch sử hội thoại thay vì tài liệu tĩnh:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import time

class LongTermMemory:
    def __init__(self, collection: str = "agent_memory"):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection = collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def remember(self, fact: str):
        vector = self.model.encode(fact, normalize_embeddings=True)
        self.client.upsert(collection_name=self.collection, points=[
            PointStruct(id=int(time.time() * 1000), vector=vector.tolist(), payload={"fact": fact})
        ])

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        vector = self.model.encode(query, normalize_embeddings=True)
        hits = self.client.search(collection_name=self.collection, query_vector=vector.tolist(), limit=top_k)
        return [hit.payload["fact"] for hit in hits]

memory = LongTermMemory()
memory.remember("Người dùng thích nhận câu trả lời ngắn gọn, có ví dụ code.")
memory.remember("Người dùng đang làm việc với Qwen2.5 chạy qua Ollama.")
print(memory.recall("Người dùng thích trả lời kiểu gì?"))
```

## 5. LangGraph

`create_react_agent` ở mục 3 phù hợp cho agent đơn giản (1 vòng lặp reasoning-action tuyến tính). Khi logic phức tạp hơn — cần rẽ nhánh có điều kiện, quay lại bước trước, hoặc nhiều agent phối hợp — cần một cách biểu diễn linh hoạt hơn: đó là [LangGraph](https://langchain-ai.github.io/langgraph/), mô hình hóa agent như một **đồ thị trạng thái (state graph)**: mỗi node là một bước xử lý (gọi LLM, chạy tool, kiểm tra điều kiện...), cạnh (edge) nối các node, có thể là cạnh cố định hoặc **cạnh có điều kiện** (routing dựa trên state hiện tại).

Tự xây một ReAct loop bằng LangGraph (tương đương những gì `create_react_agent` làm ngầm, viết tường minh để hiểu cơ chế):

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages: tự động append thay vì ghi đè

llm = ChatOllama(model="qwen2.5:7b").bind_tools([get_weather, calculate])  # tool định nghĩa ở mục 3
tool_map = {"get_weather": get_weather, "calculate": calculate}

def call_model(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_messages = []
    for call in last_message.tool_calls:
        result = tool_map[call["name"]].invoke(call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # sau khi chạy tool, quay lại agent để tiếp tục suy luận

app = graph.compile()
result = app.invoke({"messages": [{"role": "user", "content": "Đà Lạt bao nhiêu độ, và 18 * 2 bằng bao nhiêu?"}]})
print(result["messages"][-1].content)
```

Đây chính là "vòng lặp" ReAct từ [mục 1](#1-agent-là-gì--vòng-lặp-react) biểu diễn tường minh dưới dạng đồ thị: `agent -> (có tool_call?) -> tools -> agent -> ... -> END`. Khi cần logic phức tạp hơn (ví dụ: giới hạn số vòng lặp tối đa, thêm bước "human-in-the-loop" chờ người duyệt trước khi chạy tool nguy hiểm), chỉ cần thêm node/cạnh vào đồ thị này.

## 6. Multi-Agent

Khi một agent phải đảm nhiệm quá nhiều loại việc (viết code, tra cứu tài liệu, gọi API bên ngoài, kiểm tra chất lượng...), prompt hệ thống dễ trở nên cồng kềnh và model dễ nhầm lẫn vai trò. **Multi-agent** chia nhỏ thành các agent chuyên biệt, phối hợp qua một agent điều phối.

Pattern phổ biến nhất: **Supervisor** — một agent trung tâm nhận yêu cầu, quyết định giao cho sub-agent nào (dựa trên mô tả năng lực từng agent), nhận kết quả về, quyết định giao tiếp tiếp hay tổng hợp trả lời cuối:

```python
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b")

def researcher_agent(state):
    """Chuyên tra cứu thông tin (ở đây giả lập, thực tế sẽ gắn RAG/tool search)."""
    query = state["messages"][-1]["content"]
    answer = llm.invoke([{"role": "system", "content": "Bạn là chuyên gia tra cứu, chỉ trả lời fact ngắn gọn."},
                          {"role": "user", "content": query}])
    return {"messages": state["messages"] + [{"role": "assistant", "content": answer.content, "name": "researcher"}]}

def coder_agent(state):
    """Chuyên viết code."""
    query = state["messages"][-1]["content"]
    answer = llm.invoke([{"role": "system", "content": "Bạn là chuyên gia lập trình Python, chỉ trả lời bằng code."},
                          {"role": "user", "content": query}])
    return {"messages": state["messages"] + [{"role": "assistant", "content": answer.content, "name": "coder"}]}

def supervisor(state) -> str:
    """Quyết định route đến agent nào dựa trên nội dung yêu cầu."""
    query = state["messages"][-1]["content"].lower()
    if any(kw in query for kw in ["code", "viết hàm", "python", "script"]):
        return "coder"
    return "researcher"

graph = StateGraph(dict)
graph.add_node("researcher", researcher_agent)
graph.add_node("coder", coder_agent)
graph.set_conditional_entry_point(supervisor, {"researcher": "researcher", "coder": "coder"})
graph.add_edge("researcher", END)
graph.add_edge("coder", END)

app = graph.compile()
result = app.invoke({"messages": [{"role": "user", "content": "Viết hàm Python kiểm tra số nguyên tố"}]})
print(result["messages"][-1]["content"])
```

Khi nào dùng multi-agent thay vì 1 agent với nhiều tool: khi các "vai trò" cần **system prompt/persona khác nhau rõ rệt** (chuyên gia code vs chuyên gia tra cứu), khi muốn **giới hạn phạm vi tool** của từng vai trò (agent tra cứu không nên có quyền chạy code), hoặc khi luồng công việc có nhiều bước độc lập có thể chạy song song. Nếu chỉ cần thêm vài tool cho cùng một persona, một agent với nhiều tool ([mục 3](#3-langchain)) thường đơn giản và đủ dùng — multi-agent tăng độ phức tạp vận hành (khó debug hơn, tốn nhiều lượt gọi LLM hơn) nên chỉ nên dùng khi thực sự cần.

## 7. Bài tập

1. Thêm 1 tool mới vào ví dụ ở [mục 2](#2-tool-calling) (ví dụ `search_wikipedia`), thử hỏi 1 câu cần gọi 2 tool liên tiếp (dùng kết quả tool 1 làm tham số cho tool 2), quan sát model có tự nối được luồng không.
2. Chuyển ví dụ `LongTermMemory` ở [mục 4](#4-memory) thành một tool (`@tool remember_fact`, `@tool recall_facts`) để chính agent tự quyết định khi nào cần lưu/nhớ lại, thay vì gọi thủ công.
3. Sửa graph ở [mục 5](#5-langgraph) để giới hạn tối đa 3 vòng lặp `agent -> tools`, nếu vượt quá thì buộc trả lời "Không thể hoàn thành trong giới hạn cho phép".
4. Mở rộng ví dụ multi-agent ở [mục 6](#6-multi-agent) thêm 1 agent thứ 3 (ví dụ "reviewer" — kiểm tra lại output của `coder_agent` trước khi trả về), vẽ lại graph với node reviewer nằm giữa coder và END.

## 8. Tài liệu tham khảo

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)
- Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* (2023)
- [LangChain docs](https://python.langchain.com/docs/introduction/) · [LangGraph docs](https://langchain-ai.github.io/langgraph/) · [Ollama tool calling](https://ollama.com/blog/tool-support)
