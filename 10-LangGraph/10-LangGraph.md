# 10 — LangGraph

LangGraph mô hình hóa agent như một **đồ thị trạng thái (state graph)** — cần thiết khi logic phức tạp hơn một vòng lặp tuyến tính đã học ở [09-LangChain.md](../09-LangChain/09-LangChain.md) hay [11-Agent.md](../11-Agent/11-Agent.md): rẽ nhánh có điều kiện, quay lại bước trước, dừng giữa chừng chờ người duyệt, hoặc nhiều agent phối hợp.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`10-LangGraph.ipynb`](./10-LangGraph.ipynb).

## Mục lục

1. [StateGraph là gì](#chương-101--stategraph-là-gì)
2. [Tự xây ReAct loop bằng LangGraph](#chương-102--tự-xây-react-loop-bằng-langgraph)
3. [Checkpoint và Human-in-the-loop](#chương-103--checkpoint-và-human-in-the-loop)
4. [Bài tập](#chương-104--bài-tập)
5. [Tài liệu tham khảo](#chương-105--tài-liệu-tham-khảo)

---

## Chương 10.1 — StateGraph là gì

`create_react_agent` ở [09-LangChain.md](../09-LangChain/09-LangChain.md) phù hợp cho agent đơn giản (1 vòng lặp reasoning-action tuyến tính). Khi logic phức tạp hơn — cần rẽ nhánh có điều kiện, quay lại bước trước, hoặc nhiều agent phối hợp — cần một cách biểu diễn linh hoạt hơn: đó là [LangGraph](https://langchain-ai.github.io/langgraph/), mô hình hóa agent như một **đồ thị trạng thái (state graph)**: mỗi node là một bước xử lý (gọi LLM, chạy tool, kiểm tra điều kiện...), cạnh (edge) nối các node, có thể là cạnh cố định hoặc **cạnh có điều kiện** (routing dựa trên state hiện tại).

## Chương 10.2 — Tự xây ReAct loop bằng LangGraph

Tự xây một ReAct loop bằng LangGraph (tương đương những gì `create_react_agent` làm ngầm, viết tường minh để hiểu cơ chế):

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages: tự động append thay vì ghi đè

llm = ChatOllama(model="qwen2.5:7b").bind_tools([get_weather, calculate])  # tool định nghĩa ở 09-LangChain.md
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

Đây chính là "vòng lặp" ReAct từ [11-Agent.md](../11-Agent/11-Agent.md) biểu diễn tường minh dưới dạng đồ thị: `agent -> (có tool_call?) -> tools -> agent -> ... -> END`. Khi cần logic phức tạp hơn (ví dụ: giới hạn số vòng lặp tối đa, thêm bước "human-in-the-loop" chờ người duyệt trước khi chạy tool nguy hiểm), chỉ cần thêm node/cạnh vào đồ thị này.

## Chương 10.3 — Checkpoint và Human-in-the-loop

Graph ở [Chương 10.2](#chương-102--tự-xây-react-loop-bằng-langgraph) chạy xong là mất trạng thái — không thể dừng giữa chừng rồi tiếp tục sau, và không có cách nào chặn agent lại trước khi nó thực hiện một hành động rủi ro. LangGraph giải quyết bằng **checkpointer**: lưu lại state của graph sau mỗi bước, gắn với một `thread_id` — cho phép tạm dừng, khôi phục (kể cả sau khi restart chương trình nếu dùng checkpointer có persistence như `SqliteSaver`/`PostgresSaver` thay vì `MemorySaver` trong RAM), và **can thiệp giữa chừng**.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
# interrupt_before=["tools"]: graph DỪNG LẠI ngay trước khi vào node "tools",
# chờ con người duyệt thay vì tự động chạy tool
app = graph.compile(checkpointer=checkpointer, interrupt_before=["tools"])

config = {"configurable": {"thread_id": "user-123"}}

result = app.invoke(
    {"messages": [{"role": "user", "content": "Xóa toàn bộ file log cũ hơn 30 ngày trong thư mục logs/"}]},
    config=config,
)
# Graph dừng TRƯỚC node "tools" -- xem model định gọi tool gì trước khi cho phép chạy
last_message = result["messages"][-1]
print(last_message.tool_calls)  # [{'name': 'delete_files', 'args': {...}, ...}]

# Người vận hành xem qua, quyết định duyệt -> gọi lại invoke(None, ...) để graph
# tiếp tục chạy tiếp từ đúng checkpoint đã lưu (không cần truyền lại input ban đầu)
if input("Cho phép chạy tool này? (y/n): ") == "y":
    final = app.invoke(None, config=config)
    print(final["messages"][-1].content)
```

Cơ chế `thread_id` cũng chính là nền tảng để làm **multi-session chat**: mỗi user/conversation dùng một `thread_id` riêng, checkpointer tự quản lý lịch sử state cho từng thread độc lập — thay thế cho việc tự viết `ConversationMemory` thủ công như ở [11-Agent.md](../11-Agent/11-Agent.md) khi đã dùng LangGraph.

## Chương 10.4 — Bài tập

1. Sửa graph ở [Chương 10.2](#chương-102--tự-xây-react-loop-bằng-langgraph) để giới hạn tối đa 3 vòng lặp `agent -> tools`, nếu vượt quá thì buộc trả lời "Không thể hoàn thành trong giới hạn cho phép".
2. Thêm `interrupt_before` cho một graph multi-agent kiểu supervisor bạn tự xây (xem ví dụ multi-agent ở [11-Agent.md](../11-Agent/11-Agent.md)), yêu cầu người dùng duyệt trước khi một agent con "thực thi" (giả lập) đoạn code nó viết ra — cùng cơ chế `interrupt_before=["tools"]` ở [Chương 10.3](#chương-103--checkpoint-và-human-in-the-loop), nhưng áp dụng cho graph nhiều agent thay vì graph ReAct đơn.

## Chương 10.5 — Tài liệu tham khảo

- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph persistence/checkpoint](https://langchain-ai.github.io/langgraph/concepts/persistence/)

📖 Xem chú giải chi tiết ở [15-Papers.md](../15-Papers/15-Papers.md).
