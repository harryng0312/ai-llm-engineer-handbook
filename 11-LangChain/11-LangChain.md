# 11 — LangChain

Chương [13-Agent.md](../13-Agent/13-Agent.md) đã xây agent từ nền tảng thấp nhất: vòng lặp ReAct viết tay và tool calling thô qua JSON schema. **LangChain** là framework abstraction giúp không phải viết lại những phần lặp đi lặp lại đó — chuẩn hóa cách gọi nhiều provider LLM, cách định nghĩa tool, cách quản lý prompt — xây trên đúng nền tảng agent cơ bản đã học ở chương trước. Chương sau, [12-LangGraph.md](../12-LangGraph/12-LangGraph.md), sẽ đi sâu vào phần orchestration phức tạp hơn của cùng hệ sinh thái (đồ thị trạng thái, checkpoint, multi-agent).

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`11-LangChain.ipynb`](./11-LangChain.ipynb).

## Mục lục

1. [LangChain là gì](#chương-111--langchain-là-gì)
2. [Định nghĩa Tool bằng @tool](#chương-112--định-nghĩa-tool-bằng-tool)
3. [Structured Output: Pydantic và JSON Schema](#chương-113--structured-output-pydantic-và-json-schema)
4. [Bài tập](#chương-114--bài-tập)
5. [Tài liệu tham khảo](#chương-115--tài-liệu-tham-khảo)

---

## Chương 11.1 — LangChain là gì

[LangChain](https://python.langchain.com) là framework cung cấp các abstraction tái sử dụng cho ứng dụng LLM: chuẩn hóa cách gọi nhiều provider LLM khác nhau (Ollama, OpenAI, Anthropic...) qua cùng 1 interface, cách định nghĩa tool, cách quản lý prompt template, và tích hợp sẵn với các vector DB (Qdrant, FAISS...).

## Chương 11.2 — Định nghĩa Tool bằng @tool

Định nghĩa tool bằng decorator `@tool` (gọn hơn viết JSON schema tay như ở [13-Agent.md](../13-Agent/13-Agent.md) — LangChain tự sinh schema từ type hint và docstring):

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

`create_react_agent` (từ `langgraph.prebuilt`) đã đóng gói sẵn vòng lặp ReAct hoàn chỉnh: model quyết định gọi tool nào, LangGraph tự chạy tool, đưa kết quả trở lại, lặp lại cho tới khi model trả lời cuối cùng — tương đương với đoạn code thủ công ở [13-Agent.md](../13-Agent/13-Agent.md) nhưng không phải tự viết vòng lặp.

## Chương 11.3 — Structured Output: Pydantic và JSON Schema

Khi câu trả lời của agent sẽ được **code đọc lại** (không phải người đọc trực tiếp) — ví dụ để ghi vào database, gọi API tiếp theo, hiển thị lên UI theo layout cố định — text tự do không đủ tin cậy: model có thể trả lời thừa câu dẫn nhập, đổi định dạng giữa các lần gọi, hoặc quên một trường. **Structured Output** ép model trả về đúng theo một schema định trước.

Cách gọn nhất trong LangChain: định nghĩa schema bằng Pydantic, gọi `.with_structured_output()` — LangChain tự chuyển Pydantic model thành JSON Schema, gửi kèm request, và parse kết quả trả về thành đúng object Python:

```python
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class WeatherQuery(BaseModel):
    city: str = Field(description="Tên thành phố cần tra cứu thời tiết")
    unit: str = Field(default="celsius", description="Đơn vị nhiệt độ: 'celsius' hoặc 'fahrenheit'")

llm = ChatOllama(model="qwen2.5:7b")
structured_llm = llm.with_structured_output(WeatherQuery)

result = structured_llm.invoke("Cho tôi biết thời tiết Đà Nẵng, tính bằng độ F")
print(result)             # WeatherQuery(city='Đà Nẵng', unit='fahrenheit')
print(type(result))       # <class '__main__.WeatherQuery'> -- không phải string, dùng trực tiếp result.city
```

Nếu không dùng LangChain, có thể đạt hiệu quả tương tự bằng tham số `format="json"` của Ollama (ép model chỉ sinh JSON hợp lệ ở tầng decoding, không đảm bảo đúng schema — vẫn cần validate lại bằng Pydantic sau khi parse):

```python
import ollama, json
from pydantic import BaseModel, ValidationError

class WeatherQuery(BaseModel):
    city: str
    unit: str = "celsius"

response = ollama.chat(model="qwen2.5:7b", format="json", messages=[
    {"role": "system", "content": "Luôn trả về JSON với 2 field: city (string), unit (string)."},
    {"role": "user", "content": "Thời tiết Đà Nẵng bằng độ F"},
])

try:
    query = WeatherQuery.model_validate(json.loads(response["message"]["content"]))
except (json.JSONDecodeError, ValidationError) as e:
    print(f"Model trả về JSON không hợp lệ: {e}")
```

Structured output là nền tảng cho nhiều kỹ thuật ở các chương sau: planner ở [13-Agent.md](../13-Agent/13-Agent.md) cần output là 1 danh sách bước có cấu trúc, agent evaluation ở [13-Agent.md](../13-Agent/13-Agent.md) cần input/output có schema cố định để so sánh tự động.

## Chương 11.4 — Bài tập

1. Định nghĩa 1 schema Pydantic cho một task thật bạn hay làm (ví dụ trích xuất thông tin từ email), dùng `with_structured_output` ở [Chương 11.3](#chương-113--structured-output-pydantic-và-json-schema), thử với 5 input khác nhau và kiểm tra tỉ lệ parse thành công.
2. Viết 1 tool mới bằng `@tool` cho một use case khác — ví dụ `convert_currency(amount: float, from_currency: str, to_currency: str) -> float` (giả lập bảng tỷ giá cố định). Gắn tool này vào `create_react_agent` cùng với `get_weather`/`calculate` đã có ở [Chương 11.2](#chương-112--định-nghĩa-tool-bằng-tool), rồi thử hỏi 1 câu buộc agent phải tự chọn đúng tool trong 3 tool có sẵn.

## Chương 11.5 — Tài liệu tham khảo

- [LangChain docs](https://python.langchain.com/docs/introduction/)
- [Ollama tool calling](https://ollama.com/blog/tool-support)

📖 Xem chú giải chi tiết ở [17-Papers.md](../17-Papers/17-Papers.md).
