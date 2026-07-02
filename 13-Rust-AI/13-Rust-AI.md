# 13 — Rust cho AI

> Chương này không có notebook Jupyter đi kèm: Rust không có REPL tương tác kiểu "chạy từng cell" như Python — workflow chuẩn là viết, biên dịch (compile) và chạy toàn bộ chương trình. Vì vậy trọng tâm của chương không phải là "code từng bước trong notebook" mà là vai trò thực dụng của Rust trong một hệ thống AI: một lớp gateway/serving hiệu năng cao đứng **trước** phần AI (LLM, RAG, agent) vẫn được viết bằng Python.

## Mục lục

1. [Vì sao dùng Rust cho AI](#chương-131--vì-sao-dùng-rust-cho-ai)
2. [Kiến trúc: Rust Gateway đứng trước Python AI Service](#chương-132--kiến-trúc-rust-gateway-đứng-trước-python-ai-service)
3. [Rust Gateway hoàn chỉnh hơn: rate limiting và auth](#chương-133--rust-gateway-hoàn-chỉnh-hơn-rate-limiting-và-auth)
4. [Bài tập](#chương-134--bài-tập)
5. [Tài liệu tham khảo](#chương-135--tài-liệu-tham-khảo)

---

## Chương 13.1 — Vì sao dùng Rust cho AI

Nói thẳng từ đầu: **Rust không thay thế được Python ở phần model/training**. Hệ sinh thái AI mạnh nhất — `transformers`, `torch`, `langchain`, `langgraph`, các thư viện fine-tuning (LoRA/QLoRA), các framework RAG — gần như 100% được viết và duy trì bằng Python. Học Rust để "viết lại LLM bằng Rust" là đi sai trọng tâm và tốn thời gian không cần thiết đối với đa số AI Engineer.

Vậy Rust có chỗ đứng ở đâu trong một hệ thống AI? Ở **lớp hạ tầng** — nơi hiệu năng và độ tin cậy quan trọng hơn tốc độ phát triển tính năng:

- **API gateway / reverse proxy**: đứng trước service Python, xử lý hàng nghìn kết nối đồng thời với overhead thấp hơn nhiều so với một worker Python (nhờ mô hình async không GIL của Rust/Tokio).
- **Rate limiting, auth, logging tập trung**: những mối quan tâm "cross-cutting" này thường tách khỏi logic AI, phù hợp để đặt ở một lớp riêng viết bằng ngôn ngữ tối ưu cho I/O.
- **Binary gọn nhẹ khi deploy**: một binary Rust biên dịch sẵn (vài MB, không cần runtime) dễ đóng gói vào container tối giản hơn nhiều so với việc đóng gói cả Python interpreter + toàn bộ dependency (`torch`, `transformers`...) chỉ để chạy một lớp proxy mỏng.
- **Độ tin cậy về bộ nhớ và concurrency**: borrow checker của Rust loại bỏ cả lớp lỗi (data race, use-after-free, null pointer) ngay tại thời điểm biên dịch — có giá trị khi lớp gateway phải chạy ổn định 24/7 trước phần AI.

Rust cũng có một hệ sinh thái AI riêng đang phát triển, dù còn non hơn Python nhiều — chỉ cần biết tên để tra cứu khi cần, không phải phạm vi của sách này:

- **[`candle`](https://github.com/huggingface/candle)**: framework deep learning thuần Rust do Hugging Face phát triển, cho phép chạy inference (và ở mức độ hạn chế là training) mà không cần Python runtime.
- **[`ort`](https://github.com/pykeio/ort)**: binding Rust cho ONNX Runtime, cho phép load một model đã export sang định dạng ONNX (từ PyTorch/TensorFlow) và chạy inference trực tiếp trong Rust.

Hai dự án trên hữu ích khi cần nhúng inference vào một binary Rust độc lập (ví dụ ứng dụng edge/desktop không muốn kéo theo runtime Python) — đây là hướng nâng cao, nằm ngoài phạm vi chương này. Trọng tâm ở đây là kiến trúc thực dụng hơn nhiều: **giữ nguyên phần AI bằng Python, thêm Rust ở lớp gateway**.

## Chương 13.2 — Kiến trúc: Rust Gateway đứng trước Python AI Service

Giáo trình gốc đề xuất pipeline `... -> LangGraph -> Rust Backend -> Web UI`. Lý do cân nhắc Rust cho backend production: hiệu năng I/O cao hơn (nhiều client đồng thời), binary gọn nhẹ khi deploy, và tận dụng hệ sinh thái Rust nếu phần còn lại của hệ thống dùng Rust cho lớp hạ tầng (đây chính là chủ đề của chương này).

Việc cần làm khi chuyển sang Rust: **giữ nguyên phần Python** (ingest + agent LangGraph) làm một service riêng expose qua HTTP nội bộ (giống `server.py` trong dự án Trợ lý RAG + Agent, xem [14-Projects.md](../14-Projects/14-Projects.md)), rồi thêm một lớp Rust (`axum`) đứng trước làm reverse proxy / gateway kiêm xử lý các phần cần hiệu năng cao (rate limiting, auth, streaming response):

```rust
// Bộ khung tối giản minh họa ý tưởng — chi tiết đầy đủ (rate limiting, auth) ở mục 13.3 của chương này
use axum::{routing::post, Json, Router};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct ChatRequest { question: String }

#[derive(Serialize)]
struct ChatResponse { answer: String }

async fn chat(Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    // Gọi sang Python agent service (server.py) qua HTTP nội bộ
    let client = reqwest::Client::new();
    let resp: ChatResponse = client
        .post("http://localhost:8080/api/chat")
        .json(&serde_json::json!({ "question": req.question }))
        .send().await.unwrap()
        .json().await.unwrap();
    Json(resp)
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/api/chat", post(chat));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:9090").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

Đây là kiến trúc thực dụng: không cần viết lại LangGraph/model bằng Rust (hệ sinh thái AI vẫn mạnh nhất ở Python), chỉ cần đặt Rust ở lớp gateway để hưởng lợi về hiệu năng mà không đánh đổi tốc độ phát triển của phần AI. Đi sâu hơn về Rust cho AI (candle, ONNX Runtime bindings...) xem lại [mục 13.1](#chương-131--vì-sao-dùng-rust-cho-ai).

Sơ đồ tổng thể sau khi thêm lớp Rust:

```
┌──────────┐   HTTP   ┌────────────────┐   HTTP nội bộ   ┌────────────────┐   LangGraph   ┌───────────────────┐
│  Web UI  │ ───────▶ │  Rust Gateway  │ ──────────────▶ │  Python Agent   │ ────────────▶ │  Agent (Qwen)      │
│ (HTML/JS)│ ◀─────── │  (axum, :9090) │ ◀────────────── │  Service (:8080)│ ◀──────────── │  tools: rag_search │
└──────────┘  answer  └────────────────┘   answer        └────────────────┘   answer      └───────────────────┘
                        rate limit, auth
```

Lớp Rust chỉ forward request/response — toàn bộ logic AI (retrieval, tool calling, prompt) không thay đổi so với service Python gốc. Đây là lợi thế lớn nhất của kiến trúc này: có thể thêm/bớt lớp Rust mà không đụng vào code AI.

## Chương 13.3 — Rust Gateway hoàn chỉnh hơn: rate limiting và auth

Bản gateway ở mục 13.2 chỉ forward request, chưa có gì bảo vệ Python service khỏi bị quá tải hay bị gọi trái phép. Mở rộng thêm hai middleware phổ biến ở lớp gateway: **rate limiting** (giới hạn số request) và **auth** (kiểm tra token trước khi cho đi tiếp).

```rust
use axum::{
    extract::{ConnectInfo, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::Response,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tower::{limit::RateLimitLayer, ServiceBuilder};

#[derive(Deserialize)]
struct ChatRequest { question: String }

#[derive(Serialize)]
struct ChatResponse { answer: String }

// State dùng chung giữa các middleware/handler: token hợp lệ + bộ đếm request theo IP
#[derive(Clone)]
struct AppState {
    api_token: Arc<String>,
    python_service_url: Arc<String>,
    // Bộ đếm request per-IP theo cửa sổ thời gian cố định (fixed window) --
    // RateLimitLayer của tower giới hạn theo *toàn bộ service* chứ không phân biệt
    // theo key (IP), nên phần giới hạn theo IP dưới đây được cài thủ công bằng một
    // middleware riêng; RateLimitLayer vẫn được giữ lại làm lớp giới hạn tổng (global
    // ceiling) để chặn traffic tăng đột biến trước khi tới lớp per-IP.
    ip_hits: Arc<Mutex<HashMap<std::net::IpAddr, (Instant, u32)>>>,
}

const PER_IP_LIMIT: u32 = 5; // tối đa 5 request / giây / IP
const PER_IP_WINDOW: Duration = Duration::from_secs(1);

// Middleware auth: yêu cầu header "Authorization: Bearer <token>" khớp với token cấu hình
async fn auth_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let expected = format!("Bearer {}", state.api_token);
    let ok = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|v| v == expected)
        .unwrap_or(false);

    if !ok {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(next.run(request).await)
}

// Middleware rate limit theo IP: đếm số request trong cửa sổ PER_IP_WINDOW, chặn nếu vượt PER_IP_LIMIT
async fn per_ip_rate_limit_middleware(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let ip = addr.ip();
    let now = Instant::now();

    {
        let mut hits = state.ip_hits.lock().unwrap();
        let entry = hits.entry(ip).or_insert((now, 0));

        if now.duration_since(entry.0) > PER_IP_WINDOW {
            // Hết cửa sổ cũ -> reset bộ đếm
            *entry = (now, 1);
        } else {
            entry.1 += 1;
            if entry.1 > PER_IP_LIMIT {
                return Err(StatusCode::TOO_MANY_REQUESTS);
            }
        }
    }

    Ok(next.run(request).await)
}

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let client = reqwest::Client::new();
    let resp = client
        .post(state.python_service_url.as_str())
        .json(&serde_json::json!({ "question": req.question }))
        .send()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?
        .json::<ChatResponse>()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    Ok(Json(resp))
}

#[tokio::main]
async fn main() {
    let state = AppState {
        api_token: Arc::new(std::env::var("GATEWAY_TOKEN").unwrap_or_else(|_| "dev-token".into())),
        python_service_url: Arc::new("http://localhost:8080/api/chat".into()),
        ip_hits: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/api/chat", post(chat))
        .layer(
            ServiceBuilder::new()
                // Giới hạn tổng (global ceiling): tối đa 50 request/giây cho toàn bộ gateway,
                // bất kể IP -- lớp phòng thủ ngoài cùng trước khi vào các middleware chi tiết hơn.
                .layer(RateLimitLayer::new(50, Duration::from_secs(1)))
                .layer(middleware::from_fn_with_state(
                    state.clone(),
                    per_ip_rate_limit_middleware,
                ))
                .layer(middleware::from_fn_with_state(state.clone(), auth_middleware)),
        )
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9090").await.unwrap();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}
```

Vài điểm đáng chú ý trong đoạn code trên:

- `RateLimitLayer::new(50, Duration::from_secs(1))` là API thật của `tower::limit` — nó giới hạn tổng số request mà *service* xử lý trong mỗi cửa sổ thời gian, không phân biệt theo IP. Đây là lớp phòng thủ toàn cục (chặn traffic tăng đột biến), khác với `per_ip_rate_limit_middleware` là middleware tự viết để giới hạn theo từng IP — trong hệ thống thật, phần này thường được thay bằng crate chuyên dụng như `tower_governor` thay vì tự cài đặt bằng `HashMap` + `Mutex` như ví dụ trên (đơn giản hóa để dễ đọc).
- Thứ tự `.layer(...)` trong `ServiceBuilder` quan trọng: các layer được áp dụng theo thứ tự từ ngoài vào trong đối với request, nên rate limit toàn cục chạy trước, rồi đến rate limit theo IP, rồi mới đến auth — tránh tốn công kiểm tra token cho các request đã bị chặn từ trước.
- `into_make_service_with_connect_info::<SocketAddr>()` cần thiết để `ConnectInfo<SocketAddr>` (địa chỉ IP của client) có thể được `extract` trong middleware.
- Middleware auth trả `401 Unauthorized`, middleware rate limit theo IP trả `429 Too Many Requests` — đúng mã lỗi HTTP chuẩn cho từng tình huống, giúp client (Web UI) phân biệt được nguyên nhân lỗi.

## Chương 13.4 — Bài tập

1. Cài đặt đầy đủ Rust gateway ở [mục 13.2](#chương-132--kiến-trúc-rust-gateway-đứng-trước-python-ai-service), chạy song song với Python service `server.py` từ [14-Projects.md](../14-Projects/14-Projects.md). Xác nhận gọi `POST http://localhost:9090/api/chat` trả về đúng câu trả lời như khi gọi trực tiếp `POST http://localhost:8080/api/chat`.
2. Thêm rate limiting và auth như [mục 13.3](#chương-133--rust-gateway-hoàn-chỉnh-hơn-rate-limiting-và-auth). Viết một script (Python hoặc `bash` + `curl`) gửi liên tiếp hơn `PER_IP_LIMIT` request trong vòng 1 giây từ cùng một IP, xác nhận các request vượt ngưỡng nhận `429 Too Many Requests`; đồng thời thử gọi thiếu/sai header `Authorization` để xác nhận nhận `401 Unauthorized`.
3. (Nâng cao) Đo và so sánh throughput giữa gọi trực tiếp FastAPI Python (`:8080`) và gọi qua Rust gateway (`:9090`) bằng công cụ benchmark HTTP như [`wrk`](https://github.com/wg/wrk) hoặc [`hey`](https://github.com/rakyll/hey), ví dụ `hey -z 30s -c 50 http://localhost:9090/api/chat`. Nhận xét: chênh lệch latency/throughput có đáng kể không, và nó đến từ đâu (network hop thêm vào so với lợi ích về I/O concurrency)?

## Chương 13.5 — Tài liệu tham khảo

- [Axum docs](https://docs.rs/axum)
- [Tokio docs](https://tokio.rs)
- [Tower docs](https://docs.rs/tower)
- [Candle (Hugging Face)](https://github.com/huggingface/candle)
- [ort — ONNX Runtime for Rust](https://github.com/pykeio/ort)
