# 05 — LLM: Decoder-only Transformer

> Chương này trả lời câu hỏi: "Bên trong một LLM hiện đại (GPT, Qwen, Llama...) có gì, và tại sao nó được thiết kế như vậy?" Mục tiêu là hiểu đủ sâu để đọc code inference thật, tự chạy một model local bằng Ollama, và biết khi nào cần LoRA/QLoRA để fine-tune.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`05-LLM.ipynb`](./05-LLM.ipynb).

## Mục lục

1. [Kiến trúc Decoder-only](#1-kiến-trúc-decoder-only)
2. [RoPE — Rotary Positional Embedding](#2-rope--rotary-positional-embedding)
3. [KV Cache](#3-kv-cache)
4. [GQA/MQA — Grouped/Multi Query Attention](#4-gqamqa--groupedmulti-query-attention)
5. [Prompt Engineering](#5-prompt-engineering)
6. [LoRA/QLoRA](#6-loraqlora)
7. [Serving: Ollama, vLLM, llama.cpp](#7-serving-ollama-vllm-llamacpp)
8. [Bài tập](#8-bài-tập)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Kiến trúc Decoder-only

Hầu hết LLM sinh văn bản ngày nay (GPT-x, Llama, Qwen, Mistral...) đều là **decoder-only Transformer**: chỉ dùng phần decoder của kiến trúc Transformer gốc (Vaswani et al., 2017), bỏ hẳn encoder. Lý do:

- Bài toán chính là **next-token prediction** — sinh token tiếp theo dựa trên toàn bộ ngữ cảnh phía trước. Encoder-decoder (như T5) hợp với task có input/output tách biệt rõ (dịch máy, tóm tắt), còn decoder-only linh hoạt hơn cho hội thoại, code, reasoning vì input và output nằm chung một chuỗi token.
- Kiến trúc đơn giản hơn → dễ scale, dễ train song song.

Một block decoder điển hình (kiểu Llama/Qwen) gồm:

```
x -> RMSNorm -> Self-Attention (causal, RoPE) -> +residual
  -> RMSNorm -> FeedForward (SwiGLU)          -> +residual
```

Điểm khác biệt so với Transformer gốc (2017) mà các LLM hiện đại đều áp dụng:

| Thành phần | Transformer gốc | LLM hiện đại (Llama/Qwen) |
|---|---|---|
| Positional encoding | Sinusoidal cộng vào input | RoPE, áp vào Q/K trong mỗi layer |
| Normalization | LayerNorm, post-norm | RMSNorm, pre-norm |
| Activation FFN | ReLU | SwiGLU/GeGLU |
| Attention | Multi-Head Attention (MHA) | GQA hoặc MQA (tiết kiệm KV cache) |
| Attention mask | — | Causal mask (chỉ nhìn token quá khứ) |

### Bên trong Self-Attention: Scaled Dot-Product

Attention cho phép mỗi token "hỏi" các token khác để lấy thông tin liên quan. Từ embedding của mỗi token, ba ma trận trọng số học được chiếu ra 3 vector: Query `Q` (câu hỏi token đang cần), Key `K` (nhãn để token khác biết có đáng trả lời không), Value `V` (nội dung thực sự lấy về nếu liên quan):

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k  +  mask ) · V
```

Sơ đồ luồng dữ liệu của 1 attention head:

```
        Q (seq×d_k)       K (seq×d_k)        V (seq×d_k)
            │                   │                  │
            └─────────┬─────────┘                  │
                       ▼                            │
               Q · Kᵀ   (seq×seq)                   │
                       │                            │
                       ▼                            │
                  chia cho √d_k                     │
                       │                            │
                       ▼                            │
            + causal mask (-inf phía trên)          │
                       │                            │
                       ▼                            │
                 softmax (theo hàng)                 │
                       │                            │
                       └─────────────┬──────────────┘
                                     ▼
                         nhân với V (tổng có trọng số)
                                     │
                                     ▼
                        Attention output (seq×d_k)
```

Vì sao chia cho `√d_k`: khi `d_k` lớn, tích vô hướng `Q·Kᵀ` có phương sai tăng theo `d_k`, khiến softmax bị "bão hòa" (1 giá trị gần 1, còn lại gần 0) → gradient gần như triệt tiêu khi backprop. Chia cho `√d_k` giữ phương sai ổn định bất kể kích thước chiều.

**Multi-Head Attention**: thay vì attention 1 lần trên toàn bộ `d_model` chiều, chia thành `h` "đầu" (head) song song, mỗi đầu hoạt động trên 1 không gian con `d_k = d_model / h` chiều, rồi ghép (concat) kết quả và chiếu qua ma trận `W_O`:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    """q, k, v: (batch, n_heads, seq, head_dim)"""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)   # (batch, n_heads, seq, seq)
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    return weights @ v                                 # (batch, n_heads, seq, head_dim)

batch, n_heads, seq, head_dim = 1, 4, 6, 16
q = torch.randn(batch, n_heads, seq, head_dim)
k = torch.randn(batch, n_heads, seq, head_dim)
v = torch.randn(batch, n_heads, seq, head_dim)

mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
output = scaled_dot_product_attention(q, k, v, mask=mask)
print(output.shape)  # torch.Size([1, 4, 6, 16]) -- mỗi head xử lý độc lập, ghép lại thành d_model ở bước sau
```

Lý do dùng nhiều head nhỏ thay vì 1 head lớn: mỗi head có thể học "chú ý" theo một khía cạnh quan hệ khác nhau giữa các token (ví dụ 1 head học quan hệ cú pháp giữa các từ gần nhau, 1 head khác học quan hệ ngữ nghĩa ở xa) — thực nghiệm cho thấy nhiều head nhỏ tổng quát hóa tốt hơn 1 head lớn có cùng tổng số tham số.

**Causal mask** là điểm mấu chốt khiến decoder-only "chỉ nhìn về quá khứ": khi tính attention score giữa token thứ `i` và `j`, nếu `j > i` thì score bị gán `-inf` trước softmax, đảm bảo token thứ `i` không "nhìn thấy tương lai".

```python
import torch

def causal_mask(seq_len: int) -> torch.Tensor:
    """Trả về ma trận mask (seq_len, seq_len): 0 = được nhìn, -inf = bị chặn."""
    mask = torch.full((seq_len, seq_len), float("-inf"))
    return torch.triu(mask, diagonal=1)  # giữ phần trên đường chéo chính = -inf

print(causal_mask(4))
# tensor([[0., -inf, -inf, -inf],
#         [0., 0., -inf, -inf],
#         [0., 0., 0., -inf],
#         [0., 0., 0., 0.]])
```

Trực quan hóa "token nào nhìn thấy token nào" với câu "the cat sat on the mat" (✓ = được attend, `·` = bị causal mask chặn):

```
            the   cat   sat   on    the   mat
   the       ✓     ·     ·     ·     ·     ·
   cat       ✓     ✓     ·     ·     ·     ·
   sat       ✓     ✓     ✓     ·     ·     ·
   on        ✓     ✓     ✓     ✓     ·     ·
   the       ✓     ✓     ✓     ✓     ✓     ·
   mat       ✓     ✓     ✓     ✓     ✓     ✓
```

Mỗi hàng là một token đang "hỏi", mỗi cột là token có thể được nhìn thấy — nhận thấy ma trận luôn là tam giác dưới (kể cả đường chéo): token thứ `i` chỉ nhìn được chính nó và các token đứng trước.

### Sơ đồ toàn cảnh một forward pass

```
"The cat sat" (token ids)
        │
        ▼
┌─────────────────────────┐
│    Token Embedding       │   lookup bảng (vocab_size × d_model)
└─────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│               Decoder Block × N                 │
│   ┌─────────────────────────────────────────┐  │
│   │ RMSNorm                                   │  │
│   │   │                                        │  │
│   │   ▼                                        │  │
│   │ Causal Self-Attention (Q,K áp RoPE)  ─────┼──┼──▶ (+) residual
│   │   │                                        │  │
│   │ RMSNorm                                   │  │
│   │   │                                        │  │
│   │   ▼                                        │  │
│   │ FeedForward (SwiGLU)                 ─────┼──┼──▶ (+) residual
│   └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│      Final RMSNorm       │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Linear (d_model→vocab)  │
│        + Softmax          │
└─────────────────────────┘
        │
        ▼
 P(token tiếp theo | "The cat sat")  →  ví dụ: "on" (xác suất cao nhất)
```

## 2. RoPE — Rotary Positional Embedding

Vấn đề: attention không có khái niệm "thứ tự" tự nhiên (nó là tổng có trọng số, hoán vị token thì kết quả không đổi). Ta cần bơm thông tin vị trí vào.

RoPE (Su et al., 2021) không cộng một vector vị trí vào embedding như cách cũ, mà **xoay** (rotate) từng cặp chiều của vector Query/Key một góc tỉ lệ với vị trí token. Lợi ích lớn nhất: tích vô hướng `Q·K` sau khi xoay chỉ phụ thuộc vào **vị trí tương đối** `(m - n)` giữa hai token, không phụ thuộc vị trí tuyệt đối — giúp model tổng quát tốt hơn với chuỗi dài hơn lúc train.

Công thức (cho một cặp chiều `(x1, x2)` ở vị trí `m`, tần số `θ`):

```
x1' = x1 * cos(mθ) - x2 * sin(mθ)
x2' = x1 * sin(mθ) + x2 * cos(mθ)
```

Hình dung phép xoay này trong mặt phẳng 2 chiều `(x1, x2)`: độ dài vector giữ nguyên, chỉ quay đi một góc `mθ` tỉ lệ thuận với vị trí token `m` — token càng đứng xa nhau, góc lệch giữa 2 vector Q/K sau khi xoay càng khác nhau, nhờ đó tích `Q·K` "đọc" được khoảng cách tương đối mà không cần cộng thêm bất kỳ vector vị trí nào:

```
 Token ở vị trí m=0 (góc xoay = 0):        Token ở vị trí m=5 (góc xoay = 5θ):

        x2                                        x2
        │                                          │       ↗ vector đã xoay
        │   ↗ (x1, x2)                             │     ╱   (x1', x2')
        │  ╱                                       │   ╱ 5θ
        │ ╱                                        │ ╱______
        └──────────────── x1                       └──────────────── x1
```

Cài đặt tối giản bằng NumPy để thấy rõ cơ chế xoay:

```python
import numpy as np

def rope_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """theta_i cho từng cặp chiều, giống công thức trong paper RoPE / code Llama."""
    i = np.arange(0, dim, 2)
    return 1.0 / (base ** (i / dim))

def apply_rope(x: np.ndarray, position: int, base: float = 10000.0) -> np.ndarray:
    """x: vector 1 token, shape (dim,). Xoay từng cặp chiều theo vị trí `position`."""
    dim = x.shape[-1]
    theta = rope_frequencies(dim, base)          # (dim/2,)
    angles = position * theta                     # góc xoay cho từng cặp
    x1, x2 = x[0::2], x[1::2]                      # tách các cặp (even, odd)
    cos, sin = np.cos(angles), np.sin(angles)
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    out = np.empty_like(x)
    out[0::2], out[1::2] = x1_rot, x2_rot
    return out

q = np.random.randn(8)
q_at_pos_0 = apply_rope(q, position=0)
q_at_pos_5 = apply_rope(q, position=5)
print(np.allclose(q_at_pos_0, q))  # True: vị trí 0 -> góc xoay = 0, vector giữ nguyên
```

Mỗi cặp chiều trong vector có một tần số `θ_i` riêng — cặp chiều đầu quay rất nhanh theo vị trí (bắt thông tin vị trí *cục bộ*, giữa các token sát nhau), cặp chiều cuối quay rất chậm (bắt thông tin vị trí *toàn cục*, giữa các token ở xa) — giống cách một chiếc đồng hồ dùng cả kim giây (nhanh) lẫn kim giờ (chậm) để mã hóa thời gian không nhập nhằng trong một phạm vi rộng:

```
cặp chiều (0, 1):      θ_0 = 1/10000^(0/d)        ≈ 1.0      → quay nhanh   (như kim giây)
cặp chiều (2, 3):      θ_1 = 1/10000^(2/d)        ≈ 0.316    → quay vừa
        ...
cặp chiều (d-2, d-1):  θ_(d/2-1) = 1/10000^((d-2)/d) ≈ 0.0001 → quay rất chậm (như kim giờ)
```

Trực giác quan trọng cần nhớ: RoPE áp dụng lên **Q và K trước khi tính attention**, không áp lên V. Nhờ tính chất "chỉ phụ thuộc vị trí tương đối", các kỹ thuật kéo dài context (NTK-aware scaling, YaRN) đều là các biến thể chỉnh lại `base`/`theta` của RoPE.

## 3. KV Cache

Khi sinh text autoregressive (từng token một), nếu không cache, mỗi bước sinh token mới ta phải chạy lại **toàn bộ** self-attention cho tất cả token trước đó — độ phức tạp `O(n²)` cho toàn chuỗi độ dài `n`, rất lãng phí vì Key/Value của các token cũ không hề đổi.

**KV Cache**: lưu lại Key và Value đã tính của mọi token trước đó. Ở bước sinh token mới, chỉ cần tính Q/K/V cho **1 token mới**, rồi attention token đó với toàn bộ K/V đã cache — độ phức tạp mỗi bước giảm từ `O(n²)` xuống `O(n)`.

```
Bước 1 — sinh "cat":   K cache: [k_the]                           V cache: [v_the]
Bước 2 — sinh "sat":   K cache: [k_the, k_cat]                    V cache: [v_the, v_cat]
Bước 3 — sinh "on":    K cache: [k_the, k_cat, k_sat]             V cache: [v_the, v_cat, v_sat]
Bước 4 — sinh "the":   K cache: [k_the, k_cat, k_sat, k_on]       V cache: [v_the, v_cat, v_sat, v_on]
                        └── mỗi bước chỉ TÍNH THÊM 1 cặp (k, v) mới, không tính lại các cặp đã có trong cache
```

So sánh độ phức tạp cho toàn bộ quá trình sinh `n` token, không chỉ 1 bước:

```
KHÔNG cache — mỗi bước tính lại attention cho TOÀN BỘ chuỗi đã sinh tới lúc đó:
  bước 1: attend trên 1 token  -> O(1²)
  bước 2: attend trên 2 token  -> O(2²)
  ...
  bước n: attend trên n token  -> O(n²)
  Tổng: O(1² + 2² + ... + n²) = O(n³)

CÓ cache — mỗi bước chỉ tính Q/K/V cho 1 token mới rồi attend với cache có sẵn:
  bước 1: O(1)   bước 2: O(2)   ...   bước n: O(n)
  Tổng: O(1 + 2 + ... + n) = O(n²)     -- giảm hẳn 1 bậc so với không cache
```

Minh họa sự khác biệt tốc độ bằng model nhỏ (chạy được trên CPU) với thư viện `transformers`:

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"  # model nhỏ để demo trên CPU
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "The future of artificial intelligence is"
input_ids = tok(prompt, return_tensors="pt").input_ids

def generate_naive(input_ids, n_new_tokens=40):
    """KHÔNG dùng cache: mỗi bước feed lại toàn bộ chuỗi đã sinh."""
    ids = input_ids
    with torch.no_grad():
        for _ in range(n_new_tokens):
            logits = model(ids, use_cache=False).logits
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=-1)
    return ids

def generate_with_cache(input_ids, n_new_tokens=40):
    """CÓ cache: chỉ feed token mới nhất mỗi bước, tái sử dụng past_key_values."""
    ids = input_ids
    past = None
    with torch.no_grad():
        for _ in range(n_new_tokens):
            out = model(ids if past is None else ids[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=-1)
    return ids

t0 = time.time(); generate_naive(input_ids); t1 = time.time()
generate_with_cache(input_ids); t2 = time.time()
print(f"Không cache: {t1 - t0:.2f}s | Có cache: {t2 - t1:.2f}s")
```

Chạy thử sẽ thấy bản có cache nhanh hơn rõ rệt khi số token sinh ra càng lớn. Đây cũng là lý do các server inference (vLLM, TGI, llama.cpp) đều tối ưu quản lý KV cache (paged attention, quantized KV cache...) — vì ở batch lớn, **KV cache chính là phần chiếm nhiều VRAM nhất**, không phải trọng số model.

### KV cache chiếm bao nhiêu VRAM?

```
KV cache (bytes) = 2 (K và V) × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes/phần_tử
```

```python
def kv_cache_size_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per_elem=2):
    """bytes_per_elem=2 cho fp16/bf16 -- kiểu dữ liệu phổ biến nhất khi serving."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem

# Cấu hình gần giống Qwen2.5-7B: 28 layer, GQA với 4 KV head (xem mục 4), head_dim 128
for seq_len in (4_096, 32_768, 131_072):
    size = kv_cache_size_bytes(n_layers=28, n_kv_heads=4, head_dim=128, seq_len=seq_len)
    print(f"context {seq_len:>7,} token -> {size / 1024**3:.2f} GB KV cache (1 request)")

# context   4,096 token -> 0.22 GB KV cache (1 request)
# context  32,768 token -> 1.75 GB KV cache (1 request)
# context 131,072 token -> 7.00 GB KV cache (1 request)
```

Trọng số Qwen2.5-7B ở fp16 chiếm khoảng 14GB **cố định**, nhưng KV cache tăng tuyến tính theo *cả* độ dài context *lẫn* số request chạy đồng thời — đây là lý do phục vụ nhiều user với context dài là bài toán khó về bộ nhớ hơn nhiều so với chỉ việc load xong trọng số model.

**PagedAttention** (dùng trong vLLM) giải quyết lãng phí bộ nhớ khi cấp phát KV cache: thay vì cấp phát trước 1 vùng nhớ liên tục cho độ dài tối đa có thể xảy ra (lãng phí nếu câu trả lời thực tế ngắn hơn nhiều), bộ nhớ được chia thành các "block" cố định và cấp phát động — giống hệt cơ chế virtual memory paging của hệ điều hành:

```
Cấp phát truyền thống (lãng phí nếu request ngắn hơn max_len dự phòng):
  Request A: [■■■■■■■■□□□□□□□□□□□□□□□□]    ■ = đã dùng, □ = cấp sẵn nhưng chưa dùng tới

PagedAttention — chia thành block cố định, chỉ cấp đúng số block đang cần:
  Request A: [■■■■][■■■□]                   2 block, block cuối dùng 3/4
  Request B: [■■■■][■■■■][■■□□]             3 block
                     ▲
        2 request có thể DÙNG CHUNG 1 block nếu trùng prefix
        (ví dụ cùng system prompt) -> tiết kiệm thêm bộ nhớ
```

## 4. GQA/MQA — Grouped/Multi Query Attention

KV cache càng lớn thì càng tốn VRAM. Một cách giảm kích thước KV cache: giảm **số lượng KV head** so với số Query head.

| Kiểu attention | Số KV head | Đặc điểm |
|---|---|---|
| MHA (Multi-Head Attention) | = số Q head | Chất lượng tốt nhất, KV cache lớn nhất |
| MQA (Multi-Query Attention) | 1 | KV cache nhỏ nhất, có thể giảm chất lượng |
| GQA (Grouped-Query Attention) | vài nhóm (vd 8, mỗi nhóm dùng chung 1 KV head) | Cân bằng giữa MHA và MQA — Llama 2/3, Qwen2 đều dùng GQA |

Ý tưởng: nhóm nhiều Query head lại để **dùng chung** một cặp Key/Value head, thay vì mỗi Query head có KV riêng.

```
MHA — mỗi Q head có 1 KV head riêng, không chia sẻ:
  Q head:   Q1   Q2   Q3   Q4   Q5   Q6   Q7   Q8
  KV head:  K1   K2   K3   K4   K5   K6   K7   K8
            (8 KV head -> KV cache lớn nhất, chất lượng tốt nhất)

GQA — mỗi NHÓM Q head dùng chung 1 KV head:
  Q head:   Q1  Q2 │ Q3  Q4 │ Q5  Q6 │ Q7  Q8
             └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
  KV head:     K1        K2        K3        K4
            (4 KV head -> giảm 1/2 KV cache so với MHA, chất lượng gần như không đổi)

MQA — MỌI Q head dùng chung đúng 1 KV head:
  Q head:   Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8
             └─────────────────┬─────────────────┘
  KV head:                    K1
            (1 KV head -> KV cache nhỏ nhất, có thể đánh đổi 1 phần chất lượng)
```

```python
import torch

def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """kv: (batch, n_kv_heads, seq, head_dim) -> lặp mỗi kv head n_rep lần
    để khớp số lượng Q head, mô phỏng cách GQA/MQA triển khai trong thực tế
    (Llama, Qwen): không lưu KV cho từng Q head, chỉ lưu it hơn rồi "broadcast".
    """
    batch, n_kv_heads, seq, head_dim = kv.shape
    if n_rep == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq, head_dim)
    return kv.reshape(batch, n_kv_heads * n_rep, seq, head_dim)

n_q_heads, n_kv_heads, head_dim, seq, batch = 32, 8, 128, 16, 1
k = torch.randn(batch, n_kv_heads, seq, head_dim)
k_expanded = repeat_kv(k, n_rep=n_q_heads // n_kv_heads)
print(k_expanded.shape)  # torch.Size([1, 32, 16, 128]) -> khớp số Q head

# So sánh dung lượng KV cache thực tế phải lưu (không tính phần "expand" ảo):
mha_kv_size = n_q_heads * seq * head_dim
gqa_kv_size = n_kv_heads * seq * head_dim
print(f"MHA lưu {mha_kv_size} phần tử/layer, GQA (8 nhóm) chỉ lưu {gqa_kv_size} -> giảm {mha_kv_size/gqa_kv_size:.0f}x")
```

Qwen2.5 (dùng trong ví dụ ở chương này) dùng GQA; đây là lý do các model 7B–14B chạy được trên GPU 12–24GB mà vẫn giữ context dài.

## 5. Prompt Engineering

Với model đã train xong (không fine-tune thêm), prompt là công cụ chính để điều khiển hành vi. Một vài kỹ thuật cốt lõi, minh họa qua model chạy local bằng Ollama (xem cách cài ở [mục 7](#7-serving-ollama-vllm-llamacpp)):

```python
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

def chat(messages: list[dict], **kwargs) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": kwargs,
    })
    resp.raise_for_status()
    return resp.json()["message"]["content"]
```

**Zero-shot** — chỉ đưa instruction, không kèm ví dụ:

```python
print(chat([
    {"role": "user", "content": "Phân loại cảm xúc của câu sau là tích cực/tiêu cực: 'Sản phẩm dùng tệ, hỏng sau 2 ngày.'"}
]))
```

**Few-shot** — cho vài ví dụ mẫu để model bắt chước format/phong cách:

```python
print(chat([
    {"role": "user", "content": "Review: 'Giao hàng nhanh, đóng gói cẩn thận.' -> Cảm xúc: Tích cực"},
    {"role": "user", "content": "Review: 'Chờ 2 tuần mới nhận được hàng.' -> Cảm xúc: Tiêu cực"},
    {"role": "user", "content": "Review: 'Chất lượng đúng như mô tả, sẽ mua lại.' -> Cảm xúc:"},
]))
```

**Chain-of-Thought (CoT)** — yêu cầu model "suy nghĩ từng bước" trước khi trả lời, cải thiện đáng kể độ chính xác cho bài toán logic/toán:

```python
print(chat([
    {"role": "system", "content": "Hãy suy nghĩ từng bước (step by step) trước khi đưa ra đáp án cuối cùng."},
    {"role": "user", "content": "Một cửa hàng có 23 táo. Bán 12 quả buổi sáng, nhập thêm 15 quả buổi chiều, rồi bán tiếp 9 quả. Hỏi còn lại bao nhiêu táo?"},
]))
```

**System prompt** — định hình vai trò/ràng buộc xuyên suốt hội thoại (persona, ngôn ngữ, format output):

```python
SYSTEM = (
    "Bạn là trợ lý kỹ thuật, luôn trả lời bằng tiếng Việt, súc tích, "
    "và khi liệt kê phải dùng gạch đầu dòng."
)
print(chat([
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "So sánh REST và gRPC."},
]))
```

Nguyên tắc thực dụng khi viết prompt production:
- Đặt **ràng buộc format** rõ ràng (ví dụ "chỉ trả về JSON, không thêm text khác") nếu output sẽ được code parse.
- Few-shot hữu ích khi task có format đặc thù mà model chưa quen; CoT hữu ích khi task cần suy luận nhiều bước.
- Prompt càng dài, càng tốn KV cache/độ trễ (xem [mục 3](#3-kv-cache)) — không nhồi nhét ngữ cảnh thừa.

## 6. LoRA/QLoRA

Khi prompt engineering không đủ (cần model học một domain/style/task riêng), ta fine-tune. Fine-tune full-parameter một LLM 7B cần hàng chục GB VRAM cho gradient + optimizer state — không khả thi trên máy cá nhân.

**LoRA (Low-Rank Adaptation)**: đóng băng toàn bộ trọng số gốc `W`, chỉ train thêm 2 ma trận nhỏ `A (d×r)` và `B (r×d)` với `r << d` (rank thấp, ví dụ r=8, 16), sao cho:

```
W' = W + (α/r) * B @ A
```

Vì `A`, `B` rất nhỏ so với `W`, số tham số cần train giảm hàng trăm–hàng nghìn lần, VRAM cho optimizer cũng giảm tương ứng.

**QLoRA** = LoRA + lượng tử hóa (quantization) trọng số gốc `W` xuống 4-bit (NF4) trước khi train. `W` 4-bit chỉ dùng để forward/backward (dequantize tạm thời khi tính toán), còn `A`, `B` vẫn train ở độ chính xác cao (bf16/fp32). Nhờ vậy, một model 7B có thể fine-tune trên GPU 8–12GB VRAM.

Ví dụ cấu hình LoRA với thư viện `peft` (áp lên model nhỏ để chạy được trên CPU minh họa cơ chế, không phải để có kết quả tốt):

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

lora_config = LoraConfig(
    r=8,                 # rank của A, B
    lora_alpha=16,        # hệ số scale alpha/r
    target_modules=["c_attn"],  # layer nào được chèn LoRA (tùy kiến trúc model)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# ví dụ output: trainable params: 294,912 || all params: 82,207,488 || trainable%: 0.36%
```

Cấu hình QLoRA thực tế cho model 7B (yêu cầu GPU + `bitsandbytes`, chỉ để tham khảo code, không chạy được trên CPU):

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
# sau đó áp LoraConfig + get_peft_model như ví dụ trên, rồi train bằng Trainer/SFTTrainer
```

Khi nào chọn LoRA/QLoRA thay vì prompt engineering: khi cần model **nhớ một format/domain cố định** (ví dụ luôn trả lời theo văn phong công ty, luôn output đúng schema nội bộ), hoặc khi ngữ cảnh cần thiết quá dài để nhét vào mỗi prompt (tốn phí, tốn KV cache) — lúc đó "nướng" kiến thức vào trọng số qua LoRA hiệu quả hơn.

## 7. Serving: Ollama, vLLM, llama.cpp

Ba công cụ phổ biến để chạy LLM, mỗi cái phù hợp một use case:

| Công cụ | Phù hợp khi | Đặc điểm |
|---|---|---|
| **Ollama** | Chạy local để dev/thử nghiệm nhanh | Đóng gói model + runtime (dựa trên llama.cpp), API REST đơn giản, quản lý model như Docker image |
| **llama.cpp** | Chạy trên CPU/máy yếu, cần kiểm soát chi tiết | C/C++ thuần, hỗ trợ quantization GGUF (Q4_K_M, Q5_K_M...), chạy được trên điện thoại/Raspberry Pi |
| **vLLM** | Serving production, nhiều user đồng thời | PagedAttention (quản lý KV cache hiệu quả như virtual memory của OS), continuous batching, throughput cao |

### Cài Ollama và chạy Qwen local

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Tải model Qwen2.5 7B (khoảng 4-5GB, dùng cho các ví dụ trong sách này)
ollama pull qwen2.5:7b

# Chat thử qua CLI
ollama run qwen2.5:7b "Giải thích KV cache trong 3 câu"

# Ollama tự chạy server tại http://localhost:11434
```

Gọi bằng Python với thư viện chính thức `ollama`:

```python
import ollama

response = ollama.chat(model="qwen2.5:7b", messages=[
    {"role": "user", "content": "Tóm tắt sự khác biệt giữa MHA, GQA, MQA trong 3 gạch đầu dòng."}
])
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(model="qwen2.5:7b", messages=[
    {"role": "user", "content": "Viết một đoạn code Python đọc file CSV."}
], stream=True):
    print(chunk["message"]["content"], end="", flush=True)
```

### vLLM (tham khảo, cần GPU)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_compatible_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

vLLM expose một API tương thích OpenAI, nên có thể dùng lại SDK `openai` trỏ `base_url` về server local — hữu ích khi muốn tái sử dụng code viết cho OpenAI API nhưng chạy trên model self-host.

## 8. Bài tập

1. Sửa `generate_with_cache` ở [mục 3](#3-kv-cache) để in ra số token/giây, so sánh với `generate_naive` khi `n_new_tokens` = 20, 50, 100 — vẽ biểu đồ tốc độ theo độ dài chuỗi.
2. Cài Ollama, pull 2 model khác nhau (ví dụ `qwen2.5:7b` và `llama3.1:8b`), viết script so sánh câu trả lời cho cùng 5 câu hỏi kỹ thuật.
3. Thử nghiệm prompt engineering: với cùng một bài toán suy luận, so sánh độ chính xác giữa zero-shot và chain-of-thought trên 10 câu hỏi khác nhau, tự chấm điểm.
4. Đọc code `LoraConfig` của `peft`, thử đổi `target_modules` sang layer khác của `distilgpt2` (in `model` ra để xem tên các layer), quan sát số lượng trainable params thay đổi thế nào.

## 9. Tài liệu tham khảo

- Vaswani et al., *Attention Is All You Need* (2017)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023)
- [Ollama docs](https://github.com/ollama/ollama) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [vLLM docs](https://docs.vllm.ai)
