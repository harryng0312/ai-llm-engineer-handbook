# 04 — Transformer & Decoder-only LLM

Chương này gồm hai phần liền mạch. Phần đầu (4.1–4.4) dựng lại kiến trúc **Transformer gốc** (Vaswani et al., 2017): vì sao self-attention ra đời thay cho RNN/LSTM, cách văn bản được biến thành token, cấu trúc đầy đủ Encoder/Decoder, và cơ chế cross-attention nối hai khối đó lại với nhau. Phần sau (4.5–4.9) đi thẳng vào kiến trúc **decoder-only** mà mọi LLM hiện đại (GPT, Llama, Qwen...) đang dùng: self-attention với causal mask, RoPE, KV Cache, GQA/MQA, và cách điều khiển hành vi model bằng prompt engineering. Nên đọc phần đầu trước, vì nhiều khái niệm nền tảng (attention, positional encoding, normalization...) được giới thiệu lần đầu ở đó và được dùng lại xuyên suốt phần sau.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`04-Transformer.ipynb`](./04-Transformer.ipynb).

## Mục lục

1. [Vì sao cần Transformer](#chương-41--vì-sao-cần-transformer)
2. [Tokenization](#chương-42--tokenization)
3. [Kiến trúc Transformer gốc: Encoder và Decoder](#chương-43--kiến-trúc-transformer-gốc-encoder-và-decoder)
4. [Cross-Attention](#chương-44--cross-attention)
5. [Kiến trúc Decoder-only hiện đại](#chương-45--kiến-trúc-decoder-only-hiện-đại)
6. [RoPE — Rotary Positional Embedding](#chương-46--rope)
7. [KV Cache](#chương-47--kv-cache)
8. [GQA/MQA — Grouped/Multi Query Attention](#chương-48--gqamqa)
9. [Prompt Engineering](#chương-49--prompt-engineering)
10. [Bài tập](#chương-410--bài-tập)
11. [Tài liệu tham khảo](#chương-411--tài-liệu-tham-khảo)

---

## Chương 4.1 — Vì sao cần Transformer

Trước 2017, mô hình xử lý chuỗi (dịch máy, sinh văn bản) chủ yếu dùng **RNN/LSTM**: đọc token lần lượt từng bước, mỗi bước cập nhật một "hidden state" tóm tắt toàn bộ lịch sử đã đọc.

```
RNN:  x1 -> [h1] -> x2 -> [h2] -> x3 -> [h3] -> x4 -> [h4] -> ...
             │             │             │             │
        mỗi hidden state phải đợi bước trước tính xong mới bắt đầu được
```

Hai hạn chế cố hữu:

- **Không song song hóa được theo chiều thời gian**: `h4` phụ thuộc `h3`, `h3` phụ thuộc `h2`... nên dù có GPU mạnh, vẫn phải tính tuần tự từng bước — chuỗi càng dài, train càng chậm.
- **Thông tin xa bị "pha loãng"**: token đầu tiên muốn ảnh hưởng tới token thứ 1000 phải "đi qua" 999 bước trung gian, mỗi bước hidden state lại bị nén/ghi đè một phần — LSTM giảm nhẹ vấn đề này (nhờ cơ chế cổng) nhưng không giải quyết triệt để.

**Self-attention** giải quyết cả hai: mọi token có thể "nhìn thẳng" tới mọi token khác trong **1 bước duy nhất** (không qua trung gian), và phép tính giữa các token độc lập nhau nên tính song song được toàn bộ bằng nhân ma trận:

```
Self-Attention:  x1, x2, x3, x4  ─┐
                                  ├─▶  mỗi token attend TRỰC TIẾP tới mọi token khác
                                  │    cùng lúc (1 phép nhân ma trận, không tuần tự)
                                  ┘
Độ dài đường đi thông tin: RNN = O(n) bước | Self-Attention = O(1) bước
```

Đánh đổi: attention tính tương tác giữa **mọi cặp** token nên tốn `O(n²)` bộ nhớ/tính toán theo độ dài chuỗi `n` (đây chính là lý do KV cache và các kỹ thuật tối ưu ở [Chương 4.7](#chương-47--kv-cache) quan trọng), nhưng đổi lại tốc độ train nhanh hơn nhiều nhờ song song hóa, và chất lượng tốt hơn hẳn với chuỗi dài nhờ đường đi thông tin ngắn.

## Chương 4.2 — Tokenization

Mạng neural chỉ làm việc được với số, không phải chữ — bước đầu tiên của mọi pipeline NLP là **tokenization**: cắt văn bản thành các đơn vị rời rạc (token), mỗi token ánh xạ tới 1 số nguyên (token id) tra trong một bảng từ vựng (vocabulary) cố định.

```
"LLM rất thú vị" ──tokenize──▶ ["LL", "M", " rất", " thú", " vị"] ──lookup──▶ [43021, 44, 9012, 15833, 8821]
```

Ba cách tiếp cận, đánh đổi giữa kích thước vocab và độ dài chuỗi:

| Cách tách | Vocab size | Vấn đề |
|---|---|---|
| Theo từ (word-level) | Rất lớn (hàng trăm nghìn) | Từ mới/hiếm không có trong vocab (out-of-vocabulary), đặc biệt nặng với tiếng Việt (nhiều từ ghép) |
| Theo ký tự (character-level) | Rất nhỏ (~100) | Không bao giờ gặp từ lạ, nhưng chuỗi dài gấp nhiều lần, model khó học được ngữ nghĩa từ 1 ký tự đơn lẻ |
| Theo subword (BPE/WordPiece/SentencePiece) | Vừa phải (30k-150k) | Cân bằng: từ phổ biến giữ nguyên 1 token, từ hiếm tách thành các mảnh nhỏ hơn — **lựa chọn của mọi LLM hiện đại** |

**BPE (Byte-Pair Encoding)** là thuật toán subword phổ biến nhất (GPT, Llama, Qwen đều dùng biến thể của nó). Ý tưởng: bắt đầu từ từng ký tự riêng lẻ, lặp lại việc **gộp cặp ký tự/token liền kề xuất hiện nhiều nhất** thành 1 token mới, cho đến khi đạt kích thước vocab mong muốn.

```python
from collections import Counter

def get_pair_counts(corpus: list[list[str]]) -> Counter:
    """Đếm tần suất mọi cặp token liền kề trong corpus."""
    pairs = Counter()
    for word in corpus:
        for a, b in zip(word[:-1], word[1:]):
            pairs[(a, b)] += 1
    return pairs

def merge_pair(corpus: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
    """Gộp mọi lần xuất hiện của `pair` thành 1 token."""
    merged = "".join(pair)
    new_corpus = []
    for word in corpus:
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)
    return new_corpus

def train_bpe(words: list[str], num_merges: int) -> list[tuple[str, str]]:
    """Huấn luyện BPE tối giản: mỗi vòng lặp gộp 1 cặp phổ biến nhất."""
    corpus = [list(w) + ["</w>"] for w in words]  # </w>: đánh dấu kết thúc từ
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(corpus)
        if not pairs:
            break
        best_pair = pairs.most_common(1)[0][0]
        merges.append(best_pair)
        corpus = merge_pair(corpus, best_pair)
    return merges

words = ["low", "lower", "lowest", "newer", "newest", "wider"] * 5
merges = train_bpe(words, num_merges=10)
print(merges)
# ví dụ: [('e', 'r'), ('er', '</w>'), ('l', 'o'), ('lo', 'w'), ('low', '</w>'), ('n', 'e'), ...]
# -> "er" xuất hiện nhiều (lower, newer, wider) nên được gộp thành 1 token sớm
```

So sánh cách các tokenizer thật (đã train sẵn trên corpus khổng lồ) tách cùng một câu — quan sát rõ **tiếng Việt bị tách thành nhiều token hơn tiếng Anh** vì phần lớn tokenizer được train chủ yếu trên dữ liệu tiếng Anh:

```python
from transformers import AutoTokenizer

sentence_en = "Retrieval-Augmented Generation reduces hallucination."
sentence_vi = "Kỹ thuật RAG giúp giảm ảo giác của mô hình ngôn ngữ."

for model_name in ["gpt2", "bert-base-uncased", "Qwen/Qwen2.5-7B-Instruct"]:
    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"--- {model_name} ---")
    print("EN:", tok.tokenize(sentence_en), f"({len(tok.tokenize(sentence_en))} token)")
    print("VI:", tok.tokenize(sentence_vi), f"({len(tok.tokenize(sentence_vi))} token)")
```

Hệ quả thực tế: với cùng một đoạn văn, prompt tiếng Việt thường tốn **nhiều token hơn** tiếng Anh cùng độ dài — ảnh hưởng trực tiếp tới chi phí (nếu dùng API tính phí theo token) và giới hạn context window (xem [Chương 4.7](#chương-47--kv-cache) về chi phí mỗi token chiếm trong KV cache).

**WordPiece** (BERT) và **SentencePiece/Unigram** (T5, Llama, Qwen) là hai biến thể khác của cùng ý tưởng subword — khác nhau chủ yếu ở tiêu chí chọn cặp để gộp (WordPiece tối ưu likelihood thay vì tần suất thuần) và cách xử lý khoảng trắng (SentencePiece coi khoảng trắng như 1 ký tự bình thường, cho phép tokenize được cả ngôn ngữ không phân tách từ bằng dấu cách như tiếng Nhật/Trung).

## Chương 4.3 — Kiến trúc Transformer gốc: Encoder và Decoder

Bản Transformer nguyên thủy (Vaswani et al., 2017) được thiết kế cho bài toán dịch máy: **Encoder** đọc và mã hóa toàn bộ câu nguồn, **Decoder** sinh từng từ của câu đích, có "nhìn" sang thông tin đã mã hóa từ Encoder.

```
Câu nguồn: "I love cats"              Câu đích: "Tôi yêu mèo" (sinh dần)

┌─────────────────────────┐          ┌──────────────────────────────┐
│      ENCODER × N         │          │          DECODER × N          │
│  ┌─────────────────────┐ │          │  ┌──────────────────────────┐ │
│  │ Self-Attention        │ │          │  │ Masked Self-Attention     │ │
│  │ (không có mask,       │ │          │  │ (causal mask, chỉ nhìn    │ │
│  │  nhìn được cả câu)    │ │          │  │  token đích đã sinh)      │ │
│  └─────────────────────┘ │          │  └──────────────────────────┘ │
│           │               │          │             │                 │
│  ┌─────────────────────┐ │          │  ┌──────────────────────────┐ │
│  │ FeedForward           │ │  ──────▶│  │ Cross-Attention            │ │
│  └─────────────────────┘ │  encoder │  │ (Q từ decoder, K/V từ      │ │
│                           │  output  │  │  output Encoder)           │ │
└─────────────────────────┘          │  └──────────────────────────┘ │
                                       │             │                 │
                                       │  ┌──────────────────────────┐ │
                                       │  │ FeedForward                │ │
                                       │  └──────────────────────────┘ │
                                       └──────────────────────────────┘
                                                     │
                                                     ▼
                                          Linear + Softmax -> "Tôi", "yêu", "mèo"
```

So với [Chương 4.5](#chương-45--kiến-trúc-decoder-only-hiện-đại), bản gốc này khác ở 3 điểm mà bảng so sánh ở chương đó đã nêu tên nhưng chưa giải thích cơ chế — giải thích chi tiết ở đây:

**Positional Encoding kiểu sinusoidal** — trước khi RoPE ra đời ([Chương 4.6](#chương-46--rope)), vị trí được mã hóa bằng cách **cộng thẳng** một vector cố định (không học được, tính sẵn bằng công thức sin/cos) vào embedding token trước khi vào layer đầu tiên:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

```python
import numpy as np

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    position = np.arange(seq_len)[:, None]                       # (seq_len, 1)
    i = np.arange(d_model)[None, :]                                # (1, d_model)
    angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))
    angles = position * angle_rates                                # (seq_len, d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

pe = sinusoidal_positional_encoding(seq_len=6, d_model=8)
print(pe.round(2))
```

Khác biệt cốt lõi với RoPE: sinusoidal **cộng** vào embedding một lần duy nhất trước layer đầu tiên (thông tin vị trí có thể "phai" dần qua nhiều layer); RoPE **xoay** trực tiếp Q/K ở **mỗi layer** attention, và mã hóa được vị trí *tương đối* thay vì tuyệt đối — đây là lý do các LLM hiện đại đều chuyển sang RoPE.

**LayerNorm & post-norm** — bản gốc chuẩn hóa theo kiểu **post-norm**: tính sublayer trước, cộng residual, rồi mới chuẩn hóa:

```
Post-norm (gốc, 2017):   x -> Sublayer(x) -> +x -> LayerNorm -> output
Pre-norm  (LLM hiện đại): x -> LayerNorm -> Sublayer(.) -> +x -> output
```

Pre-norm (chuẩn hóa **trước** khi vào sublayer, xem lại block diagram ở [Chương 4.5](#chương-45--kiến-trúc-decoder-only-hiện-đại)) giúp gradient truyền thẳng qua nhánh residual ổn định hơn khi xếp chồng rất nhiều layer (hàng chục đến hàng trăm) — lý do mọi LLM sâu hiện nay đều dùng pre-norm dù bản gốc 2017 dùng post-norm.

**FeedForward Network (FFN)** — áp dụng **độc lập cho từng vị trí token** (không trộn thông tin giữa các token, việc đó là nhiệm vụ của attention), gồm 2 lớp Linear với ReLU ở giữa, mở rộng chiều rồi thu lại:

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # mở rộng, ví dụ d_ff = 4 * d_model
        self.fc2 = nn.Linear(d_ff, d_model)   # thu lại đúng d_model để cộng residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

ffn = FeedForward(d_model=512, d_ff=2048)
x = torch.randn(1, 6, 512)   # (batch, seq_len, d_model)
print(ffn(x).shape)          # torch.Size([1, 6, 512]) -- shape không đổi, chỉ "làm giàu" biểu diễn
```

LLM hiện đại thay ReLU bằng **SwiGLU/GeGLU** (đã nhắc ở [Chương 4.5](#chương-45--kiến-trúc-decoder-only-hiện-đại)) — về cấu trúc vẫn là "mở rộng rồi thu lại", chỉ khác activation function giúp mô hình học biểu diễn tốt hơn ở cùng số tham số.

## Chương 4.4 — Cross-Attention

Self-attention (đã học ở [Chương 4.5](#chương-45--kiến-trúc-decoder-only-hiện-đại)) có Q, K, V đều tính từ **cùng một chuỗi**. **Cross-attention** — thành phần riêng của kiến trúc encoder-decoder — có Q tính từ chuỗi decoder, còn K và V tính từ **output của Encoder**:

```
Self-Attention:                       Cross-Attention:
  Q, K, V đều từ decoder                Q từ decoder | K, V từ Encoder output
  ┌───┐ ┌───┐ ┌───┐                     ┌───┐          ┌───┐ ┌───┐
  │ Q │ │ K │ │ V │                     │ Q │          │ K │ │ V │
  └─┬─┘ └─┬─┘ └─┬─┘                     └─┬─┘          └─┬─┘ └─┬─┘
    └──────┴──────┘                       └──────┬───────┴──────┘
     cùng 1 nguồn (decoder)                  2 nguồn khác nhau (decoder Q, encoder K/V)
```

Về mặt code, cross-attention dùng **đúng công thức** scaled dot-product attention ở [Chương 4.5](#chương-45--kiến-trúc-decoder-only-hiện-đại) — chỉ khác nguồn của K, V:

```python
import torch
import torch.nn.functional as F

def cross_attention(decoder_q, encoder_k, encoder_v, mask=None):
    """decoder_q: (batch, n_heads, tgt_len, head_dim) -- từ chuỗi decoder
    encoder_k, encoder_v: (batch, n_heads, src_len, head_dim) -- từ output Encoder
    tgt_len và src_len KHÔNG cần bằng nhau (câu đích dài/ngắn khác câu nguồn đều được).
    """
    d_k = decoder_q.shape[-1]
    scores = decoder_q @ encoder_k.transpose(-2, -1) / (d_k ** 0.5)   # (batch, n_heads, tgt_len, src_len)
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    return weights @ encoder_v                                        # (batch, n_heads, tgt_len, head_dim)

batch, n_heads, head_dim = 1, 4, 16
src_len, tgt_len = 5, 3   # câu nguồn 5 token, câu đích đang sinh dở 3 token -- độ dài khác nhau, vẫn hoạt động
decoder_q = torch.randn(batch, n_heads, tgt_len, head_dim)
encoder_k = torch.randn(batch, n_heads, src_len, head_dim)
encoder_v = torch.randn(batch, n_heads, src_len, head_dim)

output = cross_attention(decoder_q, encoder_k, encoder_v)
print(output.shape)  # torch.Size([1, 4, 3, 16]) -- theo tgt_len (3), không theo src_len
```

Vì sao decoder-only (xem [Chương 4.5](#chương-45--kiến-trúc-decoder-only-hiện-đại)) **không cần** cross-attention: vì không có Encoder riêng — toàn bộ ngữ cảnh (bao gồm cả thứ tương đương "câu nguồn" như system prompt, tài liệu RAG...) được nhét thẳng vào **cùng một chuỗi token** với phần sinh ra, xử lý bằng self-attention thông thường. Đây là lý do kiến trúc decoder-only đơn giản hơn (ít loại attention hơn) nhưng vẫn xử lý được các bài toán từng cần encoder-decoder (dịch máy, tóm tắt) — chỉ cần đưa "câu nguồn" vào prompt thay vì qua một Encoder riêng.

## Chương 4.5 — Kiến trúc Decoder-only hiện đại

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

## Chương 4.6 — RoPE

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

## Chương 4.7 — KV Cache

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

# Cấu hình gần giống Qwen2.5-7B: 28 layer, GQA với 4 KV head (xem Chương 4.8), head_dim 128
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

Chi tiết cách vLLM và các server inference khác triển khai serving thực tế (bao gồm PagedAttention, continuous batching) được nói tiếp ở [12-Deployment.md](../12-Deployment/12-Deployment.md).

## Chương 4.8 — GQA/MQA

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

## Chương 4.9 — Prompt Engineering

Với model đã train xong (không fine-tune thêm), prompt là công cụ chính để điều khiển hành vi. Một vài kỹ thuật cốt lõi, minh họa qua model chạy local bằng Ollama (xem cách cài ở [12-Deployment.md](../12-Deployment/12-Deployment.md)):

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
- Prompt càng dài, càng tốn KV cache/độ trễ (xem [Chương 4.7](#chương-47--kv-cache)) — không nhồi nhét ngữ cảnh thừa.

## Chương 4.10 — Bài tập

1. Chạy `train_bpe` ở [Chương 4.2](#chương-42--tokenization) với `num_merges` tăng dần (10, 50, 100), in ra vocab cuối cùng, quan sát các từ phổ biến trong danh sách `words` dần được gộp thành 1 token duy nhất từ lúc nào.
2. So sánh tokenizer của 3 model (`gpt2`, `bert-base-uncased`, `Qwen/Qwen2.5-7B-Instruct`) trên 5 câu tiếng Việt khác nhau, tính tỉ lệ "số token / số từ" trung bình cho mỗi tokenizer — tokenizer nào hiệu quả nhất với tiếng Việt?
3. Vẽ (bằng `matplotlib` hoặc in ra dạng số) ma trận `sinusoidal_positional_encoding(seq_len=50, d_model=64)`, quan sát các cột ứng với `i` nhỏ dao động nhanh còn `i` lớn dao động chậm — so sánh với nhận xét tương tự về tần số của RoPE ở [Chương 4.6](#chương-46--rope).
4. Cài đặt đầy đủ 1 Encoder block (self-attention + FFN, post-norm) bằng PyTorch dựa trên `scaled_dot_product_attention` đã có ở [Chương 4.5](#chương-45--kiến-trúc-decoder-only-hiện-đại) và `FeedForward` ở [Chương 4.3](#chương-43--kiến-trúc-transformer-gốc-encoder-và-decoder), test forward pass với input ngẫu nhiên.
5. Mở rộng bài tập 4: thêm 1 Decoder block hoàn chỉnh (masked self-attention + cross-attention + FFN) dùng `cross_attention` ở [Chương 4.4](#chương-44--cross-attention), ghép Encoder + Decoder thành 1 mô hình encoder-decoder tối giản.
6. Sửa `generate_with_cache` ở [Chương 4.7](#chương-47--kv-cache) để in ra số token/giây, so sánh với `generate_naive` khi `n_new_tokens` = 20, 50, 100 — vẽ biểu đồ tốc độ theo độ dài chuỗi.
7. Thử nghiệm prompt engineering: với cùng một bài toán suy luận, so sánh độ chính xác giữa zero-shot và chain-of-thought trên 10 câu hỏi khác nhau, tự chấm điểm.

## Chương 4.11 — Tài liệu tham khảo

> 📖 Xem chú giải chi tiết "vì sao mỗi paper quan trọng" ở [15-Papers §2](../15-Papers/15-Papers.md#chương-152--nền-tảng-transformer--llm) (Transformer, tokenization, RoPE, GQA) và [§8](../15-Papers/15-Papers.md#chương-158--lịch-sử--scaling-laws) (BERT, GPT, T5).

- Vaswani et al., *Attention Is All You Need* (2017)
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE, 2015)
- Kudo & Richardson, *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing* (2018)
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018)
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1, 2018)
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2019)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) · [Hugging Face Tokenizers docs](https://huggingface.co/docs/tokenizers)
