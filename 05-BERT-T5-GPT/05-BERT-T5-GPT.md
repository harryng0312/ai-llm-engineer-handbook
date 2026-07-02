# 05 — BERT, GPT, T5: Ba dòng kiến trúc Transformer

[04-Transformer.md](../04-Transformer/04-Transformer.md) đã trình bày kiến trúc Transformer gốc (Encoder + Decoder đầy đủ, dùng cho dịch máy) và giới thiệu ngắn gọn ba dòng kiến trúc tách ra từ đó. Chương này đào sâu vào ba dòng đó — **Encoder-only (BERT)**, **Decoder-only (GPT family)**, **Encoder-Decoder (T5/BART)** — vì mỗi dòng phục vụ một mục đích khác hẳn nhau và là nền tảng trực tiếp cho các chương sau: [06-Embedding.md](../06-Embedding/06-Embedding.md) dùng encoder-only để làm model embedding (BGE/E5), còn toàn bộ phần còn lại của sách ([08-Fine-Tuning.md](../08-Fine-Tuning/08-Fine-Tuning.md), [09-LangChain.md](../09-LangChain/09-LangChain.md), [10-LangGraph.md](../10-LangGraph/10-LangGraph.md), [11-Agent.md](../11-Agent/11-Agent.md)) dùng decoder-only (Qwen) để sinh văn bản.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`05-BERT-T5-GPT.ipynb`](./05-BERT-T5-GPT.ipynb).

## Mục lục

1. [Encoder-only: BERT](#chương-51--encoder-only-bert)
2. [Decoder-only: GPT family](#chương-52--decoder-only-gpt-family)
3. [Encoder-Decoder: T5/BART](#chương-53--encoder-decoder-t5bart)
4. [So sánh & lựa chọn](#chương-54--so-sánh--lựa-chọn)
5. [Bài tập](#chương-55--bài-tập)
6. [Tài liệu tham khảo](#chương-56--tài-liệu-tham-khảo)

---

## Chương 5.1 — Encoder-only: BERT

### Kiến trúc

BERT (Bidirectional Encoder Representations from Transformers) chỉ giữ lại **Encoder** của Transformer gốc ([04-Transformer.md §3](../04-Transformer/04-Transformer.md)), xếp chồng nhiều Encoder block liên tiếp, bỏ hẳn Decoder và Cross-Attention. Vì không có causal mask, mỗi token **nhìn được cả hai chiều** — trái lẫn phải — trong cùng một lần forward:

```
BERT (bidirectional) -- token thứ 3 nhìn được CẢ 2 chiều:
   the cat [MASK] on the mat  ->  [MASK] nhìn thấy: the, cat, on, the, mat (toàn câu, không bị che hướng nào)
```

Đây chính là điểm khác biệt cốt lõi so với GPT ở [Chương 5.2](#chương-52--decoder-only-gpt-family): GPT chỉ nhìn được quá khứ (causal), còn BERT nhìn được toàn bộ câu — phù hợp cho các bài toán cần **hiểu** văn bản đã có sẵn đầy đủ, thay vì **sinh** văn bản mới từng token một.

### Masked Language Modeling (MLM)

Vì không có hướng sinh tuần tự (không thể huấn luyện kiểu "đoán token tiếp theo" khi model nhìn được cả hai chiều — nhìn thấy trước "đáp án" thì học không có ý nghĩa), BERT dùng một objective khác: **che ngẫu nhiên 15% token trong câu, bắt model đoán lại đúng token gốc** dựa vào ngữ cảnh hai chiều còn lại.

```
Câu gốc: "The cat sat on the mat"

BERT (Masked LM):
  Input:  "The cat [MASK] on the mat"
  Target: model phải đoán đúng "[MASK]" = "sat"
```

**Vì sao đúng 15%, không phải toàn bộ câu hay chỉ 1-2%?** Đây là sự đánh đổi giữa hai thái cực:

- Nếu che **quá nhiều** (ví dụ 100%): không còn ngữ cảnh nào sót lại để model dựa vào — bài toán trở thành đoán mù, vô nghĩa để học biểu diễn ngôn ngữ.
- Nếu che **quá ít** (ví dụ 1%): mỗi câu chỉ sinh ra rất ít tín hiệu học (1 token/câu), cần nhiều epoch hơn để hội tụ, lãng phí compute vì phần lớn forward pass "không học được gì mới".

15% là con số thực nghiệm của paper gốc (Devlin et al., 2018), cân bằng giữa hai thái cực trên. Ngoài ra, trong số 15% token được chọn che, BERT còn áp dụng thêm một chiến lược 3 phần để giảm lệch pha giữa lúc train và lúc dùng thật (khi fine-tune/inference không hề có token `[MASK]` xuất hiện):

| Trong 15% được chọn | Tỉ lệ | Xử lý |
|---|---|---|
| Thay bằng `[MASK]` | 80% | Trường hợp chính — model học đoán từ ngữ cảnh |
| Thay bằng 1 token ngẫu nhiên khác | 10% | Buộc model không được "ỷ lại" là cứ thấy `[MASK]` mới cần đoán — phải luôn kiểm tra tính hợp lý của mọi token |
| Giữ nguyên token gốc | 10% | Model vẫn phải học biểu diễn tốt cho token đó dù không bị che (vì nó có thể là 1 trong 15% được "chọn" để tính loss) |

### Ứng dụng thực tế

- **Classification**: thêm 1 token đặc biệt `[CLS]` ở đầu chuỗi, lấy vector ẩn cuối cùng của `[CLS]` (đã "gom" thông tin từ toàn câu qua nhiều lớp self-attention) đưa qua 1 lớp Linear nhỏ để phân loại (sentiment, spam, intent...).
- **NER (Named Entity Recognition)**: lấy vector ẩn của **từng token**, phân loại mỗi token thuộc nhãn nào (`B-PER`, `I-ORG`, `O`...) — tận dụng đúng khả năng nhìn ngữ cảnh hai chiều để phân biệt "Washington" là tên người hay tên bang.
- **Nền tảng cho embedding model**: đây là ứng dụng quan trọng nhất với phần còn lại của cuốn sách. Các model embedding hiện đại như BGE, E5 ([06-Embedding.md](../06-Embedding/06-Embedding.md)) đều **khởi tạo từ một encoder kiểu BERT** (đã pretrain bằng MLM), sau đó fine-tune tiếp bằng **contrastive learning** (kéo câu đồng nghĩa lại gần nhau, đẩy câu khác nghĩa ra xa trong không gian vector) để vector đầu ra phản ánh đúng độ tương đồng ngữ nghĩa. Nói cách khác: MLM dạy model "hiểu ngôn ngữ", còn contrastive learning dạy tiếp "biến sự hiểu đó thành một vector so sánh được" — hai bước huấn luyện nối tiếp trên cùng một kiến trúc encoder-only.

### Code ví dụ: điền từ vào chỗ trống với BERT

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

sentence = f"The capital of France is {tokenizer.mask_token}."
inputs = tokenizer(sentence, return_tensors="pt")
mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

with torch.no_grad():
    logits = model(**inputs).logits   # (batch, seq_len, vocab_size)

mask_logits = logits[0, mask_index, :]           # logits tại đúng vị trí [MASK]
top5 = torch.topk(mask_logits, k=5, dim=-1)

print(f"Câu: {sentence}")
for score, token_id in zip(top5.values[0], top5.indices[0]):
    word = tokenizer.decode([token_id])
    print(f"  {word:>12} — logit {score:.2f}")
# Kết quả mong đợi: "paris" đứng đầu danh sách top-5, vì model dùng
# ngữ cảnh CẢ CÂU (bidirectional) để suy luận, không chỉ phần trước [MASK]
```

## Chương 5.2 — Decoder-only: GPT family

### Nhắc lại nhanh

Kiến trúc decoder-only (self-attention với causal mask, không có cross-attention vì không có Encoder riêng) đã được học kỹ ở [04-Transformer.md §3 và §5](../04-Transformer/04-Transformer.md) — chương này **không lặp lại cơ chế**, mà tập trung vào **lịch sử phát triển** của dòng GPT, vì đây là dòng kiến trúc quan trọng nhất xuyên suốt cuốn sách (Qwen dùng ở hầu hết các chương sau).

### Bốn cột mốc

**GPT-1 (2018)** — *"Improving Language Understanding by Generative Pre-Training"*. Ý tưởng cốt lõi: pretrain một decoder-only Transformer bằng next-token prediction trên corpus văn bản không gán nhãn (BooksCorpus), sau đó **fine-tune** (cập nhật trọng số) riêng cho từng task cụ thể (classification, entailment...) bằng cách thêm một đầu ra nhỏ phía trên. Đây là lần đầu tiên "pretrain rồi fine-tune" chứng minh hiệu quả rõ rệt trên nhiều task NLP cùng lúc, nhưng vẫn cần dữ liệu gán nhãn + một vòng huấn luyện riêng cho mỗi task.

**GPT-2 (2019)** — *"Language Models are Unsupervised Multitask Learners"*. Train trên corpus lớn hơn nhiều (WebText) và model lớn hơn hẳn GPT-1. Phát hiện quan trọng nhất: khi scale đủ lớn, model bắt đầu làm được nhiều task (dịch, tóm tắt, hỏi-đáp) **mà không cần fine-tune riêng** — chỉ cần đưa đúng định dạng prompt, đây là hiện tượng **zero-shot task transfer** xuất hiện tự nhiên từ việc scale, chứ không phải do thiết kế đặc biệt.

**GPT-3 (2020)** — *"Language Models are Few-Shot Learners"*. Scale tiếp lên 175 tỷ tham số. Đóng góp mang tính bước ngoặt: **in-context learning / few-shot prompting** — chỉ cần đưa vài ví dụ mẫu (input → output) ngay trong prompt, model "học" cách làm task đó **ngay lập tức, không cần cập nhật bất kỳ trọng số nào**. Đây là lần đầu tiên ranh giới giữa "train" và "dùng" model bị xóa nhòa: toàn bộ việc thích nghi với task mới diễn ra chỉ bằng cách viết prompt khéo léo hơn.

**Thế hệ hiện đại (Llama, Qwen...)** — vẫn là decoder-only thuần túy về kiến trúc nền, nhưng cộng thêm: RoPE thay sinusoidal, GQA/MQA thay multi-head attention thường, SwiGLU thay ReLU (tất cả đã học ở [04-Transformer.md](../04-Transformer/04-Transformer.md)), cùng với instruction-tuning và RLHF để model theo sát chỉ dẫn/hội thoại tốt hơn. Đây chính là dòng model được dùng xuyên suốt phần còn lại của sách — [08-Fine-Tuning.md](../08-Fine-Tuning/08-Fine-Tuning.md), [09-LangChain.md](../09-LangChain/09-LangChain.md), [10-LangGraph.md](../10-LangGraph/10-LangGraph.md), [11-Agent.md](../11-Agent/11-Agent.md).

| Model | Năm | Số tham số | Điểm đột phá |
|---|---|---|---|
| GPT-1 | 2018 | ~117M | Pretrain (next-token) + fine-tune riêng từng task |
| GPT-2 | 2019 | ~1.5B | Zero-shot task transfer xuất hiện khi scale, không cần fine-tune |
| GPT-3 | 2020 | ~175B | Few-shot/in-context learning — thích nghi task mới chỉ bằng prompt, không cập nhật trọng số |
| Llama / Qwen (hiện đại) | 2023+ | Đa dạng (7B–70B+) | RoPE, GQA, SwiGLU, instruction-tuning/RLHF — dùng xuyên suốt sách |

### In-context learning là gì

Khác với fine-tuning (cập nhật trọng số qua backpropagation, cần dữ liệu + GPU train riêng), in-context learning tận dụng chính cơ chế attention: khi đưa vài cặp ví dụ mẫu vào đầu prompt, các token của ví dụ đó đóng vai trò ngữ cảnh mà self-attention của các token sinh sau "tham chiếu" tới — về bản chất, model **suy luận theo mẫu** ngay trong một lần forward pass, không có bước gradient nào diễn ra.

```
Prompt few-shot (in-context learning), không cần fine-tune:

  Dịch tiếng Anh sang tiếng Việt:
  cat -> mèo
  dog -> chó
  bird -> chim
  fish -> ???

Model dựa vào 3 ví dụ ngay trong prompt để "hiểu" định dạng task và
suy ra "fish -> cá" — toàn bộ diễn ra trong 1 lần forward, không train.
```

Kỹ thuật này chính là nền tảng của **Prompt Engineering** — chủ đề đã được giới thiệu ở [04-Transformer.md](../04-Transformer/04-Transformer.md): việc thiết kế prompt (system message, few-shot examples, chain-of-thought...) thực chất là đang khai thác khả năng in-context learning này của decoder-only LLM.

## Chương 5.3 — Encoder-Decoder: T5/BART

### Span corruption objective

T5 (Text-to-Text Transfer Transformer) giữ nguyên cả Encoder lẫn Decoder của Transformer gốc ([04-Transformer.md §3, §5](../04-Transformer/04-Transformer.md)), nhưng thay đổi objective pretrain so với dịch máy thuần túy: **span corruption** — che cả **một cụm token liên tiếp** (không phải từng token rời rạc như BERT) bằng 1 sentinel token duy nhất, rồi bắt Decoder sinh lại đúng cụm đã bị che:

```
Câu gốc: "The cat sat on the mat"

T5 (Span corruption):
  Input:  "The cat <X> the mat"        (cả cụm "sat on" bị che bởi 1 sentinel token <X>)
  Target: "<X> sat on"                 (model sinh lại đúng cụm đã bị che)
```

So với MLM của BERT (che từng token đơn lẻ, đoán độc lập), span corruption buộc Decoder phải **sinh một chuỗi liên tục có thứ tự** — luyện đúng kỹ năng cần cho các task sinh văn bản (dịch, tóm tắt) mà BERT (chỉ có Encoder) không làm được.

### "Text-to-text": mọi task đều là chuyển văn bản thành văn bản

Ý tưởng đặc trưng nhất của T5, cũng là nguồn gốc chữ T5 (Text-To-Text Transfer Transformer): **thay vì thiết kế kiến trúc/đầu ra riêng cho từng loại task** (như BERT cần thêm classifier head cho classification, thêm token-tagging head cho NER...), T5 đóng khung **mọi task** — dịch, tóm tắt, phân loại, trả lời câu hỏi — thành cùng một định dạng: input là văn bản kèm 1 **prefix** mô tả task, output cũng là văn bản thuần túy.

```
Task: Dịch máy
  Input:  "translate English to German: That is good."
  Output: "Das ist gut."

Task: Tóm tắt
  Input:  "summarize: <đoạn văn dài>"
  Output: "<bản tóm tắt ngắn>"

Task: Phân loại (ngay cả classification cũng thành sinh text!)
  Input:  "cola sentence: The course is jumping well."
  Output: "unacceptable"     (thay vì output 1 con số/logit như BERT)
```

Lợi ích: **một kiến trúc, một hàm loss (cross-entropy trên token sinh ra), một cách huấn luyện duy nhất** cho mọi task — không cần thiết kế lại đầu ra mỗi khi thêm task mới, chỉ cần đổi prefix và cách format input/output.

### BART — một nhánh khác của encoder-decoder

BART (Facebook, 2019) có kiến trúc gần như giống hệt T5 (encoder-decoder đầy đủ), nhưng objective pretrain là **denoising autoencoder** tổng quát hơn span corruption: kết hợp nhiều kiểu "làm nhiễu" input (xóa token, hoán đổi thứ tự câu, xoay vòng văn bản, điền khoảng trống độ dài thay đổi...) rồi bắt Decoder khôi phục lại văn bản gốc hoàn chỉnh. BART đặc biệt mạnh ở các task sinh văn bản tự nhiên như tóm tắt và sinh đối thoại, do objective huấn luyện gần với việc "khôi phục văn bản mạch lạc" hơn.

### Code ví dụ: tóm tắt văn bản với T5

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

article = (
    "summarize: The Transformer architecture, introduced in 2017, replaced "
    "recurrent neural networks in most natural language processing tasks. "
    "Instead of processing tokens sequentially, it uses a mechanism called "
    "self-attention, allowing every token to directly attend to every other "
    "token in a single step. This makes training highly parallelizable and "
    "improves the model's ability to capture long-range dependencies in text."
)

input_ids = tokenizer(article, return_tensors="pt", max_length=512, truncation=True).input_ids
summary_ids = model.generate(
    input_ids,
    max_length=40,
    num_beams=4,          # beam search: giữ 4 chuỗi ứng viên tốt nhất mỗi bước sinh
    early_stopping=True,
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
# Lưu ý: BẮT BUỘC có prefix "summarize: " -- nếu bỏ prefix, T5 không biết
# đang được yêu cầu làm task gì (đây chính là bản chất "text-to-text" ở trên)
```

## Chương 5.4 — So sánh & lựa chọn

| Tiêu chí | Encoder-only (BERT) | Decoder-only (GPT/Llama/Qwen) | Encoder-Decoder (T5/BART) |
|---|---|---|---|
| Attention pattern | Bidirectional (không mask) | Causal (chỉ nhìn quá khứ) | Encoder bidirectional + Decoder causal + Cross-attention |
| Training objective | Masked Language Modeling (che rời rạc 15% token) | Next-token prediction | Span corruption (BART: denoising autoencoder) |
| Ưu điểm | Hiểu ngữ cảnh 2 chiều tốt nhất; nhẹ, nhanh cho task phân loại | Kiến trúc đơn giản nhất (1 loại attention); scale tốt; vừa hiểu vừa sinh trong cùng 1 chuỗi | Tách biệt rõ input/output; mạnh cho task chuyển đổi văn bản có cấu trúc |
| Nhược điểm | Không sinh được văn bản tuần tự (không có decoder) | Chỉ nhìn được quá khứ — kém hơn bidirectional cho việc "hiểu" thuần túy | Kiến trúc phức tạp hơn (2 stack + cross-attention); khó scale bằng decoder-only |
| Đại diện | BERT, RoBERTa, ALBERT | GPT-1/2/3, Llama, Qwen | T5, BART |
| Dùng khi nào | Classification, NER, làm nền cho embedding model | Chat, sinh code, reasoning, agent — hầu hết ứng dụng LLM hiện đại | Dịch máy, tóm tắt — khi input/output là 2 "khối" văn bản tách biệt rõ ràng |

Trực quan hóa lại attention pattern của cả 3 dòng trên cùng một câu ví dụ (✓ = nhìn thấy):

```
BERT (bidirectional) -- token "sat" nhìn được CẢ 2 chiều:
   the cat [MASK] on the mat  ->  [MASK] nhìn thấy: the, cat, on, the, mat (toàn câu)

GPT (causal) -- token "sat" chỉ nhìn được quá khứ:
   the cat sat on the mat     ->  "sat" nhìn thấy: the, cat, sat (không thấy "on the mat")

T5 decoder (causal + cross) -- vừa nhìn quá khứ của chính nó, vừa nhìn toàn bộ output Encoder:
   decoder token "yêu" nhìn thấy: "Tôi" (quá khứ, causal) + toàn bộ "I love cats" đã mã hóa (cross-attention)
```

**Chốt lại — dòng nào dùng cho phần nào của sách:**

- **Encoder-only** → [06-Embedding.md](../06-Embedding/06-Embedding.md): mọi model embedding (BGE, E5) đều fine-tune tiếp từ một encoder kiểu BERT bằng contrastive learning, để biến văn bản thành vector so sánh được (nền tảng cho retrieval trong RAG).
- **Decoder-only** → gần như toàn bộ phần còn lại của sách: [04-Transformer.md](../04-Transformer/04-Transformer.md), [08-Fine-Tuning.md](../08-Fine-Tuning/08-Fine-Tuning.md), [09-LangChain.md](../09-LangChain/09-LangChain.md), [10-LangGraph.md](../10-LangGraph/10-LangGraph.md), [11-Agent.md](../11-Agent/11-Agent.md) đều xoay quanh Qwen — một model decoder-only. Đây là lý do dòng kiến trúc này đáng để hiểu sâu nhất trong ba dòng.
- **Encoder-Decoder** → ít xuất hiện trực tiếp trong phần thực hành của sách (các LLM hiện đại dùng decoder-only kiêm luôn vai trò dịch/tóm tắt bằng cách đưa "câu nguồn" vào prompt — như đã giải thích ở [04-Transformer.md §5](../04-Transformer/04-Transformer.md)), nhưng vẫn quan trọng để hiểu vì sao kiến trúc decoder-only "đơn giản hóa" được so với thiết kế gốc.

## Chương 5.5 — Bài tập

1. Thử lại code ở [Chương 5.1](#chương-51--encoder-only-bert) nhưng dùng model đa ngôn ngữ `bert-base-multilingual-cased` với một câu tiếng Việt, ví dụ `"Thủ đô của Việt Nam là [MASK]."`. So sánh chất lượng top-5 dự đoán với ví dụ tiếng Anh — model đa ngôn ngữ có đoán đúng "Hà Nội" không, và độ tự tin (logit) có thấp hơn so với câu tiếng Anh không?
2. Dùng code T5 ở [Chương 5.3](#chương-53--encoder-decoder-t5bart) để tóm tắt 3 đoạn văn tiếng Anh có độ dài và chủ đề khác nhau (kỹ thuật, tin tức, văn học). So sánh chất lượng bản tóm tắt của `t5-small` với `t5-base` — bản lớn hơn có giữ được nhiều thông tin quan trọng hơn không?
3. Tìm hiểu và tóm tắt (2-3 câu mỗi model) sự khác biệt giữa BERT và hai biến thể phổ biến của nó: **RoBERTa** (bỏ objective Next Sentence Prediction, dùng dynamic masking, train lâu hơn với nhiều dữ liệu hơn) và **ALBERT** (chia sẻ tham số giữa các layer để giảm kích thước model, factorize embedding matrix, thay NSP bằng Sentence Order Prediction).
4. Thử prefix `"translate English to German: "` và `"cola sentence: "` với `t5-small` (theo đúng ví dụ text-to-text ở [Chương 5.3](#chương-53--encoder-decoder-t5bart)), quan sát output — model có tự nhận diện đúng loại task chỉ nhờ đọc prefix không cần huấn luyện lại?
5. Viết một prompt few-shot (in-context learning, theo mẫu ở [Chương 5.2](#chương-52--decoder-only-gpt-family)) cho một model decoder-only local (ví dụ Qwen qua Ollama), thử với 2 và 5 ví dụ mẫu, so sánh độ chính xác kết quả — số lượng ví dụ trong prompt ảnh hưởng thế nào tới chất lượng suy luận?

## Chương 5.6 — Tài liệu tham khảo

> 📖 Xem chú giải chi tiết "vì sao mỗi paper quan trọng" ở [15-Papers.md](../15-Papers/15-Papers.md#chương-152--nền-tảng-transformer--llm).

- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018)
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1, 2018)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020)
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2019)
- Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension* (2019)
- [Hugging Face Transformers docs](https://huggingface.co/docs/transformers) · [The Illustrated BERT (Jay Alammar)](https://jalammar.github.io/illustrated-bert/)
