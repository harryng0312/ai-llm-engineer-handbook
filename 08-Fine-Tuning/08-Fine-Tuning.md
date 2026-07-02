# 08 — Fine-Tuning: LoRA và QLoRA

> Chương này trả lời câu hỏi: "Khi nào prompt engineering (đã học ở [04-Transformer.md](../04-Transformer/04-Transformer.md) và phần Prompt Engineering) không còn đủ, và làm sao 'nướng' kiến thức vào trọng số model mà không cần cả cụm GPU?" Fine-tune full-parameter một LLM 7B đòi hỏi hàng chục GB VRAM chỉ riêng cho gradient và optimizer state — vượt xa khả năng của một máy cá nhân hay một GPU Colab miễn phí. LoRA và QLoRA là hai kỹ thuật giúp fine-tune khả thi trên phần cứng khiêm tốn hơn nhiều, bằng cách train một lượng tham số nhỏ hơn gốc hàng trăm–hàng nghìn lần.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`08-Fine-Tuning.ipynb`](./08-Fine-Tuning.ipynb).

## Mục lục

1. [Chương 8.1 — Vì sao cần Fine-Tuning](#chương-81--vì-sao-cần-fine-tuning)
2. [Chương 8.2 — LoRA (Low-Rank Adaptation)](#chương-82--lora-low-rank-adaptation)
3. [Chương 8.3 — QLoRA](#chương-83--qlora)
4. [Chương 8.4 — Thực hành: chọn LoRA hay QLoRA](#chương-84--thực-hành-chọn-lora-hay-qlora)
5. [Chương 8.5 — Bài tập](#chương-85--bài-tập)
6. [Chương 8.6 — Tài liệu tham khảo](#chương-86--tài-liệu-tham-khảo)

---

## Chương 8.1 — Vì sao cần Fine-Tuning

Khi prompt engineering không đủ (cần model học một domain/style/task riêng), ta fine-tune.

Prompt engineering và fine-tuning giải quyết cùng một vấn đề — "làm cho model trả lời đúng ý mình muốn" — nhưng theo hai cách khác hẳn nhau: một bên chỉ thay đổi *đầu vào* mỗi lần gọi model, một bên thay đổi *chính trọng số* của model.

| | Prompt Engineering | Fine-Tuning |
|---|---|---|
| Cần train? | Không | Có |
| Tốc độ triển khai | Ngay lập tức, sửa prompt là xong | Cần chuẩn bị dữ liệu, chạy training, kiểm định |
| Chi phí | Thấp — chỉ tốn token mỗi lần gọi | Cao hơn — cần GPU, thời gian train, lưu trữ checkpoint |
| Tính linh hoạt | Rất cao, đổi hành vi tức thời bằng cách đổi prompt | Cứng hơn — muốn đổi hành vi phải train lại |
| Hiệu quả khi | Task đơn giản, ít ví dụ minh họa là đủ (zero/few-shot) | Task cần model "nhớ" một format/domain/style cố định, lặp lại ở quy mô lớn |
| Giới hạn | Prompt càng dài càng tốn KV cache/độ trễ, và có trần về việc "dạy" hành vi phức tạp chỉ bằng ví dụ trong ngữ cảnh | Cần dữ liệu gán nhãn, tốn tài nguyên tính toán, rủi ro overfit/quên kiến thức cũ (catastrophic forgetting) |
| Kiến thức lưu ở đâu | Trong prompt, phải gửi lại mỗi lần gọi | "Nướng" thẳng vào trọng số, không cần lặp lại trong prompt |

Nói ngắn gọn: prompt engineering thay đổi *những gì model nhìn thấy*, còn fine-tuning thay đổi *cách model phản ứng* với bất kỳ đầu vào nào, vĩnh viễn (cho tới khi fine-tune lại). Nguyên tắc thực dụng: luôn thử prompt engineering trước — rẻ, nhanh, dễ đảo ngược — chỉ chuyển sang fine-tuning khi đã xác nhận rõ prompt (kể cả few-shot, CoT) không đạt đủ chất lượng hoặc quá tốn kém khi lặp lại ở quy mô lớn.

### Vì sao không thể fine-tune full-parameter trên máy cá nhân

Fine-tune full-parameter một LLM 7B cần hàng chục GB VRAM cho gradient + optimizer state — không khả thi trên máy cá nhân.

Lý do cụ thể: với optimizer Adam (phổ biến nhất khi train LLM), mỗi tham số cần lưu không chỉ chính nó mà còn gradient và 2 "moment" nội bộ của Adam. Ước tính nhanh cho model 7B tỷ tham số, huấn luyện ở fp16/bf16:

```
Trọng số (bf16, 2 byte/tham số):        7B × 2 byte  = 14 GB
Gradient (bf16, 2 byte/tham số):        7B × 2 byte  = 14 GB
Optimizer state Adam (fp32, 2 giá trị 4 byte/tham số): 7B × 8 byte = 56 GB
                                                        ------------------
                                        Tổng tối thiểu ≈ 84 GB VRAM
```

Con số này còn chưa tính activation (bộ nhớ trung gian trong forward/backward pass) hay KV cache — 84GB đã vượt xa VRAM của hầu hết GPU tiêu dùng (thường 8–24GB). Đây chính là động lực ra đời của LoRA: nếu chỉ cần train một phần rất nhỏ tham số thay vì toàn bộ `W`, gradient và optimizer state cũng nhỏ theo tương ứng.

## Chương 8.2 — LoRA (Low-Rank Adaptation)

**LoRA (Low-Rank Adaptation)**: đóng băng toàn bộ trọng số gốc `W`, chỉ train thêm 2 ma trận nhỏ `A (d×r)` và `B (r×d)` với `r << d` (rank thấp, ví dụ r=8, 16), sao cho:

```
W' = W + (α/r) * B @ A
```

Vì `A`, `B` rất nhỏ so với `W`, số tham số cần train giảm hàng trăm–hàng nghìn lần, VRAM cho optimizer cũng giảm tương ứng.

### Vì sao rank thấp vẫn hiệu quả

Trực giác đứng sau LoRA dựa trên giả thuyết **"intrinsic rank"** (rank nội tại): khi fine-tune một model lớn cho một task cụ thể, sự **thay đổi cần thiết** của ma trận trọng số (`ΔW = W' - W`) thường không cần "trải" khắp toàn bộ không gian đầy đủ `d × d` chiều của `W` — nó nằm gọn trong một không gian con có rank thấp hơn nhiều so với kích thước đầy đủ.

Hình dung: nếu `W` là một ma trận `4096 × 4096` (~16.7 triệu tham số), việc "dạy thêm" model một task hẹp (ví dụ luôn trả lời theo một văn phong cố định) không đòi hỏi thay đổi model theo *mọi* hướng có thể trong không gian 4096 chiều đó — chỉ cần thay đổi theo một số ít hướng "quan trọng" là đủ để đạt hiệu quả gần tương đương fine-tune toàn bộ. `B @ A` (với `r` nhỏ, ví dụ 8) chính là một xấp xỉ rank-thấp của `ΔW`, buộc thay đổi trọng số chỉ được diễn ra trong không gian con `r` chiều đó:

```
ΔW (d×d, rank tối đa = d)          B @ A (d×d, nhưng rank tối đa = r << d)
┌─────────────────┐                ┌───┐   ┌─────────────────┐
│                 │                │   │   │                 │
│   d × d          │      ≈         │ B │ × │        A         │
│  (đầy đủ)        │                │d×r│   │      (r×d)       │
│                 │                │   │   │                 │
└─────────────────┘                └───┘   └─────────────────┘
 16.7M tham số (d=4096)             chỉ 2×d×r tham số (r=8 -> ~65K, giảm ~250 lần)
```

Vì `A`, `B` có tổng cộng `2 × d × r` tham số thay vì `d × d`, khi `r` càng nhỏ so với `d`, số tham số cần train càng ít — nhưng nếu chọn `r` quá nhỏ, model có thể không đủ "chỗ" để biểu diễn hết thay đổi cần thiết cho task phức tạp, nên `r` là một siêu tham số cần thử nghiệm (thường 8–64 là đủ cho phần lớn task fine-tune domain/style).

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

## Chương 8.3 — QLoRA

**QLoRA** = LoRA + lượng tử hóa (quantization) trọng số gốc `W` xuống 4-bit (NF4) trước khi train. `W` 4-bit chỉ dùng để forward/backward (dequantize tạm thời khi tính toán), còn `A`, `B` vẫn train ở độ chính xác cao (bf16/fp32). Nhờ vậy, một model 7B có thể fine-tune trên GPU 8–12GB VRAM.

### Luồng dữ liệu QLoRA

Điểm dễ nhầm lẫn: `W` không được "giữ nguyên ở 4-bit" trong suốt phép tính — nó chỉ được *lưu trữ* ở 4-bit để tiết kiệm VRAM, còn khi thực sự cần nhân ma trận, nó được giải nén (dequantize) tạm thời về bf16, tính xong rồi bỏ kết quả trung gian đó đi (không lưu lại bản bf16). Sơ đồ đơn giản hóa luồng forward/backward:

```
FORWARD PASS
────────────
  W (lưu trữ, NF4 4-bit, ĐÓNG BĂNG)
        │
        ▼ dequantize tạm thời (chỉ tồn tại trong lúc tính toán)
  W_bf16 (tạm thời)  ──┐
                       │
  A, B (bf16/fp32,     ├──▶  y = x @ W_bf16 + x @ (α/r · B @ A)
        TRAIN ĐƯỢC) ───┘            (cộng 2 nhánh: nhánh gốc đóng băng
                                      + nhánh LoRA train được)

BACKWARD PASS
─────────────
  Gradient chỉ được tính và lưu cho A, B (số tham số nhỏ)
  W 4-bit KHÔNG có gradient, KHÔNG có optimizer state -> tiết kiệm phần lớn VRAM
        │
        ▼
  Optimizer (Adam...) chỉ cập nhật A, B ở độ chính xác cao
  W 4-bit giữ nguyên, đóng băng suốt quá trình train
```

Nhờ cơ chế này, chi phí VRAM chủ đạo của QLoRA chỉ còn: (1) `W` nén 4-bit (nhỏ hơn ~4 lần so với bf16), (2) gradient + optimizer state chỉ cho `A`, `B` (rất nhỏ nhờ LoRA), (3) activation tạm thời trong lúc tính toán — không còn khoản chi phí khổng lồ cho gradient/optimizer của toàn bộ `W` như fine-tune full-parameter.

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

## Chương 8.4 — Thực hành: chọn LoRA hay QLoRA

Khi nào chọn LoRA/QLoRA thay vì prompt engineering: khi cần model **nhớ một format/domain cố định** (ví dụ luôn trả lời theo văn phong công ty, luôn output đúng schema nội bộ), hoặc khi ngữ cảnh cần thiết quá dài để nhét vào mỗi prompt (tốn phí, tốn KV cache) — lúc đó "nướng" kiến thức vào trọng số qua LoRA hiệu quả hơn.

Giữa LoRA và QLoRA, lựa chọn phụ thuộc chủ yếu vào VRAM sẵn có:

| Tiêu chí | LoRA | QLoRA |
|---|---|---|
| VRAM cần thiết (model 7B) | ~14–18GB (model gốc bf16 + A,B + optimizer nhỏ) | ~6–10GB (model gốc nén 4-bit + A,B + optimizer nhỏ) |
| Tốc độ train | Nhanh hơn (không tốn chi phí dequantize liên tục) | Chậm hơn một chút (mỗi lần forward/backward phải dequantize `W` tạm thời) |
| Độ chính xác đánh đổi | Gần như không mất mát so với full fine-tune | Mất mát rất nhỏ do lượng tử hóa `W`, nhưng NF4 + double quant được thiết kế để giảm thiểu sai số này |
| Khi nào chọn | Đã có đủ VRAM để load model gốc ở bf16 (ví dụ GPU 24GB+ cho model 7B) | VRAM hạn chế (GPU 8–12GB cho model 7B, ví dụ Colab free-tier T4/consumer GPU) |

Nguyên tắc thực dụng: nếu không chắc, bắt đầu với QLoRA — nó gần như luôn chạy được trên phần cứng khiêm tốn hơn, và chênh lệch chất lượng so với LoRA thuần thường không đáng kể cho các task fine-tune domain/style thông thường. Chỉ cần chuyển sang LoRA (không lượng tử hóa) khi đã dư VRAM và muốn tối ưu thêm tốc độ train.

## Chương 8.5 — Bài tập

1. Đọc code `LoraConfig` của `peft`, thử đổi `target_modules` sang layer khác của `distilgpt2` (in `model` ra để xem tên các layer), quan sát số lượng trainable params thay đổi thế nào.
2. Thử đổi rank `r` = 4, 8, 16, 32 trên cùng một model (`distilgpt2` hoặc tương đương), so sánh số trainable params in ra từ `print_trainable_parameters()`, và nếu có GPU, so sánh luôn thời gian train trên cùng một tập dữ liệu nhỏ.
3. Cấu hình QLoRA đầy đủ cho một model 7B thật (ví dụ `Qwen/Qwen2.5-7B-Instruct`) trên Colab free-tier GPU (T4, 16GB): load model bằng `BitsAndBytesConfig` như ví dụ ở [Chương 8.3](#chương-83--qlora), áp `LoraConfig`, chạy vài bước train thử, ghi lại VRAM thực tế dùng (`nvidia-smi` hoặc `torch.cuda.memory_allocated()`).
4. So sánh chất lượng output giữa model gốc chưa fine-tune và model đã LoRA fine-tune trên một task nhỏ tự chọn (ví dụ: luôn trả lời theo một văn phong cố định, hoặc luôn output đúng một schema JSON) — chấm điểm định tính trên cùng 10 câu hỏi.

## Chương 8.6 — Tài liệu tham khảo

> 📖 Xem chú giải chi tiết ở [15-Papers.md](../15-Papers/15-Papers.md#chương-152--nền-tảng-transformer--llm).

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
