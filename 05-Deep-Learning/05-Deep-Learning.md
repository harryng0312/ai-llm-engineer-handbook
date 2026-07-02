# 05 — Deep Learning

> Chương này dựng các khối nền tảng của mạng neural (perceptron, backpropagation, CNN, RNN/LSTM) dựa trên toán đã học ở [02-Math-for-LLM.md](../02-Math-for-LLM/02-Math-for-LLM.md), rồi đi sâu hơn vào nhánh **NLP** (xử lý ngôn ngữ tự nhiên): cách văn bản được biểu diễn thành vector qua các thời kỳ, và cách RNN/LSTM cùng cơ chế Attention từng giải quyết bài toán dịch máy trước khi Transformer ra đời. Phần cuối này ([Chương 5.5](#chương-55--word-embeddings), [5.6](#chương-56--seq2seq-và-attention)) chính là bối cảnh lịch sử trực tiếp cho [06-Transformer.md](../06-Transformer/06-Transformer.md) — nên đọc kỹ trước khi sang chương đó.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`05-Deep-Learning.ipynb`](./05-Deep-Learning.ipynb).

## Mục lục

1. [Mạng neural cơ bản](#chương-51--mạng-neural-cơ-bản)
2. [Backpropagation và Gradient Descent](#chương-52--backpropagation-và-gradient-descent)
3. [CNN](#chương-53--cnn)
4. [RNN/LSTM](#chương-54--rnnlstm)
5. [Word Embeddings: từ Bag-of-Words đến word2vec](#chương-55--word-embeddings)
6. [Seq2Seq và Attention: tiền thân của Transformer](#chương-56--seq2seq-và-attention)
7. [Regularization và Normalization](#chương-57--regularization-và-normalization)
8. [Bài tập](#chương-58--bài-tập)
9. [Tài liệu tham khảo](#chương-59--tài-liệu-tham-khảo)

---

## Chương 5.1 — Mạng neural cơ bản

Đơn vị nhỏ nhất của mạng neural là **perceptron** (hay "neuron"): nhận nhiều input số thực, nhân mỗi input với 1 trọng số (weight) học được, cộng lại cùng 1 bias, rồi đưa qua một **hàm kích hoạt** (activation function) phi tuyến:

```
x1 ──w1──┐
x2 ──w2──┼──▶ Σ(xi·wi) + b ──▶ activation ──▶ output
x3 ──w3──┘
```

Nếu không có activation phi tuyến, xếp chồng bao nhiêu lớp neuron cũng chỉ tương đương **1 phép biến đổi tuyến tính duy nhất** (tổng hợp của nhiều phép nhân ma trận tuyến tính vẫn là tuyến tính) — mạng sẽ không học được các quan hệ phức tạp (ví dụ XOR, xem [Chương 5.2](#chương-52--backpropagation-và-gradient-descent)). Ba activation phổ biến:

| Activation | Công thức | Đặc điểm |
|---|---|---|
| Sigmoid | $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ | Output trong (0,1), dễ bị "bão hòa" (gradient ≈ 0) khi $\|x\|$ lớn |
| Tanh | $\tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$ | Output trong (-1,1), cùng vấn đề bão hòa như sigmoid nhưng đối xứng quanh 0 |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ | Không bão hòa với x > 0, tính rẻ, là lựa chọn mặc định cho mạng sâu hiện đại |

Ghép nhiều neuron thành 1 lớp (layer), ghép nhiều lớp thành **Multi-Layer Perceptron (MLP)** — mỗi lớp nhận output của lớp trước làm input:

```
Input (3)        Hidden (4)        Output (1)
  x1 ──┐        ┌──● ──┐
       ├───────▶│  ●   ├───────▶  ●  ──▶ y
  x2 ──┤        │  ●   │
       ├───────▶│  ●   ├
  x3 ──┘        └──────┘
            mỗi mũi tên = 1 trọng số học được (ma trận W)
```

Cài đặt thủ công bằng NumPy để thấy rõ từng phép tính:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mlp_forward(x, W1, b1, W2, b2):
    h = sigmoid(x @ W1 + b1)   # lớp ẩn (hidden layer)
    y = sigmoid(h @ W2 + b2)   # lớp output
    return h, y

rng = np.random.default_rng(0)
input_dim, hidden_dim, output_dim = 3, 4, 1
W1 = rng.normal(size=(input_dim, hidden_dim))
b1 = np.zeros(hidden_dim)
W2 = rng.normal(size=(hidden_dim, output_dim))
b2 = np.zeros(output_dim)

x = np.array([1.0, 0.5, -1.0])
h, y = mlp_forward(x, W1, b1, W2, b2)
print("hidden:", h.round(3))
print("output:", y.round(3))
```

Trong thực tế không ai viết tay `W1, b1, W2, b2` như trên — PyTorch (`torch.nn`) tự quản lý toàn bộ tham số:

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        return y

model = MLP(input_dim=3, hidden_dim=4, output_dim=1)
x = torch.tensor([1.0, 0.5, -1.0])
print(model(x))   # nn.Linear tự khởi tạo và lưu trữ W, b bên trong -- không cần khai tay như bản NumPy
```

## Chương 5.2 — Backpropagation và Gradient Descent

Mạng neural "học" bằng cách chỉnh dần các trọng số để giảm một **hàm mất mát** (loss function) đo độ sai lệch giữa output dự đoán và giá trị đúng. Hai bước lặp lại liên tục:

```
1. FORWARD PASS:   x -> [Linear W1,b1] -> h -> sigmoid -> a -> [Linear W2,b2] -> y -> Loss(y, target)

2. BACKWARD PASS:  dL/dy  -->  dL/dW2, dL/db2  -->  dL/da  -->  dL/dh  -->  dL/dW1, dL/db1
                    (đạo hàm lan truyền NGƯỢC từ loss về từng tham số, dùng chain rule)

3. CẬP NHẬT:       W := W - learning_rate × dL/dW     (đi ngược hướng gradient -- gradient descent)
```

**Backpropagation** chỉ là ứng dụng lặp lại của **chain rule** (đã học ở [02-Math-for-LLM.md §2.2](../02-Math-for-LLM/02-Math-for-LLM.md)): đạo hàm của loss theo 1 trọng số ở lớp đầu bằng tích các đạo hàm cục bộ của mọi lớp nó đi qua. Không cần tự viết công thức này — `autograd` của PyTorch tự xây "đồ thị tính toán" lúc forward và tự tính ngược lúc gọi `.backward()`:

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
w = torch.tensor([0.5, -0.5], requires_grad=True)
y = (x * w).sum()
loss = (y - 1.0) ** 2

loss.backward()
print("dL/dx:", x.grad)   # autograd tự tính đạo hàm ngược qua toàn bộ phép toán, không cần viết tay chain rule
print("dL/dw:", w.grad)
```

Ví dụ đầy đủ 1 vòng lặp huấn luyện: bài toán XOR — không thể phân tách bằng 1 đường thẳng, buộc phải có ít nhất 1 hidden layer (minh chứng cho lý do cần activation phi tuyến ở [Chương 5.1](#chương-51--mạng-neural-cơ-bản)):

```python
import torch
import torch.nn as nn

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])   # XOR

model = nn.Sequential(
    nn.Linear(2, 8), nn.Tanh(),
    nn.Linear(8, 1), nn.Sigmoid(),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCELoss()

for epoch in range(500):
    optimizer.zero_grad()          # xóa gradient tích lũy từ vòng lặp trước
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()                # backpropagation: tính gradient cho MỌI tham số chỉ bằng 1 lệnh
    optimizer.step()               # gradient descent: cập nhật tham số theo hướng ngược gradient
    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")

print(model(X).round())   # sau khi train xong, khớp đúng bảng XOR: [0, 1, 1, 0]
```

Các biến thể của gradient descent, đánh đổi giữa tốc độ hội tụ và độ ổn định:

| Thuật toán | Cập nhật dựa trên | Đặc điểm |
|---|---|---|
| (Batch) Gradient Descent | Gradient trên **toàn bộ** dữ liệu train | Ổn định nhưng chậm, không khả thi với dataset lớn |
| SGD (Stochastic GD) | Gradient trên **1 sample** mỗi bước | Nhanh, nhiễu (noisy) nhưng nhiễu đôi khi giúp thoát local minimum |
| Mini-batch GD | Gradient trên **1 batch nhỏ** (vd 32-256 sample) | Cân bằng — lựa chọn mặc định trong thực tế |
| Momentum | Mini-batch + "đà" tích lũy từ các bước trước | Giảm dao động, hội tụ nhanh hơn ở vùng dốc thoải |
| Adam | Momentum + learning rate riêng cho từng tham số | Mặc định phổ biến nhất khi train deep learning/LLM hiện nay |

## Chương 5.3 — CNN

**Convolutional Neural Network (CNN)** dùng một "cửa sổ trượt" (kernel/filter) nhỏ quét qua toàn bộ input, học phát hiện các pattern cục bộ (cạnh, góc, texture với ảnh) mà không cần 1 trọng số riêng cho từng vị trí như MLP — trọng số của kernel được **dùng chung** (weight sharing) ở mọi vị trí nó quét qua:

```
Input (5×5)                 Kernel (3×3) trượt qua           Feature map (3×3)
■ ■ ■ ■ ■                    từng vùng 3×3 của input,
■ ■ ■ ■ ■                    mỗi vị trí tính 1 tích chập
■ ■ ■ ■ ■        ──────▶     (nhân từng phần tử rồi cộng)     ■ ■ ■
■ ■ ■ ■ ■                    ra 1 giá trị duy nhất             ■ ■ ■
■ ■ ■ ■ ■                                                       ■ ■ ■
```

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
x = torch.randn(1, 1, 5, 5)     # (batch, channels, height, width)
out = conv(x)
print(out.shape)   # torch.Size([1, 4, 3, 3]) -- 4 kernel khác nhau, mỗi kernel học phát hiện 1 pattern riêng
```

Sau lớp conv thường có **pooling** (max/average) để giảm kích thước feature map, giữ lại thông tin nổi bật nhất và giúp mô hình ít nhạy cảm hơn với dịch chuyển nhỏ của pattern trong ảnh.

CNN nổi tiếng nhất với ảnh, nhưng trước khi RNN/Transformer thống trị NLP, `Conv1d` cũng từng được dùng cho văn bản (TextCNN, Kim 2014): coi mỗi từ trong câu như 1 "pixel" theo 1 chiều duy nhất, kernel quét qua **cửa sổ vài từ liền kề** để bắt các cụm từ (n-gram) quan trọng:

```python
# Conv1d cho văn bản: in_channels = embedding_dim, độ dài chuỗi đóng vai trò "chiều rộng ảnh"
conv1d = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3)
seq = torch.randn(1, 8, 10)      # (batch, embedding_dim, seq_len) -- 10 từ, mỗi từ vector 8 chiều
out = conv1d(seq)
print(out.shape)   # torch.Size([1, 4, 8]) -- 4 kernel, mỗi kernel "quét" qua cửa sổ 3 từ liền kề
```

Hạn chế của CNN với văn bản: kernel chỉ nhìn được trong 1 cửa sổ cố định (ví dụ 3 từ) — muốn bắt quan hệ giữa 2 từ cách xa nhau phải xếp chồng rất nhiều lớp conv. Đây là một trong các lý do RNN (xử lý được chuỗi độ dài tùy ý) rồi sau đó self-attention (nhìn được toàn bộ chuỗi trong 1 bước, xem [06-Transformer.md Chương 6.1](../06-Transformer/06-Transformer.md#chương-61--vì-sao-cần-transformer)) trở thành lựa chọn chính cho NLP thay vì CNN.

## Chương 5.4 — RNN/LSTM

**Recurrent Neural Network (RNN)** xử lý dữ liệu tuần tự (văn bản, chuỗi thời gian) bằng cách duy trì 1 **hidden state** cập nhật qua từng bước thời gian — hidden state mới phụ thuộc cả input hiện tại lẫn hidden state trước đó:

```
       x1        x2        x3        x4
       │         │         │         │
       ▼         ▼         ▼         ▼
h0 -> [RNN] -> [RNN] -> [RNN] -> [RNN] -> h4
       │         │         │         │
       ▼         ▼         ▼         ▼
       y1        y2        y3        y4
```

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b)$$

```python
import numpy as np

def rnn_step(x_t, h_prev, W_xh, W_hh, b):
    return np.tanh(x_t @ W_xh + h_prev @ W_hh + b)

rng = np.random.default_rng(0)
input_dim, hidden_dim = 4, 6
W_xh = rng.normal(size=(input_dim, hidden_dim)) * 0.1
W_hh = rng.normal(size=(hidden_dim, hidden_dim)) * 0.1
b = np.zeros(hidden_dim)

h = np.zeros(hidden_dim)                      # hidden state khởi tạo = 0
sequence = rng.normal(size=(5, input_dim))    # chuỗi 5 "token", mỗi token vector 4 chiều
for t, x_t in enumerate(sequence):
    h = rnn_step(x_t, h, W_xh, W_hh, b)
    print(f"bước {t}: hidden state = {h.round(3)}")
```

**Vấn đề vanishing/exploding gradient**: vì `h_t` phụ thuộc đệ quy vào `h_(t-1)`, khi backprop qua một chuỗi dài, gradient phải nhân liên tiếp qua rất nhiều đạo hàm của `tanh` (giá trị trong (0,1)) — nhân nhiều số nhỏ liên tiếp khiến gradient **tiến về 0** (token xa không còn ảnh hưởng được tới việc học), hoặc trong một số trường hợp khác gradient có thể **bùng nổ**. Đây chính là hạn chế "thông tin xa bị pha loãng" mà [06-Transformer.md Chương 6.1](../06-Transformer/06-Transformer.md#chương-61--vì-sao-cần-transformer) nhắc tới khi giải thích lý do cần self-attention.

**LSTM (Long Short-Term Memory)** giảm nhẹ vấn đề này bằng cách tách riêng 1 "cell state" `c_t` — kênh thông tin dài hạn được cập nhật chủ yếu bằng **cộng** (không phải nhân liên tục qua tanh như RNN), điều khiển bởi 3 cổng (gate) học được:

```
                    ┌─────────────────────────────────────┐
  c_(t-1) ──────────┤  ×  forget gate f_t  (quên gì từ cell cũ)
                    │  │                                   │
                    │  +  ← input gate i_t × candidate g_t (thêm gì mới vào cell)
                    │  │                                   │
                    └──┼───────────────────────────────────┘
                       ▼
                     c_t   (cell state -- kênh "băng chuyền" dài hạn, ít bị nhân dồn qua tanh)
                       │
                     × output gate o_t
                       ▼
                     h_t   (hidden state -- xuất ra ngoài, dùng cho bước tiếp theo và output)
```

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=4, hidden_size=6, batch_first=True)
lstm = nn.LSTM(input_size=4, hidden_size=6, batch_first=True)

x = torch.randn(1, 5, 4)   # (batch, seq_len, input_dim)
rnn_out, h_n = rnn(x)
lstm_out, (h_n_lstm, c_n) = lstm(x)   # LSTM có thêm cell state c_n riêng, tách khỏi hidden state

print(rnn_out.shape, lstm_out.shape)   # cùng shape (1, 5, 6) -- khác nhau ở CƠ CHẾ nội bộ, không phải shape output
```

LSTM (và biến thể gọn hơn GRU) là kiến trúc chủ đạo cho NLP trong suốt 2014-2017, cho tới khi Transformer thay thế hoàn toàn nhờ giải quyết triệt để cả vanishing gradient (đường đi thông tin O(1) thay vì O(n)) lẫn khả năng song song hóa (LSTM vẫn phải tính tuần tự `h_t` từ `h_(t-1)`, không song song hóa được theo chiều thời gian) — xem lại bảng so sánh đầy đủ ở [06-Transformer.md Chương 6.1](../06-Transformer/06-Transformer.md#chương-61--vì-sao-cần-transformer).

## Chương 5.5 — Word Embeddings

Trước khi đưa văn bản vào mạng neural, cần biến từ thành vector số. Cách tiếp cận đơn giản nhất — **one-hot encoding**: mỗi từ là 1 vector dài bằng kích thước vocab, toàn số 0 trừ đúng 1 vị trí bằng 1:

```
vocab = ["mèo", "chó", "vua", "nữ_hoàng"]

"mèo"     -> [1, 0, 0, 0]
"vua"     -> [0, 0, 1, 0]
```

Hai vấn đề: (1) vector cực **thưa** (sparse) và dài bằng cả vocab (hàng chục nghìn chiều); (2) mọi cặp từ đều có khoảng cách bằng nhau — `"vua"` và `"nữ_hoàng"` cũng "xa" nhau như `"vua"` và `"mèo"`, dù rõ ràng 2 từ đầu liên quan ngữ nghĩa hơn nhiều. **Bag-of-Words (BoW)** và **TF-IDF** biểu diễn cả 1 câu/văn bản bằng cách đếm tần suất từ, cải thiện được bài toán so khớp văn bản nhưng vẫn không giải quyết được vấn đề (2) — chúng vẫn hoàn toàn dựa trên đếm từ trùng khớp, không hiểu ngữ nghĩa:

```python
import numpy as np

docs = [
    "con mèo ngồi trên ghế",
    "con chó ngồi trên thảm",
    "thị trường chứng khoán tăng điểm",
]
vocab = sorted(set(w for doc in docs for w in doc.split()))

def bow_vector(doc, vocab):
    counts = {w: 0 for w in vocab}
    for w in doc.split():
        counts[w] += 1
    return np.array([counts[w] for w in vocab])

def tf_idf(docs, vocab):
    """TF-IDF thủ công: TF = tần suất trong 1 doc, IDF = log(tổng số doc / số doc chứa từ đó)."""
    bow = np.array([bow_vector(d, vocab) for d in docs])
    tf = bow / bow.sum(axis=1, keepdims=True)
    df = (bow > 0).sum(axis=0)                       # số document chứa mỗi từ
    idf = np.log(len(docs) / df)
    return tf * idf

vectors = tf_idf(docs, vocab)
print(vectors.round(2))
# từ xuất hiện ở NHIỀU document (ít đặc trưng) bị giảm trọng số; từ HIẾM (đặc trưng riêng cho 1 doc) được tăng trọng số
```

**word2vec** (Mikolov et al., 2013) giải quyết đúng vấn đề (2) bằng **giả thuyết phân bố** (distributional hypothesis): *"từ xuất hiện trong ngữ cảnh giống nhau thường có nghĩa giống nhau"*. Kiến trúc **Skip-gram**: dùng từ trung tâm để dự đoán các từ xung quanh nó — trong quá trình học để dự đoán tốt, mạng buộc phải "nén" thông tin ngữ nghĩa vào 1 vector dày đặc (dense) cho mỗi từ:

```
Câu: "vua cai trị vương quốc", cửa sổ = 1, từ trung tâm = "cai_trị"

  từ trung tâm "cai_trị"  ──▶  [Embedding]  ──▶  dự đoán từ ngữ cảnh: "vua" và "vương_quốc"
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

corpus = ("vua là đàn_ông quyền_lực nữ_hoàng là phụ_nữ quyền_lực "
          "vua cai_trị vương_quốc nữ_hoàng cai_trị vương_quốc").split()
vocab = sorted(set(corpus))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

def make_skipgram_pairs(tokens, window=2):
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                pairs.append((word2idx[center], word2idx[tokens[j]]))
    return pairs

pairs = make_skipgram_pairs(corpus)
centers = torch.tensor([p[0] for p in pairs])
contexts = torch.tensor([p[1] for p in pairs])

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)    # vector khi từ là "trung tâm"
        self.out_embed = nn.Embedding(vocab_size, embed_dim)   # vector khi từ là "ngữ cảnh"

    def forward(self, center_ids):
        return self.in_embed(center_ids) @ self.out_embed.weight.T   # (batch, vocab_size)

model = SkipGram(vocab_size=len(vocab), embed_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    logits = model(centers)
    loss = F.cross_entropy(logits, contexts)   # dự đoán từ ngữ cảnh từ từ trung tâm -- ý tưởng cốt lõi của skip-gram
    loss.backward()
    optimizer.step()

def most_similar(word, k=3):
    vec = model.in_embed.weight[word2idx[word]]
    sims = F.cosine_similarity(vec.unsqueeze(0), model.in_embed.weight)
    top = sims.argsort(descending=True)[1:k + 1]   # bỏ chính nó (luôn giống 100%)
    return [(idx2word[i.item()], round(sims[i].item(), 3)) for i in top]

print(most_similar("vua"))   # kỳ vọng "nữ_hoàng" nằm trong top -- 2 từ xuất hiện ở ngữ cảnh tương tự nhau
```

Với vocab lớn thực tế, tính softmax trên **toàn bộ** vocab ở mỗi bước (như `cross_entropy` phía trên) rất tốn kém — word2vec bản gốc dùng **negative sampling**: chỉ so sánh với 1 vài từ ngẫu nhiên không liên quan thay vì toàn bộ vocab. **GloVe** (Pennington et al., 2014) đi theo hướng khác — phân rã trực tiếp ma trận đồng xuất hiện (co-occurrence) toàn corpus bằng đại số tuyến tính — nhưng cho ra loại vector cùng bản chất: **1 vector tĩnh duy nhất cho mỗi từ**.

Hạn chế cốt lõi của embedding tĩnh (word2vec/GloVe): mỗi từ chỉ có **đúng 1 vector**, bất kể ngữ cảnh. Từ "**bank**" trong "river bank" (bờ sông) và "bank account" (ngân hàng) buộc phải dùng chung 1 vector, dù nghĩa hoàn toàn khác nhau. Đây chính là động lực cho **embedding theo ngữ cảnh** (contextual embedding) — vector của một từ được tính lại tùy theo câu nó xuất hiện, dựa trên hidden state của RNN/LSTM ([Chương 5.4](#chương-54--rnnlstm)) hoặc sau này là self-attention của BERT — xem [07-BERT-T5-GPT.md](../07-BERT-T5-GPT/07-BERT-T5-GPT.md). Các model embedding hiện đại dùng cho RAG (BGE/E5) ở [08-Embedding.md Chương 8.1](../08-Embedding/08-Embedding.md#chương-81--embedding-là-gì) đi xa hơn nữa: sinh vector cho **cả câu/đoạn văn**, học bằng contrastive learning có giám sát thay vì học không giám sát từ đồng xuất hiện như word2vec.

## Chương 5.6 — Seq2Seq và Attention

RNN/LSTM ([Chương 5.4](#chương-54--rnnlstm)) có thể dùng làm **Language Model**: dự đoán từ tiếp theo dựa trên hidden state tổng hợp từ mọi từ đã đọc trước đó — đúng bài toán **next-token prediction** mà GPT/LLM hiện đại vẫn đang giải, chỉ khác ở kiến trúc bên trong (RNN tuần tự thay vì Transformer song song):

```
"tôi"  -> [RNN h1] -> dự đoán từ tiếp theo: "thích"
"thích"-> [RNN h2] -> dự đoán từ tiếp theo: "học"
"học"  -> [RNN h3] -> dự đoán từ tiếp theo: "AI"
```

### Seq2Seq: encoder-decoder với context vector cố định

Cho bài toán input/output là 2 chuỗi khác độ dài (dịch máy, tóm tắt), Sutskever et al. (2014) đề xuất **Seq2Seq**: một RNN "Encoder" đọc hết câu nguồn, nén toàn bộ thông tin vào **hidden state cuối cùng** — gọi là context vector — rồi một RNN "Decoder" khác dùng đúng 1 vector đó làm điểm khởi đầu để sinh dần câu đích:

```
ENCODER (đọc câu nguồn)                        DECODER (sinh câu đích, TỪNG TỪ MỘT)

"I"    -> [RNN] -> h1
"love" -> [RNN] -> h2
"cats" -> [RNN] -> h3 ──── context vector ────▶ h3 -> [RNN] -> "Tôi"
                    (TOÀN BỘ câu nguồn bị              -> [RNN] -> "yêu"
                     nén vào 1 vector CỐ ĐỊNH)          -> [RNN] -> "mèo"
```

**Bottleneck**: dù câu nguồn dài 5 từ hay 50 từ, toàn bộ thông tin đều phải lọt qua đúng 1 vector có kích thước cố định — thực nghiệm cho thấy chất lượng dịch giảm rõ rệt khi câu nguồn dài, vì hidden state cuối "quên" dần các từ ở đầu câu (chính vấn đề vanishing gradient/thông tin bị pha loãng đã nêu ở [Chương 5.4](#chương-54--rnnlstm)).

### Attention: bỏ bottleneck, nhìn lại toàn bộ Encoder

Bahdanau et al. (2015) giải quyết bottleneck bằng ý tưởng đơn giản nhưng mang tính bước ngoặt: thay vì decoder chỉ nhận **1 context vector cố định** từ hidden state cuối, ở **mỗi bước decode**, decoder tính 1 context vector **riêng** bằng cách nhìn lại **toàn bộ** hidden state của Encoder, có trọng số:

```
Decoder ở bước t cần sinh từ tiếp theo, nhìn lại TOÀN BỘ hidden state Encoder:

  h1   h2   h3   h4   h5     (mọi hidden state của Encoder, KHÔNG chỉ h cuối như Seq2Seq gốc)
   │    │    │    │    │
   ▼    ▼    ▼    ▼    ▼
 score(h_i, trạng_thái_decoder)   -- "độ liên quan" giữa từng vị trí nguồn và decoder hiện tại
   │    │    │    │    │
   ▼    ▼    ▼    ▼    ▼
      softmax  ──▶   α1   α2   α3   α4   α5     (trọng số attention, tổng = 1)
   │    │    │    │    │
   └────┴────┴────┴────┘
             ▼
        context_t
```

$$\text{context}_t = \sum_i \alpha_i \cdot h_i$$

(context vector **riêng cho mỗi bước decode**, không cố định như Seq2Seq gốc)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        return self.rnn(self.embed(x))   # (toàn bộ hidden state, hidden state cuối)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        """decoder_state: (batch, hidden_dim) -- trạng thái decoder hiện tại
        encoder_outputs: (batch, src_len, hidden_dim) -- toàn bộ h1..hn của Encoder
        """
        src_len = encoder_outputs.shape[1]
        decoder_state = decoder_state.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W(torch.cat([decoder_state, encoder_outputs], dim=-1)))
        scores = self.v(energy).squeeze(-1)                # (batch, src_len) -- điểm liên quan
        weights = F.softmax(scores, dim=-1)                 # α trong sơ đồ ở trên
        context = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)   # context vector riêng cho bước này
        return context, weights

encoder = Encoder(vocab_size=20, embed_dim=8, hidden_dim=16)
attention = BahdanauAttention(hidden_dim=16)

src = torch.randint(0, 20, (1, 5))            # câu nguồn giả lập, 5 token
encoder_outputs, h_n = encoder(src)           # encoder_outputs: (1, 5, 16) -- LƯU LẠI hidden state MỌI vị trí
decoder_state = h_n.squeeze(0)                # (1, 16) -- trạng thái decoder ở 1 bước decode

context, weights = attention(decoder_state, encoder_outputs)
print("attention weights:", weights.detach().round(decimals=3))
print("context vector shape:", context.shape)   # khác nhau ở MỖI bước decode, không cố định như Seq2Seq gốc
```

Đây chính là tiền thân trực tiếp của **self-attention** trong Transformer. So sánh 2 điểm khác biệt cốt lõi mà [06-Transformer.md Chương 6.1](../06-Transformer/06-Transformer.md#chương-61--vì-sao-cần-transformer) và [Chương 6.3](../06-Transformer/06-Transformer.md#chương-63--kiến-trúc-transformer-gốc-encoder-và-decoder) trình bày chi tiết cơ chế:

- **Attention ở đây** chỉ dùng ở decoder, để "nhìn sang" Encoder (tương đương cross-attention, [Chương 6.4](../06-Transformer/06-Transformer.md#chương-64--cross-attention)) — Encoder/Decoder vẫn là RNN tuần tự bên trong.
- **Transformer tổng quát hóa** hoàn toàn: bỏ hẳn RNN, để **mọi vị trí** (không riêng decoder) tính attention score với **mọi vị trí khác** cùng lúc bằng 1 phép nhân ma trận — vừa bỏ được bottleneck của Seq2Seq, vừa song song hóa hoàn toàn thay vì phải chờ từng bước decode tuần tự.

## Chương 5.7 — Regularization và Normalization

Khi mạng đủ lớn, nó có thể "học thuộc lòng" dữ liệu train (bao gồm cả nhiễu) thay vì học quy luật tổng quát — gọi là **overfitting**: loss trên tập train tiếp tục giảm nhưng loss trên tập validation lại tăng trở lại:

```
Loss
 │         val loss
 │             ╭─────────────
 │            ╱
 │           ╱        train loss
 │          ╱     ╲________________
 │         ╱
 └──────────────────────────────────▶ epoch
        điểm val loss bắt đầu TĂNG trong khi train loss vẫn giảm = overfitting
```

**Dropout** (Srivastava et al., 2014): trong lúc train, ngẫu nhiên "tắt" (đặt về 0) một tỉ lệ neuron ở mỗi bước — buộc mạng không được phụ thuộc quá nhiều vào 1 vài neuron cụ thể, giúp tổng quát hóa tốt hơn:

```python
import torch
import torch.nn as nn

dropout = nn.Dropout(p=0.5)
x = torch.ones(10)

dropout.train()
print(dropout(x))   # train: ngẫu nhiên zero ~50% phần tử, phần còn lại nhân thêm 1/(1-p) để giữ nguyên kỳ vọng tổng

dropout.eval()
print(dropout(x))   # eval: KHÔNG drop gì, dùng nguyên vẹn toàn bộ mạng đã học để suy luận
```

**Normalization** chuẩn hóa phân phối giá trị trong mạng (đưa về mean≈0, variance≈1) để việc train ổn định hơn — nhưng **BatchNorm** và **LayerNorm** chuẩn hóa theo 2 chiều khác nhau:

```python
import torch
import torch.nn as nn

x = torch.randn(4, 6)   # (batch=4, features=6)

batch_norm = nn.BatchNorm1d(6)
layer_norm = nn.LayerNorm(6)

print(batch_norm(x).mean(dim=0).round(decimals=3))   # BatchNorm: chuẩn hóa theo CỘT (qua batch) -> mean/feature ≈ 0
print(layer_norm(x).mean(dim=1).round(decimals=3))   # LayerNorm: chuẩn hóa theo HÀNG (qua feature) -> mean/sample ≈ 0
```

| | BatchNorm | LayerNorm |
|---|---|---|
| Chuẩn hóa theo | Toàn batch, cho từng feature | Từng sample riêng lẻ, qua mọi feature |
| Phụ thuộc batch size | Có — batch nhỏ khiến ước lượng mean/var không ổn định | Không — tính độc lập cho từng sample |
| Dùng ở | CNN, MLP (batch cố định khi train/inference) | RNN, Transformer/LLM |

LLM hiện đại dùng LayerNorm (hoặc **RMSNorm** — biến thể đơn giản hóa, bỏ bước trừ mean, chỉ chia cho root-mean-square, xem [06-Transformer.md Chương 6.5](../06-Transformer/06-Transformer.md#chương-65--kiến-trúc-decoder-only-hiện-đại)) chứ không dùng BatchNorm — lý do: khi inference, LLM sinh **từng token một** (batch theo chiều token không ổn định/không tồn tại theo nghĩa như ảnh), và độ dài chuỗi thay đổi liên tục giữa các request — thống kê theo batch của BatchNorm không còn ý nghĩa ổn định trong bối cảnh đó, trong khi LayerNorm luôn chuẩn hóa độc lập theo từng token, không phụ thuộc các token khác trong batch.

## Chương 5.8 — Bài tập

1. Thêm 1 hidden layer nữa vào `MLP` ở [Chương 5.1](#chương-51--mạng-neural-cơ-bản) (3 lớp Linear thay vì 2), huấn luyện lại bài toán XOR ở [Chương 5.2](#chương-52--backpropagation-và-gradient-descent) — quan sát tốc độ hội tụ có đổi không.
2. So sánh optimizer `SGD` (không momentum) với `Adam` trên cùng bài toán XOR, giữ nguyên `lr` và số epoch — vẽ đường loss theo epoch của cả 2 để thấy sự khác biệt.
3. Cài đặt 1 mạng CNN nhỏ (2 lớp `Conv2d` + pooling) bằng PyTorch, huấn luyện phân loại chữ số viết tay trên tập MNIST (`torchvision.datasets.MNIST`).
4. Sửa `rnn_step` ở [Chương 5.4](#chương-54--rnnlstm) để in thêm norm (`np.linalg.norm`) của hidden state qua từng bước với chuỗi dài 50 bước thay vì 5 — quan sát hiện tượng gradient/hidden state co lại hoặc phình to dần.
5. Huấn luyện lại `SkipGram` ở [Chương 5.5](#chương-55--word-embeddings) với 1 corpus tiếng Việt lớn hơn (vài đoạn văn bản thật), thử `most_similar` với nhiều từ khác nhau — so sánh chất lượng với corpus đồ chơi trong bài.
6. Mở rộng ví dụ attention ở [Chương 5.6](#chương-56--seq2seq-và-attention) thành 1 vòng lặp decode đầy đủ (sinh nhiều token liên tiếp, mỗi bước cập nhật `decoder_state` bằng 1 cell RNN/GRU nhận `context` làm input) thay vì chỉ tính attention cho 1 bước.
7. Thêm `nn.Dropout(p=0.3)` và `nn.LayerNorm` vào `MLP` ở [Chương 5.1](#chương-51--mạng-neural-cơ-bản), huấn luyện trên 1 dataset nhỏ dễ overfit (ví dụ chỉ 20 sample), so sánh loss trên tập validation có/không có regularization.

## Chương 5.9 — Tài liệu tham khảo

- Rumelhart, Hinton & Williams, *Learning Representations by Back-Propagating Errors* (1986)
- LeCun et al., *Gradient-Based Learning Applied to Document Recognition* (CNN, 1998)
- Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997)
- Kim, *Convolutional Neural Networks for Sentence Classification* (TextCNN, 2014)
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (word2vec, 2013)
- Pennington, Socher & Manning, *GloVe: Global Vectors for Word Representation* (2014)
- Sutskever, Vinyals & Le, *Sequence to Sequence Learning with Neural Networks* (2014)
- Bahdanau, Cho & Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate* (2015)
- Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (2014)
- Ioffe & Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* (2015)
- Ba, Kiros & Hinton, *Layer Normalization* (2016)

📖 Đọc tiếp [06-Transformer.md](../06-Transformer/06-Transformer.md) để thấy self-attention thay thế hoàn toàn RNN và cơ chế attention thủ công ở trên như thế nào.
