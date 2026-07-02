# 07 — Embedding & Vector Database

> Chương này trả lời câu hỏi: "Làm sao biến văn bản thành vector để máy tính so sánh được 'độ giống nhau về ý nghĩa', và lưu/tìm hàng triệu vector đó nhanh thế nào?" Đây là nền tảng bắt buộc trước khi học RAG ở chương sau.

## Mục lục

1. [Embedding là gì](#1-embedding-là-gì)
2. [Contrastive Learning](#2-contrastive-learning)
3. [Triplet Loss](#3-triplet-loss)
4. [Các model embedding: BGE/E5/GTE](#4-các-model-embedding-bgee5gte)
5. [Tìm kiếm vector: từ brute-force đến ANN](#5-tìm-kiếm-vector-từ-brute-force-đến-ann)
6. [FAISS](#6-faiss)
7. [Qdrant](#7-qdrant)
8. [Bài tập](#8-bài-tập)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Embedding là gì

**Embedding** là một hàm ánh xạ dữ liệu (văn bản, ảnh...) thành một vector số thực có chiều cố định (ví dụ 768, 1024 chiều), sao cho **khoảng cách hình học giữa các vector phản ánh độ tương đồng về ngữ nghĩa**: hai câu có ý nghĩa gần nhau → vector gần nhau (cosine similarity cao); hai câu không liên quan → vector xa nhau.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

sentences = [
    "Con mèo đang ngủ trên ghế sofa.",
    "Một chú mèo nằm nghỉ trên trường kỷ.",   # gần nghĩa với câu 1
    "Thị trường chứng khoán tăng điểm hôm nay.",  # không liên quan
]
vectors = model.encode(sentences, normalize_embeddings=True)  # (3, 384)

def cosine(a, b):
    return float(np.dot(a, b))  # đã normalize nên dot product = cosine similarity

print("Câu 1 vs Câu 2:", cosine(vectors[0], vectors[1]))  # ~0.7-0.9, cao
print("Câu 1 vs Câu 3:", cosine(vectors[0], vectors[2]))  # thấp
```

Đây chính là "viên gạch" cho RAG: thay vì so khớp từ khóa (BM25/full-text search), ta so khớp **ý nghĩa** — câu hỏi "thú cưng nào hay ngủ nhiều" vẫn tìm ra đoạn văn nói về mèo dù không chung từ nào.

## 2. Contrastive Learning

Câu hỏi cốt lõi: model embedding được train thế nào để "câu giống nhau → vector gần nhau"? Câu trả lời phổ biến nhất: **contrastive learning**.

Ý tưởng: với mỗi mẫu **anchor**, ta có một mẫu **positive** (giống nghĩa/liên quan) và nhiều mẫu **negative** (không liên quan, thường lấy ngẫu nhiên từ batch — gọi là *in-batch negatives*). Loss huấn luyện model kéo `anchor` lại gần `positive` và đẩy xa khỏi tất cả `negative`.

Hàm loss phổ biến nhất là **InfoNCE** (Info Noise-Contrastive Estimation):

```
L = -log( exp(sim(a, p) / τ) / Σ_i exp(sim(a, n_i) / τ) )
```

trong đó `sim` thường là cosine similarity, `τ` (temperature) là hệ số làm "sắc nét" phân phối — τ nhỏ khiến model phân biệt gắt gao hơn giữa positive/negative.

Cài đặt tối giản bằng PyTorch để thấy rõ cơ chế (huấn luyện trên batch câu anchor/positive):

```python
import torch
import torch.nn.functional as F

def info_nce_loss(anchor_emb: torch.Tensor, positive_emb: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """anchor_emb, positive_emb: (batch, dim), đã normalize (L2 norm = 1).
    Với mỗi anchor[i], positive[i] là mẫu dương, còn positive[j != i] trong
    cùng batch đóng vai trò negative (in-batch negatives) -- đây là cách
    BGE/E5/GTE và hầu hết sentence embedding model hiện đại được train.
    """
    sim_matrix = anchor_emb @ positive_emb.T / temperature   # (batch, batch)
    labels = torch.arange(sim_matrix.size(0))                # đường chéo = cặp đúng
    return F.cross_entropy(sim_matrix, labels)

batch = 4
anchor = F.normalize(torch.randn(batch, 32), dim=-1)
positive = F.normalize(torch.randn(batch, 32), dim=-1)
loss = info_nce_loss(anchor, positive)
print(loss.item())
```

Vì sao dùng `cross_entropy` trên ma trận similarity? Nó tương đương việc coi bài toán như "phân loại": trong hàng `i` của `sim_matrix`, đáp án đúng là cột `i` (chính positive tương ứng) — cross-entropy ép model cho điểm số cao nhất ở đúng vị trí đó.

## 3. Triplet Loss

Trước InfoNCE, **triplet loss** là hàm loss phổ biến cho embedding (dùng nhiều trong face recognition — FaceNet, 2015). Mỗi mẫu train là bộ ba `(anchor, positive, negative)` tường minh (không dựa vào in-batch negatives), loss ép khoảng cách `d(anchor, positive)` nhỏ hơn `d(anchor, negative)` ít nhất một `margin`:

```
L = max(0, d(a, p) - d(a, n) + margin)
```

```python
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin: float = 0.2):
    d_pos = 1 - F.cosine_similarity(anchor, positive)  # "khoảng cách" = 1 - cos sim
    d_neg = 1 - F.cosine_similarity(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()

anchor = F.normalize(torch.randn(4, 32), dim=-1)
positive = F.normalize(torch.randn(4, 32), dim=-1)
negative = F.normalize(torch.randn(4, 32), dim=-1)
print(triplet_loss(anchor, positive, negative).item())
```

**So sánh với InfoNCE**: triplet loss cần chọn negative một cách thủ công/heuristic (hard negative mining — chọn negative "khó", gần giống positive, để việc học hiệu quả hơn); InfoNCE tận dụng toàn bộ batch làm negative miễn phí nên hiệu quả hơn khi batch size lớn — đây là lý do các model embedding SOTA hiện nay (BGE, E5, GTE) hầu hết dùng biến thể của InfoNCE (kết hợp thêm hard negatives được mine trước).

## 4. Các model embedding: BGE/E5/GTE

Ba họ model embedding mã nguồn mở phổ biến nhất hiện nay, đều dựa trên kiến trúc encoder (BERT-like) và train bằng contrastive learning ở quy mô lớn:

| Model | Tổ chức | Đặc điểm |
|---|---|---|
| **BGE** (BAAI General Embedding) | BAAI | Có bản `small`/`base`/`large`, hỗ trợ tiếng Trung + đa ngôn ngữ (`bge-m3`), thường yêu cầu thêm *instruction prefix* cho câu query |
| **E5** | Microsoft | Yêu cầu prefix `"query: "` / `"passage: "` khi encode, phân biệt rõ vai trò query vs document |
| **GTE** (General Text Embedding) | Alibaba | Không cần prefix đặc biệt, hỗ trợ context dài hơn ở các bản mới |

Điểm cần nhớ khi dùng: nhiều model (E5, BGE) train với **prefix bất đối xứng** cho query và passage — nếu bỏ qua bước này, chất lượng retrieval giảm rõ rệt.

```python
from sentence_transformers import SentenceTransformer

# BGE: khuyến nghị thêm instruction cho query (không cần cho passage)
bge = SentenceTransformer("BAAI/bge-base-en-v1.5")
query_emb = bge.encode("Represent this sentence for searching relevant passages: LLM là gì?")
passage_emb = bge.encode("LLM (Large Language Model) là mô hình ngôn ngữ lớn...")

# E5: bắt buộc prefix "query: " và "passage: "
e5 = SentenceTransformer("intfloat/multilingual-e5-base")
query_emb_e5 = e5.encode("query: LLM là gì?")
passage_emb_e5 = e5.encode("passage: LLM (Large Language Model) là mô hình ngôn ngữ lớn...")
```

Cách chọn model thực dụng: dùng benchmark [MTEB](https://huggingface.co/spaces/mteb/leaderboard) để so sánh theo ngôn ngữ/task cần dùng, ưu tiên bản `base`/`small` cho production (độ trễ thấp) trừ khi độ chính xác retrieval là ưu tiên số 1.

## 5. Tìm kiếm vector: từ brute-force đến ANN

Có embedding rồi, bài toán tiếp theo: cho một vector query, tìm `k` vector gần nhất trong hàng triệu vector đã lưu (**k-NN search**).

**Brute-force** (tính cosine similarity với từng vector) chính xác 100% nhưng độ phức tạp `O(n × d)` — với `n` = vài triệu vector, `d` = 1024 chiều, mỗi query mất hàng trăm ms đến vài giây, không chấp nhận được cho production.

**ANN (Approximate Nearest Neighbor)**: đánh đổi một chút độ chính xác (recall < 100%) để lấy tốc độ nhanh hơn hàng trăm–hàng nghìn lần. Hai họ thuật toán phổ biến:

- **IVF (Inverted File Index)**: chia không gian vector thành `nlist` cụm (bằng k-means), khi query chỉ tìm trong `nprobe` cụm gần centroid nhất thay vì toàn bộ dữ liệu.
- **HNSW (Hierarchical Navigable Small World)**: xây một đồ thị nhiều tầng, tầng trên thưa (nhảy xa, nhanh), tầng dưới dày (chính xác), tìm kiếm bằng cách "đi từ trên xuống" — đây là thuật toán mặc định của Qdrant và nhiều vector DB hiện đại vì recall cao, không cần train (không như IVF cần k-means trước).

## 6. FAISS

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) là thư viện ANN chạy **in-memory**, không phải một database đầy đủ (không có persistence/metadata filter tích hợp sẵn) — phù hợp để prototype nhanh hoặc nhúng vào ứng dụng không cần server riêng.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
docs = [
    "RAG kết hợp retrieval và generation để giảm ảo giác của LLM.",
    "FAISS là thư viện tìm kiếm vector hiệu năng cao của Meta.",
    "LoRA giúp fine-tune LLM tiết kiệm VRAM.",
    "Ollama cho phép chạy LLM local qua API đơn giản.",
]
doc_vectors = model.encode(docs, normalize_embeddings=True).astype("float32")

dim = doc_vectors.shape[1]
index = faiss.IndexFlatIP(dim)   # IP = Inner Product; với vector đã normalize -> tương đương cosine sim
index.add(doc_vectors)

query = "Làm sao giảm chi phí VRAM khi fine-tune model?"
query_vector = model.encode([query], normalize_embeddings=True).astype("float32")

scores, indices = index.search(query_vector, k=2)
for score, idx in zip(scores[0], indices[0]):
    print(f"{score:.3f} — {docs[idx]}")
```

Với dữ liệu lớn hơn (>100k vector), thay `IndexFlatIP` bằng `IndexIVFFlat` hoặc `IndexHNSWFlat` để đánh đổi lấy tốc độ:

```python
quantizer = faiss.IndexFlatIP(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, 100)  # 100 cụm (nlist)
index_ivf.train(doc_vectors)   # IVF cần bước "train" k-means trước khi add
index_ivf.add(doc_vectors)
index_ivf.nprobe = 10           # tìm trong 10/100 cụm gần nhất -> đánh đổi tốc độ/độ chính xác
```

## 7. Qdrant

[Qdrant](https://qdrant.tech) là vector database đầy đủ: có server, persistence, **metadata filter** (lọc kết hợp điều kiện thông thường với tìm kiếm vector — ví dụ "tìm đoạn văn gần nghĩa nhất NHƯNG chỉ trong tài liệu có `category = 'legal'`"), phù hợp production hơn FAISS.

Chạy Qdrant local bằng Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

Tạo collection, upsert và search bằng `qdrant-client`:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(url="http://localhost:6333")

COLLECTION = "handbook_docs"
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

docs = [
    {"text": "RAG kết hợp retrieval và generation.", "category": "rag"},
    {"text": "LoRA giúp fine-tune tiết kiệm VRAM.", "category": "llm"},
    {"text": "Qdrant hỗ trợ lọc metadata khi search.", "category": "vector-db"},
]
vectors = model.encode([d["text"] for d in docs], normalize_embeddings=True)

client.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(id=i, vector=vectors[i].tolist(), payload=docs[i])
        for i in range(len(docs))
    ],
)

query_vector = model.encode("Công cụ nào giúp lọc theo metadata khi tìm vector?", normalize_embeddings=True)

results = client.search(
    collection_name=COLLECTION,
    query_vector=query_vector.tolist(),
    query_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="vector-db"))]),
    limit=3,
)
for r in results:
    print(r.score, r.payload["text"])
```

**Khi nào dùng FAISS, khi nào dùng Qdrant**: FAISS phù hợp khi dữ liệu tương đối tĩnh, chạy embedded trong 1 process, không cần filter phức tạp (ví dụ demo, batch job). Qdrant (hoặc Milvus, Weaviate, pgvector) phù hợp khi cần server riêng, dữ liệu cập nhật liên tục (insert/update/delete), cần filter theo metadata, hoặc cần scale ra nhiều máy.

## 8. Bài tập

1. Encode 20 câu tiếng Việt thuộc 3 chủ đề khác nhau (thể thao, công nghệ, ẩm thực) bằng `bge-small`, dùng `sklearn` để vẽ chúng ra 2D (t-SNE hoặc PCA), quan sát các cụm có tách biệt theo chủ đề không.
2. Cài đặt lại `info_nce_loss` nhưng thêm **hard negative** tường minh (không chỉ dựa vào in-batch negative) — so sánh giá trị loss.
3. Build một `IndexHNSWFlat` trong FAISS với 50.000 vector ngẫu nhiên, so sánh thời gian search với `IndexFlatIP` trên cùng dữ liệu, đo recall@10 (so với kết quả brute-force làm ground truth).
4. Dựng Qdrant bằng Docker, nạp 100 đoạn văn bản từ một file bất kỳ, viết hàm `search(query, category=None)` hỗ trợ filter tùy chọn.

## 9. Tài liệu tham khảo

- Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE, 2018)
- Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering* (Triplet Loss, 2015)
- Xiao et al., *C-Pack: Packed Resources For General Chinese Embeddings* (BGE, 2023)
- Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5, 2022)
- Li et al., *Towards General Text Embeddings with Multi-stage Contrastive Learning* (GTE, 2023)
- Johnson et al., *Billion-scale similarity search with GPUs* (FAISS, 2017)
- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using HNSW* (2018)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) · [Qdrant docs](https://qdrant.tech/documentation/) · [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
