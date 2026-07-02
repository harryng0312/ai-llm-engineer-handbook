# 06 — RAG (Retrieval-Augmented Generation)

> Chương này ghép LLM ([05-LLM](../05-LLM/README.md)) và Vector Database ([07-Vector-Database](../07-Vector-Database/README.md)) lại thành một pipeline hoàn chỉnh: cho LLM "tra cứu" tài liệu riêng trước khi trả lời, thay vì chỉ dựa vào kiến thức đã train.

> 📓 Toàn bộ code ví dụ trong chương này cũng có ở dạng notebook chạy được: [`06-RAG.ipynb`](./06-RAG.ipynb).

## Mục lục

1. [RAG là gì và tại sao cần](#1-rag-là-gì-và-tại-sao-cần)
2. [Chunking](#2-chunking)
3. [Embedding & Indexing](#3-embedding--indexing)
4. [Retriever: dense, sparse, hybrid](#4-retriever-dense-sparse-hybrid)
5. [Re-ranking](#5-re-ranking)
6. [Ráp thành pipeline hoàn chỉnh](#6-ráp-thành-pipeline-hoàn-chỉnh)
7. [Evaluation](#7-evaluation)
8. [Bài tập](#8-bài-tập)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. RAG là gì và tại sao cần

LLM có hai giới hạn cố hữu: (1) kiến thức bị "đóng băng" tại thời điểm train (không biết dữ liệu mới/nội bộ), (2) có xu hướng **ảo giác** (hallucination) — bịa thông tin nghe có vẻ hợp lý khi không chắc chắn.

**RAG (Retrieval-Augmented Generation)** giải quyết bằng cách: trước khi LLM sinh câu trả lời, hệ thống **truy xuất** (retrieve) các đoạn văn bản liên quan nhất từ một kho tài liệu riêng (qua vector search), rồi **nhét** các đoạn đó vào prompt làm ngữ cảnh — LLM chỉ cần "đọc hiểu và tổng hợp" thay vì "nhớ lại".

```
Câu hỏi -> [Retriever: tìm top-k đoạn văn liên quan] -> [Re-rank (tùy chọn)]
        -> Ghép (câu hỏi + các đoạn văn) thành prompt -> LLM sinh câu trả lời
```

Lợi ích: cập nhật kiến thức chỉ cần thay đổi dữ liệu trong vector DB (không cần train lại model), câu trả lời có thể **trích dẫn nguồn** (traceable), giảm hallucination vì model được "neo" vào văn bản thật.

## 2. Chunking

Tài liệu gốc (PDF, docx, trang web...) thường quá dài để nhét nguyên văn vào một embedding hay một prompt. **Chunking** là bước chia tài liệu thành các đoạn nhỏ trước khi embed và lưu vào vector DB.

Đánh đổi khi chọn kích thước chunk:
- **Chunk quá nhỏ** (vài chục từ): mất ngữ cảnh, một câu bị cắt giữa chừng làm embedding kém chính xác.
- **Chunk quá lớn** (vài nghìn từ): embedding bị "pha loãng" (trộn nhiều ý khác nhau vào 1 vector), và tốn budget prompt khi nhét vào LLM.

Kỹ thuật phổ biến, từ đơn giản đến phức tạp:

**Fixed-size chunking với overlap** — chia theo số ký tự/token cố định, cho các chunk liên tiếp **chồng lấn** một phần để tránh cắt đứt ý ở ranh giới:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ~500 ký tự mỗi chunk
    chunk_overlap=50,     # chồng lấn 50 ký tự giữa 2 chunk liên tiếp
    separators=["\n\n", "\n", ". ", " ", ""],  # ưu tiên cắt tại ranh giới đoạn/câu trước khi cắt cứng
)

text = open("document.txt", encoding="utf-8").read()
chunks = splitter.split_text(text)
print(f"Chia thành {len(chunks)} chunks, chunk đầu tiên dài {len(chunks[0])} ký tự")
```

`RecursiveCharacterTextSplitter` "đệ quy": nó thử cắt theo `\n\n` (đoạn văn) trước; nếu đoạn vẫn dài hơn `chunk_size`, thử `\n` (dòng); rồi đến `. ` (câu); cuối cùng mới cắt cứng theo ký tự. Điều này giữ chunk gọn theo ranh giới ngữ nghĩa (đoạn/câu) nhiều nhất có thể.

**Semantic chunking** — thay vì cắt theo số ký tự cố định, cắt tại điểm mà **độ tương đồng embedding giữa 2 câu liên tiếp giảm mạnh** (dấu hiệu chuyển chủ đề):

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def semantic_chunk(sentences: list[str], threshold: float = 0.5) -> list[list[str]]:
    embeddings = model.encode(sentences, normalize_embeddings=True)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        if sim < threshold:          # độ tương đồng giảm mạnh -> có thể đã đổi chủ đề
            chunks.append(current)
            current = []
        current.append(sentences[i])
    chunks.append(current)
    return chunks
```

Trong thực tế, đa số hệ thống production bắt đầu với fixed-size + overlap (rẻ, dễ đoán, đủ tốt) và chỉ chuyển sang semantic chunking khi đo được retrieval quality chưa đạt (xem [mục 7](#7-evaluation)).

## 3. Embedding & Indexing

Sau khi có chunks, dùng model embedding ([chương 07](../07-Vector-Database/README.md#4-các-model-embedding-bgee5gte)) để encode và lưu vào vector DB. Payload nên giữ lại **text gốc** và metadata (nguồn, số trang...) để khi retrieve xong còn dùng được:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(url="http://localhost:6333")

COLLECTION = "rag_docs"
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def index_chunks(chunks: list[str], source: str):
    vectors = model.encode(chunks, normalize_embeddings=True)
    client.upsert(collection_name=COLLECTION, points=[
        PointStruct(
            id=hash(f"{source}-{i}") & 0x7FFFFFFF,
            vector=vectors[i].tolist(),
            payload={"text": chunks[i], "source": source, "chunk_index": i},
        )
        for i in range(len(chunks))
    ])
```

## 4. Retriever: dense, sparse, hybrid

**Dense retrieval** (những gì đã làm ở trên): so khớp theo embedding, mạnh về mặt **ngữ nghĩa** (đồng nghĩa, diễn giải khác cách nói) nhưng đôi khi yếu với từ khóa hiếm/chính xác (mã lỗi, tên riêng, số hiệu).

**Sparse retrieval** (BM25 — hậu duệ của TF-IDF): so khớp theo **từ khóa xuất hiện chung**, mạnh với truy vấn có thuật ngữ chính xác, không cần embedding:

```python
from rank_bm25 import BM25Okapi

corpus = ["RAG kết hợp retrieval và generation.", "LoRA giúp fine-tune tiết kiệm VRAM.", "Qdrant hỗ trợ lọc metadata."]
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "fine-tune tiết kiệm VRAM"
scores = bm25.get_scores(query.lower().split())
print(scores)  # điểm BM25 cho từng document trong corpus
```

**Hybrid retrieval** — kết hợp cả hai, thường bằng cách chuẩn hóa điểm số của mỗi phương pháp rồi cộng có trọng số (hoặc dùng Reciprocal Rank Fusion — RRF):

```python
def reciprocal_rank_fusion(dense_ranked_ids: list[int], sparse_ranked_ids: list[int], k: int = 60) -> dict[int, float]:
    """RRF: điểm mỗi doc = tổng 1/(k + rank) qua các danh sách xếp hạng.
    Ưu điểm so với cộng điểm trực tiếp: không cần chuẩn hóa thang điểm giữa
    cosine similarity (dense) và BM25 score (sparse) vốn không cùng đơn vị.
    """
    scores: dict[int, float] = {}
    for ranked_list in (dense_ranked_ids, sparse_ranked_ids):
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))
```

Hybrid retrieval thường cho recall cao hơn dense-only hoặc sparse-only, đặc biệt với dữ liệu kỹ thuật (code, tài liệu có mã số, tên riêng) — đây là setup mặc định của nhiều framework RAG production hiện nay.

## 5. Re-ranking

Retriever (dense/sparse/hybrid) cần nhanh vì phải quét qua hàng triệu document, nên dùng các phép so khớp "rẻ" (cosine similarity, BM25) — độ chính xác có giới hạn. **Re-ranking** thêm một bước lọc lại: lấy top-k (ví dụ k=20) kết quả từ retriever, rồi dùng một model **cross-encoder** (đắt hơn nhưng chính xác hơn nhiều) để chấm điểm lại và chọn ra top-n cuối cùng (ví dụ n=5) đưa vào prompt.

Khác biệt cốt lõi: embedding model (bi-encoder) encode query và document **riêng biệt** rồi so sánh vector — nhanh vì document có thể encode trước (offline); cross-encoder nhận **cả cặp (query, document) cùng lúc** làm input, cho phép model học sự tương tác giữa 2 văn bản trực tiếp — chính xác hơn nhưng phải chạy online cho từng cặp, không thể tính trước.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

query = "Làm sao giảm chi phí VRAM khi fine-tune LLM?"
candidates = [
    "LoRA đóng băng trọng số gốc, chỉ train 2 ma trận rank thấp để tiết kiệm VRAM.",
    "Qdrant hỗ trợ lọc metadata khi tìm kiếm vector.",
    "Prompt engineering giúp điều khiển hành vi model mà không cần train lại.",
]

pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)   # điểm relevance, càng cao càng liên quan

ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
for score, doc in ranked:
    print(f"{score:.3f} — {doc}")
```

## 6. Ráp thành pipeline hoàn chỉnh

Ghép chunking → indexing → hybrid retrieval → re-ranking → generation (qua Ollama/Qwen như ở [chương 05](../05-LLM/README.md#7-serving-ollama-vllm-llamacpp)):

```python
import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION = "rag_docs"

def retrieve(query: str, top_k: int = 10, rerank_top_n: int = 3) -> list[str]:
    query_vector = embed_model.encode(query, normalize_embeddings=True)
    hits = qdrant.search(collection_name=COLLECTION, query_vector=query_vector.tolist(), limit=top_k)
    candidates = [hit.payload["text"] for hit in hits]

    if not candidates:
        return []

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
    return [doc for _, doc in reranked[:rerank_top_n]]

def rag_answer(query: str) -> str:
    context_chunks = retrieve(query)
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))

    prompt = f"""Trả lời câu hỏi CHỈ dựa trên ngữ cảnh dưới đây. Nếu ngữ cảnh không đủ thông tin, nói rõ "Không tìm thấy thông tin liên quan".

Ngữ cảnh:
{context}

Câu hỏi: {query}

Trả lời (trích dẫn số [1], [2]... tương ứng nguồn đã dùng):"""

    response = ollama.chat(model="qwen2.5:7b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

print(rag_answer("Làm sao giảm chi phí VRAM khi fine-tune LLM?"))
```

Lưu ý quan trọng trong prompt trên: ràng buộc **"chỉ dựa trên ngữ cảnh"** + yêu cầu trích dẫn nguồn — đây là kỹ thuật giảm hallucination hiệu quả nhất trong RAG (ép model "bám" vào context thay vì dùng kiến thức nội tại khi không chắc).

## 7. Evaluation

RAG có 2 thành phần độc lập cần đánh giá riêng: **retrieval** (tìm đúng đoạn văn chưa?) và **generation** (trả lời có đúng/trung thực với ngữ cảnh chưa?). Đánh giá gộp cả pipeline dễ che lấp lỗi — ví dụ retrieval tệ nhưng LLM "đoán đúng" nhờ kiến thức nội tại, hoặc retrieval tốt nhưng LLM vẫn hallucinate.

**Metric cho retrieval** (cần bộ câu hỏi kèm ground-truth document/chunk liên quan):

- **Recall@k**: trong top-k kết quả trả về, có bao nhiêu % chứa document đúng (ground truth)?
- **MRR (Mean Reciprocal Rank)**: trung bình của `1/rank` của document đúng đầu tiên — phạt nặng nếu document đúng nằm ở vị trí thấp.

```python
def recall_at_k(retrieved_ids: list[list[int]], relevant_ids: list[set[int]], k: int) -> float:
    hits = sum(1 for ret, rel in zip(retrieved_ids, relevant_ids) if set(ret[:k]) & rel)
    return hits / len(retrieved_ids)

def mrr(retrieved_ids: list[list[int]], relevant_ids: list[set[int]]) -> float:
    reciprocal_ranks = []
    for ret, rel in zip(retrieved_ids, relevant_ids):
        rank = next((i + 1 for i, doc_id in enumerate(ret) if doc_id in rel), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

**Metric cho generation** — thường không có "đáp án đúng duy nhất" nên phổ biến nhất là dùng chính LLM làm giám khảo (**LLM-as-judge**), ví dụ theo framework [RAGAS](https://github.com/explodinggradients/ragas):

- **Faithfulness**: câu trả lời có bịa thông tin ngoài context không? (đo bằng cách tách câu trả lời thành các claim nhỏ, hỏi LLM từng claim có được context hỗ trợ không).
- **Answer Relevance**: câu trả lời có thực sự trả lời đúng câu hỏi không (không lạc đề)?
- **Context Precision/Recall**: trong các chunk được retrieve, bao nhiêu % thực sự hữu ích cho câu trả lời (precision), và có bỏ sót chunk cần thiết không (recall)?

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question": ["Làm sao giảm chi phí VRAM khi fine-tune LLM?"],
    "answer": ["Dùng LoRA/QLoRA để chỉ train một phần nhỏ tham số và lượng tử hóa trọng số gốc."],
    "contexts": [["LoRA đóng băng trọng số gốc, chỉ train 2 ma trận rank thấp để tiết kiệm VRAM."]],
    "ground_truth": ["Sử dụng LoRA hoặc QLoRA."],
})

result = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
```

Thực dụng: xây một bộ ~20-50 câu hỏi thật (kèm câu trả lời mẫu) đại diện cho use case của bạn, chạy lại bộ eval này **mỗi khi đổi** chunk size, model embedding, hoặc prompt — đây là cách duy nhất biết một thay đổi có thực sự cải thiện hệ thống hay chỉ "cảm giác tốt hơn".

## 8. Bài tập

1. Lấy 1 tài liệu dài (ví dụ 1 chương sách này), thử 3 cách chia chunk (`chunk_size=200/500/1000`), index cả 3 vào 3 collection Qdrant riêng, so sánh câu trả lời cho cùng 5 câu hỏi.
2. Cài đặt hybrid retrieval bằng Reciprocal Rank Fusion ở [mục 4](#4-retriever-dense-sparse-hybrid), so sánh recall@5 với dense-only trên một bộ câu hỏi có chứa thuật ngữ/mã số chính xác.
3. Thêm bước re-ranking vào pipeline ở [mục 6](#6-ráp-thành-pipeline-hoàn-chỉnh), đo thời gian tăng thêm và so sánh chất lượng câu trả lời trước/sau khi có re-ranking.
4. Viết 15 cặp (câu hỏi, câu trả lời mẫu, context mẫu), chạy `ragas.evaluate`, thử cố ý viết một câu trả lời "bịa thêm thông tin ngoài context" và quan sát điểm `faithfulness` giảm thế nào.

## 9. Tài liệu tham khảo

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (RAG gốc, 2020)
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (RRF, 2009)
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023)
- [LangChain text splitters docs](https://python.langchain.com/docs/how_to/#text-splitters) · [Qdrant hybrid search](https://qdrant.tech/documentation/concepts/hybrid-queries/) · [RAGAS docs](https://docs.ragas.io)
