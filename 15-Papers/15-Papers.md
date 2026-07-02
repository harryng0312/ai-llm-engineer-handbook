# 15 — Papers: Danh sách đọc có chú giải

> Các chương 04-11 đều có mục "Tài liệu tham khảo" riêng, nhưng rải rác qua nhiều file thì khó thấy được bức tranh tổng thể: paper nào là nền tảng bắt buộc, paper nào chỉ cần biết tên, paper nào nên để dành đọc sau. Chương này gom tất cả lại theo chủ đề, thêm chú giải ngắn "vì sao paper này quan trọng", và bổ sung thêm những paper nâng cao chưa được nhắc tới trong phần thân sách.

## Mục lục

1. [Cách dùng chương này](#chương-151--cách-dùng-chương-này)
2. [Nền tảng Transformer & LLM](#chương-152--nền-tảng-transformer--llm)
3. [Embedding & Vector Search](#chương-153--embedding--vector-search)
4. [RAG](#chương-154--rag)
5. [Agent & Reasoning](#chương-155--agent--reasoning)
6. [Alignment & Fine-tuning nâng cao](#chương-156--alignment--fine-tuning-nâng-cao)
7. [Hiệu năng & Serving nâng cao](#chương-157--hiệu-năng--serving-nâng-cao)
8. [Lịch sử & Scaling Laws](#chương-158--lịch-sử--scaling-laws)
9. [RAG & Agent nâng cao](#chương-159--rag--agent-nâng-cao)
10. [Xem trước: Diffusion & Multimodal](#chương-1510--xem-trước-diffusion--multimodal)
11. [Cách đọc một paper AI/ML hiệu quả](#chương-1511--cách-đọc-một-paper-aiml-hiệu-quả)
12. [Bảng tổng hợp theo dòng thời gian](#chương-1512--bảng-tổng-hợp-theo-dòng-thời-gian)
13. [Tài liệu tham khảo](#chương-1513--tài-liệu-tham-khảo)

---

## Chương 15.1 — Cách dùng chương này

Không cần đọc hết danh sách này theo thứ tự trước khi học các chương khác — ngược lại, nên đọc **song song**: khi học tới một khái niệm trong chương 04-11 mà muốn hiểu sâu hơn phần "vì sao nó hoạt động", quay lại đây tìm paper tương ứng. Mỗi mục 2-9 tương ứng gần như 1-1 với một chương lý thuyết đã viết, cộng thêm phần mở rộng "nâng cao" cho ai muốn đọc thêm ngoài phạm vi sách.

Ba mức độ ưu tiên gợi ý (đánh dấu trong ngoặc ở mỗi paper):

- **[Bắt buộc]** — nên đọc ít nhất phần abstract + hình minh họa trước khi coi là hiểu chương liên quan.
- **[Nên đọc]** — giúp hiểu sâu hơn, không đọc cũng không cản trở việc học tiếp.
- **[Tham khảo thêm]** — dành cho ai muốn đi sâu hơn phạm vi cuốn sách này (hướng nghiên cứu, đọc khi cần).

## Chương 15.2 — Nền tảng Transformer & LLM

*Song song với [04-Transformer.md](../04-Transformer/04-Transformer.md) (kiến trúc gốc, tokenization, decoder-only hiện đại).*

- **[Bắt buộc] Vaswani et al., *Attention Is All You Need* (2017)** — paper gốc giới thiệu kiến trúc Transformer, nền tảng của mọi LLM hiện đại. Đọc kỹ Hình 1 (kiến trúc encoder-decoder đầy đủ) cùng phần "Kiến trúc Transformer gốc" trong [04-Transformer.md](../04-Transformer/04-Transformer.md), rồi phần công thức scaled dot-product attention cùng phần "Kiến trúc Decoder-only hiện đại".
- **[Nên đọc] Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE, 2015)** — thuật toán Byte-Pair Encoding, nền tảng tokenizer của GPT/Llama/Qwen. Xem cài đặt tối giản ở phần "Tokenization" trong [04-Transformer.md](../04-Transformer/04-Transformer.md).
- **[Tham khảo thêm] Kudo & Richardson, *SentencePiece* (2018)** — biến thể tokenizer dùng trong T5, Llama, Qwen, xử lý được cả ngôn ngữ không phân tách từ bằng khoảng trắng.
- **[Bắt buộc] Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)** — nguồn gốc của RoPE, kỹ thuật mã hóa vị trí dùng trong hầu hết LLM hiện đại (Llama, Qwen). Đọc cùng phần "RoPE" trong [04-Transformer.md](../04-Transformer/04-Transformer.md).
- **[Nên đọc] Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)** — giải thích cách "chuyển đổi" một model MHA có sẵn sang GQA mà không cần train lại từ đầu, chi tiết hơn phần tóm tắt ở phần "GQA/MQA" trong [04-Transformer.md](../04-Transformer/04-Transformer.md).
- **[Bắt buộc] Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)** — kỹ thuật fine-tune tiết kiệm tham số đứng sau hầu hết công cụ fine-tune LLM ngày nay. Xem thực hành ở [08-Fine-Tuning.md](../08-Fine-Tuning/08-Fine-Tuning.md).
- **[Nên đọc] Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)** — kết hợp lượng tử hóa 4-bit với LoRA, lý do một model 7B fine-tune được trên GPU 8-12GB. Xem [08-Fine-Tuning.md](../08-Fine-Tuning/08-Fine-Tuning.md).
- **[Nên đọc] Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)** — paper của vLLM, giải thích PagedAttention chi tiết hơn sơ đồ tóm tắt ở phần "KV Cache" trong [04-Transformer.md](../04-Transformer/04-Transformer.md) và phần Serving trong [12-Deployment.md](../12-Deployment/12-Deployment.md).

## Chương 15.3 — Embedding & Vector Search

*Song song với [06-Embedding.md](../06-Embedding/06-Embedding.md).*

- **[Bắt buộc] Oord et al., *Representation Learning with Contrastive Predictive Coding* (2018)** — nguồn gốc hàm loss InfoNCE, nền tảng lý thuyết cho cách train hầu hết embedding model hiện đại. Xem phần "Contrastive Learning" trong [06-Embedding.md](../06-Embedding/06-Embedding.md).
- **[Nên đọc] Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering* (2015)** — nguồn gốc triplet loss, hữu ích để hiểu vì sao các phương pháp mới (InfoNCE) lại thay thế nó ở hầu hết bài toán embedding văn bản. Xem phần "Triplet Loss" trong [06-Embedding.md](../06-Embedding/06-Embedding.md).
- **[Tham khảo thêm] Xiao et al., *C-Pack: Packed Resources For General Chinese Embeddings* (2023)** — paper của BGE.
- **[Tham khảo thêm] Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (2022)** — paper của E5, giải thích lý do cần prefix `"query: "`/`"passage: "` khi encode.
- **[Nên đọc] Johnson et al., *Billion-scale similarity search with GPUs* (2017)** — paper của FAISS.
- **[Bắt buộc] Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs* (2018)** — thuật toán HNSW, cơ chế index mặc định của Qdrant và hầu hết vector DB hiện đại. Nên đọc để hiểu vì sao search trên hàng triệu vector vẫn nhanh — xem phần "Tìm kiếm vector" trong [06-Embedding.md](../06-Embedding/06-Embedding.md).

## Chương 15.4 — RAG

*Song song với [07-RAG.md](../07-RAG/07-RAG.md).*

- **[Bắt buộc] Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)** — paper đặt tên và định hình kiến trúc RAG. Đáng đọc kỹ vì bản gốc train luôn retriever cùng generator (end-to-end), khác với cách dùng retriever "đóng băng" phổ biến hiện nay ở [07-RAG.md](../07-RAG/07-RAG.md).
- **[Nên đọc] Izacard & Grave, *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering* (2020)** — Fusion-in-Decoder (FiD), một hướng khác của RAG: đưa nhiều passage vào decoder cùng lúc thay vì chỉ nối chuỗi vào 1 prompt.
- **[Nên đọc] Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)** — nền tảng lý thuyết của BM25, vẫn là baseline mạnh cho sparse retrieval. Xem phần "Retriever" trong [07-RAG.md](../07-RAG/07-RAG.md).
- **[Tham khảo thêm] Cormack et al., *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (2009)** — RRF, kỹ thuật hợp nhất kết quả dense + sparse retrieval.
- **[Bắt buộc] Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023)** — framework đánh giá RAG tự động (faithfulness, answer relevance, context precision/recall) dùng ở phần "Evaluation" trong [07-RAG.md](../07-RAG/07-RAG.md).

## Chương 15.5 — Agent & Reasoning

*Song song với [09-LangChain.md](../09-LangChain/09-LangChain.md), [10-LangGraph.md](../10-LangGraph/10-LangGraph.md), [11-Agent.md](../11-Agent/11-Agent.md).*

- **[Bắt buộc] Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)** — paper định hình vòng lặp reasoning + action mà gần như mọi agent framework hiện nay (LangGraph, AutoGen...) đều dựa trên. Đọc kỹ trước phần "Agent là gì" trong [11-Agent.md](../11-Agent/11-Agent.md).
- **[Nên đọc] Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (2023)** — mở rộng ReAct thành tìm kiếm đa nhánh có backtrack, xem phần "Chiến lược Planning nâng cao" trong [11-Agent.md](../11-Agent/11-Agent.md).
- **[Nên đọc] Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* (2023)** — agent tự phê bình câu trả lời của chính mình để cải thiện mà không cần train lại trọng số.
- **[Tham khảo thêm] Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* (2023)** — paper đứng sau framework AutoGen, xem phần "Multi-Agent" trong [11-Agent.md](../11-Agent/11-Agent.md).
- **[Tham khảo thêm] Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* (2023)**, **Mialon et al., *GAIA: A Benchmark for General AI Assistants* (2023)**, **Liu et al., *AgentBench: Evaluating LLMs as Agents* (2023)** — ba benchmark chuẩn để đánh giá agent, xem phần "Agent Evaluation" trong [11-Agent.md](../11-Agent/11-Agent.md).
- **[Nên đọc] Anthropic, *Model Context Protocol Specification* (2024)** và **Anthropic, *Building Effective Agents* (2024)** — đặc tả kỹ thuật của MCP và bài viết thực dụng về khi nào nên/không nên dùng agent phức tạp thay vì workflow đơn giản.

## Chương 15.6 — Alignment & Fine-tuning nâng cao

*Mở rộng thêm cho [08-Fine-Tuning.md](../08-Fine-Tuning/08-Fine-Tuning.md) — sách mới dừng ở LoRA/QLoRA (fine-tune để model học domain/format), còn alignment (dạy model "cư xử đúng ý người") là một chủ đề riêng.*

- **[Nên đọc] Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (2022)** — paper InstructGPT, giới thiệu quy trình RLHF (Reinforcement Learning from Human Feedback) 3 bước (SFT → reward model → PPO) mà ChatGPT được xây dựng theo.
- **[Nên đọc] Rafailov et al., *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model* (2023)** — DPO, thay thế RLHF bằng 1 hàm loss đơn giản hơn (không cần train reward model + PPO riêng), lý do phần lớn model mở hiện nay (Llama, Qwen instruct) dùng DPO thay vì RLHF đầy đủ.
- **[Tham khảo thêm] Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (2022)** — thay một phần feedback của con người bằng feedback từ chính LLM dựa trên một bộ "hiến pháp" (nguyên tắc) viết sẵn, giảm phụ thuộc vào lượng lớn dữ liệu gán nhãn thủ công.
- **[Tham khảo thêm] Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (2021)** — FLAN, paper đặt nền móng cho instruction tuning (fine-tune trên tập lớn các task được viết dưới dạng instruction) trước khi RLHF/DPO xuất hiện.

## Chương 15.7 — Hiệu năng & Serving nâng cao

*Mở rộng thêm cho [12-Deployment.md](../12-Deployment/12-Deployment.md) và phần KV Cache trong [04-Transformer.md](../04-Transformer/04-Transformer.md) — các kỹ thuật tối ưu tốc độ/bộ nhớ dùng trong production nhưng sách chưa đi sâu.*

- **[Nên đọc] Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)** — tối ưu cách attention đọc/ghi bộ nhớ GPU (không phải giảm số phép tính, mà giảm số lần truy cập HBM chậm), tăng tốc đáng kể mà không đổi kết quả toán học. Gần như mọi framework serving hiện nay (vLLM, TGI, Ollama) đều dùng biến thể của kỹ thuật này.
- **[Tham khảo thêm] Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)** — bản cải tiến, tối ưu thêm việc chia công việc giữa các luồng GPU.
- **[Tham khảo thêm] Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023)** — dùng một model nhỏ "đoán trước" nhiều token, model lớn chỉ cần verify (rẻ hơn tự sinh từng token), tăng tốc inference mà không đổi chất lượng output.
- **[Tham khảo thêm] Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022)** — lượng tử hóa sau khi train (khác QLoRA lượng tử hóa để fine-tune), nền tảng của nhiều định dạng GGUF dùng trong llama.cpp/Ollama.
- **[Tham khảo thêm] Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017)** và **Fedus et al., *Switch Transformers* (2021)** — kiến trúc Mixture-of-Experts (MoE): tăng số tham số model mà không tăng tương ứng chi phí tính toán mỗi token, dùng trong nhiều LLM SOTA hiện nay (Mixtral, DeepSeek, Qwen-MoE...).

## Chương 15.8 — Lịch sử & Scaling Laws

*Bối cảnh: hiểu LLM hiện đại đến từ đâu, không bắt buộc để dùng được các chương 04-11 nhưng giúp có bức tranh đầy đủ hơn.*

- **[Tham khảo thêm] Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)** — Word2Vec, thế hệ embedding đầu tiên phổ biến (trước BERT/BGE/E5 ở chương 06), đáng đọc để thấy ý tưởng "vector hóa ý nghĩa" đã có từ rất sớm.
- **[Nên đọc] Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018)** — hướng "encoder-only", vẫn là nền tảng của nhiều embedding model hiện nay. Xem so sánh 3 dòng kiến trúc ở [05-BERT-T5-GPT.md](../05-BERT-T5-GPT/05-BERT-T5-GPT.md).
- **[Nên đọc] Radford et al., *Improving Language Understanding by Generative Pre-Training* (2018, GPT-1)** và **Brown et al., *Language Models Are Few-Shot Learners* (2020, GPT-3)** — hai cột mốc cho thấy decoder-only scale lên thì xuất hiện khả năng few-shot mà không cần fine-tune, tiền đề trực tiếp cho prompt engineering ở [04-Transformer.md](../04-Transformer/04-Transformer.md). Xem thêm lịch sử GPT ở [05-BERT-T5-GPT.md](../05-BERT-T5-GPT/05-BERT-T5-GPT.md).
- **[Tham khảo thêm] Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (2019)** — T5, đại diện hướng encoder-decoder (có cross-attention, xem [04-Transformer.md](../04-Transformer/04-Transformer.md)), hữu ích để so sánh với lựa chọn decoder-only. Xem thêm ở [05-BERT-T5-GPT.md](../05-BERT-T5-GPT/05-BERT-T5-GPT.md).
- **[Nên đọc] Kaplan et al., *Scaling Laws for Neural Language Models* (2020)** và **Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022, Chinchilla)** — quan hệ giữa số tham số, dữ liệu train, và compute; paper Chinchilla chỉ ra nhiều model thế hệ trước (GPT-3) bị "under-trained" so với số tham số, thay đổi cách ngành thiết kế model từ đó.
- **[Tham khảo thêm] Touvron et al., *LLaMA: Open and Efficient Foundation Language Models* (2023)** — mở đầu làn sóng LLM mở hiệu quả, kiến trúc mà Qwen (dùng xuyên suốt sách này) cũng kế thừa nhiều lựa chọn thiết kế (RMSNorm, SwiGLU, RoPE).

## Chương 15.9 — RAG & Agent nâng cao

*Các biến thể mới hơn RAG/Agent cơ bản đã học ở chương 07, 09-11 — phù hợp khi hệ thống ở [14-Projects.md](../14-Projects/14-Projects.md) cần cải thiện thêm.*

- **[Nên đọc] Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (2023)** — model tự quyết định khi nào cần retrieve (không phải lúc nào cũng retrieve cố định như đã học ở [07-RAG.md](../07-RAG/07-RAG.md)), và tự đánh giá độ liên quan/độ trung thực của câu trả lời bằng các token đặc biệt được train riêng.
- **[Tham khảo thêm] Yan et al., *Corrective Retrieval Augmented Generation* (2024, CRAG)** — thêm một bước đánh giá chất lượng kết quả retrieve, tự động tìm kiếm bổ sung (ví dụ trên web) nếu context không đủ tốt, thay vì luôn tin tưởng top-k trả về.
- **[Nên đọc] Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023)** — model tự học khi nào nên chèn lời gọi tool vào text bằng self-supervised learning, một hướng khác với tool calling native (train sẵn) đã dùng ở phần "Tool Calling" trong [11-Agent.md](../11-Agent/11-Agent.md).
- **[Tham khảo thêm] Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (2023)** — mô phỏng một "thị trấn ảo" nhiều agent có memory, lập kế hoạch, và tương tác xã hội với nhau — ví dụ mở rộng thú vị cho ý tưởng multi-agent ở [11-Agent.md](../11-Agent/11-Agent.md).

## Chương 15.10 — Xem trước: Diffusion & Multimodal

*Chuẩn bị cho [17-Diffusion.md](../17-Diffusion/17-Diffusion.md) (chưa viết) — 3 paper nền tảng nếu muốn đọc trước.*

- **[Bắt buộc khi học ch.17] Ho et al., *Denoising Diffusion Probabilistic Models* (2020, DDPM)** — công thức hóa quá trình thêm nhiễu/khử nhiễu làm nền tảng cho mọi model diffusion hiện đại.
- **[Bắt buộc khi học ch.17] Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022)** — kiến trúc đứng sau Stable Diffusion: chạy quá trình diffusion trong không gian latent (đã nén bởi VAE) thay vì trực tiếp trên pixel, giảm chi phí tính toán hàng chục lần.
- **[Nên đọc khi học ch.17] Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (2021, CLIP)** — model nối ảnh và text vào chung 1 không gian embedding, là thành phần "hiểu prompt text" trong Stable Diffusion và nhiều hệ multimodal khác.

## Chương 15.11 — Cách đọc một paper AI/ML hiệu quả

Đọc tuyến tính từ đầu đến cuối thường lãng phí thời gian với paper kỹ thuật. Thứ tự thực dụng hơn:

1. **Abstract + Hình 1** — nắm ý tưởng cốt lõi trong 2 phút. Nếu Hình 1 không hiểu, đọc phần Introduction trước khi đi tiếp.
2. **Kết luận (Conclusion)** — thường tóm tắt lại đóng góp chính và hạn chế, giúp quyết định có cần đọc sâu phần method không.
3. **Phần Method** — chỉ đọc kỹ nếu cần **cài đặt lại** ý tưởng đó (giống các ví dụ code trong sách này). Nếu chỉ cần biết "nó làm gì", phần tóm tắt + hình vẽ là đủ.
4. **Experiments/Ablation** — quan trọng nhất là bảng ablation (bỏ từng thành phần xem kết quả giảm bao nhiêu) — cho biết phần nào của ý tưởng thực sự tạo ra khác biệt, phần nào chỉ là chi tiết kỹ thuật phụ.
5. **Related Work** — hữu ích để tìm thêm paper liên quan (backward reference), nhưng nên đọc sau cùng, không đọc trước khi hiểu paper chính.

Với paper toán nặng (như RoPE, InfoNCE), một mẹo hiệu quả: **tự code lại phần công thức cốt lõi** bằng vài dòng NumPy/PyTorch trước khi đọc phần chứng minh toán học đầy đủ — đây chính là cách tiếp cận các ví dụ trong sách này đã dùng (xem phần "RoPE" trong [04-Transformer.md](../04-Transformer/04-Transformer.md), phần "Contrastive Learning" trong [06-Embedding.md](../06-Embedding/06-Embedding.md)) — code chạy được thường làm rõ trực giác nhanh hơn đọc công thức suông.

## Chương 15.12 — Bảng tổng hợp theo dòng thời gian

Nhìn nhanh thứ tự xuất hiện giúp thấy được các ý tưởng "kế thừa" nhau như thế nào:

| Năm | Paper | Chủ đề |
|---|---|---|
| 2013 | Word2Vec | Embedding |
| 2015 | FaceNet (Triplet Loss) | Embedding |
| 2017 | Attention Is All You Need | Transformer |
| 2017 | Mixture-of-Experts (Sparsely-Gated) | Kiến trúc/Hiệu năng |
| 2018 | BERT | Transformer (encoder) |
| 2018 | GPT-1 | Transformer (decoder) |
| 2018 | InfoNCE (CPC) | Embedding |
| 2019 | T5 | Transformer (encoder-decoder) |
| 2020 | GPT-3 | Scale/Few-shot |
| 2020 | Scaling Laws | Lý thuyết scale |
| 2020 | RAG (Lewis et al.) | RAG |
| 2020 | DDPM | Diffusion |
| 2021 | RoPE | Transformer |
| 2021 | LoRA | Fine-tuning |
| 2021 | CLIP | Multimodal |
| 2021 | Switch Transformers | Kiến trúc/Hiệu năng |
| 2022 | InstructGPT (RLHF) | Alignment |
| 2022 | Chinchilla | Lý thuyết scale |
| 2022 | ReAct | Agent |
| 2022 | FlashAttention | Hiệu năng |
| 2022 | Constitutional AI | Alignment |
| 2022 | Latent Diffusion (Stable Diffusion) | Diffusion |
| 2023 | QLoRA | Fine-tuning |
| 2023 | GQA | Kiến trúc |
| 2023 | Tree of Thoughts | Agent |
| 2023 | Reflexion | Agent |
| 2023 | Toolformer | Agent |
| 2023 | Self-RAG | RAG |
| 2023 | DPO | Alignment |
| 2023 | PagedAttention (vLLM) | Serving |
| 2023 | LLaMA / Llama 2 | LLM mở |
| 2023 | AutoGen | Multi-Agent |
| 2024 | CRAG | RAG |
| 2024 | Model Context Protocol | Agent/Tool |

## Chương 15.13 — Tài liệu tham khảo

Toàn bộ paper trong chương này đã được trích dẫn đầy đủ (tác giả, tên, năm) trong các mục 2-10 ở trên — không lặp lại danh sách ở đây. Để tra cứu bản PDF/arXiv, tìm trực tiếp theo tên paper trên [arXiv.org](https://arxiv.org) hoặc [Google Scholar](https://scholar.google.com) — cuốn sách này chủ động không kèm link trực tiếp vì ID arXiv của một số paper có thể đổi phiên bản (v1, v2...) theo thời gian, tìm theo tên đảm bảo luôn ra đúng bản mới nhất.
