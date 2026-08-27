# Chunk size ablation

Same 45 questions, same LSA-256 dense model, chunk overlap = 20% of chunk size.
Produced with `make ablation`. One question is worth 0.022, so gaps under ~0.05 are within noise.

| Chunk size (words) | Retriever | Hit@1 | Hit@5 | Hit@10 | MRR |
|---:|---|---:|---:|---:|---:|
| 120 | BM25 | 0.489 | 0.756 | 0.844 | 0.633 |
| 120 | Hybrid RRF (BM25 + LSA) | 0.511 | 0.800 | 0.867 | 0.644 |
| 220 | BM25 | 0.556 | 0.844 | 0.889 | 0.677 |
| 220 | Hybrid RRF (BM25 + LSA) | 0.489 | 0.800 | 0.867 | 0.614 |
| 400 | BM25 | 0.556 | 0.800 | 0.889 | 0.677 |
| 400 | Hybrid RRF (BM25 + LSA) | 0.511 | 0.778 | 0.844 | 0.629 |

# Maintenance pass: 2026-08-27 14:42
