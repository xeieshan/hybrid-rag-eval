# hybrid-rag-eval

Retrieval-augmented question answering over all 744 Python Enhancement Proposals (PEPs), built to be measured. It combines BM25 and dense retrieval with reciprocal rank fusion, optionally reranks with an LLM, and answers with citations. A hand-written set of 45 questions, each labeled with the PEP that answers it, scores every change.

## Results so far

Document-level retrieval on 45 questions. The corpus is pinned to commit `545c670` of `python/peps`, giving 18,330 chunks of up to 220 words. Every number below comes from `make eval` and is checked into [`results/`](results/).

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 (from scratch) | 0.556 | 0.733 | **0.844** | **0.889** | **0.677** |
| Dense: LSA-256 (TF-IDF + SVD, local) | 0.511 | 0.667 | 0.733 | 0.800 | 0.604 |
| Hybrid RRF (BM25 + LSA) | 0.489 | 0.667 | 0.800 | 0.867 | 0.614 |

What this shows:

- **BM25 is a hard baseline here.** PEPs are technical and keyword-dense, so exact terms like `ParamSpec`, `zoneinfo` or `except*` do most of the work.
- **A weak dense model makes hybrid worse, not better.** With LSA as the dense half, fusion pulls the right PEP out of first place more often than it rescues a miss. For example, LSA tends to send queries containing "operator" to PEP 465 (the `@` operator), so "walrus operator" goes from PEP 572 at rank 3 under BM25 to PEP 465 at the top under hybrid. Hybrid search only pays off when the dense model gets *different* questions right.
