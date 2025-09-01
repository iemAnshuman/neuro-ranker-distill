# README.md
--teacher runs/teacher/best.pt \
--student sentence-transformers/all-MiniLM-L6-v2 \
--epochs 1 --lr 3e-5 --batch 64 --temp 3.0 \
--in_batch_negs 1 --hard_negs 3 \
--out_dir runs/student
```

### 6) Evaluate BM25 vs Teacher vs Student
```
python eval_rankers.py \
--data_dir data/msmarco-mini \
--bm25_index data/msmarco-mini/lucene-index \
--teacher runs/teacher/best.pt \
--student runs/student/best.pt \
--k 10 --cand_k 100
```
Emits CSV with NDCG@10, MRR@10, Recall@100 plus latency microbenchmarks.

### 7) Export student to ONNX + quantize INT8
```
python scripts/export_onnx.py --ckpt runs/student/best.pt --onnx out/student.onnx
python scripts/quantize_onnx.py --in_onnx out/student.onnx --out_onnx out/student.int8.onnx
```

### 8) Run the `/rerank` service (ONNX Runtime)
```
uvicorn src.neuro_ranker.service.app:app --host 0.0.0.0 --port 8000 --workers 2
```
Test:
```
curl -X POST http://localhost:8000/rerank \
-H 'Content-Type: application/json' \
-d '{"query":"what is a neural ranking model","texts":["...","..."],"freshness_timestamps":[1724899200,1726800000]}'
```

### 9) Latency/QPS bench (p50/p95)
```
python scripts/latency_bench.py --endpoint http://localhost:8000/rerank --qps 100 --seconds 20
```

---

## Goals to track
- Student P95 ≤ 50–80 ms for reranking 100 passages on CPU
- QPS ≥ 100 on a 4‑core laptop (offline bench)
- Student NDCG@10 within 90–95% of teacher and >7–10% over BM25

## Tips
- Start with the mini dataset; once working, switch to full MS MARCO Passage.
- Use `--hard_negs` from BM25 negatives for better student.
- Enable IPS (`--ips`) to handle position bias when training from clicks.

```

