import argparse, time
from tqdm import tqdm  # <-- MODIFICATION 1: Added this import
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.metrics import ndcg_at_k, mrr_at_k, recall_at_k
from src.neuro_ranker.bm25 import BM25
from src.neuro_ranker.trainer_teacher import TeacherTrainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--bm25_index", required=True)
p.add_argument("--teacher", required=True)
p.add_argument("--student", required=True)
p.add_argument("--k", type=int, default=10)
p.add_argument("--cand_k", type=int, default=100)
args = p.parse_args()

mini = MSMini(args.data_dir)
qmap = dict(mini.qs)
bm25 = BM25(args.bm25_index)

# Load teacher
ckpt_t = torch.load(args.teacher, map_location="cpu")
T_tok = AutoTokenizer.from_pretrained(
    ckpt_t.get("name", "microsoft/MiniLM-L12-H384-uncased")
)
T = AutoModelForSequenceClassification.from_pretrained(
    ckpt_t.get("name", "microsoft/MiniLM-L12-H384-uncased"), num_labels=1
)
T.load_state_dict(ckpt_t["model"])
T.eval()

# Load student
ckpt_s = torch.load(args.student, map_location="cpu")
from src.neuro_ranker.trainer_student import BiEncoder

S = BiEncoder("sentence-transformers/all-MiniLM-L6-v2")
S.load_state_dict(ckpt_s["model"])
S.eval()
S_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

ndcgs_b = []
mrrs_b = []
rec_b = []
ndcgs_t = []
mrrs_t = []
ndcgs_s = []
mrrs_s = []

# --- MODIFICATION 2: Wrapped mini.qs with tqdm ---
for qid, qtext in tqdm(mini.qs, desc="Evaluating Rankers"):
    hits = bm25.topk(qtext, args.cand_k)
    labels = [1 if (qid, pid) in mini.qrels else 0 for pid, _, _ in hits]

    # Teacher rerank latency
    s = time.time()
    scored_t = []
    for pid, score, raw in hits:
        enc = T_tok(
            qtext,
            raw,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logit = (
                T(**{k: enc[k] for k in ["input_ids", "attention_mask"]})
                .logits[0, 0]
                .item()
            )
        scored_t.append((pid, logit, raw))
    tlat = (time.time() - s) * 1000

    # Student rerank latency (batch)
    s = time.time()
    # Encode query once
    q_enc = S_tok(
        qtext, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        q_vec = S.encode(q_enc["input_ids"], q_enc["attention_mask"])

    # Batch encode all passages
    passages = [raw for _, _, raw in hits]
    p_enc = S_tok(
        passages, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        p_vecs = S.encode(p_enc["input_ids"], p_enc["attention_mask"])

    # Normalize vectors and compute all scores at once via matrix multiplication
    q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=1)
    p_vecs = torch.nn.functional.normalize(p_vecs, p=2, dim=1)
    sim_scores = (q_vec @ p_vecs.T).squeeze(0).cpu().numpy()

    scored_s = [(hits[i][0], sim_scores[i], hits[i][2]) for i in range(len(hits))]
    slat = (time.time() - s) * 1000

    # Compute ranking metrics
    by_bm25 = list(range(len(hits)))  # original bm25 order
    by_t = sorted(range(len(hits)), key=lambda i: scored_t[i][1], reverse=True)
    by_s = sorted(range(len(hits)), key=lambda i: scored_s[i][1], reverse=True)

    lab = labels
    ndcgs_b.append(ndcg_at_k([lab[i] for i in by_bm25], args.k))
    mrrs_b.append(mrr_at_k([lab[i] for i in by_bm25], args.k))

    ndcgs_t.append(ndcg_at_k([lab[i] for i in by_t], args.k))
    mrrs_t.append(mrr_at_k([lab[i] for i in by_t], args.k))

    ndcgs_s.append(ndcg_at_k([lab[i] for i in by_s], args.k))
    mrrs_s.append(mrr_at_k([lab[i] for i in by_s], args.k))

print(
    f"BM25: NDCG@{args.k}={sum(ndcgs_b)/len(ndcgs_b):.3f} MRR@{args.k}={sum(mrrs_b)/len(mrrs_b):.3f}"
)
print(
    f"Teach: NDCG@{args.k}={sum(ndcgs_t)/len(ndcgs_t):.3f} MRR@{args.k}={sum(mrrs_t)/len(mrrs_t):.3f}"
)
print(
    f"Stud.: NDCG@{args.k}={sum(ndcgs_s)/len(ndcgs_s):.3f} MRR@{args.k}={sum(mrrs_s)/len(mrrs_s):.3f}"
)