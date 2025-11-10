from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import time
from ..freshness import freshness_boost
from ..safety import filter_banned
from .onnx_encoder import OnnxEncoder

# Load once at startup (path configurable via env)
import os

ONNX_PATH = os.getenv("STUDENT_ONNX", "out/student.int8.onnx")
MODEL_NAME = os.getenv("STUDENT_NAME", "sentence-transformers/all-MiniLM-L6-v2")
enc = OnnxEncoder(ONNX_PATH, MODEL_NAME)

app = FastAPI(title="NeuroRanker Distill Service")


class RerankRequest(BaseModel):
    query: str
    texts: List[str]
    freshness_timestamps: Optional[List[Optional[int]]] = None  # epoch seconds per text
    use_freshness: bool = True
    banned_terms: Optional[List[str]] = None
    top_k: int = 10


class RerankResponse(BaseModel):
    indices: List[int]
    scores: List[float]
    latency_ms: float


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    t0 = time.time()
    qv = enc.embed([req.query])[0]
    pv = enc.embed(req.texts)
    sims = pv @ qv
    sims = sims / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv) + 1e-6)

    # Optional freshness boost (additive)
    if req.use_freshness and req.freshness_timestamps:
        boost = np.array(freshness_boost(req.freshness_timestamps))
        sims = sims + 0.1 * boost  # small prior

    # Optional safety filter → penalize banned
    if req.banned_terms:
        banned = filter_banned(req.texts, req.banned_terms)
        sims = sims - 1.0 * np.array(banned, dtype=np.float32)

    order = np.argsort(-sims)[: req.top_k]
    latency = (time.time() - t0) * 1000
    return RerankResponse(
        indices=order.tolist(), scores=sims[order].tolist(), latency_ms=float(latency)
    )
