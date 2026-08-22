from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Protocol

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..freshness import freshness_boost
from ..safety import filter_banned


class Encoder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    texts: list[str] = Field(min_length=1)
    freshness_timestamps: list[int | None] | None = None
    use_freshness: bool = False
    banned_terms: list[str] | None = None
    top_k: int = Field(default=10, ge=1)


class RerankResponse(BaseModel):
    indices: list[int]
    scores: list[float]
    latency_ms: float


def create_app(encoder: Encoder | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if encoder is not None:
            app.state.encoder = encoder
        else:
            model_path = os.environ.get("NEURORANKER_ONNX")
            tokenizer_path = os.environ.get("NEURORANKER_TOKENIZER")
            if not model_path or not tokenizer_path:
                raise RuntimeError("NEURORANKER_ONNX and NEURORANKER_TOKENIZER must be configured")
            from .onnx_encoder import OnnxEncoder

            app.state.encoder = OnnxEncoder(model_path, tokenizer_path)
        yield

    app = FastAPI(title="NeuroRanker", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": type(app.state.encoder).__name__}

    @app.post("/rerank", response_model=RerankResponse)
    def rerank(request: RerankRequest) -> RerankResponse:
        if request.freshness_timestamps is not None and len(request.freshness_timestamps) != len(
            request.texts
        ):
            raise HTTPException(
                status_code=400,
                detail="freshness_timestamps must have the same length as texts",
            )
        started = time.perf_counter()
        query_vector = app.state.encoder.embed([request.query])[0]
        passage_vectors = app.state.encoder.embed(request.texts)
        scores = passage_vectors @ query_vector
        if request.use_freshness and request.freshness_timestamps is not None:
            scores = scores + 0.1 * np.asarray(
                freshness_boost(request.freshness_timestamps), dtype=np.float32
            )
        if request.banned_terms:
            scores = scores - np.asarray(
                filter_banned(request.texts, request.banned_terms), dtype=np.float32
            )
        top_k = min(request.top_k, len(request.texts))
        order = np.argsort(-scores, kind="stable")[:top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return RerankResponse(
            indices=order.tolist(),
            scores=scores[order].astype(float).tolist(),
            latency_ms=elapsed_ms,
        )

    return app
