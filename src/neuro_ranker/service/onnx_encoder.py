from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class OnnxEncoder:
    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_path: str | Path,
        *,
        max_length: int = 256,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
        self.max_length = max_length

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return self.session.run(
            None,
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )[0].astype(np.float32)
