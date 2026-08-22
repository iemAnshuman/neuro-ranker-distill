from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BiEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path: str, revision: str | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            attn_implementation="eager",
        )

    @staticmethod
    def pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1e-9)
        return F.normalize(summed / counts, p=2, dim=1)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool(outputs.last_hidden_state, attention_mask)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.encode(query_input_ids, query_attention_mask)
        positive = self.encode(positive_input_ids, positive_attention_mask)
        negative = self.encode(negative_input_ids, negative_attention_mask)
        positive_score = (query * positive).sum(dim=1)
        negative_score = (query * negative).sum(dim=1)
        return positive_score, negative_score, positive_score - negative_score


def load_tokenizer(model_name_or_path: str | Path, revision: str | None = None):
    return AutoTokenizer.from_pretrained(str(model_name_or_path), revision=revision, use_fast=True)


def encode_texts(
    model: BiEncoder,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            vectors = model.encode(
                tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            )
            encoded.append(vectors.cpu().numpy())
    if not encoded:
        hidden_size = int(model.encoder.config.hidden_size)
        return np.empty((0, hidden_size), dtype=np.float32)
    return np.concatenate(encoded, axis=0)
