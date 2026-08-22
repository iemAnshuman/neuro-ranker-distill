from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import sha256_file
from .model import BiEncoder, load_tokenizer
from .utils import set_seed


class ScoredTriplets(Dataset):
    def __init__(self, path: str | Path):
        with Path(path).open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle]
        if not self.records:
            raise ValueError("teacher-scored training data is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_student(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    revision: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    device: str = "auto",
) -> dict[str, Any]:
    source = Path(input_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite model directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed)
    torch_device = _resolve_device(device)
    tokenizer = load_tokenizer(model_name, revision)
    model = BiEncoder(model_name, revision).to(torch_device)
    dataset = ScoredTriplets(source)

    def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "teacher_margin": torch.tensor(
                [float(record["teacher_margin"]) for record in records], dtype=torch.float32
            )
        }
        for name, field in (
            ("query", "query"),
            ("positive", "positive"),
            ("negative", "negative"),
        ):
            tokens = tokenizer(
                [record[field] for record in records],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            result[f"{name}_input_ids"] = tokens["input_ids"]
            result[f"{name}_attention_mask"] = tokens["attention_mask"]
        return result

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(loader, desc=f"student epoch {epoch + 1}", unit="batch")
        for batch in progress:
            batch = {
                key: value.to(torch_device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            _positive, _negative, student_margin = model(
                batch["query_input_ids"],
                batch["query_attention_mask"],
                batch["positive_input_ids"],
                batch["positive_attention_mask"],
                batch["negative_input_ids"],
                batch["negative_attention_mask"],
            )
            loss = F.mse_loss(student_margin, batch["teacher_margin"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.5f}")
        epoch_losses.append(total_loss / max(1, len(loader)))

    model.encoder.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "mean-pooled-bi-encoder",
        "base_model": model_name,
        "base_revision": revision,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "training_records": len(dataset),
        "training_data_sha256": sha256_file(source),
        "epoch_losses": epoch_losses,
        "device": str(torch_device),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
