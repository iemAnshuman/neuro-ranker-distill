from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import count_lines, sha256_file
from .utils import set_seed


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def score_teacher_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    revision: str,
    batch_size: int,
    max_length: int,
    seed: int,
    device: str = "auto",
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)

    set_seed(seed)
    torch_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision).to(
        torch_device
    )
    model.eval()

    total = count_lines(source)
    pending: list[dict[str, str]] = []
    written = 0

    def flush(handle) -> int:
        if not pending:
            return 0
        pairs: list[tuple[str, str]] = []
        for record in pending:
            pairs.append((record["query"], record["positive"]))
            pairs.append((record["query"], record["negative"]))
        queries = [query for query, _passage in pairs]
        passages = [passage for _query, passage in pairs]
        tokens = tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(torch_device) for key, value in tokens.items()}
        with torch.inference_mode():
            scores = model(**inputs).logits.reshape(-1).float().cpu().tolist()
        for index, record in enumerate(pending):
            teacher_positive = float(scores[index * 2])
            teacher_negative = float(scores[index * 2 + 1])
            scored = {
                **record,
                "teacher_positive": teacher_positive,
                "teacher_negative": teacher_negative,
                "teacher_margin": teacher_positive - teacher_negative,
            }
            handle.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")
        count = len(pending)
        pending.clear()
        return count

    with source.open(encoding="utf-8") as input_handle:
        with destination.open("w", encoding="utf-8", newline="\n") as output_handle:
            progress = tqdm(total=total, desc="teacher scoring", unit="triplet")
            for line in input_handle:
                pending.append(json.loads(line))
                if len(pending) >= batch_size:
                    count = flush(output_handle)
                    written += count
                    progress.update(count)
            count = flush(output_handle)
            written += count
            progress.update(count)
            progress.close()

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "revision": revision,
        "seed": seed,
        "device": str(torch_device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "input": {"path": str(source), "sha256": sha256_file(source), "records": total},
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "records": written,
        },
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
