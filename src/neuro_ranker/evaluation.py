from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import sha256_file
from .metrics import aggregate_ranking_metrics
from .model import BiEncoder, encode_texts, load_tokenizer


def iter_candidate_groups(path: str | Path) -> Iterator[list[dict[str, Any]]]:
    current_qid: str | None = None
    group: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if current_qid is not None and record["qid"] != current_qid:
                yield group
                group = []
            current_qid = record["qid"]
            group.append(record)
    if group:
        yield group


def evaluate_student(
    candidates_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int,
    max_length: int,
    k: int,
    device: str = "auto",
) -> dict[str, Any]:
    candidates = Path(candidates_path)
    model_path = Path(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else device
        if device != "auto"
        else "cpu"
    )
    tokenizer = load_tokenizer(model_path)
    model = BiEncoder(str(model_path)).to(torch_device)
    model.eval()

    baseline_rankings: list[list[int]] = []
    student_rankings: list[list[int]] = []
    query_count = 0
    candidate_count = 0
    started = time.perf_counter()
    run_path = output / "student.run.tsv"
    with run_path.open("w", encoding="utf-8", newline="\n") as run_handle:
        for group in iter_candidate_groups(candidates):
            query_count += 1
            candidate_count += len(group)
            query_vector = encode_texts(
                model,
                tokenizer,
                [group[0]["query"]],
                device=torch_device,
                batch_size=1,
                max_length=max_length,
            )[0]
            passage_vectors = encode_texts(
                model,
                tokenizer,
                [record["passage"] for record in group],
                device=torch_device,
                batch_size=batch_size,
                max_length=max_length,
            )
            scores = passage_vectors @ query_vector
            order = np.argsort(-scores, kind="stable")
            baseline = sorted(group, key=lambda record: int(record["baseline_rank"]))
            baseline_rankings.append([int(record["relevance"]) for record in baseline])
            student_rankings.append([int(group[index]["relevance"]) for index in order])
            for rank, index in enumerate(order, start=1):
                record = group[int(index)]
                run_handle.write(
                    f"{record['qid']}\tQ0\t{record['pid']}\t{rank}\t"
                    f"{float(scores[index]):.8f}\tneuro-ranker\n"
                )

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "evaluation_scope": "reranking the candidates present in eval_candidates.jsonl",
        "k": k,
        "queries": query_count,
        "candidates": candidate_count,
        "candidate_data_sha256": sha256_file(candidates),
        "model_artifact_sha256": sha256_file(model_path / "artifact.json"),
        "baseline": aggregate_ranking_metrics(baseline_rankings, k),
        "student": aggregate_ranking_metrics(student_rankings, k),
        "device": str(torch_device),
        "elapsed_seconds": elapsed,
        "queries_per_second": query_count / elapsed if elapsed else None,
        "run_file": run_path.name,
        "run_sha256": sha256_file(run_path),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
