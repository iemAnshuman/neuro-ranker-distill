from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config
from .data import prepare_dataset
from .evaluation import evaluate_student
from .exporting import export_student
from .provenance import write_provenance
from .teacher import score_teacher_file
from .training import train_student


def run_artifact(
    config_path: str | Path,
    work_dir: str | Path,
    *,
    accept_msmarco_terms: bool,
    repo_root: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    config = load_config(config_path)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    data_dir = work / "data"
    scores_path = work / "teacher_scores.jsonl"
    student_dir = work / "student"
    evaluation_dir = work / "evaluation"
    export_dir = work / "export"

    data_manifest = prepare_dataset(config, data_dir, accept_msmarco_terms)
    teacher_manifest = score_teacher_file(
        data_dir / "train_triplets.jsonl",
        scores_path,
        model_name=config["teacher"]["model"],
        revision=config["teacher"]["revision"],
        batch_size=int(config["teacher"]["batch_size"]),
        max_length=int(config["teacher"]["max_length"]),
        seed=int(config["seed"]),
        device=device,
    )
    student_artifact = train_student(
        scores_path,
        student_dir,
        model_name=config["student"]["model"],
        revision=config["student"]["revision"],
        epochs=int(config["student"]["epochs"]),
        batch_size=int(config["student"]["batch_size"]),
        learning_rate=float(config["student"]["learning_rate"]),
        max_length=int(config["student"]["max_length"]),
        seed=int(config["seed"]),
        device=device,
    )
    evaluation = evaluate_student(
        data_dir / "eval_candidates.jsonl",
        student_dir,
        evaluation_dir,
        batch_size=int(config["evaluation"]["batch_size"]),
        max_length=int(config["student"]["max_length"]),
        k=int(config["evaluation"]["k"]),
        device=device,
    )
    export = export_student(
        student_dir, export_dir, max_length=int(config["student"]["max_length"])
    )
    provenance = write_provenance(repo_root, work / "provenance.json")
    run_manifest = {
        "schema_version": 1,
        "profile": config["profile"],
        "data": data_manifest,
        "teacher": teacher_manifest,
        "student": student_artifact,
        "evaluation": evaluation,
        "export": export,
        "provenance": provenance,
    }
    (work / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    return run_manifest
