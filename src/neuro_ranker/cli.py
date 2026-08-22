from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuro-ranker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare a fixture or MS MARCO profile")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--accept-msmarco-terms", action="store_true")

    verify = subparsers.add_parser("verify-data", help="verify a prepared data manifest")
    verify.add_argument("--data-dir", required=True)

    teacher = subparsers.add_parser("score-teacher", help="precompute teacher margins")
    teacher.add_argument("--config", required=True)
    teacher.add_argument("--data-dir", required=True)
    teacher.add_argument("--output", required=True)
    teacher.add_argument("--device", default="auto")

    student = subparsers.add_parser("train-student", help="train the distilled bi-encoder")
    student.add_argument("--config", required=True)
    student.add_argument("--input", required=True)
    student.add_argument("--output", required=True)
    student.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate", help="evaluate baseline and student")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--data-dir", required=True)
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")

    export = subparsers.add_parser("export", help="export and validate FP32/INT8 ONNX")
    export.add_argument("--config", required=True)
    export.add_argument("--model-dir", required=True)
    export.add_argument("--output", required=True)

    doctor = subparsers.add_parser("doctor", help="capture software and hardware provenance")
    doctor.add_argument("--repo-root", default=".")
    doctor.add_argument("--output")

    run = subparsers.add_parser("run", help="run preparation through export")
    run.add_argument("--config", required=True)
    run.add_argument("--work-dir", required=True)
    run.add_argument("--repo-root", default=".")
    run.add_argument("--device", default="auto")
    run.add_argument("--accept-msmarco-terms", action="store_true")

    serve = subparsers.add_parser("serve", help="serve a validated ONNX export")
    serve.add_argument("--model", required=True)
    serve.add_argument("--tokenizer", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_dataset

        _print_json(
            prepare_dataset(load_config(args.config), args.output, args.accept_msmarco_terms)
        )
    elif args.command == "verify-data":
        from .data import verify_dataset

        _print_json(verify_dataset(args.data_dir))
    elif args.command == "score-teacher":
        from .teacher import score_teacher_file

        config = load_config(args.config)
        _print_json(
            score_teacher_file(
                Path(args.data_dir) / "train_triplets.jsonl",
                args.output,
                model_name=config["teacher"]["model"],
                revision=config["teacher"]["revision"],
                batch_size=int(config["teacher"]["batch_size"]),
                max_length=int(config["teacher"]["max_length"]),
                seed=int(config["seed"]),
                device=args.device,
            )
        )
    elif args.command == "train-student":
        from .training import train_student

        config = load_config(args.config)
        _print_json(
            train_student(
                args.input,
                args.output,
                model_name=config["student"]["model"],
                revision=config["student"]["revision"],
                epochs=int(config["student"]["epochs"]),
                batch_size=int(config["student"]["batch_size"]),
                learning_rate=float(config["student"]["learning_rate"]),
                max_length=int(config["student"]["max_length"]),
                seed=int(config["seed"]),
                device=args.device,
            )
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate_student

        config = load_config(args.config)
        _print_json(
            evaluate_student(
                Path(args.data_dir) / "eval_candidates.jsonl",
                args.model_dir,
                args.output,
                batch_size=int(config["evaluation"]["batch_size"]),
                max_length=int(config["student"]["max_length"]),
                k=int(config["evaluation"]["k"]),
                device=args.device,
            )
        )
    elif args.command == "export":
        from .exporting import export_student

        config = load_config(args.config)
        _print_json(
            export_student(
                args.model_dir,
                args.output,
                max_length=int(config["student"]["max_length"]),
            )
        )
    elif args.command == "doctor":
        from .provenance import collect_provenance, write_provenance

        report = (
            write_provenance(args.repo_root, args.output)
            if args.output
            else collect_provenance(args.repo_root)
        )
        _print_json(report)
    elif args.command == "run":
        from .runner import run_artifact

        _print_json(
            run_artifact(
                args.config,
                args.work_dir,
                accept_msmarco_terms=args.accept_msmarco_terms,
                repo_root=args.repo_root,
                device=args.device,
            )
        )
    elif args.command == "serve":
        import os

        import uvicorn

        from .service.app import create_app

        os.environ["NEURORANKER_ONNX"] = args.model
        os.environ["NEURORANKER_TOKENIZER"] = args.tokenizer
        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
