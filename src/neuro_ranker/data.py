from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from .config import config_digest

MSMARCO_TERMS_URL = "https://microsoft.github.io/msmarco/"
MSMARCO_DOWNLOAD_ROOT = "https://msmarco.z22.web.core.windows.net/msmarcoranking"
MSMARCO_SOURCES = {
    "train_triples": {
        "url": f"{MSMARCO_DOWNLOAD_ROOT}/triples.train.small.tar.gz",
        "content_length": "7930881353",
        "etag": '"0x8DBE166AC95195B"',
        "last_modified": "Thu, 09 Nov 2023 20:58:50 GMT",
    },
    "dev_candidates": {
        "url": f"{MSMARCO_DOWNLOAD_ROOT}/top1000.dev.tar.gz",
        "content_length": "687414398",
        "etag": '"0x8DBE166A92A3437"',
        "last_modified": "Thu, 09 Nov 2023 20:58:44 GMT",
    },
    "dev_qrels": {
        "url": f"{MSMARCO_DOWNLOAD_ROOT}/qrels.dev.tsv",
        "content_length": "1201626",
        "content_md5_base64": "kVfMrqqCJ/kXIrpXcHh7Fg==",
        "etag": '"0x8DBE166ABFFF421"',
        "last_modified": "Thu, 09 Nov 2023 20:58:49 GMT",
    },
}


class TermsNotAcceptedError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: str | Path) -> int:
    with Path(path).open("rb") as handle:
        return sum(1 for _ in handle)


def _request(source_name: str) -> BinaryIO:
    source = MSMARCO_SOURCES[source_name]
    url = source["url"]
    request = urllib.request.Request(url, headers={"User-Agent": "neuro-ranker-distill/0.2"})
    response = urllib.request.urlopen(request, timeout=120)  # noqa: S310
    expected_headers = {
        "Content-Length": source["content_length"],
        "ETag": source["etag"],
        "Last-Modified": source["last_modified"],
    }
    if "content_md5_base64" in source:
        expected_headers["Content-MD5"] = source["content_md5_base64"]
    for header, expected in expected_headers.items():
        actual = response.headers.get(header)
        if actual != expected:
            response.close()
            raise RuntimeError(
                f"upstream {source_name} changed: {header} expected {expected!r}, got {actual!r}"
            )
    return response


def _iter_tar_lines(source_name: str) -> Iterator[str]:
    with _request(source_name) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                for raw_line in extracted:
                    yield raw_line.decode("utf-8").rstrip("\n")
                return
    raise RuntimeError(f"archive had no regular file: {MSMARCO_SOURCES[source_name]['url']}")


def _read_qrels(source_name: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with _request(source_name) as response:
        for raw_line in response:
            fields = raw_line.decode("utf-8").strip().split()
            if len(fields) != 4:
                continue
            qid, _iteration, pid, relevance = fields
            qrels.setdefault(qid, {})[pid] = int(relevance)
    return qrels


def _write_jsonl(path: Path, records: Iterator[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _fixture_triplets() -> Iterator[dict[str, Any]]:
    examples = [
        ("capital of france", "Paris is the capital of France.", "Berlin is in Germany."),
        ("largest ocean", "The Pacific Ocean is the largest ocean.", "The Nile is a river."),
        ("speed of light", "Light travels at about 299792458 m/s.", "Sound is slower."),
        ("python list append", "list.append adds one item to a list.", "A tuple is immutable."),
        ("water freezing point", "Water freezes at 0 degrees Celsius.", "Iron melts hot."),
        ("author of hamlet", "William Shakespeare wrote Hamlet.", "Dickens wrote Oliver Twist."),
        (
            "gpu tensor cores",
            "Tensor Cores accelerate matrix operations.",
            "A CPU cache stores data.",
        ),
        (
            "meaning of bm25",
            "BM25 is a probabilistic ranking function.",
            "BLEU scores translation.",
        ),
    ]
    for query, positive, negative in examples:
        yield {"query": query, "positive": positive, "negative": negative}


def _fixture_candidates() -> Iterator[dict[str, Any]]:
    groups = {
        "q1": (
            "capital of france",
            [
                ("p1", "Berlin is in Germany.", 0),
                ("p2", "Paris is the capital of France.", 1),
                ("p3", "Madrid is in Spain.", 0),
                ("p4", "Rome is in Italy.", 0),
            ],
        ),
        "q2": (
            "largest ocean",
            [
                ("p5", "The Pacific Ocean is the largest ocean.", 1),
                ("p6", "The Nile is a river.", 0),
                ("p7", "Everest is a mountain.", 0),
                ("p8", "Mercury is a planet.", 0),
            ],
        ),
    }
    for qid, (query, candidates) in groups.items():
        for rank, (pid, passage, relevance) in enumerate(candidates, start=1):
            yield {
                "qid": qid,
                "pid": pid,
                "query": query,
                "passage": passage,
                "baseline_rank": rank,
                "relevance": relevance,
            }


def _msmarco_triplets(limit: int) -> Iterator[dict[str, Any]]:
    produced = 0
    for line in _iter_tar_lines("train_triples"):
        fields = line.split("\t", maxsplit=2)
        if len(fields) != 3:
            continue
        query, positive, negative = fields
        yield {"query": query, "positive": positive, "negative": negative}
        produced += 1
        if produced >= limit:
            return


def _msmarco_candidates(query_limit: int, candidates_per_query: int) -> Iterator[dict[str, Any]]:
    qrels = _read_qrels("dev_qrels")
    selected: list[str] = []
    per_query: dict[str, int] = {}
    for line in _iter_tar_lines("dev_candidates"):
        fields = line.split("\t", maxsplit=3)
        if len(fields) != 4:
            continue
        qid, pid, query, passage = fields
        if qid not in qrels:
            continue
        if qid not in per_query:
            if len(selected) >= query_limit:
                return
            selected.append(qid)
            per_query[qid] = 0
        if per_query[qid] >= candidates_per_query:
            continue
        per_query[qid] += 1
        yield {
            "qid": qid,
            "pid": pid,
            "query": query,
            "passage": passage,
            "baseline_rank": per_query[qid],
            "relevance": qrels[qid].get(pid, 0),
        }


def _file_record(path: Path, records: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_prepared_counts(
    config: dict[str, Any], *, train_records: int, eval_records: int
) -> None:
    data_config = config["data"]
    expected = {
        "train_triplets": int(data_config["train_records"]),
        "eval_candidates": int(data_config["eval_queries"])
        * int(data_config["candidates_per_query"]),
    }
    actual = {"train_triplets": train_records, "eval_candidates": eval_records}
    for name, expected_count in expected.items():
        if actual[name] != expected_count:
            raise RuntimeError(
                f"{name} produced {actual[name]} records; expected {expected_count}"
            )


def prepare_dataset(
    config: dict[str, Any], output_dir: str | Path, accept_msmarco_terms: bool = False
) -> dict[str, Any]:
    profile = config["profile"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = verify_dataset(output)
        if manifest["config_sha256"] != config_digest(config):
            raise ValueError(f"existing dataset uses a different config: {output}")
        return manifest

    train_path = output / "train_triplets.jsonl"
    eval_path = output / "eval_candidates.jsonl"
    if train_path.exists() or eval_path.exists():
        raise FileExistsError(f"refusing to overwrite incomplete dataset: {output}")

    if profile == "fixture":
        train_records = _write_jsonl(train_path, _fixture_triplets())
        eval_records = _write_jsonl(eval_path, _fixture_candidates())
        sources: dict[str, str] = {"fixture": "generated locally; no external dataset"}
        terms_accepted = False
    else:
        if not accept_msmarco_terms:
            raise TermsNotAcceptedError(
                f"review {MSMARCO_TERMS_URL} and pass --accept-msmarco-terms"
            )
        data_config = config["data"]
        train_records = _write_jsonl(
            train_path, _msmarco_triplets(int(data_config["train_records"]))
        )
        eval_records = _write_jsonl(
            eval_path,
            _msmarco_candidates(
                int(data_config["eval_queries"]),
                int(data_config["candidates_per_query"]),
            ),
        )
        sources = MSMARCO_SOURCES.copy()
        terms_accepted = True

    _validate_prepared_counts(
        config, train_records=train_records, eval_records=eval_records
    )
    manifest = {
        "schema_version": 1,
        "profile": profile,
        "config_sha256": config_digest(config),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "msmarco_terms": MSMARCO_TERMS_URL,
        "msmarco_terms_accepted": terms_accepted,
        "sources": sources,
        "limits": config["data"],
        "files": {
            "train_triplets": _file_record(train_path, train_records),
            "eval_candidates": _file_record(eval_path, eval_records),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_dataset(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("dataset manifest schema_version must be 1")
    for record in manifest["files"].values():
        path = output / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"size mismatch: {path}")
        if count_lines(path) != record["records"]:
            raise ValueError(f"record-count mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    return manifest
