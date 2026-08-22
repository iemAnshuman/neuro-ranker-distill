import json

import pytest

from neuro_ranker import data as data_module
from neuro_ranker.config import load_config
from neuro_ranker.data import (
    TermsNotAcceptedError,
    prepare_dataset,
    verify_dataset,
)


def test_fixture_is_reproducible_and_verifiable(tmp_path):
    config = load_config("configs/fixture.json")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = prepare_dataset(config, first_dir)
    second = prepare_dataset(config, second_dir)

    assert first["files"]["train_triplets"]["sha256"] == second["files"]["train_triplets"]["sha256"]
    assert (
        first["files"]["eval_candidates"]["sha256"] == second["files"]["eval_candidates"]["sha256"]
    )
    assert verify_dataset(first_dir)["profile"] == "fixture"


def test_manifest_detects_tampering(tmp_path):
    config = load_config("configs/fixture.json")
    output = tmp_path / "fixture"
    prepare_dataset(config, output)
    with (output / "train_triplets.jsonl").open("a") as handle:
        handle.write(json.dumps({"query": "tampered"}) + "\n")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_dataset(output)


def test_msmarco_requires_explicit_terms_acceptance(tmp_path):
    config = load_config("configs/smoke.json")
    with pytest.raises(TermsNotAcceptedError):
        prepare_dataset(config, tmp_path / "smoke")


def test_source_identity_mismatch_closes_response(monkeypatch):
    class Response:
        headers = {"Content-Length": "0"}
        closed = False

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(data_module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    with pytest.raises(RuntimeError, match="upstream dev_qrels changed: Content-Length"):
        data_module._request("dev_qrels")
    assert response.closed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("train_records", 9, "train_triplets produced 8 records; expected 9"),
        ("eval_queries", 3, "eval_candidates produced 8 records; expected 12"),
    ],
)
def test_preparation_rejects_incomplete_record_counts(tmp_path, field, value, message):
    config = load_config("configs/fixture.json")
    config["data"][field] = value
    with pytest.raises(RuntimeError, match=message):
        prepare_dataset(config, tmp_path / field)
