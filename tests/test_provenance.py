from neuro_ranker.provenance import collect_provenance


def test_provenance_records_embedded_source_revision(monkeypatch, tmp_path):
    revision = "a" * 40
    monkeypatch.setenv("NEURORANKER_SOURCE_REVISION", revision)

    report = collect_provenance(tmp_path)

    assert report["git"]["embedded_revision"] == revision
    assert report["environment"]["NEURORANKER_SOURCE_REVISION"] == revision
