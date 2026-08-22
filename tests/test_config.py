import json

import pytest

from neuro_ranker.config import config_digest, load_config


def test_config_digest_ignores_mapping_order(tmp_path):
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}
    assert config_digest(left) == config_digest(right)


def test_invalid_profile_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "unknown",
                "seed": 1,
                "data": {},
                "teacher": {},
                "student": {},
                "evaluation": {},
            }
        )
    )
    with pytest.raises(ValueError, match="profile"):
        load_config(path)
