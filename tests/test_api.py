import numpy as np
from fastapi.testclient import TestClient

from neuro_ranker.service.app import create_app


class FakeEncoder:
    vectors = {
        "query": [1.0, 0.0],
        "irrelevant": [0.0, 1.0],
        "relevant": [1.0, 0.0],
    }

    def embed(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_health_and_rerank_are_model_independent():
    with TestClient(create_app(FakeEncoder())) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model": "FakeEncoder"}

        response = client.post(
            "/rerank",
            json={
                "query": "query",
                "texts": ["irrelevant", "relevant"],
                "top_k": 2,
            },
        )
        assert response.status_code == 200
        assert response.json()["indices"] == [1, 0]


def test_timestamp_length_is_validated():
    with TestClient(create_app(FakeEncoder())) as client:
        response = client.post(
            "/rerank",
            json={
                "query": "query",
                "texts": ["relevant", "irrelevant"],
                "freshness_timestamps": [1],
            },
        )
        assert response.status_code == 400


def test_banned_passages_are_excluded_instead_of_penalized():
    with TestClient(create_app(FakeEncoder())) as client:
        response = client.post(
            "/rerank",
            json={
                "query": "query",
                "texts": ["relevant", "irrelevant"],
                "banned_terms": ["relevant"],
                "top_k": 2,
            },
        )
        assert response.status_code == 200
        assert response.json()["indices"] == [1]


def test_all_banned_passages_return_an_empty_ranking():
    with TestClient(create_app(FakeEncoder())) as client:
        response = client.post(
            "/rerank",
            json={
                "query": "query",
                "texts": ["relevant"],
                "banned_terms": ["relevant"],
            },
        )
        assert response.status_code == 200
        assert response.json()["indices"] == []
        assert response.json()["scores"] == []
