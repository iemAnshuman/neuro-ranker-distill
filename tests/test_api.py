from fastapi.testclient import TestClient
from ranker_service.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    # Adjust this based on what your root endpoint returns, if anything.
    # If you don't have a root endpoint, test /health or similar.
