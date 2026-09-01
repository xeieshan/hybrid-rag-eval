from fastapi.testclient import TestClient

from hybrid_rag.api import create_app


def test_health_search_and_ask(bundle, fake_completion):
    fn = fake_completion("Lines should be at most 79 characters [PEP 8].")
    with TestClient(create_app(bundle=bundle, model="m", completion_fn=fn)) as client:
        assert client.get("/health").json()["chunks"] == len(bundle.chunks)

