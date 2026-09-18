from fastapi.testclient import TestClient

from hybrid_rag.api import create_app


def test_health_search_and_ask(bundle, fake_completion):
    fn = fake_completion("Lines should be at most 79 characters [PEP 8].")
    with TestClient(create_app(bundle=bundle, model="m", completion_fn=fn)) as client:
        assert client.get("/health").json()["chunks"] == len(bundle.chunks)

        r = client.post("/search", json={"query": "line length", "k": 2, "mode": "bm25"})
        assert r.status_code == 200
        assert r.json()["results"][0]["pep"] == 8

        r = client.post("/ask", json={"question": "How long can a line be?"})
        assert r.status_code == 200
        body = r.json()
        assert body["cited_peps"] == [8]
        assert body["sources"]


def test_validation_errors(bundle, fake_completion):
    with TestClient(create_app(bundle=bundle, completion_fn=fake_completion("x"))) as client:
        assert client.post("/search", json={"query": ""}).status_code == 422
        assert client.post("/search", json={"query": "x", "mode": "nope"}).status_code == 422


def test_llm_failure_returns_502(bundle):
    def boom(**_):
        raise RuntimeError("provider down")

    with TestClient(create_app(bundle=bundle, completion_fn=boom)) as client:
        r = client.post("/ask", json={"question": "anything"})
        assert r.status_code == 502

# Maintenance pass: 2026-09-18 12:55
