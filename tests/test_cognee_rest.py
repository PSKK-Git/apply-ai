import httpx
from apply_ai import cognee_rest
from apply_ai.cognee_rest import CogneeRestMemory
from apply_ai.memory import Evidence
from apply_ai.schemas import load_resume


class _Resp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def test_ingest_posts_remember_multipart(monkeypatch):
    calls = {}

    def fake_post(url, **kw):
        calls.update(url=url, data=kw.get("data"), files=kw.get("files"),
                     headers=kw.get("headers"))
        return _Resp({"ok": True})

    monkeypatch.setattr(httpx, "post", fake_post)
    mem = CogneeRestMemory(base_url="https://x", api_key="k", dataset="apply_ai_resume")
    mem.ingest(load_resume("data/resume.json"))
    assert calls["url"].endswith("/api/v1/remember")
    assert calls["data"]["datasetName"] == "apply_ai_resume"
    assert calls["headers"]["X-Api-Key"] == "k"
    assert "data" in calls["files"]
    assert "ev-1" in mem._index


def test_retrieve_parses_provenance_tags(monkeypatch):
    def fake_post(url, **kw):
        return _Resp([{"text": "work involves [ev-1] and later [ev-2]"}])

    monkeypatch.setattr(httpx, "post", fake_post)
    mem = CogneeRestMemory(base_url="https://x", api_key="k")
    mem.load_index(load_resume("data/resume.json"))
    ev = mem.retrieve_evidence("fastapi", k=5)
    assert [e.evidence_id for e in ev] == ["ev-1", "ev-2"]
    assert all(isinstance(e, Evidence) for e in ev)
