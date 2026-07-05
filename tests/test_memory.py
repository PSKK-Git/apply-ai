import os
import pytest
from apply_ai.memory import LocalMemory, CogneeMemory, Evidence, get_memory
from apply_ai.schemas import load_resume


def _fixture_resume():
    # Fixed fixture so ranking assertions don't depend on the live data/resume.json
    # (which now holds the user's real résumé).
    from apply_ai.schemas import Bullet, Experience, Resume
    return Resume(name="T", skills=["python"], experiences=[
        Experience(id="exp-1", company="Acme", role="Eng", start="2020", bullets=[
            Bullet(id="ev-1", text="Built a FastAPI service cutting p95 latency",
                   skills=["python", "fastapi"]),
            Bullet(id="ev-2", text="Designed SQL pipelines", skills=["sql"]),
        ])])


def test_local_retrieval_ranks_by_overlap():
    mem = LocalMemory()
    mem.ingest(_fixture_resume())
    ev = mem.retrieve_evidence("fastapi latency service", k=3)
    assert ev and isinstance(ev[0], Evidence)
    assert ev[0].evidence_id == "ev-1"


def test_local_improve_is_noop_safe():
    mem = LocalMemory()
    mem.ingest(load_resume("data/resume.json"))
    mem.improve({"job_id": "AAI-0001", "accepted": ["ev-1"]})  # must not raise


def test_cognee_provenance_bridge_offline():
    # The [evidence_id] tag bridge maps retrieved chunks back to real bullets,
    # with no network — this is what guarantees provenance through Cognee.
    mem = CogneeMemory()
    mem.load_index(load_resume("data/resume.json"))
    chunks = ["noise [ev-2] designed sql pipelines", "top [ev-1] built a fastapi service"]
    ev = mem._evidence_from_texts(chunks)
    assert [e.evidence_id for e in ev] == ["ev-2", "ev-1"]
    assert all(isinstance(e, Evidence) for e in ev)


def test_get_memory_defaults_to_cognee(monkeypatch):
    monkeypatch.delenv("APPLY_AI_MEMORY", raising=False)
    assert isinstance(get_memory(), CogneeMemory)
    assert isinstance(get_memory(force_local=True), LocalMemory)
    monkeypatch.setenv("APPLY_AI_MEMORY", "local")
    assert isinstance(get_memory(), LocalMemory)


@pytest.mark.skipif(os.getenv("APPLY_AI_RUN_COGNEE") != "1",
                    reason="live cognee: set APPLY_AI_RUN_COGNEE=1 and provide keys in .env")
def test_cognee_live_roundtrip():
    mem = CogneeMemory(dataset="apply_ai_test")
    mem.ingest(load_resume("data/resume.json"))
    ev = mem.retrieve_evidence("fastapi latency service", k=5)
    assert any(e.evidence_id == "ev-1" for e in ev)
