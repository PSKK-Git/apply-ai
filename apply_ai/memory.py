"""Memory: Cognee is the knowledge spine.

`CogneeMemory` ingests the real experience corpus with `remember()`, grounds
tailoring with `search(SearchType.CHUNKS)`, and folds signals back with
`improve()`. Each bullet is stored tagged as `[evidence_id] ...` so retrieval can
recover provenance — the anchor for the no-fabrication guarantee.

`LocalMemory` is an explicit offline double for tests / no-key demos; it is never
the silent default. `get_memory()` returns the Cognee spine unless
`APPLY_AI_MEMORY=local` (or `force_local=True`).
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import os
import re
from pydantic import BaseModel, Field
from .schemas import Resume
from .cognee_config import configure

_TAG = re.compile(r"\[([A-Za-z0-9_-]+)\]")


class Evidence(BaseModel):
    evidence_id: str
    text: str
    skills: list[str] = Field(default_factory=list)
    source: str = ""


def _all_bullets(resume: Resume) -> list[Evidence]:
    out: list[Evidence] = []
    for e in resume.experiences:
        for b in e.bullets:
            out.append(Evidence(evidence_id=b.id, text=b.text, skills=b.skills,
                                source=f"{e.role} @ {e.company}"))
    for p in resume.projects:
        for b in p.bullets:
            out.append(Evidence(evidence_id=b.id, text=b.text, skills=b.skills,
                                source=f"project:{p.name}"))
    return out


def _run(coro):
    """Run an async cognee coroutine from sync code, even inside a live loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


class CogneeMemory:
    """The knowledge spine, backed by Cognee's remember/search/improve."""

    def __init__(self, dataset: str = "apply_ai_resume") -> None:
        self.dataset = dataset
        self._index: dict[str, Evidence] = {}   # evidence_id -> Evidence
        self._feedback: list[dict] = []

    # -- pure helpers (unit-testable offline) --------------------------------
    def _docs(self, resume: Resume) -> list[str]:
        """Build the tagged documents + the evidence index. No network."""
        ev = _all_bullets(resume)
        self._index = {e.evidence_id: e for e in ev}
        return [f"[{e.evidence_id}] {e.text} "
                f"(skills: {', '.join(e.skills)}; source: {e.source})" for e in ev]

    def load_index(self, resume: Resume) -> None:
        """Populate the provenance index without ingesting (retrieve-only procs)."""
        self._docs(resume)

    def _evidence_from_texts(self, texts) -> list[Evidence]:
        """Recover Evidence objects from retrieved chunk text via [id] tags."""
        seen: set[str] = set()
        out: list[Evidence] = []
        for t in texts:
            for eid in _TAG.findall(str(t)):
                if eid in self._index and eid not in seen:
                    seen.add(eid)
                    out.append(self._index[eid])
        return out

    # -- Cognee-backed operations --------------------------------------------
    def ingest(self, resume: Resume) -> None:
        configure()
        import cognee
        docs = self._docs(resume)
        try:
            _run(cognee.remember(docs, dataset_name=self.dataset, self_improvement=True))
        except Exception as exc:  # pragma: no cover - surfaces misconfig clearly
            raise RuntimeError(
                "Cognee ingest failed. Ensure LLM + embedding keys are set in .env "
                "(see .env.example). Original error: " + repr(exc)
            ) from exc

    def retrieve_evidence(self, query: str, k: int = 5) -> list[Evidence]:
        configure()
        import cognee
        from cognee import SearchType
        results = _run(cognee.search(query, query_type=SearchType.CHUNKS,
                                     top_k=k, datasets=[self.dataset]))
        texts = [getattr(r, "search_result", r) for r in results]
        return self._evidence_from_texts(texts)[:k]

    def improve(self, signal: dict) -> None:
        configure()
        import cognee
        self._feedback.append(signal)
        _run(cognee.improve(dataset=self.dataset))


class LocalMemory:
    """Explicit offline double: keyword-overlap retrieval. Never the default."""

    def __init__(self) -> None:
        self._ev: list[Evidence] = []
        self._weights: dict[str, float] = {}

    def ingest(self, resume: Resume) -> None:
        self._ev = _all_bullets(resume)

    def retrieve_evidence(self, query: str, k: int = 5) -> list[Evidence]:
        q = {t for t in query.lower().split() if t}

        def score(ev: Evidence) -> float:
            hay = (ev.text + " " + " ".join(ev.skills)).lower()
            overlap = sum(1 for t in q if t in hay)
            return overlap + self._weights.get(ev.evidence_id, 0.0)

        return sorted(self._ev, key=score, reverse=True)[:k]

    def improve(self, signal: dict) -> None:
        for eid in signal.get("accepted", []):
            self._weights[eid] = self._weights.get(eid, 0.0) + 0.5


def get_memory(force_local: bool | None = None):
    """Return the Cognee spine by default; LocalMemory only when explicitly asked."""
    if force_local is None:
        force_local = os.getenv("APPLY_AI_MEMORY", "").lower() == "local"
    return LocalMemory() if force_local else CogneeMemory()
