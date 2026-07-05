"""Hosted Cognee via REST — the app's persistent knowledge spine.

Talks to a hosted Cognee tenant over HTTP (POST /api/v1/remember, /api/v1/recall)
with X-Api-Key auth. No local graph DB, so it sidesteps the missing embedded
native lib entirely. Persistence lives in the tenant's cloud graph + vector DBs.

Optional DNS pin (COGNEE_RESOLVE_IP) works around resolvers that REFUSE the tenant
hostname, while keeping the hostname for TLS SNI/cert validation.

Same interface as LocalMemory/CogneeMemory: ingest / retrieve_evidence / improve.
Provenance is preserved by tagging each bullet `[evidence_id] ...` on ingest and
recovering those ids from recalled text.
"""
from __future__ import annotations
import io
import os
import re
import socket
import httpx
from .schemas import Resume
from .memory import Evidence, _all_bullets

_TAG = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def _pin_host(host: str, ip: str) -> None:
    """Force `host` to resolve to `ip` (keeps hostname so TLS SNI/cert still match)."""
    _orig = socket.getaddrinfo

    def patched(h, *args, **kwargs):
        return _orig(ip if h == host else h, *args, **kwargs)

    socket.getaddrinfo = patched


class CogneeRestMemory:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 dataset: str = "apply_ai_resume", resolve_ip: str | None = None) -> None:
        self.base_url = (base_url or os.environ["COGNEE_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["COGNEE_API_KEY"]
        self.dataset = dataset
        self._index: dict[str, Evidence] = {}
        ip = resolve_ip or os.getenv("COGNEE_RESOLVE_IP")
        if ip:
            _pin_host(httpx.URL(self.base_url).host, ip)

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key}

    def _docs(self, resume: Resume) -> list[str]:
        ev = _all_bullets(resume)
        self._index = {e.evidence_id: e for e in ev}
        return [f"[{e.evidence_id}] {e.text} "
                f"(skills: {', '.join(e.skills)}; source: {e.source})" for e in ev]

    def load_index(self, resume: Resume) -> None:
        self._docs(resume)

    def ingest(self, resume: Resume) -> dict:
        text = "\n".join(self._docs(resume))
        files = {"data": ("resume_evidence.txt", io.BytesIO(text.encode()), "text/plain")}
        data = {"datasetName": self.dataset}
        r = httpx.post(f"{self.base_url}/api/v1/remember", headers=self._headers,
                       data=data, files=files, timeout=180)
        r.raise_for_status()
        return r.json()

    def _evidence_from_texts(self, texts) -> list[Evidence]:
        seen: set[str] = set()
        out: list[Evidence] = []
        for t in texts:
            for eid in _TAG.findall(str(t)):
                if eid in self._index and eid not in seen:
                    seen.add(eid)
                    out.append(self._index[eid])
        return out

    def _keyword_rank(self, query: str, k: int) -> list[Evidence]:
        """Local keyword-overlap fallback over the indexed evidence — used when the
        graph isn't cognified yet or the tenant is unreachable, so Cognee-first never
        leaves tailoring empty."""
        q = {t for t in query.lower().split() if t}
        def score(e: Evidence) -> int:
            hay = (e.text + " " + " ".join(e.skills)).lower()
            return sum(1 for t in q if t in hay)
        return sorted(self._index.values(), key=score, reverse=True)[:k]

    def retrieve_evidence(self, query: str, k: int = 5) -> list[Evidence]:
        # CHUNKS returns the raw retrieved text (which still carries the [evidence_id]
        # tags) rather than an LLM completion that would strip them — so provenance
        # recovery in _evidence_from_texts works.
        payload = {"query": query, "datasets": [self.dataset],
                   "searchType": "CHUNKS", "topK": k}
        try:
            r = httpx.post(f"{self.base_url}/api/v1/recall", headers=self._headers,
                           json=payload, timeout=180)
            r.raise_for_status()
            results = r.json()
            if isinstance(results, list):
                texts = [d.get("text", "") if isinstance(d, dict) else str(d) for d in results]
            else:
                texts = [str(results)]
            ev = self._evidence_from_texts(texts)[:k]
            if ev:
                return ev
        except Exception:
            pass  # tenant unreachable / graph not ready -> fall back below
        return self._keyword_rank(query, k)

    def improve(self, signal: dict) -> None:
        # Hosted Cognee self-improves during remember(self_improvement=True);
        # nothing to push from the client for now.
        return None
