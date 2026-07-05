"""Tailoring is the core value + the no-fabrication guarantee.

These tests run fully offline: the memory store and the Claude call are both
substituted, so we exercise the grounding guard deterministically. The one rule
that must never break: every rendered bullet cites at least one REAL evidence_id
that was actually retrieved; anything the model invents is dropped.
"""
from apply_ai import tailor
from apply_ai.memory import Evidence
from apply_ai.schemas import JD, Resume, TailorResult, TailoredBullet


class FakeMemory:
    """Returns a fixed evidence set regardless of query."""

    def __init__(self, evidence):
        self._ev = evidence

    def retrieve_evidence(self, query, k=5):
        return self._ev[:k]


def _resume():
    return Resume(name="Sai", skills=["python", "sql"])


def _jd():
    return JD(title="Backend Engineer", company="Acme",
              required_skills=["python", "fastapi"], nice_to_have=["aws"],
              responsibilities=["build low-latency services"])


def _evidence():
    return [
        Evidence(evidence_id="ev-1", text="Built a FastAPI service, cut p95 40%",
                 skills=["python", "fastapi", "aws"], source="SWE @ Acme"),
        Evidence(evidence_id="ev-2", text="Designed SQL pipelines over 5M rows/day",
                 skills=["sql"], source="SWE @ Acme"),
    ]


def test_grounded_bullets_kept_fabricated_dropped(monkeypatch):
    # Claude returns 3 candidates: two grounded, one fabricated (cites ev-99).
    def fake_call(jd, evidence):
        return [
            {"text": "Engineered a FastAPI service cutting p95 latency 40%",
             "evidence_ids": ["ev-1"], "skills": ["python", "fastapi", "aws"]},
            {"text": "Built SQL data pipelines processing millions of rows daily",
             "evidence_ids": ["ev-2"], "skills": ["sql"]},
            {"text": "Led a team of 12 engineers (INVENTED)",
             "evidence_ids": ["ev-99"], "skills": ["leadership"]},
        ]

    monkeypatch.setattr(tailor, "_claude_tailor", fake_call)
    res = tailor.tailor(_resume(), _jd(), FakeMemory(_evidence()))

    assert isinstance(res, TailorResult)
    # fabricated bullet dropped
    assert len(res.bullets) == 2
    assert any("INVENTED" in d for d in res.dropped)
    # THE GUARANTEE: every kept bullet has non-empty provenance, all valid ids
    valid = {"ev-1", "ev-2"}
    for b in res.bullets:
        assert isinstance(b, TailoredBullet)
        assert b.provenance
        assert set(b.provenance) <= valid
    # provenance map is keyed by bullet id and mirrors the bullets
    assert res.provenance == {b.id: b.provenance for b in res.bullets}


def test_partial_fabricated_ids_are_filtered_not_whole_bullet(monkeypatch):
    # A bullet citing one real + one invented id keeps only the real id.
    def fake_call(jd, evidence):
        return [{"text": "Shipped a FastAPI service", "evidence_ids": ["ev-1", "ev-77"],
                 "skills": ["python"]}]

    monkeypatch.setattr(tailor, "_claude_tailor", fake_call)
    res = tailor.tailor(_resume(), _jd(), FakeMemory(_evidence()))
    assert len(res.bullets) == 1
    assert res.bullets[0].provenance == ["ev-1"]


def test_skills_delta_surfaces_jd_skills(monkeypatch):
    # aws + fastapi are JD skills grounded in evidence but not in base resume skills.
    def fake_call(jd, evidence):
        return [{"text": "FastAPI service on AWS", "evidence_ids": ["ev-1"],
                 "skills": ["python", "fastapi", "aws"]}]

    monkeypatch.setattr(tailor, "_claude_tailor", fake_call)
    res = tailor.tailor(_resume(), _jd(), FakeMemory(_evidence()))
    # base resume has python, sql. JD-relevant skills newly surfaced: aws, fastapi
    assert set(res.skills_delta) == {"aws", "fastapi"}


def test_empty_when_nothing_grounded(monkeypatch):
    def fake_call(jd, evidence):
        return [{"text": "totally made up", "evidence_ids": ["ev-x"], "skills": []}]

    monkeypatch.setattr(tailor, "_claude_tailor", fake_call)
    res = tailor.tailor(_resume(), _jd(), FakeMemory(_evidence()))
    assert res.bullets == []
    assert res.provenance == {}
    assert res.dropped == ["totally made up"]
