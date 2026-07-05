"""Tailor resume bullets to a JD, grounded in real retrieved evidence.

The pipeline: build a query from the JD -> `memory.retrieve_evidence` (the Cognee
spine) -> Claude rewrites/selects bullets, each citing the `evidence_id`s it used
-> a hard grounding guard drops anything not anchored in a REAL retrieved id.

The no-fabrication guarantee lives in `_ground`, not in the prompt: even if the
model invents an id or a whole bullet, it never reaches the rendered result. That
is why the guard is unit-tested offline with the Claude call substituted.
"""
from __future__ import annotations
import json
import os
import re

from .schemas import JD, Resume, TailoredBullet, TailorResult
from .fit_score import normalize, normalize_set

_MODEL = os.getenv("APPLY_AI_LLM_MODEL_TAILOR", "claude-sonnet-5")

_PROMPT = """You tailor resume bullets to a job description. You may ONLY use the
evidence provided; never invent achievements, numbers, or skills. Rewrite/select
the evidence into strong, JD-aligned bullets. Each output bullet MUST cite the
evidence_id(s) it is grounded in.

Return ONLY a JSON array; each item:
  {{"text": "<tailored bullet>", "evidence_ids": ["ev-1", ...], "skills": ["python", ...]}}

Job: {title} at {company}
Required skills: {required}
Nice to have: {nice}
Responsibilities: {resp}

Evidence (id | text | skills):
{evidence}
"""


def _query_for_jd(jd: JD) -> str:
    parts = list(jd.required_skills) + list(jd.nice_to_have) + list(jd.responsibilities)
    parts.append(jd.title)
    return " ".join(p for p in parts if p)


def _claude_tailor(jd: JD, evidence) -> list[dict]:
    """Call Claude and return the raw candidate list. Substituted in tests."""
    from . import llm

    ev_block = "\n".join(
        f"{e.evidence_id} | {e.text} | {', '.join(e.skills)}" for e in evidence
    )
    prompt = _PROMPT.format(
        title=jd.title, company=jd.company,
        required=", ".join(jd.required_skills) or "(none)",
        nice=", ".join(jd.nice_to_have) or "(none)",
        resp="; ".join(jd.responsibilities) or "(none)",
        evidence=ev_block or "(none)",
    )
    return _parse_candidates(llm.complete(prompt, max_tokens=1500))


def _parse_candidates(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _ground(candidates: list[dict], valid_ids: set[str]):
    """Keep only bullets anchored in a real retrieved evidence_id.

    Partially-fabricated citations are filtered id-by-id; a bullet with no valid id
    left is dropped whole. This is the enforcement point of the no-fabrication rule.
    """
    kept: list[TailoredBullet] = []
    dropped: list[str] = []
    for c in candidates:
        text = str(c.get("text", "")).strip()
        prov = [e for e in c.get("evidence_ids", []) if e in valid_ids]
        if not text or not prov:
            if text:
                dropped.append(text)
            continue
        kept.append(TailoredBullet(
            id=f"tb-{len(kept) + 1}",
            text=text,
            provenance=prov,
            skills=[s for s in c.get("skills", []) if s],
        ))
    return kept, dropped


def tailor(resume: Resume, jd: JD, memory, *, k: int = 6) -> TailorResult:
    """Produce grounded, provenance-tagged bullets for `jd` from `resume`'s evidence."""
    evidence = memory.retrieve_evidence(_query_for_jd(jd), k=k)
    valid_ids = {e.evidence_id for e in evidence}
    kept, dropped = _ground(_claude_tailor(jd, evidence), valid_ids)

    base = normalize_set(resume.skills)
    jd_skills = normalize_set(jd.required_skills) | normalize_set(jd.nice_to_have)
    surfaced = {normalize(s) for b in kept for s in b.skills}
    skills_delta = sorted((surfaced & jd_skills) - base)

    return TailorResult(
        bullets=kept,
        provenance={b.id: b.provenance for b in kept},
        skills_delta=skills_delta,
        dropped=dropped,
    )
