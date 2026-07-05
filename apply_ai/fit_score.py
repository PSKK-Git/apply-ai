from __future__ import annotations
from .schemas import JD, FitResult

ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "postgres": "postgresql", "k8s": "kubernetes",
    "ml": "machine learning", "golang": "go",
}


def normalize(skill: str) -> str:
    s = skill.strip().lower()
    return ALIASES.get(s, s)


def normalize_set(skills) -> set[str]:
    return {normalize(s) for s in skills if s and s.strip()}


def score(resume_skills, jd: JD) -> FitResult:
    R = normalize_set(resume_skills)
    req = normalize_set(jd.required_skills)
    nice = normalize_set(jd.nice_to_have)

    req_matched = R & req
    nice_matched = R & nice
    required_hit = len(req_matched) / len(req) if req else 0.0
    nice_hit = len(nice_matched) / len(nice) if nice else 0.0
    total = round(100 * (0.75 * required_hit + 0.25 * nice_hit))

    contributions: dict[str, float] = {}
    if req:
        per = 75.0 / len(req)
        for s in sorted(req_matched):
            contributions[s] = round(per, 2)
    if nice:
        per = 25.0 / len(nice)
        for s in sorted(nice_matched):
            contributions[s] = round(per, 2)

    return FitResult(
        score=total,
        matched=sorted(req_matched | nice_matched),
        missing_required=sorted(req - R),
        missing_nice=sorted(nice - R),
        contributions=contributions,
    )
