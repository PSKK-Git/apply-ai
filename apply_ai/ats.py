"""Deterministic ATS coverage: does the tailored resume actually contain the JD's
keywords an ATS parser scans for? Pure text/set math over normalized skills — no
LLM guess. Missing keywords become a gap list that separates 'you can ground this
in real evidence' from 'this needs new real input from you' (the add-if-true flow).
"""
from __future__ import annotations
import re

from .schemas import ATSReport, Gap, JD, Resume, TailorResult
from .fit_score import normalize, normalize_set

_TOKEN = re.compile(r"[a-z0-9+#.]+")


def _norm_tokens(text: str) -> set[str]:
    return {normalize(t) for t in _TOKEN.findall(text.lower())}


def _present(keyword: str, text_lower: str, tokens: set[str]) -> bool:
    n = normalize(keyword)
    if not n:
        return False
    # multiword keywords ("machine learning") -> substring; single -> exact token
    return n in text_lower if " " in n else n in tokens


def _corpus_haystack(corpus) -> tuple[str, set[str]]:
    parts: list[str] = []
    for e in corpus or []:
        parts.append(e.text)
        parts.extend(e.skills)
    text = " ".join(parts).lower()
    return text, _norm_tokens(text)


def ats_report(jd: JD, tailored: TailorResult, resume: Resume, corpus=None) -> ATSReport:
    keywords = normalize_set(jd.required_skills) | normalize_set(jd.nice_to_have)

    # what an ATS would parse from the tailored resume: bullet text + skills + base skills
    parts: list[str] = list(resume.skills)
    for b in tailored.bullets:
        parts.append(b.text)
        parts.extend(b.skills)
    text_lower = " ".join(parts).lower()
    tokens = _norm_tokens(text_lower)

    matched = sorted(k for k in keywords if _present(k, text_lower, tokens))
    missing = sorted(keywords - set(matched))

    score = 100 if not keywords else round(100 * len(matched) / len(keywords))

    corpus_text, corpus_tokens = _corpus_haystack(corpus)
    gaps = [
        Gap(keyword=k, groundable=_present(k, corpus_text, corpus_tokens))
        for k in missing
    ]

    return ATSReport(
        score=score,
        matched_keywords=matched,
        missing_keywords=missing,
        gaps=gaps,
    )
