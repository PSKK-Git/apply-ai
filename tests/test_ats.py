"""ATS scoring = deterministic JD-keyword coverage of the tailored resume, plus a
gap list that distinguishes 'groundable in your real evidence' from 'needs new
real input'. No LLM — the number must be stable and explainable.
"""
from apply_ai import ats
from apply_ai.memory import Evidence
from apply_ai.schemas import ATSReport, JD, Resume, TailoredBullet, TailorResult


def _jd():
    return JD(title="Backend Engineer", company="Acme",
              required_skills=["python", "fastapi", "kubernetes"],
              nice_to_have=["aws"])


def _tailored(*texts_skills):
    bullets = [TailoredBullet(id=f"tb-{i+1}", text=t, provenance=["ev-1"], skills=s)
               for i, (t, s) in enumerate(texts_skills)]
    return TailorResult(bullets=bullets, provenance={b.id: b.provenance for b in bullets})


def test_full_coverage_scores_100():
    jd = _jd()
    resume = Resume(name="Sai", skills=["python", "aws"])
    tailored = _tailored(
        ("Built a FastAPI service on Kubernetes", ["python", "fastapi", "kubernetes", "aws"]),
    )
    rep = ats.ats_report(jd, tailored, resume)
    assert isinstance(rep, ATSReport)
    assert rep.score == 100
    assert rep.missing_keywords == []
    assert rep.gaps == []


def test_partial_coverage_and_alias():
    jd = _jd()  # python, fastapi, kubernetes, aws
    resume = Resume(name="Sai", skills=["python"])
    # "k8s" must normalize to kubernetes and count as covered; fastapi + aws missing
    tailored = _tailored(("Shipped python microservices on k8s", ["python", "k8s"]))
    rep = ats.ats_report(jd, tailored, resume)
    # matched: python, kubernetes -> 2 of 4 = 50
    assert rep.score == 50
    assert set(rep.matched_keywords) == {"python", "kubernetes"}
    assert set(rep.missing_keywords) == {"fastapi", "aws"}


def test_gaps_flag_groundable_vs_new():
    jd = _jd()
    resume = Resume(name="Sai", skills=["python"])
    tailored = _tailored(("Wrote python services", ["python"]))
    # corpus proves the candidate really has fastapi+aws evidence (groundable);
    # kubernetes appears nowhere -> needs new input.
    corpus = [
        Evidence(evidence_id="ev-1", text="Built FastAPI APIs on AWS", skills=["fastapi", "aws"]),
    ]
    rep = ats.ats_report(jd, tailored, resume, corpus=corpus)
    by_kw = {g.keyword: g.groundable for g in rep.gaps}
    assert by_kw == {"fastapi": True, "aws": True, "kubernetes": False}


def test_no_jd_keywords_is_vacuously_full():
    rep = ats.ats_report(JD(title="x", company="y"), _tailored(), Resume(name="Sai"))
    assert rep.score == 100
    assert rep.gaps == []
