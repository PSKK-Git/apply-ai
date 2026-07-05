"""Multi-model cross-verification. Claude writes the resume; ChatGPT and Gemini
independently check it. Tests inject fake provider runners so no network is hit:
we verify aggregation (consensus across models, averaged ATS alignment) and that
a failing/absent provider degrades gracefully instead of crashing the run.
"""
from apply_ai import verify
from apply_ai.schemas import JD, TailoredBullet, TailorResult, VerifyReport, VerifyVerdict


def _tailored():
    b = [TailoredBullet(id="tb-1", text="FastAPI service", provenance=["ev-1"]),
         TailoredBullet(id="tb-2", text="Led 12 engineers", provenance=["ev-2"])]
    return TailorResult(bullets=b, provenance={x.id: x.provenance for x in b})


def _jd():
    return JD(title="Backend Engineer", company="Acme", required_skills=["python"])


def test_consensus_and_average(monkeypatch):
    def claude(jd, tailored, ats):
        return VerifyVerdict(provider="claude", ats_alignment=90, fabrication_flags=["tb-2"])

    def openai(jd, tailored, ats):
        return VerifyVerdict(provider="openai", ats_alignment=80,
                             fabrication_flags=["tb-2"], suggestions=["add metrics"])

    def gemini(jd, tailored, ats):
        return VerifyVerdict(provider="gemini", ats_alignment=70, fabrication_flags=["tb-1"])

    rep = verify.cross_verify(_jd(), _tailored(), None,
                              runners={"claude": claude, "openai": openai, "gemini": gemini})
    assert isinstance(rep, VerifyReport)
    # tb-2 doubted by claude+openai -> consensus; tb-1 only by gemini -> not
    assert rep.consensus_flags == ["tb-2"]
    assert rep.avg_ats_alignment == 80          # (90+80+70)/3
    assert {v.provider for v in rep.verdicts} == {"claude", "openai", "gemini"}


def test_failing_provider_is_recorded_not_fatal():
    def claude(jd, tailored, ats):
        return VerifyVerdict(provider="claude", ats_alignment=88)

    def gemini(jd, tailored, ats):
        raise RuntimeError("no gemini key")

    rep = verify.cross_verify(_jd(), _tailored(), None,
                              runners={"claude": claude, "gemini": gemini})
    errs = {v.provider: v.error for v in rep.verdicts}
    assert errs["claude"] is None
    assert "gemini" in errs and errs["gemini"]
    assert rep.avg_ats_alignment == 88          # only the successful one counts


def test_available_providers_reflects_keys(monkeypatch):
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY",
              "LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "y")
    avail = verify.available_providers()
    assert "claude" in avail and "openai" in avail
    assert "gemini" not in avail
