import json
from apply_ai import jd_ingest
from apply_ai.schemas import JD


def test_ingest_uses_raw_text_when_not_url(monkeypatch):
    captured = {}

    def fake_extract(raw, company_job_id=None):
        captured["raw"] = raw
        return JD(title="Engineer", company="Acme",
                  required_skills=["python"], company_job_id=company_job_id)

    monkeypatch.setattr(jd_ingest, "extract_jd", fake_extract)
    jd = jd_ingest.ingest("We need a Python engineer", company_job_id="REQ-9")
    assert captured["raw"] == "We need a Python engineer"
    assert jd.company == "Acme"
    assert jd.company_job_id == "REQ-9"


def test_extract_parses_claude_json(monkeypatch):
    payload = {"title": "Backend Engineer", "company": "Beta",
               "required_skills": ["python", "sql"], "nice_to_have": ["aws"],
               "responsibilities": ["build APIs"]}

    def fake_call(prompt: str) -> str:
        return "```json\n" + json.dumps(payload) + "\n```"

    monkeypatch.setattr(jd_ingest, "_claude_json", fake_call)
    jd = jd_ingest.extract_jd("raw jd text here", company_job_id="X-1")
    assert jd.title == "Backend Engineer"
    assert jd.required_skills == ["python", "sql"]
    assert jd.company_job_id == "X-1"
    assert jd.raw_text == "raw jd text here"
