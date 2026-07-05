from apply_ai.schemas import Resume, JD, load_resume


def test_load_sample_resume():
    r = load_resume("data/resume.json")
    assert isinstance(r, Resume)
    assert r.name
    assert r.skills
    # every bullet has a stable evidence id
    ids = [b.id for e in r.experiences for b in e.bullets]
    assert len(ids) == len(set(ids))
    assert all(ids)


def test_jd_defaults():
    jd = JD(title="Engineer", company="Acme")
    assert jd.required_skills == []
    assert jd.company_job_id is None
