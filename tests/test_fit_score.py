from apply_ai.fit_score import score, normalize
from apply_ai.schemas import JD


def test_perfect_required_no_nice():
    jd = JD(title="x", company="y", required_skills=["python", "sql"])
    r = score(["python", "sql", "extra"], jd)
    assert r.score == 75
    assert r.missing_required == []
    assert set(r.matched) == {"python", "sql"}


def test_perfect_required_and_nice():
    jd = JD(title="x", company="y", required_skills=["python", "sql"],
            nice_to_have=["aws", "docker"])
    r = score(["python", "sql", "aws", "docker"], jd)
    assert r.score == 100


def test_partial_required_clean_integer():
    # 3 of 5 required -> 0.6 * 75 = 45
    jd = JD(title="x", company="y",
            required_skills=["python", "sql", "go", "rust", "scala"])
    r = score(["python", "sql", "go"], jd)
    assert r.score == 45
    assert r.missing_required == ["rust", "scala"]


def test_no_match():
    jd = JD(title="x", company="y", required_skills=["python"])
    r = score([], jd)
    assert r.score == 0
    assert r.missing_required == ["python"]


def test_alias_normalisation():
    jd = JD(title="x", company="y", required_skills=["javascript"])
    r = score(["JS"], jd)
    assert r.score == 75
    assert normalize("Postgres") == "postgresql"
