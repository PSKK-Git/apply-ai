from apply_ai import tracker
from apply_ai.schemas import JD, FitResult


def _fit(score=80):
    return FitResult(score=score, matched=["python"], missing_required=["go"],
                     missing_nice=[], contributions={"python": 37.5})


def test_ids_increment_and_add():
    conn = tracker.init_db(":memory:")
    jd1 = JD(title="Eng", company="Acme", company_job_id="REQ-1")
    jd2 = JD(title="Dev", company="Beta")
    id1 = tracker.add_job(conn, jd1, _fit(80))
    id2 = tracker.add_job(conn, jd2, _fit(50))
    assert id1 == "AAI-0001"
    assert id2 == "AAI-0002"
    row = tracker.get_job(conn, id1)
    assert row["company_job_id"] == "REQ-1"
    assert row["fit_score"] == 80
    assert row["status"] == "discovered"


def test_filters():
    conn = tracker.init_db(":memory:")
    a = tracker.add_job(conn, JD(title="A", company="Acme"), _fit(90))
    b = tracker.add_job(conn, JD(title="B", company="Beta"), _fit(40))
    tracker.set_status(conn, a, "applied", applied_date="2026-07-04")
    tracker.set_status(conn, b, "interview", interview_date="2026-07-05")
    assert [r["local_job_id"] for r in tracker.list_jobs(conn, status="applied")] == [a]
    assert [r["local_job_id"] for r in tracker.list_jobs(conn, min_fit=80)] == [a]
    assert [r["local_job_id"] for r in tracker.list_jobs(conn, interview_date="2026-07-05")] == [b]
    assert [r["local_job_id"] for r in tracker.list_jobs(conn, applied_from="2026-07-01", applied_to="2026-07-04")] == [a]


def test_find_job_by_url_dedupes():
    conn = tracker.init_db(":memory:")
    url = "https://apply.workable.com/x/j/ABC/"
    a = tracker.add_job(conn, JD(title="Eng", company="Acme"), _fit(80), url=url)
    tracker.add_job(conn, JD(title="Other", company="Beta"), _fit(50), url="")
    found = tracker.find_job_by_url(conn, url)
    assert found is not None and found["local_job_id"] == a
    assert tracker.find_job_by_url(conn, "https://different.com/j/1") is None
    assert tracker.find_job_by_url(conn, "") is None
