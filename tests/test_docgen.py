"""docgen renders a real PDF from the tailored résumé — verify it produces a valid,
non-trivial PDF (correct magic header) and works even before tailoring (falls back
to the base résumé bullets)."""
from apply_ai import docgen
from apply_ai.schemas import (Bullet, Contact, Education, Experience, Resume,
                              TailoredBullet, TailorResult)


def _resume():
    return Resume(name="Jane Doe", contact=Contact(email="jane@x.com"),
                  skills=["python", "aws"],
                  experiences=[Experience(id="exp-1", company="Acme", role="Eng", start="2020",
                                          bullets=[Bullet(id="ev-1", text="Built X", skills=["python"])])],
                  education=[Education(school="MIT", degree="BS CS", year="2020")])


def test_build_pdf_from_tailored():
    t = TailorResult(bullets=[TailoredBullet(id="tb-1", text="Engineered a scalable API",
                                             provenance=["ev-1"], skills=["python"])])
    data = docgen.build_pdf_bytes(_resume(), t)
    assert data[:4] == b"%PDF"
    assert len(data) > 900


def test_build_pdf_falls_back_without_tailoring():
    data = docgen.build_pdf_bytes(_resume(), TailorResult())
    assert data[:4] == b"%PDF"


def test_build_pdf_writes_file(tmp_path):
    out = tmp_path / "resume.pdf"
    p = docgen.build_pdf(_resume(), TailorResult(), str(out))
    assert out.exists() and out.stat().st_size > 900 and p == str(out)
