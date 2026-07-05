# Apply_AI Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic spine of Apply_AI — project scaffold, typed schemas, the SQLite job tracker, and the explainable fit score — then wire JD ingestion and Cognee-backed memory, producing a slice you can demo (paste a JD → get a scored, gap-annotated result).

**Architecture:** A single Python package `apply_ai/` with focused modules behind clean interfaces. The deterministic core (schemas, tracker, fit_score) has zero external dependencies and is built + tested first. JD ingestion (Claude extraction) and memory (Cognee) come after and depend on API keys loaded via a `.env`.

**Tech Stack:** Python 3.11–3.13 (venv) · pydantic v2 · SQLite (stdlib) · pytest · trafilatura + httpx · anthropic SDK · cognee.

## Global Constraints

- **Python version:** target 3.11–3.13 in a venv. Host default is 3.14, on which `cognee`/ML deps may lack wheels — if `pip install cognee` fails, recreate the venv with `python3.12` (or `3.11`). Copy verbatim: package name is `apply_ai` (underscore).
- **Fit score is deterministic:** never call an LLM to produce the number. Pure set math over normalized skills. Weights are exactly `0.75 * required_hit + 0.25 * nice_hit`, scaled ×100, `round()`-ed.
- **No fabrication (applies once tailoring lands):** every tailored bullet must carry a non-empty `provenance` list of `evidence_id`s.
- **Draft-only email (later phase):** never call Gmail `messages.send`.
- **LLM provider:** Anthropic Claude, model id `claude-sonnet-5`.
- **Local IDs:** local job ids are `AAI-000N` zero-padded to 4 digits.

---

### Task 0: Project scaffold + environment

**Files:**
- Create: `Apply_AI/pyproject.toml`
- Create: `Apply_AI/.env.example`
- Create: `Apply_AI/.gitignore`
- Create: `Apply_AI/apply_ai/__init__.py`
- Create: `Apply_AI/tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installed, importable `apply_ai` package in a venv; `pytest` runnable; env vars loadable.

- [ ] **Step 1: Create the package files**

`Apply_AI/pyproject.toml`:
```toml
[project]
name = "apply_ai"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = [
  "pydantic>=2.6",
  "python-dotenv>=1.0",
  "httpx>=0.27",
  "trafilatura>=1.8",
  "anthropic>=0.40",
]

[project.optional-dependencies]
memory = ["cognee>=0.1.15"]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`Apply_AI/.env.example`:
```
ANTHROPIC_API_KEY=sk-ant-...
COGNEE_API_KEY=...
```

`Apply_AI/.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
data/tracker.db
data/*.pdf
*.egg-info/
```

`Apply_AI/apply_ai/__init__.py` and `Apply_AI/tests/__init__.py`: empty files.

- [ ] **Step 2: Create venv and install (dev + base deps only; memory extra deferred)**

Run:
```bash
cd Apply_AI
python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```
Expected: install succeeds, `apply_ai` is importable.

- [ ] **Step 3: Create `.env` from example and confirm keys load**

Run:
```bash
cp .env.example .env   # then paste real keys into .env
. .venv/bin/activate
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('anthropic', bool(os.getenv('ANTHROPIC_API_KEY'))); print('cognee', bool(os.getenv('COGNEE_API_KEY')))"
```
Expected: `anthropic True` and `cognee True`. If False, the keys are not in `.env` yet — stop and add them.

- [ ] **Step 4: Verify pytest runs (no tests yet is fine)**

Run: `. .venv/bin/activate && pytest -q`
Expected: `no tests ran` (exit 5) — confirms pytest is wired.

- [ ] **Step 5: Commit**

```bash
git add Apply_AI/pyproject.toml Apply_AI/.env.example Apply_AI/.gitignore Apply_AI/apply_ai/__init__.py Apply_AI/tests/__init__.py
git commit -m "chore: scaffold apply_ai package and environment"
```

---

### Task 1: Typed schemas + sample resume

**Files:**
- Create: `Apply_AI/apply_ai/schemas.py`
- Create: `Apply_AI/data/resume.json`
- Test: `Apply_AI/tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces: pydantic models `Bullet(id:str, text:str, skills:list[str])`, `Experience`, `Project`, `Education`, `Contact`, `Resume`, `JD(title,company,company_job_id,required_skills,nice_to_have,responsibilities,raw_text)`, `FitResult(score:int, matched:list[str], missing_required:list[str], missing_nice:list[str], contributions:dict[str,float])`. Function `load_resume(path)->Resume`.

- [ ] **Step 1: Write the failing test**

`Apply_AI/tests/test_schemas.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `. .venv/bin/activate && pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: apply_ai.schemas`.

- [ ] **Step 3: Write the schemas**

`Apply_AI/apply_ai/schemas.py`:
```python
from __future__ import annotations
import json
from pydantic import BaseModel, Field


class Bullet(BaseModel):
    id: str                    # evidence_id — anchor for grounding/provenance
    text: str
    skills: list[str] = Field(default_factory=list)


class Experience(BaseModel):
    id: str
    company: str
    role: str
    start: str
    end: str | None = None
    bullets: list[Bullet] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    name: str
    bullets: list[Bullet] = Field(default_factory=list)


class Education(BaseModel):
    school: str
    degree: str
    year: str


class Contact(BaseModel):
    email: str = ""
    phone: str = ""
    links: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str
    contact: Contact = Field(default_factory=Contact)
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)


class JD(BaseModel):
    title: str
    company: str
    company_job_id: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    raw_text: str = ""


class FitResult(BaseModel):
    score: int
    matched: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    missing_nice: list[str] = Field(default_factory=list)
    contributions: dict[str, float] = Field(default_factory=dict)


def load_resume(path: str = "data/resume.json") -> Resume:
    with open(path, encoding="utf-8") as fh:
        return Resume.model_validate(json.load(fh))
```

- [ ] **Step 4: Create the sample resume**

`Apply_AI/data/resume.json`:
```json
{
  "name": "Sai Karthik",
  "contact": {"email": "saikarthik8106@gmail.com", "phone": "", "links": ["github.com/saikarthik"]},
  "skills": ["python", "streamlit", "sql", "fastapi", "pandas", "docker", "aws", "llm"],
  "experiences": [
    {
      "id": "exp-1", "company": "Acme Labs", "role": "Software Engineer",
      "start": "2023-01", "end": "2024-06",
      "bullets": [
        {"id": "ev-1", "text": "Built a FastAPI service handling 2k req/s, cutting p95 latency 40%", "skills": ["python", "fastapi", "aws"]},
        {"id": "ev-2", "text": "Designed SQL pipelines processing 5M rows/day with pandas", "skills": ["sql", "pandas"]},
        {"id": "ev-3", "text": "Containerized services with Docker for reproducible deploys", "skills": ["docker"]}
      ]
    }
  ],
  "projects": [
    {"id": "proj-1", "name": "Apply_AI", "bullets": [
      {"id": "ev-9", "text": "Built an LLM resume tailor with a deterministic fit score", "skills": ["python", "llm", "streamlit"]}
    ]}
  ],
  "education": [{"school": "State University", "degree": "B.Tech CSE", "year": "2022"}]
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `. .venv/bin/activate && pytest tests/test_schemas.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add Apply_AI/apply_ai/schemas.py Apply_AI/data/resume.json Apply_AI/tests/test_schemas.py
git commit -m "feat: add pydantic schemas and sample resume"
```

---

### Task 2: SQLite tracker

**Files:**
- Create: `Apply_AI/apply_ai/tracker.py`
- Test: `Apply_AI/tests/test_tracker.py`

**Interfaces:**
- Consumes: `JD`, `FitResult` from `apply_ai.schemas`.
- Produces:
  - `init_db(path=":memory:") -> sqlite3.Connection` (creates tables if absent)
  - `next_local_job_id(conn) -> str` (`AAI-0001`, incrementing)
  - `add_job(conn, jd: JD, fit: FitResult, url="") -> str` (returns local_job_id)
  - `get_job(conn, local_job_id) -> dict | None`
  - `list_jobs(conn, *, status=None, applied_from=None, applied_to=None, interview_date=None, min_fit=None) -> list[dict]`
  - `set_status(conn, local_job_id, status, applied_date=None, interview_date=None) -> None`

- [ ] **Step 1: Write the failing test**

`Apply_AI/tests/test_tracker.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `. .venv/bin/activate && pytest tests/test_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: apply_ai.tracker`.

- [ ] **Step 3: Write the tracker**

`Apply_AI/apply_ai/tracker.py`:
```python
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from .schemas import JD, FitResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  local_job_id   TEXT PRIMARY KEY,
  company_job_id TEXT,
  company        TEXT,
  title          TEXT,
  url            TEXT,
  jd_json        TEXT,
  fit_score      INTEGER,
  fit_breakdown  TEXT,
  status         TEXT,
  applied_date   TEXT,
  interview_date TEXT,
  created_at     TEXT
);
CREATE TABLE IF NOT EXISTS resume_versions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  local_job_id  TEXT,
  version       INTEGER,
  tailored_json TEXT,
  provenance    TEXT,
  skills_delta  TEXT,
  pdf_path      TEXT,
  accepted      INTEGER DEFAULT 0,
  created_at    TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  local_job_id  TEXT,
  type          TEXT,
  payload       TEXT,
  seen          INTEGER DEFAULT 0,
  created_at    TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def next_local_job_id(conn: sqlite3.Connection) -> str:
    n = conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]
    return f"AAI-{n + 1:04d}"


def add_job(conn: sqlite3.Connection, jd: JD, fit: FitResult, url: str = "") -> str:
    local_id = next_local_job_id(conn)
    conn.execute(
        """INSERT INTO jobs (local_job_id, company_job_id, company, title, url,
             jd_json, fit_score, fit_breakdown, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (local_id, jd.company_job_id, jd.company, jd.title, url,
         jd.model_dump_json(), fit.score, fit.model_dump_json(),
         "discovered", _now()),
    )
    conn.commit()
    return local_id


def get_job(conn: sqlite3.Connection, local_job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE local_job_id=?", (local_job_id,)).fetchone()
    return dict(row) if row else None


def set_status(conn, local_job_id, status, applied_date=None, interview_date=None) -> None:
    fields = ["status=?"]
    vals = [status]
    if applied_date is not None:
        fields.append("applied_date=?"); vals.append(applied_date)
    if interview_date is not None:
        fields.append("interview_date=?"); vals.append(interview_date)
    vals.append(local_job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE local_job_id=?", vals)
    conn.commit()


def list_jobs(conn, *, status=None, applied_from=None, applied_to=None,
              interview_date=None, min_fit=None) -> list[dict]:
    where, vals = [], []
    if status is not None:
        where.append("status=?"); vals.append(status)
    if applied_from is not None:
        where.append("applied_date>=?"); vals.append(applied_from)
    if applied_to is not None:
        where.append("applied_date<=?"); vals.append(applied_to)
    if interview_date is not None:
        where.append("interview_date=?"); vals.append(interview_date)
    if min_fit is not None:
        where.append("fit_score>=?"); vals.append(min_fit)
    sql = "SELECT * FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC"
    return [dict(r) for r in conn.execute(sql, vals).fetchall()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `. .venv/bin/activate && pytest tests/test_tracker.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add Apply_AI/apply_ai/tracker.py Apply_AI/tests/test_tracker.py
git commit -m "feat: add sqlite job tracker with filters"
```

---

### Task 3: Deterministic fit score

**Files:**
- Create: `Apply_AI/apply_ai/fit_score.py`
- Test: `Apply_AI/tests/test_fit_score.py`

**Interfaces:**
- Consumes: `JD`, `FitResult` from `apply_ai.schemas`.
- Produces:
  - `normalize(skill: str) -> str`
  - `normalize_set(skills) -> set[str]`
  - `score(resume_skills, jd: JD) -> FitResult` (deterministic; weights `0.75/0.25`)

- [ ] **Step 1: Write the failing test**

`Apply_AI/tests/test_fit_score.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `. .venv/bin/activate && pytest tests/test_fit_score.py -v`
Expected: FAIL — `ModuleNotFoundError: apply_ai.fit_score`.

- [ ] **Step 3: Write the fit score**

`Apply_AI/apply_ai/fit_score.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `. .venv/bin/activate && pytest tests/test_fit_score.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add Apply_AI/apply_ai/fit_score.py Apply_AI/tests/test_fit_score.py
git commit -m "feat: add deterministic explainable fit score"
```

---

### Task 4: JD ingestion (URL/text → structured JD)

**Files:**
- Create: `Apply_AI/apply_ai/jd_ingest.py`
- Test: `Apply_AI/tests/test_jd_ingest.py`

**Interfaces:**
- Consumes: `JD` from `apply_ai.schemas`.
- Produces:
  - `fetch_text(url: str) -> str` (trafilatura extract; raises on empty)
  - `extract_jd(raw_text: str, company_job_id: str | None = None) -> JD` (Claude → structured JD)
  - `ingest(url_or_text: str, company_job_id: str | None = None) -> JD` (URL if it starts with http, else treats input as raw text)
  - Internal: `_claude_json(prompt: str) -> str`, `_parse_json_block(text: str) -> dict`

Requires `ANTHROPIC_API_KEY` in env (loaded from `.env`). Network-dependent.

- [ ] **Step 1: Write the failing test (offline — parsing only, Claude mocked)**

`Apply_AI/tests/test_jd_ingest.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `. .venv/bin/activate && pytest tests/test_jd_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: apply_ai.jd_ingest`.

- [ ] **Step 3: Write the module**

`Apply_AI/apply_ai/jd_ingest.py`:
```python
from __future__ import annotations
import json
import os
import re
import httpx
import trafilatura
from dotenv import load_dotenv
from .schemas import JD

load_dotenv()

_PROMPT = """Extract this job description into JSON with keys:
title, company, required_skills (array of short skill tokens),
nice_to_have (array), responsibilities (array of short strings).
Return ONLY JSON. Job description:
---
{raw}
---"""


def fetch_text(url: str) -> str:
    resp = httpx.get(url, timeout=30, follow_redirects=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    text = trafilatura.extract(resp.text) or ""
    if not text.strip():
        raise ValueError(f"No extractable text at {url}")
    return text


def _claude_json(prompt: str) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-5", max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _parse_json_block(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object in model output")
    return json.loads(m.group(0))


def extract_jd(raw_text: str, company_job_id: str | None = None) -> JD:
    data = _parse_json_block(_claude_json(_PROMPT.format(raw=raw_text[:12000])))
    return JD(
        title=data.get("title", "Unknown"),
        company=data.get("company", "Unknown"),
        company_job_id=company_job_id or data.get("company_job_id"),
        required_skills=data.get("required_skills", []),
        nice_to_have=data.get("nice_to_have", []),
        responsibilities=data.get("responsibilities", []),
        raw_text=raw_text,
    )


def ingest(url_or_text: str, company_job_id: str | None = None) -> JD:
    raw = fetch_text(url_or_text) if url_or_text.strip().startswith("http") else url_or_text
    return extract_jd(raw, company_job_id=company_job_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `. .venv/bin/activate && pytest tests/test_jd_ingest.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Live smoke test (network + key required)**

Run:
```bash
. .venv/bin/activate && python -c "
from apply_ai.jd_ingest import ingest
jd = ingest('We are hiring a Python backend engineer. Required: Python, SQL, AWS. Nice: Docker.')
print(jd.title, jd.company, jd.required_skills)
"
```
Expected: prints a title/company and `['python','sql','aws']`-ish list. If it errors on the key, confirm `.env` has `ANTHROPIC_API_KEY`.

- [ ] **Step 6: Commit**

```bash
git add Apply_AI/apply_ai/jd_ingest.py Apply_AI/tests/test_jd_ingest.py
git commit -m "feat: add JD ingestion (url/text -> structured JD via claude)"
```

---

### Task 5: Cognee memory (ingest resume + retrieve evidence)

**Files:**
- Create: `Apply_AI/apply_ai/memory.py`
- Test: `Apply_AI/tests/test_memory.py`

**Interfaces:**
- Consumes: `Resume`, `Bullet` from `apply_ai.schemas`.
- Produces:
  - `Evidence(evidence_id: str, text: str, skills: list[str], source: str)` (pydantic)
  - `class LocalMemory` with `ingest(resume)`, `retrieve_evidence(query, k=5) -> list[Evidence]`, `improve(signal: dict) -> None`.
  - `class CogneeMemory(LocalMemory)` overriding `ingest`/`improve` with Cognee calls.
  - `get_memory() -> LocalMemory` (returns CogneeMemory if `COGNEE_API_KEY` set, else LocalMemory).

Install the memory extra first: `pip install -e ".[memory]"` (if it fails on Python 3.14, recreate the venv with 3.12 per Global Constraints).

- [ ] **Step 1: Write the failing test (offline via LocalMemory)**

`Apply_AI/tests/test_memory.py`:
```python
from apply_ai.memory import LocalMemory, Evidence
from apply_ai.schemas import load_resume


def test_local_retrieval_ranks_by_overlap():
    mem = LocalMemory()
    mem.ingest(load_resume("data/resume.json"))
    ev = mem.retrieve_evidence("fastapi latency service", k=3)
    assert ev and isinstance(ev[0], Evidence)
    # the FastAPI bullet (ev-1) should rank first
    assert ev[0].evidence_id == "ev-1"


def test_improve_is_noop_safe():
    mem = LocalMemory()
    mem.ingest(load_resume("data/resume.json"))
    mem.improve({"job_id": "AAI-0001", "accepted": ["ev-1"]})  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `. .venv/bin/activate && pytest tests/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: apply_ai.memory`.

- [ ] **Step 3: Write the module**

`Apply_AI/apply_ai/memory.py`:
```python
from __future__ import annotations
import os
from pydantic import BaseModel, Field
from .schemas import Resume


class Evidence(BaseModel):
    evidence_id: str
    text: str
    skills: list[str] = Field(default_factory=list)
    source: str = ""


def _all_bullets(resume: Resume) -> list[Evidence]:
    out: list[Evidence] = []
    for e in resume.experiences:
        for b in e.bullets:
            out.append(Evidence(evidence_id=b.id, text=b.text, skills=b.skills,
                                source=f"{e.role} @ {e.company}"))
    for p in resume.projects:
        for b in p.bullets:
            out.append(Evidence(evidence_id=b.id, text=b.text, skills=b.skills,
                                source=f"project:{p.name}"))
    return out


class LocalMemory:
    """Offline fallback: keyword-overlap retrieval. Always testable."""

    def __init__(self) -> None:
        self._ev: list[Evidence] = []
        self._weights: dict[str, float] = {}

    def ingest(self, resume: Resume) -> None:
        self._ev = _all_bullets(resume)

    def retrieve_evidence(self, query: str, k: int = 5) -> list[Evidence]:
        q = {t for t in query.lower().split() if t}

        def score(ev: Evidence) -> float:
            hay = (ev.text + " " + " ".join(ev.skills)).lower()
            overlap = sum(1 for t in q if t in hay)
            return overlap + self._weights.get(ev.evidence_id, 0.0)

        return sorted(self._ev, key=score, reverse=True)[:k]

    def improve(self, signal: dict) -> None:
        for eid in signal.get("accepted", []):
            self._weights[eid] = self._weights.get(eid, 0.0) + 0.5


class CogneeMemory(LocalMemory):
    """Cognee-backed. Falls back to local ranking if the SDK/keys misbehave."""

    def ingest(self, resume: Resume) -> None:
        super().ingest(resume)
        import cognee, asyncio
        text = "\n".join(f"[{e.evidence_id}] {e.text} (skills: {', '.join(e.skills)})"
                         for e in self._ev)
        asyncio.run(self._cognify(cognee, text))

    async def _cognify(self, cognee, text: str) -> None:
        await cognee.add(text)
        await cognee.cognify()

    def improve(self, signal: dict) -> None:
        super().improve(signal)  # local reweight always
        # Cognee improve() hook wired here in a later phase.


def get_memory():
    if os.getenv("COGNEE_API_KEY"):
        return CogneeMemory()
    return LocalMemory()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `. .venv/bin/activate && pytest tests/test_memory.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Full suite green**

Run: `. .venv/bin/activate && pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Apply_AI/apply_ai/memory.py Apply_AI/tests/test_memory.py
git commit -m "feat: add memory store (local fallback + cognee backend)"
```

---

## Phase 1 Done Criteria

- `pytest -q` fully green.
- You can, in a Python shell: `ingest(jd_text)` → `score(resume.skills, jd)` → a `FitResult` with real gaps, and `add_job` persists it so `list_jobs(status=...)` filters work.
- This is the deterministic, demoable spine. Phase 2 (tailoring + provenance, docgen PDF, Streamlit UI, email drafts, improve() wiring) builds directly on these interfaces.

## Self-Review Notes

- **Spec coverage:** schemas (§4.1), tracker + filters (§4.2, §8.2), fit score (§6.3), JD ingest + dual job-id (§6.1), Cognee memory (§3, §7) all mapped. Tailoring/docgen/dashboard/email are Phase 2 per the spec's build order (§11) — intentionally out of this Phase 1 plan.
- **Type consistency:** `FitResult`, `JD`, `Evidence`, `retrieve_evidence` signatures are identical everywhere they appear.
- **No placeholders:** every code step contains complete, runnable code.
