from __future__ import annotations
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
    # check_same_thread=False: Streamlit caches this conn and reuses it across its
    # per-run worker threads; sqlite would otherwise refuse the cross-thread access.
    conn = sqlite3.connect(path, check_same_thread=False)
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


def find_job_by_url(conn: sqlite3.Connection, url: str) -> dict | None:
    """Return the most recent existing job for this URL, so re-pasting the same link
    continues that job instead of creating a duplicate. Empty url never matches."""
    if not url or not url.strip():
        return None
    row = conn.execute(
        "SELECT * FROM jobs WHERE url=? AND url<>'' ORDER BY created_at DESC LIMIT 1",
        (url.strip(),),
    ).fetchone()
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
