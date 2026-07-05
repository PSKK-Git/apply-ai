# Apply_AI — Low-Level Design (MVP)

**Date:** 2026-07-04
**Deadline:** 24-hour end-to-end MVP (hackathon: Hangover_we_make_devs)
**Owner:** sai karthik

---

## 1. Goal

Tailor resume bullets to a pasted/linked Job Description, **grounded in real experience**
(no fabrication), with a **deterministic, explainable fit score** and missing-skill gaps.
A **self-improving loop** (`improve()`) folds accepted edits + application outcomes back into
memory. A minimal, polished **Streamlit dashboard** tracks the whole pipeline. Runs on
**Cognee Cloud** via `serve()`.

## 2. Scope

### In (build in 24h)
1. **Resume tailoring** — JD → grounded bullets + provenance, deterministic fit score, skill gaps, `improve()` loop.
2. **Dashboard** — Streamlit: Add Job, Pipeline (filters), Notifications.
3. **Job-link ingest** — paste URL **or** text → structured JD → `local_job_id` + `company_job_id`.
4. **LaTeX doc generation** — tailored resume JSON → `.tex` → tectonic PDF (local, no external service).
5. **Email (DRAFT-ONLY)** — scan inbox, classify replies (reject/interview/other), surface in
   Notifications, generate Gmail **drafts** for rejection follow-ups. **Never auto-sends.**

### Out (Phase 2 — interfaces stubbed, not built)
- Overleaf MCP adapter (behind `docgen` interface; local `.tex` used until core is stable).
- Cover letter + motivation letter generation (checkbox-gated).
- Interview-question prep + per-company browser input.
- Auto-send of any email.

### Non-negotiable constraints
- **No fabrication:** every tailored bullet must cite the `evidence_id` of the real experience it is grounded in. Bullets without grounding are rejected before render.
- **Deterministic fit score:** the number is pure set math over normalized skills — never an LLM guess.
- **Draft-only email:** the system creates Gmail drafts; a human sends.

## 3. Architecture

```
┌─────────────────────────── Streamlit UI (3 pages) ───────────────────────────┐
│  Add Job          │  Pipeline (filters)        │  Notifications              │
└───────┬───────────┴───────────┬────────────────┴───────────┬─────────────────┘
        │                       │                            │
┌───────▼───────────────────────▼────────────────────────────▼─────────────────┐
│                              Service layer                                    │
│  jd_ingest · fit_score · tailor · docgen · tracker · improve_loop · email_scan│
└───┬──────────────┬───────────────┬───────────────┬───────────────┬───────────┘
    │              │               │               │               │
 Cognee        Anthropic       tectonic         SQLite          Gmail API
 (memory,      Claude          (.tex→PDF)       (tracker,       (read + create
  improve)     (extract,                         filters,        DRAFT only)
               tailor)                           versions)
```

- **Cognee** = knowledge/memory layer: ingests real experience, returns grounding evidence, runs `improve()`.
- **SQLite** = operational store: jobs, resume versions, events. Powers every dashboard filter.
- **tectonic** = local LaTeX→PDF. No network doc service in the MVP.
- **Anthropic Claude** (`claude-sonnet-5`) = JD extraction + bullet tailoring only. Not the fit score.

## 4. Data model

### 4.1 `resume.json` (ground truth — read-only source)
```json
{
  "name": "Sai Karthik",
  "contact": {"email": "...", "phone": "...", "links": ["..."]},
  "skills": ["python", "streamlit", "sql", "..."],
  "experiences": [
    {
      "id": "exp-1",
      "company": "...", "role": "...", "start": "2023-01", "end": "2024-06",
      "bullets": [
        {"id": "ev-1", "text": "Built X that reduced Y by 30%", "skills": ["python","sql"]}
      ]
    }
  ],
  "projects": [ {"id": "proj-1", "name": "...", "bullets": [{"id":"ev-9","text":"...","skills":[...]}]} ],
  "education": [ {"school":"...","degree":"...","year":"..."} ]
}
```
Each bullet `id` is an **`evidence_id`** — the anchor for grounding and provenance.

### 4.2 SQLite schema (`tracker.db`)
```sql
CREATE TABLE jobs (
  local_job_id   TEXT PRIMARY KEY,      -- AAI-0001
  company_job_id TEXT,                  -- as parsed / entered
  company        TEXT,
  title          TEXT,
  url            TEXT,
  jd_json        TEXT,                  -- structured JD
  fit_score      INTEGER,
  fit_breakdown  TEXT,                  -- JSON: matched/missing/points
  status         TEXT,                  -- discovered|tailored|applied|interview|rejected|offer
  applied_date   TEXT,
  interview_date TEXT,
  created_at     TEXT
);

CREATE TABLE resume_versions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  local_job_id  TEXT REFERENCES jobs(local_job_id),
  version       INTEGER,
  tailored_json TEXT,                   -- full tailored resume
  provenance    TEXT,                   -- {bullet_id: [evidence_id,...]}
  skills_delta  TEXT,                   -- skills surfaced vs base resume
  pdf_path      TEXT,
  accepted      INTEGER DEFAULT 0,      -- user accepted this version
  created_at    TEXT
);

CREATE TABLE events (                   -- email replies, status changes, notifications
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  local_job_id  TEXT,
  type          TEXT,                   -- reply|reject|interview|draft_created|status_change
  payload       TEXT,                   -- JSON (snippet, thread_id, draft_id, ...)
  seen          INTEGER DEFAULT 0,
  created_at    TEXT
);
```

### 4.3 Structured JD (stored in `jobs.jd_json`)
```json
{ "title": "...", "company": "...", "company_job_id": "...",
  "required_skills": ["..."], "nice_to_have": ["..."],
  "responsibilities": ["..."], "raw_text": "..." }
```

## 5. Module interfaces (LLD)

```python
# resume_store.py
def load_resume(path="resume.json") -> Resume: ...
def evidence_index(resume) -> dict[str, Bullet]:  # evidence_id -> bullet

# memory.py  (Cognee-backed)
class MemoryStore:
    def ingest(self, resume: Resume) -> None: ...            # cognify real experience
    def retrieve_evidence(self, query: str, k=5) -> list[Evidence]: ...
    def improve(self, signal: ImproveSignal) -> None: ...    # Cognee improve() + reweight

# jd_ingest.py
def ingest(url_or_text: str, company_job_id: str | None) -> JD: ...  # trafilatura + Claude extract
def next_local_job_id(db) -> str: ...                        # AAI-0001, AAI-0002 ...

# fit_score.py  (DETERMINISTIC — no LLM)
def score(resume_skills: set[str], jd: JD) -> FitResult:
    # FitResult: score:int, matched:[...], missing_required:[...], missing_nice:[...], contributions:{...}

# tailor.py
def tailor(resume: Resume, jd: JD, memory: MemoryStore) -> TailorResult:
    # per JD cluster -> retrieve_evidence -> Claude rewrite/select
    # GUARANTEE: every out bullet carries provenance=[evidence_id]; ungrounded bullets dropped
    # returns tailored_resume, diff, provenance, skills_delta

# docgen.py
class DocGen(Protocol):
    def build_pdf(self, tailored: Resume) -> str: ...        # returns pdf path
class LocalTexDocGen:  ...                                   # Jinja2 .tex -> tectonic  (MVP)
class OverleafDocGen:  ...                                   # Phase 2 stub, same interface

# tracker.py
def add_job(db, jd, fit) -> str
def save_version(db, job_id, tailored, provenance, skills_delta, pdf_path) -> int
def list_jobs(db, *, status=None, applied_from=None, applied_to=None,
              interview_date=None, min_fit=None) -> list[Job]     # powers filters
def set_status(db, job_id, status, **dates) -> None

# improve_loop.py
def on_edit_decision(memory, job_id, bullet_id, decision) -> None  # accept|edit|reject
def on_outcome(memory, job_id, outcome) -> None                    # applied|interview|reject|offer

# email_scan.py  (DRAFT-ONLY)
def scan(db, jobs) -> list[Event]:      # search Gmail for replies to tracked companies
def classify(thread) -> str:            # Claude -> reject|interview|other
def draft_followup(thread, job) -> str: # create Gmail DRAFT (why-not-shortlisted); returns draft_id
                                        # NEVER calls messages.send
```

## 6. Key flows

### 6.1 Add & tailor a job
1. User pastes URL/text (+ optional company job-id) on **Add Job**.
2. `jd_ingest.ingest` → structured JD; `next_local_job_id` → `AAI-000N`; `tracker.add_job`.
3. `fit_score.score(resume_skills, jd)` → score + gaps → **explainability card**.
4. User clicks *Tailor* → `tailor.tailor` retrieves real evidence from Cognee, Claude rewrites
   bullets, attaches provenance, computes **skills_delta** (surfaced vs base).
5. UI shows tailored bullets + provenance + diff. User accepts/edits/rejects per bullet.
6. `docgen.build_pdf` → PDF; `tracker.save_version` stores the version (history preserved).
7. Accept/reject decisions → `improve_loop.on_edit_decision` → `memory.improve`.

### 6.2 Email (draft-only), run on demand / daily
1. **Notifications** page "Scan inbox" (and optional daily schedule) → `email_scan.scan`.
2. Each new reply → `classify` → write `events` row (reply/reject/interview).
3. Interview → set `jobs.interview_date` (feeds "tomorrow's interviews").
4. Rejection → `draft_followup` creates a Gmail **draft** asking for feedback; `events` row
   `draft_created` with `draft_id`. UI shows "Draft ready — review & send." No auto-send.

### 6.3 Fit score (deterministic)
```
norm(s)      = lowercase + alias_map   (e.g. "js" -> "javascript")
required_hit = |R ∩ J_req| / |J_req|
nice_hit     = |R ∩ J_nice| / |J_nice|   (0 if J_nice empty)
score        = round(100 * (0.75*required_hit + 0.25*nice_hit))
gaps         = J_req \ R   (ranked)  +  J_nice \ R
```
Every matched skill contributes explicit points shown in the breakdown card.

## 7. `improve()` loop
- **Signals:** per-bullet accept/edit/reject; job outcome (applied→interview→reject/offer).
- **Action:** `memory.improve(signal)` (Cognee) + local reweight of evidence retrieval so future
  tailoring favors evidence that produced accepted bullets / interviews.

## 8. Dashboard pages
1. **Add Job** — ingest → fit card → tailor → provenance/diff → accept → build PDF → save.
   Includes **Skills-delta panel** (what this JD surfaced vs base) and version history.
2. **Pipeline** — table of all jobs; filters: **status, applied-date range, interview date, min fit**;
   inline status edit; shows both job-ids; counts (applied / in-progress).
3. **Notifications** — *tomorrow's interviews* (from `interview_date`) + *today's replies*
   (from `events`) + rejection drafts pending review.

## 9. Tech stack
Python 3.11 · Streamlit · Cognee (primary memory, `serve()`) · Anthropic Claude `claude-sonnet-5`
· SQLite · Jinja2 + tectonic · trafilatura + httpx · Gmail API (read + drafts) · pydantic.

## 10. Repo layout
```
Apply_AI/
  app.py                      # streamlit entry / serve()
  apply_ai/
    resume_store.py  memory.py  jd_ingest.py  fit_score.py
    tailor.py  docgen.py  tracker.py  improve_loop.py  email_scan.py
    schemas.py                # pydantic models
    templates/resume.tex.j2
  data/resume.json  data/tracker.db
  tests/                      # fit_score (deterministic) + provenance guarantee
  docs/specs/2026-07-04-apply-ai-design.md
```

## 11. 24-hour build order
| Window | Deliverable |
|--------|-------------|
| H0–2   | Scaffold, pydantic schemas, SQLite tracker, Cognee ingest of `resume.json` |
| H2–5   | `jd_ingest` (URL + paste, Claude extract, dual job-id) |
| H5–8   | `fit_score` (deterministic) + gaps + explainability card |
| H8–14  | `tailor` (Cognee-grounded, provenance, no-fabrication guard) + diff + skills-delta |
| H14–17 | `docgen` LocalTexDocGen (.tex → tectonic PDF) + version history |
| H17–20 | Streamlit 3 pages + filters |
| H20–22 | `email_scan` draft-only + Notifications feed |
| H22–23 | `improve()` wiring + `serve()` deploy |
| H23–24 | Seed demo data + full dry-run |

## 12. Testing (minimum)
- `fit_score`: table tests — exact scores for known skill sets (deterministic, must be stable).
- `tailor`: provenance guarantee — assert no rendered bullet lacks an `evidence_id`.
- `email_scan`: assert `send` is never called; only `drafts.create`.
