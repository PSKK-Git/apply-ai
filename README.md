# Apply_AI

Tailor your resume to a job description — grounded in your **real** experience,
never fabricated. Built for the *Hangover_we_make_devs* hackathon.

Paste a job description (or its URL), and Apply_AI:

1. extracts the JD into structured requirements,
2. scores a deterministic **fit %** and an **ATS keyword-coverage %** against your résumé,
3. **tailors your real bullets** to the JD — every bullet must cite the real evidence
   it's grounded in; anything ungrounded is dropped, never invented,
4. shows **skill gaps** and lets you add real proof for ones you actually have
   (never auto-invents a skill for you),
5. optionally **cross-verifies** the tailoring with a second/third model,
6. exports a clean **PDF résumé**,
7. tracks the job in a local pipeline (status, fit, applied/interview dates).

---

## How this project uses Cognee

Cognee is the **knowledge/memory spine** — the thing that lets tailoring cite
*real* experience instead of an LLM guessing a plausible-sounding bullet.

An earlier attempt used Cognee's **embedded** graph backend, which doesn't work on
macOS out of the box (its `kuzu`/`ladybug` provider needs a native shared library
that isn't shipped). So this app talks to **Cognee Cloud over its REST API**
instead (`apply_ai/cognee_rest.py`, class `CogneeRestMemory`) — no local graph DB
required.

Cognee is used in three places:

- **Ingest** — when you load your résumé (My Resume page), every bullet is tagged
  `[evidence_id] text (skills: …; source: …)` and uploaded via `POST /api/v1/remember`
  into a dataset called `apply_ai_resume`. This is what builds your persistent
  knowledge graph of real experience.
- **Retrieve** — at tailor time, `tailor.py` asks memory for the evidence most
  relevant to the JD. Against Cognee this is `POST /api/v1/recall` with
  `searchType=CHUNKS` — chunk search returns raw retrieved text (still carrying the
  `[evidence_id]` tags) rather than an LLM summary that would strip them. A small
  regex recovers the real `Evidence` objects from those tags. This tag round-trip
  is *the* mechanism that ties Cognee's semantic search back to citable evidence,
  which is what makes the no-fabrication guarantee possible: a tailored bullet is
  only kept if it cites a real, retrieved `evidence_id`.
- **Improve** — hosted Cognee self-improves server-side during `remember(...,
  self_improvement=True)`; there's nothing extra the client needs to push today.

If the graph isn't cognified yet, or the tenant is briefly unreachable, retrieval
**falls back to local keyword ranking** over the same evidence — so a Cognee hiccup
degrades gracefully instead of breaking tailoring.

`apply_ai/memory.py` also ships a `LocalMemory` (pure keyword ranking, no network —
used by the test suite and as an explicit offline mode) and a `CogneeMemory`
(the embedded-SDK path, kept only for reference since it's broken on macOS).

---

## Setup

### 1. Environment

```bash
git clone https://github.com/PSKK-Git/apply-ai.git
cd apply-ai
python3.12 -m venv .venv   # 3.11–3.13; cognee/ML deps may lack wheels on 3.14
source .venv/bin/activate
pip install -e ".[dev]"
pip install streamlit openai google-generativeai pypdf reportlab   # not yet in pyproject
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Then fill in `.env`:

- **A generator LLM** (`APPLY_AI_GEN_PROVIDER` + its key) — pick whichever you have
  quota for: `gemini`, `claude`, `mistral`, `openai`, `ollama` (free/local), or
  `openrouter`/`custom` (any OpenAI-compatible gateway, incl. free-tier models).
  All of this is also changeable **live from the app's sidebar** — no restart needed.
- **Cognee Cloud** — `COGNEE_BASE_URL` + `COGNEE_API_KEY` from your Cognee tenant.
  If your resolver refuses the tenant hostname, set `COGNEE_RESOLVE_IP` to its
  resolved IP (found via `nslookup <host> 1.1.1.1`) — the app pins DNS while
  keeping the hostname for TLS. Leaving Cognee unset, or setting
  `APPLY_AI_MEMORY=local`, runs fully offline on keyword matching instead.

### 3. Add your résumé

```bash
streamlit run app.py
```

Open **My Resume** → upload a PDF or paste your résumé text → *Use this resume*.
It's parsed into structured evidence, synced to Cognee, and saved to
`data/resume.json` — it persists across restarts, so you only do this once
(re-open the page any time and hit *Continue with this résumé*).

### 4. Run the tests

```bash
pytest -q       # 32 passed, 1 skipped (a live-Cognee test gated behind an env flag)
```

---

## Getting the most out of it

- **Keep your evidence rich.** Tailoring can only cite what's in your résumé's
  bullets. The more specific, quantified real bullets you have (metrics, tech,
  scale), the more the ATS/fit scores and tailored output actually reflect you.
- **Use the gap-fill loop.** After tailoring, the ATS panel lists JD keywords
  you're missing. Groundable ones (already in your evidence, just not surfaced)
  show as chips. For the rest, add a short *real* bullet only if it's true — it
  becomes new evidence, gets pushed to Cognee, and the next tailor pass can use
  it. This is the fastest way to raise your ATS score honestly.
- **Cross-verify before you trust a big rewrite.** The verify step runs your
  tailored bullets past a second/third model as an independent auditor — it flags
  bullets more than one model doubts and estimates ATS alignment. Treat a
  cross-model consensus flag as a signal to double-check that bullet yourself.
- **Re-paste the same job URL to resume, not duplicate.** The tracker matches on
  URL — pasting a link you've already added continues that job (and its tailored
  version) instead of creating `AAI-000N+1`.
- **Free LLM tiers are rate-limited by nature.** Gemini's free tier is ~20
  requests/day; free OpenRouter models get throttled upstream under load. If
  tailoring feels slow or 429s, switch providers in the sidebar — Ollama (local,
  unlimited) or a few dollars of OpenRouter credit on a non-`:free` model are the
  most reliable paths.
- **The PDF is generated fresh each time**, from whatever bullets are currently
  tailored — download after every JD you care about, or hit *Accept & save this
  version* to also persist it to `data/<job_id>_v<n>.pdf` alongside that job.

---

## Project layout

```
app.py                  # Streamlit UI (My Resume / Add Job / Pipeline / Notifications)
apply_ai/
  schemas.py             # pydantic models (Resume, JD, FitResult, TailorResult, ATSReport, ...)
  resume_ingest.py        # PDF/text -> structured Resume (LLM-parsed, evidence-tagged)
  jd_ingest.py            # URL/text -> structured JD (with a JS-page reader fallback)
  fit_score.py            # deterministic skill-overlap score (no LLM)
  ats.py                  # deterministic ATS keyword coverage + gap list
  tailor.py                # JD-grounded tailoring with the no-fabrication guard
  memory.py / cognee_rest.py   # LocalMemory, CogneeMemory (embedded, broken), CogneeRestMemory
  llm.py                    # provider-agnostic generator (gemini/claude/mistral/openai/ollama/custom)
  verify.py                  # multi-model cross-verification
  docgen.py                   # tailored resume -> PDF (reportlab)
  tracker.py                    # SQLite job tracker (+ URL dedupe)
tests/                          # 32 tests, all offline/mocked except one gated live-Cognee test
docs/specs/, docs/plans/         # original design doc + Phase-1 implementation plan
```
