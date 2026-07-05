"""Apply_AI — Streamlit dashboard (Phase 2).

Three pages wiring the pipeline: Add Job (ingest -> fit -> tailor -> ATS ->
cross-verify -> save), Pipeline (filterable tracker), Notifications.

Grounding is the product's spine: tailored bullets only ever come from real
retrieved evidence, and ATS gaps that can't be grounded are offered back to you
to add in your own words (which then becomes new evidence) — never auto-invented.

Run: `streamlit run app.py`  (uses local memory by default; toggle in the sidebar).
"""
from __future__ import annotations
import json
import os
from datetime import date, timedelta

import streamlit as st
from dotenv import load_dotenv

from apply_ai import ats as ats_mod
from apply_ai import docgen, fit_score, jd_ingest, resume_ingest, tailor as tailor_mod, tracker, verify
from apply_ai.memory import Evidence, LocalMemory, _all_bullets, get_memory
from apply_ai.schemas import Bullet, FitResult, JD, Resume, load_resume

load_dotenv()
st.set_page_config(page_title="Apply_AI", page_icon="🎯", layout="wide")

_APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@300;400;500;600&display=swap');
:root{--bg:#130f24;--bg2:#181334;--violet:#8b5cf6;--violet2:#c4b5fd;
      --emerald:#34d399;--text:#eceafc;--muted:#a49dc8;--line:rgba(196,181,253,.13);}
.stApp{background:linear-gradient(180deg,#130f24 0%,#171130 100%);color:var(--text);
  font-family:'Inter',sans-serif;}
.block-container{padding-top:2.6rem;max-width:1100px;}
/* royal serif headings — solid, restrained, no glow */
h1,h2,h3{font-family:'Fraunces',serif!important;color:#f4f1ff!important;font-weight:600!important;
  letter-spacing:.2px;-webkit-text-fill-color:initial;}
h1{font-size:2.3rem!important;}
h1::after{content:"";display:block;width:52px;height:2px;margin-top:.5rem;
  background:var(--emerald);border-radius:2px;opacity:.9;}
.stCaption,[data-testid="stCaptionContainer"]{color:var(--muted)!important;}
/* sidebar: quiet, one hairline */
[data-testid="stSidebar"]{background:#100c20;border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] h1{font-size:1.5rem!important;}
[data-testid="stSidebar"] h1::after{display:none;}
/* secondary buttons: outline, minimal. primary: single emerald accent */
.stButton>button,.stFormSubmitButton>button{background:transparent;color:var(--violet2);
  border:1px solid var(--line);border-radius:10px;padding:.5rem 1.1rem;font-weight:500;
  box-shadow:none;transition:border-color .15s ease,color .15s ease,background .15s ease;}
.stButton>button:hover,.stFormSubmitButton>button:hover{border-color:var(--violet);
  color:#fff;background:rgba(139,92,246,.10);}
[data-testid="stBaseButton-primary"]{background:var(--emerald)!important;color:#06251a!important;
  border:0!important;font-weight:600!important;box-shadow:none!important;}
[data-testid="stBaseButton-primary"]:hover{background:#57e0b0!important;}
/* inputs: hairline, no glow */
textarea,input,.stTextInput input,.stTextArea textarea,div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{
  background:rgba(255,255,255,.025)!important;border:1px solid var(--line)!important;
  border-radius:10px!important;color:var(--text)!important;}
textarea:focus,input:focus{border-color:var(--violet)!important;box-shadow:none!important;}
/* metric + form + expander: subtle elevated panels, one hairline each */
[data-testid="stMetric"]{background:rgba(255,255,255,.028);border:1px solid var(--line);
  border-radius:12px;padding:14px 18px;}
[data-testid="stMetricValue"]{color:var(--emerald)!important;font-family:'Fraunces',serif!important;}
[data-testid="stExpander"],div[data-testid="stForm"]{background:rgba(255,255,255,.02)!important;
  border:1px solid var(--line)!important;border-radius:12px!important;}
.stProgress>div>div>div{background:var(--emerald)!important;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{color:var(--violet2)!important;}
a{color:var(--emerald)!important;text-decoration:none;}
hr{border-color:var(--line)!important;}
::-webkit-scrollbar{width:9px}::-webkit-scrollbar-thumb{background:rgba(139,92,246,.35);border-radius:8px}
::-webkit-scrollbar-track{background:transparent}
</style>
"""
st.markdown(_APP_CSS, unsafe_allow_html=True)

DB_PATH = os.getenv("APPLY_AI_DB", "data/tracker.db")
RESUME_PATH = os.getenv("APPLY_AI_RESUME", "data/resume.json")
STATUSES = ["discovered", "tailored", "applied", "interview", "rejected", "offer"]


# --------------------------------------------------------------------------- state
@st.cache_resource
def _db():
    return tracker.init_db(DB_PATH)


def _resume() -> Resume:
    # not cached: gap-fill mutates the in-session resume copy
    if "resume" not in st.session_state:
        st.session_state.resume = load_resume(RESUME_PATH)
    return st.session_state.resume


def _memory(use_local: bool):
    if use_local:
        mem = LocalMemory()
        mem.ingest(_resume())
        return mem
    # Cognee knowledge spine over REST (hosted tenant). The résumé is pushed to the
    # graph on load; here we only rebuild the provenance index for tag recovery.
    from apply_ai.cognee_rest import CogneeRestMemory
    mem = CogneeRestMemory()
    mem.load_index(_resume())
    return mem


def _corpus() -> list[Evidence]:
    return _all_bullets(_resume())


def _init_session():
    st.session_state.setdefault("jd", None)
    st.session_state.setdefault("fit", None)
    st.session_state.setdefault("tailored", None)
    st.session_state.setdefault("ats", None)
    st.session_state.setdefault("verify", None)
    st.session_state.setdefault("job_id", None)


# ----------------------------------------------------------------------- add job
def page_add_job(conn, use_local):
    st.header("Add Job")
    col_in, col_meta = st.columns([3, 1])
    with col_in:
        raw = st.text_area("Job description (paste text or a URL)", height=180,
                           placeholder="Paste the JD, or a job posting URL…")
    with col_meta:
        company_job_id = st.text_input("Company job id (optional)")

    if st.button("Ingest & score", type="primary", disabled=not raw.strip()):
        url = raw.strip() if raw.strip().startswith("http") else ""
        existing = tracker.find_job_by_url(conn, url)
        if existing:
            # Same URL as before -> continue that job, don't create a duplicate.
            st.info(f"↩︎ You've added this link before — continuing from "
                    f"**{existing['local_job_id']}** ({existing['title']} · {existing['company']}). "
                    "No duplicate created.")
            st.session_state.jd = JD.model_validate_json(existing["jd_json"])
            st.session_state.fit = FitResult.model_validate_json(existing["fit_breakdown"])
            st.session_state.job_id = existing["local_job_id"]
            st.session_state.tailored = st.session_state.ats = st.session_state.verify = None
        else:
            with st.spinner("Extracting JD…"):
                try:
                    jd = jd_ingest.ingest(raw.strip(), company_job_id or None)
                except Exception as exc:
                    st.error(f"JD ingest failed: {exc}")
                    return
            fit = fit_score.score(_resume().skills, jd)
            st.session_state.jd = jd
            st.session_state.fit = fit
            st.session_state.tailored = st.session_state.ats = st.session_state.verify = None
            st.session_state.job_id = tracker.add_job(conn, jd, fit, url=url)

    jd: JD = st.session_state.jd
    if not jd:
        return

    st.subheader(f"{jd.title} — {jd.company}  ·  `{st.session_state.job_id}`")
    fit = st.session_state.fit
    c1, c2, c3 = st.columns(3)
    c1.metric("Fit score", f"{fit.score}%")
    c2.metric("Matched", len(fit.matched))
    c3.metric("Missing required", len(fit.missing_required))
    with st.expander("Fit breakdown"):
        st.write("**Matched:**", ", ".join(fit.matched) or "—")
        st.write("**Missing required:**", ", ".join(fit.missing_required) or "—")
        st.write("**Missing nice-to-have:**", ", ".join(fit.missing_nice) or "—")
        st.json(fit.contributions)

    st.divider()
    if st.button("Tailor to this JD", type="primary"):
        with st.spinner("Grounding + tailoring with Claude…"):
            try:
                mem = _memory(use_local)
                tailored = tailor_mod.tailor(_resume(), jd, mem)
                st.session_state.tailored = tailored
                st.session_state.ats = ats_mod.ats_report(jd, tailored, _resume(), corpus=_corpus())
                st.session_state.verify = None
            except Exception as exc:
                st.error(f"Tailoring failed: {exc}")

    _render_tailored(conn, jd, use_local)


def _render_tailored(conn, jd, use_local):
    tailored = st.session_state.tailored
    ats = st.session_state.ats
    if not tailored:
        return

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Tailored bullets (grounded)")
        if not tailored.bullets:
            st.warning("No bullet could be grounded in your evidence for this JD.")
        for b in tailored.bullets:
            st.markdown(f"- {b.text}  \n  <span style='color:gray'>provenance: "
                        f"{', '.join(b.provenance)}</span>", unsafe_allow_html=True)
        if tailored.dropped:
            with st.expander(f"Dropped {len(tailored.dropped)} ungrounded suggestion(s)"):
                for d in tailored.dropped:
                    st.write("•", d)
        if tailored.skills_delta:
            st.info("**Skills surfaced for this JD:** " + ", ".join(tailored.skills_delta))

    with right:
        st.subheader("ATS")
        st.metric("ATS keyword coverage", f"{ats.score}%")
        st.progress(ats.score / 100)
        st.caption("Matched: " + (", ".join(ats.matched_keywords) or "—"))

    _render_gaps(ats, jd, use_local)   # full-width section, below the two columns

    st.divider()
    _render_verify(jd, tailored, ats)

    if tailored.bullets:
        st.divider()
        st.subheader("Final résumé")
        st.caption("Your base résumé with the grounded, JD-tailored bullets — as a real PDF.")
        try:
            pdf = docgen.build_pdf_bytes(_resume(), tailored)
            fname = f"{_resume().name.replace(' ', '_')}_{jd.company.replace(' ', '_')}.pdf"
            st.download_button("📄  Download résumé (PDF)", data=pdf, file_name=fname,
                               mime="application/pdf", type="primary")
        except Exception as exc:
            st.error(f"PDF build failed: {exc}")
        if st.button("Accept & save this version"):
            path = _save_version(conn, tailored)
            tracker.set_status(conn, st.session_state.job_id, "tailored")
            st.success(f"Saved {st.session_state.job_id} → tailored"
                       + (f" · PDF written to `{path}`" if path else "") + ".")


def _render_gaps(ats, jd, use_local):
    if not ats.gaps:
        st.success("✓ No ATS gaps — every JD keyword is covered by your résumé.")
        return
    groundable = [g for g in ats.gaps if g.groundable]
    missing = [g for g in ats.gaps if not g.groundable]

    st.markdown("###### Skill gaps &nbsp;<span style='font-weight:400;color:#a49dc8;"
                "font-size:.78em'>· add-if-true, never auto-invented</span>",
                unsafe_allow_html=True)

    if groundable:
        chips = " ".join(
            "<span style='background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.4);"
            "color:#8ef0c4;padding:2px 10px;border-radius:20px;margin:2px 5px 2px 0;"
            f"display:inline-block;font-size:.8em'>✓ {g.keyword}</span>" for g in groundable)
        st.markdown("<div style='color:#a49dc8;font-size:.8em;margin-bottom:3px'>"
                    "In your experience — surfaces on re-tailor:</div>" + chips,
                    unsafe_allow_html=True)

    if missing:
        with st.expander(f"➕  Add real evidence for {len(missing)} missing skill(s) — optional"):
            st.caption("Add a bullet only if it's true; it becomes grounded evidence and re-tailors.")
            for g in missing:
                with st.form(f"gap-{g.keyword}", clear_on_submit=True):
                    c1, c2, c3 = st.columns([1.5, 4, 1])
                    c1.markdown(f"<div style='padding-top:8px'>🔴 <b>{g.keyword}</b></div>",
                                unsafe_allow_html=True)
                    txt = c2.text_input(
                        f"Real evidence for {g.keyword}", key=f"in-{g.keyword}",
                        label_visibility="collapsed", placeholder=f"what you did with {g.keyword}…")
                    if c3.form_submit_button("Add") and txt.strip():
                        _add_evidence(txt.strip(), g.keyword)
                        st.session_state.tailored = None
                        st.rerun()


def _add_evidence(text: str, keyword: str):
    """User-supplied real experience becomes new grounded evidence on the resume."""
    r = _resume()
    new_id = f"ev-user-{len(_all_bullets(r)) + 1}"
    if not r.projects:
        from apply_ai.schemas import Project
        r.projects.append(Project(id="proj-user", name="Additional experience"))
    r.projects[-1].bullets.append(Bullet(id=new_id, text=text, skills=[keyword]))
    if keyword not in r.skills:
        r.skills.append(keyword)


def _render_verify(jd, tailored, ats):
    avail = list(verify.available_providers())
    st.subheader("Cross-verification")
    st.caption("Available verifiers: " + (", ".join(avail) or "none") +
               ("" if "gemini" in avail else "  ·  add GEMINI_API_KEY to enable Gemini"))
    if st.button("Cross-verify with other models", disabled=not avail):
        with st.spinner("Auditing with " + ", ".join(avail) + "…"):
            st.session_state.verify = verify.cross_verify(jd, tailored, ats)
    rep = st.session_state.verify
    if not rep:
        return
    if rep.avg_ats_alignment is not None:
        st.metric("Avg ATS alignment (models)", f"{rep.avg_ats_alignment}%")
    if rep.consensus_flags:
        st.warning("Bullets doubted by >1 model: " + ", ".join(rep.consensus_flags))
    for v in rep.verdicts:
        with st.expander(f"{v.provider} {'⚠️ ' + v.error if v.error else ''}"):
            if v.error:
                st.write(v.error)
                continue
            st.write("ATS alignment:", v.ats_alignment)
            st.write("Flags:", ", ".join(v.fabrication_flags) or "—")
            for s in v.suggestions:
                st.write("💡", s)


def _save_version(conn, tailored):
    jid = st.session_state.job_id
    n = conn.execute("SELECT COUNT(*) c FROM resume_versions WHERE local_job_id=?",
                     (jid,)).fetchone()["c"]
    pdf_path = f"data/{jid}_v{n + 1}.pdf"
    try:
        docgen.build_pdf(_resume(), tailored, pdf_path)
    except Exception:
        pdf_path = ""
    conn.execute(
        """INSERT INTO resume_versions
             (local_job_id, version, tailored_json, provenance, skills_delta, pdf_path, accepted, created_at)
           VALUES (?,?,?,?,?,?,1,?)""",
        (jid, n + 1, tailored.model_dump_json(), json.dumps(tailored.provenance),
         json.dumps(tailored.skills_delta), pdf_path, tracker._now()),
    )
    conn.commit()
    return pdf_path


# ---------------------------------------------------------------------- pipeline
def page_pipeline(conn):
    st.header("Pipeline")

    with st.expander("➕ Log a job you've applied to (manual — no AI needed)", expanded=False):
        with st.form("manual_job", clear_on_submit=True):
            m1, m2 = st.columns(2)
            company = m1.text_input("Company *")
            title = m2.text_input("Title *")
            m3, m4 = st.columns(2)
            cid = m3.text_input("Company job id")
            url = m4.text_input("Job URL")
            m5, m6, m7 = st.columns(3)
            mstatus = m5.selectbox("Status", STATUSES, index=STATUSES.index("applied"))
            applied = m6.text_input("Applied date", value=date.today().isoformat())
            interview = m7.text_input("Interview date (YYYY-MM-DD)")
            if st.form_submit_button("Add to tracker") and company.strip() and title.strip():
                jd = JD(title=title.strip(), company=company.strip(), company_job_id=cid.strip() or None)
                jid = tracker.add_job(conn, jd, FitResult(score=0), url=url.strip())
                tracker.set_status(conn, jid, mstatus,
                                   applied_date=applied.strip() or None,
                                   interview_date=interview.strip() or None)
                st.success(f"Added {jid}: {title} @ {company} — {mstatus}")
                st.rerun()

    f1, f2, f3, f4 = st.columns(4)
    status = f1.selectbox("Status", ["(any)"] + STATUSES)
    min_fit = f2.slider("Min fit", 0, 100, 0)
    applied_from = f3.text_input("Applied from (YYYY-MM-DD)")
    applied_to = f4.text_input("Applied to (YYYY-MM-DD)")

    rows = tracker.list_jobs(
        conn,
        status=None if status == "(any)" else status,
        min_fit=min_fit or None,
        applied_from=applied_from or None,
        applied_to=applied_to or None,
    )
    st.caption(f"{len(rows)} job(s)")
    for r in rows:
        c = st.columns([1.2, 3, 1, 1.4, 1.4])
        c[0].write(f"`{r['local_job_id']}`")
        c[1].write(f"**{r['title']}** — {r['company']}  ·  {r['company_job_id'] or ''}")
        c[2].write(f"{r['fit_score']}%")
        new = c[3].selectbox("status", STATUSES, index=STATUSES.index(r["status"]),
                             key=f"st-{r['local_job_id']}", label_visibility="collapsed")
        if new != r["status"]:
            extra = {}
            if new == "applied":
                extra["applied_date"] = date.today().isoformat()
            tracker.set_status(conn, r["local_job_id"], new, **extra)
            st.rerun()
        c[4].write(r["applied_date"] or "")


# ------------------------------------------------------------------ notifications
def page_notifications(conn):
    st.header("Notifications")
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    st.subheader("Tomorrow's interviews")
    ivs = tracker.list_jobs(conn, interview_date=tomorrow)
    if ivs:
        for r in ivs:
            st.write(f"📅 `{r['local_job_id']}` {r['title']} — {r['company']}")
    else:
        st.caption("None scheduled.")

    st.subheader("Recent replies")
    events = conn.execute(
        "SELECT * FROM events WHERE type IN ('reply','reject','interview') "
        "ORDER BY created_at DESC LIMIT 20").fetchall()
    if events:
        for e in events:
            st.write(f"✉️ {e['type']} · {e['local_job_id']} · {e['payload']}")
    else:
        st.caption("No replies yet. (Email scan is a later Phase-2 module — draft-only.)")


# ------------------------------------------------------------------------- main
def page_resume():
    st.header("① My Resume")
    _r0 = _resume()
    _is_sample = (_r0.name == "Sai Karthik"
                  and any(e.company == "Acme Labs" for e in _r0.experiences))
    if not _is_sample:
        st.success(f"✓ Your résumé is saved & loaded: **{_r0.name}** — "
                   f"{len(_corpus())} bullets · {len(_r0.skills)} skills. "
                   "Reused every session — you never need to re-upload.")
        if st.button("Continue with this résumé  →", type="primary"):
            st.session_state.resume = load_resume(RESUME_PATH)
            st.info("Locked in. Open **Add Job** in the sidebar to tailor against it.")
        st.caption("Upload below only if you want to **replace** your saved résumé.")
    else:
        st.caption("Upload or paste YOUR resume — fit, tailoring and ATS are all built on this base.")
    up = st.file_uploader("Upload (PDF / TXT / MD / JSON)", type=["pdf", "txt", "md", "json"])
    txt = st.text_area("…or paste your resume text", height=200)
    if st.button("Use this resume", type="primary"):
        try:
            raw = resume_ingest.extract_text(up) if up is not None else txt
        except Exception as exc:
            st.error(f"Could not read file: {exc}"); return
        if not (raw or "").strip():
            st.warning("Upload a file or paste some text first."); return
        with st.spinner("Parsing your resume…"):
            try:
                r = resume_ingest.parse_resume(raw)
            except Exception as exc:
                st.error(f"Parse failed: {exc}"); return
        st.session_state.resume = r
        with open(RESUME_PATH, "w", encoding="utf-8") as fh:
            fh.write(r.model_dump_json(indent=2))
        st.success(f"Loaded **{r.name}** — {len(_corpus())} evidence bullets, "
                   f"{len(r.skills)} skills. Saved & reused automatically.")
        if os.getenv("COGNEE_BASE_URL"):
            with st.spinner("Syncing to the Cognee knowledge graph…"):
                try:
                    from apply_ai.cognee_rest import CogneeRestMemory
                    CogneeRestMemory().ingest(r)
                    st.caption("🧠 Synced to Cognee (dataset `apply_ai_resume`).")
                except Exception as exc:
                    st.caption(f"Cognee sync skipped: {str(exc)[:90]}")

    r = _resume()
    st.divider()
    st.subheader(f"Current base résumé — {r.name}")
    if r.name == "Sai Karthik" and any(e.company == "Acme Labs" for e in r.experiences):
        st.warning("⚠️ This is the **sample** resume. Load yours above — until then, all "
                   "scores/tailoring compare jobs against this placeholder.")
    else:
        st.success("✓ Using your saved résumé — persisted and reused automatically. "
                   "No need to re-upload; replace it above only if you want to.")
    st.write("**Skills:**", ", ".join(r.skills) or "—")
    for e in r.experiences:
        st.markdown(f"**{e.role} @ {e.company}**")
        for b in e.bullets:
            st.write("•", b.text)
    for p in r.projects:
        st.markdown(f"**project: {p.name}**")
        for b in p.bullets:
            st.write("•", b.text)


def main():
    _init_session()
    conn = _db()
    st.sidebar.title("🎯 Apply_AI")
    page = st.sidebar.radio("", ["My Resume", "Add Job", "Pipeline", "Notifications"])
    use_local = st.sidebar.toggle(
        "Use local memory instead of Cognee",
        value=os.getenv("APPLY_AI_MEMORY", "").lower() == "local",
        help="Default is the Cognee knowledge graph. Toggle on only for offline keyword mode.")
    st.sidebar.caption(f"Memory: **{'Local keyword' if use_local else '🧠 Cognee graph'}**")
    st.sidebar.caption(f"Résumé: {_resume().name} · {len(_corpus())} bullets")

    st.sidebar.divider()
    st.sidebar.markdown("**LLM** · JD parsing + tailoring")
    provs = ["openrouter", "gemini", "claude", "mistral", "openai", "ollama", "custom"]
    cur = os.getenv("APPLY_AI_GEN_PROVIDER", "gemini").lower()
    prov = st.sidebar.selectbox("Provider", provs,
                                index=provs.index(cur) if cur in provs else 0,
                                label_visibility="collapsed")
    os.environ["APPLY_AI_GEN_PROVIDER"] = prov
    os.environ.pop("APPLY_AI_GEN_MODEL", None)   # reset; each provider uses its own default
    _keymap = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY",
               "mistral": "MISTRAL_API_KEY", "openai": "OPENAI_API_KEY"}
    if prov == "ollama":
        os.environ["OLLAMA_BASE_URL"] = st.sidebar.text_input(
            "Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        os.environ["APPLY_AI_GEN_MODEL"] = st.sidebar.text_input("Model", value="llama3.1")
        st.sidebar.caption("🆓 Free / local — no key needed.")
    elif prov in ("custom", "openrouter"):
        _dbase = "https://openrouter.ai/api/v1" if prov == "openrouter" else ""
        _dmodel = "meta-llama/llama-3.3-70b-instruct:free" if prov == "openrouter" else ""
        os.environ["APPLY_AI_CUSTOM_BASE_URL"] = st.sidebar.text_input(
            "Base URL (OpenAI-compatible)",
            value=os.getenv("APPLY_AI_CUSTOM_BASE_URL", "") or _dbase,
            placeholder="https://your-gateway/v1")
        ck = st.sidebar.text_input("API key", type="password",
                                   placeholder="sk-or-… (blank = use .env)")
        if ck.strip():
            os.environ["APPLY_AI_CUSTOM_KEY"] = ck.strip()
        os.environ["APPLY_AI_GEN_MODEL"] = st.sidebar.text_input(
            "Model (TEXT, not image)",
            value=os.getenv("APPLY_AI_GEN_MODEL", "") or _dmodel,
            placeholder="e.g. meta-llama/llama-3.3-70b-instruct:free")
        st.sidebar.caption(("🔑 key set" if os.getenv("APPLY_AI_CUSTOM_KEY") else "⚠️ no key")
                           + " · free models may rate-limit; retries built in.")
    else:
        k = st.sidebar.text_input(f"{prov} API key", type="password",
                                  placeholder="paste to use (this session)")
        if k.strip():
            os.environ[_keymap[prov]] = k.strip()
        st.sidebar.caption("🔑 key set" if os.getenv(_keymap[prov]) else "⚠️ no key — paste above")

    if page == "My Resume":
        page_resume()
    elif page == "Add Job":
        page_add_job(conn, use_local)
    elif page == "Pipeline":
        page_pipeline(conn)
    else:
        page_notifications(conn)


if __name__ == "__main__":
    main()
