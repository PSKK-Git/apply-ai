"""Turn a real resume (uploaded PDF/text or pasted) into the app's Resume model.

This is the base everything builds on: once your real resume is parsed here, the
fit score, tailoring, and ATS all compare the JD against *your* experience instead
of the sample. Every bullet is given a stable evidence_id so grounding/provenance
works exactly as with the seed resume.
"""
from __future__ import annotations
import io
import json
import re

from . import llm
from .schemas import Resume

_PROMPT = """Extract this resume into JSON with EXACTLY this shape:
{{
 "name": "", "contact": {{"email": "", "phone": "", "links": []}},
 "skills": [],
 "experiences": [{{"id": "exp-1", "company": "", "role": "", "start": "", "end": "",
   "bullets": [{{"id": "ev-1", "text": "", "skills": []}}]}}],
 "projects": [{{"id": "proj-1", "name": "", "bullets": [{{"id": "ev-9", "text": "", "skills": []}}]}}],
 "education": [{{"school": "", "degree": "", "year": ""}}]
}}
Rules: give EVERY bullet a unique id (ev-1, ev-2, ...). Extract skills as short
lowercase tokens (e.g. "python", "aws"). Keep bullet text faithful — do not invent.
Return ONLY JSON. Resume:
---
{raw}
---"""


def extract_text(uploaded) -> str:
    """Get plain text from a Streamlit UploadedFile (pdf/txt/md/json)."""
    name = (getattr(uploaded, "name", "") or "").lower()
    data = uploaded.read()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", "replace")


def _coerce_ids(data: dict) -> None:
    """Guarantee every bullet has a non-empty id (LLMs sometimes drop them)."""
    n = 0
    for group in ("experiences", "projects"):
        for item in data.get(group) or []:
            for b in item.get("bullets") or []:
                n += 1
                if not b.get("id"):
                    b["id"] = f"ev-{n}"
                b.setdefault("skills", [])


def parse_resume(raw_text: str) -> Resume:
    """Structured JSON passes through; anything else is parsed by the LLM."""
    raw = (raw_text or "").strip()
    if not raw:
        raise ValueError("empty resume text")

    # already valid Resume JSON? load directly.
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("name"):
            _coerce_ids(data)
            return Resume.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        pass

    out = llm.complete(_PROMPT.format(raw=raw[:16000]), max_tokens=2000)
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        raise ValueError("model did not return JSON for the resume")
    data = json.loads(m.group(0))
    data["name"] = data.get("name") or "Candidate"
    _coerce_ids(data)
    return Resume.model_validate(data)
