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
    if text.strip():
        return text
    # JS-rendered boards (Workable/Greenhouse/LinkedIn) return an empty shell to a
    # plain GET; fall back to the r.jina.ai reader, which renders the page server-side
    # and returns clean text. (Sends the public job URL to that third-party service.)
    reader = httpx.get(f"https://r.jina.ai/{url}", timeout=60,
                       headers={"User-Agent": "Mozilla/5.0"})
    reader.raise_for_status()
    text = (reader.text or "").strip()
    if not text:
        raise ValueError(f"No extractable text at {url}")
    return text


def _claude_json(prompt: str) -> str:
    # provider-agnostic generator (name kept for the test hook)
    from . import llm
    return llm.complete(prompt, max_tokens=1200)


def _parse_json_block(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object in model output")
    return json.loads(m.group(0))


def extract_jd(raw_text: str, company_job_id: str | None = None) -> JD:
    data = _parse_json_block(_claude_json(_PROMPT.format(raw=raw_text[:12000])))
    return JD(
        title=data.get("title") or "Unknown",
        company=data.get("company") or "Unknown",
        company_job_id=company_job_id or data.get("company_job_id"),
        required_skills=data.get("required_skills") or [],
        nice_to_have=data.get("nice_to_have") or [],
        responsibilities=data.get("responsibilities") or [],
        raw_text=raw_text,
    )


def ingest(url_or_text: str, company_job_id: str | None = None) -> JD:
    raw = fetch_text(url_or_text) if url_or_text.strip().startswith("http") else url_or_text
    return extract_jd(raw, company_job_id=company_job_id)
