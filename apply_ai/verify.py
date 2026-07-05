"""Cross-verify a Claude-tailored resume with other models (ChatGPT, Gemini).

Claude generates; independent models audit. Each provider is asked the same thing:
rate ATS alignment to the JD (0-100), flag any bullet that looks unsupported by
its cited evidence, and suggest fixes. We aggregate into a consensus (a bullet
doubted by more than one model is worth your attention) and an average alignment.

Providers with no API key are simply skipped — the run degrades, never crashes.
Only Anthropic + OpenAI keys exist today; add GEMINI_API_KEY to light up Gemini.
"""
from __future__ import annotations
import json
import os
import re
from collections import Counter

from .schemas import JD, TailorResult, VerifyReport, VerifyVerdict

_PROMPT = """You are auditing a resume tailored to a job description. For each
tailored bullet, judge whether it is a plausible, non-exaggerated rewrite. Then
rate how well the whole set matches the JD for an ATS (0-100).

Return ONLY JSON: {{"ats_alignment": <int 0-100>,
  "fabrication_flags": ["<bullet id you doubt>", ...],
  "suggestions": ["<short fix>", ...]}}

Job: {title} at {company}. Required: {required}. Nice: {nice}.
Tailored bullets:
{bullets}
"""


def _prompt(jd: JD, tailored: TailorResult) -> str:
    bullets = "\n".join(f"{b.id}: {b.text}  (cites: {', '.join(b.provenance)})"
                        for b in tailored.bullets)
    return _PROMPT.format(
        title=jd.title, company=jd.company,
        required=", ".join(jd.required_skills) or "(none)",
        nice=", ".join(jd.nice_to_have) or "(none)",
        bullets=bullets or "(none)",
    )


def _parse(provider: str, text: str) -> VerifyVerdict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(m.group(0)) if m else {}
    align = data.get("ats_alignment")
    return VerifyVerdict(
        provider=provider,
        ats_alignment=int(align) if isinstance(align, (int, float)) else None,
        fabrication_flags=[str(x) for x in data.get("fabrication_flags", [])],
        suggestions=[str(x) for x in data.get("suggestions", [])],
    )


# -- provider runners (each raises if its key/SDK is unavailable) ----------------
def _verify_claude(jd, tailored, ats) -> VerifyVerdict:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=os.getenv("APPLY_AI_VERIFY_CLAUDE_MODEL", "claude-sonnet-5"),
        max_tokens=800, messages=[{"role": "user", "content": _prompt(jd, tailored)}],
    )
    return _parse("claude", msg.content[0].text)


def _openai_key() -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if os.getenv("LLM_PROVIDER", "").lower() == "openai" and os.getenv("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    return None


def _verify_openai(jd, tailored, ats) -> VerifyVerdict:
    from openai import OpenAI
    key = _openai_key()
    if not key:
        raise RuntimeError("no OpenAI key")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=os.getenv("APPLY_AI_VERIFY_OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": _prompt(jd, tailored)}],
    )
    return _parse("openai", resp.choices[0].message.content)


def _gemini_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _verify_gemini(jd, tailored, ats) -> VerifyVerdict:
    import google.generativeai as genai
    key = _gemini_key()
    if not key:
        raise RuntimeError("no Gemini key")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(os.getenv("APPLY_AI_VERIFY_GEMINI_MODEL", "gemini-2.5-flash"))
    resp = model.generate_content(_prompt(jd, tailored))
    return _parse("gemini", resp.text)


def available_providers() -> dict:
    """Runners whose API key is present in the environment."""
    out: dict = {}
    if os.getenv("ANTHROPIC_API_KEY"):
        out["claude"] = _verify_claude
    if _openai_key():
        out["openai"] = _verify_openai
    if _gemini_key():
        out["gemini"] = _verify_gemini
    return out


def cross_verify(jd: JD, tailored: TailorResult, ats, *, runners=None) -> VerifyReport:
    runners = runners if runners is not None else available_providers()
    verdicts: list[VerifyVerdict] = []
    for name, fn in runners.items():
        try:
            verdicts.append(fn(jd, tailored, ats))
        except Exception as exc:  # a provider failing must not sink the others
            verdicts.append(VerifyVerdict(provider=name, error=str(exc)[:200]))

    counts: Counter = Counter()
    for v in verdicts:
        for flag in set(v.fabrication_flags):
            counts[flag] += 1
    consensus = sorted(f for f, c in counts.items() if c > 1)

    aligns = [v.ats_alignment for v in verdicts if v.ats_alignment is not None]
    avg = round(sum(aligns) / len(aligns)) if aligns else None

    return VerifyReport(verdicts=verdicts, consensus_flags=consensus, avg_ats_alignment=avg)
