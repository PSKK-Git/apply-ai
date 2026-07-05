"""Render a tailored résumé to a real PDF document.

Pure-Python via reportlab — no LaTeX/tectonic/system deps, so it just works. Takes
the base Resume (header, skills, education) + the TailorResult (the grounded,
JD-tailored bullets) and lays out a clean one-column résumé. Returned as bytes for
an in-browser download, or written to disk when a version is accepted.
"""
from __future__ import annotations
import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from .schemas import Resume, TailorResult

_ACCENT = colors.HexColor("#4b2e83")   # royal violet
_RULE = colors.HexColor("#c9c2e0")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "name": ParagraphStyle("Name", parent=ss["Title"], fontSize=19,
                               spaceAfter=2, alignment=TA_CENTER),
        "contact": ParagraphStyle("Contact", parent=ss["Normal"], fontSize=9,
                                  alignment=TA_CENTER, textColor=colors.grey, spaceAfter=8),
        "section": ParagraphStyle("Section", parent=ss["Heading2"], fontSize=11,
                                  spaceBefore=11, spaceAfter=3, textColor=_ACCENT),
        "body": ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=13),
        "bullet": ParagraphStyle("Bullet", parent=ss["Normal"], fontSize=9.5,
                                 leading=13, leftIndent=11),
    }


def _flow(resume: Resume, tailored: TailorResult):
    s = _styles()
    els = [Paragraph(resume.name or "Résumé", s["name"])]
    c = resume.contact
    bits = [x for x in [c.email, c.phone, *c.links] if x]
    if bits:
        els.append(Paragraph(" · ".join(bits), s["contact"]))
    els.append(HRFlowable(width="100%", color=_RULE, spaceAfter=4))

    if resume.skills:
        els.append(Paragraph("SKILLS", s["section"]))
        els.append(Paragraph(", ".join(resume.skills), s["body"]))

    els.append(Paragraph("EXPERIENCE", s["section"]))
    bullets = tailored.bullets if tailored and tailored.bullets else None
    if bullets:
        for b in bullets:
            els.append(Paragraph("&bull;&nbsp; " + b.text, s["bullet"]))
    else:  # no tailoring yet -> fall back to the base résumé bullets
        for e in resume.experiences:
            els.append(Paragraph(f"<b>{e.role} — {e.company}</b>", s["body"]))
            for b in e.bullets:
                els.append(Paragraph("&bull;&nbsp; " + b.text, s["bullet"]))

    if resume.education:
        els.append(Paragraph("EDUCATION", s["section"]))
        for ed in resume.education:
            els.append(Paragraph(f"{ed.degree}, {ed.school} ({ed.year})", s["body"]))
    els.append(Spacer(1, 2))
    return els


def build_pdf_bytes(resume: Resume, tailored: TailorResult) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, topMargin=0.6 * inch,
                            bottomMargin=0.6 * inch, leftMargin=0.7 * inch,
                            rightMargin=0.7 * inch, title=f"{resume.name} — résumé")
    doc.build(_flow(resume, tailored))
    return buf.getvalue()


def build_pdf(resume: Resume, tailored: TailorResult, path: str) -> str:
    with open(path, "wb") as fh:
        fh.write(build_pdf_bytes(resume, tailored))
    return path
