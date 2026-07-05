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


class TailoredBullet(BaseModel):
    id: str                                 # tb-1, tb-2 ... (stable within a result)
    text: str
    provenance: list[str] = Field(default_factory=list)   # evidence_ids — GUARANTEED non-empty
    skills: list[str] = Field(default_factory=list)


class TailorResult(BaseModel):
    bullets: list[TailoredBullet] = Field(default_factory=list)
    provenance: dict[str, list[str]] = Field(default_factory=dict)  # tb-id -> [evidence_id]
    skills_delta: list[str] = Field(default_factory=list)           # JD skills surfaced vs base
    dropped: list[str] = Field(default_factory=list)                # ungrounded candidate texts


class Gap(BaseModel):
    keyword: str
    groundable: bool = False   # True = present somewhere in the real evidence corpus
                               #        (surface it); False = needs new real input to add


class ATSReport(BaseModel):
    score: int                                          # 0-100 JD-keyword coverage of the resume
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)       # missing JD keywords, flagged addable-vs-new


class VerifyVerdict(BaseModel):
    provider: str                                       # claude | openai | gemini
    ats_alignment: int | None = None                    # 0-100 the model's ATS-alignment estimate
    fabrication_flags: list[str] = Field(default_factory=list)  # bullet ids/text it doubts
    suggestions: list[str] = Field(default_factory=list)
    error: str | None = None                            # set when the provider was skipped/failed


class VerifyReport(BaseModel):
    verdicts: list[VerifyVerdict] = Field(default_factory=list)
    consensus_flags: list[str] = Field(default_factory=list)  # doubted by >1 provider
    avg_ats_alignment: int | None = None


def load_resume(path: str = "data/resume.json") -> Resume:
    with open(path, encoding="utf-8") as fh:
        return Resume.model_validate(json.load(fh))
