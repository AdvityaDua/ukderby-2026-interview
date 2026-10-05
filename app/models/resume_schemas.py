from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra='allow')

class GeminiBaseSchema(BaseModel):
    """Schema for Gemini output must not have extra properties."""
    model_config = ConfigDict(from_attributes=True)

# Resume Analytics Inputs
class CVQualitySubscore(BaseSchema):
    dimension: Optional[str] = None
    score: Optional[int] = 0
    max_score: Optional[int] = 100
    evidence: Optional[List[str]] = None

class CVQuality(BaseSchema):
    overall_score: Optional[int] = 0
    subscores: Optional[List[CVQualitySubscore]] = None

class JDMatchSubscore(BaseSchema):
    dimension: Optional[str] = None
    score: Optional[int] = 0
    max_score: Optional[int] = 100
    evidence: Optional[List[str]] = None

class JDMatch(BaseSchema):
    overall_score: Optional[int] = 0
    subscores: Optional[List[JDMatchSubscore]] = None

class KeyTakeaways(BaseSchema):
    red_flags: Optional[List[str]] = None
    green_flags: Optional[List[str]] = None

class Analytics(BaseSchema):
    cv_quality: Optional[CVQuality] = None
    jd_match: Optional[JDMatch] = None
    key_takeaways: Optional[KeyTakeaways] = None
    overall_score: Optional[int] = 0
    sections: Optional[List[Any]] = None

class TailoredResume(BaseSchema):
    summary: Optional[str] = None
    experience: Optional[List[Any]] = None
    skills: Optional[List[Any]] = None
    projects: Optional[List[Any]] = None

class Top1PercentGap(BaseSchema):
    strengths: Optional[List[str]] = None
    gaps: Optional[List[str]] = None
    actionable_next_steps: Optional[List[str]] = None

class Enhancement(BaseSchema):
    tailored_resume: Optional[TailoredResume] = None
    top_1_percent_gap: Optional[Top1PercentGap] = None
    cover_letter: Optional[Any] = None

class ResumeData(BaseSchema):
    id: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    analytics: Optional[Analytics] = None
    enhancement: Optional[Enhancement] = None

class ResumeAnalysisRequest(BaseSchema):
    message: Optional[str] = None
    resume: Optional[ResumeData] = None
    id: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    analytics: Optional[Analytics] = None
    enhancement: Optional[Enhancement] = None

# Resume Built Output Schemas (Explicitly Typed for Gemini Structured Output)
class EducationDetail(GeminiBaseSchema):
    degree: str
    institution: str
    year: str
    details: Optional[str] = None

class ExperienceDetail(GeminiBaseSchema):
    role: str
    company: str
    duration: str
    responsibilities: List[str]

class ProjectDetail(GeminiBaseSchema):
    title: str
    description: str
    technologies: List[str]
    link: Optional[str] = None

class PersonalInfo(GeminiBaseSchema):
    name: str
    email: str
    phone: str
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None

class CertificationDetail(GeminiBaseSchema):
    name: str
    issuer: str
    year: str

class LanguageDetail(GeminiBaseSchema):
    language: str
    proficiency: str

class SkillCategories(GeminiBaseSchema):
    programming_languages: List[str]
    frameworks: List[str]
    tools: List[str]
    other: List[str]

class ResumeBuilderContent(GeminiBaseSchema):
    personal_info: PersonalInfo
    professional_summary: str
    skills: SkillCategories
    experience: List[ExperienceDetail]
    projects: List[ProjectDetail]
    education: List[EducationDetail]
    achievements: List[str]
    certifications: List[CertificationDetail]
    languages: List[LanguageDetail]

class ResumeBuilderResponse(GeminiBaseSchema):
    status: Optional[str] = "success"
    resume_content: Optional[ResumeBuilderContent] = None
    formatting_tips: Optional[List[str]] = None
    message: Optional[str] = ""
