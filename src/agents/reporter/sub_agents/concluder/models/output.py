from typing import List, Literal
from pydantic import BaseModel, Field

class ConclusionOutput(BaseModel):
    conclusion: str = Field(..., description="Report conclusion based on sections content")

class IntroductionOutput(BaseModel):
    title: str = Field(..., description="Report title")
    introduction: str = Field(..., description="Report introduction based on sections content")

class EvaluationOutput(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1 on how good is the introduction and conclusion")
    sections_to_improve: Literal["conclusion", "introduction"] = Field(..., description="Sections to improve. If both -> 'conclusion', if only introduction -> 'introduction'")
    improvements: List[str] = Field(..., min_length=0, max_length=5, description="Key points to improve the introduction/conclusion")