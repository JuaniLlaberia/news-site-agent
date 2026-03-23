from pydantic import BaseModel, Field

class Section(BaseModel):
    id: str = Field(..., description="Section identifier id")
    title: str = Field(..., description="Section name or temporal title")
    description: str = Field(..., description="Section description or goal (what we want to say here)")
    expected_format: list[str] = Field(min_length=1, max_length=3, description="Section format can be narrative, bullets or table. And can be combined")

class PlannerOutput(BaseModel):
    sections: list[Section] = Field(..., min_length=2, max_length=6, description="Sections that will be part of the final report")

class EvaluationOutput(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1 on how good is the plan")
    improvements: list[str] = Field(..., min_length=0, max_length=5, description="Key points to improve the plan")