from typing import List, Literal
from pydantic import BaseModel, Field

class ContentItem(BaseModel):
    order: int
    content_type: Literal["narrative", "bullets"]
    text: str = ""
    items_subtitle: str = ""
    items: list[str] = []

class WriterOutput(BaseModel):
    section_title: str = Field(..., description="Section title")
    section_content: List[ContentItem] = Field(..., max_length=4, description="Content for this section that will be part of the final report")

class ValidatorOutput(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1 on how good section content is")
    improvements: List[str] = Field(..., min_length=0, max_length=5, description="Key points to improve the content")