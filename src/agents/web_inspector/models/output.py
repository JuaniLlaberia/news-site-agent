from pydantic import BaseModel, Field

class MainRoutesOutput(BaseModel):
    routes: list[str] = Field(..., description="All routes (web navigation routes) that have articles that could interest us", min_length=1)

class ArticlesTagOutput(BaseModel):
    css_selector: str = Field(..., description="CSS selector to get all articles from the page")

class ArticleContentOutput(BaseModel):
    title_selector: str = Field(..., description="CSS selector for article title")
    content_selector: str = Field(..., description="CSS selectors for the article content")
    subtitle_selector: str | None = Field(..., description="CSS selector for article sub-title")
    author_selector: str | None = Field(..., description="CSS selector for article author")
    img_url_selector: str | None = Field(..., description="CSS selector for article main image")
    date_selector: str | None  = Field(..., description="CSS selector for article date selector")