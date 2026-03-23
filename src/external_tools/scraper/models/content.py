from pydantic import BaseModel

class SiteConfigModel(BaseModel):
    name: str
    base_url: str
    url_dict: dict[str, str]
    title_selector: str
    content_selector: list[str]
    date_selector: str | None = None
    subtitle_selector: str | None = None
    author_selector: str | None = None
    img_url_selector: str | None = None
    prevent_redirection: bool = False
    rate_limit: float = 0.0
    strip_chars: int = 0
