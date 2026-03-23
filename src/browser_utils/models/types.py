from enum import Enum
from pydantic import BaseModel

class FetchStatus(str, Enum):
    SUCCESSFUL = "successful"
    ERROR = "error"

class FetchResult(BaseModel):
    """
    Results of site's HTML fetch
    """
    status: FetchStatus
    html_content: str | None
    url: str
    fetch_duration: float
    error: str | None