from enum import Enum
from pydantic import BaseModel

class AgentType(str, Enum):
    ORCHESTRATOR = "orchestrator"
    WEB_INSPECTOR = "web_inspector"
    TESTER = "tester"
    RATE_LIMIT_TEST = "rate_limit_tester"

class ObservationType(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

class Observation(BaseModel):
    type: ObservationType
    message: str
    agent_type: AgentType