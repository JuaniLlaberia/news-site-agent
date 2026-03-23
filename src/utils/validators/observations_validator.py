from src.agents.orchestrator.models.content import Observation

REQUIRED_FIELDS = ["type", "message", "agent_type"]

def validate_observation(observation: Observation) -> list[str]:
    missing = []
    for field in REQUIRED_FIELDS:
        if not observation.get(field, False):
            missing.append(field)
    return missing

def validate_observations(observations: list[Observation]) -> tuple[bool, None | str]:
    """
    Validates that observations have all required fields

    Args:
        observations (list[Observation]): List of observations comming from the site analysis
    Returns:
        tuple[bool, None | str]: Tuple containing a bool to know whether the observations are valid or not, and an error string in case it's not valid
    """
    errors = []

    for idx, observation in enumerate(observations):
        missing = validate_observation(observation)
        if missing:
            errors.append({"index": idx, "missing_fields": missing})

    if errors:
        return False, errors

    return True, None
