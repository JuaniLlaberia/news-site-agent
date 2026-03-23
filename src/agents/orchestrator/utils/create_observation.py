from src.agents.orchestrator.models.content import Observation, ObservationType, AgentType

def create_observation(observations: list[Observation],
                       observation_type: ObservationType,
                       message: str,
                       agent_type: AgentType) -> list[Observation]:
    """
    Helper function to create observation and return full list

    Args:
        observations (list[Observation]): List of current observations
        observation_type (ObservationType): Observation type
        message (str): Observation msg
        agent_type (AgentType): Name of the agent for the report
    Returns:
        list[Observation]: List of observation containing previous ones and the new one
    """
    observation = Observation(
        type=observation_type,
        message=message,
        agent_type=agent_type,
    )

    return [*observations, observation]
