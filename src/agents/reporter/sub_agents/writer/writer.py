import logging
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from src.agents.orchestrator.models.content import Observation
from src.agents.reporter.sub_agents.planner.models.output import Section
from src.agents.reporter.sub_agents.writer.models.output import WriterOutput, ValidatorOutput
from src.agents.reporter.sub_agents.writer.utils.prompts import GENERATE_SECTION_CONTENT_PROMPT, EVALUATE_CONTENT_PROMPT
from src.utils.decorators.retry import retry_with_backoff

class State(TypedDict):
    # Site data
    observations: list[Observation]
    url_dict: dict[str, any]
    selectors: dict[str, any]

    # Writer data
    section: Section
    section_content: WriterOutput
    score: float
    improvements: list[str]
    revision_count: int

class Writer:
    """
    Writer agent, generates the content for the provided section
    """
    def __init__(self,
                ollama_model: str,
                ollama_base_url: str,
                temperature: float = 0.05,
                top_p: float = 0.2):
        """
        Initializes planner agent

        Args:
            ollama_model (str): Name of the ollama model
            ollama_base_url (str): Base URL for ollama
            temperature (float): Controls the randomness of the output
            top_p (float): Lowering top_p narrows the field of possible tokens.
        """
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=temperature,
            top_p=top_p,
            format="json",
            num_predict=1024,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the writer graph
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("section_writer", self._section_writer)
        graph.add_node("section_validator", self._section_validator)

        # Add edges
        graph.set_entry_point("section_writer")
        graph.add_edge("section_writer", "section_validator")
        graph.add_conditional_edges(
            "section_validator",
            self._decide_next_step,
            {
                "continue": "section_writer",
                "end": END,
            },
        )

        return graph.compile()

    @retry_with_backoff()
    def _section_writer(self, state: State) -> dict[str, any]:
        """
        Generate section content from section data

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        try:
            logging.info(f"Running writer for {state['section'].title} section")

            structured_llm = self.llm.with_structured_output(WriterOutput)
            chain = GENERATE_SECTION_CONTENT_PROMPT | structured_llm

            response = chain.invoke({
                    "title": state["section"].title,
                    "description": state["section"].description,
                    "expected_format": state["section"].expected_format,
                    "observations": state["observations"],
                    "url_dict": state["url_dict"],
                    "selectors": state["selectors"],
                    "improvements": state["improvements"]
                })

            if isinstance(response, WriterOutput):
                content_data = {
                    "section_title": response.section_title,
                    "section_content": response.section_content,
                }
            else:
                response_data = response.model_dump()
                content_data = {
                    "section_title": response_data.get("section_title"),
                    "section_content": response_data.get("section_content"),
                }

            logging.info(f"Sucessfully generated content for section: {state['section'].title}")
            return {"section_content": content_data}

        except Exception as e:
            logging.error(f"Failed to generate content for secion `{state['section'].title}`: {e}")
            return {
                    "section_title": "",
                    "section_content": "",
                }

    @retry_with_backoff()
    def _section_validator(self, state: State) -> dict[str, any]:
        """
        Evaluate section content and generate score

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        try:
            logging.info(f"Running writer validator for {state['section'].title} section")

            structured_llm = self.llm.with_structured_output(ValidatorOutput)
            chain = EVALUATE_CONTENT_PROMPT | structured_llm

            response = chain.invoke({
                "observations": state["observations"],
                "url_dict": state["url_dict"],
                "selectors": state["selectors"],
                "section_content": state["section_content"],
                "title": state["section"].title,
                "description": state["section"].description,
                "expected_format": state["section"].expected_format,
            })

            if isinstance(response, ValidatorOutput):
                evaluation_data = {
                    "score": response.score,
                    "improvements": response.improvements
                    }
            else:
                response_data = response.model_dump()
                evaluation_data = {
                    "score": response_data.get("score"),
                    "improvements": response_data.get("improvements")
                    }

            return {**evaluation_data, "revision_count": state["revision_count"] + 1}

        except Exception as e:
            logging.error(f"Failed to generate content for section {state['section'].title}: {e}")
            return {
                "score": 0,
                "improvements": None,
                "revision_count": state["revision_count"] + 1
            }

    def _decide_next_step(self, state: State) -> Literal["end", "continue"]:
        """
        Validator checkpoint to route the graph to either continue or end process

        Args:
            state (State): Graph state that the node receives
        Returns:
            Literal["continue" | "end"]: Literal to know if we continue or we go to the next node
        """
        score = state["score"]
        revision_count = state["revision_count"]

        if score >= 0.8 or revision_count >= 2:
            logging.info("Content validation passed")
            return "end"
        else:
            logging.warning("Failed to validate content. Re-iterating")
            return "continue"