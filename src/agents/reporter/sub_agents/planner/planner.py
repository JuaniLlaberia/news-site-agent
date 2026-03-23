import logging
from typing import TypedDict, Literal

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from src.utils.decorators.retry import retry_with_backoff
from src.agents.orchestrator.models.content import Observation
from .models.output import Section, PlannerOutput, EvaluationOutput
from .utils.prompts import GENERATE_PLAN_PROMPT, EVALUATE_PLAN_PROMPT

class State(TypedDict):
    site_name: str
    url: str
    url_dict: dict[str, str]
    title_selector: str
    subtitle_selector: str | None
    author_selector: str | None
    img_url_selector: str | None
    content_selector: list[str]
    date_selector: str | None
    rate_limit: float
    observations: list[Observation]

    sections: list[Section]
    score: float
    improvements: str | None

class Planner:
    """
    Planner agent, generates a structured plan for the report
    """
    def __init__(self, ollama_model: str,
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
        Build the planner graph with conditional validation
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("plan_generator", self._plan_generator)
        graph.add_node("evaluate_plan", self._evaluate_plan)

        # Add edges
        graph.set_entry_point("plan_generator")
        graph.add_conditional_edges(
            "plan_generator",
            self._validate_generation,
            {
                "continue": "plan_generator",
                "end": "evaluate_plan"
            }
        )
        graph.add_conditional_edges(
            "evaluate_plan",
            self._validate_plan,
            {
                "continue": "plan_generator",
                "end": END
            }
        )

        return graph.compile()

    @retry_with_backoff()
    def _plan_generator(self, state: State) -> dict[str, any]:
        """
        Generate report plan from information

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        try:
            logging.info(f"Generating plan based on {len(state['observations'])} observations")
            structured_llm = self.llm.with_structured_output(PlannerOutput)
            chain = GENERATE_PLAN_PROMPT | structured_llm

            response = chain.invoke({
                        "observations": state["observations"],
                        "url_dict": state["url_dict"],
                        "selectors": {
                            "title_selector": state["title_selector"],
                            "subtitle_selector": state["subtitle_selector"],
                            "author_selector": state["author_selector"],
                            "img_url_selector": state["img_url_selector"],
                            "content_selector": state["content_selector"],
                            "date_selector": state["date_selector"]
                        },
                        "url": state["url"],
                        "site_name": state["site_name"],
                        "improvements": state["improvements"]
                    })

            if isinstance(response, PlannerOutput):
                plan_data = {"sections": response.sections}
            else:
                response_data = response.model_dump()
                plan_data = {"sections": response_data.get("sections")}

            logging.info("Sucessfully generated plan")
            return {**plan_data}

        except Exception as e:
            logging.error(f"Unexpected error when generating plan: {e}")
            return {"sections": []}

    def _validate_generation(self, state: State) -> Literal["continue", "end"]:
        """
        Validator method to check if plan generation failed and we need to re-iterate

        Args:
            state (State): Graph state that the node receives
        Returns:
            Literal["continue" | "end"]: Literal to know if we continue or we go to the next node
        """
        return "end" if len(state["sections"]) > 0 else "continue"

    @retry_with_backoff()
    def _evaluate_plan(self, state: State) -> dict[str, any]:
        """
        Evaluates plan and generates a score

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        try:
            logging.info("Evaluating generated plan")
            structured_llm = self.llm.with_structured_output(EvaluationOutput)
            chain = EVALUATE_PLAN_PROMPT | structured_llm

            response = chain.invoke({"sections": state["sections"]})

            if isinstance(response, EvaluationOutput):
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

            logging.info("Successfully evaluated the plan")
            return {**evaluation_data}

        except Exception as e:
            logging.error(f"Failed plan evaluation: {e}")
            return {
                "score": 0,
                "improvements": None
            }

    def _validate_plan(self, state: State) -> Literal["continue", "end"]:
        """
        Validator method to check score and decide if we should re-iterate or end process

        Args:
            state (State): Graph state that the node receives
        Returns:
            Literal["continue" | "end"]: Literal to know if we continue or we go to the next node
        """
        if state["score"] >= 0.8:
            logging.info("Plan validation passed")
            return "end"
        else:
            logging.warning("Failed to validate plan. Re-iterating")
            return "continue"

    def run(self,
            site_name: str,
            url: str,
            url_dict: dict[str, str],
            title_selector: str,
            content_selector: list[str],
            subtitle_selector: str | None,
            author_selector: str | None,
            img_url_selector: str | None,
            date_selector: str | None,
            rate_limit: float,
            observations: list[Observation]) -> list[Section]:
        """
        Run the planner agent

        Args:
            site_name (str): News site name
            url (str): Site's URL
            title_selector (str): CSS selector for the article's title
            content_selector (list[str]): CSS selector for the article's content
            subtitle_selector (str | None): CSS selector for the article's subtitle
            author_selector (str): CSS selector for the article's author
            img_url_selector (str): CSS selector for the article's image url
            date_selector (str): CSS selector for the article's date
            rate_limit (float): Rate limit delay in seconds
            observations (list[Observation]): List of observations made during the analysis pipeline
        Returns:
            list[Section]: List of generated sections
        """
        initial_state = State(
            site_name=site_name,
            url=url,
            url_dict=url_dict,
            title_selector=title_selector,
            content_selector=content_selector,
            subtitle_selector=subtitle_selector,
            author_selector=author_selector,
            img_url_selector=img_url_selector,
            date_selector=date_selector,
            rate_limit=rate_limit,
            observations=observations,
            sections=[],
            score=0,
            improvements=None
        )

        logging.info("Running report planner...")
        result = self.graph.invoke(initial_state)

        return result["sections"]