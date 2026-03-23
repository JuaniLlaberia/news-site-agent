import logging
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph
from langgraph.types import Send

from src.agents.orchestrator.models.content import Observation
from src.agents.reporter.sub_agents.concluder.concluder import Concluder
from src.agents.reporter.sub_agents.planner.planner import Planner
from src.agents.reporter.sub_agents.writer.writer import Writer
from src.agents.reporter.sub_agents.planner.models.output import Section
from src.agents.reporter.sub_agents.writer.models.output import WriterOutput
from src.playwright.pdf_generator import PDFGenerator

class State(TypedDict):
    # Report data
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
    # Report
    plan_sections: list[Section]
    completed_sections: Annotated[list[WriterOutput], add]
    title: str
    introduction: str
    conclusion: str
    report_bytes: bytes

class Reporter:
    """
    Reporter class generates reports based on the agent observations and extracted data
    """
    def __init__(self,
                 ollama_model: str,
                 ollama_base_url: str,
                 temperature: float = 0.05,
                 top_p: float = 0.2):
        """
        Initializes Reporter instance

        Args:
            ollama_model (str): Name of the ollama model
            ollama_base_url (str): Base URL for ollama
            temperature (float): Controls the randomness of the output
            top_p (float): Lowering top_p narrows the field of possible tokens.
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.top_p = top_p
        self.writer = Writer(ollama_model=ollama_model,
                            ollama_base_url=ollama_base_url,
                            temperature=temperature,
                            top_p=top_p)
        self.graph: StateGraph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the reporter agent graph
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("planner", self._planner_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("concluder", self._concluder_node)
        graph.add_node("file_generator", self._reporter_file_generator_node)

        # Add edges
        graph.set_entry_point("planner")
        graph.add_conditional_edges(
            "planner",
            self._assign_workers,
            {"writer": "writer"}
        )
        graph.add_edge("writer", "concluder")
        graph.add_edge("concluder", "file_generator")
        graph.set_finish_point("file_generator")

        return graph.compile()

    def _planner_node(self, state: State) -> dict[str, any]:
        """
        Generates plan based on the observations and selectors data

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        planner = Planner(
            ollama_model=self.ollama_model,
            ollama_base_url=self.ollama_base_url,
            temperature=self.temperature,
            top_p=self.top_p
        )
        sections = planner.run(
            site_name=state["site_name"],
            url=state["url"],
            url_dict=state["url_dict"],
            title_selector=state["title_selector"],
            subtitle_selector=state["subtitle_selector"],
            author_selector=state["author_selector"],
            img_url_selector=state["img_url_selector"],
            content_selector=state["content_selector"],
            date_selector=state["date_selector"],
            rate_limit=state["rate_limit"],
            observations=state["observations"],
        )

        selectors = {
            "title_selector": state["title_selector"],
            "subtitle_selector": state["subtitle_selector"],
            "author_selector": state["author_selector"],
            "img_url_selector": state["img_url_selector"],
            "content_selector": state["content_selector"],
            "date_selector": state["date_selector"]
        }
        default_config_section = Section(
            id="99",
            title="Final Site Configuration",
            description=f"Display the url_dict ({state['url_dict']}) and the selectors ({selectors}) without changing them.",
            expected_format=["narrative", "bullets"]
        )

        print(sections + [default_config_section])

        return {
            "plan_sections": sections + [default_config_section]
        }

    def _assign_workers(self, state: State) -> list[Send]:
        """
        Assign a worker to each section

        Args:
            state (State): Graph state that the node receives
        """
        logging.info(f"Sending {len(state['plan_sections'])} workers to process sections")

        return [Send("writer", {
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
            "section": s,
            "section_content": WriterOutput(section_title="", section_content=[]),
            "score": 0.0,
            "improvements": [],
            "revision_count": 0
        }) for s in state["plan_sections"]]

    def _writer_node(self, state: State) -> dict[str, any]:
        """
        Generates content for the report based on data and plan

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        writer_result = self.writer.graph.invoke(state)
        section_content = writer_result["section_content"]

        return {"completed_sections": [section_content]}

    def _concluder_node(self, state: State) -> dict[str, any]:
        """
        Generates title, introduction and conclusion for the report based on data and plan

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        concluder = Concluder(ollama_model="gemma3:4b", ollama_base_url="http://localhost:11434")
        title, introduction, conclusion = concluder.run(content=state["completed_sections"])

        return {
            "title": title,
            "introduction": introduction,
            "conclusion": conclusion
        }

    def _reporter_file_generator_node(self, state: State) -> dict[str, any]:
        """
        Generates PDF bytes based on all the generated content

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        generator = PDFGenerator(format="A4")
        pdf_bytes = generator.run(
            title=state["title"],
            intro=state["introduction"],
            sections=state["completed_sections"],
            conclusion=state["conclusion"]
        )

        return {
            "report_bytes": pdf_bytes
        }

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
            observations: list[Observation]) -> bytes:
        """
        Runs the Reporter agent

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
            bytes
        """
        logging.info("Running reporter agent...")
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
            plan_sections=[],
            completed_sections=[],
            title="",
            introduction="",
            conclusion="",
            report_bytes=0
        )

        result: State = self.graph.invoke(initial_state)
        return result["report_bytes"]


