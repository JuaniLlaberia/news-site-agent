import os
from typing import TypedDict
from langgraph.graph import StateGraph

from src.agents.web_inspector.web_inspector import WebInspector
from src.browser_utils.rate_limit_analyzer import RateLimitAnalyzer
from src.external_tools.scraper.scraper_api import ScraperAPI
from src.external_tools.scraper.models.content import SiteConfigModel
from .models.content import Observation

class State(TypedDict):
    # Core site data
    site_name: str
    url: str
    # Main scraping data
    url_dict: dict[str, str]
    title_selector: str
    subtitle_selector: str | None
    author_selector: str | None
    img_url_selector: str | None
    content_selector: list[str]
    date_selector: str | None
    # Extra scraping data
    rate_limit: float
    # Workflow data
    observations: list[Observation]
    # Tester
    is_valid_config: bool = False

class Orchestrator:
    """
    Orchestrator class
    """
    def __init__(self):
        """
        Initializes Orchestrator agent instance
        """
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build orchestrator graph
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("web_inspector", self._web_inspector_node)
        graph.add_node("rate_limit_tester", self._rate_limit_tester_node)
        graph.add_node("tester", self._tester_node)

        # Add edges
        graph.set_entry_point("web_inspector")
        graph.add_edge("web_inspector", "rate_limit_tester")
        graph.add_edge("rate_limit_tester", "tester")
        graph.set_finish_point("tester")

        return graph.compile()

    def _web_inspector_node(self, state: State) -> dict[str, any]:
        """
        Extracts & Validates content with its HTML/CSS tags/classes to create news site scrape config

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        web_inspector = WebInspector(gemini_model=os.getenv("GEMINI_MODEL"),
                                     gemini_key=os.getenv("GEMINI_KEY"),
                                     temperature=0.05,
                                     top_p=0.2,
                                     tok_k=10)
        selectors, url_dict, observations = web_inspector.run(url=state["url"])

        return {
            **selectors,
            "url_dict": url_dict,
            "observations": state["observations"] + observations
        }

    def _rate_limit_tester_node(self, state: State) -> dict[str, any]:
        """
        Performs rate limit testing to site

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        rate_limit_analyzer = RateLimitAnalyzer(url=state["url"])
        rate_limit_delay, observations = rate_limit_analyzer.run()

        return {
            "rate_limit": rate_limit_delay,
            "observations": state["observations"] + observations
        }

    def _tester_node(self, state: State) -> dict[str, any]:
        """
        Performs url_dict and selectors testing. It simulates a scrape to the url and validates the extract data

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        scraper_tester = ScraperAPI(site_dict=SiteConfigModel(
            name=state["site_name"],
            base_url=state["url"],
            url_dict=state["url_dict"],
            title_selector=state["title_selector"],
            content_selector=state["content_selector"],
            subtitle_selector=state["subtitle_selector"],
            date_selector=state["date_selector"],
            author_selector=state["author_selector"],
            img_url_selector=state["img_url_selector"],
            rate_limit=state["rate_limit"],
            strip_chars=0
        ))
        is_valid, observations = scraper_tester.run()

        return {
            "is_valid_config": is_valid,
            "observations": state["observations"] + observations
        }

    def run(self, site_name: str, url: str) -> tuple[dict[str, any], list[Observation]]:
        """
        Run orchestrator agent

        Args:
            site_name (str): Name of the site we are researching
            url (str): URL of the site we are researching

        Returns:
            tuple[dict[str, any], list[Observation]]:
                - Dictionary containing the news_site_config information
                - List of observation gather during the analysis
        """
        initial_state = State(
            site_name=site_name,
            url=url,
            url_dict={},
            title_selector="",
            subtitle_selector=None,
            author_selector=None,
            img_url_selector=None,
            date_selector=None,
            content_selector=[],
            rate_limit=0.0,
            observations=[],
        )

        result: State = self.graph.invoke(initial_state)
        news_site_config = {
            "name": result["site_name"],
            "base_url": result["url"],
            "url_dict": result["url_dict"],
            "title_selector": result["title_selector"],
            "subtitle_selector": result["subtitle_selector"],
            "author_selector": result["author_selector"],
            "img_url_selector": result["img_url_selector"],
            "content_selector": result["content_selector"],
            "date_selector": result["date_selector"],
            "rate_limit": result["rate_limit"],
        }

        return news_site_config, result["observations"]
