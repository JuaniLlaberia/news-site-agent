import logging
from random import sample
from typing import TypedDict, Literal
from time import sleep

from langgraph.graph import StateGraph
from bs4 import BeautifulSoup

from src.browser_utils.fetchers import HTMLFetcher, FetchResult
from src.browser_utils.filters import HTMLFilter
from src.browser_utils.models.types import FetchStatus
from src.agents.orchestrator.models.content import Observation, ObservationType, AgentType
from src.agents.orchestrator.utils.create_observation import create_observation
from src.utils.decorators.retry import retry_with_backoff
from src.llm.gemini import Gemini
from .utils.prompts import EXTRACT_ROUTES_PROMPT, EXTRACT_ARTICLES_TAGS_PROMPT, EXTRACT_ARTICLE_CONTENT_PROMPT
from .models.output import MainRoutesOutput, ArticlesTagOutput, ArticleContentOutput

# DOUBLE CHECK THIS TAGS AND CLASSES
HTML_TAGS = ["header", "nav", "aside"]
CSS_CLASSES = ["nav", "navigation", "navbar", "menu", "main-nav", "primary-nav", "site-nav", "header",
               "header-nav", "top-nav", "topnav", "mobile-nav", "mobile-menu", "sidebar", "side-nav", "sidenav",
               "main-menu", "primary-menu", "site-menu", "nav-menu"]

class State(TypedDict):
    url: str
    # HTML analysis
    routes_list: list[str]

    html_data_per_route: dict[str, str]
    routes_to_analyze: list[str]
    url_dict: dict[str, str]

    articles_urls: list[str]
    articles_html: list[str]

    # Scraping data
    title_selector: str
    subtitle_selector: str | None
    author_selector: str | None
    img_url_selector: str | None
    content_selector: list[str]
    date_selector: str | None

    # General data
    observations: list[Observation]

    # Error handling
    retry_count: dict[str, int]
    max_retry_analysis: int = 3

class WebInspector:
    """
    Web Inspector Agent for extracting article structure and selectors from news sites
    """
    def __init__(self,
                 gemini_model: str,
                 gemini_key: str,
                 temperature: float,
                 top_p: float,
                 tok_k: int):
        """
        Initializes WebInspector agent instance

        Args:
            gemini_model (str): Name of the gemini model
            gemini_key (str): Gemini API key (comes from .env)
            temperature (float): Controls the randomness of the output
            top_p (float): Lowering top_p narrows the field of possible tokens
            top_k (int): Limits the token selection to the top_k most likely tokens at each step
        """
        self.agent_name = AgentType.WEB_INSPECTOR
        # Instance gemini model
        self.llm = Gemini(temperature=temperature,
                           top_p=top_p,
                           top_k=tok_k)
        # Instance HTML fetcher
        self.html_fetcher = HTMLFetcher()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the WebInspector agent
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("main_extractor", self._main_extractor_node)

        graph.add_node("routes_data_extractor", self._routes_data_extractor_node)
        graph.add_node("articles_tags_extractor", self._articles_tags_extractor_node)

        graph.add_node("get_articles_to_analyze", self._get_articles_to_analyze_node)
        graph.add_node("articles_content_extractor", self._articles_content_extractor_node)

        # Add edge
        graph.set_entry_point("main_extractor")
        graph.add_conditional_edges(
            "main_extractor",
            self._main_extractor_validator_node,
            {
                "continue": "main_extractor",
                "end": "routes_data_extractor"
            }
        )

        graph.add_edge("routes_data_extractor", "articles_tags_extractor")
        graph.add_conditional_edges(
            "articles_tags_extractor",
            self._articles_extractor_validator_node,
            {
                "continue": "articles_tags_extractor",
                "end": "get_articles_to_analyze"
            }
        )

        graph.add_conditional_edges(
            "get_articles_to_analyze",
            self._articles_to_analyze_validator_node,
            {
                "continue": "get_articles_to_analyze",
                "end": "articles_content_extractor"
            }
        )
        graph.set_finish_point("articles_content_extractor")

        return graph.compile()

    @retry_with_backoff()
    def _main_extractor_node(self, state: State) -> dict[str, any]:
        """
        Handles the extraction of the main content structure using the base_url

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        observations = state.get("observations", [])
        try:
            # Step 1: Fetch base_url HTML content
            main_fetch_results = self.html_fetcher.run(url=state["url"])

            if main_fetch_results.status != FetchStatus.SUCCESSFUL:
                observations = create_observation(observations=observations,
                                                observation_type=ObservationType.ERROR,
                                                message=f"Error when fetching main html: {main_fetch_results.error}",
                                                agent_type=self.agent_name)

                return {**state, "routes_list": [], "observations": observations}

            main_html = main_fetch_results.html_content

            # Step 1.1: Filter HTML to reduce content size (in this case we want to only keep <a> tags in the navigation areas)
            html_filter = HTMLFilter(html=main_html)
            filtered_html = html_filter.keep_only(tags=HTML_TAGS,
                                                classes=CSS_CLASSES, partial_match=True).keep_only(tags=["a"]).to_string()

            # Fallback in case the filtered_html is empty
            if len(filtered_html) == 0:
                logging.warning("None of the tags or classes were found, defaulting to fallback (all <a> tags)")
                filtered_html = html_filter.keep_only(tags=["a"]).to_string()

            logging.info(f"HTML was sanitized successfully for {state['url']} with {len(filtered_html)} bytes")

            # Step 2: Extract relevant <a> tag's href -> Relative links that takes us to "important" pages
            try:
                response = self.llm.invoke_model(prompt=EXTRACT_ROUTES_PROMPT,
                                                  output_schema=MainRoutesOutput,
                                                  input={"html_content": filtered_html})

                if isinstance(response, MainRoutesOutput):
                    routes = response.routes
                else:
                    response_data = response.model_dump()
                    routes = response_data.get("routes", [])

                observations = create_observation(observations=observations,
                                                observation_type=ObservationType.SUCCESS,
                                                message=f"Extracted {len(routes)} routes from main page. Extrated routes: {routes}",
                                                agent_type=self.agent_name)

                logging.info(f"Successfully extracted {len(routes)} from {state['url']}")
                return {
                    **state,
                    "routes_list": routes,
                    "obervations": observations
                }
            except Exception as e:
                observations = create_observation(observations=observations,
                                            observation_type=ObservationType.ERROR,
                                            message=f"LLM failed to extract routes: {e}",
                                            agent_type=self.agent_name)

                logging.error(f"Unexpected error in main_extractor_node: {e}")
                return {**state, "routes_list": [], "observations": observations}

        except Exception as e:
            observations = create_observation(observations=observations,
                                            observation_type=ObservationType.ERROR,
                                            message=f"Unexpected error in main extractor: {e}",
                                            agent_type=self.agent_name)

            logging.error(f"Unexpected error in main_extractor_node: {e}")
            return {**state, "routes_list": [], "observations": observations}

    def _main_extractor_validator_node(self, state: State) -> Literal["end", "continue"]:
        """
        Validates the main structure extraction

        Args:
            state (State): Graph state that the node receives
        Returns:
            Literal["end", "continue"]: Either `end` (if it pass the threshold) or `continue` if we need to go back to prev. node
        """
        if len(state["routes_list"]) > 0:
            return "end"
        else:
            observations = state.get("observations", [])
            observations = create_observation(observations=observations,
                                            observation_type=ObservationType.WARNING,
                                            message="No routes found, retrying main extractor",
                                            agent_type=self.agent_name)
            state["observations"] = observations

            return "continue"

    @retry_with_backoff()
    def _routes_data_extractor_node(self, state) -> dict[str, any]:
        """
        Handles the extraction of the routes content

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        base_url = state["url"]
        routes_list = state["routes_list"]

        # Check if "/" is in the list, if not add it
        if "/" not in routes_list:
            routes_list.append("/")

        observations = state.get("observations", [])

        validated_routes: dict[str, str] = {}
        validated_routes_list: list[str] = []
        failed_routes: list[str] = []

        def fetch_routes(base_url: str,
                         route: str) -> tuple[str, str | None]:
            """
            It calls the main logic of the HTMLFetcher instance

            Args:
                base_url (str): Site base URL
                route (str): Relative route to fetch
            """
            full_url = f"{base_url}{route}"

            try:
                result: FetchResult = self.html_fetcher.run(url=full_url)

                if result.status == FetchStatus.SUCCESSFUL and result.html_content is not None:
                    return route, result.html_content
                else:
                    logging.warning(f"Failed to fetch {route}: {result.error}")
                    return route, None
            except Exception as e:
                logging.error(f"Exception fetching {route}: {e}")
                return route, None


        for route in routes_list:
            try:
                # Step 1: Extract HTML for route
                route_path, html_content = fetch_routes(base_url, route)

                # Step 2: Filter and remove unnecessary HTML tags
                if html_content is not None:
                    html_filter = HTMLFilter(html=html_content)
                    filtered_html = html_filter.keep_only(tags=["body"]).remove(
                        tags=["head", "script", "style", "iframe", "noscript", "svg", "path", "nav", "footer", "aside"],
                        classes=["navbar", "footer"],
                        partial_match=True
                    ).keep_only(tags=["a"], include_parents=True).to_string()

                    validated_routes[route_path] = filtered_html
                    validated_routes_list.append(route_path)

                    observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.SUCCESS,
                        message=f"Successfully scraped route: {route_path}",
                        agent_type=self.agent_name
                    )

                    logging.info(f"Scraped successfully: {route_path}")
                else:
                    failed_routes.append(route_path)
                    observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.ERROR,
                        message=f"Failed to fetch HTML for route: {route_path}",
                        agent_type=self.agent_name
                    )

                    logging.error(f"Warning: Failed to fetch route {route_path}")

            except Exception as exc:
                failed_routes.append(route_path)
                observations = create_observation(
                    observations=observations,
                    observation_type=ObservationType.ERROR,
                    message=f"Unexpected error processing route: {route_path}",
                    agent_type=self.agent_name
                )

                logging.error(f"Error fetching route {route}: {exc}")

        observations = create_observation(
                    observations=observations,
                    observation_type=ObservationType.INFO,
                    message=f"Route extraction complete: {len(validated_routes_list)} successful ({validated_routes_list}), {len(failed_routes)} failed ({failed_routes})",
                    agent_type=self.agent_name
                )

        return {**state,
                "html_data_per_route": validated_routes,
                "routes_to_analyze": validated_routes_list,
                "observations": observations
            }

    @retry_with_backoff()
    def _articles_tags_extractor_node(self, state: State) -> dict[str, any]:
        """
        Handles the extraction of the articles CSS selector (for the list of articles)

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        url_dict: dict[str, str] = state.get("url_dict", {})
        observations = state.get("observations", [])
        retry_count: dict[str, int] = state.get("retry_count", {})
        max_retries = 3

        routes_to_analyze = state["routes_to_analyze"]
        html_per_route = state["html_data_per_route"]

        # Only analyze routes that haven't been successfully extracted yet and haven't exceeded max retries
        routes_to_process = [
            route for route in routes_to_analyze
            if route not in url_dict and retry_count.get(route, 0) < max_retries
        ]

        failed_extractions = []
        permanently_failed = []

        # Analyze routes that need processing
        for route in routes_to_process:
            sleep(3)
            logging.info("Adding delay of 3 seconds to avoid oversaturating the gemini API")
            try:
                # Increment retry count for this route
                retry_count[route] = retry_count.get(route, 0) + 1

                logging.info(f"Analyzing {route} (attempt {retry_count[route]}/{max_retries})...")
                response = self.llm.invoke_model(prompt=EXTRACT_ARTICLES_TAGS_PROMPT,
                                                 output_schema=ArticlesTagOutput,
                                                 input={"html_content": html_per_route[route]})

                if isinstance(response, ArticlesTagOutput):
                    data = {"css_selector": response.css_selector}
                else:
                    response_data = response.model_dump()
                    data = {"css_selector": response_data.get("css_selector")}

                if len(data["css_selector"]) > 0:
                    url_dict[route] = data["css_selector"]
                    logging.info(f"Successfully analyzed {route}")
                else:
                    failed_extractions.append(route)

                    if retry_count[route] >= max_retries:
                        permanently_failed.append(route)
                        observations = create_observation(
                            observations=observations,
                            observation_type=ObservationType.ERROR,
                            message=f"Route {route} exceeded max retries ({max_retries}) with empty CSS selector",
                            agent_type=self.agent_name
                        )
                    else:
                        observations = create_observation(
                            observations=observations,
                            observation_type=ObservationType.WARNING,
                            message=f"Empty CSS selector returned for route: {route} (attempt {retry_count[route]}/{max_retries})",
                            agent_type=self.agent_name
                        )

                    logging.info(f"Failed to extract css selector for: {route}")

            except Exception as e:
                failed_extractions.append(route)

                if retry_count[route] >= max_retries:
                    permanently_failed.append(route)
                    observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.ERROR,
                        message=f"Route {route} exceeded max retries ({max_retries}): {str(e)}",
                        agent_type=self.agent_name
                    )
                else:
                    observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.WARNING,
                        message=f"Error extracting CSS selector for route: {route} (attempt {retry_count[route]}/{max_retries})",
                        agent_type=self.agent_name
                    )

                logging.error(f"Error analyzing route {route}: {e}")

        successful_count = len(url_dict)
        pending_retry_count = len([r for r in failed_extractions if r not in permanently_failed])

        observations = create_observation(
            observations=observations,
            observation_type=ObservationType.INFO,
            message=f"CSS extraction status: {successful_count} successful, {pending_retry_count} pending retry, {len(permanently_failed)} permanently failed",
            agent_type=self.agent_name
        )
        logging.info(f"CSS extraction status: {successful_count} successful, {pending_retry_count} pending retry, {len(permanently_failed)} permanently failed")

        return {
            **state,
            "url_dict": url_dict,
            "observations": observations,
            "retry_count": retry_count
        }

    def _articles_extractor_validator_node(self, state: State) -> Literal["end", "continue"]:
        """
        Validates the routes css selector

        Args:
            state (State): Graph state that the node receives
        Returns:
            Literal["end", "continue"]: Either `end` (if it pass the threshold) or `continue` if we need to go back to prev. node
        """
        routes_count = len(state["routes_to_analyze"])
        extracted_count = len(state["url_dict"].keys())
        retry_count = state.get("retry_count", {})
        max_retries = 3

        routes_with_retries_available = sum(
            1 for route in state["routes_to_analyze"]
            if route not in state["url_dict"] and retry_count.get(route, 0) < max_retries
        )

        # If all routes are either extracted or have exceeded max retries, we're done
        if routes_with_retries_available == 0:
            observations = state.get("observations", [])

            if extracted_count < routes_count:
                failed_count = routes_count - extracted_count
                observations = create_observation(
                    observations=observations,
                    observation_type=ObservationType.WARNING,
                    message=f"CSS selector extraction complete with {failed_count} routes ignored after max retries ({extracted_count}/{routes_count} successful)",
                    agent_type=self.agent_name
                )
            else:
                observations = create_observation(
                    observations=observations,
                    observation_type=ObservationType.INFO,
                    message=f"CSS selector extraction complete: all {extracted_count} routes successful",
                    agent_type=self.agent_name
                )

            state["observations"] = observations
            return "end"
        else:
            observations = state.get("observations", [])
            observations = create_observation(
                observations=observations,
                observation_type=ObservationType.INFO,
                message=f"CSS selector extraction incomplete ({extracted_count}/{routes_count}), retrying {routes_with_retries_available} routes",
                agent_type=self.agent_name
            )

            state["observations"] = observations
            return "continue"

    @retry_with_backoff()
    def _get_articles_to_analyze_node(self, state: State) -> dict[str, any]:
        """
        Get and fetch 3 random articles using the current routes structure to later analyze.

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        url_dict = state["url_dict"]
        observations = state.get("observations", [])

        if not url_dict:
            observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.ERROR,
                        message="No routes with CSS selectors available",
                        agent_type=self.agent_name)
            return {**state, "articles_urls": [], "articles_html": [], "observations": observations}

        try:
            # Step 1: Get one 'random' key from the dict (one relative route)
            route_key = sample(list(url_dict.keys()), 1)[0]
            # Step 1.1: Create full URL based on the 'ranom' route
            url_to_scrape = f"{state['url']}{route_key}"

            observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.INFO,
                        message=f"Selected route for article sampling: {route_key}",
                        agent_type=self.agent_name)

            # Step 2: Fetch the full page content
            route_results = self.html_fetcher.run(url=url_to_scrape)

            if route_results.status != FetchStatus.SUCCESSFUL:
                observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.ERROR,
                        message="Failed to fetch route page",
                        agent_type=self.agent_name)

                logging.error(f"Failed to get HTML for {route_key} route")
                return {**state, "articles_urls": [], "articles_html": [], "observations": observations}

            route_html = route_results.html_content

            try:
                soup = BeautifulSoup(route_html, "html.parser")
                css_selector = url_dict[route_key]

                # Step 3: Find all links matching the CSS selector inside the route's HTML
                article_links = []
                for selector in css_selector.split(","):
                    links = soup.select(selector)

                    for link in links:
                        href = link.get("href")
                        if href:
                            if href.startswith("/"):
                                href = f"{state['url']}{href}"
                            article_links.append(href)
                            if len(article_links) >= 3:
                                break

                    if len(article_links) >= 3:
                        break

                # Ensure we only get 3 articles links (Keep 3 or just 1 -> ???)
                logging.info(f"Successfully extracted 3 articles to analyze ({article_links})")
                article_links = article_links[:3]
            except Exception as e:
                observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.ERROR,
                        message=f"Error parsing article links from HTML: {e}",
                        agent_type=self.agent_name)
                return {**state, "articles_urls": [], "articles_html": [], "observations": observations}

            # Step 4: Fetch the content for all articles
            articles_html = []
            failed_articles = []

            for article_url in article_links:
                try:
                    results = self.html_fetcher.run(url=article_url)
                    if results.status != FetchStatus.SUCCESSFUL:
                        failed_articles.append(article_url)
                        observations = create_observation(
                            observations=observations,
                            observation_type=ObservationType.WARNING,
                            message=f"Failed to fetch article: {article_url}",
                            agent_type=self.agent_name)

                        logging.error(f"Failed to fetch: {article_url}")
                        continue

                    html = results.html_content

                    # Step 4.1 Clean HTML content
                    html_filter = HTMLFilter(html=html)
                    filtered_html = html_filter.keep_only(tags=["body"]).remove(
                        tags=["script", "style", "iframe", "noscript", "svg", "path", "nav", "footer", "aside"],
                        classes=["navbar", "footer"],
                        partial_match=True
                    ).to_string()

                    articles_html.append(filtered_html)
                    logging.info(f"Successfully fetched: {article_url}")

                except Exception as e:
                    failed_articles.append(article_url)
                    observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.ERROR,
                        message="Error processing article",
                        agent_type=self.agent_name)

                    logging.error(f"Error processing article {article_url}: {e}")

            observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.INFO,
                        message=f"Article fetching complete: {len(articles_html)} successful, {len(failed_articles)} failed",
                        agent_type=self.agent_name)
            return {
                **state,
                "article_links": article_links,
                "articles_html": articles_html,
                "observations": observations
            }
        except Exception as e:
            observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.ERROR,
                        message=f"Unexpected error in article sampling: {e}",
                        agent_type=self.agent_name)

            logging.error(f"Unexpected error in get_articles_to_analyze_node: {e}")
            return {**state, "articles_urls": [], "articles_html": [], "observations": observations}

    def _articles_to_analyze_validator_node(self, state: State) -> Literal["end", "continue"]:
        """
        Validates the extractions of 3 random articles in the prev. node

        Args:
            state (State): Graph state that the node receives
        Returns:
            Literal["end", "continue"]: Either `end` (if it pass the threshold) or `continue` if we need to go back to prev. node
        """
        MIN_TOTAL_NEEDED_ARTICLES = 3
        total_extracted_articles = len(state.get("articles_html", []))

        if total_extracted_articles >= 3 or state["max_retry_analysis"] <= 0:
            return "end"
        else:
            observations = state.get("observations", [])
            observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.WARNING,
                        message=f"Missing articles for further analysis ({total_extracted_articles}/{MIN_TOTAL_NEEDED_ARTICLES}), retrying",
                        agent_type=self.agent_name
                    )

            state["max_retry_analysis"] = state["max_retry_analysis"] - 1
            state["observations"] = observations
            return "continue"

    def _validate_selectors(self, selectors: dict[str, str], articles_html: list[str]) -> dict[str, any]:
        """
        Validate selectors across sampled articles based on selectors importance

        Args:
            selectors (dict[str, str]): Dictionary containing the selectors name and it's CSS selector
            articles_html (list[str]): List of articles html
        """
        REQUIRED_FIELDS = ["title_selector", "content_selector"]
        OPTIONAL_FIELDS = ["subtitle_selector", "author_selector", "img_url_selector", "date_selector"]

        all_required_valid = True
        failed_required = []
        failed_optional = []

        try:
            for html in articles_html:
                soup = BeautifulSoup(html, "html.parser")

                # Check required selectors (MUST BE IN THE HTML)
                for key in REQUIRED_FIELDS:
                    selector = selectors.get(key)

                    if selector:
                        element = soup.select_one(selector)
                        if not element or not element.get_text(strip=True):
                            all_required_valid = False
                            failed_required.append(key)
                            break

                # Check optional selectors
                for key in OPTIONAL_FIELDS:
                    selector = selectors.get(key)
                    if selector:
                        element = soup.select_one(selector)
                        if not element or not element.get_text(strip=True):
                            failed_optional.append(key)

                if not all_required_valid:
                    break

            return {
                "all_valid": all_required_valid,
                "failed_required": list(set(failed_required)),
                "failed_optional": list(set(failed_optional)),
            }
        except Exception as e:
            logging.error(f"Unexpected error in _validate_selectors: {e}")
            return {
                "all_valid": False,
                "failed_required": REQUIRED_FIELDS,
                "failed_optional": OPTIONAL_FIELDS,
                "error": str(e)
            }

    def _find_consensus_selectors(self, responses: list[dict]) -> dict[str, any]:
        """
        Return selectors with fallback options for content, single values for others.

        Args:
            responses (list[dict])
        Returns:
            dict[str, str]: Dictionary containing the selectors based on all 3 articles
        """
        # Dynamic import -> Just when we use this method
        from collections import Counter

        consensus = {}

        FIELDS_WITH_FALLBACKS = ['content_selector']

        try:
            for key in responses[0].keys():
                values = [r[key] for r in responses if r.get(key)]

                if not values:
                    consensus[key] = None
                    continue

                counter = Counter(values)

                # For content_selector: store multiple options if they exist
                if key in FIELDS_WITH_FALLBACKS and len(counter) > 1:
                    consensus[key] = [selector for selector, _ in counter.most_common()]
                else:
                    # For all other fields: just use the most common
                    consensus[key] = counter.most_common(1)[0][0]

            return consensus
        except Exception as e:
            logging.error(f"Error finding consensus selectors: {e}")
            return responses[0] if responses else {} # Return first as fallback

    def _extract_from_multiple_articles(self, first_article_response: dict[str, any], articles_html: list[str]) -> dict[str, str]:
        """
        Analyze multiple articles and find consensus selectors

        Args:
            articles_html (list[str]): List of articles html
            first_article_response (dict[str, any]): Selectors from first article (not being analyze here)
        Returns:
            dict[str, str]: Dictionary with selectors
        """
        try:
            all_responses = [first_article_response]

            for html in articles_html:
                response = self.llm.invoke_model(prompt=EXTRACT_ARTICLE_CONTENT_PROMPT,
                                                 output_schema=ArticleContentOutput,
                                                 input={"html_content": html})

                if isinstance(response, ArticleContentOutput):
                    selectors = {"title_selector": response.title_selector,
                            "subtitle_selector": response.subtitle_selector,
                            "author_selector": response.author_selector,
                            "img_url_selector": response.img_url_selector,
                            "content_selector": response.content_selector,
                            "date_selector": response.date_selector}
                else:
                    response_data = response.model_dump()
                    selectors = {"title_selector": response_data.get("title_selector"),
                            "subtitle_selector": response_data.get("subtitle_selector"),
                            "author_selector": response_data.get("author_selector"),
                            "img_url_selector": response_data.get("img_url_selector"),
                            "content_selector": response_data.get("content_selector"),
                            "date_selector": response_data.get("date_selector")}

                all_responses.append(selectors)

            # Find most common selectors
            return self._find_consensus_selectors(all_responses)
        except Exception as e:
            logging.error(f"Error in _extract_from_multiple_articles: {e}")
            return first_article_response

    @retry_with_backoff()
    def _articles_content_extractor_node(self, state: State) -> dict[str, any]:
        """
        Extract needed CSS selector from fetched articles

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        logging.info("Analyzing and Extracting first article selectors")
        articles_html = state["articles_html"]
        observations = state.get("observations", [])
        max_retries = 3

        if not articles_html:
            observations = create_observation(
                observations=observations,
                observation_type=ObservationType.ERROR,
                message="No articles available for content extraction",
                agent_type=self.agent_name)
            return {
                **state,
                "title_selector": "",
                "subtitle_selector": None,
                "author_selector": None,
                "img_url_selector": None,
                "content_selector": [],
                "date_selector": None,
                "observations": observations
            }

        # Try multiple articles if the first ones fail due to safety filters
        articles_to_try = min(5, len(articles_html))  # Try up to 5 different articles

        for article_index in range(articles_to_try):
            retry_attempt = 0

            while retry_attempt < max_retries:
                try:
                    retry_attempt += 1

                    observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.INFO,
                        message=f"Analyzing article {article_index + 1} for content selectors (attempt {retry_attempt}/{max_retries})",
                        agent_type=self.agent_name)

                    # Step 1: Extract selectors from the current article HTML
                    response = self.llm.invoke_model(prompt=EXTRACT_ARTICLE_CONTENT_PROMPT,
                                                     output_schema=ArticleContentOutput,
                                                     input={"html_content": articles_html[article_index]})

                    if isinstance(response, ArticleContentOutput):
                        selectors = {
                            "title_selector": response.title_selector,
                            "subtitle_selector": response.subtitle_selector,
                            "author_selector": response.author_selector,
                            "img_url_selector": response.img_url_selector,
                            "content_selector": response.content_selector,
                            "date_selector": response.date_selector
                        }
                    else:
                        response_data = response.model_dump()
                        selectors = {
                            "title_selector": response_data.get("title_selector", ""),
                            "subtitle_selector": response_data.get("subtitle_selector", ""),
                            "author_selector": response_data.get("author_selector", ""),
                            "img_url_selector": response_data.get("img_url_selector", ""),
                            "content_selector": response_data.get("content_selector", []),
                            "date_selector": response_data.get("date_selector", "")
                        }

                    observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.INFO,
                        message=f"Successfully extracted selectors from article {article_index + 1}",
                        agent_type=self.agent_name)
                    logging.info(f"Successfully extracted selectors from article {article_index + 1}")

                    if len(articles_html) > 1:
                        # Step 2: Validate if selectors work in other articles
                        validation_results = self._validate_selectors(
                            selectors=selectors,
                            articles_html=articles_html[1:min(6, len(articles_html))]  # Validate on next few articles
                        )

                        if "error" in validation_results:
                            observations = create_observation(
                                observations=observations,
                                observation_type=ObservationType.ERROR,
                                message=f"Error during selector validation: {validation_results['error']}",
                                agent_type=self.agent_name)

                        if not validation_results["all_valid"]:
                            # Step 3: Analyze multiple articles for consensus
                            failed_required = validation_results["failed_required"]
                            failed_optional = validation_results["failed_optional"]

                            observations = create_observation(
                                observations=observations,
                                observation_type=ObservationType.WARNING,
                                message=f"Some selectors didn't pass validation ({[*failed_required, *failed_optional]})",
                                agent_type=self.agent_name)
                            logging.warning(f"Some selectors didn't pass validation ({[*failed_required, *failed_optional]})")

                            # Extract selectors from multiple articles for consensus
                            selectors = self._extract_from_multiple_articles(
                                first_article_response=selectors,
                                articles_html=articles_html[1:min(6, len(articles_html))]
                            )
                            observations = create_observation(
                                observations=observations,
                                observation_type=ObservationType.INFO,
                                message="Using consensus selectors from multiple articles",
                                agent_type=self.agent_name)
                        else:
                            logging.info("Selectors passed validation")

                    observations = create_observation(
                        observations=observations,
                        observation_type=ObservationType.INFO,
                        message=f"Selectors validation and extraction successful. Final selectors: {selectors}",
                        agent_type=self.agent_name)

                    return {
                        **state,
                        **selectors,
                        "content_selector": selectors["content_selector"] if isinstance(selectors["content_selector"], list) else [selectors["content_selector"]],
                        "observations": observations
                    }

                except Exception as e:
                    error_msg = str(e).lower()

                    # Detect safety filter blocks
                    if any(keyword in error_msg for keyword in ['safety', 'blocked', 'harm', 'policy', 'prohibited']):
                        logging.warning(f"Article {article_index + 1} blocked by safety filters: {e}")

                        observations = create_observation(
                            observations=observations,
                            observation_type=ObservationType.WARNING,
                            message=f"Article {article_index + 1} blocked by content safety filters, trying next article",
                            agent_type=self.agent_name)

                        # Break retry loop and try next article
                        break

                    else:
                        # Regular error - retry on same article
                        if retry_attempt >= max_retries:
                            logging.error(f"Max retries reached for article {article_index + 1}: {e}")
                            observations = create_observation(
                                observations=observations,
                                observation_type=ObservationType.ERROR,
                                message=f"Failed to extract from article {article_index + 1} after {max_retries} attempts: {str(e)}",
                                agent_type=self.agent_name)
                            # Break and try next article
                            break
                        else:
                            logging.warning(f"Error on article {article_index + 1}, attempt {retry_attempt}/{max_retries}: {e}")
                            observations = create_observation(
                                observations=observations,
                                observation_type=ObservationType.WARNING,
                                message=f"Error extracting from article {article_index + 1} (attempt {retry_attempt}/{max_retries}), retrying",
                                agent_type=self.agent_name)
                            # Continue retry loop
                            continue

        # If we've exhausted all articles
        observations = create_observation(
            observations=observations,
            observation_type=ObservationType.ERROR,
            message=f"Failed to extract selectors from any of the first {articles_to_try} articles",
            agent_type=self.agent_name)

        logging.error(f"Exhausted all {articles_to_try} articles without successful extraction")

        return {
            **state,
            "title_selector": "",
            "subtitle_selector": None,
            "author_selector": None,
            "img_url_selector": None,
            "content_selector": [],
            "date_selector": None,
            "observations": observations
        }

    def run(self, url: str) -> tuple[dict[str, any], dict[str, str], list[Observation]]:
        """
        Run WebInspector agent

        Args:
            url (str): News site URL

        Returns:
            tuple[dict[str, any], list[Observation]]: Tuple containing the final selectors in the first position, in the second the url_dict and in the last the observation made during the process
        """
        logging.info("Running WebInspector agent...")
        initial_state = State(
            url=url,
            routes_list=[],
            html_data_per_route={},
            routes_to_analyze=[],
            url_dict={},
            articles_urls=[],
            articles_html=[],

            # Scraping data
            title_selector="",
            subtitle_selector=None,
            author_selector=None,
            img_url_selector=None,
            date_selector=None,
            content_selector=[],

            # General data
            observation=[],

            # Error handling
            retry_count={},
            max_retry_analysis=3
        )

        result: State = self.graph.invoke(initial_state)

        selectors = {
            "title_selector": result["title_selector"],
            "subtitle_selector": result["subtitle_selector"],
            "author_selector": result["author_selector"],
            "img_url_selector": result["img_url_selector"],
            "content_selector": result["content_selector"],
            "date_selector": result["date_selector"]
        }

        return selectors, result["url_dict"], result["observations"]