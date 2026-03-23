import os
import logging
import requests

from src.agents.orchestrator.models.content import Observation, ObservationType, AgentType
from src.agents.orchestrator.utils.create_observation import create_observation
from src.external_tools.scraper.models.content import SiteConfigModel

class ScraperAPI:
    """
    Scraper API servie class. Allows us to communicate with the scraper server
    """
    def __init__(self,
                 site_dict: SiteConfigModel,
                 scraper_key: str = "SCRAPER_URL"):
        """
        Initializes ScraperAPI class

        Args:
            site_dict (SiteConfigModel):
            scraper_key (str):
        """
        self.url = os.getenv(scraper_key, None)
        self.site_dict = site_dict

        self.observations: list[Observation] = []

    def _check_health(self) -> tuple[bool, str | None]:
        """
        Method to check the health status of the Scraper API

        Returns:
            tuple[bool, str | None]: First value is a bool (True if server is healthy) and the second is an error msg in case something is wrong
        """
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.get(url=f"{self.url}/health",
                                    headers=headers)

            if not response.ok:
                raise Exception("Failed to check scraper health status")

            if response.status_code == 200:
                status = response.json()["status"]

                if status == "healthy":
                    logging.info("Scraper API is healthy and ready to be use")
                    return True, None
                else:
                    logging.info("Scraper API is unhealthy")
                    return False, "Scraper server is unhealthy"

            elif response.status_code == 503:
                raise Exception("The scraper server is unhealthy")
        except Exception as e:
            logging.error("Unexpected error in the scraper API")
            return False, f"Unexpected error in the scraper API: {e}"

    def _scrape_urls(self) -> tuple[list[tuple[dict[str, any], list[str]]] | None, str | None]:
        """
        Get all articles URLs for this news site

        Returns:
        tuple[list[tuple[dict[str, any], list[str]]] | None, str | None]:
            - List of tuples containing the site info and scraped urls
            - Error string if something fails or None
        """
        try:
            json = {"news_sites": [self.site_dict.model_dump()]}
            headers = {"Content-Type": "application/json"}

            response = requests.post(f"{self.url}/scrape-urls",
                                     headers=headers,
                                     json=json)

            if not response.ok:
                raise Exception(f"Failed to scrape urls for {self.url}")

            articles_tuple = response.json()

            logging.info(f"Sucessfully scraped {len(articles_tuple[0][1])} articles tuples")
            return articles_tuple, None
        except Exception as e:
            logging.error(f"Unexpected error when scraping {self.url} articles links: {e}")
            return None, f"Unexpected error when scraping {self.url} articles links: {e}"

    def _scrape_articles(self, articles_tuple: list[tuple[dict[str, any], list[str]]]) -> tuple[list[dict[str, any]] | None, str | None]:
        """
        Scrapes articles using url tuples and scraper

        Args:
            url_tuples (list[tuple[dict[str, any], list[str]]]): List of tuples containing sites info and it's latest urls
        Returns:
            tuple[list[dict[str, any]] | None, str | None]:
                - List of scraped articles (if no error)
                - Error string if something fails or None
        """
        try:
            json = {"link_tuples": articles_tuple}
            headers = {"Content-Type": "application/json"}

            response = requests.post(f"{self.url}/scrape-articles",
                                     headers=headers,
                                     json=json)

            if not response.ok:
                raise Exception(f"Failed to scrape articles for {self.url}")

            data = response.json()

            logging.info(f"Sucessfully scraped {len(data)} articles HTML content")
            return data, None
        except Exception as e:
            logging.error(f"Unexpected error when scraping {self.url} articles: {e}")
            return None, f"Unexpected error when scraping {self.url} articles: {e}"

    def _validate_performance(
        self,
        articles: list[dict[str, any]],
        url_tuples: list[tuple[dict[str, any], list[str]]]
    ) -> tuple[bool, str | None]:
        """
        Validates the performance of the scraper by checking article and field completeness

        Distinguishes between critical fields (title, content) and non-critical fields
        Provides detailed per-field statistics and logs URLs with critical field failures

        Args:
            articles: List of successfully scraped articles
            url_tuples: Original site info and scraped URLs
        Returns:
            Validation success status and a detailed summary message/error
        """
        if not articles:
            logging.error("Validation failed: No articles were scraped.")
            return False, "No articles were scraped, validation failed."

        # Configuration
        CRITICAL_FIELDS = {"title", "content"}
        NON_CRITICAL_FIELDS = {"subtitle", "date", "img_url"}
        ALL_FIELDS = CRITICAL_FIELDS | NON_CRITICAL_FIELDS

        MIN_CONTENT_LENGTH = 50
        SUCCESS_THRESHOLD_COMPLETENESS = 0.80
        SUCCESS_THRESHOLD_CRITICAL_FAILURE = 0.10

        # Calculate URL completeness
        all_expected_urls = [url for _, urls in url_tuples for url in urls]
        total_expected = len(all_expected_urls)
        total_scraped = len(articles)
        completeness_ratio = total_scraped / total_expected if total_expected > 0 else 0

        # Initialize tracking structures
        field_missing_count = {field: 0 for field in ALL_FIELDS}
        field_missing_count["content_quality"] = 0

        articles_with_critical_issues = []
        articles_with_any_missing = 0

        # Analyze each article
        for article in articles:
            has_any_missing = False
            has_critical_issue = False
            missing_fields_in_article = []

            for field in ALL_FIELDS:
                value = article.get(field)
                is_missing = False

                # Check if field is missing or empty
                if value is None or (isinstance(value, (str, list)) and not value):
                    is_missing = True
                # Special handling for content quality
                elif field == "content" and isinstance(value, list):
                    full_content = " ".join(value).strip()
                    if len(full_content) < MIN_CONTENT_LENGTH:
                        field_missing_count["content_quality"] += 1
                        is_missing = True
                        missing_fields_in_article.append(f"{field} (quality)")
                        logging.warning(f"Article ({article.get('url', 'URL not available')}) hasn't pass the content_quality validation")

                if is_missing:
                    field_missing_count[field] += 1
                    has_any_missing = True

                    if field in CRITICAL_FIELDS:
                        has_critical_issue = True
                        if field not in [mf.split(" ")[0] for mf in missing_fields_in_article]:
                            missing_fields_in_article.append(field)
                            logging.warning(f"Article ({article.get('url', 'URL not available')}) is missing critical field: {field}")

            if has_any_missing:
                articles_with_any_missing += 1

            if has_critical_issue:
                articles_with_critical_issues.append({
                    "url": article.get("url", "URL not available"),
                    "missing_fields": missing_fields_in_article
                })

        # Calculate rates
        critical_failure_rate = len(articles_with_critical_issues) / total_scraped if total_scraped > 0 else 0
        any_missing_rate = articles_with_any_missing / total_scraped if total_scraped > 0 else 0

        logging.info(f"Scraping Validation Report for {self.site_dict.name}")
        logging.info(f"URL Completeness: {total_scraped}/{total_expected} ({completeness_ratio:.1%})")
        logging.info(f"\nPer-Field Missing Statistics:")

        for field in sorted(ALL_FIELDS):
            count = field_missing_count[field]
            percentage = (count / total_scraped * 100) if total_scraped > 0 else 0
            field_type = "CRITICAL" if field in CRITICAL_FIELDS else "non-critical"
            logging.info(f"  • {field:12} [{field_type:12}]: {count:4}/{total_scraped} missing ({percentage:5.1f}%)")

        # Log content quality issues separately
        quality_count = field_missing_count["content_quality"]
        quality_pct = (quality_count / total_scraped * 100) if total_scraped > 0 else 0
        logging.info(f"  • {'content_quality':12} [CRITICAL    ]: {quality_count:4}/{total_scraped} too short ({quality_pct:5.1f}%)")

        # Log articles with critical issues
        if articles_with_critical_issues:
            logging.warning(f"\n{len(articles_with_critical_issues)} articles with CRITICAL field failures:")
            for i, issue in enumerate(articles_with_critical_issues[:10], 1):  # Log first 10
                logging.warning(f"  {i}. URL: {issue['url']}")
                logging.warning(f"     Missing: {', '.join(issue['missing_fields'])}")

            if len(articles_with_critical_issues) > 10:
                logging.warning(f"  ... and {len(articles_with_critical_issues) - 10} more")

        observations = []
        for field in NON_CRITICAL_FIELDS:
            count = field_missing_count[field]
            if count == total_scraped:
                observations.append(f"All articles missing '{field}' - likely not available on this site")
            elif count > total_scraped * 0.9:
                observations.append(f"Most articles missing '{field}' ({count}/{total_scraped})")

        if observations:
            logging.info("\nSite Observations:")
            for obs in observations:
                logging.info(f"  • {obs}")

        # Determine success
        is_successful = (
            completeness_ratio >= SUCCESS_THRESHOLD_COMPLETENESS and
            critical_failure_rate <= SUCCESS_THRESHOLD_CRITICAL_FAILURE
        )

        # Build summary message
        summary_message = (
            f"Scraping Performance Summary for **{self.site_dict.name}**:\n"
            f"* **URL Completeness**: {total_scraped}/{total_expected} articles ({completeness_ratio:.2%})\n"
            f"* **Critical Field Failures**: {len(articles_with_critical_issues)}/{total_scraped} articles ({critical_failure_rate:.2%})\n"
            f"* **Any Missing Fields**: {articles_with_any_missing}/{total_scraped} articles ({any_missing_rate:.2%})\n"
            f"* **Selectors  Count**: {field_missing_count}"
            f"* **Content Quality Issues**: {quality_count} articles below {MIN_CONTENT_LENGTH} chars\n"
        )

        if observations:
            summary_message += "\n**Observations**:\n"
            for obs in observations:
                summary_message += f"* {obs}\n"

        if is_successful:
            logging.info("Validation PASSED")
            return True, summary_message
        else:
            error_message = (
                f"Validation FAILED! Did not meet required thresholds.\n{summary_message}\n"
                f"Thresholds: Completeness ≥{SUCCESS_THRESHOLD_COMPLETENESS:.0%}, "
                f"Critical Failures ≤{SUCCESS_THRESHOLD_CRITICAL_FAILURE:.0%}"
            )
            logging.error(error_message)
            return False, error_message

    def run(self) -> tuple[bool, list[Observation]]:
        """
        Runs the full Scraper API test pipeline, including health checks, URL scraping, article content scraping, and performance validation.
        The method manages the sequence of operations and logs observations at each stage. If any critical step fails
        (e.g., missing URL, unhealthy server, or failed scrape), the pipeline stops immediately.

        Returns:
            Tuple[bool, List[Observation]]:
                - bool: True if the entire pipeline, including validation, was successful, False otherwise.
                - List[Observation]: A history of all logs/events generated during the pipeline execution.
        """
        if not self.url:
            self.observations = create_observation(
                observations=self.observations,
                observation_type=ObservationType.ERROR,
                message="Missing `scraper` url. It's a required .env to run this pipeline.",
                agent_type=AgentType.TESTER
            )

            logging.error("Missing `scraper` url. It's a required .env to run this pipeline.")
            return False, self.observations

        # Step 1: Check scraper health status
        is_healthy, error = self._check_health()
        if not is_healthy and error is not None:
            self.observations = create_observation(
                observations=self.observations,
                observation_type=ObservationType.ERROR,
                message="Scraper API server is not healthy. The pipeline can't run.",
                agent_type=AgentType.TESTER
            )

            logging.error("Scraper API server is not healthy. The pipeline can't run.")
            return False, self.observations

        # Step 2: Scrape urls
        articles_tuple, error = self._scrape_urls()
        if error is not None:
            self.observations = create_observation(
                observations=self.observations,
                observation_type=ObservationType.ERROR,
                message="Failed to scrape articles urls.",
                agent_type=AgentType.TESTER
            )

            logging.error("Failed to scrape articles urls.")
            return False, self.observations

        logging.info(f"Successfully scraped {len(articles_tuple[0][1])} articles urls")
        self.observations = create_observation(
                observations=self.observations,
                observation_type=ObservationType.INFO,
                message=f"Successfully scraped {len(articles_tuple[0][1])} articles urls",
                agent_type=AgentType.TESTER
            )

        # Step 3: Scrape articles
        articles_html_content, error = self._scrape_articles(articles_tuple=articles_tuple)
        if error is not None:
            self.observations = create_observation(
                observations=self.observations,
                observation_type=ObservationType.ERROR,
                message="Failed to scrape articles HTML content.",
                agent_type=AgentType.TESTER
            )

            logging.error("Failed to scrape articles HTML content.")
            return False, self.observations

        logging.info(f"Successfully scraped {len(articles_html_content)} articles HTML content.")
        self.observations = create_observation(
                observations=self.observations,
                observation_type=ObservationType.INFO,
                message=f"Successfully scraped {len(articles_html_content)} articles HTML content.",
                agent_type=AgentType.TESTER
            )

        # Step 4: Perform validation of content based on selectors
        is_valid, validation_msg = self._validate_performance(articles=articles_html_content,
                                   url_tuples=articles_tuple)

        observation_type = ObservationType.INFO if is_valid else ObservationType.ERROR
        self.observations = create_observation(
            observations=self.observations,
            observation_type=observation_type,
            message=f"Validation result after scraper API pipeline:\n{validation_msg}",
            agent_type=AgentType.TESTER
        )

        return is_valid, self.observations


