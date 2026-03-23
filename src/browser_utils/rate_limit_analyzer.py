import requests
import logging
from time import sleep, time
from dataclasses import dataclass, field
from statistics import mean, median

from src.agents.orchestrator.models.content import Observation, ObservationType, AgentType
from src.agents.orchestrator.utils.create_observation import create_observation

@dataclass
class RateLimitMetrics:
    """
    Metrics for a single study case
    """
    case_id: int
    requests_sent: int
    requests_successful: int
    requests_blocked: int
    requests_failed: int
    delay_between_requests: float
    response_codes: dict[int, int] = field(default_factory=dict)
    response_times: list[float] = field(default_factory=list)
    avg_response_time: float = 0.0
    median_response_time: float = 0.0
    total_duration: float = 0.0
    rate_limit_triggered: bool = False
    rate_limit_headers: dict[str, str] | None = None

    def calculate_stats(self):
        """Calculate aggregate statistics"""
        if self.response_times:
            self.avg_response_time = mean(self.response_times)
            self.median_response_time = median(self.response_times)

class RateLimitAnalyzer:
    """
    Rate Limiter to analyze and understand site rate limits
    """
    RATE_LIMIT_STATUS_CODES = {429, 503, 403}
    RATE_LIMIT_HEADERS = [
        'x-ratelimit-limit',
        'x-ratelimit-remaining',
        'x-ratelimit-reset',
        'x-rate-limit-limit',
        'x-rate-limit-remaining',
        'x-rate-limit-reset',
        'retry-after',
        'ratelimit-limit',
        'ratelimit-remaining',
        'ratelimit-reset',
    ]

    def __init__(self, url: str, timeout: int = 10):
        """
        Initializes rate limit analyzer class
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        })
        self.url = url
        self.timeout = timeout
        self.observations: list[Observation] = []
        self.metrics: list[RateLimitMetrics] = []

        self.study_cases = {
            1: {"requests": 10, "delay": 2},
            2: {"requests": 10, "delay": 1},
            3: {"requests": 10, "delay": 0.5},
            4: {"requests": 10, "delay": 0.1},
            5: {"requests": 20, "delay": 0}
        }

    def _extract_rate_limit_headers(self, response: requests.Response) -> dict[str, str]:
        """
        Extract rate limit related headers from response

        Args:
            response (Reponse): Web browser response
        Returns:
            dict[str, str]: Dictionary containing rate-limit headers and it's values (in case they are present)
        """
        rate_limit_info = {}

        for header in self.RATE_LIMIT_HEADERS:
            value = response.headers.get(header)
            if value:
                rate_limit_info[header] = value

        return rate_limit_info

    def _is_rate_limited(self, response: requests.Response) -> bool:
        """
        Check if response indicates rate limiting

        Args:
            response (Reponse): Web browser response
        Returns:
            bool: True if the status code is one of the rate limit ones or False if not
        """
        return response.status_code in self.RATE_LIMIT_STATUS_CODES

    def _validate_response(self, response: requests.Response, metrics: RateLimitMetrics) -> None:
        """
        Validate and categorize the response

        Args:
            response (Reponse): Web browser response
            metrics (RateLimitMetrics): Metrics object
        """
        status_code = response.status_code

        metrics.response_codes[status_code] = metrics.response_codes.get(status_code, 0) + 1

        if self._is_rate_limited(response):
            metrics.rate_limit_triggered = True
            metrics.requests_blocked += 1

            if not metrics.rate_limit_headers:
                metrics.rate_limit_headers = self._extract_rate_limit_headers(response)

            self.observations  = create_observation(
                    observations=self.observations,
                    observation_type=ObservationType.WARNING,
                    message=f"Rate limit detected. Status: {status_code}, Headers: {metrics.rate_limit_headers}",
                    agent_type=AgentType.RATE_LIMIT_TEST)
            logging.warning(f"Rate limit detected. Status: {status_code}, Headers: {metrics.rate_limit_headers}")

        elif 200 <= status_code < 300 or status_code in (301, 302, 303, 307, 308):
            metrics.requests_successful += 1
        else:
            metrics.requests_failed += 1

            self.observations  = create_observation(
                    observations=self.observations,
                    observation_type=ObservationType.WARNING,
                    message=f"Request failed with status code: {status_code}",
                    agent_type=AgentType.RATE_LIMIT_TEST)
            logging.warning(f"Request failed with status code: {status_code}")

    def _run_study_case(self, case_id: int, requests_amount: int, delay: float) -> RateLimitMetrics:
        """
        Run a single study case and return metrics

        Args:
            case_id (int): Case study id
            requests_amount (int): Number of requests
            delay (float): Delay (in seconds)
        Returns:
            RateLimitMetrics: Metrics object
        """
        metrics = RateLimitMetrics(
            case_id=case_id,
            requests_sent=0,
            requests_successful=0,
            requests_blocked=0,
            requests_failed=0,
            delay_between_requests=delay
        )

        logging.info(f"Starting case {case_id}: {requests_amount} requests with {delay}s delay")

        case_start = time()
        for i in range(requests_amount):
            if i > 0:  # Don't delay before first request
                sleep(delay)

            try:
                request_start = time()
                response = self.session.get(
                    url=self.url,
                    allow_redirects=False,
                    stream=False,
                    verify=True,
                    timeout=self.timeout
                )
                request_time = time() - request_start

                metrics.requests_sent += 1
                metrics.response_times.append(request_time)

                self._validate_response(response, metrics)

                logging.info(f"Request {i+1}/{requests_amount}: Status {response.status_code}, Time {request_time:.3f}s")

            except requests.exceptions.Timeout:
                metrics.requests_failed += 1

                self.observations  = create_observation(
                    observations=self.observations,
                    observation_type=ObservationType.ERROR,
                    message=f"Request {i+1}/{requests_amount} timed out",
                    agent_type=AgentType.RATE_LIMIT_TEST)
                logging.error(f"Request {i+1}/{requests_amount} timed out")

            except requests.exceptions.ConnectionError as e:
                metrics.requests_failed += 1

                self.observations  = create_observation(
                    observations=self.observations,
                    observation_type=ObservationType.ERROR,
                    message=f"Request {i+1}/{requests_amount} connection error: {e}",
                    agent_type=AgentType.RATE_LIMIT_TEST)
                logging.error(f"Request {i+1}/{requests_amount} connection error: {e}")

            except Exception as e:
                metrics.requests_failed += 1

                self.observations  = create_observation(
                    observations=self.observations,
                    observation_type=ObservationType.ERROR,
                    message=f"Request {i+1}/{requests_amount} unexpected error: {e}",
                    agent_type=AgentType.RATE_LIMIT_TEST)
                logging.error(f"Request {i+1}/{requests_amount} unexpected error: {e}")

        metrics.total_duration = time() - case_start
        metrics.calculate_stats()

        return metrics

    def _run_study_cases(self, cooldown_between_cases: int = 10) -> list[RateLimitMetrics]:
        """
        Run all study cases with cooldown periods between them

        Args:
            cooldown_between_cases (int): Cooldown between study cases so it "resets"
        Returns:
            list[RateLimitMetrics]: List of metrics for all case studies
        """
        for case_id, case_study in self.study_cases.items():
            requests_amount = case_study["requests"]
            delay = case_study["delay"]

            # Run the study case
            metrics = self._run_study_case(case_id, requests_amount, delay)
            self.metrics.append(metrics)

            study_case_msg = f"""
                Case {case_id} complete:
                - {metrics.requests_successful} successful
                - {metrics.requests_blocked} blocked
                - {metrics.requests_failed} failed
                - Rate limited: {metrics.rate_limit_triggered}
            """
            self.observations  = create_observation(
                    observations=self.observations,
                    observation_type=ObservationType.INFO,
                    message=study_case_msg,
                    agent_type=AgentType.RATE_LIMIT_TEST)
            logging.info(study_case_msg)

            # Cooldown between cases (except after last case)
            if case_id < len(self.study_cases):
                logging.info(f"Cooling down for {cooldown_between_cases}s...")
                sleep(cooldown_between_cases)

        self.observations  = create_observation(
                    observations=self.observations,
                    observation_type=ObservationType.SUCCESS,
                    message="All rate limit case studies ran successfully",
                    agent_type=AgentType.RATE_LIMIT_TEST)

        logging.info("Rate limit analysis complete")
        return self.metrics

    def _calculate_recommended_delay(self, min_delay: float = 1, safety_margin: float = 1.5) -> dict[str, any]:
        """
        Calculate the recommended delay between requests based on test results.

        Args:
            min_delay (float): Minimum delay if no rate limiting detected (seconds)
            safety_margin (float): Multiplier for safety buffer (default 1.5 = 50% extra)
        Returns:
            dict[str, any]: Dictionary with recommended delay and reasoning
        """
        if not self.metrics:
            logging.info("No metrics were found. Returning min_delay")
            return {
                "recommended_delay": min_delay,
                "reasoning": "No test data available",
                "rate_limited": False
            }

        # Find the first case that got blocked
        blocked_cases = [m for m in self.metrics if m.rate_limit_triggered]

        # Min delay in case there was NO block
        if not blocked_cases:
            logging.info("No study case failed. Returning min_delay")
            return {
                "recommended_delay": min_delay,
                "reasoning": f"No rate limiting detected. Safe to use {min_delay}s delay.",
                "rate_limited": False
            }

        # Find the fastest delay that caused blocking
        first_blocked = min(blocked_cases, key=lambda m: m.delay_between_requests)
        blocked_delay = first_blocked.delay_between_requests

        # Find successful cases that were slower than the blocked one
        successful_cases = [
            m for m in self.metrics
            if not m.rate_limit_triggered and m.delay_between_requests > blocked_delay
        ]

        if successful_cases:
            # Use the slowest successful case + safety margin
            safe_case = min(successful_cases, key=lambda m: m.delay_between_requests)
            recommended = safe_case.delay_between_requests * safety_margin
            reasoning = f"Blocked at {blocked_delay}s, successful at {safe_case.delay_between_requests}s. Using {recommended:.2f}s with safety margin."
        else:
            # No successful cases slower than blocking, extrapolate
            recommended = blocked_delay * safety_margin * 2
            reasoning = f"Blocked at {blocked_delay}s. No successful slower cases found. Using {recommended:.2f}s (conservative estimate)."

        return {
            "recommended_delay": round(recommended, 3),
            "reasoning": reasoning,
            "rate_limited": True,
            "blocked_at": blocked_delay
        }

    def run(self, min_delay: float = 0.1, safety_margin: float = 1.5) -> tuple[float, list[Observation]]:
        """
        Run the rate limit analyzer flow

        Args:
            min_delay (float): Minimum delay if no rate limiting detected (seconds)
            safety_margin (float): Multiplier for safety buffer
        Returns:
            tuple[float, list[Observation]]: First value is the rate_limit_delay val (float) and second the list of observations
        """
        logging.info(f"Starting rate limit analysis on {self.url}")

        COOLDOWN_NUM = 10
        # Run flow to calculate rate limit delay
        self._run_study_cases(cooldown_between_cases=COOLDOWN_NUM)
        rate_limit_delay_info = self._calculate_recommended_delay(min_delay=min_delay,
                                                                  safety_margin=safety_margin)

        self.observations  = create_observation(
                    observations=self.observations,
                    observation_type=ObservationType.INFO,
                    message=f"Sucessfully calculated delay: {rate_limit_delay_info}",
                    agent_type=AgentType.RATE_LIMIT_TEST)
        logging.info(f"Sucessfully ran rate limit analyzer")

        return rate_limit_delay_info["recommended_delay"], self.observations
