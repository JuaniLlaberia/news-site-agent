import logging
import requests
import time
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Comment
from .models.types import FetchResult, FetchStatus

class SecurityError(Exception):
    """
    Raised for security-related issues
    """
    pass

class ContentError(Exception):
    """
    Raised for page content issues (redirects, paywalls, no content, etc...)
    """
    pass

class FetchError(Exception):
    """
    Raised when fetching fails
    """
    pass

class HTMLFetcher:
    """
    HTML fetcher using requests (Works for static sites)
    """
    def __init__(self,
                 min_delay: float = 1.0,
                 timeout: int = 10):
        """
        Initializes a new instance of the HTMLFetcher class

        Args:
            min_delay (float): Minimum delay between requests
            timeout (int): Request timeout in seconds
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
        self.min_delay = min_delay
        self.timeout = timeout
        self.last_request_time = {}

    def _fetch(self, url: str, timeout: int = 10) -> str:
        """
        Fetch HTML using current session. It can be one or multiple urls

        Args:
            url (str): URL to fetch
            timeout (int): Request timeout in seconds

        Returns:
            str: HTML content

        Raises:
            SecurityError: If URL fails security validation
            ContentError: If content is empty
            FetchError: If fetch operation fails
        """
        # URL Validation Step
        validated_url = self._validate_url(url)

        # URL Ratelimit Step
        self._rate_limit(validated_url)

        try:
            logging.info(f"Fetching {validated_url} HTML...")
            response = self.session.get(
                url=validated_url,
                timeout=(5, timeout), # First is connect time, second is read time
                allow_redirects=False,
                stream=False,
                verify=True
            )
            response.raise_for_status()

            logging.info(f"URL response with code {response.status_code}")

            if response.status_code in (301, 302, 303, 307, 308, 402):
                logging.error(f"The URL needs a suscriptions to enter ({validated_url})")
                raise ContentError(f"The URL needs a suscriptions to enter ({validated_url}).")
            elif response.status_code == 404:
                logging.error(f"Page {validated_url} not found")
                raise ContentError(f"Page {validated_url} not found.")
            elif response.status_code == 200 and response.text != None:
                sanitized_html = self._sanitize_html(response_content=response.text)
                logging.info(f"HTML was sanitized successfully for {url} with {len(sanitized_html)} bytes")

                return sanitized_html

        except requests.exceptions.Timeout:
            logging.error(f"Request timed out after {timeout}s")
            raise FetchError(f"Request timed out after {timeout}s.")
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error: {e}")
            raise FetchError(f"HTTP error: {e}.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            raise FetchError(f"Request failed: {e}")

    def _validate_url(self, url: str) -> str | None:
        """
        Performs validation checks on URL

        Args:
            url (str): URL to validate

        Returns:
            str: The validated url in case is valid, else None

        Raises:
            SecurityError: If the URL is not valid
        """
        parsed_url = urlparse(url)

        if not parsed_url.scheme or not parsed_url.netloc or not parsed_url.hostname:
            raise SecurityError(f"Invalid URL: {url} is not a valid url.")

        if parsed_url.scheme != "https":
            raise SecurityError(f"Invalid URL scheme: {parsed_url.scheme} is not 'https'.")

        localhost_variations = ['localhost', '0.0.0.0', '127.0.0.1', '::1', 'local', 'localdomain']

        hostname_lower = parsed_url.hostname.lower()
        if any(hostname_lower == loc or hostname_lower.endswith(f'.{loc}') for loc in localhost_variations):
            raise SecurityError(f"Access to localhost blocked: {parsed_url.hostname}")

        return url

    def _rate_limit(self, url: str) -> None:
        """
        Rate limit requests per domain

        Args:
            url (str): URL to fetch
        """
        domain = urlparse(url).hostname

        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]

            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed
                logging.info(f"Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
                time.sleep(sleep_time)

        self.last_request_time[domain] = time.time()

    def _sanitize_html(self, response_content: str) -> str:
        """
        Cleans HTML content by removing specific tags and comments for security and to reduce noise

        Args:
            response_content (str): The raw HTML content.

        Returns:
            str: The sanitized HTML as string.
        """
        soup = BeautifulSoup(response_content, 'lxml')

        # HTML tags to remove
        tags_to_remove = [
            'script',
            'style',
            'link',
            'meta',
            'noscript',
            'svg',
            'form',
            'iframe',
        ]

        # Remove all instances of the specified tags
        for tag_name in tags_to_remove:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()

        clean_html = str(soup)
        # Remove excessive newlines and spaces before returning
        clean_html = re.sub(r'\n\s*\n', '\n', clean_html)
        clean_html = re.sub(r' +', ' ', clean_html)

        return clean_html.strip()

    def run(self, url: str) -> FetchResult:
        """
        Runs HTML fetcher pipeline to get sanitized HTML content

        Args:
            url (str): URL fetch

        Returns:
            FetchResult: Pydantic object with URL fetch results
        """
        logging.info("Running HTML fetcher pipeline...")
        start_time = time.time()

        try:
            html = self._fetch(url=url, timeout=self.timeout)
            return FetchResult(
                status=FetchStatus.SUCCESSFUL,
                html_content=html,
                url=url,
                fetch_duration=time.time() - start_time,
                error=None
            )

        except SecurityError as e:
            logging.error(f"Security error when fetching {url}: {e}")
            return FetchResult(
                status=FetchStatus.ERROR,
                html_content=None,
                url=url,
                fetch_duration=time.time() - start_time,
                error=f"Security error: {e}"
            )
        except ContentError as e:
            logging.error(f"Content error when fetching {url}: {e}")
            return FetchResult(
                status=FetchStatus.ERROR,
                html_content=None,
                url=url,
                fetch_duration=time.time() - start_time,
                error=f"Content error: {e}"
            )
        except FetchError as e:
            logging.error(f"Fetch error {url}: {e}")
            return FetchResult(
                status=FetchStatus.ERROR,
                html_content=None,
                url=url,
                fetch_duration=time.time() - start_time,
                error=f"Fetch error: {e}"
            )
        except Exception as e:
            logging.error(f"Unexpected error when fetching {url}: {e}")
            return FetchResult(
                status=FetchStatus.ERROR,
                html_content=None,
                url=url,
                fetch_duration=time.time() - start_time,
                error=f"Unexpected error: {e}"
            )
