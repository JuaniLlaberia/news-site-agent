import os
import logging
import time

from langchain_google_genai import (ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold)
from google.api_core.exceptions import ResourceExhausted
from pydantic import BaseModel

class Gemini:
    """
    Light wrapper for one or more `ChatGoogleGenerativeAI` model instances.

    This class manages multiple Gemini models with round-robin key rotation
    and intelligent retry logic that distinguishes between rate limits and quota exhaustion.
    """
    def __init__(self,
                 temperature: float = 0.05,
                 top_p: float = 0.3,
                 top_k: int = 10,
                 requests_per_key_daily: int = 20,
                 max_retries: int = 3,
                 retry_delay: float = 2.0) -> None:
        """
        Initializes Gemini class instance

        Args:
            temperature (float): Controls the randomness of the output.
            top_p (float): Lowering top_p narrows the field of possible tokens.
            top_k (int): Limits the token selection to the top_k most likely tokens at each step.
            requests_per_key_daily (int): Maximum requests per API key per day.
            max_retries (int): Maximum retries per key when hitting rate limits.
            retry_delay (float): Seconds to wait between retries.
        """
        self.api_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_API_KEY_BACKUP_1"),
            os.getenv("GOOGLE_API_KEY_BACKUP_2"),
        ]
        self.api_keys = [key for key in self.api_keys if key]

        if not self.api_keys:
            raise ValueError("No Google API keys found in environment variables")

        self.successful_usage = {key: 0 for key in self.api_keys}
        self.daily_limit = requests_per_key_daily
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Model hyperparameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.current_key_index = 0

        logging.info(f"Initialized Gemini with {len(self.api_keys)} API keys")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Determine if the error is a rate limit (per-minute) vs quota exhaustion (daily).

        Args:
            error: The exception raised.
        Returns:
            bool: True if it's a rate limit error (retry-able), False if quota exhausted.
        """
        error_str = str(error).lower()

        rate_limit_indicators = [
            "rate limit",
            "quota exceeded",
            "resource has been exhausted",
            "429",
            "rpm",
        ]

        return any(indicator in error_str for indicator in rate_limit_indicators)

    def _get_next_available_key_index(self) -> int | None:
        """
        Find the next available API key that hasn't exceeded its daily limit.

        Returns:
            int | None: Index of the next available key, or None if all are exhausted.
        """
        # Try each key starting from current position
        for i in range(len(self.api_keys)):
            key_index = (self.current_key_index + i) % len(self.api_keys)
            key = self.api_keys[key_index]

            if self.successful_usage[key] < self.daily_limit:
                self.current_key_index = (key_index + 1) % len(self.api_keys)
                return key_index

        return None

    def _create_model(self,
                      api_key: str,
                      name: str) -> ChatGoogleGenerativeAI:
        """
        Initializes and returns a ChatGoogleGenerativeAI model with customizable
        generation settings.

        Args:
            api_key (str): The Google API key.
            name (str): Model instance name for logging and debugging.
        Returns:
            ChatGoogleGenerativeAI: An instance of the initialized model.
        """
        model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            },
            max_retries=0,
            name=name
        )
        return model

    def invoke_model(self,
                     prompt: any,
                     output_schema: BaseModel,
                     input: dict[str, any]) -> any:
        """
        Invoke the configured model chain with structured output.

        Automatically rotates through available API keys with intelligent retry logic:
        - Rate limit errors: Retry with same key after delay, then try other keys
        - Quota exhausted: Move to next key immediately
        - Only successful requests count against daily quota

        Args:
            prompt: A prompt object or prompt string compatible with the
                langchain prompt operators used here.
            output_schema: A Pydantic `BaseModel` class describing the
                structured output schema.
            input: A dictionary of inputs to pass to the chain's `invoke` call.
        Returns:
            The raw result returned by the chain.
        Raises:
            ResourceExhausted: If all configured models are exhausted.
            Exception: Any unexpected exception raised while invoking the chain.
        """
        attempted_keys = set()
        last_exception = None

        while len(attempted_keys) < len(self.api_keys):
            # Get next available key
            primary_key_index = self._get_next_available_key_index()

            if primary_key_index is None:
                raise ResourceExhausted(
                    f"All {len(self.api_keys)} API keys have exhausted their daily quota "
                    f"({self.daily_limit} requests each)"
                )

            if primary_key_index in attempted_keys:
                continue

            attempted_keys.add(primary_key_index)
            primary_key = self.api_keys[primary_key_index]

            for retry in range(self.max_retries):
                try:
                    primary_model = self._create_model(
                        api_key=primary_key,
                        name=f"llm_key_{primary_key_index + 1}"
                    )

                    structured_llm = primary_model.with_structured_output(output_schema)
                    chain = prompt | structured_llm

                    result = chain.invoke(input)

                    self.successful_usage[primary_key] += 1
                    logging.info(
                        f"Chain successfully ran with key #{primary_key_index + 1} "
                        f"(usage: {self.successful_usage[primary_key]}/{self.daily_limit})"
                    )
                    return result

                except ResourceExhausted as e:
                    last_exception = e

                    if self._is_rate_limit_error(e):
                        if retry < self.max_retries - 1:
                            wait_time = self.retry_delay * (retry + 1)
                            logging.warning(
                                f"Rate limit hit on key #{primary_key_index + 1}, "
                                f"retrying in {wait_time}s (attempt {retry + 2}/{self.max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            logging.warning(
                                f"Rate limit persists on key #{primary_key_index + 1} after "
                                f"{self.max_retries} retries, trying next key"
                            )
                            break
                    else:
                        logging.warning(
                            f"Daily quota exhausted on key #{primary_key_index + 1}, trying next key"
                        )
                        self.successful_usage[primary_key] = self.daily_limit
                        break

                except Exception as e:
                    logging.error(f"Unexpected error with key #{primary_key_index + 1}: {e}")
                    last_exception = e
                    break

        if last_exception:
            raise last_exception
        else:
            raise ResourceExhausted("All API keys failed")