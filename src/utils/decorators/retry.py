import logging
from time import sleep
from functools import wraps
from typing import Tuple, Type, Callable, Any
from langchain_core.exceptions import OutputParserException
from json import JSONDecodeError
from pydantic import ValidationError

def retry_with_backoff(
    max_retries: int = 3,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OutputParserException,
        JSONDecodeError,
        ValidationError
    ),
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    logger: logging.Logger = None
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts
        retryable_exceptions (tuple[Exceptions]): Tuple of exceptions that should trigger a retry
        base_delay (float): Base delay in seconds for exponential backoff
        exponential_base (float): Base for exponential calculation (delay = base_delay * base^attempt)
        logger (Logger): Logger instance to use (defaults to function's module logger)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            log = logger or logging.getLogger(f"{func.__module__}.{func.__name__}")

            for attempt in range(max_retries + 1):
                log.info(f"Executing {func.__name__}: Attempt #{attempt + 1}/{max_retries}")

                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        log.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                    return result

                except retryable_exceptions as e:
                    log.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")

                    if attempt < max_retries:
                        delay = base_delay * (exponential_base ** attempt)

                        log.info(f"Waiting {delay:.2f}s before next attempt")
                        sleep(delay)
                    else:
                        log.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise e

                except Exception as e:
                    log.error(f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}")
                    raise e

        return wrapper
    return decorator