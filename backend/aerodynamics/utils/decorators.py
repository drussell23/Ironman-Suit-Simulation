"""
Common decorators for timing, deprecation, retry, caching, and type validation.
"""
import time
import logging
import warnings
import functools
import inspect

logger = logging.getLogger(__name__)


def timeit(func):
    """
    Measure execution time of a function and log at DEBUG level.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} executed in {elapsed:.6f}s")
        return result
    return wrapper


def deprecated(reason: str = ""):
    """
    Mark a function as deprecated.

    Args:
        reason: Optional explanation for deprecation.
    """
    def decorator(func):
        msg = f"Function {func.__name__} is deprecated. {reason}".strip()
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry(exceptions: tuple = (Exception,), tries: int = 3, delay: float = 0.0):
    """
    Retry a function upon exception.

    Args:
        exceptions: Tuple of exception types to catch.
        tries: Number of attempts.
        delay: Delay between retries in seconds.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for _ in range(tries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logger.warning(f"Retrying {func.__name__} after exception: {e}")
                    if delay:
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def lru_cache(maxsize: int = 128):
    """
    LRU cache decorator using functools.lru_cache.

    Args:
        maxsize: Maximum cache size.
    """
    def decorator(func):
        return functools.lru_cache(maxsize=maxsize)(func)
    return decorator


def validate_types(func):
    """
    Validate argument types against function annotations.
    Raises TypeError on mismatch.
    """
    sig = inspect.signature(func)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            if name in func.__annotations__:
                expected = func.__annotations__[name]
                if not isinstance(value, expected):
                    raise TypeError(f"{func.__name__}: argument '{name}' must be {expected}, got {type(value)}")
        result = func(*args, **kwargs)
        if 'return' in func.__annotations__:
            expected = func.__annotations__['return']
            if not isinstance(result, expected):
                raise TypeError(f"{func.__name__}: return must be {expected}, got {type(result)}")
        return result
    return wrapper