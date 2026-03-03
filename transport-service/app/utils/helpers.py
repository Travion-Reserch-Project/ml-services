import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def log_execution(func: Callable) -> Callable:
    """Decorator to log function execution"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        logger.info(f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"✓ {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"✗ {func.__name__} failed: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        logger.info(f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✓ {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"✗ {func.__name__} failed: {e}")
            raise
    
    # Return async wrapper if function is async, else sync
    if hasattr(func, '__await__'):
        return async_wrapper
    return sync_wrapper


def validate_distance(distance: float) -> bool:
    """Validate distance value"""
    return 2 <= distance <= 120


def validate_traffic_score(score: float) -> bool:
    """Validate traffic score value"""
    return 0 <= score <= 1


def validate_binary_field(value: int) -> bool:
    """Validate binary field (0 or 1)"""
    return value in (0, 1)
