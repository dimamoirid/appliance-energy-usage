import logging
from functools import wraps


def log_step(step_name, method=logging.info):
    """
    Decorator for logging start and finish statements before and after executing a function

    Args:
        step_name: (string) Text describing the step
        method: (callable) [print, logging.info, logging.debug]

    Returns:
        decorated function that prints or logs start/finish messages
    """
    def log_step_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            method(f"[ STARTING STEP: {step_name} ]")
            res = func(*args, **kwargs)
            method(f"[ FINISHED STEP: {step_name} ]")
            return res
        return wrapper
    return log_step_decorator
