import os
import functools
from flask_jwt_extended import verify_jwt_in_request
from flask_jwt_extended.exceptions import JWTExtendedException


def restrict_in_tasks(restricted_tasks):
    """
    Decorator that enforces JWT auth only if the current RGMC_TASK_ENV
    is in the provided list of restricted_tasks.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            current_env = os.environ.get('RGMC_TASK_ENV')

            if current_env in restricted_tasks:
                try:
                    verify_jwt_in_request()
                except JWTExtendedException as e:
                    return {
                        "msg": f"Action forbidden. Endpoint is locked during '{current_env}' task."
                    }, 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator
