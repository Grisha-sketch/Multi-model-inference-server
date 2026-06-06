import time
import logging
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Create logs folder if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logger to write to logs/requests.log
logging.basicConfig(
    filename="logs/requests.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("request_logger")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Intercepts every HTTP request and logs:
    - HTTP method and path
    - Response status code
    - Time taken (in ms)
    Runs automatically on every request — no manual calls needed.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # Calculate how long it took
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Log it
        log_message = (
            f"{request.method} {request.url.path} | "
            f"status={response.status_code} | "
            f"latency={duration_ms}ms"
        )
        logger.info(log_message)
        print(f"[LOG] {log_message}")

        return response