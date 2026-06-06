from collections import defaultdict
import time


class MetricsTracker:
    """
    Tracks performance metrics for each model.
    Stores: request count, total latency, error count.
    """

    def __init__(self):
        self._request_count = defaultdict(int)     # { "model:version": count }
        self._total_latency = defaultdict(float)   # { "model:version": total_ms }
        self._error_count = defaultdict(int)       # { "model:version": count }

    def record_request(self, model_name: str, version: str, latency_ms: float):
        """Record a successful prediction request."""
        key = f"{model_name}:{version}"
        self._request_count[key] += 1
        self._total_latency[key] += latency_ms

    def record_error(self, model_name: str, version: str):
        """Record a failed prediction request."""
        key = f"{model_name}:{version}"
        self._error_count[key] += 1

    def get_metrics(self):
        """Return all metrics as a summary dict."""
        result = {}

        all_keys = set(self._request_count) | set(self._error_count)

        for key in all_keys:
            count = self._request_count[key]
            avg_latency = (
                round(self._total_latency[key] / count, 2) if count > 0 else 0
            )
            result[key] = {
                "request_count": count,
                "avg_latency_ms": avg_latency,
                "total_latency_ms": round(self._total_latency[key], 2),
                "error_count": self._error_count[key]
            }

        return result


# Single shared instance used across the app
tracker = MetricsTracker()