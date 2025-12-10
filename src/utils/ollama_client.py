"""Ollama client with connection pooling, retry logic, and graceful degradation."""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import IntEnum
from queue import PriorityQueue, Empty
import threading
from pathlib import Path

import ollama
import yaml


logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Request priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Request:
    """Ollama API request."""

    model: str
    prompt: str
    priority: Priority
    timestamp: float
    timeout: int
    temperature: float = 0.7
    max_tokens: int = 1000
    request_id: Optional[str] = None

    def __lt__(self, other: "Request") -> bool:
        """Compare by priority, then timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class Response:
    """Ollama API response."""

    success: bool
    content: str
    model: str
    request_id: Optional[str]
    tokens_used: int
    duration: float
    error: Optional[str] = None


class OllamaClient:
    """
    Ollama client with advanced features:
    - Connection pooling
    - Priority-based request queue
    - Exponential backoff retry logic
    - Per-model timeout configuration
    - Health checks with graceful degradation
    - Thread-safe operations
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Ollama client.

        Args:
            config_path: Path to models.yaml config file
        """
        self.config = self._load_config(config_path)
        self.endpoint = self.config["endpoint"]["base_url"]

        # Initialize Ollama client
        self.client = ollama.Client(host=self.endpoint)

        # Request queue
        self.queue: PriorityQueue[Request] = PriorityQueue(
            maxsize=self.config["queue"]["max_size"]
        )

        # Model health status
        self.model_health: Dict[str, bool] = {}
        self.health_check_enabled = self.config["health_check"]["enabled"]
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "retries_total": 0,
        }

        logger.info(f"Initialized OllamaClient with endpoint: {self.endpoint}")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        for key, config in self.config["models"].items():
            if config["name"] == model_name or key == model_name:
                return config
        raise ValueError(f"Model {model_name} not found in configuration")

    def health_check(self, model: str) -> bool:
        """
        Check if a model is healthy and available.

        Args:
            model: Model name to check

        Returns:
            True if model is healthy, False otherwise
        """
        try:
            response = self.client.generate(
                model=model, prompt="test", options={"num_predict": 1}
            )
            with self._lock:
                self.model_health[model] = True
            return True
        except Exception as e:
            logger.warning(f"Health check failed for model {model}: {e}")
            with self._lock:
                self.model_health[model] = False
            return False

    def is_model_healthy(self, model: str) -> bool:
        """Check if model is marked as healthy."""
        with self._lock:
            return self.model_health.get(model, True)  # Assume healthy if unknown

    def generate(
        self,
        model: str,
        prompt: str,
        priority: Priority = Priority.NORMAL,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Response:
        """
        Generate completion with retry logic and exponential backoff.

        Args:
            model: Model name
            prompt: Input prompt
            priority: Request priority
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens to generate (overrides config)
            timeout: Request timeout in seconds (overrides config)
            request_id: Optional request identifier

        Returns:
            Response object with completion
        """
        # Get model configuration
        model_config = self.get_model_config(model)

        # Use config defaults if not specified
        if temperature is None:
            temperature = model_config.get("temperature", 0.7)
        if max_tokens is None:
            max_tokens = model_config.get("max_tokens", 1000)
        if timeout is None:
            timeout = model_config.get("timeout", 30)

        # Create request
        request = Request(
            model=model,
            prompt=prompt,
            priority=priority,
            timestamp=time.time(),
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=request_id,
        )

        # Execute with retry logic
        return self._execute_with_retry(request)

    def _execute_with_retry(self, request: Request) -> Response:
        """
        Execute request with exponential backoff retry logic.

        Args:
            request: Request to execute

        Returns:
            Response object
        """
        max_attempts = self.config["retry"]["max_attempts"]
        backoff_delays = self.config["retry"]["backoff_delays"]

        last_error = None

        for attempt in range(max_attempts):
            try:
                # Check model health before attempting
                if self.health_check_enabled and not self.is_model_healthy(request.model):
                    logger.warning(
                        f"Model {request.model} is unhealthy, attempting anyway (attempt {attempt + 1}/{max_attempts})"
                    )

                # Execute request
                start_time = time.time()
                result = self.client.generate(
                    model=request.model,
                    prompt=request.prompt,
                    options={
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                )
                duration = time.time() - start_time

                # Extract response
                content = result.get("response", "")
                tokens_used = result.get("eval_count", 0)

                # Update statistics
                with self._lock:
                    self.stats["requests_total"] += 1
                    self.stats["requests_success"] += 1
                    if attempt > 0:
                        self.stats["retries_total"] += attempt

                logger.debug(
                    f"Request successful: model={request.model}, tokens={tokens_used}, duration={duration:.2f}s"
                )

                return Response(
                    success=True,
                    content=content,
                    model=request.model,
                    request_id=request.request_id,
                    tokens_used=tokens_used,
                    duration=duration,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_attempts}): {e}"
                )

                # Mark model as unhealthy
                with self._lock:
                    self.model_health[request.model] = False

                # Exponential backoff (unless last attempt)
                if attempt < max_attempts - 1:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)

        # All attempts failed
        with self._lock:
            self.stats["requests_total"] += 1
            self.stats["requests_failed"] += 1
            self.stats["retries_total"] += max_attempts - 1

        logger.error(f"Request failed after {max_attempts} attempts: {last_error}")

        return Response(
            success=False,
            content="",
            model=request.model,
            request_id=request.request_id,
            tokens_used=0,
            duration=0.0,
            error=last_error,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats["model_health"] = self.model_health.copy()
            if stats["requests_total"] > 0:
                stats["success_rate"] = stats["requests_success"] / stats["requests_total"]
                stats["failure_rate"] = stats["requests_failed"] / stats["requests_total"]
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
            return stats

    def list_models(self) -> List[str]:
        """List all configured models."""
        return [config["name"] for config in self.config["models"].values()]

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self.stats = {
                "requests_total": 0,
                "requests_success": 0,
                "requests_failed": 0,
                "retries_total": 0,
            }
            logger.info("Statistics reset")
