"""Unit tests for OllamaClient."""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from utils.ollama_client import OllamaClient, Priority, Request, Response


@pytest.fixture
def config_path():
    """Path to test configuration."""
    return Path(__file__).parent.parent / "config" / "models.yaml"


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client."""
    with patch("utils.ollama_client.ollama.Client") as mock:
        yield mock


@pytest.fixture
def client(config_path, mock_ollama_client):
    """Create OllamaClient instance with mocked backend."""
    return OllamaClient(config_path=config_path)


class TestOllamaClientInitialization:
    """Test client initialization."""

    def test_initialization(self, client):
        """Test client initializes correctly."""
        assert client.endpoint == "http://192.168.254.17:11434"
        assert client.config is not None
        assert "models" in client.config
        assert "endpoint" in client.config

    def test_load_config(self, config_path):
        """Test configuration loading."""
        with patch("utils.ollama_client.ollama.Client"):
            client = OllamaClient(config_path=config_path)
            assert "phi3-mini" in client.config["models"]
            assert "deepseek-coder" in client.config["models"]
            assert "qwen-1.5b" in client.config["models"]
            assert "qwen-7b" in client.config["models"]
            assert "tinyllama" in client.config["models"]

    def test_get_model_config(self, client):
        """Test retrieving model configuration."""
        config = client.get_model_config("phi3-mini")
        assert config["size"] == "3.8B"
        assert config["role"] == "leader"
        assert config["timeout"] == 30

    def test_get_model_config_not_found(self, client):
        """Test error when model not found."""
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            client.get_model_config("nonexistent")

    def test_list_models(self, client):
        """Test listing all models."""
        models = client.list_models()
        assert "phi3:mini" in models
        assert "deepseek-coder:1.3b" in models
        assert "qwen2.5:1.5b" in models
        assert "qwen2.5:7b" in models
        assert "tinyllama:1.1b" in models


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        mock_response = {"response": "test"}
        client.client.generate = Mock(return_value=mock_response)

        result = client.health_check("phi3:mini")
        assert result is True
        assert client.is_model_healthy("phi3:mini") is True

    def test_health_check_failure(self, client):
        """Test failed health check."""
        client.client.generate = Mock(side_effect=Exception("Connection error"))

        result = client.health_check("phi3:mini")
        assert result is False
        assert client.is_model_healthy("phi3:mini") is False

    def test_is_model_healthy_unknown(self, client):
        """Test checking health of unknown model."""
        # Should assume healthy if unknown
        assert client.is_model_healthy("unknown-model") is True


class TestGenerate:
    """Test generate method."""

    def test_generate_success(self, client):
        """Test successful generation."""
        mock_result = {
            "response": "This is a test response",
            "eval_count": 10,
        }
        client.client.generate = Mock(return_value=mock_result)

        response = client.generate(model="phi3:mini", prompt="Test prompt")

        assert response.success is True
        assert response.content == "This is a test response"
        assert response.tokens_used == 10
        assert response.model == "phi3:mini"
        assert response.error is None

    def test_generate_with_custom_params(self, client):
        """Test generation with custom parameters."""
        mock_result = {"response": "Test", "eval_count": 5}
        client.client.generate = Mock(return_value=mock_result)

        response = client.generate(
            model="phi3:mini",
            prompt="Test",
            temperature=0.5,
            max_tokens=500,
            priority=Priority.HIGH,
        )

        assert response.success is True
        client.client.generate.assert_called_once()
        call_args = client.client.generate.call_args
        assert call_args[1]["options"]["temperature"] == 0.5
        assert call_args[1]["options"]["num_predict"] == 500

    def test_generate_uses_config_defaults(self, client):
        """Test that generate uses config defaults."""
        mock_result = {"response": "Test", "eval_count": 5}
        client.client.generate = Mock(return_value=mock_result)

        response = client.generate(model="phi3:mini", prompt="Test")

        call_args = client.client.generate.call_args
        # Should use temperature from config (0.4 for phi3:mini)
        assert call_args[1]["options"]["temperature"] == 0.4
        assert call_args[1]["options"]["num_predict"] == 1000

    def test_generate_failure_all_retries(self, client):
        """Test generation failure after all retries."""
        client.client.generate = Mock(side_effect=Exception("Connection failed"))

        response = client.generate(model="phi3:mini", prompt="Test")

        assert response.success is False
        assert response.error == "Connection failed"
        assert response.tokens_used == 0
        # Should have tried 3 times (max_attempts)
        assert client.client.generate.call_count == 3

    @patch("time.sleep")  # Mock sleep to speed up test
    def test_generate_retry_logic(self, mock_sleep, client):
        """Test exponential backoff retry logic."""
        # Fail twice, succeed on third attempt
        mock_result = {"response": "Success", "eval_count": 5}
        client.client.generate = Mock(
            side_effect=[
                Exception("Error 1"),
                Exception("Error 2"),
                mock_result,
            ]
        )

        response = client.generate(model="phi3:mini", prompt="Test")

        assert response.success is True
        assert response.content == "Success"
        # Should have called sleep twice (before 2nd and 3rd attempts)
        assert mock_sleep.call_count == 2
        # Check backoff delays: [1, 2, 4]
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)


class TestStatistics:
    """Test statistics tracking."""

    def test_statistics_initial(self, client):
        """Test initial statistics."""
        stats = client.get_statistics()
        assert stats["requests_total"] == 0
        assert stats["requests_success"] == 0
        assert stats["requests_failed"] == 0
        assert stats["retries_total"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["failure_rate"] == 0.0

    def test_statistics_after_success(self, client):
        """Test statistics after successful request."""
        mock_result = {"response": "Test", "eval_count": 5}
        client.client.generate = Mock(return_value=mock_result)

        client.generate(model="phi3:mini", prompt="Test")

        stats = client.get_statistics()
        assert stats["requests_total"] == 1
        assert stats["requests_success"] == 1
        assert stats["requests_failed"] == 0
        assert stats["success_rate"] == 1.0

    def test_statistics_after_failure(self, client):
        """Test statistics after failed request."""
        client.client.generate = Mock(side_effect=Exception("Error"))

        client.generate(model="phi3:mini", prompt="Test")

        stats = client.get_statistics()
        assert stats["requests_total"] == 1
        assert stats["requests_success"] == 0
        assert stats["requests_failed"] == 1
        assert stats["failure_rate"] == 1.0
        assert stats["retries_total"] == 2  # Failed 3 times, so 2 retries

    @patch("time.sleep")
    def test_statistics_after_retry_success(self, mock_sleep, client):
        """Test statistics after successful retry."""
        mock_result = {"response": "Success", "eval_count": 5}
        client.client.generate = Mock(
            side_effect=[Exception("Error"), mock_result]
        )

        client.generate(model="phi3:mini", prompt="Test")

        stats = client.get_statistics()
        assert stats["requests_total"] == 1
        assert stats["requests_success"] == 1
        assert stats["retries_total"] == 1  # 1 retry before success

    def test_reset_statistics(self, client):
        """Test resetting statistics."""
        mock_result = {"response": "Test", "eval_count": 5}
        client.client.generate = Mock(return_value=mock_result)

        client.generate(model="phi3:mini", prompt="Test")
        assert client.get_statistics()["requests_total"] == 1

        client.reset_statistics()
        stats = client.get_statistics()
        assert stats["requests_total"] == 0
        assert stats["requests_success"] == 0


class TestPriority:
    """Test priority handling."""

    def test_priority_comparison(self):
        """Test priority enum comparison."""
        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH < Priority.NORMAL
        assert Priority.NORMAL < Priority.LOW

    def test_request_priority_ordering(self):
        """Test requests are ordered by priority."""
        req1 = Request(
            model="test",
            prompt="test1",
            priority=Priority.LOW,
            timestamp=1.0,
            timeout=30,
        )
        req2 = Request(
            model="test",
            prompt="test2",
            priority=Priority.CRITICAL,
            timestamp=2.0,
            timeout=30,
        )
        req3 = Request(
            model="test",
            prompt="test3",
            priority=Priority.NORMAL,
            timestamp=1.5,
            timeout=30,
        )

        # Sort by priority (critical first)
        sorted_reqs = sorted([req1, req2, req3])
        assert sorted_reqs[0].priority == Priority.CRITICAL
        assert sorted_reqs[1].priority == Priority.NORMAL
        assert sorted_reqs[2].priority == Priority.LOW

    def test_request_timestamp_tiebreaker(self):
        """Test timestamp breaks priority ties."""
        req1 = Request(
            model="test",
            prompt="test1",
            priority=Priority.NORMAL,
            timestamp=2.0,
            timeout=30,
        )
        req2 = Request(
            model="test",
            prompt="test2",
            priority=Priority.NORMAL,
            timestamp=1.0,
            timeout=30,
        )

        sorted_reqs = sorted([req1, req2])
        # Earlier timestamp should come first
        assert sorted_reqs[0].timestamp == 1.0
        assert sorted_reqs[1].timestamp == 2.0


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_statistics_updates(self, client):
        """Test statistics updates are thread-safe."""
        import threading

        mock_result = {"response": "Test", "eval_count": 5}
        client.client.generate = Mock(return_value=mock_result)

        def make_request():
            client.generate(model="phi3:mini", prompt="Test")

        threads = [threading.Thread(target=make_request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = client.get_statistics()
        assert stats["requests_total"] == 10
        assert stats["requests_success"] == 10
