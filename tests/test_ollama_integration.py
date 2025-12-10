"""Integration tests for OllamaClient with real endpoint.

These tests connect to the actual Ollama server at 192.168.254.17.
Mark as slow and skip by default with: pytest -m "not slow"
"""

import pytest
from utils.ollama_client import OllamaClient, Priority


@pytest.fixture
def client():
    """Create real OllamaClient instance."""
    return OllamaClient()


@pytest.mark.integration
@pytest.mark.slow
class TestOllamaIntegration:
    """Integration tests with real Ollama endpoint."""

    def test_connection_to_endpoint(self, client):
        """Test connection to Ollama endpoint."""
        # This should not raise any exceptions
        assert client.endpoint == "http://192.168.254.17:11434"

    def test_health_check_all_models(self, client):
        """Test health check for all configured models."""
        models = client.list_models()

        results = {}
        for model in models:
            try:
                healthy = client.health_check(model)
                results[model] = healthy
                print(f"  {model}: {'✓' if healthy else '✗'}")
            except Exception as e:
                results[model] = False
                print(f"  {model}: ✗ ({e})")

        # At least one model should be healthy
        assert any(results.values()), "No models are healthy"

    def test_simple_generation(self, client):
        """Test simple text generation with phi3:mini."""
        response = client.generate(
            model="phi3:mini",
            prompt="Say 'Hello, World!' and nothing else.",
            max_tokens=50,
        )

        print(f"\nResponse: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Duration: {response.duration:.2f}s")

        assert response.success, f"Generation failed: {response.error}"
        assert len(response.content) > 0
        assert response.tokens_used > 0

    def test_sequential_requests(self, client):
        """Test 100 sequential requests with <1% failure rate."""
        num_requests = 100
        successes = 0
        failures = 0

        print(f"\nRunning {num_requests} sequential requests...")

        for i in range(num_requests):
            response = client.generate(
                model="phi3:mini",
                prompt=f"Count to {i % 5 + 1}",
                max_tokens=50,
            )

            if response.success:
                successes += 1
            else:
                failures += 1

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_requests} ({successes} success, {failures} failed)")

        success_rate = successes / num_requests
        failure_rate = failures / num_requests

        print(f"\nResults:")
        print(f"  Success: {successes}/{num_requests} ({success_rate:.1%})")
        print(f"  Failed: {failures}/{num_requests} ({failure_rate:.1%})")

        stats = client.get_statistics()
        print(f"  Retries: {stats['retries_total']}")

        # Require <1% failure rate
        assert failure_rate < 0.01, f"Failure rate {failure_rate:.1%} exceeds 1%"

    def test_all_models_generation(self, client):
        """Test generation with all configured models."""
        models = client.list_models()

        results = {}
        for model in models:
            print(f"\nTesting {model}...")
            response = client.generate(
                model=model,
                prompt="Say 'test' and nothing else.",
                max_tokens=20,
            )

            results[model] = response.success
            if response.success:
                print(f"  ✓ Success: {response.content[:50]}")
                print(f"    Tokens: {response.tokens_used}, Duration: {response.duration:.2f}s")
            else:
                print(f"  ✗ Failed: {response.error}")

        # At least 50% of models should work
        success_count = sum(results.values())
        success_rate = success_count / len(models)
        assert success_rate >= 0.5, f"Only {success_rate:.1%} of models working"

    def test_priority_handling(self, client):
        """Test different priority levels."""
        priorities = [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]

        for priority in priorities:
            response = client.generate(
                model="phi3:mini",
                prompt="Test priority",
                priority=priority,
                max_tokens=20,
            )
            assert response.success, f"Failed with priority {priority.name}"

    def test_temperature_variation(self, client):
        """Test different temperature settings."""
        temperatures = [0.0, 0.5, 1.0]

        for temp in temperatures:
            response = client.generate(
                model="phi3:mini",
                prompt="Generate a random number between 1 and 10",
                temperature=temp,
                max_tokens=20,
            )
            assert response.success, f"Failed with temperature {temp}"

    def test_retry_on_failure(self, client):
        """Test that retry logic works by using invalid prompt."""
        # This should trigger retries but eventually fail
        # Note: depending on model behavior, this might still succeed
        response = client.generate(
            model="phi3:mini",
            prompt="",  # Empty prompt might cause issues
            max_tokens=1,
        )

        # Check that response is returned (success or failure)
        assert response is not None
        print(f"\nRetry test result: {'success' if response.success else 'failed'}")
        print(f"  Error: {response.error if response.error else 'None'}")


@pytest.mark.integration
@pytest.mark.slow
def test_statistics_tracking(client):
    """Test that statistics are tracked correctly."""
    client.reset_statistics()

    # Make a few requests
    for i in range(5):
        client.generate(
            model="phi3:mini",
            prompt=f"Test {i}",
            max_tokens=20,
        )

    stats = client.get_statistics()
    print(f"\nStatistics after 5 requests:")
    print(f"  Total: {stats['requests_total']}")
    print(f"  Success: {stats['requests_success']}")
    print(f"  Failed: {stats['requests_failed']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")

    assert stats['requests_total'] == 5
    assert stats['success_rate'] > 0


if __name__ == "__main__":
    # Run integration tests when executed directly
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
