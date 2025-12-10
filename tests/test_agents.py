"""Tests for agent framework."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Tuple, Optional

from agents.base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentState,
    Permission,
)
from agents.registry import AgentRegistry, load_agent_configs, get_registry
from communication.message import Message, MessageType, Priority


# Test implementation of BaseAgent
class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def process_task(self, task: Any, context: Dict[str, Any]) -> Optional[Message]:
        """Process a task."""
        self.state.tasks_completed += 1
        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.TASK_RESULT,
            priority=Priority.NORMAL,
            content={"result": "success"},
        )

    def validate_output(self, output: Any) -> Tuple[bool, str]:
        """Validate output."""
        if output is None:
            return False, "Output is None"
        return True, "Valid"


@pytest.fixture
def mock_ollama():
    """Create mock Ollama client."""
    client = Mock()
    client.generate = Mock(
        return_value=Mock(
            success=True,
            content="Test response",
            tokens_used=10,
        )
    )
    return client


@pytest.fixture
def agent_config():
    """Create test agent config."""
    return AgentConfig(
        name="test_agent",
        role=AgentRole.LEADER,
        model="phi3:mini",
        temperature=0.5,
        max_tokens=500,
        token_budget=2000,
        permissions={
            "current_goal": Permission.READ_WRITE,
            "facts": Permission.READ,
        },
        system_prompt="You are a test agent.",
    )


@pytest.fixture
def mock_agent(agent_config, mock_ollama):
    """Create mock agent."""
    return MockAgent(agent_config, mock_ollama)


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_config_creation(self, agent_config):
        """Test creating agent config."""
        assert agent_config.name == "test_agent"
        assert agent_config.role == AgentRole.LEADER
        assert agent_config.model == "phi3:mini"
        assert agent_config.temperature == 0.5

    def test_config_to_dict(self, agent_config):
        """Test config serialization."""
        data = agent_config.to_dict()
        assert data["name"] == "test_agent"
        assert data["role"] == "leader"
        assert data["permissions"]["current_goal"] == "read_write"


class TestAgentState:
    """Test AgentState dataclass."""

    def test_state_creation(self):
        """Test creating agent state."""
        state = AgentState()
        assert state.tokens_used == 0
        assert state.tasks_completed == 0
        assert state.context_window == []

    def test_state_serialization(self):
        """Test state serialization."""
        state = AgentState(tokens_used=100, tasks_completed=5)
        data = state.to_dict()
        assert data["tokens_used"] == 100
        assert data["tasks_completed"] == 5

        restored = AgentState.from_dict(data)
        assert restored.tokens_used == 100


class TestBaseAgent:
    """Test BaseAgent class."""

    def test_agent_creation(self, mock_agent):
        """Test creating an agent."""
        assert mock_agent.name == "test_agent"
        assert mock_agent.role == AgentRole.LEADER
        assert mock_agent.model == "phi3:mini"
        assert not mock_agent.is_active

    def test_agent_activation(self, mock_agent):
        """Test agent activation/deactivation."""
        mock_agent.activate()
        assert mock_agent.is_active
        assert mock_agent.state.last_active > 0

        mock_agent.deactivate()
        assert not mock_agent.is_active

    def test_token_budget(self, mock_agent):
        """Test token budget management."""
        assert mock_agent.tokens_remaining == 2000

        mock_agent.use_tokens(500)
        assert mock_agent.tokens_remaining == 1500

        mock_agent.reset_budget()
        assert mock_agent.tokens_remaining == 2000

    def test_token_estimation(self, mock_agent):
        """Test token estimation."""
        tokens = mock_agent.estimate_tokens("Hello world")
        assert tokens == 2  # ~4 chars per token

    def test_permission_check(self, mock_agent):
        """Test permission checking."""
        assert mock_agent.has_permission("current_goal", Permission.READ)
        assert mock_agent.has_permission("current_goal", Permission.WRITE)
        assert mock_agent.has_permission("facts", Permission.READ)
        assert not mock_agent.has_permission("facts", Permission.WRITE)
        assert not mock_agent.has_permission("unknown", Permission.READ)

    def test_generate(self, mock_agent):
        """Test text generation."""
        content, tokens = mock_agent.generate("Test prompt")
        assert content == "Test response"
        assert tokens == 10
        assert mock_agent.state.tokens_used == 10

    def test_generate_budget_exceeded(self, mock_agent):
        """Test generation with insufficient budget."""
        mock_agent.use_tokens(1990)  # Only 10 tokens left

        content, tokens = mock_agent.generate("Long prompt " * 100)
        assert content == ""
        assert tokens == 0

    def test_process_task(self, mock_agent):
        """Test task processing."""
        result = mock_agent.process_task({}, {})
        assert result is not None
        assert result.type == MessageType.TASK_RESULT
        assert mock_agent.state.tasks_completed == 1

    def test_validate_output(self, mock_agent):
        """Test output validation."""
        valid, reason = mock_agent.validate_output("test")
        assert valid is True

        valid, reason = mock_agent.validate_output(None)
        assert valid is False

    def test_context_window(self, mock_agent):
        """Test context window management."""
        for i in range(15):
            mock_agent.add_to_context({"entry": i})

        context = mock_agent.get_context()
        assert len(context) == 10  # Max entries

        mock_agent.clear_context()
        assert len(mock_agent.get_context()) == 0

    def test_state_persistence(self, mock_agent):
        """Test state save/load."""
        mock_agent.use_tokens(100)
        mock_agent.state.tasks_completed = 5

        state_data = mock_agent.save_state()
        assert state_data["state"]["tokens_used"] == 100

        # Create new agent and load state
        mock_agent.reset_budget()
        mock_agent.load_state(state_data)
        assert mock_agent.state.tasks_completed == 5


class TestAgentRegistry:
    """Test AgentRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create registry."""
        return AgentRegistry()

    def test_register_class(self, registry):
        """Test registering agent class."""
        registry.register_class("mock", MockAgent)
        assert "mock" in registry.list_classes()

    def test_register_invalid_class(self, registry):
        """Test registering invalid class."""
        with pytest.raises(ValueError):
            registry.register_class("invalid", dict)

    def test_create_agent(self, registry, agent_config, mock_ollama):
        """Test creating agent from registry."""
        registry.register_class("mock", MockAgent)
        agent = registry.create_agent("mock", agent_config, mock_ollama)

        assert agent.name == "test_agent"
        assert "test_agent" in registry.list_agents()

    def test_get_agent(self, registry, agent_config, mock_ollama):
        """Test getting agent by name."""
        registry.register_class("mock", MockAgent)
        registry.create_agent("mock", agent_config, mock_ollama)

        agent = registry.get_agent("test_agent")
        assert agent is not None
        assert agent.name == "test_agent"

    def test_get_agents_by_role(self, registry, mock_ollama):
        """Test getting agents by role."""
        registry.register_class("mock", MockAgent)

        # Create agents with different roles
        config1 = AgentConfig(
            name="leader1", role=AgentRole.LEADER, model="phi3:mini"
        )
        config2 = AgentConfig(
            name="coder1", role=AgentRole.CODE_SPECIALIST, model="deepseek"
        )
        config3 = AgentConfig(
            name="leader2", role=AgentRole.LEADER, model="phi3:mini"
        )

        registry.create_agent("mock", config1, mock_ollama)
        registry.create_agent("mock", config2, mock_ollama)
        registry.create_agent("mock", config3, mock_ollama)

        leaders = registry.get_agents_by_role(AgentRole.LEADER)
        assert len(leaders) == 2

    def test_remove_agent(self, registry, agent_config, mock_ollama):
        """Test removing agent."""
        registry.register_class("mock", MockAgent)
        registry.create_agent("mock", agent_config, mock_ollama)

        assert registry.remove_agent("test_agent")
        assert registry.get_agent("test_agent") is None

    def test_activate_deactivate_all(self, registry, agent_config, mock_ollama):
        """Test bulk activation/deactivation."""
        registry.register_class("mock", MockAgent)
        registry.create_agent("mock", agent_config, mock_ollama)

        registry.activate_all()
        agent = registry.get_agent("test_agent")
        assert agent.is_active

        registry.deactivate_all()
        assert not agent.is_active

    def test_statistics(self, registry, agent_config, mock_ollama):
        """Test registry statistics."""
        registry.register_class("mock", MockAgent)
        registry.create_agent("mock", agent_config, mock_ollama)

        stats = registry.get_statistics()
        assert stats["total_agents"] == 1
        assert stats["active_agents"] == 0


class TestGlobalRegistry:
    """Test global registry singleton."""

    def test_get_registry(self):
        """Test getting global registry."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2
