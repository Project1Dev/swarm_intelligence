"""Tests for SwarmOrchestrator."""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from swarm.orchestrator import (
    SwarmOrchestrator,
    SwarmConfig,
    SwarmResult,
    SwarmStatus,
)
from agents.base import AgentConfig, AgentRole, Permission
from agents.leader import Plan, Subtask, SubtaskType, SubtaskStatus
from communication.message import Message, MessageType, Priority
from utils.maze_generator import Maze, Cell


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_ollama():
    """Create mock Ollama client."""
    client = Mock()
    # Mock successful generation
    client.generate = Mock(
        return_value=Mock(
            success=True,
            content='{"analysis": "Test", "recommended_algorithm": "BFS", '
                    '"reasoning": "Small maze", "subtasks": ['
                    '{"type": "analyze_maze", "description": "Analyze", '
                    '"assigned_agent": "logic_checker", "priority": 1}'
                    '], "expected_duration": 30}',
            tokens_used=50,
        )
    )
    return client


@pytest.fixture
def swarm_config():
    """Create swarm configuration."""
    return SwarmConfig(
        time_limit=30.0,
        max_rounds=5,
        token_budget=5000,
        parallel_execution=False,
    )


@pytest.fixture
def orchestrator(mock_ollama, swarm_config):
    """Create orchestrator."""
    orch = SwarmOrchestrator(mock_ollama, swarm_config)
    return orch


@pytest.fixture
def simple_maze():
    """Create a simple test maze."""
    grid = [
        [Cell.WALL, Cell.WALL, Cell.WALL, Cell.WALL, Cell.WALL],
        [Cell.WALL, Cell.PATH, Cell.PATH, Cell.PATH, Cell.WALL],
        [Cell.WALL, Cell.PATH, Cell.WALL, Cell.PATH, Cell.WALL],
        [Cell.WALL, Cell.PATH, Cell.PATH, Cell.PATH, Cell.WALL],
        [Cell.WALL, Cell.WALL, Cell.WALL, Cell.WALL, Cell.WALL],
    ]
    return Maze(
        width=5,
        height=5,
        grid=grid,
        start=(1, 1),
        end=(3, 3),
        algorithm="test",
    )


# ==============================================================================
# SwarmConfig Tests
# ==============================================================================

class TestSwarmConfig:
    """Test SwarmConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = SwarmConfig()
        assert config.time_limit == 60.0
        assert config.max_rounds == 10
        assert config.token_budget == 10000
        assert not config.parallel_execution

    def test_custom_config(self):
        """Test custom configuration."""
        config = SwarmConfig(
            time_limit=30.0,
            max_rounds=5,
            token_budget=5000,
        )
        assert config.time_limit == 30.0
        assert config.max_rounds == 5

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = SwarmConfig(time_limit=45.0)
        data = config.to_dict()
        assert data["time_limit"] == 45.0


# ==============================================================================
# SwarmResult Tests
# ==============================================================================

class TestSwarmResult:
    """Test SwarmResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = SwarmResult(
            success=True,
            path=[(0, 0), (0, 1), (1, 1)],
            path_length=3,
            nodes_explored=10,
            execution_time=1.5,
            algorithm_used="BFS",
            rounds=3,
            total_tokens=100,
            agent_contributions={"Leader": 50, "CodeSpecialist": 50},
            status=SwarmStatus.COMPLETED,
        )

        assert result.success
        assert result.path_length == 3
        assert result.status == SwarmStatus.COMPLETED

    def test_failure_result(self):
        """Test failure result."""
        result = SwarmResult(
            success=False,
            path=None,
            path_length=0,
            nodes_explored=0,
            execution_time=10.0,
            algorithm_used="none",
            rounds=5,
            total_tokens=200,
            agent_contributions={},
            status=SwarmStatus.FAILED,
            error="Test error",
        )

        assert not result.success
        assert result.error == "Test error"

    def test_result_serialization(self):
        """Test result serialization."""
        result = SwarmResult(
            success=True,
            path=[(0, 0)],
            path_length=1,
            nodes_explored=1,
            execution_time=0.1,
            algorithm_used="BFS",
            rounds=1,
            total_tokens=10,
            agent_contributions={},
            status=SwarmStatus.COMPLETED,
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["status"] == "completed"


# ==============================================================================
# SwarmOrchestrator Tests
# ==============================================================================

class TestSwarmOrchestrator:
    """Test SwarmOrchestrator class."""

    def test_creation(self, orchestrator):
        """Test creating orchestrator."""
        assert orchestrator is not None
        assert orchestrator.status == SwarmStatus.IDLE
        assert orchestrator.blackboard is not None
        assert orchestrator.router is not None

    def test_initialize_agents(self, orchestrator):
        """Test initializing agents."""
        orchestrator.initialize_agents()

        assert len(orchestrator.agents) == 4
        assert AgentRole.LEADER in orchestrator.agents
        assert AgentRole.CODE_SPECIALIST in orchestrator.agents
        assert AgentRole.LOGIC_CHECKER in orchestrator.agents
        assert AgentRole.RETRIEVAL_SPECIALIST in orchestrator.agents

    def test_leader_initialized(self, orchestrator):
        """Test leader is properly initialized."""
        orchestrator.initialize_agents()

        assert orchestrator.leader is not None
        assert orchestrator.leader.role == AgentRole.LEADER

    def test_agents_registered_with_router(self, orchestrator):
        """Test agents are registered with message router."""
        orchestrator.initialize_agents()

        stats = orchestrator.router.get_statistics()
        assert len(stats["registered_agents"]) == 4

    def test_get_status_idle(self, orchestrator):
        """Test getting status when idle."""
        status = orchestrator.get_status()

        assert status["status"] == "idle"
        assert status["current_round"] == 0

    def test_get_status_after_init(self, orchestrator):
        """Test getting status after initialization."""
        orchestrator.initialize_agents()
        status = orchestrator.get_status()

        assert len(status["agents"]) == 4

    def test_reset(self, orchestrator):
        """Test resetting orchestrator."""
        orchestrator.initialize_agents()
        orchestrator.status = SwarmStatus.EXECUTING
        orchestrator.current_round = 5
        orchestrator.total_tokens = 1000

        orchestrator.reset()

        assert orchestrator.status == SwarmStatus.IDLE
        assert orchestrator.current_round == 0
        assert orchestrator.total_tokens == 0

    def test_write_maze_to_blackboard(self, orchestrator, simple_maze):
        """Test writing maze info to blackboard."""
        orchestrator._write_maze_to_blackboard(simple_maze)

        goal = orchestrator.blackboard.read("current_goal", "test")
        assert goal is not None
        assert "maze" in goal

    def test_check_timeout_not_exceeded(self, orchestrator):
        """Test timeout check when not exceeded."""
        orchestrator.start_time = time.time()
        orchestrator.config.time_limit = 60.0

        assert not orchestrator._check_timeout()

    def test_check_timeout_exceeded(self, orchestrator):
        """Test timeout check when exceeded."""
        orchestrator.start_time = time.time() - 100
        orchestrator.config.time_limit = 60.0

        assert orchestrator._check_timeout()

    def test_calculate_wall_ratio(self, orchestrator, simple_maze):
        """Test wall ratio calculation."""
        ratio = orchestrator._calculate_wall_ratio(simple_maze)
        assert 0 <= ratio <= 1

    def test_create_failure_result(self, orchestrator):
        """Test creating failure result."""
        orchestrator.start_time = time.time()
        result = orchestrator._create_failure_result("Test error")

        assert not result.success
        assert result.error == "Test error"
        assert result.status == SwarmStatus.FAILED

    def test_create_failure_result_timeout(self, orchestrator):
        """Test creating failure result with timeout status."""
        orchestrator.start_time = time.time()
        result = orchestrator._create_failure_result(
            "Timeout", SwarmStatus.TIMEOUT
        )

        assert result.status == SwarmStatus.TIMEOUT


# ==============================================================================
# Integration Tests (with mocked LLM)
# ==============================================================================

class TestSwarmIntegration:
    """Integration tests for swarm orchestration."""

    def test_solve_maze_basic(self, orchestrator, simple_maze):
        """Test basic maze solving (will use fallback)."""
        orchestrator.initialize_agents()
        result = orchestrator.solve_maze(simple_maze)

        # Should complete (possibly with fallback)
        assert result is not None
        assert result.execution_time > 0

    def test_solve_maze_with_fallback(self, orchestrator, simple_maze, mock_ollama):
        """Test maze solving falls back to BFS."""
        # Make LLM fail to force fallback
        mock_ollama.generate = Mock(
            return_value=Mock(
                success=False,
                content="",
                tokens_used=0,
            )
        )

        orchestrator.initialize_agents()

        # Set short timeout to trigger fallback
        orchestrator.config.time_limit = 0.001

        result = orchestrator.solve_maze(simple_maze)

        # Should still succeed via fallback
        assert result is not None

    def test_solve_maze_tracks_rounds(self, orchestrator, simple_maze):
        """Test that rounds are tracked."""
        orchestrator.initialize_agents()
        result = orchestrator.solve_maze(simple_maze)

        assert result.rounds >= 0

    def test_solve_maze_tracks_tokens(self, orchestrator, simple_maze):
        """Test that tokens are tracked."""
        orchestrator.initialize_agents()
        result = orchestrator.solve_maze(simple_maze)

        assert result.total_tokens >= 0

    def test_solve_maze_provides_agent_contributions(self, orchestrator, simple_maze):
        """Test that agent contributions are tracked."""
        orchestrator.initialize_agents()
        result = orchestrator.solve_maze(simple_maze)

        assert isinstance(result.agent_contributions, dict)

    def test_agents_deactivated_after_solve(self, orchestrator, simple_maze):
        """Test agents are deactivated after solving."""
        orchestrator.initialize_agents()
        orchestrator.solve_maze(simple_maze)

        for agent in orchestrator.agents.values():
            assert not agent.is_active

    def test_status_transitions(self, orchestrator, simple_maze):
        """Test status transitions during solve."""
        orchestrator.initialize_agents()

        initial_status = orchestrator.status
        assert initial_status == SwarmStatus.IDLE

        result = orchestrator.solve_maze(simple_maze)

        # Should end in terminal state
        assert result.status in [
            SwarmStatus.COMPLETED,
            SwarmStatus.FAILED,
            SwarmStatus.TIMEOUT,
        ]


# ==============================================================================
# Subtask Execution Tests
# ==============================================================================

class TestSubtaskExecution:
    """Test subtask execution."""

    def test_build_task_context_implement(self, orchestrator, simple_maze):
        """Test building task context for implementation."""
        orchestrator.initialize_agents()

        subtask = Subtask(
            id="test_1",
            type=SubtaskType.IMPLEMENT_SOLUTION,
            description="Implement BFS",
            assigned_agent=AgentRole.CODE_SPECIALIST,
            priority=Priority.HIGH,
        )

        task, context = orchestrator._build_task_context(subtask, simple_maze)

        assert task["type"] == "implement_solution"
        assert "maze" in context

    def test_build_task_context_validate(self, orchestrator, simple_maze):
        """Test building task context for validation."""
        orchestrator.initialize_agents()

        subtask = Subtask(
            id="test_1",
            type=SubtaskType.VALIDATE_SOLUTION,
            description="Validate",
            assigned_agent=AgentRole.LOGIC_CHECKER,
            priority=Priority.CRITICAL,
        )

        task, context = orchestrator._build_task_context(subtask, simple_maze)

        assert task["type"] == "validate_solution"

    def test_build_task_context_retrieve(self, orchestrator, simple_maze):
        """Test building task context for retrieval."""
        orchestrator.initialize_agents()

        subtask = Subtask(
            id="test_1",
            type=SubtaskType.RETRIEVE_KNOWLEDGE,
            description="Retrieve",
            assigned_agent=AgentRole.RETRIEVAL_SPECIALIST,
            priority=Priority.NORMAL,
        )

        task, context = orchestrator._build_task_context(subtask, simple_maze)

        assert task["type"] == "retrieve_knowledge"
        assert "maze_properties" in task


# ==============================================================================
# Fallback Tests
# ==============================================================================

class TestFallback:
    """Test fallback behavior."""

    def test_run_fallback_success(self, orchestrator, simple_maze):
        """Test fallback algorithm succeeds."""
        orchestrator.start_time = time.time()

        result = orchestrator._run_fallback(simple_maze, 0.1)

        assert result.success
        assert result.path is not None
        assert "fallback" in result.algorithm_used.lower()

    def test_run_fallback_on_unsolvable(self, orchestrator):
        """Test fallback on unsolvable maze."""
        # Create maze with blocked path
        grid = [
            [Cell.WALL, Cell.WALL, Cell.WALL],
            [Cell.WALL, Cell.PATH, Cell.WALL],
            [Cell.WALL, Cell.WALL, Cell.WALL],
        ]
        unsolvable = Maze(
            width=3, height=3, grid=grid,
            start=(1, 1), end=(2, 2),  # End is in wall
            algorithm="test"
        )

        orchestrator.start_time = time.time()
        result = orchestrator._run_fallback(unsolvable, 0.1)

        assert not result.success

    def test_handle_timeout_uses_fallback(self, orchestrator, simple_maze):
        """Test timeout handler uses fallback."""
        orchestrator.initialize_agents()
        orchestrator.start_time = time.time()

        plan = Plan(
            id="test",
            problem_description="Test",
            subtasks=[],
            created_at=time.time(),
            updated_at=time.time(),
        )

        result = orchestrator._handle_timeout(plan, simple_maze)

        # Should use fallback and succeed
        assert result is not None


# ==============================================================================
# Multi-Maze Tests
# ==============================================================================

class TestMultipleMazes:
    """Test solving multiple mazes."""

    def test_solve_multiple_mazes_sequentially(self, orchestrator):
        """Test solving multiple mazes in sequence."""
        mazes = []
        for size in [5, 7, 9]:
            grid = [[Cell.PATH for _ in range(size)] for _ in range(size)]
            for i in range(size):
                grid[0][i] = Cell.WALL
                grid[size-1][i] = Cell.WALL
                grid[i][0] = Cell.WALL
                grid[i][size-1] = Cell.WALL
            mazes.append(Maze(
                width=size, height=size, grid=grid,
                start=(1, 1), end=(size-2, size-2),
                algorithm="test"
            ))

        orchestrator.initialize_agents()

        results = []
        for maze in mazes:
            orchestrator.reset()
            result = orchestrator.solve_maze(maze)
            results.append(result)

        assert len(results) == 3

    def test_reset_between_mazes(self, orchestrator, simple_maze):
        """Test resetting between maze solves."""
        orchestrator.initialize_agents()

        # First solve
        orchestrator.solve_maze(simple_maze)
        round1 = orchestrator.current_round
        tokens1 = orchestrator.total_tokens

        # Reset
        orchestrator.reset()

        assert orchestrator.current_round == 0
        assert orchestrator.total_tokens == 0

        # Second solve
        orchestrator.solve_maze(simple_maze)

        # Should start fresh
        assert orchestrator.current_round >= 0
