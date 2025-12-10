"""Tests for LeaderAgent class."""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from agents.base import AgentConfig, AgentRole, Permission
from agents.leader import (
    LeaderAgent,
    Plan,
    Subtask,
    SubtaskType,
    SubtaskStatus,
)
from communication.message import Message, MessageType, Priority
from utils.maze_generator import Maze, Cell


@pytest.fixture
def mock_ollama():
    """Create mock Ollama client."""
    client = Mock()
    # Default response for plan generation
    client.generate = Mock(
        return_value=Mock(
            success=True,
            content='{"analysis": "Test maze", "recommended_algorithm": "BFS", '
                    '"reasoning": "Small maze", "subtasks": ['
                    '{"type": "analyze_maze", "description": "Analyze", '
                    '"assigned_agent": "logic_checker", "priority": 1},'
                    '{"type": "implement_solution", "description": "Implement BFS", '
                    '"assigned_agent": "code_specialist", "priority": 1},'
                    '{"type": "validate_solution", "description": "Validate", '
                    '"assigned_agent": "logic_checker", "priority": 0}'
                    '], "expected_duration": 30}',
            tokens_used=100,
        )
    )
    return client


@pytest.fixture
def mock_blackboard():
    """Create mock blackboard."""
    bb = Mock()
    bb.read = Mock(return_value={})
    bb.write = Mock(return_value=True)
    return bb


@pytest.fixture
def leader_config():
    """Create leader agent config."""
    return AgentConfig(
        name="test_leader",
        role=AgentRole.LEADER,
        model="phi3:mini",
        temperature=0.4,
        max_tokens=1000,
        token_budget=5000,
        permissions={
            "current_goal": Permission.READ_WRITE,
            "facts": Permission.READ_WRITE,
            "hypotheses": Permission.READ_WRITE,
            "artifacts": Permission.READ,
            "working_memory": Permission.READ_WRITE,
            "metrics": Permission.READ,
        },
    )


@pytest.fixture
def leader_agent(leader_config, mock_ollama, mock_blackboard):
    """Create leader agent."""
    agent = LeaderAgent(leader_config, mock_ollama, mock_blackboard)
    return agent


@pytest.fixture
def simple_maze():
    """Create a simple test maze."""
    # Create a 5x5 maze with a clear path
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


@pytest.fixture
def medium_maze():
    """Create a medium test maze."""
    # Create a 15x15 maze
    grid = [[Cell.WALL for _ in range(15)] for _ in range(15)]
    # Create path
    for i in range(1, 14):
        grid[1][i] = Cell.PATH
        grid[i][1] = Cell.PATH
        grid[13][i] = Cell.PATH
        grid[i][13] = Cell.PATH

    return Maze(
        width=15,
        height=15,
        grid=grid,
        start=(1, 1),
        end=(13, 13),
        algorithm="test",
    )


class TestSubtask:
    """Test Subtask dataclass."""

    def test_subtask_creation(self):
        """Test creating a subtask."""
        subtask = Subtask(
            id="task_1",
            type=SubtaskType.ANALYZE_MAZE,
            description="Analyze maze structure",
            assigned_agent=AgentRole.LOGIC_CHECKER,
            priority=Priority.HIGH,
        )

        assert subtask.id == "task_1"
        assert subtask.type == SubtaskType.ANALYZE_MAZE
        assert subtask.status == SubtaskStatus.PENDING
        assert subtask.retry_count == 0

    def test_subtask_serialization(self):
        """Test subtask serialization."""
        subtask = Subtask(
            id="task_1",
            type=SubtaskType.IMPLEMENT_SOLUTION,
            description="Implement BFS",
            assigned_agent=AgentRole.CODE_SPECIALIST,
            priority=Priority.CRITICAL,
            dependencies=["task_0"],
        )

        data = subtask.to_dict()
        assert data["id"] == "task_1"
        assert data["type"] == "implement_solution"
        assert data["assigned_agent"] == "code_specialist"
        assert data["dependencies"] == ["task_0"]

        restored = Subtask.from_dict(data)
        assert restored.id == subtask.id
        assert restored.type == subtask.type

    def test_subtask_with_result(self):
        """Test subtask with result."""
        subtask = Subtask(
            id="task_1",
            type=SubtaskType.VALIDATE_SOLUTION,
            description="Validate",
            assigned_agent=AgentRole.LOGIC_CHECKER,
            priority=Priority.HIGH,
            status=SubtaskStatus.COMPLETED,
            result={"valid": True, "confidence": 0.95},
        )

        assert subtask.status == SubtaskStatus.COMPLETED
        assert subtask.result["valid"] is True


class TestPlan:
    """Test Plan dataclass."""

    def test_plan_creation(self):
        """Test creating a plan."""
        subtasks = [
            Subtask(
                id="task_0",
                type=SubtaskType.ANALYZE_MAZE,
                description="Analyze",
                assigned_agent=AgentRole.LOGIC_CHECKER,
                priority=Priority.HIGH,
            ),
            Subtask(
                id="task_1",
                type=SubtaskType.IMPLEMENT_SOLUTION,
                description="Implement",
                assigned_agent=AgentRole.CODE_SPECIALIST,
                priority=Priority.HIGH,
                dependencies=["task_0"],
            ),
        ]

        plan = Plan(
            id="plan_1",
            problem_description="Test maze",
            subtasks=subtasks,
            created_at=time.time(),
            updated_at=time.time(),
        )

        assert plan.id == "plan_1"
        assert len(plan.subtasks) == 2

    def test_get_next_subtask(self):
        """Test getting next subtask."""
        subtasks = [
            Subtask(
                id="task_0",
                type=SubtaskType.ANALYZE_MAZE,
                description="Analyze",
                assigned_agent=AgentRole.LOGIC_CHECKER,
                priority=Priority.HIGH,
            ),
            Subtask(
                id="task_1",
                type=SubtaskType.IMPLEMENT_SOLUTION,
                description="Implement",
                assigned_agent=AgentRole.CODE_SPECIALIST,
                priority=Priority.HIGH,
                dependencies=["task_0"],
            ),
        ]

        plan = Plan(
            id="plan_1",
            problem_description="Test",
            subtasks=subtasks,
            created_at=time.time(),
            updated_at=time.time(),
        )

        # First task should be next (no dependencies)
        next_task = plan.get_next_subtask()
        assert next_task.id == "task_0"

        # Complete first task
        plan.update_subtask_status("task_0", SubtaskStatus.COMPLETED)

        # Second task should now be next
        next_task = plan.get_next_subtask()
        assert next_task.id == "task_1"

    def test_plan_completion(self):
        """Test plan completion check."""
        subtask = Subtask(
            id="task_0",
            type=SubtaskType.ANALYZE_MAZE,
            description="Analyze",
            assigned_agent=AgentRole.LOGIC_CHECKER,
            priority=Priority.HIGH,
        )

        plan = Plan(
            id="plan_1",
            problem_description="Test",
            subtasks=[subtask],
            created_at=time.time(),
            updated_at=time.time(),
        )

        assert not plan.is_complete()

        plan.update_subtask_status("task_0", SubtaskStatus.COMPLETED)
        assert plan.is_complete()

    def test_plan_failure_detection(self):
        """Test detecting plan failure."""
        subtask = Subtask(
            id="task_0",
            type=SubtaskType.IMPLEMENT_SOLUTION,
            description="Implement",
            assigned_agent=AgentRole.CODE_SPECIALIST,
            priority=Priority.CRITICAL,
        )

        plan = Plan(
            id="plan_1",
            problem_description="Test",
            subtasks=[subtask],
            created_at=time.time(),
            updated_at=time.time(),
        )

        assert not plan.has_failed()

        plan.update_subtask_status("task_0", SubtaskStatus.FAILED)
        assert plan.has_failed()

    def test_plan_serialization(self):
        """Test plan serialization."""
        subtask = Subtask(
            id="task_0",
            type=SubtaskType.ANALYZE_MAZE,
            description="Analyze",
            assigned_agent=AgentRole.LOGIC_CHECKER,
            priority=Priority.HIGH,
        )

        plan = Plan(
            id="plan_1",
            problem_description="Test maze 5x5",
            subtasks=[subtask],
            created_at=time.time(),
            updated_at=time.time(),
            analysis="Simple maze",
            metadata={"recommended_algorithm": "BFS"},
        )

        data = plan.to_dict()
        assert data["id"] == "plan_1"
        assert data["metadata"]["recommended_algorithm"] == "BFS"

        restored = Plan.from_dict(data)
        assert restored.id == plan.id
        assert restored.metadata["recommended_algorithm"] == "BFS"


class TestLeaderAgent:
    """Test LeaderAgent class."""

    def test_leader_creation(self, leader_agent):
        """Test creating leader agent."""
        assert leader_agent.name == "test_leader"
        assert leader_agent.role == AgentRole.LEADER
        assert leader_agent.current_plan is None

    def test_leader_permissions(self, leader_agent):
        """Test leader has correct permissions."""
        assert leader_agent.has_permission("current_goal", Permission.READ)
        assert leader_agent.has_permission("current_goal", Permission.WRITE)
        assert leader_agent.has_permission("facts", Permission.READ_WRITE)

    def test_decompose_task_llm(self, leader_agent, simple_maze):
        """Test task decomposition with LLM."""
        plan = leader_agent.decompose_task(simple_maze)

        assert plan is not None
        assert len(plan.subtasks) >= 1
        assert leader_agent.current_plan is plan

    def test_decompose_task_fallback(self, leader_agent, simple_maze, mock_ollama):
        """Test task decomposition falls back to rule-based."""
        # Make LLM fail
        mock_ollama.generate = Mock(
            return_value=Mock(success=False, content="", tokens_used=0)
        )

        plan = leader_agent.decompose_task(simple_maze)

        assert plan is not None
        assert len(plan.subtasks) >= 2  # Rule-based has 5 tasks

    def test_rule_based_decompose_small_maze(self, leader_agent, simple_maze):
        """Test rule-based decomposition for small maze."""
        # Force rule-based by making LLM fail
        leader_agent.ollama_client.generate = Mock(
            return_value=Mock(success=False, content="", tokens_used=0)
        )

        plan = leader_agent.decompose_task(simple_maze)

        assert plan.metadata.get("recommended_algorithm") == "BFS"
        assert "rule-based" in plan.analysis.lower()

    def test_rule_based_decompose_large_maze(self, leader_agent, mock_ollama):
        """Test rule-based decomposition for large maze."""
        # Create a large maze
        grid = [[Cell.PATH for _ in range(35)] for _ in range(35)]
        for i in range(35):
            grid[0][i] = Cell.WALL
            grid[34][i] = Cell.WALL
            grid[i][0] = Cell.WALL
            grid[i][34] = Cell.WALL

        large_maze = Maze(
            width=35, height=35, grid=grid, start=(1, 1), end=(33, 33),
            algorithm="test"
        )

        mock_ollama.generate = Mock(
            return_value=Mock(success=False, content="", tokens_used=0)
        )

        plan = leader_agent.decompose_task(large_maze)

        # Should recommend A* for large mazes
        assert plan.metadata.get("recommended_algorithm") == "A*"

    def test_create_maze_description(self, leader_agent, simple_maze):
        """Test maze description creation."""
        description = leader_agent._create_maze_description(simple_maze)

        assert "5x5" in description
        assert "Start position" in description
        assert "End position" in description
        assert "simple" in description

    def test_select_agent_default(self, leader_agent):
        """Test agent selection returns default."""
        subtask = Subtask(
            id="task_0",
            type=SubtaskType.IMPLEMENT_SOLUTION,
            description="Implement",
            assigned_agent=AgentRole.CODE_SPECIALIST,
            priority=Priority.HIGH,
        )

        selected = leader_agent.select_agent(subtask, {})
        assert selected == AgentRole.CODE_SPECIALIST

    def test_select_agent_validation(self, leader_agent):
        """Test agent selection for validation always uses logic checker."""
        subtask = Subtask(
            id="task_0",
            type=SubtaskType.VALIDATE_SOLUTION,
            description="Validate",
            assigned_agent=AgentRole.CODE_SPECIALIST,  # Wrong assignment
            priority=Priority.HIGH,
        )

        selected = leader_agent.select_agent(subtask, {})
        assert selected == AgentRole.LOGIC_CHECKER

    def test_monitor_progress_no_plan(self, leader_agent):
        """Test progress monitoring with no plan."""
        is_progressing, status = leader_agent.monitor_progress()
        assert not is_progressing
        assert "No active plan" in status

    def test_monitor_progress_with_plan(self, leader_agent, simple_maze):
        """Test progress monitoring with active plan."""
        leader_agent.decompose_task(simple_maze)

        is_progressing, status = leader_agent.monitor_progress()
        assert is_progressing
        assert "Progress" in status

    def test_monitor_progress_completion(self, leader_agent, simple_maze):
        """Test progress monitoring detects completion."""
        plan = leader_agent.decompose_task(simple_maze)

        # Complete all tasks
        for subtask in plan.subtasks:
            subtask.status = SubtaskStatus.COMPLETED

        is_progressing, status = leader_agent.monitor_progress()
        assert is_progressing
        assert "complete" in status.lower()

    def test_replan_retry(self, leader_agent, simple_maze):
        """Test replanning with retry."""
        plan = leader_agent.decompose_task(simple_maze)
        original_algo = plan.metadata.get("recommended_algorithm")

        # Mark a task as failed
        failed_task = None
        for subtask in plan.subtasks:
            if subtask.type == SubtaskType.IMPLEMENT_SOLUTION:
                failed_task = subtask
                break

        if failed_task:
            failure_info = {
                "subtask_id": failed_task.id,
                "error": "Implementation error",
                "retry_count": 0,
            }

            new_plan = leader_agent.replan(failure_info)
            assert new_plan is not None

            # Check task was reset to pending
            retried_task = new_plan.get_subtask_by_id(failed_task.id)
            if retried_task:
                assert retried_task.retry_count == 1

    def test_replan_skip_low_priority(self, leader_agent, simple_maze):
        """Test replanning skips low priority task after max retries."""
        plan = leader_agent.decompose_task(simple_maze)

        # Find a low priority task or modify one
        for subtask in plan.subtasks:
            if subtask.priority == Priority.NORMAL:
                subtask.max_retries = 0  # Exhaust retries
                failure_info = {
                    "subtask_id": subtask.id,
                    "error": "Test error",
                    "retry_count": 1,
                }

                new_plan = leader_agent.replan(failure_info)
                assert new_plan is not None

                skipped_task = new_plan.get_subtask_by_id(subtask.id)
                assert skipped_task.status == SubtaskStatus.SKIPPED
                break

    def test_replan_fallback(self, leader_agent, simple_maze):
        """Test replanning creates fallback for critical failure."""
        plan = leader_agent.decompose_task(simple_maze)

        # Find critical task and exhaust retries
        for subtask in plan.subtasks:
            if subtask.priority == Priority.CRITICAL:
                subtask.max_retries = 0

                failure_info = {
                    "subtask_id": subtask.id,
                    "error": "Critical failure",
                    "retry_count": 1,
                }

                new_plan = leader_agent.replan(failure_info)
                assert new_plan is not None
                assert new_plan.metadata.get("is_fallback", False) or \
                       new_plan.metadata.get("recommended_algorithm") == "BFS"
                break

    def test_process_task_analyze(self, leader_agent, simple_maze):
        """Test processing analyze_problem task."""
        result = leader_agent.process_task(
            {"type": "analyze_problem"},
            {"maze": simple_maze}
        )

        assert result is not None
        assert result.type == MessageType.STATUS_UPDATE
        assert result.content["task"] == "analyze_problem"

    def test_process_task_create_plan(self, leader_agent, simple_maze):
        """Test processing create_plan task."""
        result = leader_agent.process_task(
            {"type": "create_plan"},
            {"maze": simple_maze}
        )

        assert result is not None
        assert result.type == MessageType.PLAN_UPDATE

    def test_process_task_no_maze(self, leader_agent):
        """Test processing task without maze returns error."""
        result = leader_agent.process_task(
            {"type": "analyze_problem"},
            {}
        )

        assert result is not None
        assert result.type == MessageType.ERROR

    def test_process_task_unknown(self, leader_agent):
        """Test processing unknown task type."""
        result = leader_agent.process_task(
            {"type": "unknown_task"},
            {}
        )

        assert result is None

    def test_validate_output_message(self, leader_agent):
        """Test validating message output."""
        msg = Message(
            from_agent="leader",
            to_agent="broadcast",
            type=MessageType.STATUS_UPDATE,
            priority=Priority.NORMAL,
            content={"test": True},
        )

        valid, reason = leader_agent.validate_output(msg)
        assert valid

    def test_validate_output_plan(self, leader_agent, simple_maze):
        """Test validating plan output."""
        plan = leader_agent.decompose_task(simple_maze)

        valid, reason = leader_agent.validate_output(plan)
        assert valid

    def test_validate_output_none(self, leader_agent):
        """Test validating None output."""
        valid, reason = leader_agent.validate_output(None)
        assert not valid

    def test_get_plan_summary_no_plan(self, leader_agent):
        """Test getting summary without plan."""
        summary = leader_agent.get_plan_summary()
        assert summary["status"] == "no_plan"

    def test_get_plan_summary(self, leader_agent, simple_maze):
        """Test getting plan summary."""
        leader_agent.decompose_task(simple_maze)
        summary = leader_agent.get_plan_summary()

        assert "plan_id" in summary
        assert summary["total_subtasks"] >= 1
        assert "completed" in summary
        assert "failed" in summary

    def test_plan_history(self, leader_agent, simple_maze, medium_maze):
        """Test plan history is maintained."""
        leader_agent.decompose_task(simple_maze)
        leader_agent.decompose_task(medium_maze)

        assert len(leader_agent.plan_history) == 2


class TestJSONParsing:
    """Test JSON parsing from LLM responses."""

    def test_parse_direct_json(self, leader_agent):
        """Test parsing direct JSON."""
        response = '{"key": "value"}'
        result = leader_agent._parse_json_response(response)
        assert result == {"key": "value"}

    def test_parse_markdown_block(self, leader_agent):
        """Test parsing JSON from markdown code block."""
        response = '''Here's the plan:
```json
{"key": "value"}
```'''
        result = leader_agent._parse_json_response(response)
        assert result == {"key": "value"}

    def test_parse_embedded_json(self, leader_agent):
        """Test parsing embedded JSON."""
        response = 'The analysis shows {"key": "value"} as result.'
        result = leader_agent._parse_json_response(response)
        assert result == {"key": "value"}

    def test_parse_invalid_json(self, leader_agent):
        """Test parsing invalid JSON returns None."""
        response = 'Not valid JSON at all'
        result = leader_agent._parse_json_response(response)
        assert result is None


class TestDecisionTree:
    """Test decision tree scenarios."""

    def test_decision_small_maze_bfs(self, leader_agent):
        """Test decision for small maze recommends BFS."""
        grid = [[Cell.PATH for _ in range(10)] for _ in range(10)]
        small_maze = Maze(width=10, height=10, grid=grid, start=(1, 1), end=(8, 8), algorithm="test")

        leader_agent.ollama_client.generate = Mock(
            return_value=Mock(success=False, content="", tokens_used=0)
        )

        plan = leader_agent.decompose_task(small_maze)
        assert plan.metadata.get("recommended_algorithm") == "BFS"

    def test_decision_medium_maze_astar(self, leader_agent):
        """Test decision for medium maze recommends A*."""
        grid = [[Cell.PATH for _ in range(25)] for _ in range(25)]
        medium_maze = Maze(width=25, height=25, grid=grid, start=(1, 1), end=(23, 23), algorithm="test")

        leader_agent.ollama_client.generate = Mock(
            return_value=Mock(success=False, content="", tokens_used=0)
        )

        plan = leader_agent.decompose_task(medium_maze)
        assert plan.metadata.get("recommended_algorithm") == "A*"


class TestStateManagement:
    """Test state management."""

    def test_state_persistence(self, leader_agent, simple_maze):
        """Test state is saved correctly."""
        leader_agent.decompose_task(simple_maze)

        state = leader_agent.save_state()
        assert "state" in state
        assert state["state"]["tasks_completed"] >= 0

    def test_tasks_completed_count(self, leader_agent, simple_maze):
        """Test tasks completed counter."""
        initial_count = leader_agent.state.tasks_completed

        leader_agent.process_task(
            {"type": "analyze_problem"},
            {"maze": simple_maze}
        )

        assert leader_agent.state.tasks_completed > initial_count
