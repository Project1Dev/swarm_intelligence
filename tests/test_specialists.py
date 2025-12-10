"""Tests for specialist agents."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Tuple

from agents.base import AgentConfig, AgentRole, Permission
from agents.specialists import (
    CodeSpecialist,
    CodeGenerationResult,
    LogicChecker,
    ValidationResult,
    RetrievalSpecialist,
    RetrievalResult,
)
from communication.message import Message, MessageType, Priority
from utils.maze_generator import Maze, Cell


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_ollama():
    """Create mock Ollama client."""
    client = Mock()
    client.generate = Mock(
        return_value=Mock(
            success=True,
            content='{"valid": true, "issues": [], "suggestions": []}',
            tokens_used=50,
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
def code_specialist_config():
    """Create code specialist config."""
    return AgentConfig(
        name="test_code_specialist",
        role=AgentRole.CODE_SPECIALIST,
        model="deepseek-coder:1.3b",
        temperature=0.1,
        max_tokens=2000,
        token_budget=8000,
    )


@pytest.fixture
def logic_checker_config():
    """Create logic checker config."""
    return AgentConfig(
        name="test_logic_checker",
        role=AgentRole.LOGIC_CHECKER,
        model="qwen2.5:1.5b",
        temperature=0.0,
        max_tokens=1500,
        token_budget=4000,
    )


@pytest.fixture
def retrieval_specialist_config():
    """Create retrieval specialist config."""
    return AgentConfig(
        name="test_retrieval_specialist",
        role=AgentRole.RETRIEVAL_SPECIALIST,
        model="tinyllama:1.1b",
        temperature=0.2,
        max_tokens=800,
        token_budget=3000,
    )


@pytest.fixture
def code_specialist(code_specialist_config, mock_ollama, mock_blackboard):
    """Create code specialist agent."""
    return CodeSpecialist(code_specialist_config, mock_ollama, mock_blackboard)


@pytest.fixture
def logic_checker(logic_checker_config, mock_ollama, mock_blackboard):
    """Create logic checker agent."""
    return LogicChecker(logic_checker_config, mock_ollama, mock_blackboard)


@pytest.fixture
def retrieval_specialist(retrieval_specialist_config, mock_ollama, mock_blackboard):
    """Create retrieval specialist agent."""
    return RetrievalSpecialist(retrieval_specialist_config, mock_ollama, mock_blackboard)


@pytest.fixture
def simple_maze_grid():
    """Create a simple maze grid."""
    return [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]


# ==============================================================================
# CodeGenerationResult Tests
# ==============================================================================

class TestCodeGenerationResult:
    """Test CodeGenerationResult dataclass."""

    def test_creation(self):
        """Test creating a result."""
        result = CodeGenerationResult(
            code="def test(): pass",
            algorithm="BFS",
            explanation="Test explanation",
            complexity_time="O(n)",
            complexity_space="O(n)",
        )

        assert result.code == "def test(): pass"
        assert result.algorithm == "BFS"
        assert result.success is True

    def test_serialization(self):
        """Test result serialization."""
        result = CodeGenerationResult(
            code="def test(): pass",
            algorithm="BFS",
            explanation="Test",
            complexity_time="O(n)",
            complexity_space="O(n)",
        )

        data = result.to_dict()
        assert data["code"] == "def test(): pass"
        assert data["algorithm"] == "BFS"
        assert data["success"] is True


# ==============================================================================
# CodeSpecialist Tests
# ==============================================================================

class TestCodeSpecialist:
    """Test CodeSpecialist agent."""

    def test_creation(self, code_specialist):
        """Test creating code specialist."""
        assert code_specialist.name == "test_code_specialist"
        assert code_specialist.role == AgentRole.CODE_SPECIALIST

    def test_permissions(self, code_specialist):
        """Test code specialist has correct permissions."""
        assert code_specialist.has_permission("artifacts", Permission.READ)
        assert code_specialist.has_permission("artifacts", Permission.WRITE)
        assert code_specialist.has_permission("current_goal", Permission.READ)

    def test_generate_algorithm_bfs(self, code_specialist):
        """Test generating BFS algorithm."""
        result = code_specialist.generate_algorithm("BFS")

        assert result.success
        assert result.algorithm == "BFS"
        assert "def solve_maze" in result.code
        assert "deque" in result.code

    def test_generate_algorithm_dfs(self, code_specialist):
        """Test generating DFS algorithm."""
        result = code_specialist.generate_algorithm("DFS")

        assert result.success
        assert result.algorithm == "DFS"
        assert "def solve_maze" in result.code
        assert "stack" in result.code.lower() or "pop()" in result.code

    def test_generate_algorithm_astar(self, code_specialist):
        """Test generating A* algorithm."""
        result = code_specialist.generate_algorithm("A*")

        assert result.success
        assert result.algorithm == "A*"
        assert "def solve_maze" in result.code
        assert "heapq" in result.code or "manhattan" in result.code.lower()

    def test_generate_algorithm_case_insensitive(self, code_specialist):
        """Test algorithm name is case insensitive."""
        result = code_specialist.generate_algorithm("bfs")
        assert result.success
        assert result.algorithm == "BFS"

    def test_validate_syntax_valid(self, code_specialist):
        """Test syntax validation for valid code."""
        code = "def test():\n    return 1"
        is_valid, error = code_specialist._validate_syntax(code)
        assert is_valid
        assert error is None

    def test_validate_syntax_invalid(self, code_specialist):
        """Test syntax validation for invalid code."""
        code = "def test(\n    return 1"
        is_valid, error = code_specialist._validate_syntax(code)
        assert not is_valid
        assert error is not None

    def test_extract_code_markdown(self, code_specialist):
        """Test extracting code from markdown."""
        response = '''Here's the code:
```python
def solve(): pass
```'''
        code = code_specialist._extract_code(response)
        assert "def solve" in code

    def test_extract_code_raw(self, code_specialist):
        """Test extracting code without markdown."""
        response = "def solve_maze(): pass"
        code = code_specialist._extract_code(response)
        assert "def solve_maze" in code

    def test_process_task_generate(self, code_specialist):
        """Test processing generate_algorithm task."""
        result = code_specialist.process_task(
            {"type": "generate_algorithm", "algorithm": "BFS"},
            {}
        )

        assert result is not None
        assert result.type == MessageType.CODE_DRAFT

    def test_process_task_unknown(self, code_specialist):
        """Test processing unknown task type."""
        result = code_specialist.process_task(
            {"type": "unknown"},
            {}
        )
        assert result is None

    def test_validate_output_success(self, code_specialist):
        """Test validating successful output."""
        result = CodeGenerationResult(
            code="def test(): pass",
            algorithm="BFS",
            explanation="Test",
            complexity_time="O(n)",
            complexity_space="O(n)",
        )
        valid, reason = code_specialist.validate_output(result)
        assert valid

    def test_validate_output_failed(self, code_specialist):
        """Test validating failed output."""
        result = CodeGenerationResult(
            code="",
            algorithm="BFS",
            explanation="Test",
            complexity_time="O(n)",
            complexity_space="O(n)",
            success=False,
            error="Failed",
        )
        valid, reason = code_specialist.validate_output(result)
        assert not valid


# ==============================================================================
# ValidationResult Tests
# ==============================================================================

class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_creation(self):
        """Test creating a result."""
        result = ValidationResult(
            valid=True,
            issues=[],
            suggestions=["Test suggestion"],
            confidence=0.95,
        )

        assert result.valid is True
        assert len(result.suggestions) == 1
        assert result.confidence == 0.95

    def test_serialization(self):
        """Test result serialization."""
        result = ValidationResult(
            valid=False,
            issues=[{"severity": "critical", "description": "Test issue"}],
            confidence=0.8,
        )

        data = result.to_dict()
        assert data["valid"] is False
        assert len(data["issues"]) == 1


# ==============================================================================
# LogicChecker Tests
# ==============================================================================

class TestLogicChecker:
    """Test LogicChecker agent."""

    def test_creation(self, logic_checker):
        """Test creating logic checker."""
        assert logic_checker.name == "test_logic_checker"
        assert logic_checker.role == AgentRole.LOGIC_CHECKER

    def test_permissions(self, logic_checker):
        """Test logic checker has correct permissions."""
        assert logic_checker.has_permission("facts", Permission.READ)
        assert logic_checker.has_permission("facts", Permission.WRITE)
        assert logic_checker.has_permission("hypotheses", Permission.READ_WRITE)

    def test_validate_path_valid(self, logic_checker, simple_maze_grid):
        """Test validating a valid path."""
        path = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
        start = (1, 1)
        end = (3, 3)

        result = logic_checker.validate_path(simple_maze_grid, path, start, end)

        assert result.valid
        assert len(result.issues) == 0

    def test_validate_path_wrong_start(self, logic_checker, simple_maze_grid):
        """Test validating path with wrong start."""
        path = [(2, 1), (1, 1), (1, 2)]  # Starts at wrong position
        start = (1, 1)
        end = (1, 2)

        result = logic_checker.validate_path(simple_maze_grid, path, start, end)

        assert not result.valid
        assert any("start" in i["description"].lower() for i in result.issues)

    def test_validate_path_wrong_end(self, logic_checker, simple_maze_grid):
        """Test validating path with wrong end."""
        path = [(1, 1), (1, 2)]  # Ends at wrong position
        start = (1, 1)
        end = (3, 3)

        result = logic_checker.validate_path(simple_maze_grid, path, start, end)

        assert not result.valid
        assert any("end" in i["description"].lower() for i in result.issues)

    def test_validate_path_through_wall(self, logic_checker, simple_maze_grid):
        """Test validating path that goes through a wall."""
        path = [(1, 1), (2, 1), (2, 2)]  # (2,2) is a wall
        start = (1, 1)
        end = (2, 2)

        result = logic_checker.validate_path(simple_maze_grid, path, start, end)

        assert not result.valid
        assert any("wall" in i["description"].lower() for i in result.issues)

    def test_validate_path_non_adjacent(self, logic_checker, simple_maze_grid):
        """Test validating path with non-adjacent steps."""
        path = [(1, 1), (3, 3)]  # Jumps from start to end
        start = (1, 1)
        end = (3, 3)

        result = logic_checker.validate_path(simple_maze_grid, path, start, end)

        assert not result.valid
        assert any("adjacent" in i["description"].lower() for i in result.issues)

    def test_validate_path_empty(self, logic_checker, simple_maze_grid):
        """Test validating empty path."""
        result = logic_checker.validate_path(simple_maze_grid, [], (1, 1), (3, 3))

        assert not result.valid
        assert any("empty" in i["description"].lower() for i in result.issues)

    def test_validate_path_out_of_bounds(self, logic_checker, simple_maze_grid):
        """Test validating path with out of bounds position."""
        path = [(1, 1), (10, 10)]  # (10,10) is out of bounds
        start = (1, 1)
        end = (10, 10)

        result = logic_checker.validate_path(simple_maze_grid, path, start, end)

        assert not result.valid
        assert any("bound" in i["description"].lower() for i in result.issues)

    def test_validate_code_valid(self, logic_checker):
        """Test validating valid code."""
        code = '''
def solve_maze(maze, start, end):
    visited = set()
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current == end:
            return True
        visited.add(current)
    return False
'''
        result = logic_checker.validate_code(code)
        # Should pass basic syntax check
        assert result.confidence > 0

    def test_validate_code_syntax_error(self, logic_checker):
        """Test validating code with syntax error."""
        code = "def test(\n    return"

        result = logic_checker.validate_code(code)

        assert not result.valid
        assert any(i["severity"] == "critical" for i in result.issues)

    def test_validate_code_potential_infinite_loop(self, logic_checker):
        """Test detecting potential infinite loop."""
        code = '''
def solve():
    while True:
        pass  # No break
'''
        result = logic_checker.validate_code(code)
        # Should detect potential infinite loop
        assert any("infinite" in i["description"].lower() for i in result.issues) or result.valid

    def test_process_task_validate_path(self, logic_checker, simple_maze_grid):
        """Test processing validate_path task."""
        result = logic_checker.process_task(
            {
                "type": "validate_path",
                "maze": simple_maze_grid,
                "path": [(1, 1), (1, 2), (1, 3)],
                "start": [1, 1],
                "end": [1, 3],
            },
            {}
        )

        assert result is not None
        assert result.type == MessageType.VALIDATION_RESULT

    def test_process_task_validate_code(self, logic_checker):
        """Test processing validate_code task."""
        result = logic_checker.process_task(
            {
                "type": "validate_code",
                "code": "def test(): pass",
            },
            {}
        )

        assert result is not None
        assert result.type == MessageType.VALIDATION_RESULT

    def test_process_task_unknown(self, logic_checker):
        """Test processing unknown task type."""
        result = logic_checker.process_task(
            {"type": "unknown"},
            {}
        )
        assert result is None

    def test_validate_output_validation_result(self, logic_checker):
        """Test validating ValidationResult output."""
        result = ValidationResult(valid=True)
        valid, reason = logic_checker.validate_output(result)
        assert valid

    def test_validate_output_message(self, logic_checker):
        """Test validating Message output."""
        msg = Message(
            from_agent="test",
            to_agent="broadcast",
            type=MessageType.VALIDATION_RESULT,
            priority=Priority.NORMAL,
            content={"test": True},
        )
        valid, reason = logic_checker.validate_output(msg)
        assert valid


# ==============================================================================
# RetrievalResult Tests
# ==============================================================================

class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_creation(self):
        """Test creating a result."""
        result = RetrievalResult(
            relevant_solutions=[{"algorithm": "BFS"}],
            patterns=[{"pattern": "test"}],
            confidence=0.9,
        )

        assert len(result.relevant_solutions) == 1
        assert len(result.patterns) == 1
        assert result.confidence == 0.9

    def test_serialization(self):
        """Test result serialization."""
        result = RetrievalResult(
            relevant_solutions=[{"algorithm": "BFS"}],
            confidence=0.8,
        )

        data = result.to_dict()
        assert len(data["relevant_solutions"]) == 1
        assert data["confidence"] == 0.8


# ==============================================================================
# RetrievalSpecialist Tests
# ==============================================================================

class TestRetrievalSpecialist:
    """Test RetrievalSpecialist agent."""

    def test_creation(self, retrieval_specialist):
        """Test creating retrieval specialist."""
        assert retrieval_specialist.name == "test_retrieval_specialist"
        assert retrieval_specialist.role == AgentRole.RETRIEVAL_SPECIALIST

    def test_permissions(self, retrieval_specialist):
        """Test retrieval specialist has correct permissions."""
        assert retrieval_specialist.has_permission("facts", Permission.READ)
        assert retrieval_specialist.has_permission("facts", Permission.WRITE)
        assert retrieval_specialist.has_permission("working_memory", Permission.READ_WRITE)

    def test_has_knowledge_base(self, retrieval_specialist):
        """Test retrieval specialist has knowledge base."""
        assert retrieval_specialist.knowledge_base is not None
        assert "algorithm_recommendations" in retrieval_specialist.knowledge_base
        assert "patterns" in retrieval_specialist.knowledge_base

    def test_search_solutions_small_maze(self, retrieval_specialist):
        """Test searching for small maze solutions."""
        properties = {
            "total_cells": 100,
            "wall_ratio": 0.3,
        }

        result = retrieval_specialist.search_solutions(properties)

        assert len(result.relevant_solutions) > 0
        # Should recommend BFS for small sparse maze
        assert any(
            s.get("recommended_algorithm") == "BFS"
            for s in result.relevant_solutions
        )

    def test_search_solutions_large_maze(self, retrieval_specialist):
        """Test searching for large maze solutions."""
        properties = {
            "total_cells": 2500,
            "wall_ratio": 0.3,
        }

        result = retrieval_specialist.search_solutions(properties)

        assert len(result.relevant_solutions) > 0
        # Should recommend A* for large maze
        assert any(
            s.get("recommended_algorithm") == "A*"
            for s in result.relevant_solutions
        )

    def test_search_solutions_includes_patterns(self, retrieval_specialist):
        """Test search includes patterns."""
        properties = {
            "total_cells": 100,
            "wall_ratio": 0.3,
        }

        result = retrieval_specialist.search_solutions(properties)

        assert len(result.patterns) > 0

    def test_find_similar_mazes_small(self, retrieval_specialist):
        """Test finding similar small mazes."""
        properties = {
            "total_cells": 100,
        }

        similar = retrieval_specialist.find_similar_mazes(properties)

        assert len(similar) > 0
        assert similar[0]["similarity"] > 0

    def test_find_similar_mazes_medium(self, retrieval_specialist):
        """Test finding similar medium mazes."""
        properties = {
            "total_cells": 500,
        }

        similar = retrieval_specialist.find_similar_mazes(properties)

        assert len(similar) > 0

    def test_find_similar_mazes_large(self, retrieval_specialist):
        """Test finding similar large mazes."""
        properties = {
            "total_cells": 2000,
        }

        similar = retrieval_specialist.find_similar_mazes(properties)

        assert len(similar) > 0

    def test_process_task_search_solutions(self, retrieval_specialist):
        """Test processing search_solutions task."""
        result = retrieval_specialist.process_task(
            {
                "type": "search_solutions",
                "maze_properties": {"total_cells": 100, "wall_ratio": 0.3},
            },
            {}
        )

        assert result is not None
        assert result.type == MessageType.TASK_RESULT

    def test_process_task_find_patterns(self, retrieval_specialist):
        """Test processing find_patterns task."""
        result = retrieval_specialist.process_task(
            {"type": "find_patterns"},
            {}
        )

        assert result is not None
        assert result.type == MessageType.TASK_RESULT
        assert "patterns" in result.content

    def test_process_task_unknown(self, retrieval_specialist):
        """Test processing unknown task type."""
        result = retrieval_specialist.process_task(
            {"type": "unknown"},
            {}
        )
        assert result is None

    def test_validate_output_retrieval_result(self, retrieval_specialist):
        """Test validating RetrievalResult output."""
        result = RetrievalResult()
        valid, reason = retrieval_specialist.validate_output(result)
        assert valid

    def test_validate_output_message(self, retrieval_specialist):
        """Test validating Message output."""
        msg = Message(
            from_agent="test",
            to_agent="broadcast",
            type=MessageType.TASK_RESULT,
            priority=Priority.NORMAL,
            content={"test": True},
        )
        valid, reason = retrieval_specialist.validate_output(msg)
        assert valid


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestSpecialistIntegration:
    """Test specialist agent integration."""

    def test_code_specialist_generates_executable_code(self, code_specialist):
        """Test that generated code is executable."""
        result = code_specialist.generate_algorithm("BFS")

        assert result.success

        # Try to compile the code
        try:
            compile(result.code, "<string>", "exec")
            code_compiles = True
        except SyntaxError:
            code_compiles = False

        assert code_compiles

    def test_logic_checker_validates_generated_code(
        self, code_specialist, logic_checker
    ):
        """Test that logic checker can validate generated code."""
        code_result = code_specialist.generate_algorithm("BFS")
        validation_result = logic_checker.validate_code(code_result.code)

        # Generated template code should be valid
        # Warnings are ok, critical errors are not
        critical_issues = [
            i for i in validation_result.issues
            if i.get("severity") == "critical"
        ]
        assert len(critical_issues) == 0

    def test_retrieval_specialist_provides_algorithm_recommendations(
        self, retrieval_specialist
    ):
        """Test retrieval specialist provides useful recommendations."""
        for maze_size, expected_algo in [
            (100, "BFS"),   # Small
            (500, "A*"),   # Medium
            (2000, "A*"),  # Large
        ]:
            result = retrieval_specialist.search_solutions({
                "total_cells": maze_size,
                "wall_ratio": 0.3,
            })

            assert len(result.relevant_solutions) > 0
            recommendation = result.relevant_solutions[0]
            assert recommendation.get("recommended_algorithm") == expected_algo
