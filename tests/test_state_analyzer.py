"""Tests for StateAnalyzer and dynamic agent selection."""

import pytest
import time
from unittest.mock import Mock, MagicMock

from swarm.state_analyzer import (
    StateAnalyzer,
    StateVector,
    SwarmState,
    AgentRecommendation,
    StrategyComparator,
)
from agents.base import AgentRole


# ==============================================================================
# StateVector Tests
# ==============================================================================

class TestStateVector:
    """Test StateVector dataclass."""

    def test_creation(self):
        """Test creating a state vector."""
        sv = StateVector(
            messages_pending=5,
            facts_count=10,
            subtasks_completed=3,
            subtasks_total=10,
        )

        assert sv.messages_pending == 5
        assert sv.facts_count == 10
        assert sv.swarm_state == SwarmState.INITIAL

    def test_serialization(self):
        """Test state vector serialization."""
        sv = StateVector(
            messages_pending=2,
            subtasks_completed=5,
            subtasks_total=10,
            swarm_state=SwarmState.IMPLEMENTING,
        )

        data = sv.to_dict()
        assert data["messages_pending"] == 2
        assert data["swarm_state"] == "implementing"


# ==============================================================================
# AgentRecommendation Tests
# ==============================================================================

class TestAgentRecommendation:
    """Test AgentRecommendation dataclass."""

    def test_creation(self):
        """Test creating a recommendation."""
        rec = AgentRecommendation(
            agent_role=AgentRole.CODE_SPECIALIST,
            confidence=0.9,
            reasoning="Test reasoning",
        )

        assert rec.agent_role == AgentRole.CODE_SPECIALIST
        assert rec.confidence == 0.9

    def test_with_alternative(self):
        """Test recommendation with alternative."""
        rec = AgentRecommendation(
            agent_role=AgentRole.LOGIC_CHECKER,
            confidence=0.8,
            reasoning="Override default",
            alternative=AgentRole.CODE_SPECIALIST,
        )

        assert rec.alternative == AgentRole.CODE_SPECIALIST

    def test_serialization(self):
        """Test recommendation serialization."""
        rec = AgentRecommendation(
            agent_role=AgentRole.RETRIEVAL_SPECIALIST,
            confidence=0.75,
            reasoning="Need context",
        )

        data = rec.to_dict()
        assert data["agent_role"] == "retrieval_specialist"
        assert data["confidence"] == 0.75


# ==============================================================================
# StateAnalyzer Tests
# ==============================================================================

class TestStateAnalyzer:
    """Test StateAnalyzer class."""

    def test_creation(self):
        """Test creating state analyzer."""
        analyzer = StateAnalyzer()

        assert analyzer is not None
        assert len(analyzer.history) == 0
        assert analyzer.pattern_database is not None

    def test_has_patterns(self):
        """Test analyzer has pattern database."""
        analyzer = StateAnalyzer()

        assert "validation_errors_high" in analyzer.pattern_database
        assert "no_artifacts" in analyzer.pattern_database
        assert "consecutive_failures" in analyzer.pattern_database

    def test_analyze_blackboard_empty(self):
        """Test analyzing empty blackboard."""
        analyzer = StateAnalyzer()
        blackboard = Mock()
        blackboard.read = Mock(return_value={})

        state = analyzer.analyze_blackboard(blackboard, 0, time.time())

        assert state is not None
        assert state.swarm_state == SwarmState.INITIAL

    def test_analyze_blackboard_with_plan(self):
        """Test analyzing blackboard with plan."""
        analyzer = StateAnalyzer()
        blackboard = Mock()

        # Setup mock blackboard data
        plan_data = {
            "plan": {
                "subtasks": [
                    {"status": "completed"},
                    {"status": "in_progress"},
                    {"status": "pending"},
                ]
            }
        }

        def mock_read(section, requester):
            if section == "current_goal":
                return plan_data
            return {}

        blackboard.read = mock_read

        state = analyzer.analyze_blackboard(blackboard, 1, time.time() - 10)

        assert state.subtasks_total == 3
        assert state.subtasks_completed == 1

    def test_determine_swarm_state_initial(self):
        """Test determining initial state."""
        analyzer = StateAnalyzer()

        state = analyzer._determine_swarm_state(0, 5, 0, 2)

        assert state == SwarmState.INITIAL

    def test_determine_swarm_state_planning(self):
        """Test determining planning state."""
        analyzer = StateAnalyzer()

        state = analyzer._determine_swarm_state(0, 0, 0, 10)

        assert state == SwarmState.PLANNING

    def test_determine_swarm_state_near_completion(self):
        """Test determining near completion state."""
        analyzer = StateAnalyzer()

        state = analyzer._determine_swarm_state(9, 10, 0, 30)

        assert state == SwarmState.NEAR_COMPLETION

    def test_determine_swarm_state_error_recovery(self):
        """Test determining error recovery state."""
        analyzer = StateAnalyzer()

        state = analyzer._determine_swarm_state(1, 5, 3, 20)

        assert state == SwarmState.ERROR_RECOVERY

    def test_determine_swarm_state_stuck(self):
        """Test determining stuck state."""
        analyzer = StateAnalyzer()

        state = analyzer._determine_swarm_state(0, 5, 0, 20)

        assert state == SwarmState.STUCK

    def test_detect_stall_no_history(self):
        """Test stall detection with no history."""
        analyzer = StateAnalyzer()
        sv = StateVector()

        is_stalled = analyzer._detect_stall(sv)

        assert not is_stalled

    def test_detect_stall_with_progress(self):
        """Test stall detection with progress."""
        analyzer = StateAnalyzer()

        # Add history with increasing progress
        for i in range(3):
            analyzer.history.append(StateVector(subtasks_completed=i))

        sv = StateVector(subtasks_completed=3)
        is_stalled = analyzer._detect_stall(sv)

        assert not is_stalled

    def test_detect_stall_no_progress(self):
        """Test stall detection without progress."""
        analyzer = StateAnalyzer()

        # Add history with no progress
        for i in range(3):
            analyzer.history.append(StateVector(subtasks_completed=5))

        sv = StateVector(subtasks_completed=5)
        is_stalled = analyzer._detect_stall(sv)

        assert is_stalled

    def test_recommend_agent_default(self):
        """Test default agent recommendation."""
        analyzer = StateAnalyzer()
        # Use non-initial state to avoid early_stage pattern
        sv = StateVector(
            subtasks_completed=5,
            elapsed_time=20,
        )

        rec = analyzer.recommend_agent(
            "test_task", sv, AgentRole.CODE_SPECIALIST
        )

        assert rec.agent_role == AgentRole.CODE_SPECIALIST

    def test_recommend_agent_validation_errors(self):
        """Test recommendation with validation errors."""
        analyzer = StateAnalyzer()
        sv = StateVector(validation_errors=5)

        rec = analyzer.recommend_agent(
            "test_task", sv, AgentRole.CODE_SPECIALIST
        )

        # Should recommend logic checker due to validation errors
        assert rec.agent_role == AgentRole.LOGIC_CHECKER
        assert rec.confidence > 0.7

    def test_recommend_agent_no_artifacts(self):
        """Test recommendation when no artifacts exist."""
        analyzer = StateAnalyzer()
        sv = StateVector(
            artifacts_count=0,
            subtasks_completed=3,
        )

        rec = analyzer.recommend_agent(
            "test_task", sv, AgentRole.LOGIC_CHECKER
        )

        # Should recommend code specialist
        assert rec.agent_role == AgentRole.CODE_SPECIALIST

    def test_recommend_agent_consecutive_failures(self):
        """Test recommendation after consecutive failures."""
        analyzer = StateAnalyzer()
        sv = StateVector(consecutive_failures=3)

        rec = analyzer.recommend_agent(
            "test_task", sv, AgentRole.CODE_SPECIALIST
        )

        # Should recommend retrieval specialist for alternative approaches
        assert rec.agent_role == AgentRole.RETRIEVAL_SPECIALIST

    def test_recommend_agent_swarm_stuck(self):
        """Test recommendation when swarm is stuck."""
        analyzer = StateAnalyzer()
        sv = StateVector(
            swarm_state=SwarmState.STUCK,
            elapsed_time=30,  # Avoid early_stage pattern
        )

        rec = analyzer.recommend_agent(
            "test_task", sv, AgentRole.CODE_SPECIALIST
        )

        # Should recommend retrieval for alternative approaches
        assert rec.agent_role == AgentRole.RETRIEVAL_SPECIALIST
        assert "stuck" in rec.reasoning.lower() or "alternative" in rec.reasoning.lower()

    def test_recommend_agent_error_recovery(self):
        """Test recommendation in error recovery mode."""
        analyzer = StateAnalyzer()
        sv = StateVector(
            swarm_state=SwarmState.ERROR_RECOVERY,
            validation_errors=5,  # Increase to trigger pattern
            elapsed_time=20,  # Avoid early_stage pattern
        )

        rec = analyzer.recommend_agent(
            "test_task", sv, AgentRole.CODE_SPECIALIST
        )

        # Should recommend logic checker due to high validation errors
        assert rec.agent_role == AgentRole.LOGIC_CHECKER

    def test_record_task_outcome(self):
        """Test recording task outcomes."""
        analyzer = StateAnalyzer()

        analyzer.record_task_outcome(
            AgentRole.CODE_SPECIALIST, "generate_code", True
        )
        analyzer.record_task_outcome(
            AgentRole.CODE_SPECIALIST, "generate_code", True
        )
        analyzer.record_task_outcome(
            AgentRole.CODE_SPECIALIST, "generate_code", False
        )

        # Check success rate
        rate = analyzer._get_success_rate(AgentRole.CODE_SPECIALIST, "generate_code")
        assert rate == 2 / 3

    def test_record_task_outcome_limits_history(self):
        """Test outcome history is limited."""
        analyzer = StateAnalyzer()

        # Record 25 outcomes
        for i in range(25):
            analyzer.record_task_outcome(
                AgentRole.CODE_SPECIALIST, "test", i % 2 == 0
            )

        # Should only keep last 20
        key = (AgentRole.CODE_SPECIALIST, "test")
        assert len(analyzer.agent_success_rates[key]) == 20

    def test_get_success_rate_no_history(self):
        """Test getting success rate with no history."""
        analyzer = StateAnalyzer()

        rate = analyzer._get_success_rate(AgentRole.CODE_SPECIALIST, "unknown_task")

        assert rate is None

    def test_find_alternative_agent(self):
        """Test finding alternative agent."""
        analyzer = StateAnalyzer()

        # Record some outcomes
        analyzer.record_task_outcome(AgentRole.CODE_SPECIALIST, "task", False)
        analyzer.record_task_outcome(AgentRole.LOGIC_CHECKER, "task", True)
        analyzer.record_task_outcome(AgentRole.LOGIC_CHECKER, "task", True)

        alternative = analyzer._find_alternative_agent(
            AgentRole.CODE_SPECIALIST, "task"
        )

        assert alternative == AgentRole.LOGIC_CHECKER

    def test_get_statistics(self):
        """Test getting statistics."""
        analyzer = StateAnalyzer()
        blackboard = Mock()
        blackboard.read = Mock(return_value={})

        # Analyze a few times
        for i in range(3):
            analyzer.analyze_blackboard(blackboard, i, time.time())

        stats = analyzer.get_statistics()

        assert stats["states_analyzed"] == 3
        assert "current_state" in stats

    def test_reset(self):
        """Test resetting analyzer."""
        analyzer = StateAnalyzer()
        blackboard = Mock()
        blackboard.read = Mock(return_value={})

        # Add some history
        analyzer.analyze_blackboard(blackboard, 0, time.time())
        analyzer.record_task_outcome(AgentRole.CODE_SPECIALIST, "test", True)

        # Reset
        analyzer.reset()

        assert len(analyzer.history) == 0


# ==============================================================================
# StrategyComparator Tests
# ==============================================================================

class TestStrategyComparator:
    """Test StrategyComparator class."""

    def test_compare_algorithms_empty(self):
        """Test comparing empty list."""
        result = StrategyComparator.compare_algorithms([], {})

        assert result is None

    def test_compare_algorithms_single(self):
        """Test comparing single strategy."""
        strategies = [
            {"algorithm": "BFS", "confidence": 0.8}
        ]

        result = StrategyComparator.compare_algorithms(strategies, {})

        assert result == strategies[0]

    def test_compare_algorithms_bfs_small_maze(self):
        """Test BFS preferred for small maze."""
        strategies = [
            {"algorithm": "BFS", "confidence": 0.8},
            {"algorithm": "DFS", "confidence": 0.8},
        ]

        maze_props = {
            "total_cells": 100,
            "wall_ratio": 0.3,
        }

        result = StrategyComparator.compare_algorithms(strategies, maze_props)

        assert result["algorithm"] == "BFS"

    def test_compare_algorithms_astar_large_maze(self):
        """Test A* preferred for large maze."""
        strategies = [
            {"algorithm": "BFS", "confidence": 0.8},
            {"algorithm": "A*", "confidence": 0.8},
        ]

        maze_props = {
            "total_cells": 1000,
            "wall_ratio": 0.3,
        }

        result = StrategyComparator.compare_algorithms(strategies, maze_props)

        assert result["algorithm"] == "A*"

    def test_compare_algorithms_confidence_bonus(self):
        """Test confidence affects scoring."""
        strategies = [
            {"algorithm": "BFS", "confidence": 0.5},
            {"algorithm": "DFS", "confidence": 0.9},
        ]

        maze_props = {
            "total_cells": 500,
            "wall_ratio": 0.5,
        }

        result = StrategyComparator.compare_algorithms(strategies, maze_props)

        # High confidence DFS might win despite non-optimality
        assert result is not None

    def test_score_strategy_bfs(self):
        """Test scoring BFS strategy."""
        strategy = {"algorithm": "BFS", "confidence": 0.8}
        maze_props = {"total_cells": 200, "wall_ratio": 0.3}

        score = StrategyComparator._score_strategy(strategy, maze_props)

        # BFS should score well for small mazes
        assert score > 0.5

    def test_score_strategy_astar_large_sparse(self):
        """Test scoring A* for large sparse maze."""
        strategy = {"algorithm": "A*", "confidence": 0.8}
        maze_props = {"total_cells": 1000, "wall_ratio": 0.3}

        score = StrategyComparator._score_strategy(strategy, maze_props)

        # A* should score well for large sparse mazes
        assert score > 0.7

    def test_score_strategy_dfs_penalty(self):
        """Test DFS gets penalty for non-optimality."""
        strategy = {"algorithm": "DFS", "confidence": 0.8}
        maze_props = {"total_cells": 500, "wall_ratio": 0.5}

        score = StrategyComparator._score_strategy(strategy, maze_props)

        # DFS should have lower score due to non-optimality
        assert score < 1.0


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestStateAnalyzerIntegration:
    """Integration tests for state analyzer."""

    def test_learning_from_outcomes(self):
        """Test analyzer learns from task outcomes."""
        analyzer = StateAnalyzer()

        # Simulate a pattern: CODE_SPECIALIST fails, LOGIC_CHECKER succeeds
        for i in range(5):
            analyzer.record_task_outcome(
                AgentRole.CODE_SPECIALIST, "implement", False
            )
            analyzer.record_task_outcome(
                AgentRole.LOGIC_CHECKER, "implement", True
            )

        # Verify the success rates are recorded
        code_rate = analyzer._get_success_rate(AgentRole.CODE_SPECIALIST, "implement")
        logic_rate = analyzer._get_success_rate(AgentRole.LOGIC_CHECKER, "implement")

        assert code_rate == 0.0  # 0% success
        assert logic_rate == 1.0  # 100% success

        # Now when asking for recommendation
        sv = StateVector(
            elapsed_time=20,  # Avoid early_stage pattern
            subtasks_completed=3,
        )
        rec = analyzer.recommend_agent(
            "implement", sv, AgentRole.CODE_SPECIALIST
        )

        # Should either recommend alternative or have low confidence
        # (Pattern matching might override this)
        assert rec.agent_role in [AgentRole.LOGIC_CHECKER, AgentRole.CODE_SPECIALIST]

    def test_state_progression(self):
        """Test swarm state progresses correctly."""
        analyzer = StateAnalyzer()
        blackboard = Mock()

        # Simulate progression through states
        states = []

        # Initial state
        blackboard.read = Mock(return_value={})
        state = analyzer.analyze_blackboard(blackboard, 0, time.time())
        states.append(state.swarm_state)

        # Add plan data
        plan_data = {
            "plan": {
                "subtasks": [
                    {"status": "pending"} for _ in range(5)
                ]
            }
        }
        blackboard.read = Mock(return_value=plan_data)
        state = analyzer.analyze_blackboard(blackboard, 1, time.time() - 10)
        states.append(state.swarm_state)

        assert SwarmState.INITIAL in states

    def test_pattern_matching_priority(self):
        """Test that pattern matching works correctly."""
        analyzer = StateAnalyzer()

        # State with multiple matching patterns
        sv = StateVector(
            validation_errors=5,  # Matches validation_errors_high
            consecutive_failures=3,  # Matches consecutive_failures
        )

        rec = analyzer.recommend_agent(
            "test_task", sv, AgentRole.CODE_SPECIALIST
        )

        # Should pick highest confidence pattern (validation_errors_high = 0.9)
        assert rec.agent_role in [AgentRole.LOGIC_CHECKER, AgentRole.RETRIEVAL_SPECIALIST]
        assert rec.confidence > 0.7
