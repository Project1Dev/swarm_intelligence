"""State analysis for dynamic agent selection and strategy optimization."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import time

from agents.base import AgentRole
from communication.message import Priority


logger = logging.getLogger(__name__)


class SwarmState(Enum):
    """High-level swarm state categories."""

    INITIAL = "initial"  # Just started, minimal information
    PLANNING = "planning"  # Creating or refining plan
    IMPLEMENTING = "implementing"  # Generating code/solutions
    VALIDATING = "validating"  # Checking correctness
    STUCK = "stuck"  # No progress, may need intervention
    NEAR_COMPLETION = "near_completion"  # Most tasks done
    ERROR_RECOVERY = "error_recovery"  # Handling failures


@dataclass
class StateVector:
    """
    Feature vector representing current swarm state.

    Used for pattern matching and agent selection decisions.
    """

    # Message queue metrics
    messages_pending: int = 0
    messages_critical: int = 0
    messages_by_type: Dict[str, int] = field(default_factory=dict)

    # Knowledge base metrics
    facts_count: int = 0
    hypotheses_count: int = 0
    artifacts_count: int = 0

    # Progress indicators
    subtasks_completed: int = 0
    subtasks_failed: int = 0
    subtasks_total: int = 0
    progress_rate: float = 0.0  # Tasks per second

    # Error tracking
    validation_errors: int = 0
    retry_count: int = 0
    consecutive_failures: int = 0

    # Agent activity
    agent_confidence: Dict[str, float] = field(default_factory=dict)
    tokens_used: int = 0
    tokens_remaining: int = 0

    # Time metrics
    elapsed_time: float = 0.0
    time_since_progress: float = 0.0

    # Derived state
    swarm_state: SwarmState = SwarmState.INITIAL
    is_stalled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messages_pending": self.messages_pending,
            "messages_critical": self.messages_critical,
            "facts_count": self.facts_count,
            "hypotheses_count": self.hypotheses_count,
            "artifacts_count": self.artifacts_count,
            "subtasks_completed": self.subtasks_completed,
            "subtasks_failed": self.subtasks_failed,
            "subtasks_total": self.subtasks_total,
            "progress_rate": self.progress_rate,
            "validation_errors": self.validation_errors,
            "retry_count": self.retry_count,
            "consecutive_failures": self.consecutive_failures,
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.tokens_remaining,
            "elapsed_time": self.elapsed_time,
            "time_since_progress": self.time_since_progress,
            "swarm_state": self.swarm_state.value,
            "is_stalled": self.is_stalled,
        }


@dataclass
class AgentRecommendation:
    """Recommendation for which agent to use for a task."""

    agent_role: AgentRole
    confidence: float
    reasoning: str
    alternative: Optional[AgentRole] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_role": self.agent_role.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternative": self.alternative.value if self.alternative else None,
        }


class StateAnalyzer:
    """
    Analyzes blackboard state to enable dynamic agent selection.

    Responsibilities:
    - Extract features from blackboard
    - Detect swarm state (stuck, progressing, etc.)
    - Recommend agents based on current context
    - Track patterns and success rates
    """

    def __init__(self):
        """Initialize state analyzer."""
        self.history: List[StateVector] = []
        self.pattern_database: Dict[str, Any] = self._initialize_patterns()
        self.agent_success_rates: Dict[Tuple[AgentRole, str], List[bool]] = {}
        self.stall_threshold = 30.0  # seconds without progress

        logger.info("StateAnalyzer initialized")

    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize pattern database for agent selection."""
        return {
            # Pattern: validation errors -> use Logic Checker
            "validation_errors_high": {
                "condition": lambda sv: sv.validation_errors > 2,
                "recommendation": AgentRole.LOGIC_CHECKER,
                "confidence": 0.9,
                "reasoning": "Multiple validation errors detected",
            },

            # Pattern: no code yet -> use Code Specialist
            "no_artifacts": {
                "condition": lambda sv: sv.artifacts_count == 0 and sv.subtasks_completed > 0,
                "recommendation": AgentRole.CODE_SPECIALIST,
                "confidence": 0.85,
                "reasoning": "No code artifacts generated yet",
            },

            # Pattern: consecutive failures -> use Retrieval for similar solutions
            "consecutive_failures": {
                "condition": lambda sv: sv.consecutive_failures >= 2,
                "recommendation": AgentRole.RETRIEVAL_SPECIALIST,
                "confidence": 0.8,
                "reasoning": "Multiple failures, need alternative approach",
            },

            # Pattern: early stage -> use Retrieval for context
            "early_stage": {
                "condition": lambda sv: sv.subtasks_completed < 2 and sv.elapsed_time < 10,
                "recommendation": AgentRole.RETRIEVAL_SPECIALIST,
                "confidence": 0.75,
                "reasoning": "Early stage, gather context first",
            },

            # Pattern: high confidence artifacts -> skip validation
            "high_confidence_code": {
                "condition": lambda sv: sv.artifacts_count > 0 and
                    sv.agent_confidence.get("CodeSpecialist", 0) > 0.9,
                "recommendation": None,  # Skip task
                "confidence": 0.7,
                "reasoning": "High confidence code, validation optional",
            },
        }

    def analyze_blackboard(
        self, blackboard: Any, current_round: int, start_time: float
    ) -> StateVector:
        """
        Analyze current blackboard state.

        Args:
            blackboard: The blackboard to analyze
            current_round: Current execution round
            start_time: When execution started

        Returns:
            StateVector with extracted features
        """
        # Read all sections
        current_goal = blackboard.read("current_goal", "analyzer") or {}
        facts = blackboard.read("facts", "analyzer") or {}
        hypotheses = blackboard.read("hypotheses", "analyzer") or {}
        artifacts = blackboard.read("artifacts", "analyzer") or {}
        metrics = blackboard.read("metrics", "analyzer") or {}
        working_memory = blackboard.read("working_memory", "analyzer") or {}

        # Calculate metrics
        elapsed = time.time() - start_time

        # Get plan info if available
        plan_data = current_goal.get("plan", {})
        subtasks_total = len(plan_data.get("subtasks", []))
        subtasks_completed = sum(
            1 for st in plan_data.get("subtasks", [])
            if st.get("status") == "completed"
        )
        subtasks_failed = sum(
            1 for st in plan_data.get("subtasks", [])
            if st.get("status") == "failed"
        )

        # Calculate progress rate
        progress_rate = subtasks_completed / elapsed if elapsed > 0 else 0

        # Extract validation errors
        validation_data = facts.get("code_validation", {})
        validation_errors = len(validation_data.get("issues", []))

        # Count retries
        retry_count = sum(
            st.get("retry_count", 0)
            for st in plan_data.get("subtasks", [])
        )

        # Calculate consecutive failures
        consecutive_failures = 0
        for st in reversed(plan_data.get("subtasks", [])):
            if st.get("status") == "failed":
                consecutive_failures += 1
            elif st.get("status") in ["completed", "in_progress"]:
                break

        # Build state vector
        state_vector = StateVector(
            messages_pending=0,  # Would read from message queue
            messages_critical=0,
            facts_count=len(facts),
            hypotheses_count=len(hypotheses),
            artifacts_count=len(artifacts),
            subtasks_completed=subtasks_completed,
            subtasks_failed=subtasks_failed,
            subtasks_total=subtasks_total,
            progress_rate=progress_rate,
            validation_errors=validation_errors,
            retry_count=retry_count,
            consecutive_failures=consecutive_failures,
            elapsed_time=elapsed,
            swarm_state=self._determine_swarm_state(
                subtasks_completed, subtasks_total, subtasks_failed, elapsed
            ),
        )

        # Detect stall
        state_vector.is_stalled = self._detect_stall(state_vector)

        # Record in history
        self.history.append(state_vector)

        logger.debug(
            f"State: {state_vector.swarm_state.value}, "
            f"Progress: {subtasks_completed}/{subtasks_total}, "
            f"Stalled: {state_vector.is_stalled}"
        )

        return state_vector

    def _determine_swarm_state(
        self, completed: int, total: int, failed: int, elapsed: float
    ) -> SwarmState:
        """Determine high-level swarm state."""
        if elapsed < 5:
            return SwarmState.INITIAL

        if total == 0:
            return SwarmState.PLANNING

        completion_ratio = completed / total if total > 0 else 0

        if completion_ratio > 0.8:
            return SwarmState.NEAR_COMPLETION

        if failed > 0 and failed >= completed:
            return SwarmState.ERROR_RECOVERY

        if completed == 0 and elapsed > 15:
            return SwarmState.STUCK

        return SwarmState.IMPLEMENTING

    def _detect_stall(self, state_vector: StateVector) -> bool:
        """Detect if swarm is stalled."""
        # Check history for progress
        if len(self.history) < 3:
            return False

        # Look at last 3 states
        recent = self.history[-3:]

        # No progress in last 3 states
        no_progress = all(
            sv.subtasks_completed == state_vector.subtasks_completed
            for sv in recent
        )

        # Been a while since last progress
        time_stalled = state_vector.time_since_progress > self.stall_threshold

        return no_progress or time_stalled

    def recommend_agent(
        self,
        task_type: str,
        state_vector: StateVector,
        default_agent: AgentRole
    ) -> AgentRecommendation:
        """
        Recommend best agent for a task based on current state.

        Args:
            task_type: Type of task to perform
            state_vector: Current state vector
            default_agent: Default agent assignment

        Returns:
            AgentRecommendation with selected agent and reasoning
        """
        # Check pattern-based rules
        for pattern_name, pattern in self.pattern_database.items():
            if pattern["condition"](state_vector):
                recommended_role = pattern["recommendation"]

                # Skip task if pattern recommends it
                if recommended_role is None:
                    return AgentRecommendation(
                        agent_role=default_agent,
                        confidence=pattern["confidence"],
                        reasoning=f"Pattern '{pattern_name}': {pattern['reasoning']} (skip recommended)",
                        alternative=None,
                    )

                # Override default if confidence is high enough
                if pattern["confidence"] > 0.7:
                    logger.info(
                        f"Pattern '{pattern_name}' recommends {recommended_role.value}"
                    )
                    return AgentRecommendation(
                        agent_role=recommended_role,
                        confidence=pattern["confidence"],
                        reasoning=f"Pattern '{pattern_name}': {pattern['reasoning']}",
                        alternative=default_agent,
                    )

        # Check historical success rates
        success_rate = self._get_success_rate(default_agent, task_type)

        if success_rate is not None and success_rate < 0.5:
            # Low success rate, consider alternative
            alternative = self._find_alternative_agent(default_agent, task_type)
            if alternative:
                return AgentRecommendation(
                    agent_role=alternative,
                    confidence=0.6,
                    reasoning=f"Low historical success rate ({success_rate:.1%}) for {default_agent.value}",
                    alternative=default_agent,
                )

        # State-based selection
        state_recommendation = self._state_based_selection(
            task_type, state_vector, default_agent
        )
        if state_recommendation:
            return state_recommendation

        # Default recommendation
        return AgentRecommendation(
            agent_role=default_agent,
            confidence=0.5,
            reasoning="Using default agent assignment",
            alternative=None,
        )

    def _state_based_selection(
        self, task_type: str, state_vector: StateVector, default_agent: AgentRole
    ) -> Optional[AgentRecommendation]:
        """Make recommendations based on swarm state."""
        state = state_vector.swarm_state

        if state == SwarmState.STUCK:
            # Stuck - try retrieval for alternative approaches
            return AgentRecommendation(
                agent_role=AgentRole.RETRIEVAL_SPECIALIST,
                confidence=0.8,
                reasoning="Swarm stuck, retrieving alternative approaches",
                alternative=default_agent,
            )

        if state == SwarmState.ERROR_RECOVERY and state_vector.validation_errors > 0:
            # Errors present - prioritize validation
            return AgentRecommendation(
                agent_role=AgentRole.LOGIC_CHECKER,
                confidence=0.85,
                reasoning="Error recovery mode with validation errors",
                alternative=default_agent,
            )

        return None

    def _get_success_rate(
        self, agent_role: AgentRole, task_type: str
    ) -> Optional[float]:
        """Get historical success rate for agent on task type."""
        key = (agent_role, task_type)
        if key not in self.agent_success_rates or not self.agent_success_rates[key]:
            return None

        successes = self.agent_success_rates[key]
        return sum(successes) / len(successes)

    def _find_alternative_agent(
        self, current_agent: AgentRole, task_type: str
    ) -> Optional[AgentRole]:
        """Find alternative agent with better success rate."""
        best_agent = None
        best_rate = 0.0

        for agent_role in AgentRole:
            if agent_role == current_agent:
                continue

            rate = self._get_success_rate(agent_role, task_type)
            if rate is not None and rate > best_rate:
                best_rate = rate
                best_agent = agent_role

        return best_agent if best_rate > 0.5 else None

    def record_task_outcome(
        self, agent_role: AgentRole, task_type: str, success: bool
    ):
        """Record outcome of a task for learning."""
        key = (agent_role, task_type)
        if key not in self.agent_success_rates:
            self.agent_success_rates[key] = []

        self.agent_success_rates[key].append(success)

        # Keep last 20 outcomes only
        if len(self.agent_success_rates[key]) > 20:
            self.agent_success_rates[key] = self.agent_success_rates[key][-20:]

        logger.debug(
            f"Recorded {agent_role.value} on {task_type}: "
            f"{'success' if success else 'failure'}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        if not self.history:
            return {"states_analyzed": 0}

        latest = self.history[-1]

        return {
            "states_analyzed": len(self.history),
            "current_state": latest.swarm_state.value,
            "is_stalled": latest.is_stalled,
            "progress_rate": latest.progress_rate,
            "validation_errors": latest.validation_errors,
            "agent_success_rates": {
                f"{role.value}_{task}": sum(outcomes) / len(outcomes)
                for (role, task), outcomes in self.agent_success_rates.items()
                if outcomes
            },
        }

    def reset(self):
        """Reset analyzer state."""
        self.history = []
        logger.debug("StateAnalyzer reset")


class StrategyComparator:
    """
    Compares multiple strategies and selects the best one.

    Used when multiple algorithms or approaches are generated.
    """

    @staticmethod
    def compare_algorithms(
        strategies: List[Dict[str, Any]],
        maze_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithm strategies.

        Args:
            strategies: List of strategy dictionaries
            maze_properties: Properties of the maze being solved

        Returns:
            Best strategy based on analysis
        """
        if not strategies:
            return None

        if len(strategies) == 1:
            return strategies[0]

        # Score each strategy
        scored = []
        for strategy in strategies:
            score = StrategyComparator._score_strategy(strategy, maze_properties)
            scored.append((score, strategy))

        # Return highest scoring
        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_strategy = scored[0]

        logger.info(
            f"Selected strategy with score {best_score:.2f}: "
            f"{best_strategy.get('algorithm', 'unknown')}"
        )

        return best_strategy

    @staticmethod
    def _score_strategy(
        strategy: Dict[str, Any],
        maze_properties: Dict[str, Any]
    ) -> float:
        """Score a strategy based on maze properties."""
        score = 0.5  # Base score

        algorithm = strategy.get("algorithm", "").upper()
        total_cells = maze_properties.get("total_cells", 0)
        wall_ratio = maze_properties.get("wall_ratio", 0.5)

        # Algorithm-specific scoring
        if algorithm == "BFS":
            # BFS good for small mazes, guarantees optimal
            if total_cells < 300:
                score += 0.3
            score += 0.2  # Bonus for optimality

        elif algorithm == "A*":
            # A* good for medium/large sparse mazes
            if total_cells >= 300:
                score += 0.3
            if wall_ratio < 0.4:  # Sparse
                score += 0.2

        elif algorithm == "DFS":
            # DFS uses less memory but not optimal
            if total_cells > 1000:
                score += 0.1  # Memory advantage
            score -= 0.1  # Penalty for non-optimality

        # Confidence bonus
        confidence = strategy.get("confidence", 0.5)
        score += confidence * 0.2

        return score
