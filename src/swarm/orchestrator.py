"""Swarm orchestrator for coordinating agent collaboration."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type
from enum import Enum

from agents.base import BaseAgent, AgentConfig, AgentRole, Permission
from agents.leader import LeaderAgent, Plan, Subtask, SubtaskType, SubtaskStatus
from agents.specialists import CodeSpecialist, LogicChecker, RetrievalSpecialist
from blackboard.blackboard import Blackboard, SectionConfig
from communication.message import Message, MessageType, Priority, MessageRouter
from swarm.state_analyzer import StateAnalyzer, StateVector, AgentRecommendation
from utils.maze_generator import Maze
from experiments.baseline_algorithms import PathfindingResult


logger = logging.getLogger(__name__)


class SwarmStatus(Enum):
    """Status of the swarm."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SwarmResult:
    """Result of swarm execution."""

    success: bool
    path: Optional[List[tuple]]
    path_length: int
    nodes_explored: int
    execution_time: float
    algorithm_used: str
    rounds: int
    total_tokens: int
    agent_contributions: Dict[str, int]
    status: SwarmStatus
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "path": self.path,
            "path_length": self.path_length,
            "nodes_explored": self.nodes_explored,
            "execution_time": self.execution_time,
            "algorithm_used": self.algorithm_used,
            "rounds": self.rounds,
            "total_tokens": self.total_tokens,
            "agent_contributions": self.agent_contributions,
            "status": self.status.value,
            "error": self.error,
        }


@dataclass
class SwarmConfig:
    """Configuration for the swarm."""

    time_limit: float = 60.0  # seconds
    max_rounds: int = 10
    token_budget: int = 10000
    parallel_execution: bool = False  # GPU constraint
    progress_check_interval: float = 5.0
    fallback_algorithm: str = "BFS"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_limit": self.time_limit,
            "max_rounds": self.max_rounds,
            "token_budget": self.token_budget,
            "parallel_execution": self.parallel_execution,
            "progress_check_interval": self.progress_check_interval,
            "fallback_algorithm": self.fallback_algorithm,
        }


class SwarmOrchestrator:
    """
    Orchestrates the multi-agent swarm for maze solving.

    Responsibilities:
    - Manage agent lifecycle and communication
    - Coordinate task execution according to plan
    - Handle timeouts and error recovery
    - Collect metrics and produce results
    """

    def __init__(
        self,
        ollama_client: Any,
        config: Optional[SwarmConfig] = None,
    ):
        """
        Initialize swarm orchestrator.

        Args:
            ollama_client: Ollama client for LLM inference
            config: Swarm configuration
        """
        self.ollama_client = ollama_client
        self.config = config or SwarmConfig()

        # Initialize shared components
        self.blackboard = Blackboard()
        self.router = MessageRouter()
        self.state_analyzer = StateAnalyzer()

        # Initialize agents
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self.leader: Optional[LeaderAgent] = None

        # State tracking
        self.status = SwarmStatus.IDLE
        self.current_round = 0
        self.total_tokens = 0
        self.start_time = 0.0

        # Metrics
        self.agent_contributions: Dict[str, int] = {}

        logger.info("SwarmOrchestrator initialized with StateAnalyzer")

    def initialize_agents(self, agent_configs: Optional[Dict[str, AgentConfig]] = None):
        """
        Initialize all agents in the swarm.

        Args:
            agent_configs: Optional custom configurations for agents
        """
        if agent_configs is None:
            agent_configs = self._default_agent_configs()

        # Create leader
        leader_config = agent_configs.get("leader", self._default_leader_config())
        self.leader = LeaderAgent(leader_config, self.ollama_client, self.blackboard)
        self.agents[AgentRole.LEADER] = self.leader
        self.router.register_agent(self.leader.name)

        # Create specialists
        for role, cls in [
            (AgentRole.CODE_SPECIALIST, CodeSpecialist),
            (AgentRole.LOGIC_CHECKER, LogicChecker),
            (AgentRole.RETRIEVAL_SPECIALIST, RetrievalSpecialist),
        ]:
            config = agent_configs.get(role.value, self._default_config_for_role(role))
            agent = cls(config, self.ollama_client, self.blackboard)
            self.agents[role] = agent
            self.router.register_agent(agent.name)
            self.agent_contributions[agent.name] = 0

        self.agent_contributions[self.leader.name] = 0

        logger.info(f"Initialized {len(self.agents)} agents")

    def _default_agent_configs(self) -> Dict[str, AgentConfig]:
        """Get default agent configurations."""
        return {
            "leader": self._default_leader_config(),
            AgentRole.CODE_SPECIALIST.value: self._default_config_for_role(
                AgentRole.CODE_SPECIALIST
            ),
            AgentRole.LOGIC_CHECKER.value: self._default_config_for_role(
                AgentRole.LOGIC_CHECKER
            ),
            AgentRole.RETRIEVAL_SPECIALIST.value: self._default_config_for_role(
                AgentRole.RETRIEVAL_SPECIALIST
            ),
        }

    def _default_leader_config(self) -> AgentConfig:
        """Get default leader configuration."""
        return AgentConfig(
            name="Leader",
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

    def _default_config_for_role(self, role: AgentRole) -> AgentConfig:
        """Get default configuration for a role."""
        configs = {
            AgentRole.CODE_SPECIALIST: AgentConfig(
                name="CodeSpecialist",
                role=AgentRole.CODE_SPECIALIST,
                model="deepseek-coder:1.3b",
                temperature=0.1,
                max_tokens=2000,
                token_budget=8000,
                permissions={
                    "current_goal": Permission.READ,
                    "facts": Permission.READ,
                    "hypotheses": Permission.READ,
                    "artifacts": Permission.READ_WRITE,
                    "working_memory": Permission.READ_WRITE,
                    "metrics": Permission.READ,
                },
            ),
            AgentRole.LOGIC_CHECKER: AgentConfig(
                name="LogicChecker",
                role=AgentRole.LOGIC_CHECKER,
                model="qwen2.5:1.5b",
                temperature=0.0,
                max_tokens=1500,
                token_budget=4000,
                permissions={
                    "current_goal": Permission.READ,
                    "facts": Permission.READ_WRITE,
                    "hypotheses": Permission.READ_WRITE,
                    "artifacts": Permission.READ,
                    "working_memory": Permission.READ,
                    "metrics": Permission.READ_WRITE,
                },
            ),
            AgentRole.RETRIEVAL_SPECIALIST: AgentConfig(
                name="RetrievalSpecialist",
                role=AgentRole.RETRIEVAL_SPECIALIST,
                model="tinyllama:1.1b",
                temperature=0.2,
                max_tokens=800,
                token_budget=3000,
                permissions={
                    "current_goal": Permission.READ,
                    "facts": Permission.READ_WRITE,
                    "hypotheses": Permission.READ,
                    "artifacts": Permission.READ,
                    "working_memory": Permission.READ_WRITE,
                    "metrics": Permission.READ,
                },
            ),
        }
        return configs.get(role)

    def solve_maze(self, maze: Maze) -> SwarmResult:
        """
        Solve a maze using the agent swarm.

        Args:
            maze: The maze to solve

        Returns:
            SwarmResult with solution and metrics
        """
        self.start_time = time.time()
        self.status = SwarmStatus.PLANNING
        self.current_round = 0
        self.total_tokens = 0

        logger.info(f"Starting maze solve: {maze.width}x{maze.height}")

        # Ensure agents are initialized
        if not self.agents:
            self.initialize_agents()

        # Activate all agents
        for agent in self.agents.values():
            agent.activate()

        # Write maze info to blackboard
        self._write_maze_to_blackboard(maze)

        try:
            # Phase 1: Leader creates plan
            plan = self._planning_phase(maze)

            if plan is None:
                return self._create_failure_result("Planning failed")

            # Phase 2: Execute plan
            self.status = SwarmStatus.EXECUTING
            result = self._execution_phase(plan, maze)

            return result

        except TimeoutError:
            logger.error("Swarm execution timed out")
            return self._create_failure_result("Timeout", SwarmStatus.TIMEOUT)

        except Exception as e:
            logger.error(f"Swarm execution error: {e}")
            return self._create_failure_result(str(e))

        finally:
            # Deactivate all agents
            for agent in self.agents.values():
                agent.deactivate()

    def _write_maze_to_blackboard(self, maze: Maze):
        """Write maze information to blackboard."""
        maze_info = {
            "width": maze.width,
            "height": maze.height,
            "start": maze.start,
            "end": maze.end,
            "total_cells": maze.width * maze.height,
            "algorithm": maze.algorithm,
        }
        self.blackboard.write("current_goal", "maze", maze_info, "orchestrator")
        self.blackboard.write("facts", "maze_dimensions", f"{maze.width}x{maze.height}", "orchestrator")

    def _planning_phase(self, maze: Maze) -> Optional[Plan]:
        """Execute planning phase with leader."""
        logger.info("Starting planning phase")

        # Check timeout
        if self._check_timeout():
            raise TimeoutError("Timeout during planning")

        # Leader decomposes task
        plan = self.leader.decompose_task(maze)
        self._update_tokens(self.leader)

        if not plan or not plan.subtasks:
            logger.error("Leader failed to create plan")
            return None

        logger.info(f"Plan created with {len(plan.subtasks)} subtasks")
        return plan

    def _execution_phase(self, plan: Plan, maze: Maze) -> SwarmResult:
        """Execute the plan's subtasks."""
        logger.info("Starting execution phase")

        while not plan.is_complete() and self.current_round < self.config.max_rounds:
            self.current_round += 1

            # Check timeout
            if self._check_timeout():
                logger.warning("Timeout during execution")
                return self._handle_timeout(plan, maze)

            # Get next subtask
            subtask = plan.get_next_subtask()

            if subtask is None:
                # No more subtasks ready, check if we're stuck
                if plan.has_failed():
                    return self._handle_failure(plan, maze)
                break

            # Execute subtask
            logger.info(f"Round {self.current_round}: Executing {subtask.type.value}")
            success = self._execute_subtask(subtask, plan, maze)

            if not success and subtask.priority == Priority.CRITICAL:
                # Critical failure - try to replan
                failure_info = {
                    "subtask_id": subtask.id,
                    "error": "Execution failed",
                    "retry_count": subtask.retry_count,
                }

                new_plan = self.leader.replan(failure_info)
                if new_plan and new_plan != plan:
                    plan = new_plan
                    logger.info("Replanning successful")

        # Extract result from blackboard
        return self._extract_result(plan, maze)

    def _execute_subtask(
        self, subtask: Subtask, plan: Plan, maze: Maze
    ) -> bool:
        """Execute a single subtask."""
        subtask.status = SubtaskStatus.IN_PROGRESS
        plan.updated_at = time.time()

        # Analyze current state
        state_vector = self.state_analyzer.analyze_blackboard(
            self.blackboard, self.current_round, self.start_time
        )

        # Get leader's default recommendation
        default_agent_role = self.leader.select_agent(
            subtask, self.blackboard.read("facts", "orchestrator") or {}
        )

        # Use state analyzer for dynamic selection
        recommendation = self.state_analyzer.recommend_agent(
            subtask.type.value, state_vector, default_agent_role
        )

        agent_role = recommendation.agent_role
        agent = self.agents.get(agent_role)

        if agent is None:
            logger.error(f"No agent found for role {agent_role}")
            subtask.status = SubtaskStatus.FAILED
            return False

        logger.info(
            f"Selected {agent_role.value} (confidence: {recommendation.confidence:.2f}) - "
            f"{recommendation.reasoning}"
        )

        # Build task and context
        task, context = self._build_task_context(subtask, maze)

        try:
            # Execute
            start = time.time()
            result = agent.process_task(task, context)
            elapsed = time.time() - start

            self._update_tokens(agent)

            if result is not None:
                # Route message
                self.router.route(result)

                # Update subtask
                subtask.status = SubtaskStatus.COMPLETED
                subtask.result = result.content if isinstance(result, Message) else result

                # Record success for learning
                self.state_analyzer.record_task_outcome(
                    agent_role, subtask.type.value, True
                )

                logger.info(
                    f"Subtask {subtask.id} completed by {agent.name} in {elapsed:.2f}s"
                )
                return True
            else:
                subtask.status = SubtaskStatus.FAILED
                subtask.retry_count += 1

                # Record failure for learning
                self.state_analyzer.record_task_outcome(
                    agent_role, subtask.type.value, False
                )

                logger.warning(f"Subtask {subtask.id} returned None")
                return False

        except Exception as e:
            logger.error(f"Subtask {subtask.id} failed: {e}")
            subtask.status = SubtaskStatus.FAILED
            subtask.retry_count += 1

            # Record failure for learning
            self.state_analyzer.record_task_outcome(
                agent_role, subtask.type.value, False
            )

            return False

    def _build_task_context(
        self, subtask: Subtask, maze: Maze
    ) -> tuple:
        """Build task dict and context for a subtask."""
        task = {"type": subtask.type.value, "description": subtask.description}
        context = {"maze": maze}

        # Add type-specific data
        if subtask.type == SubtaskType.IMPLEMENT_SOLUTION:
            # Get recommended algorithm from plan
            algorithm = "BFS"  # Default
            if self.leader and self.leader.current_plan:
                algorithm = self.leader.current_plan.metadata.get(
                    "recommended_algorithm", "BFS"
                )
            task["algorithm"] = algorithm

        elif subtask.type == SubtaskType.VALIDATE_SOLUTION:
            # Get code from blackboard
            artifacts = self.blackboard.read("artifacts", "orchestrator") or {}
            task["code"] = artifacts.get("code_BFS", {}).get("code", "")

        elif subtask.type == SubtaskType.RETRIEVE_KNOWLEDGE:
            # Add maze properties
            task["maze_properties"] = {
                "total_cells": maze.width * maze.height,
                "wall_ratio": self._calculate_wall_ratio(maze),
            }

        return task, context

    def _calculate_wall_ratio(self, maze: Maze) -> float:
        """Calculate wall ratio of maze."""
        total = maze.width * maze.height
        paths = sum(1 for row in maze.grid for cell in row if cell.value == 0)
        return 1 - (paths / total)

    def _update_tokens(self, agent: BaseAgent):
        """Update token tracking."""
        tokens_used = agent.state.tokens_used
        if agent.name in self.agent_contributions:
            self.agent_contributions[agent.name] += tokens_used
        self.total_tokens += tokens_used

    def _check_timeout(self) -> bool:
        """Check if time limit exceeded."""
        return time.time() - self.start_time > self.config.time_limit

    def _handle_timeout(self, plan: Plan, maze: Maze) -> SwarmResult:
        """Handle timeout by using fallback."""
        logger.warning("Handling timeout with fallback algorithm")
        self.status = SwarmStatus.TIMEOUT

        # Try to use whatever solution we have
        return self._extract_result(plan, maze, fallback=True)

    def _handle_failure(self, plan: Plan, maze: Maze) -> SwarmResult:
        """Handle plan failure."""
        logger.warning("Handling plan failure with fallback")
        self.status = SwarmStatus.FAILED

        return self._extract_result(plan, maze, fallback=True)

    def _extract_result(
        self, plan: Plan, maze: Maze, fallback: bool = False
    ) -> SwarmResult:
        """Extract final result from blackboard and plan."""
        execution_time = time.time() - self.start_time

        # Try to get solution from blackboard
        artifacts = self.blackboard.read("artifacts", "orchestrator") or {}
        metrics = self.blackboard.read("metrics", "orchestrator") or {}

        # Check for executed solution
        solution = artifacts.get("solution")
        if solution and solution.get("success"):
            self.status = SwarmStatus.COMPLETED
            return SwarmResult(
                success=True,
                path=solution.get("path"),
                path_length=solution.get("path_length", 0),
                nodes_explored=solution.get("nodes_explored", 0),
                execution_time=execution_time,
                algorithm_used=solution.get("algorithm", "unknown"),
                rounds=self.current_round,
                total_tokens=self.total_tokens,
                agent_contributions=self.agent_contributions.copy(),
                status=self.status,
            )

        # If no solution and fallback requested, run baseline algorithm
        if fallback:
            return self._run_fallback(maze, execution_time)

        # No solution found
        self.status = SwarmStatus.FAILED
        return SwarmResult(
            success=False,
            path=None,
            path_length=0,
            nodes_explored=0,
            execution_time=execution_time,
            algorithm_used="none",
            rounds=self.current_round,
            total_tokens=self.total_tokens,
            agent_contributions=self.agent_contributions.copy(),
            status=self.status,
            error="No solution found",
        )

    def _run_fallback(self, maze: Maze, elapsed_time: float) -> SwarmResult:
        """Run fallback algorithm directly."""
        from experiments.baseline_algorithms import BFSPathfinder

        logger.info("Running fallback BFS algorithm")

        pathfinder = BFSPathfinder()
        result = pathfinder.find_path(maze)

        total_time = time.time() - self.start_time

        if result.success:
            self.status = SwarmStatus.COMPLETED
            return SwarmResult(
                success=True,
                path=result.path,
                path_length=result.path_length,
                nodes_explored=result.nodes_explored,
                execution_time=total_time,
                algorithm_used="BFS (fallback)",
                rounds=self.current_round,
                total_tokens=self.total_tokens,
                agent_contributions=self.agent_contributions.copy(),
                status=self.status,
            )
        else:
            self.status = SwarmStatus.FAILED
            return SwarmResult(
                success=False,
                path=None,
                path_length=0,
                nodes_explored=result.nodes_explored,
                execution_time=total_time,
                algorithm_used="BFS (fallback)",
                rounds=self.current_round,
                total_tokens=self.total_tokens,
                agent_contributions=self.agent_contributions.copy(),
                status=self.status,
                error="Fallback also failed",
            )

    def _create_failure_result(
        self, error: str, status: SwarmStatus = SwarmStatus.FAILED
    ) -> SwarmResult:
        """Create a failure result."""
        self.status = status
        return SwarmResult(
            success=False,
            path=None,
            path_length=0,
            nodes_explored=0,
            execution_time=time.time() - self.start_time,
            algorithm_used="none",
            rounds=self.current_round,
            total_tokens=self.total_tokens,
            agent_contributions=self.agent_contributions.copy(),
            status=status,
            error=error,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        return {
            "status": self.status.value,
            "current_round": self.current_round,
            "total_tokens": self.total_tokens,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            "agents": list(self.agents.keys()),
            "agent_contributions": self.agent_contributions,
        }

    def reset(self):
        """Reset swarm state for new task."""
        self.status = SwarmStatus.IDLE
        self.current_round = 0
        self.total_tokens = 0
        self.start_time = 0.0

        # Reset agent budgets
        for agent in self.agents.values():
            agent.reset_budget()

        # Clear blackboard
        self.blackboard.clear()

        # Reset state analyzer
        self.state_analyzer.reset()

        # Reset contributions
        for name in self.agent_contributions:
            self.agent_contributions[name] = 0

        logger.info("Swarm reset")
