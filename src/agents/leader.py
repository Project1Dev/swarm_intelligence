"""Leader agent for task decomposition and orchestration."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from agents.base import BaseAgent, AgentConfig, AgentRole, Permission
from communication.message import Message, MessageType, Priority
from utils.maze_generator import Maze


logger = logging.getLogger(__name__)


class SubtaskType(Enum):
    """Types of subtasks the leader can create."""

    ANALYZE_MAZE = "analyze_maze"
    DESIGN_ALGORITHM = "design_algorithm"
    IMPLEMENT_SOLUTION = "implement_solution"
    VALIDATE_SOLUTION = "validate_solution"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    EXECUTE_SOLUTION = "execute_solution"


class SubtaskStatus(Enum):
    """Status of a subtask."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """A subtask in the leader's plan."""

    id: str
    type: SubtaskType
    description: str
    assigned_agent: AgentRole
    priority: Priority
    dependencies: List[str] = field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 2
    timeout: float = 30.0  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "assigned_agent": self.assigned_agent.value,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subtask":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=SubtaskType(data["type"]),
            description=data["description"],
            assigned_agent=AgentRole(data["assigned_agent"]),
            priority=Priority(data["priority"]),
            dependencies=data.get("dependencies", []),
            status=SubtaskStatus(data.get("status", "pending")),
            result=data.get("result"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 2),
            timeout=data.get("timeout", 30.0),
        )


@dataclass
class Plan:
    """A plan created by the leader."""

    id: str
    problem_description: str
    subtasks: List[Subtask]
    created_at: float
    updated_at: float
    status: str = "active"
    analysis: str = ""
    expected_duration: float = 60.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "problem_description": self.problem_description,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "analysis": self.analysis,
            "expected_duration": self.expected_duration,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            problem_description=data["problem_description"],
            subtasks=[Subtask.from_dict(s) for s in data["subtasks"]],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=data.get("status", "active"),
            analysis=data.get("analysis", ""),
            expected_duration=data.get("expected_duration", 60.0),
            metadata=data.get("metadata", {}),
        )

    def get_next_subtask(self) -> Optional[Subtask]:
        """Get the next pending subtask with all dependencies met."""
        completed_ids = {
            s.id for s in self.subtasks if s.status == SubtaskStatus.COMPLETED
        }

        for subtask in self.subtasks:
            if subtask.status == SubtaskStatus.PENDING:
                # Check if all dependencies are completed
                if all(dep in completed_ids for dep in subtask.dependencies):
                    return subtask

        return None

    def get_subtask_by_id(self, subtask_id: str) -> Optional[Subtask]:
        """Get subtask by ID."""
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                return subtask
        return None

    def update_subtask_status(
        self, subtask_id: str, status: SubtaskStatus, result: Optional[Dict] = None
    ) -> bool:
        """Update a subtask's status."""
        subtask = self.get_subtask_by_id(subtask_id)
        if subtask:
            subtask.status = status
            if result:
                subtask.result = result
            self.updated_at = time.time()
            return True
        return False

    def is_complete(self) -> bool:
        """Check if all subtasks are completed or skipped."""
        return all(
            s.status in (SubtaskStatus.COMPLETED, SubtaskStatus.SKIPPED)
            for s in self.subtasks
        )

    def has_failed(self) -> bool:
        """Check if any critical subtask has failed."""
        return any(
            s.status == SubtaskStatus.FAILED and s.priority == Priority.CRITICAL
            for s in self.subtasks
        )


class LeaderAgent(BaseAgent):
    """
    Leader agent for task decomposition and swarm orchestration.

    Responsibilities:
    - Analyze problems and create strategic plans
    - Decompose complex tasks into subtasks
    - Select appropriate specialist agents
    - Monitor progress and adapt strategy
    - Handle failures and replanning
    """

    # System prompt template for the leader
    SYSTEM_PROMPT_TEMPLATE = """You are the Leader Agent in a collaborative swarm intelligence system for maze solving.

Your responsibilities:
1. Analyze the maze problem and create a high-level strategy
2. Decompose complex tasks into smaller subtasks
3. Coordinate specialist agents by assigning appropriate tasks
4. Monitor progress and adjust strategy based on feedback
5. Synthesize results from specialists into coherent solutions

Available specialist agents:
- CODE_SPECIALIST: Implements pathfinding algorithms (BFS, DFS, A*)
- LOGIC_CHECKER: Validates solutions and code for correctness
- RETRIEVAL_SPECIALIST: Searches past solutions and patterns

Communication guidelines:
- Be concise and clear in your instructions
- Provide context when delegating tasks
- Acknowledge and integrate feedback from specialists
- Make decisive choices when multiple approaches are valid

When solving mazes:
- Start by analyzing maze dimensions and structure
- Consider multiple algorithmic approaches (BFS, DFS, A*)
- Request code implementation from Code Specialist
- Request validation from Logic Checker
- Report final solution path and metrics

CRITICAL: Always respond in valid JSON format."""

    def __init__(
        self,
        config: AgentConfig,
        ollama_client: Any,
        blackboard: Optional[Any] = None,
    ):
        """Initialize leader agent."""
        # Ensure leader has full permissions
        if not config.permissions:
            config.permissions = {
                "current_goal": Permission.READ_WRITE,
                "facts": Permission.READ_WRITE,
                "hypotheses": Permission.READ_WRITE,
                "artifacts": Permission.READ,
                "working_memory": Permission.READ_WRITE,
                "metrics": Permission.READ,
            }

        if not config.system_prompt:
            config.system_prompt = self.SYSTEM_PROMPT_TEMPLATE

        super().__init__(config, ollama_client, blackboard)

        self.current_plan: Optional[Plan] = None
        self.plan_history: List[Plan] = []
        self._progress_check_interval = 10.0  # seconds
        self._last_progress_check = 0.0
        self._stall_threshold = 3  # Number of checks without progress

    def decompose_task(self, maze: Maze) -> Plan:
        """
        Decompose a maze-solving task into subtasks.

        Args:
            maze: The maze to solve

        Returns:
            Plan with subtasks
        """
        logger.info(f"Leader: Decomposing task for maze {maze.width}x{maze.height}")

        # Create maze description for the LLM
        maze_description = self._create_maze_description(maze)

        # Try to get LLM-based decomposition
        plan = self._llm_decompose(maze, maze_description)

        if plan is None:
            # Fallback to rule-based decomposition
            logger.warning("Leader: LLM decomposition failed, using rule-based")
            plan = self._rule_based_decompose(maze, maze_description)

        self.current_plan = plan
        self.plan_history.append(plan)

        # Write plan to blackboard
        self.write_blackboard("current_goal", "plan", plan.to_dict())

        logger.info(
            f"Leader: Created plan with {len(plan.subtasks)} subtasks"
        )

        return plan

    def _create_maze_description(self, maze: Maze) -> str:
        """Create a text description of the maze for LLM."""
        total_cells = maze.width * maze.height
        path_count = sum(
            1 for row in maze.grid for cell in row if cell.value == 0
        )
        wall_ratio = 1 - (path_count / total_cells)

        return (
            f"Maze dimensions: {maze.width}x{maze.height}\n"
            f"Total cells: {total_cells}\n"
            f"Path cells: {path_count}\n"
            f"Wall ratio: {wall_ratio:.2%}\n"
            f"Start position: {maze.start}\n"
            f"End position: {maze.end}\n"
            f"Category: {'simple' if total_cells <= 200 else 'medium' if total_cells <= 600 else 'complex' if total_cells <= 1000 else 'stress'}"
        )

    def _llm_decompose(self, maze: Maze, description: str) -> Optional[Plan]:
        """Use LLM to decompose the task."""
        prompt = f"""Analyze this maze problem and create a solving plan.

{description}

Create a plan with the following subtasks:
1. Analyze the maze structure
2. Select the best algorithm based on maze properties
3. Implement the pathfinding solution
4. Validate the solution is correct

For each subtask, specify:
- Type (analyze_maze, design_algorithm, implement_solution, validate_solution)
- Assigned agent (code_specialist, logic_checker, or retrieval_specialist)
- Priority (0=critical, 1=high, 2=normal, 3=low)

Respond in JSON format:
{{
  "analysis": "Brief analysis of the maze",
  "recommended_algorithm": "BFS|DFS|A*",
  "reasoning": "Why this algorithm",
  "subtasks": [
    {{
      "type": "analyze_maze",
      "description": "Analyze maze structure",
      "assigned_agent": "logic_checker",
      "priority": 1
    }},
    ...
  ],
  "expected_duration": 30
}}"""

        response, tokens = self.generate(prompt, max_tokens=800)

        if not response:
            return None

        try:
            # Try to parse JSON from response
            data = self._parse_json_response(response)
            if data is None:
                return None

            # Create plan from LLM response
            plan_id = f"plan_{int(time.time())}"
            now = time.time()

            subtasks = []
            for i, st_data in enumerate(data.get("subtasks", [])):
                subtask = Subtask(
                    id=f"{plan_id}_task_{i}",
                    type=SubtaskType(st_data["type"]),
                    description=st_data["description"],
                    assigned_agent=AgentRole(st_data["assigned_agent"]),
                    priority=Priority(st_data.get("priority", 2)),
                    dependencies=[f"{plan_id}_task_{j}" for j in range(i)],
                )
                subtasks.append(subtask)

            return Plan(
                id=plan_id,
                problem_description=description,
                subtasks=subtasks,
                created_at=now,
                updated_at=now,
                analysis=data.get("analysis", ""),
                expected_duration=data.get("expected_duration", 60.0),
                metadata={
                    "recommended_algorithm": data.get("recommended_algorithm"),
                    "reasoning": data.get("reasoning"),
                },
            )

        except Exception as e:
            logger.error(f"Leader: Failed to parse LLM response: {e}")
            return None

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON-like structure
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _rule_based_decompose(self, maze: Maze, description: str) -> Plan:
        """Create a rule-based plan when LLM fails."""
        plan_id = f"plan_{int(time.time())}"
        now = time.time()

        # Determine algorithm based on maze size
        total_cells = maze.width * maze.height
        if total_cells <= 200:
            algorithm = "BFS"
            reasoning = "Small maze, BFS guarantees shortest path"
        elif total_cells <= 1000:
            algorithm = "A*"
            reasoning = "Medium maze, A* with heuristic is efficient"
        else:
            algorithm = "A*"
            reasoning = "Large maze, A* with heuristic minimizes exploration"

        subtasks = [
            Subtask(
                id=f"{plan_id}_task_0",
                type=SubtaskType.ANALYZE_MAZE,
                description="Analyze maze structure and identify key properties",
                assigned_agent=AgentRole.LOGIC_CHECKER,
                priority=Priority.HIGH,
                dependencies=[],
            ),
            Subtask(
                id=f"{plan_id}_task_1",
                type=SubtaskType.RETRIEVE_KNOWLEDGE,
                description="Search for similar mazes and successful solutions",
                assigned_agent=AgentRole.RETRIEVAL_SPECIALIST,
                priority=Priority.NORMAL,
                dependencies=[],  # Can run in parallel with analysis
            ),
            Subtask(
                id=f"{plan_id}_task_2",
                type=SubtaskType.IMPLEMENT_SOLUTION,
                description=f"Implement {algorithm} pathfinding algorithm",
                assigned_agent=AgentRole.CODE_SPECIALIST,
                priority=Priority.HIGH,
                dependencies=[f"{plan_id}_task_0"],
            ),
            Subtask(
                id=f"{plan_id}_task_3",
                type=SubtaskType.VALIDATE_SOLUTION,
                description="Validate the implemented solution for correctness",
                assigned_agent=AgentRole.LOGIC_CHECKER,
                priority=Priority.CRITICAL,
                dependencies=[f"{plan_id}_task_2"],
            ),
            Subtask(
                id=f"{plan_id}_task_4",
                type=SubtaskType.EXECUTE_SOLUTION,
                description="Execute validated solution and record metrics",
                assigned_agent=AgentRole.LEADER,
                priority=Priority.CRITICAL,
                dependencies=[f"{plan_id}_task_3"],
            ),
        ]

        return Plan(
            id=plan_id,
            problem_description=description,
            subtasks=subtasks,
            created_at=now,
            updated_at=now,
            analysis=f"Rule-based decomposition for {maze.width}x{maze.height} maze",
            expected_duration=30.0 + (total_cells / 100),
            metadata={
                "recommended_algorithm": algorithm,
                "reasoning": reasoning,
            },
        )

    def select_agent(
        self, subtask: Subtask, blackboard_state: Dict[str, Any]
    ) -> AgentRole:
        """
        Select the best agent for a subtask based on blackboard state.

        Args:
            subtask: The subtask to assign
            blackboard_state: Current blackboard state

        Returns:
            Selected agent role
        """
        # Start with the pre-assigned agent
        selected = subtask.assigned_agent

        # Check blackboard state for dynamic selection
        facts = blackboard_state.get("facts", {})
        hypotheses = blackboard_state.get("hypotheses", {})

        # Dynamic selection rules
        if subtask.type == SubtaskType.IMPLEMENT_SOLUTION:
            # If we have validation errors, might want to reassign
            errors = facts.get("validation_errors", [])
            if errors and subtask.retry_count > 0:
                # Logic checker found errors, code specialist should fix
                selected = AgentRole.CODE_SPECIALIST

        elif subtask.type == SubtaskType.VALIDATE_SOLUTION:
            # Always use logic checker for validation
            selected = AgentRole.LOGIC_CHECKER

        elif subtask.type == SubtaskType.RETRIEVE_KNOWLEDGE:
            # Check if similar problem was already retrieved
            similar = facts.get("similar_solutions", [])
            if similar:
                # Skip retrieval if we already have good matches
                logger.info("Leader: Skipping retrieval, similar solutions found")
                # Mark as skipped in a real implementation

        logger.debug(
            f"Leader: Selected {selected.value} for subtask {subtask.id}"
        )

        return selected

    def monitor_progress(self) -> Tuple[bool, str]:
        """
        Monitor swarm progress and detect stalls.

        Returns:
            Tuple of (is_progressing, status_message)
        """
        if self.current_plan is None:
            return False, "No active plan"

        now = time.time()

        # Count completed tasks
        completed = sum(
            1 for s in self.current_plan.subtasks
            if s.status == SubtaskStatus.COMPLETED
        )
        in_progress = sum(
            1 for s in self.current_plan.subtasks
            if s.status == SubtaskStatus.IN_PROGRESS
        )
        failed = sum(
            1 for s in self.current_plan.subtasks
            if s.status == SubtaskStatus.FAILED
        )
        total = len(self.current_plan.subtasks)

        # Check for completion
        if self.current_plan.is_complete():
            return True, f"Plan complete: {completed}/{total} tasks done"

        # Check for critical failures
        if self.current_plan.has_failed():
            return False, f"Critical task failed: {failed} failures"

        # Check progress since last check
        elapsed = now - self._last_progress_check

        if elapsed > self._progress_check_interval:
            self._last_progress_check = now

            # Check if we're making progress
            progress_key = f"progress_{int(now / self._progress_check_interval)}"
            current_progress = completed + (in_progress * 0.5)

            # Store for stall detection
            self.state.metadata[progress_key] = current_progress

        status = (
            f"Progress: {completed}/{total} complete, "
            f"{in_progress} in progress, {failed} failed"
        )

        return True, status

    def replan(self, failure_info: Dict[str, Any]) -> Optional[Plan]:
        """
        Create a new plan after failure.

        Args:
            failure_info: Information about the failure

        Returns:
            New plan or None if recovery not possible
        """
        if self.current_plan is None:
            logger.error("Leader: Cannot replan without current plan")
            return None

        failed_subtask_id = failure_info.get("subtask_id")
        error_message = failure_info.get("error", "Unknown error")
        retry_count = failure_info.get("retry_count", 0)

        logger.warning(
            f"Leader: Replanning after failure in {failed_subtask_id}: {error_message}"
        )

        # Find the failed subtask
        failed_subtask = self.current_plan.get_subtask_by_id(failed_subtask_id)

        if failed_subtask is None:
            return None

        # Strategy 1: Retry with different agent or approach
        if retry_count < failed_subtask.max_retries:
            failed_subtask.retry_count += 1
            failed_subtask.status = SubtaskStatus.PENDING

            # Modify the subtask for retry
            if failed_subtask.type == SubtaskType.IMPLEMENT_SOLUTION:
                # Try a different algorithm
                current_algo = self.current_plan.metadata.get("recommended_algorithm")
                alternatives = ["BFS", "DFS", "A*"]
                alternatives.remove(current_algo) if current_algo in alternatives else None

                if alternatives:
                    new_algo = alternatives[0]
                    failed_subtask.description = (
                        f"Retry: Implement {new_algo} pathfinding algorithm"
                    )
                    self.current_plan.metadata["recommended_algorithm"] = new_algo
                    logger.info(f"Leader: Retrying with {new_algo}")

            self.current_plan.updated_at = time.time()
            return self.current_plan

        # Strategy 2: Skip non-critical task
        if failed_subtask.priority not in (Priority.CRITICAL, Priority.HIGH):
            failed_subtask.status = SubtaskStatus.SKIPPED
            logger.info(f"Leader: Skipping non-critical task {failed_subtask_id}")
            return self.current_plan

        # Strategy 3: Create simplified fallback plan
        logger.warning("Leader: Creating fallback plan")
        return self._create_fallback_plan()

    def _create_fallback_plan(self) -> Plan:
        """Create a minimal fallback plan using default algorithm."""
        plan_id = f"fallback_{int(time.time())}"
        now = time.time()

        subtasks = [
            Subtask(
                id=f"{plan_id}_task_0",
                type=SubtaskType.IMPLEMENT_SOLUTION,
                description="Execute BFS pathfinding (guaranteed to work)",
                assigned_agent=AgentRole.CODE_SPECIALIST,
                priority=Priority.CRITICAL,
                dependencies=[],
            ),
            Subtask(
                id=f"{plan_id}_task_1",
                type=SubtaskType.EXECUTE_SOLUTION,
                description="Execute BFS and return result",
                assigned_agent=AgentRole.LEADER,
                priority=Priority.CRITICAL,
                dependencies=[f"{plan_id}_task_0"],
            ),
        ]

        return Plan(
            id=plan_id,
            problem_description="Fallback plan using guaranteed BFS",
            subtasks=subtasks,
            created_at=now,
            updated_at=now,
            analysis="Fallback after repeated failures",
            expected_duration=30.0,
            metadata={"recommended_algorithm": "BFS", "is_fallback": True},
        )

    def process_task(self, task: Any, context: Dict[str, Any]) -> Optional[Message]:
        """
        Process a task assigned to the leader.

        The leader handles high-level tasks like:
        - Initial problem analysis
        - Plan creation
        - Progress monitoring
        - Final result synthesis
        """
        task_type = task.get("type") if isinstance(task, dict) else str(task)

        if task_type == "analyze_problem":
            return self._handle_analyze_problem(task, context)
        elif task_type == "create_plan":
            return self._handle_create_plan(task, context)
        elif task_type == "monitor_progress":
            return self._handle_monitor_progress(context)
        elif task_type == "synthesize_result":
            return self._handle_synthesize_result(context)
        else:
            logger.warning(f"Leader: Unknown task type: {task_type}")
            return None

    def _handle_analyze_problem(
        self, task: Dict, context: Dict[str, Any]
    ) -> Message:
        """Handle problem analysis request."""
        maze = context.get("maze")

        if maze is None:
            return Message(
                from_agent=self.name,
                to_agent="broadcast",
                type=MessageType.ERROR,
                priority=Priority.HIGH,
                content={"error": "No maze provided for analysis"},
            )

        # Analyze maze
        analysis = self._create_maze_description(maze)

        # Write analysis to blackboard
        self.write_blackboard("facts", "maze_analysis", analysis)

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.STATUS_UPDATE,
            priority=Priority.NORMAL,
            content={
                "task": "analyze_problem",
                "status": "completed",
                "analysis": analysis,
            },
        )

    def _handle_create_plan(
        self, task: Dict, context: Dict[str, Any]
    ) -> Message:
        """Handle plan creation request."""
        maze = context.get("maze")

        if maze is None:
            return Message(
                from_agent=self.name,
                to_agent="broadcast",
                type=MessageType.ERROR,
                priority=Priority.HIGH,
                content={"error": "No maze provided for planning"},
            )

        plan = self.decompose_task(maze)

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.PLAN_UPDATE,
            priority=Priority.HIGH,
            content={
                "plan_id": plan.id,
                "subtask_count": len(plan.subtasks),
                "expected_duration": plan.expected_duration,
                "recommended_algorithm": plan.metadata.get("recommended_algorithm"),
            },
        )

    def _handle_monitor_progress(self, context: Dict[str, Any]) -> Message:
        """Handle progress monitoring."""
        is_progressing, status = self.monitor_progress()

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.STATUS_UPDATE,
            priority=Priority.LOW,
            content={
                "task": "monitor_progress",
                "is_progressing": is_progressing,
                "status": status,
            },
        )

    def _handle_synthesize_result(self, context: Dict[str, Any]) -> Message:
        """Handle result synthesis."""
        # Read results from blackboard
        bb_data = self.read_blackboard(["artifacts", "facts", "metrics"])

        solution = bb_data.get("artifacts", {}).get("solution")
        metrics = bb_data.get("metrics", {})

        if solution:
            return Message(
                from_agent=self.name,
                to_agent="broadcast",
                type=MessageType.TASK_RESULT,
                priority=Priority.CRITICAL,
                content={
                    "task": "synthesize_result",
                    "success": True,
                    "solution": solution,
                    "metrics": metrics,
                },
            )
        else:
            return Message(
                from_agent=self.name,
                to_agent="broadcast",
                type=MessageType.TASK_FAILED,
                priority=Priority.CRITICAL,
                content={
                    "task": "synthesize_result",
                    "success": False,
                    "reason": "No solution found in artifacts",
                },
            )

    def validate_output(self, output: Any) -> Tuple[bool, str]:
        """Validate leader output."""
        if output is None:
            return False, "Output is None"

        if isinstance(output, Message):
            # Check message has required fields
            if output.content is None:
                return False, "Message content is None"
            return True, "Valid message"

        if isinstance(output, Plan):
            # Check plan has subtasks
            if not output.subtasks:
                return False, "Plan has no subtasks"
            return True, "Valid plan"

        return False, f"Unknown output type: {type(output)}"

    def get_plan_summary(self) -> Dict[str, Any]:
        """Get summary of current plan status."""
        if self.current_plan is None:
            return {"status": "no_plan"}

        completed = sum(
            1 for s in self.current_plan.subtasks
            if s.status == SubtaskStatus.COMPLETED
        )
        failed = sum(
            1 for s in self.current_plan.subtasks
            if s.status == SubtaskStatus.FAILED
        )
        in_progress = sum(
            1 for s in self.current_plan.subtasks
            if s.status == SubtaskStatus.IN_PROGRESS
        )

        return {
            "plan_id": self.current_plan.id,
            "status": self.current_plan.status,
            "total_subtasks": len(self.current_plan.subtasks),
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "is_complete": self.current_plan.is_complete(),
            "has_failed": self.current_plan.has_failed(),
            "recommended_algorithm": self.current_plan.metadata.get(
                "recommended_algorithm"
            ),
        }
