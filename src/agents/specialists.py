"""Specialist agents for the swarm intelligence system."""

import json
import logging
import ast
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from agents.base import BaseAgent, AgentConfig, AgentRole, Permission
from communication.message import Message, MessageType, Priority
from utils.maze_generator import Maze


logger = logging.getLogger(__name__)


# ==============================================================================
# Code Specialist Agent
# ==============================================================================

@dataclass
class CodeGenerationResult:
    """Result of code generation."""

    code: str
    algorithm: str
    explanation: str
    complexity_time: str
    complexity_space: str
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "algorithm": self.algorithm,
            "explanation": self.explanation,
            "complexity_time": self.complexity_time,
            "complexity_space": self.complexity_space,
            "success": self.success,
            "error": self.error,
        }


class CodeSpecialist(BaseAgent):
    """
    Code Specialist Agent for algorithm implementation.

    Responsibilities:
    - Implement pathfinding algorithms (BFS, DFS, A*)
    - Write clean, efficient, well-documented Python code
    - Optimize code for performance when requested
    - Debug and fix issues in existing implementations
    """

    SYSTEM_PROMPT_TEMPLATE = """You are the Code Specialist Agent in a collaborative swarm intelligence system.

Your responsibilities:
1. Implement pathfinding algorithms (BFS, DFS, A*, custom heuristics)
2. Write clean, efficient, well-documented Python code
3. Optimize code for performance when requested
4. Debug and fix issues in existing implementations
5. Explain code logic when asked

Code guidelines:
- Use type hints for all function parameters and returns
- Include docstrings with clear descriptions
- Handle edge cases (empty maze, no path, etc.)
- Return structured results with metrics

When implementing maze solvers:
- Use standard data structures (deque for BFS, list for DFS, heapq for A*)
- Track nodes explored and execution time
- Return both the path and solving statistics
- Validate inputs before processing

CRITICAL: Always respond with valid Python code in a markdown code block."""

    # Pre-defined algorithm templates for reliability
    ALGORITHM_TEMPLATES = {
        "BFS": '''
def solve_maze_bfs(maze: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> Dict[str, Any]:
    """
    Solve maze using Breadth-First Search.

    BFS guarantees the shortest path in terms of number of steps.
    Time: O(V + E) where V is vertices, E is edges
    Space: O(V)

    Args:
        maze: 2D grid where 0 is path, 1 is wall
        start: Starting position (row, col)
        end: End position (row, col)

    Returns:
        Dictionary with path, nodes_explored, and success status
    """
    from collections import deque
    import time

    start_time = time.time()
    rows, cols = len(maze), len(maze[0])

    # Validate inputs
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return {"success": False, "error": "Invalid start position", "path": [], "nodes_explored": 0}
    if not (0 <= end[0] < rows and 0 <= end[1] < cols):
        return {"success": False, "error": "Invalid end position", "path": [], "nodes_explored": 0}
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:
        return {"success": False, "error": "Start or end is a wall", "path": [], "nodes_explored": 0}

    queue = deque([start])
    visited = {start}
    came_from = {}
    nodes_explored = 0

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if current == end:
            # Reconstruct path
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            return {
                "success": True,
                "path": path,
                "path_length": len(path),
                "nodes_explored": nodes_explored,
                "execution_time": time.time() - start_time,
                "algorithm": "BFS",
                "optimal": True
            }

        row, col = current
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)

            if (0 <= nr < rows and 0 <= nc < cols and
                neighbor not in visited and maze[nr][nc] == 0):
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return {
        "success": False,
        "error": "No path found",
        "path": [],
        "nodes_explored": nodes_explored,
        "execution_time": time.time() - start_time
    }
''',
        "DFS": '''
def solve_maze_dfs(maze: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> Dict[str, Any]:
    """
    Solve maze using Depth-First Search.

    DFS does NOT guarantee the shortest path.
    Time: O(V + E)
    Space: O(V)

    Args:
        maze: 2D grid where 0 is path, 1 is wall
        start: Starting position (row, col)
        end: End position (row, col)

    Returns:
        Dictionary with path, nodes_explored, and success status
    """
    import time

    start_time = time.time()
    rows, cols = len(maze), len(maze[0])

    # Validate inputs
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return {"success": False, "error": "Invalid start position", "path": [], "nodes_explored": 0}
    if not (0 <= end[0] < rows and 0 <= end[1] < cols):
        return {"success": False, "error": "Invalid end position", "path": [], "nodes_explored": 0}
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:
        return {"success": False, "error": "Start or end is a wall", "path": [], "nodes_explored": 0}

    stack = [start]
    visited = {start}
    came_from = {}
    nodes_explored = 0

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while stack:
        current = stack.pop()
        nodes_explored += 1

        if current == end:
            # Reconstruct path
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            return {
                "success": True,
                "path": path,
                "path_length": len(path),
                "nodes_explored": nodes_explored,
                "execution_time": time.time() - start_time,
                "algorithm": "DFS",
                "optimal": False
            }

        row, col = current
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)

            if (0 <= nr < rows and 0 <= nc < cols and
                neighbor not in visited and maze[nr][nc] == 0):
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

    return {
        "success": False,
        "error": "No path found",
        "path": [],
        "nodes_explored": nodes_explored,
        "execution_time": time.time() - start_time
    }
''',
        "A*": '''
def solve_maze_astar(maze: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> Dict[str, Any]:
    """
    Solve maze using A* algorithm with Manhattan distance heuristic.

    A* guarantees the shortest path with admissible heuristic.
    Time: O(E) with good heuristic
    Space: O(V)

    Args:
        maze: 2D grid where 0 is path, 1 is wall
        start: Starting position (row, col)
        end: End position (row, col)

    Returns:
        Dictionary with path, nodes_explored, and success status
    """
    import heapq
    import time

    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    start_time = time.time()
    rows, cols = len(maze), len(maze[0])

    # Validate inputs
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return {"success": False, "error": "Invalid start position", "path": [], "nodes_explored": 0}
    if not (0 <= end[0] < rows and 0 <= end[1] < cols):
        return {"success": False, "error": "Invalid end position", "path": [], "nodes_explored": 0}
    if maze[start[0]][start[1]] == 1 or maze[end[0]][end[1]] == 1:
        return {"success": False, "error": "Start or end is a wall", "path": [], "nodes_explored": 0}

    # Priority queue: (f_score, counter, position)
    counter = 0
    open_set = [(manhattan_distance(start, end), counter, start)]
    came_from = {}

    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, end)}
    open_set_hash = {start}
    nodes_explored = 0

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while open_set:
        _, _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)
        nodes_explored += 1

        if current == end:
            # Reconstruct path
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            return {
                "success": True,
                "path": path,
                "path_length": len(path),
                "nodes_explored": nodes_explored,
                "execution_time": time.time() - start_time,
                "algorithm": "A*",
                "optimal": True
            }

        row, col = current
        current_g = g_score[current]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if maze[nr][nc] == 1:
                continue

            tentative_g = current_g + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan_distance(neighbor, end)
                f_score[neighbor] = f

                if neighbor not in open_set_hash:
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
                    open_set_hash.add(neighbor)

    return {
        "success": False,
        "error": "No path found",
        "path": [],
        "nodes_explored": nodes_explored,
        "execution_time": time.time() - start_time
    }
'''
    }

    def __init__(
        self,
        config: AgentConfig,
        ollama_client: Any,
        blackboard: Optional[Any] = None,
    ):
        """Initialize code specialist agent."""
        # Ensure code specialist has correct permissions
        if not config.permissions:
            config.permissions = {
                "current_goal": Permission.READ,
                "facts": Permission.READ,
                "hypotheses": Permission.READ,
                "artifacts": Permission.READ_WRITE,
                "working_memory": Permission.READ_WRITE,
                "metrics": Permission.READ,
            }

        if not config.system_prompt:
            config.system_prompt = self.SYSTEM_PROMPT_TEMPLATE

        super().__init__(config, ollama_client, blackboard)

    def generate_algorithm(
        self, algorithm: str, maze_info: Optional[Dict] = None
    ) -> CodeGenerationResult:
        """
        Generate pathfinding algorithm code.

        Args:
            algorithm: Algorithm name (BFS, DFS, A*)
            maze_info: Optional maze information for context

        Returns:
            CodeGenerationResult with generated code
        """
        algorithm = algorithm.upper()

        # Use template if available (more reliable)
        if algorithm in self.ALGORITHM_TEMPLATES:
            logger.info(f"CodeSpecialist: Using template for {algorithm}")
            return CodeGenerationResult(
                code=self.ALGORITHM_TEMPLATES[algorithm].strip(),
                algorithm=algorithm,
                explanation=f"Standard {algorithm} implementation for maze solving",
                complexity_time="O(V + E)" if algorithm in ["BFS", "DFS"] else "O(E)",
                complexity_space="O(V)",
            )

        # Try LLM generation for custom algorithms
        return self._llm_generate_algorithm(algorithm, maze_info)

    def _llm_generate_algorithm(
        self, algorithm: str, maze_info: Optional[Dict] = None
    ) -> CodeGenerationResult:
        """Generate algorithm using LLM."""
        context = ""
        if maze_info:
            context = f"\nMaze info: {maze_info}"

        prompt = f"""Generate a Python implementation of {algorithm} pathfinding algorithm.
{context}

Requirements:
1. Function signature: solve_maze(maze, start, end) -> Dict
2. maze is List[List[int]] where 0 is path, 1 is wall
3. start and end are Tuple[int, int] (row, col)
4. Return dict with: success, path, path_length, nodes_explored, execution_time, algorithm, optimal
5. Include input validation
6. Use type hints and docstrings

Provide the code in a Python code block."""

        response, tokens = self.generate(prompt, max_tokens=1500)

        if not response:
            return CodeGenerationResult(
                code="",
                algorithm=algorithm,
                explanation="LLM generation failed",
                complexity_time="Unknown",
                complexity_space="Unknown",
                success=False,
                error="No response from LLM",
            )

        # Extract code from response
        code = self._extract_code(response)

        if not code:
            return CodeGenerationResult(
                code="",
                algorithm=algorithm,
                explanation="Failed to extract code from response",
                complexity_time="Unknown",
                complexity_space="Unknown",
                success=False,
                error="Code extraction failed",
            )

        # Validate the code
        is_valid, error = self._validate_syntax(code)

        if not is_valid:
            return CodeGenerationResult(
                code=code,
                algorithm=algorithm,
                explanation="Generated code has syntax errors",
                complexity_time="Unknown",
                complexity_space="Unknown",
                success=False,
                error=error,
            )

        return CodeGenerationResult(
            code=code,
            algorithm=algorithm,
            explanation=f"LLM-generated {algorithm} implementation",
            complexity_time="See code comments",
            complexity_space="See code comments",
        )

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        import re

        # Try to find code in markdown code block
        code_match = re.search(r'```(?:python)?\s*([\s\S]*?)```', response)
        if code_match:
            return code_match.group(1).strip()

        # Try to find function definition
        func_match = re.search(r'(def\s+solve_maze[\s\S]*)', response)
        if func_match:
            return func_match.group(1).strip()

        return ""

    def _validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def optimize_code(
        self, code: str, optimization_type: str = "speed"
    ) -> CodeGenerationResult:
        """
        Optimize existing code.

        Args:
            code: Code to optimize
            optimization_type: Type of optimization (speed, memory)

        Returns:
            CodeGenerationResult with optimized code
        """
        prompt = f"""Optimize this Python code for {optimization_type}:

```python
{code}
```

Provide optimized code with explanation of changes."""

        response, tokens = self.generate(prompt, max_tokens=1500)

        if not response:
            return CodeGenerationResult(
                code=code,
                algorithm="optimized",
                explanation="Optimization failed",
                complexity_time="Unknown",
                complexity_space="Unknown",
                success=False,
                error="No response from LLM",
            )

        optimized_code = self._extract_code(response)

        if not optimized_code:
            return CodeGenerationResult(
                code=code,
                algorithm="optimized",
                explanation="Failed to extract optimized code",
                complexity_time="Unknown",
                complexity_space="Unknown",
                success=False,
                error="Code extraction failed",
            )

        return CodeGenerationResult(
            code=optimized_code,
            algorithm="optimized",
            explanation=f"Code optimized for {optimization_type}",
            complexity_time="Improved",
            complexity_space="Improved",
        )

    def process_task(self, task: Any, context: Dict[str, Any]) -> Optional[Message]:
        """Process a task assigned to code specialist."""
        task_type = task.get("type") if isinstance(task, dict) else str(task)

        if task_type == "generate_algorithm":
            return self._handle_generate_algorithm(task, context)
        elif task_type == "optimize_code":
            return self._handle_optimize_code(task, context)
        elif task_type == "debug_code":
            return self._handle_debug_code(task, context)
        else:
            logger.warning(f"CodeSpecialist: Unknown task type: {task_type}")
            return None

    def _handle_generate_algorithm(
        self, task: Dict, context: Dict[str, Any]
    ) -> Message:
        """Handle algorithm generation task."""
        algorithm = task.get("algorithm", "BFS")
        maze_info = context.get("maze_info")

        result = self.generate_algorithm(algorithm, maze_info)

        # Write to blackboard
        self.write_blackboard("artifacts", f"code_{algorithm}", result.to_dict())

        self.state.tasks_completed += 1

        if result.success:
            return Message(
                from_agent=self.name,
                to_agent="broadcast",
                type=MessageType.CODE_DRAFT,
                priority=Priority.NORMAL,
                content={
                    "task": "generate_algorithm",
                    "algorithm": algorithm,
                    "success": True,
                    "code_length": len(result.code),
                },
            )
        else:
            self.state.tasks_failed += 1
            return Message(
                from_agent=self.name,
                to_agent="broadcast",
                type=MessageType.TASK_FAILED,
                priority=Priority.HIGH,
                content={
                    "task": "generate_algorithm",
                    "algorithm": algorithm,
                    "success": False,
                    "error": result.error,
                },
            )

    def _handle_optimize_code(self, task: Dict, context: Dict[str, Any]) -> Message:
        """Handle code optimization task."""
        code = task.get("code", "")
        optimization_type = task.get("optimization_type", "speed")

        result = self.optimize_code(code, optimization_type)

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.CODE_DRAFT,
            priority=Priority.NORMAL,
            content={
                "task": "optimize_code",
                "success": result.success,
                "optimization_type": optimization_type,
            },
        )

    def _handle_debug_code(self, task: Dict, context: Dict[str, Any]) -> Message:
        """Handle code debugging task."""
        code = task.get("code", "")
        error_info = task.get("error", "")

        prompt = f"""Debug this Python code:

```python
{code}
```

Error: {error_info}

Provide the fixed code."""

        response, tokens = self.generate(prompt, max_tokens=1500)
        fixed_code = self._extract_code(response) if response else ""

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.CODE_DRAFT,
            priority=Priority.NORMAL,
            content={
                "task": "debug_code",
                "success": bool(fixed_code),
                "has_fix": bool(fixed_code),
            },
        )

    def validate_output(self, output: Any) -> Tuple[bool, str]:
        """Validate code specialist output."""
        if output is None:
            return False, "Output is None"

        if isinstance(output, CodeGenerationResult):
            if not output.success:
                return False, f"Code generation failed: {output.error}"
            if not output.code:
                return False, "No code generated"
            return True, "Valid code generation result"

        if isinstance(output, Message):
            if output.content is None:
                return False, "Message content is None"
            return True, "Valid message"

        return False, f"Unknown output type: {type(output)}"


# ==============================================================================
# Logic Checker Agent
# ==============================================================================

@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "details": self.details,
        }


class LogicChecker(BaseAgent):
    """
    Logic Checker Agent for validation and verification.

    Responsibilities:
    - Validate proposed solutions for correctness
    - Verify that paths are valid (continuous, no wall crossings)
    - Check code implementations for logical errors
    - Identify edge cases and potential bugs
    - Assess solution optimality when applicable
    """

    SYSTEM_PROMPT_TEMPLATE = """You are the Logic Checker Agent in a collaborative swarm intelligence system.

Your responsibilities:
1. Validate proposed solutions for correctness
2. Verify that paths are valid (continuous, no wall crossings)
3. Check code implementations for logical errors
4. Identify edge cases and potential bugs
5. Assess solution optimality when applicable

Validation guidelines:
- Be thorough and systematic in your checks
- Provide specific feedback on any issues found
- Suggest fixes when problems are identified
- Confirm when solutions are verified correct

When validating maze solutions:
- Check that path starts at start position and ends at goal
- Verify each step is to an adjacent, walkable cell
- Confirm no diagonal moves (if not allowed)
- Compare path length to known optimal if available
- Report validation result with detailed reasoning

CRITICAL: Always respond in valid JSON format with 'valid', 'issues', and 'suggestions' fields."""

    def __init__(
        self,
        config: AgentConfig,
        ollama_client: Any,
        blackboard: Optional[Any] = None,
    ):
        """Initialize logic checker agent."""
        if not config.permissions:
            config.permissions = {
                "current_goal": Permission.READ,
                "facts": Permission.READ_WRITE,
                "hypotheses": Permission.READ_WRITE,
                "artifacts": Permission.READ,
                "working_memory": Permission.READ,
                "metrics": Permission.READ_WRITE,
            }

        if not config.system_prompt:
            config.system_prompt = self.SYSTEM_PROMPT_TEMPLATE

        super().__init__(config, ollama_client, blackboard)

    def validate_path(
        self,
        maze: List[List[int]],
        path: List[Tuple[int, int]],
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> ValidationResult:
        """
        Validate a path through the maze.

        Args:
            maze: 2D grid where 0 is path, 1 is wall
            path: List of positions forming the path
            start: Expected start position
            end: Expected end position

        Returns:
            ValidationResult with validation details
        """
        issues = []
        suggestions = []

        # Check empty path
        if not path:
            return ValidationResult(
                valid=False,
                issues=[{"severity": "critical", "description": "Path is empty"}],
                confidence=1.0,
            )

        # Check start position
        if path[0] != start:
            issues.append({
                "severity": "critical",
                "description": f"Path does not start at start position. Expected {start}, got {path[0]}",
            })

        # Check end position
        if path[-1] != end:
            issues.append({
                "severity": "critical",
                "description": f"Path does not end at end position. Expected {end}, got {path[-1]}",
            })

        rows = len(maze)
        cols = len(maze[0]) if rows > 0 else 0

        # Check each position
        for i, pos in enumerate(path):
            row, col = pos

            # Check bounds
            if not (0 <= row < rows and 0 <= col < cols):
                issues.append({
                    "severity": "critical",
                    "description": f"Position {pos} at index {i} is out of bounds",
                })
                continue

            # Check not a wall
            if maze[row][col] == 1:
                issues.append({
                    "severity": "critical",
                    "description": f"Position {pos} at index {i} is a wall",
                })

        # Check continuity (each step is adjacent)
        for i in range(1, len(path)):
            prev = path[i - 1]
            curr = path[i]

            # Check Manhattan distance is exactly 1
            distance = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
            if distance != 1:
                issues.append({
                    "severity": "critical",
                    "description": f"Non-adjacent step from {prev} to {curr} (distance {distance})",
                })

            # Check no diagonal moves
            if curr[0] != prev[0] and curr[1] != prev[1]:
                issues.append({
                    "severity": "warning",
                    "description": f"Diagonal move detected from {prev} to {curr}",
                })

        # Generate suggestions
        if issues:
            suggestions.append("Review path generation algorithm for correctness")
            if any(i["severity"] == "critical" for i in issues):
                suggestions.append("Critical issues found - path is invalid")

        return ValidationResult(
            valid=len([i for i in issues if i["severity"] == "critical"]) == 0,
            issues=issues,
            suggestions=suggestions,
            confidence=1.0,
            details={
                "path_length": len(path),
                "start_correct": path[0] == start if path else False,
                "end_correct": path[-1] == end if path else False,
            },
        )

    def validate_code(self, code: str) -> ValidationResult:
        """
        Validate Python code for potential issues.

        Args:
            code: Python code to validate

        Returns:
            ValidationResult with validation details
        """
        issues = []
        suggestions = []

        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                issues=[{
                    "severity": "critical",
                    "description": f"Syntax error: {e.msg}",
                    "line": e.lineno,
                }],
                confidence=1.0,
            )

        # Check for common issues
        code_lower = code.lower()

        # Infinite loop risks
        if "while true" in code_lower and "break" not in code_lower:
            issues.append({
                "severity": "warning",
                "description": "Potential infinite loop: 'while True' without break",
            })
            suggestions.append("Add a break condition or counter to prevent infinite loops")

        # Missing visited set for graph traversal
        if ("bfs" in code_lower or "dfs" in code_lower) and "visited" not in code_lower:
            issues.append({
                "severity": "warning",
                "description": "Graph traversal without visited set may cause infinite loops",
            })
            suggestions.append("Add a visited set to track explored nodes")

        # Bounds checking
        if "[" in code and "len(" not in code and "range(" not in code:
            issues.append({
                "severity": "warning",
                "description": "Array access without explicit bounds checking",
            })
            suggestions.append("Add bounds checking before array access")

        # Try LLM validation for deeper analysis
        llm_result = self._llm_validate_code(code)
        if llm_result.issues:
            issues.extend(llm_result.issues)
        if llm_result.suggestions:
            suggestions.extend(llm_result.suggestions)

        return ValidationResult(
            valid=len([i for i in issues if i["severity"] == "critical"]) == 0,
            issues=issues,
            suggestions=suggestions,
            confidence=0.9 if issues else 1.0,
        )

    def _llm_validate_code(self, code: str) -> ValidationResult:
        """Use LLM for deeper code validation."""
        prompt = f"""Analyze this Python pathfinding code for potential issues:

```python
{code}
```

Check for:
1. Infinite loop risks
2. Boundary check issues
3. Logic errors
4. Edge case handling
5. Performance issues

Respond in JSON:
{{"valid": true/false, "issues": [{{"severity": "critical/warning", "description": "..."}}], "suggestions": ["..."]}}"""

        response, tokens = self.generate(prompt, max_tokens=500)

        if not response:
            return ValidationResult(valid=True, confidence=0.5)

        try:
            data = self._parse_json_response(response)
            if data:
                return ValidationResult(
                    valid=data.get("valid", True),
                    issues=data.get("issues", []),
                    suggestions=data.get("suggestions", []),
                    confidence=0.8,
                )
        except Exception:
            pass

        return ValidationResult(valid=True, confidence=0.5)

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        import re

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def process_task(self, task: Any, context: Dict[str, Any]) -> Optional[Message]:
        """Process a task assigned to logic checker."""
        task_type = task.get("type") if isinstance(task, dict) else str(task)

        if task_type == "validate_path":
            return self._handle_validate_path(task, context)
        elif task_type == "validate_code":
            return self._handle_validate_code(task, context)
        elif task_type == "analyze_maze":
            return self._handle_analyze_maze(task, context)
        else:
            logger.warning(f"LogicChecker: Unknown task type: {task_type}")
            return None

    def _handle_validate_path(self, task: Dict, context: Dict[str, Any]) -> Message:
        """Handle path validation task."""
        maze = task.get("maze", [])
        path = task.get("path", [])
        start = tuple(task.get("start", (0, 0)))
        end = tuple(task.get("end", (0, 0)))

        # Convert path items to tuples if needed
        path = [tuple(p) if isinstance(p, list) else p for p in path]

        result = self.validate_path(maze, path, start, end)

        # Write to blackboard
        self.write_blackboard("facts", "path_validation", result.to_dict())

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.VALIDATION_RESULT,
            priority=Priority.HIGH,
            content={
                "task": "validate_path",
                "valid": result.valid,
                "issue_count": len(result.issues),
                "confidence": result.confidence,
            },
        )

    def _handle_validate_code(self, task: Dict, context: Dict[str, Any]) -> Message:
        """Handle code validation task."""
        code = task.get("code", "")

        result = self.validate_code(code)

        # Write to blackboard
        self.write_blackboard("facts", "code_validation", result.to_dict())

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.VALIDATION_RESULT,
            priority=Priority.HIGH,
            content={
                "task": "validate_code",
                "valid": result.valid,
                "issue_count": len(result.issues),
                "confidence": result.confidence,
            },
        )

    def _handle_analyze_maze(self, task: Dict, context: Dict[str, Any]) -> Message:
        """Handle maze analysis task."""
        maze = context.get("maze")

        if maze is None:
            return Message(
                from_agent=self.name,
                to_agent="broadcast",
                type=MessageType.ERROR,
                priority=Priority.HIGH,
                content={"error": "No maze provided for analysis"},
            )

        # Analyze maze properties
        analysis = self._analyze_maze_properties(maze)

        # Write to blackboard
        self.write_blackboard("facts", "maze_analysis", analysis)

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.TASK_RESULT,
            priority=Priority.NORMAL,
            content={
                "task": "analyze_maze",
                "analysis": analysis,
            },
        )

    def _analyze_maze_properties(self, maze: Maze) -> Dict[str, Any]:
        """Analyze maze properties."""
        total_cells = maze.width * maze.height
        path_cells = sum(
            1 for row in maze.grid for cell in row if cell.value == 0
        )

        return {
            "dimensions": f"{maze.width}x{maze.height}",
            "total_cells": total_cells,
            "path_cells": path_cells,
            "wall_ratio": 1 - (path_cells / total_cells),
            "start": maze.start,
            "end": maze.end,
            "manhattan_distance": abs(maze.end[0] - maze.start[0]) + abs(maze.end[1] - maze.start[1]),
            "complexity": "simple" if total_cells <= 200 else "medium" if total_cells <= 1000 else "complex",
        }

    def validate_output(self, output: Any) -> Tuple[bool, str]:
        """Validate logic checker output."""
        if output is None:
            return False, "Output is None"

        if isinstance(output, ValidationResult):
            return True, "Valid validation result"

        if isinstance(output, Message):
            if output.content is None:
                return False, "Message content is None"
            return True, "Valid message"

        return False, f"Unknown output type: {type(output)}"


# ==============================================================================
# Retrieval Specialist Agent
# ==============================================================================

@dataclass
class RetrievalResult:
    """Result of knowledge retrieval."""

    relevant_solutions: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relevant_solutions": self.relevant_solutions,
            "patterns": self.patterns,
            "confidence": self.confidence,
        }


class RetrievalSpecialist(BaseAgent):
    """
    Retrieval Specialist Agent for knowledge search.

    Responsibilities:
    - Search and retrieve relevant information from the knowledge base
    - Find similar problems and their solutions
    - Extract patterns from past successful approaches
    - Summarize relevant context for other agents
    """

    SYSTEM_PROMPT_TEMPLATE = """You are the Retrieval Specialist Agent in a collaborative swarm intelligence system.

Your responsibilities:
1. Search and retrieve relevant information from the knowledge base
2. Find similar problems and their solutions
3. Extract patterns from past successful approaches
4. Summarize relevant context for other agents
5. Index new knowledge for future retrieval

Retrieval guidelines:
- Focus on the most relevant information
- Prioritize recent and high-quality sources
- Provide concise summaries
- Note confidence levels in retrieved information

When supporting maze solving:
- Retrieve relevant algorithm information
- Find examples of similar maze types
- Provide heuristic suggestions
- Report any relevant past solutions"""

    def __init__(
        self,
        config: AgentConfig,
        ollama_client: Any,
        blackboard: Optional[Any] = None,
    ):
        """Initialize retrieval specialist agent."""
        if not config.permissions:
            config.permissions = {
                "current_goal": Permission.READ,
                "facts": Permission.READ_WRITE,
                "hypotheses": Permission.READ,
                "artifacts": Permission.READ,
                "working_memory": Permission.READ_WRITE,
                "metrics": Permission.READ,
            }

        if not config.system_prompt:
            config.system_prompt = self.SYSTEM_PROMPT_TEMPLATE

        super().__init__(config, ollama_client, blackboard)

        # Knowledge base (in-memory for now)
        self.knowledge_base: Dict[str, Any] = {
            "algorithm_recommendations": {
                "small_sparse": {"algorithm": "BFS", "confidence": 0.95},
                "small_dense": {"algorithm": "BFS", "confidence": 0.90},
                "medium_sparse": {"algorithm": "A*", "confidence": 0.95},
                "medium_dense": {"algorithm": "A*", "confidence": 0.85},
                "large_sparse": {"algorithm": "A*", "confidence": 0.95},
                "large_dense": {"algorithm": "A*", "confidence": 0.80},
            },
            "patterns": [
                {
                    "pattern": "BFS guarantees shortest path",
                    "context": "When optimality is required",
                    "confidence": 1.0,
                },
                {
                    "pattern": "A* is faster for large sparse mazes",
                    "context": "When maze has low wall density",
                    "confidence": 0.9,
                },
                {
                    "pattern": "DFS uses less memory but may not find optimal",
                    "context": "Memory-constrained environments",
                    "confidence": 0.95,
                },
            ],
        }

    def search_solutions(
        self, maze_properties: Dict[str, Any]
    ) -> RetrievalResult:
        """
        Search for relevant solutions based on maze properties.

        Args:
            maze_properties: Properties of the current maze

        Returns:
            RetrievalResult with relevant solutions
        """
        relevant = []

        # Determine maze category
        total_cells = maze_properties.get("total_cells", 0)
        wall_ratio = maze_properties.get("wall_ratio", 0.5)

        if total_cells <= 200:
            size = "small"
        elif total_cells <= 1000:
            size = "medium"
        else:
            size = "large"

        density = "dense" if wall_ratio > 0.4 else "sparse"
        category = f"{size}_{density}"

        # Get recommendation from knowledge base
        recommendation = self.knowledge_base["algorithm_recommendations"].get(category)
        if recommendation:
            relevant.append({
                "maze_category": category,
                "recommended_algorithm": recommendation["algorithm"],
                "confidence": recommendation["confidence"],
                "reason": f"Based on maze size ({size}) and density ({density})",
            })

        # Get relevant patterns
        patterns = []
        for pattern in self.knowledge_base["patterns"]:
            patterns.append({
                "pattern": pattern["pattern"],
                "confidence": pattern["confidence"],
            })

        return RetrievalResult(
            relevant_solutions=relevant,
            patterns=patterns,
            confidence=0.9 if relevant else 0.5,
        )

    def find_similar_mazes(
        self, maze_properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find similar mazes from past solutions.

        Args:
            maze_properties: Properties of the current maze

        Returns:
            List of similar maze info
        """
        # In a real implementation, this would search a database
        # For now, return synthetic examples
        total_cells = maze_properties.get("total_cells", 0)

        similar = []
        if total_cells <= 200:
            similar.append({
                "maze_id": "example_small_1",
                "dimensions": "10x10",
                "best_algorithm": "BFS",
                "avg_solve_time": 0.001,
                "similarity": 0.9,
            })
        elif total_cells <= 1000:
            similar.append({
                "maze_id": "example_medium_1",
                "dimensions": "25x25",
                "best_algorithm": "A*",
                "avg_solve_time": 0.01,
                "similarity": 0.85,
            })
        else:
            similar.append({
                "maze_id": "example_large_1",
                "dimensions": "50x50",
                "best_algorithm": "A*",
                "avg_solve_time": 0.1,
                "similarity": 0.8,
            })

        return similar

    def process_task(self, task: Any, context: Dict[str, Any]) -> Optional[Message]:
        """Process a task assigned to retrieval specialist."""
        task_type = task.get("type") if isinstance(task, dict) else str(task)

        if task_type == "search_solutions":
            return self._handle_search_solutions(task, context)
        elif task_type == "find_patterns":
            return self._handle_find_patterns(task, context)
        else:
            logger.warning(f"RetrievalSpecialist: Unknown task type: {task_type}")
            return None

    def _handle_search_solutions(
        self, task: Dict, context: Dict[str, Any]
    ) -> Message:
        """Handle solution search task."""
        maze_properties = task.get("maze_properties", {})

        result = self.search_solutions(maze_properties)
        similar = self.find_similar_mazes(maze_properties)

        # Write to blackboard
        self.write_blackboard("facts", "similar_solutions", {
            "solutions": result.relevant_solutions,
            "similar_mazes": similar,
            "patterns": result.patterns,
        })

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.TASK_RESULT,
            priority=Priority.NORMAL,
            content={
                "task": "search_solutions",
                "found_solutions": len(result.relevant_solutions),
                "found_patterns": len(result.patterns),
                "confidence": result.confidence,
            },
        )

    def _handle_find_patterns(self, task: Dict, context: Dict[str, Any]) -> Message:
        """Handle pattern finding task."""
        patterns = self.knowledge_base.get("patterns", [])

        self.state.tasks_completed += 1

        return Message(
            from_agent=self.name,
            to_agent="broadcast",
            type=MessageType.TASK_RESULT,
            priority=Priority.LOW,
            content={
                "task": "find_patterns",
                "patterns": patterns,
            },
        )

    def validate_output(self, output: Any) -> Tuple[bool, str]:
        """Validate retrieval specialist output."""
        if output is None:
            return False, "Output is None"

        if isinstance(output, RetrievalResult):
            return True, "Valid retrieval result"

        if isinstance(output, Message):
            if output.content is None:
                return False, "Message content is None"
            return True, "Valid message"

        return False, f"Unknown output type: {type(output)}"
