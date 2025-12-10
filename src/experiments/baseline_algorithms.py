"""Baseline pathfinding algorithms for performance comparison."""

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from collections import deque
import heapq

from utils.maze_generator import Maze, Cell


@dataclass
class PathfindingResult:
    """Result of a pathfinding algorithm execution."""

    algorithm: str
    success: bool
    path: Optional[List[Tuple[int, int]]]
    path_length: int
    nodes_explored: int
    execution_time: float  # seconds
    optimal: bool = False  # True if path is proven optimal

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "algorithm": self.algorithm,
            "success": self.success,
            "path": [list(pos) for pos in self.path] if self.path else None,
            "path_length": self.path_length,
            "nodes_explored": self.nodes_explored,
            "execution_time": self.execution_time,
            "optimal": self.optimal,
        }


class PathfindingAlgorithm:
    """Base class for pathfinding algorithms."""

    def __init__(self, name: str):
        """
        Initialize pathfinding algorithm.

        Args:
            name: Algorithm name
        """
        self.name = name
        self.nodes_explored = 0

    def find_path(self, maze: Maze) -> PathfindingResult:
        """Find path from start to end. Must be implemented by subclasses."""
        raise NotImplementedError

    def _reconstruct_path(
        self, came_from: Dict[Tuple[int, int], Tuple[int, int]], start: Tuple[int, int], end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary."""
        path = []
        current = end

        while current != start:
            path.append(current)
            if current not in came_from:
                return []  # No path found
            current = came_from[current]

        path.append(start)
        path.reverse()
        return path


class BFSPathfinder(PathfindingAlgorithm):
    """Breadth-First Search pathfinding algorithm."""

    def __init__(self):
        """Initialize BFS pathfinder."""
        super().__init__("BFS")

    def find_path(self, maze: Maze) -> PathfindingResult:
        """
        Find path using BFS.

        BFS guarantees the shortest path in terms of number of steps.

        Time Complexity: O(V + E) where V is vertices, E is edges
        Space Complexity: O(V)
        """
        start_time = time.time()
        self.nodes_explored = 0

        queue = deque([maze.start])
        visited = {maze.start}
        came_from = {}

        while queue:
            current = queue.popleft()
            self.nodes_explored += 1

            if current == maze.end:
                # Found the end
                path = self._reconstruct_path(came_from, maze.start, maze.end)
                execution_time = time.time() - start_time

                return PathfindingResult(
                    algorithm=self.name,
                    success=True,
                    path=path,
                    path_length=len(path),
                    nodes_explored=self.nodes_explored,
                    execution_time=execution_time,
                    optimal=True,  # BFS guarantees optimal path
                )

            row, col = current
            for neighbor in maze.get_neighbors(row, col):
                if neighbor not in visited and maze.is_walkable(*neighbor):
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

        # No path found
        execution_time = time.time() - start_time
        return PathfindingResult(
            algorithm=self.name,
            success=False,
            path=None,
            path_length=0,
            nodes_explored=self.nodes_explored,
            execution_time=execution_time,
            optimal=False,
        )


class DFSPathfinder(PathfindingAlgorithm):
    """Depth-First Search pathfinding algorithm."""

    def __init__(self):
        """Initialize DFS pathfinder."""
        super().__init__("DFS")

    def find_path(self, maze: Maze) -> PathfindingResult:
        """
        Find path using DFS.

        DFS does NOT guarantee the shortest path.

        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        start_time = time.time()
        self.nodes_explored = 0

        stack = [maze.start]
        visited = {maze.start}
        came_from = {}

        while stack:
            current = stack.pop()
            self.nodes_explored += 1

            if current == maze.end:
                # Found the end
                path = self._reconstruct_path(came_from, maze.start, maze.end)
                execution_time = time.time() - start_time

                return PathfindingResult(
                    algorithm=self.name,
                    success=True,
                    path=path,
                    path_length=len(path),
                    nodes_explored=self.nodes_explored,
                    execution_time=execution_time,
                    optimal=False,  # DFS does not guarantee optimal path
                )

            row, col = current
            for neighbor in maze.get_neighbors(row, col):
                if neighbor not in visited and maze.is_walkable(*neighbor):
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append(neighbor)

        # No path found
        execution_time = time.time() - start_time
        return PathfindingResult(
            algorithm=self.name,
            success=False,
            path=None,
            path_length=0,
            nodes_explored=self.nodes_explored,
            execution_time=execution_time,
            optimal=False,
        )


class AStarPathfinder(PathfindingAlgorithm):
    """A* pathfinding algorithm with Manhattan distance heuristic."""

    def __init__(self):
        """Initialize A* pathfinder."""
        super().__init__("A*")

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def find_path(self, maze: Maze) -> PathfindingResult:
        """
        Find path using A* with Manhattan distance heuristic.

        A* guarantees the shortest path when using an admissible heuristic.

        Time Complexity: O(E) where E is edges (with good heuristic)
        Space Complexity: O(V)
        """
        start_time = time.time()
        self.nodes_explored = 0

        # Priority queue: (f_score, counter, position)
        # f_score = g_score + h_score
        counter = 0
        open_set = [(0, counter, maze.start)]
        came_from = {}

        # g_score: cost from start to node
        g_score = {maze.start: 0}

        # f_score: estimated cost from start through node to end
        f_score = {maze.start: self._manhattan_distance(maze.start, maze.end)}

        open_set_hash = {maze.start}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            self.nodes_explored += 1

            if current == maze.end:
                # Found the end
                path = self._reconstruct_path(came_from, maze.start, maze.end)
                execution_time = time.time() - start_time

                return PathfindingResult(
                    algorithm=self.name,
                    success=True,
                    path=path,
                    path_length=len(path),
                    nodes_explored=self.nodes_explored,
                    execution_time=execution_time,
                    optimal=True,  # A* with admissible heuristic guarantees optimal path
                )

            row, col = current
            current_g = g_score[current]

            for neighbor in maze.get_neighbors(row, col):
                if not maze.is_walkable(*neighbor):
                    continue

                # Tentative g_score (cost to reach neighbor through current)
                tentative_g = current_g + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._manhattan_distance(neighbor, maze.end)
                    f_score[neighbor] = f

                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor))
                        open_set_hash.add(neighbor)

        # No path found
        execution_time = time.time() - start_time
        return PathfindingResult(
            algorithm=self.name,
            success=False,
            path=None,
            path_length=0,
            nodes_explored=self.nodes_explored,
            execution_time=execution_time,
            optimal=False,
        )


def run_all_algorithms(maze: Maze) -> List[PathfindingResult]:
    """
    Run all baseline algorithms on a maze.

    Args:
        maze: Maze to solve

    Returns:
        List of results from each algorithm
    """
    algorithms = [BFSPathfinder(), DFSPathfinder(), AStarPathfinder()]
    results = []

    for algorithm in algorithms:
        result = algorithm.find_path(maze)
        results.append(result)

    return results


def compare_algorithms(results: List[PathfindingResult]) -> dict:
    """
    Compare algorithm performance.

    Args:
        results: List of results from different algorithms

    Returns:
        Comparison dictionary with statistics
    """
    comparison = {
        "algorithms": {},
        "fastest": None,
        "most_efficient": None,  # Fewest nodes explored
        "shortest_path": None,
    }

    fastest_time = float("inf")
    fewest_nodes = float("inf")
    shortest_path = float("inf")

    for result in results:
        if result.success:
            comparison["algorithms"][result.algorithm] = {
                "path_length": result.path_length,
                "nodes_explored": result.nodes_explored,
                "execution_time": result.execution_time,
                "optimal": result.optimal,
            }

            if result.execution_time < fastest_time:
                fastest_time = result.execution_time
                comparison["fastest"] = result.algorithm

            if result.nodes_explored < fewest_nodes:
                fewest_nodes = result.nodes_explored
                comparison["most_efficient"] = result.algorithm

            if result.path_length < shortest_path:
                shortest_path = result.path_length
                comparison["shortest_path"] = result.algorithm

    return comparison


def visualize_result(maze: Maze, result: PathfindingResult) -> str:
    """
    Create visualization of pathfinding result.

    Args:
        maze: The maze
        result: Pathfinding result

    Returns:
        ASCII visualization with path marked
    """
    if not result.success or result.path is None:
        return f"{result.algorithm} failed to find a path.\n{maze.to_ascii()}"

    visualization = f"{result.algorithm} Result:\n"
    visualization += f"  Path Length: {result.path_length}\n"
    visualization += f"  Nodes Explored: {result.nodes_explored}\n"
    visualization += f"  Execution Time: {result.execution_time:.6f}s\n"
    visualization += f"  Optimal: {result.optimal}\n\n"
    visualization += maze.to_ascii(path=result.path)

    return visualization
