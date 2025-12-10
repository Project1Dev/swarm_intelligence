"""Maze generation and representation system."""

import json
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Set, Optional
from enum import IntEnum
from pathlib import Path
from collections import deque


class Cell(IntEnum):
    """Cell types in the maze."""

    WALL = 0
    PATH = 1
    START = 2
    END = 3


@dataclass
class Maze:
    """Maze data structure."""

    width: int
    height: int
    grid: List[List[int]]
    start: Tuple[int, int]
    end: Tuple[int, int]
    algorithm: str  # Generator algorithm used

    def __post_init__(self) -> None:
        """Validate maze after initialization."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Maze dimensions must be positive")
        if len(self.grid) != self.height:
            raise ValueError(f"Grid height {len(self.grid)} != maze height {self.height}")
        if any(len(row) != self.width for row in self.grid):
            raise ValueError("All grid rows must have same width as maze")
        if not (0 <= self.start[0] < self.height and 0 <= self.start[1] < self.width):
            raise ValueError(f"Start position {self.start} out of bounds")
        if not (0 <= self.end[0] < self.height and 0 <= self.end[1] < self.width):
            raise ValueError(f"End position {self.end} out of bounds")

    def get_cell(self, row: int, col: int) -> int:
        """Get cell value at position."""
        return self.grid[row][col]

    def set_cell(self, row: int, col: int, value: int) -> None:
        """Set cell value at position."""
        self.grid[row][col] = value

    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within bounds."""
        return 0 <= row < self.height and 0 <= col < self.width

    def is_walkable(self, row: int, col: int) -> bool:
        """Check if cell is walkable (not a wall)."""
        return self.is_valid_position(row, col) and self.grid[row][col] != Cell.WALL

    def get_neighbors(self, row: int, col: int, include_diagonals: bool = False) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        neighbors = []
        # Cardinal directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if include_diagonals:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                neighbors.append((new_row, new_col))

        return neighbors

    def to_ascii(self, path: Optional[List[Tuple[int, int]]] = None) -> str:
        """Convert maze to ASCII representation."""
        path_set = set(path) if path else set()
        lines = []

        for r in range(self.height):
            line = ""
            for c in range(self.width):
                pos = (r, c)
                if pos == self.start:
                    line += "S"
                elif pos == self.end:
                    line += "E"
                elif pos in path_set:
                    line += "."
                elif self.grid[r][c] == Cell.WALL:
                    line += "#"
                else:
                    line += " "
            lines.append(line)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "width": self.width,
            "height": self.height,
            "grid": self.grid,
            "start": list(self.start),
            "end": list(self.end),
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Maze":
        """Create maze from dictionary."""
        return cls(
            width=data["width"],
            height=data["height"],
            grid=data["grid"],
            start=tuple(data["start"]),
            end=tuple(data["end"]),
            algorithm=data["algorithm"],
        )

    def save(self, filepath: Path) -> None:
        """Save maze to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "Maze":
        """Load maze from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class MazeGenerator:
    """Base class for maze generation algorithms."""

    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        """
        Initialize maze generator.

        Args:
            width: Maze width (columns)
            height: Maze height (rows)
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def generate(self) -> Maze:
        """Generate a maze. Must be implemented by subclasses."""
        raise NotImplementedError

    def _initialize_grid(self, fill_value: int = Cell.WALL) -> List[List[int]]:
        """Initialize a grid filled with given value."""
        return [[fill_value for _ in range(self.width)] for _ in range(self.height)]

    def _choose_start_end(self, grid: List[List[int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Choose valid start and end positions that are reachable from each other."""
        # Find all path cells
        path_cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if grid[r][c] == Cell.PATH
        ]

        if len(path_cells) < 2:
            raise ValueError("Not enough path cells for start and end")

        # Find the largest connected component using flood fill
        visited = set()
        components = []

        def flood_fill(start_pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
            """Flood fill to find connected component."""
            component = set()
            queue = deque([start_pos])
            component.add(start_pos)

            while queue:
                r, c = queue.popleft()
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self.height
                        and 0 <= nc < self.width
                        and grid[nr][nc] == Cell.PATH
                        and (nr, nc) not in component
                    ):
                        component.add((nr, nc))
                        queue.append((nr, nc))

            return component

        # Find all connected components
        for cell in path_cells:
            if cell not in visited:
                component = flood_fill(cell)
                visited.update(component)
                components.append(list(component))

        # Use the largest component
        largest_component = max(components, key=len)

        # Choose start from top half, end from bottom half within the component
        top_half = [cell for cell in largest_component if cell[0] < self.height // 2]
        bottom_half = [cell for cell in largest_component if cell[0] >= self.height // 2]

        start = random.choice(top_half) if top_half else random.choice(largest_component)
        end = random.choice(bottom_half) if bottom_half else random.choice(largest_component)

        # Ensure start != end
        if start == end and len(largest_component) > 1:
            remaining = [c for c in largest_component if c != start]
            end = random.choice(remaining)

        return start, end


class RecursiveBacktrackingGenerator(MazeGenerator):
    """Generate maze using recursive backtracking (DFS-based)."""

    def generate(self) -> Maze:
        """Generate maze using recursive backtracking algorithm."""
        # Start with all walls
        grid = self._initialize_grid(Cell.WALL)

        # Choose random starting cell (odd coordinates for proper wall structure)
        start_row = random.randrange(1, self.height, 2)
        start_col = random.randrange(1, self.width, 2)
        grid[start_row][start_col] = Cell.PATH

        # Stack for backtracking
        stack = [(start_row, start_col)]
        visited = {(start_row, start_col)}

        while stack:
            current_row, current_col = stack[-1]

            # Find unvisited neighbors (2 cells away in cardinal directions)
            neighbors = []
            for dr, dc in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                new_row, new_col = current_row + dr, current_col + dc
                if (
                    0 < new_row < self.height
                    and 0 < new_col < self.width
                    and (new_row, new_col) not in visited
                ):
                    neighbors.append((new_row, new_col, dr // 2, dc // 2))

            if neighbors:
                # Choose random unvisited neighbor
                new_row, new_col, wall_dr, wall_dc = random.choice(neighbors)

                # Carve path to neighbor
                grid[new_row][new_col] = Cell.PATH
                grid[current_row + wall_dr][current_col + wall_dc] = Cell.PATH

                visited.add((new_row, new_col))
                stack.append((new_row, new_col))
            else:
                # Backtrack
                stack.pop()

        # Choose start and end positions
        start, end = self._choose_start_end(grid)

        return Maze(
            width=self.width,
            height=self.height,
            grid=grid,
            start=start,
            end=end,
            algorithm="recursive_backtracking",
        )


class PrimsAlgorithmGenerator(MazeGenerator):
    """Generate maze using Prim's algorithm (minimum spanning tree)."""

    def generate(self) -> Maze:
        """Generate maze using Prim's algorithm."""
        # Start with all walls
        grid = self._initialize_grid(Cell.WALL)

        # Choose random starting cell
        start_row = random.randrange(1, self.height, 2)
        start_col = random.randrange(1, self.width, 2)
        grid[start_row][start_col] = Cell.PATH

        # Frontier cells (walls adjacent to path)
        frontier = []
        for dr, dc in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            new_row, new_col = start_row + dr, start_col + dc
            if 0 < new_row < self.height and 0 < new_col < self.width:
                frontier.append((new_row, new_col))

        while frontier:
            # Choose random frontier cell
            current_row, current_col = random.choice(frontier)
            frontier.remove((current_row, current_col))

            # Find adjacent path cells
            path_neighbors = []
            for dr, dc in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                new_row, new_col = current_row + dr, current_col + dc
                if (
                    0 < new_row < self.height
                    and 0 < new_col < self.width
                    and grid[new_row][new_col] == Cell.PATH
                ):
                    path_neighbors.append((new_row, new_col, dr // 2, dc // 2))

            if path_neighbors:
                # Connect to random path neighbor
                path_row, path_col, wall_dr, wall_dc = random.choice(path_neighbors)
                grid[current_row][current_col] = Cell.PATH
                grid[current_row - wall_dr][current_col - wall_dc] = Cell.PATH

                # Add new frontier cells
                for dr, dc in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                    new_row, new_col = current_row + dr, current_col + dc
                    if (
                        0 < new_row < self.height
                        and 0 < new_col < self.width
                        and grid[new_row][new_col] == Cell.WALL
                        and (new_row, new_col) not in frontier
                    ):
                        frontier.append((new_row, new_col))

        # Choose start and end positions
        start, end = self._choose_start_end(grid)

        return Maze(
            width=self.width,
            height=self.height,
            grid=grid,
            start=start,
            end=end,
            algorithm="prims_algorithm",
        )


class RandomGrowthGenerator(MazeGenerator):
    """Generate maze using random growth (BFS-like)."""

    def generate(self) -> Maze:
        """Generate maze using random growth algorithm."""
        # Start with all walls
        grid = self._initialize_grid(Cell.WALL)

        # Choose random starting cell
        start_row = random.randrange(1, self.height, 2)
        start_col = random.randrange(1, self.width, 2)
        grid[start_row][start_col] = Cell.PATH

        # Cells to grow from
        growing = [(start_row, start_col)]

        while growing:
            # Choose random growing cell
            current_row, current_col = random.choice(growing)

            # Find unvisited neighbors
            neighbors = []
            for dr, dc in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                new_row, new_col = current_row + dr, current_col + dc
                if (
                    0 < new_row < self.height
                    and 0 < new_col < self.width
                    and grid[new_row][new_col] == Cell.WALL
                ):
                    neighbors.append((new_row, new_col, dr // 2, dc // 2))

            if neighbors:
                # Carve path to random neighbor
                new_row, new_col, wall_dr, wall_dc = random.choice(neighbors)
                grid[new_row][new_col] = Cell.PATH
                grid[current_row + wall_dr][current_col + wall_dc] = Cell.PATH

                # Add new cell to growing list
                growing.append((new_row, new_col))
            else:
                # No more neighbors, stop growing from this cell
                growing.remove((current_row, current_col))

        # Choose start and end positions
        start, end = self._choose_start_end(grid)

        return Maze(
            width=self.width,
            height=self.height,
            grid=grid,
            start=start,
            end=end,
            algorithm="random_growth",
        )


def validate_maze_solvable(maze: Maze) -> bool:
    """
    Validate that maze has a path from start to end using BFS.

    Args:
        maze: Maze to validate

    Returns:
        True if maze is solvable, False otherwise
    """
    queue = deque([maze.start])
    visited = {maze.start}

    while queue:
        current = queue.popleft()

        if current == maze.end:
            return True

        row, col = current
        for neighbor in maze.get_neighbors(row, col):
            if neighbor not in visited and maze.is_walkable(*neighbor):
                visited.add(neighbor)
                queue.append(neighbor)

    return False


def generate_benchmark_suite(output_dir: Path) -> List[Maze]:
    """
    Generate benchmark maze suite with varying complexity.

    Args:
        output_dir: Directory to save generated mazes

    Returns:
        List of generated mazes
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mazes = []
    maze_id = 0

    # Define benchmark configurations
    configs = [
        (10, 10, "simple", 5),      # 5x 10x10 (simple)
        (20, 20, "medium", 5),      # 5x 20x20 (medium)
        (30, 30, "complex", 5),     # 5x 30x30 (complex)
        (50, 50, "stress", 5),      # 5x 50x50 (stress test)
    ]

    generators = [
        RecursiveBacktrackingGenerator,
        PrimsAlgorithmGenerator,
        RandomGrowthGenerator,
    ]

    for width, height, label, count in configs:
        # Ensure odd dimensions for proper maze structure
        width = width + 1 if width % 2 == 0 else width
        height = height + 1 if height % 2 == 0 else height

        for i in range(count):
            # Cycle through generator algorithms
            generator_class = generators[i % len(generators)]
            generator = generator_class(width=width, height=height, seed=maze_id)

            maze = generator.generate()

            # Validate solvability
            if not validate_maze_solvable(maze):
                print(f"Warning: Maze {maze_id} ({label}) is not solvable, regenerating...")
                # Try again with different seed
                generator = generator_class(width=width, height=height, seed=maze_id + 1000)
                maze = generator.generate()

            # Save maze
            filename = f"maze_{label}_{width}x{height}_{i:02d}.json"
            filepath = output_dir / filename
            maze.save(filepath)

            mazes.append(maze)
            maze_id += 1

            print(f"Generated maze {maze_id}: {filename} ({maze.algorithm})")

    return mazes
