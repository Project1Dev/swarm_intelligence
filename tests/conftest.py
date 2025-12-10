"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
from pathlib import Path

from utils.maze_generator import Maze, Cell


@pytest.fixture
def simple_3x3_maze():
    """Create a simple 3x3 maze for testing."""
    grid = [
        [Cell.PATH, Cell.PATH, Cell.PATH],
        [Cell.WALL, Cell.WALL, Cell.PATH],
        [Cell.PATH, Cell.PATH, Cell.PATH],
    ]
    return Maze(
        width=3,
        height=3,
        grid=grid,
        start=(0, 0),
        end=(2, 2),
        algorithm="test",
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def benchmark_mazes(temp_dir):
    """Generate small benchmark mazes for testing."""
    from utils.maze_generator import RecursiveBacktrackingGenerator

    mazes = []
    for i in range(3):
        generator = RecursiveBacktrackingGenerator(width=11, height=11, seed=i)
        maze = generator.generate()
        filepath = temp_dir / f"test_maze_{i}.json"
        maze.save(filepath)
        mazes.append(maze)

    return mazes, temp_dir
