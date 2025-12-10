"""Unit tests for maze generation system."""

import pytest
import tempfile
from pathlib import Path

from utils.maze_generator import (
    Maze,
    Cell,
    MazeGenerator,
    RecursiveBacktrackingGenerator,
    PrimsAlgorithmGenerator,
    RandomGrowthGenerator,
    validate_maze_solvable,
    generate_benchmark_suite,
)


class TestMaze:
    """Test Maze data structure."""

    def test_maze_creation(self):
        """Test creating a valid maze."""
        grid = [[Cell.PATH, Cell.WALL], [Cell.WALL, Cell.PATH]]
        maze = Maze(
            width=2,
            height=2,
            grid=grid,
            start=(0, 0),
            end=(1, 1),
            algorithm="test",
        )

        assert maze.width == 2
        assert maze.height == 2
        assert maze.start == (0, 0)
        assert maze.end == (1, 1)
        assert maze.algorithm == "test"

    def test_maze_invalid_dimensions(self):
        """Test creating maze with invalid dimensions."""
        grid = [[Cell.PATH, Cell.WALL]]
        with pytest.raises(ValueError, match="must be positive"):
            Maze(width=0, height=1, grid=grid, start=(0, 0), end=(0, 1), algorithm="test")

    def test_maze_invalid_grid_height(self):
        """Test creating maze with mismatched grid height."""
        grid = [[Cell.PATH]]
        with pytest.raises(ValueError, match="Grid height"):
            Maze(width=1, height=2, grid=grid, start=(0, 0), end=(1, 0), algorithm="test")

    def test_maze_invalid_grid_width(self):
        """Test creating maze with mismatched grid width."""
        grid = [[Cell.PATH, Cell.WALL], [Cell.PATH]]
        with pytest.raises(ValueError, match="same width"):
            Maze(width=2, height=2, grid=grid, start=(0, 0), end=(1, 0), algorithm="test")

    def test_maze_invalid_start_position(self):
        """Test creating maze with invalid start position."""
        grid = [[Cell.PATH]]
        with pytest.raises(ValueError, match="Start position"):
            Maze(width=1, height=1, grid=grid, start=(5, 5), end=(0, 0), algorithm="test")

    def test_maze_invalid_end_position(self):
        """Test creating maze with invalid end position."""
        grid = [[Cell.PATH]]
        with pytest.raises(ValueError, match="End position"):
            Maze(width=1, height=1, grid=grid, start=(0, 0), end=(5, 5), algorithm="test")

    def test_get_cell(self):
        """Test getting cell value."""
        grid = [[Cell.PATH, Cell.WALL], [Cell.WALL, Cell.PATH]]
        maze = Maze(width=2, height=2, grid=grid, start=(0, 0), end=(1, 1), algorithm="test")

        assert maze.get_cell(0, 0) == Cell.PATH
        assert maze.get_cell(0, 1) == Cell.WALL
        assert maze.get_cell(1, 0) == Cell.WALL
        assert maze.get_cell(1, 1) == Cell.PATH

    def test_set_cell(self):
        """Test setting cell value."""
        grid = [[Cell.PATH, Cell.PATH]]
        maze = Maze(width=2, height=1, grid=grid, start=(0, 0), end=(0, 1), algorithm="test")

        maze.set_cell(0, 0, Cell.WALL)
        assert maze.get_cell(0, 0) == Cell.WALL

    def test_is_valid_position(self):
        """Test checking valid positions."""
        grid = [[Cell.PATH, Cell.PATH]]
        maze = Maze(width=2, height=1, grid=grid, start=(0, 0), end=(0, 1), algorithm="test")

        assert maze.is_valid_position(0, 0) is True
        assert maze.is_valid_position(0, 1) is True
        assert maze.is_valid_position(1, 0) is False
        assert maze.is_valid_position(-1, 0) is False

    def test_is_walkable(self):
        """Test checking walkable cells."""
        grid = [[Cell.PATH, Cell.WALL], [Cell.WALL, Cell.PATH]]
        maze = Maze(width=2, height=2, grid=grid, start=(0, 0), end=(1, 1), algorithm="test")

        assert maze.is_walkable(0, 0) is True
        assert maze.is_walkable(0, 1) is False
        assert maze.is_walkable(1, 0) is False
        assert maze.is_walkable(1, 1) is True
        assert maze.is_walkable(5, 5) is False

    def test_get_neighbors_cardinal(self):
        """Test getting cardinal neighbors."""
        grid = [[Cell.PATH] * 3 for _ in range(3)]
        maze = Maze(width=3, height=3, grid=grid, start=(0, 0), end=(2, 2), algorithm="test")

        neighbors = maze.get_neighbors(1, 1)
        assert len(neighbors) == 4
        assert (0, 1) in neighbors  # up
        assert (2, 1) in neighbors  # down
        assert (1, 0) in neighbors  # left
        assert (1, 2) in neighbors  # right

    def test_get_neighbors_with_diagonals(self):
        """Test getting neighbors including diagonals."""
        grid = [[Cell.PATH] * 3 for _ in range(3)]
        maze = Maze(width=3, height=3, grid=grid, start=(0, 0), end=(2, 2), algorithm="test")

        neighbors = maze.get_neighbors(1, 1, include_diagonals=True)
        assert len(neighbors) == 8

    def test_get_neighbors_edge(self):
        """Test getting neighbors at maze edge."""
        grid = [[Cell.PATH] * 3 for _ in range(3)]
        maze = Maze(width=3, height=3, grid=grid, start=(0, 0), end=(2, 2), algorithm="test")

        neighbors = maze.get_neighbors(0, 0)
        assert len(neighbors) == 2  # Only right and down

    def test_to_ascii_simple(self):
        """Test ASCII representation."""
        grid = [[Cell.PATH, Cell.WALL, Cell.PATH], [Cell.WALL, Cell.PATH, Cell.WALL]]
        maze = Maze(width=3, height=2, grid=grid, start=(0, 0), end=(0, 2), algorithm="test")

        ascii_repr = maze.to_ascii()
        lines = ascii_repr.split("\n")

        assert len(lines) == 2
        assert lines[0] == "S#E"
        assert lines[1] == "# #"

    def test_to_ascii_with_path(self):
        """Test ASCII representation with solution path."""
        grid = [[Cell.PATH] * 3, [Cell.WALL, Cell.PATH, Cell.WALL], [Cell.PATH] * 3]
        maze = Maze(width=3, height=3, grid=grid, start=(0, 0), end=(2, 2), algorithm="test")

        path = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
        ascii_repr = maze.to_ascii(path)

        assert "S" in ascii_repr
        assert "E" in ascii_repr
        assert "." in ascii_repr  # Path markers

    def test_serialization(self):
        """Test maze serialization to dict."""
        grid = [[Cell.PATH, Cell.WALL]]
        maze = Maze(width=2, height=1, grid=grid, start=(0, 0), end=(0, 1), algorithm="test")

        data = maze.to_dict()

        assert data["width"] == 2
        assert data["height"] == 1
        assert data["grid"] == grid
        assert data["start"] == [0, 0]
        assert data["end"] == [0, 1]
        assert data["algorithm"] == "test"

    def test_deserialization(self):
        """Test maze deserialization from dict."""
        data = {
            "width": 2,
            "height": 1,
            "grid": [[Cell.PATH, Cell.WALL]],
            "start": [0, 0],
            "end": [0, 1],
            "algorithm": "test",
        }

        maze = Maze.from_dict(data)

        assert maze.width == 2
        assert maze.height == 1
        assert maze.start == (0, 0)
        assert maze.end == (0, 1)
        assert maze.algorithm == "test"

    def test_save_and_load(self):
        """Test saving and loading maze from file."""
        grid = [[Cell.PATH, Cell.WALL]]
        maze = Maze(width=2, height=1, grid=grid, start=(0, 0), end=(0, 1), algorithm="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_maze.json"
            maze.save(filepath)

            loaded_maze = Maze.load(filepath)

            assert loaded_maze.width == maze.width
            assert loaded_maze.height == maze.height
            assert loaded_maze.grid == maze.grid
            assert loaded_maze.start == maze.start
            assert loaded_maze.end == maze.end
            assert loaded_maze.algorithm == maze.algorithm


class TestMazeGenerators:
    """Test maze generation algorithms."""

    def test_recursive_backtracking_generator(self):
        """Test recursive backtracking generator."""
        generator = RecursiveBacktrackingGenerator(width=11, height=11, seed=42)
        maze = generator.generate()

        assert maze.width == 11
        assert maze.height == 11
        assert maze.algorithm == "recursive_backtracking"
        assert maze.start != maze.end
        assert validate_maze_solvable(maze)

    def test_prims_algorithm_generator(self):
        """Test Prim's algorithm generator."""
        generator = PrimsAlgorithmGenerator(width=11, height=11, seed=42)
        maze = generator.generate()

        assert maze.width == 11
        assert maze.height == 11
        assert maze.algorithm == "prims_algorithm"
        assert maze.start != maze.end
        assert validate_maze_solvable(maze)

    def test_random_growth_generator(self):
        """Test random growth generator."""
        generator = RandomGrowthGenerator(width=11, height=11, seed=42)
        maze = generator.generate()

        assert maze.width == 11
        assert maze.height == 11
        assert maze.algorithm == "random_growth"
        assert maze.start != maze.end
        assert validate_maze_solvable(maze)

    def test_generator_reproducibility(self):
        """Test that same seed produces same maze."""
        generator1 = RecursiveBacktrackingGenerator(width=11, height=11, seed=123)
        maze1 = generator1.generate()

        generator2 = RecursiveBacktrackingGenerator(width=11, height=11, seed=123)
        maze2 = generator2.generate()

        assert maze1.grid == maze2.grid
        assert maze1.start == maze2.start
        assert maze1.end == maze2.end

    def test_generator_different_seeds(self):
        """Test that different seeds produce different mazes."""
        generator1 = RecursiveBacktrackingGenerator(width=11, height=11, seed=123)
        maze1 = generator1.generate()

        generator2 = RecursiveBacktrackingGenerator(width=11, height=11, seed=456)
        maze2 = generator2.generate()

        # Grids should be different (very high probability)
        assert maze1.grid != maze2.grid or maze1.start != maze2.start

    def test_generator_sizes(self):
        """Test generating mazes of various sizes."""
        sizes = [(5, 5), (11, 11), (21, 21), (31, 31)]

        for width, height in sizes:
            generator = RecursiveBacktrackingGenerator(width=width, height=height, seed=42)
            maze = generator.generate()

            assert maze.width == width
            assert maze.height == height
            assert validate_maze_solvable(maze)

    def test_all_generators_produce_solvable_mazes(self):
        """Test that all generator types produce solvable mazes."""
        generators = [
            RecursiveBacktrackingGenerator,
            PrimsAlgorithmGenerator,
            RandomGrowthGenerator,
        ]

        for generator_class in generators:
            generator = generator_class(width=11, height=11, seed=42)
            maze = generator.generate()

            assert validate_maze_solvable(maze), f"{generator_class.__name__} produced unsolvable maze"


class TestMazeValidation:
    """Test maze validation."""

    def test_validate_solvable_maze(self):
        """Test validating a solvable maze."""
        grid = [
            [Cell.PATH, Cell.PATH, Cell.PATH],
            [Cell.WALL, Cell.WALL, Cell.PATH],
            [Cell.PATH, Cell.PATH, Cell.PATH],
        ]
        maze = Maze(width=3, height=3, grid=grid, start=(0, 0), end=(2, 2), algorithm="test")

        assert validate_maze_solvable(maze) is True

    def test_validate_unsolvable_maze(self):
        """Test validating an unsolvable maze."""
        grid = [
            [Cell.PATH, Cell.WALL, Cell.PATH],
            [Cell.WALL, Cell.WALL, Cell.WALL],
            [Cell.PATH, Cell.WALL, Cell.PATH],
        ]
        maze = Maze(width=3, height=3, grid=grid, start=(0, 0), end=(2, 2), algorithm="test")

        assert validate_maze_solvable(maze) is False

    def test_validate_trivial_maze(self):
        """Test validating a trivial maze (start == end)."""
        grid = [[Cell.PATH]]
        maze = Maze(width=1, height=1, grid=grid, start=(0, 0), end=(0, 0), algorithm="test")

        assert validate_maze_solvable(maze) is True


class TestBenchmarkGeneration:
    """Test benchmark suite generation."""

    def test_generate_benchmark_suite(self):
        """Test generating full benchmark suite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mazes = generate_benchmark_suite(output_dir)

            # Should generate 20 mazes total (4 sizes × 5 mazes each)
            assert len(mazes) == 20

            # Check that all mazes are solvable
            for maze in mazes:
                assert validate_maze_solvable(maze)

            # Check that files were created
            maze_files = list(output_dir.glob("*.json"))
            assert len(maze_files) == 20

    def test_benchmark_suite_sizes(self):
        """Test that benchmark suite includes correct sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mazes = generate_benchmark_suite(output_dir)

            # Check size distribution
            size_counts = {}
            for maze in mazes:
                size = (maze.width, maze.height)
                size_counts[size] = size_counts.get(size, 0) + 1

            # Should have 5 mazes of each size (with odd dimensions)
            assert len(size_counts) == 4
            assert all(count == 5 for count in size_counts.values())

    def test_benchmark_suite_algorithms(self):
        """Test that benchmark suite uses different algorithms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mazes = generate_benchmark_suite(output_dir)

            algorithms = {maze.algorithm for maze in mazes}

            # Should use all 3 algorithms
            assert len(algorithms) == 3
            assert "recursive_backtracking" in algorithms
            assert "prims_algorithm" in algorithms
            assert "random_growth" in algorithms
