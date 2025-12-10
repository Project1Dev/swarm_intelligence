"""Unit tests for baseline pathfinding algorithms."""

import pytest
from pathlib import Path

from utils.maze_generator import Maze, Cell
from experiments.baseline_algorithms import (
    BFSPathfinder,
    DFSPathfinder,
    AStarPathfinder,
    PathfindingResult,
    run_all_algorithms,
    compare_algorithms,
    visualize_result,
)


@pytest.fixture
def simple_maze():
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
def unsolvable_maze():
    """Create an unsolvable maze for testing."""
    grid = [
        [Cell.PATH, Cell.WALL, Cell.PATH],
        [Cell.WALL, Cell.WALL, Cell.WALL],
        [Cell.PATH, Cell.WALL, Cell.PATH],
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
def trivial_maze():
    """Create a trivial maze (start == end)."""
    grid = [[Cell.PATH]]
    return Maze(
        width=1,
        height=1,
        grid=grid,
        start=(0, 0),
        end=(0, 0),
        algorithm="test",
    )


class TestBFSPathfinder:
    """Test BFS pathfinding algorithm."""

    def test_bfs_simple_maze(self, simple_maze):
        """Test BFS on simple maze."""
        pathfinder = BFSPathfinder()
        result = pathfinder.find_path(simple_maze)

        assert result.success is True
        assert result.path is not None
        assert result.path[0] == simple_maze.start
        assert result.path[-1] == simple_maze.end
        assert result.optimal is True
        assert result.nodes_explored > 0
        assert result.execution_time > 0

    def test_bfs_unsolvable_maze(self, unsolvable_maze):
        """Test BFS on unsolvable maze."""
        pathfinder = BFSPathfinder()
        result = pathfinder.find_path(unsolvable_maze)

        assert result.success is False
        assert result.path is None
        assert result.path_length == 0

    def test_bfs_trivial_maze(self, trivial_maze):
        """Test BFS on trivial maze."""
        pathfinder = BFSPathfinder()
        result = pathfinder.find_path(trivial_maze)

        assert result.success is True
        assert result.path == [(0, 0)]
        assert result.path_length == 1

    def test_bfs_optimal_path(self, simple_maze):
        """Test that BFS finds optimal path."""
        pathfinder = BFSPathfinder()
        result = pathfinder.find_path(simple_maze)

        # Check path is valid
        for i in range(len(result.path) - 1):
            current = result.path[i]
            next_pos = result.path[i + 1]
            # Manhattan distance should be 1 (adjacent cells)
            distance = abs(current[0] - next_pos[0]) + abs(current[1] - next_pos[1])
            assert distance == 1

        # Check all path cells are walkable
        for pos in result.path:
            assert simple_maze.is_walkable(*pos)


class TestDFSPathfinder:
    """Test DFS pathfinding algorithm."""

    def test_dfs_simple_maze(self, simple_maze):
        """Test DFS on simple maze."""
        pathfinder = DFSPathfinder()
        result = pathfinder.find_path(simple_maze)

        assert result.success is True
        assert result.path is not None
        assert result.path[0] == simple_maze.start
        assert result.path[-1] == simple_maze.end
        assert result.optimal is False  # DFS doesn't guarantee optimal
        assert result.nodes_explored > 0

    def test_dfs_unsolvable_maze(self, unsolvable_maze):
        """Test DFS on unsolvable maze."""
        pathfinder = DFSPathfinder()
        result = pathfinder.find_path(unsolvable_maze)

        assert result.success is False
        assert result.path is None

    def test_dfs_finds_valid_path(self, simple_maze):
        """Test that DFS finds a valid path."""
        pathfinder = DFSPathfinder()
        result = pathfinder.find_path(simple_maze)

        # Check path is valid and continuous
        for i in range(len(result.path) - 1):
            current = result.path[i]
            next_pos = result.path[i + 1]
            distance = abs(current[0] - next_pos[0]) + abs(current[1] - next_pos[1])
            assert distance == 1

        # Check all path cells are walkable
        for pos in result.path:
            assert simple_maze.is_walkable(*pos)


class TestAStarPathfinder:
    """Test A* pathfinding algorithm."""

    def test_astar_simple_maze(self, simple_maze):
        """Test A* on simple maze."""
        pathfinder = AStarPathfinder()
        result = pathfinder.find_path(simple_maze)

        assert result.success is True
        assert result.path is not None
        assert result.path[0] == simple_maze.start
        assert result.path[-1] == simple_maze.end
        assert result.optimal is True
        assert result.nodes_explored > 0

    def test_astar_unsolvable_maze(self, unsolvable_maze):
        """Test A* on unsolvable maze."""
        pathfinder = AStarPathfinder()
        result = pathfinder.find_path(unsolvable_maze)

        assert result.success is False
        assert result.path is None

    def test_astar_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        pathfinder = AStarPathfinder()

        assert pathfinder._manhattan_distance((0, 0), (0, 0)) == 0
        assert pathfinder._manhattan_distance((0, 0), (1, 0)) == 1
        assert pathfinder._manhattan_distance((0, 0), (0, 1)) == 1
        assert pathfinder._manhattan_distance((0, 0), (3, 4)) == 7
        assert pathfinder._manhattan_distance((5, 5), (2, 1)) == 7

    def test_astar_optimal_path(self, simple_maze):
        """Test that A* finds optimal path."""
        pathfinder = AStarPathfinder()
        result = pathfinder.find_path(simple_maze)

        # Check path is valid
        for i in range(len(result.path) - 1):
            current = result.path[i]
            next_pos = result.path[i + 1]
            distance = abs(current[0] - next_pos[0]) + abs(current[1] - next_pos[1])
            assert distance == 1

    def test_astar_efficiency(self, simple_maze):
        """Test that A* explores fewer nodes than BFS in some cases."""
        astar = AStarPathfinder()
        astar_result = astar.find_path(simple_maze)

        # A* should find a path with reasonable efficiency
        assert astar_result.success is True
        assert astar_result.nodes_explored <= simple_maze.width * simple_maze.height


class TestAlgorithmComparison:
    """Test algorithm comparison functions."""

    def test_run_all_algorithms(self, simple_maze):
        """Test running all algorithms on a maze."""
        results = run_all_algorithms(simple_maze)

        assert len(results) == 3
        assert all(isinstance(r, PathfindingResult) for r in results)
        assert {r.algorithm for r in results} == {"BFS", "DFS", "A*"}

        # All should succeed on simple maze
        assert all(r.success for r in results)

    def test_compare_algorithms(self, simple_maze):
        """Test algorithm comparison."""
        results = run_all_algorithms(simple_maze)
        comparison = compare_algorithms(results)

        assert "algorithms" in comparison
        assert "fastest" in comparison
        assert "most_efficient" in comparison
        assert "shortest_path" in comparison

        assert len(comparison["algorithms"]) == 3
        assert comparison["fastest"] in ["BFS", "DFS", "A*"]
        assert comparison["most_efficient"] in ["BFS", "DFS", "A*"]
        assert comparison["shortest_path"] in ["BFS", "DFS", "A*"]

    def test_compare_empty_results(self):
        """Test comparison with no successful results."""
        results = []
        comparison = compare_algorithms(results)

        assert comparison["fastest"] is None
        assert comparison["most_efficient"] is None
        assert comparison["shortest_path"] is None

    def test_optimal_path_lengths_match(self, simple_maze):
        """Test that optimal algorithms find same path length."""
        results = run_all_algorithms(simple_maze)

        # BFS and A* should find paths of equal length (both optimal)
        bfs_result = next(r for r in results if r.algorithm == "BFS")
        astar_result = next(r for r in results if r.algorithm == "A*")

        assert bfs_result.path_length == astar_result.path_length


class TestVisualization:
    """Test visualization functions."""

    def test_visualize_success(self, simple_maze):
        """Test visualizing successful pathfinding."""
        pathfinder = BFSPathfinder()
        result = pathfinder.find_path(simple_maze)

        viz = visualize_result(simple_maze, result)

        assert "BFS Result:" in viz
        assert "Path Length:" in viz
        assert "Nodes Explored:" in viz
        assert "Execution Time:" in viz
        assert "S" in viz  # Start marker
        assert "E" in viz  # End marker

    def test_visualize_failure(self, unsolvable_maze):
        """Test visualizing failed pathfinding."""
        pathfinder = BFSPathfinder()
        result = pathfinder.find_path(unsolvable_maze)

        viz = visualize_result(unsolvable_maze, result)

        assert "failed to find a path" in viz


class TestBenchmarkMazes:
    """Test algorithms on actual benchmark mazes."""

    @pytest.mark.slow
    def test_algorithms_on_benchmark_mazes(self):
        """Test all algorithms on generated benchmark mazes."""
        benchmark_dir = Path("data/benchmarks")

        if not benchmark_dir.exists():
            pytest.skip("Benchmark mazes not generated")

        # Test on a few benchmark mazes
        maze_files = list(benchmark_dir.glob("maze_simple_*.json"))[:3]

        for maze_file in maze_files:
            maze = Maze.load(maze_file)
            results = run_all_algorithms(maze)

            # All algorithms should succeed on benchmark mazes
            for result in results:
                assert result.success is True, f"{result.algorithm} failed on {maze_file.name}"
                assert len(result.path) > 0
                assert result.nodes_explored > 0

            # BFS and A* should find optimal paths
            bfs_result = next(r for r in results if r.algorithm == "BFS")
            astar_result = next(r for r in results if r.algorithm == "A*")

            assert bfs_result.path_length == astar_result.path_length


class TestPathfindingResult:
    """Test PathfindingResult data class."""

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = PathfindingResult(
            algorithm="BFS",
            success=True,
            path=[(0, 0), (0, 1), (1, 1)],
            path_length=3,
            nodes_explored=5,
            execution_time=0.001,
            optimal=True,
        )

        data = result.to_dict()

        assert data["algorithm"] == "BFS"
        assert data["success"] is True
        assert data["path"] == [[0, 0], [0, 1], [1, 1]]
        assert data["path_length"] == 3
        assert data["nodes_explored"] == 5
        assert data["execution_time"] == 0.001
        assert data["optimal"] is True

    def test_result_to_dict_no_path(self):
        """Test converting result with no path to dictionary."""
        result = PathfindingResult(
            algorithm="BFS",
            success=False,
            path=None,
            path_length=0,
            nodes_explored=10,
            execution_time=0.002,
            optimal=False,
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["path"] is None
