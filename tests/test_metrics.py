"""Unit tests for metrics and scoring system."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from utils.metrics import (
    MetricsDatabase,
    PerformanceMetrics,
    HybridScorer,
    generate_report,
)
from experiments.baseline_algorithms import PathfindingResult


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_metrics.db"
        yield MetricsDatabase(db_path)


@pytest.fixture
def sample_metrics():
    """Create sample metrics for testing."""
    return PerformanceMetrics(
        run_id="test-run-123",
        timestamp=datetime.now().isoformat(),
        maze_id="maze_simple_11x11_00",
        maze_size="11x11",
        algorithm="BFS",
        success=True,
        path_length=12,
        nodes_explored=25,
        execution_time=0.001,
        optimal=True,
        hybrid_score=0.95,
    )


@pytest.fixture
def sample_results():
    """Create sample pathfinding results."""
    return [
        PathfindingResult(
            algorithm="BFS",
            success=True,
            path=[(0, 0), (0, 1), (1, 1)],
            path_length=3,
            nodes_explored=5,
            execution_time=0.001,
            optimal=True,
        ),
        PathfindingResult(
            algorithm="DFS",
            success=True,
            path=[(0, 0), (1, 0), (1, 1)],
            path_length=3,
            nodes_explored=4,
            execution_time=0.0008,
            optimal=False,
        ),
        PathfindingResult(
            algorithm="A*",
            success=True,
            path=[(0, 0), (0, 1), (1, 1)],
            path_length=3,
            nodes_explored=3,
            execution_time=0.0012,
            optimal=True,
        ),
    ]


class TestPerformanceMetrics:
    """Test PerformanceMetrics data class."""

    def test_metrics_creation(self, sample_metrics):
        """Test creating performance metrics."""
        assert sample_metrics.run_id == "test-run-123"
        assert sample_metrics.algorithm == "BFS"
        assert sample_metrics.success is True
        assert sample_metrics.path_length == 12
        assert sample_metrics.nodes_explored == 25

    def test_metrics_to_dict(self, sample_metrics):
        """Test converting metrics to dictionary."""
        data = sample_metrics.to_dict()

        assert data["run_id"] == "test-run-123"
        assert data["algorithm"] == "BFS"
        assert data["success"] is True
        assert data["path_length"] == 12
        assert data["hybrid_score"] == 0.95


class TestMetricsDatabase:
    """Test MetricsDatabase operations."""

    def test_database_initialization(self, temp_db):
        """Test database is initialized correctly."""
        assert temp_db.db_path.exists()

    def test_insert_metrics(self, temp_db, sample_metrics):
        """Test inserting metrics."""
        temp_db.insert_metrics(sample_metrics)

        # Retrieve and verify
        metrics = temp_db.get_metrics_by_maze("maze_simple_11x11_00")
        assert len(metrics) == 1
        assert metrics[0]["algorithm"] == "BFS"
        assert metrics[0]["path_length"] == 12

    def test_insert_multiple_metrics(self, temp_db):
        """Test inserting multiple metrics."""
        for i in range(5):
            metrics = PerformanceMetrics(
                run_id=f"run-{i}",
                timestamp=datetime.now().isoformat(),
                maze_id="test_maze",
                maze_size="10x10",
                algorithm="BFS" if i % 2 == 0 else "DFS",
                success=True,
                path_length=10 + i,
                nodes_explored=20 + i,
                execution_time=0.001 * (i + 1),
                optimal=True,
                hybrid_score=0.9,
            )
            temp_db.insert_metrics(metrics)

        all_metrics = temp_db.get_metrics_by_maze("test_maze")
        assert len(all_metrics) == 5

    def test_insert_baseline_stats(self, temp_db, sample_results):
        """Test inserting baseline statistics."""
        temp_db.insert_baseline_stats(
            maze_id="test_maze",
            maze_size="5x5",
            optimal_path_length=3,
            total_cells=25,
            results=sample_results,
        )

        stats = temp_db.get_baseline_stats("test_maze")
        assert stats is not None
        assert stats["maze_size"] == "5x5"
        assert stats["optimal_path_length"] == 3
        assert stats["total_cells"] == 25

    def test_get_baseline_stats_not_found(self, temp_db):
        """Test getting baseline stats for non-existent maze."""
        stats = temp_db.get_baseline_stats("nonexistent")
        assert stats is None

    def test_get_metrics_by_algorithm(self, temp_db):
        """Test getting metrics by algorithm."""
        # Insert metrics for different algorithms
        for algo in ["BFS", "BFS", "DFS", "A*"]:
            metrics = PerformanceMetrics(
                run_id=f"run-{algo}",
                timestamp=datetime.now().isoformat(),
                maze_id="test_maze",
                maze_size="10x10",
                algorithm=algo,
                success=True,
                path_length=10,
                nodes_explored=20,
                execution_time=0.001,
                optimal=True,
                hybrid_score=0.9,
            )
            temp_db.insert_metrics(metrics)

        bfs_metrics = temp_db.get_metrics_by_algorithm("BFS")
        assert len(bfs_metrics) == 2

        dfs_metrics = temp_db.get_metrics_by_algorithm("DFS")
        assert len(dfs_metrics) == 1

    def test_get_summary_statistics(self, temp_db):
        """Test getting summary statistics."""
        # Insert some test data
        for i in range(10):
            metrics = PerformanceMetrics(
                run_id=f"run-{i}",
                timestamp=datetime.now().isoformat(),
                maze_id=f"maze_{i}",
                maze_size="10x10",
                algorithm="BFS" if i < 5 else "DFS",
                success=i < 8,  # 2 failures
                path_length=10 if i < 8 else 0,
                nodes_explored=20,
                execution_time=0.001,
                optimal=True,
                hybrid_score=0.9 if i < 8 else 0.0,
            )
            temp_db.insert_metrics(metrics)

        summary = temp_db.get_summary_statistics()

        assert summary["total_runs"] == 10
        assert summary["successful_runs"] == 8
        assert summary["success_rate"] == 0.8
        assert len(summary["per_algorithm"]) == 2

    def test_empty_summary_statistics(self, temp_db):
        """Test summary statistics with empty database."""
        summary = temp_db.get_summary_statistics()

        assert summary["total_runs"] == 0
        assert summary["success_rate"] == 0


class TestHybridScorer:
    """Test HybridScorer calculations."""

    def test_perfect_score(self):
        """Test scoring a perfect result."""
        result = PathfindingResult(
            algorithm="BFS",
            success=True,
            path=[(0, 0), (0, 1), (1, 1)],
            path_length=3,
            nodes_explored=3,  # No dead ends
            execution_time=0.001,
            optimal=True,
        )

        score = HybridScorer.calculate_score(
            result=result,
            baseline_optimal_length=3,
            baseline_mean_time=0.001,
            total_cells=9,
        )

        # Perfect accuracy (no deviation, no dead ends) + efficiency = 1.0
        assert score >= 0.9

    def test_failed_result_score(self):
        """Test scoring a failed result."""
        result = PathfindingResult(
            algorithm="BFS",
            success=False,
            path=None,
            path_length=0,
            nodes_explored=10,
            execution_time=0.002,
            optimal=False,
        )

        score = HybridScorer.calculate_score(
            result=result,
            baseline_optimal_length=5,
            baseline_mean_time=0.001,
            total_cells=25,
        )

        assert score == 0.0

    def test_suboptimal_path_score(self):
        """Test scoring a suboptimal path."""
        result = PathfindingResult(
            algorithm="DFS",
            success=True,
            path=[(0, 0), (1, 0), (2, 0), (2, 1), (1, 1)],
            path_length=5,  # Longer than optimal
            nodes_explored=10,
            execution_time=0.001,
            optimal=False,
        )

        score = HybridScorer.calculate_score(
            result=result,
            baseline_optimal_length=3,  # Optimal is 3
            baseline_mean_time=0.001,
            total_cells=9,
        )

        # Score should be lower due to path deviation
        assert 0 < score < 1.0

    def test_efficiency_score_component(self):
        """Test efficiency affects score."""
        # Fast result
        fast_result = PathfindingResult(
            algorithm="BFS",
            success=True,
            path=[(0, 0), (0, 1), (1, 1)],
            path_length=3,
            nodes_explored=3,
            execution_time=0.0005,  # Faster than baseline
            optimal=True,
        )

        # Slow result
        slow_result = PathfindingResult(
            algorithm="BFS",
            success=True,
            path=[(0, 0), (0, 1), (1, 1)],
            path_length=3,
            nodes_explored=3,
            execution_time=0.002,  # Slower than baseline
            optimal=True,
        )

        fast_score = HybridScorer.calculate_score(
            result=fast_result,
            baseline_optimal_length=3,
            baseline_mean_time=0.001,
            total_cells=9,
        )

        slow_score = HybridScorer.calculate_score(
            result=slow_result,
            baseline_optimal_length=3,
            baseline_mean_time=0.001,
            total_cells=9,
        )

        # Fast should score higher than slow
        assert fast_score > slow_score

    def test_custom_weights(self):
        """Test custom accuracy/efficiency weights."""
        result = PathfindingResult(
            algorithm="BFS",
            success=True,
            path=[(0, 0), (0, 1), (1, 1)],
            path_length=3,
            nodes_explored=5,
            execution_time=0.001,
            optimal=True,
        )

        # More weight on accuracy
        accuracy_weighted = HybridScorer.calculate_score(
            result=result,
            baseline_optimal_length=3,
            baseline_mean_time=0.001,
            total_cells=9,
            accuracy_weight=0.9,
            efficiency_weight=0.1,
        )

        # More weight on efficiency
        efficiency_weighted = HybridScorer.calculate_score(
            result=result,
            baseline_optimal_length=3,
            baseline_mean_time=0.001,
            total_cells=9,
            accuracy_weight=0.1,
            efficiency_weight=0.9,
        )

        # Both should be valid scores
        assert 0 <= accuracy_weighted <= 2
        assert 0 <= efficiency_weighted <= 2


class TestReportGeneration:
    """Test report generation."""

    def test_generate_report(self, temp_db):
        """Test generating a performance report."""
        # Insert some test data
        for i in range(5):
            metrics = PerformanceMetrics(
                run_id=f"run-{i}",
                timestamp=datetime.now().isoformat(),
                maze_id=f"maze_{i}",
                maze_size="10x10",
                algorithm="BFS",
                success=True,
                path_length=10,
                nodes_explored=20,
                execution_time=0.001,
                optimal=True,
                hybrid_score=0.9,
            )
            temp_db.insert_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "test_report.md"
            generate_report(temp_db, report_path)

            assert report_path.exists()

            content = report_path.read_text()
            assert "# Swarm Intelligence Performance Report" in content
            assert "Overall Statistics" in content
            assert "Per-Algorithm Statistics" in content
            assert "BFS" in content
