"""Performance metrics tracking and scoring system."""

import sqlite3
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from experiments.baseline_algorithms import PathfindingResult


@dataclass
class PerformanceMetrics:
    """Performance metrics for a maze-solving attempt."""

    run_id: str
    timestamp: str
    maze_id: str
    maze_size: str  # e.g., "11x11"
    algorithm: str
    success: bool
    path_length: int
    nodes_explored: int
    execution_time: float
    optimal: bool
    hybrid_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsDatabase:
    """SQLite database for storing performance metrics."""

    def __init__(self, db_path: Path):
        """
        Initialize metrics database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    maze_id TEXT NOT NULL,
                    maze_size TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    path_length INTEGER NOT NULL,
                    nodes_explored INTEGER NOT NULL,
                    execution_time REAL NOT NULL,
                    optimal BOOLEAN NOT NULL,
                    hybrid_score REAL
                )
            """)

            # Create baseline statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS baseline_stats (
                    maze_id TEXT PRIMARY KEY,
                    maze_size TEXT NOT NULL,
                    optimal_path_length INTEGER NOT NULL,
                    total_cells INTEGER NOT NULL,
                    bfs_time REAL,
                    dfs_time REAL,
                    astar_time REAL,
                    bfs_nodes INTEGER,
                    dfs_nodes INTEGER,
                    astar_nodes INTEGER,
                    timestamp TEXT NOT NULL
                )
            """)

            conn.commit()

    def insert_metrics(self, metrics: PerformanceMetrics) -> None:
        """
        Insert performance metrics into database.

        Args:
            metrics: Performance metrics to insert
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (
                    run_id, timestamp, maze_id, maze_size, algorithm,
                    success, path_length, nodes_explored, execution_time,
                    optimal, hybrid_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.run_id,
                metrics.timestamp,
                metrics.maze_id,
                metrics.maze_size,
                metrics.algorithm,
                metrics.success,
                metrics.path_length,
                metrics.nodes_explored,
                metrics.execution_time,
                metrics.optimal,
                metrics.hybrid_score,
            ))
            conn.commit()

    def insert_baseline_stats(
        self,
        maze_id: str,
        maze_size: str,
        optimal_path_length: int,
        total_cells: int,
        results: List[PathfindingResult],
    ) -> None:
        """
        Insert baseline statistics for a maze.

        Args:
            maze_id: Maze identifier
            maze_size: Maze size (e.g., "11x11")
            optimal_path_length: Optimal path length
            total_cells: Total cells in maze
            results: List of pathfinding results from baseline algorithms
        """
        # Extract statistics by algorithm
        stats = {}
        for result in results:
            stats[result.algorithm.lower().replace("*", "star")] = {
                "time": result.execution_time,
                "nodes": result.nodes_explored,
            }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO baseline_stats (
                    maze_id, maze_size, optimal_path_length, total_cells,
                    bfs_time, dfs_time, astar_time,
                    bfs_nodes, dfs_nodes, astar_nodes,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                maze_id,
                maze_size,
                optimal_path_length,
                total_cells,
                stats.get("bfs", {}).get("time"),
                stats.get("dfs", {}).get("time"),
                stats.get("astar", {}).get("time"),
                stats.get("bfs", {}).get("nodes"),
                stats.get("dfs", {}).get("nodes"),
                stats.get("astar", {}).get("nodes"),
                datetime.now().isoformat(),
            ))
            conn.commit()

    def get_baseline_stats(self, maze_id: str) -> Optional[Dict[str, Any]]:
        """
        Get baseline statistics for a maze.

        Args:
            maze_id: Maze identifier

        Returns:
            Baseline statistics dictionary or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM baseline_stats WHERE maze_id = ?
            """, (maze_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_metrics_by_maze(self, maze_id: str) -> List[Dict[str, Any]]:
        """
        Get all metrics for a specific maze.

        Args:
            maze_id: Maze identifier

        Returns:
            List of metrics dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM metrics WHERE maze_id = ? ORDER BY timestamp DESC
            """, (maze_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_metrics_by_algorithm(self, algorithm: str) -> List[Dict[str, Any]]:
        """
        Get all metrics for a specific algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            List of metrics dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM metrics WHERE algorithm = ? ORDER BY timestamp DESC
            """, (algorithm,))
            return [dict(row) for row in cursor.fetchall()]

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all runs.

        Returns:
            Summary statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Overall statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_runs,
                    AVG(execution_time) as avg_execution_time,
                    AVG(nodes_explored) as avg_nodes_explored,
                    AVG(hybrid_score) as avg_hybrid_score
                FROM metrics
            """)
            overall = cursor.fetchone()

            # Per-algorithm statistics
            cursor.execute("""
                SELECT
                    algorithm,
                    COUNT(*) as runs,
                    AVG(execution_time) as avg_time,
                    AVG(nodes_explored) as avg_nodes,
                    AVG(hybrid_score) as avg_score,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
                FROM metrics
                GROUP BY algorithm
            """)
            per_algorithm = cursor.fetchall()

            return {
                "total_runs": overall[0],
                "successful_runs": overall[1],
                "success_rate": overall[1] / overall[0] if overall[0] > 0 else 0,
                "avg_execution_time": overall[2],
                "avg_nodes_explored": overall[3],
                "avg_hybrid_score": overall[4],
                "per_algorithm": [
                    {
                        "algorithm": row[0],
                        "runs": row[1],
                        "avg_time": row[2],
                        "avg_nodes": row[3],
                        "avg_score": row[4],
                        "successes": row[5],
                        "success_rate": row[5] / row[1] if row[1] > 0 else 0,
                    }
                    for row in per_algorithm
                ],
            }


class HybridScorer:
    """Calculate hybrid performance scores."""

    @staticmethod
    def calculate_score(
        result: PathfindingResult,
        baseline_optimal_length: int,
        baseline_mean_time: float,
        total_cells: int,
        accuracy_weight: float = 0.6,
        efficiency_weight: float = 0.4,
    ) -> float:
        """
        Calculate hybrid performance score.

        Formula:
            Hybrid Score = (accuracy_weight * accuracy_score) + (efficiency_weight * efficiency_score)

            accuracy_score = (1 - path_deviation) * (1 - dead_end_ratio)
            path_deviation = (path_length - optimal_length) / optimal_length
            dead_end_ratio = dead_ends_visited / total_cells

            efficiency_score = baseline_time / actual_time

        Args:
            result: Pathfinding result
            baseline_optimal_length: Optimal path length from baseline
            baseline_mean_time: Mean execution time from baseline
            total_cells: Total cells in maze
            accuracy_weight: Weight for accuracy component (default 0.6)
            efficiency_weight: Weight for efficiency component (default 0.4)

        Returns:
            Hybrid score (higher is better)
        """
        if not result.success:
            return 0.0

        # Calculate path deviation
        path_deviation = (result.path_length - baseline_optimal_length) / baseline_optimal_length
        path_deviation = max(0.0, path_deviation)  # Clamp to [0, inf)

        # Calculate dead-end ratio (nodes explored but not in path)
        dead_ends = max(0, result.nodes_explored - result.path_length)
        dead_end_ratio = dead_ends / total_cells

        # Calculate accuracy score
        accuracy_score = (1.0 - min(1.0, path_deviation)) * (1.0 - dead_end_ratio)

        # Calculate efficiency score
        if result.execution_time > 0:
            efficiency_score = baseline_mean_time / result.execution_time
        else:
            efficiency_score = 1.0

        # Calculate hybrid score
        hybrid_score = (accuracy_weight * accuracy_score) + (efficiency_weight * efficiency_score)

        return max(0.0, hybrid_score)  # Ensure non-negative


def generate_report(db: MetricsDatabase, output_path: Path) -> None:
    """
    Generate performance report.

    Args:
        db: Metrics database
        output_path: Path to save report
    """
    summary = db.get_summary_statistics()

    report = ["# Swarm Intelligence Performance Report", ""]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## Overall Statistics")
    report.append(f"- Total Runs: {summary['total_runs']}")
    report.append(f"- Successful Runs: {summary['successful_runs']}")
    report.append(f"- Success Rate: {summary['success_rate']:.1%}")

    if summary['avg_execution_time']:
        report.append(f"- Average Execution Time: {summary['avg_execution_time']*1000:.3f}ms")
    if summary['avg_nodes_explored']:
        report.append(f"- Average Nodes Explored: {summary['avg_nodes_explored']:.1f}")
    if summary['avg_hybrid_score']:
        report.append(f"- Average Hybrid Score: {summary['avg_hybrid_score']:.3f}")
    report.append("")

    report.append("## Per-Algorithm Statistics")
    report.append("")
    for algo in summary['per_algorithm']:
        report.append(f"### {algo['algorithm']}")
        report.append(f"- Runs: {algo['runs']}")
        report.append(f"- Success Rate: {algo['success_rate']:.1%}")
        if algo['avg_time']:
            report.append(f"- Average Time: {algo['avg_time']*1000:.3f}ms")
        if algo['avg_nodes']:
            report.append(f"- Average Nodes: {algo['avg_nodes']:.1f}")
        if algo['avg_score']:
            report.append(f"- Average Score: {algo['avg_score']:.3f}")
        report.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report))
