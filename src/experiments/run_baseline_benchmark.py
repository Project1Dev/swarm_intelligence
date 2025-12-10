"""Run baseline algorithms on benchmark mazes and collect metrics."""

import argparse
from pathlib import Path
from datetime import datetime
import uuid

from utils.maze_generator import Maze
from experiments.baseline_algorithms import run_all_algorithms
from utils.metrics import MetricsDatabase, PerformanceMetrics, HybridScorer, generate_report
from utils.logging_config import setup_logging, get_logger


def run_benchmark(
    benchmark_dir: Path,
    metrics_db_path: Path,
    output_report: Path,
    log_level: str = "INFO",
) -> None:
    """
    Run baseline benchmark on all mazes.

    Args:
        benchmark_dir: Directory containing benchmark mazes
        metrics_db_path: Path to metrics database
        output_report: Path to save performance report
        log_level: Logging level
    """
    # Setup logging
    setup_logging(log_level=getattr(__import__("logging"), log_level))
    logger = get_logger(__name__)

    logger.info("Starting baseline benchmark")
    logger.info(f"Benchmark directory: {benchmark_dir}")
    logger.info(f"Metrics database: {metrics_db_path}")

    # Initialize database
    db = MetricsDatabase(metrics_db_path)
    scorer = HybridScorer()

    # Get all maze files
    maze_files = sorted(benchmark_dir.glob("*.json"))
    logger.info(f"Found {len(maze_files)} mazes")

    # Generate run ID
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()

    # Run algorithms on each maze
    for i, maze_file in enumerate(maze_files, 1):
        logger.info(f"\n[{i}/{len(maze_files)}] Processing {maze_file.name}")

        # Load maze
        maze = Maze.load(maze_file)
        maze_id = maze_file.stem
        maze_size = f"{maze.width}x{maze.height}"
        total_cells = maze.width * maze.height

        # Run all algorithms
        results = run_all_algorithms(maze)

        # Find optimal path length and mean time
        optimal_results = [r for r in results if r.optimal and r.success]
        if optimal_results:
            optimal_path_length = min(r.path_length for r in optimal_results)
            baseline_mean_time = sum(r.execution_time for r in results if r.success) / len(
                [r for r in results if r.success]
            )
        else:
            optimal_path_length = min(r.path_length for r in results if r.success) if any(
                r.success for r in results
            ) else 0
            baseline_mean_time = sum(r.execution_time for r in results) / len(results)

        # Store baseline stats
        db.insert_baseline_stats(
            maze_id=maze_id,
            maze_size=maze_size,
            optimal_path_length=optimal_path_length,
            total_cells=total_cells,
            results=results,
        )

        # Store individual results and calculate scores
        for result in results:
            # Calculate hybrid score
            hybrid_score = scorer.calculate_score(
                result=result,
                baseline_optimal_length=optimal_path_length,
                baseline_mean_time=baseline_mean_time,
                total_cells=total_cells,
            )

            # Create metrics
            metrics = PerformanceMetrics(
                run_id=run_id,
                timestamp=timestamp,
                maze_id=maze_id,
                maze_size=maze_size,
                algorithm=result.algorithm,
                success=result.success,
                path_length=result.path_length,
                nodes_explored=result.nodes_explored,
                execution_time=result.execution_time,
                optimal=result.optimal,
                hybrid_score=hybrid_score,
            )

            # Store metrics
            db.insert_metrics(metrics)

            logger.info(
                f"  {result.algorithm}: "
                f"{'✓' if result.success else '✗'} "
                f"path={result.path_length}, "
                f"nodes={result.nodes_explored}, "
                f"time={result.execution_time*1000:.3f}ms, "
                f"score={hybrid_score:.3f}"
            )

    # Generate report
    logger.info(f"\nGenerating report: {output_report}")
    generate_report(db, output_report)

    # Print summary
    summary = db.get_summary_statistics()
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total Runs: {summary['total_runs']}")
    logger.info(f"Success Rate: {summary['success_rate']:.1%}")
    logger.info(f"Average Execution Time: {summary['avg_execution_time']*1000:.3f}ms")
    logger.info(f"Average Hybrid Score: {summary['avg_hybrid_score']:.3f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run baseline benchmark")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/benchmarks"),
        help="Directory containing benchmark mazes",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/metrics.db"),
        help="Path to metrics database",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/results/baseline_report.md"),
        help="Path to output report",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Create output directory
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    run_benchmark(
        benchmark_dir=args.benchmark_dir,
        metrics_db_path=args.db,
        output_report=args.report,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
