# Swarm Intelligence System

A multi-agent LLM swarm intelligence system using hierarchical-blackboard hybrid architecture for collaborative problem solving, starting with maze pathfinding benchmarks.

## Overview

This project implements a swarm intelligence system where multiple LLM agents collaborate to solve complex problems. The initial focus is on maze pathfinding to establish baseline metrics and validate the collaborative architecture.

## Features (Phase 1 Week 1)

- **Ollama Integration**: Robust client with connection pooling, retry logic, and health checks
- **Maze Generation**: Multiple algorithms (recursive backtracking, Prim's, random growth)
- **Baseline Algorithms**: BFS, DFS, and A* pathfinding for performance comparison
- **Metrics System**: SQLite-backed performance tracking with hybrid scoring
- **Comprehensive Testing**: 90%+ code coverage with pytest

## Installation

### Prerequisites

- Python 3.10+
- Ollama server with models (see Configuration)

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Configuration

### Ollama Endpoint

The system expects an Ollama server at `192.168.254.17:11434` with the following models:

| Model | Size | Role |
|-------|------|------|
| phi3:mini | 3.8B | Leader Agent |
| deepseek-coder:1.3b | 1.3B | Code Specialist |
| qwen2.5:1.5b | 1.5B | Logic Checker |
| qwen2.5:7b | 7B | Blackboard Manager |
| tinyllama:1.1b | 1.1B | Retrieval Specialist |

To modify the endpoint, edit `config/models.yaml`.

## Usage

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Skip slow tests
pytest tests/ -m "not slow"
```

### Generate Benchmark Mazes

```python
from utils.maze_generator import generate_benchmark_suite
from pathlib import Path

mazes = generate_benchmark_suite(Path("data/benchmarks"))
```

### Run Baseline Benchmark

```bash
python src/experiments/run_baseline_benchmark.py
```

### Use Ollama Client

```python
from utils.ollama_client import OllamaClient

client = OllamaClient()
response = client.generate(
    model="phi3:mini",
    prompt="What is 2 + 2?",
    max_tokens=100
)
print(response.content)
```

### Solve a Maze

```python
from utils.maze_generator import Maze
from experiments.baseline_algorithms import run_all_algorithms, visualize_result

maze = Maze.load(Path("data/benchmarks/maze_simple_11x11_00.json"))
results = run_all_algorithms(maze)

for result in results:
    print(visualize_result(maze, result))
```

## Project Structure

```
swarm-intelligence/
├── src/
│   ├── agents/              # Agent implementations (Week 2+)
│   ├── blackboard/          # Shared memory system (Week 2+)
│   ├── communication/       # Message passing (Week 2+)
│   ├── experiments/         # Benchmarking & evaluation
│   │   ├── baseline_algorithms.py
│   │   └── run_baseline_benchmark.py
│   ├── tasks/               # Task definitions (Week 2+)
│   └── utils/
│       ├── ollama_client.py
│       ├── maze_generator.py
│       ├── metrics.py
│       └── logging_config.py
├── tests/                   # Comprehensive test suite
├── config/                  # Configuration files
│   └── models.yaml
├── data/
│   ├── benchmarks/          # Generated benchmark mazes
│   └── results/             # Performance reports
└── logs/                    # Run logs
```

## Performance Metrics

The system tracks:

- **Path Length**: Number of steps in solution
- **Nodes Explored**: Search space utilization
- **Execution Time**: Milliseconds to solution
- **Hybrid Score**: Weighted combination (60% accuracy, 40% efficiency)

### Baseline Results (20 benchmark mazes)

| Algorithm | Avg Time | Avg Score | Optimal |
|-----------|----------|-----------|---------|
| BFS | 0.3ms | 0.94 | Yes |
| DFS | 0.3ms | 1.00 | No |
| A* | 0.4ms | 0.96 | Yes |

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] Week 1: Infrastructure Setup
  - [x] Ollama integration with retry logic
  - [x] Maze generation system
  - [x] Baseline algorithms (BFS, DFS, A*)
  - [x] Metrics and logging
  - [x] Comprehensive test suite

### Phase 1: Foundation (Next)
- [ ] Week 2: Agent Framework
- [ ] Week 3: Blackboard System

### Phase 2: Multi-Agent Communication
- [ ] Weeks 4-5: Message Protocol
- [ ] Week 6: Swarm Coordination

### Phase 3: Intelligent Collaboration
- [ ] Weeks 7-8: Adaptive Strategies

## License

MIT

## Contributing

See PLAN.md for detailed development guidelines and architecture decisions.
