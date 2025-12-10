# Swarm Intelligence System - Comprehensive Development Plan

## Executive Summary

This document outlines a structured, phase-based approach to building a state-of-the-art multi-agent LLM swarm intelligence system. The project begins with a measurable maze-solving benchmark and scales toward general-purpose swarm intelligence capable of tackling complex, undefined problems through emergent collaboration.

**Core Philosophy**: Build incrementally from first principles, validate at each phase, and iterate based on empirical results.

---

## Table of Contents

1. [Project Vision & Goals](#project-vision--goals)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1: Foundation (Weeks 1-3)](#phase-1-foundation-weeks-1-3)
4. [Phase 2: Basic Swarm (Weeks 4-6)](#phase-2-basic-swarm-weeks-4-6)
5. [Phase 3: Advanced Coordination (Weeks 7-9)](#phase-3-advanced-coordination-weeks-7-9)
6. [Phase 4: Optimization & Intelligence (Weeks 10-12)](#phase-4-optimization--intelligence-weeks-10-12)
7. [Phase 5: Advanced Features (Future)](#phase-5-advanced-features-future)
8. [Technical Specifications](#technical-specifications)
9. [Research Insights & Best Practices](#research-insights--best-practices)
10. [Testing & Validation](#testing--validation)

---

## Project Vision & Goals

### Primary Objective
Create a **hierarchical-blackboard hybrid swarm** that demonstrates measurable intelligence through collaborative problem-solving, starting with maze pathfinding and scaling to general-purpose problem solving.

### Success Criteria
- **Measurability**: Quantifiable performance metrics (accuracy, efficiency, convergence)
- **Scalability**: System adapts from simple to complex problems
- **Emergent Intelligence**: Collective performance exceeds individual agent capabilities
- **Reproducibility**: Consistent results across runs with documented variance

### Key Differentiators
Based on latest research (2024-2025):
- **Dynamic agent selection** based on blackboard state (not pre-defined workflows)
- **Event-driven communication** for resilience and scalability
- **Hybrid architecture** combining hierarchical control with decentralized swarm behavior
- **Real-time adaptation** without supervised training overhead

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    CONTROL LAYER                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Leader Agent (Phi-3 Mini 3.8B)                  │  │
│  │  - Strategic planning & task decomposition       │  │
│  │  - Agent selection & orchestration               │  │
│  │  - Progress monitoring & replanning              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────┐
│                 BLACKBOARD MEMORY                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Shared Knowledge Base (Qwen2.5 0.5B)           │  │
│  │  - Current goals & constraints                   │  │
│  │  - Agent communications (priority queues)        │  │
│  │  - Solutions & hypotheses                        │  │
│  │  - Event log & metrics                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────┐
│                 SPECIALIST LAYER                        │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────┐   │
│  │   Code    │  │   Logic   │  │    Retrieval     │   │
│  │ Specialist│  │  Checker  │  │    Specialist    │   │
│  │(DeepSeek) │  │ (Qwen1.5B)│  │  (TinyLlama)     │   │
│  └───────────┘  └───────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Information Flow
1. **Leader** reads blackboard, selects agents, decomposes tasks
2. **Specialists** read relevant sections, execute, write results
3. **Blackboard** aggregates knowledge, maintains state
4. **Event system** triggers actions based on state changes

---

## Phase 1: Foundation (Weeks 1-3)

### Goal
Establish core infrastructure and validate individual components before integration.

### Week 1: Infrastructure Setup

#### Task 1.1: Environment Configuration
- [ ] Set up Python 3.10+ development environment
- [ ] Configure Ollama endpoint (192.168.254.17)
- [ ] Test model loading and inference
  - GPU: Phi-3 Mini (3.8B), DeepSeekCoder (1.3B), Qwen (1.5B)
  - CPU: Qwen (0.5B), TinyLlama (1.1B)
- [ ] Verify VRAM/RAM allocation (11GB GPU, 32GB RAM target)
- [ ] Create project structure:
  ```
  swarm-intelligence/
  ├── src/
  │   ├── agents/          # Agent implementations
  │   ├── blackboard/      # Shared memory system
  │   ├── communication/   # Message passing
  │   ├── tasks/           # Task definitions
  │   └── utils/           # Helpers
  ├── tests/
  ├── config/              # Model configs, hyperparameters
  ├── data/                # Mazes, benchmarks
  └── logs/                # Run logs, metrics
  ```

**Validation**: All models respond to test prompts within timeout limits.

#### Task 1.2: Ollama Integration Layer
- [ ] Create `OllamaClient` class with connection pooling
- [ ] Implement request queue with priority handling
- [ ] Add retry logic with exponential backoff (1s, 2s, 4s)
- [ ] Configure timeouts per model size:
  - 3.8B: 30s
  - 1.3B: 20s
  - 0.5-1B: 10s
- [ ] Add health check endpoints
- [ ] Implement graceful degradation on model failure

**Validation**: 100 sequential requests complete with <1% failure rate.

#### Task 1.3: Maze Generation & Representation
- [ ] Implement maze data structure (2D grid with walls)
- [ ] Create maze generator with complexity controls:
  - Recursive backtracking (DFS mazes)
  - Prim's algorithm (minimum spanning tree)
  - Random growth (BFS-like)
- [ ] Generate benchmark suite:
  - 5x 10x10 (simple baseline)
  - 5x 20x20 (medium complexity)
  - 5x 30x30 (complex with cycles)
  - 5x 50x50 (stress test)
- [ ] Implement maze visualization (ASCII + optional GUI)
- [ ] Create maze validation (ensure solvability)

**Validation**: All generated mazes have valid start-to-goal paths.

### Week 2: Base Agent Framework

#### Task 2.1: Agent Base Class
```python
class BaseAgent:
    def __init__(self, name, role, model, permissions, temperature):
        self.name = name
        self.role = role
        self.model = model
        self.permissions = permissions  # Blackboard access rights
        self.temperature = temperature
        self.token_budget = 0  # Per-round budget
        
    def read_blackboard(self, sections: List[str]) -> Dict:
        """Read allowed sections from blackboard"""
        pass
        
    def write_blackboard(self, section: str, content: Any) -> bool:
        """Write to allowed sections"""
        pass
        
    def process_task(self, task: Task, context: Dict) -> Message:
        """Execute task with context"""
        pass
        
    def validate_output(self, output: Any) -> Tuple[bool, str]:
        """Self-validation of output quality"""
        pass
```

**Implementation**:
- [ ] Define agent interface with abstract methods
- [ ] Create agent registry for dynamic instantiation
- [ ] Implement token counting and budget enforcement
- [ ] Add output validation framework
- [ ] Create agent state persistence

**Validation**: Mock agent responds to dummy tasks, respects permissions.

#### Task 2.2: Blackboard System (Basic)
```python
class Blackboard:
    def __init__(self):
        self.data = {
            "current_goal": {},      # 200 tokens max
            "facts": {},             # 500 tokens max
            "hypotheses": {},        # 300 tokens max
            "artifacts": {},         # 5000 tokens max (code)
            "message_queues": {
                "critical": [],      # 1000 tokens max
                "normal": [],        # 2000 tokens max
                "log": []            # 500 tokens max
            },
            "metrics": {},
            "event_log": []          # Last 50 events
        }
        self.locks = {}  # Section-level locks
        self.version = 0
        
    def read(self, sections: List[str], requester: str) -> Dict:
        """Read sections with permission check"""
        pass
        
    def write(self, section: str, key: str, value: Any, 
              requester: str) -> bool:
        """Write with permission check & size limits"""
        pass
        
    def subscribe(self, event_type: str, callback: Callable):
        """Event subscription for reactive agents"""
        pass
        
    def prune(self):
        """Auto-prune when size limits exceeded"""
        pass
```

**Implementation**:
- [ ] Build in-memory data structure with nested dicts
- [ ] Implement size tracking per section (token count)
- [ ] Add auto-pruning when limits exceeded (keep recent/important)
- [ ] Create snapshot system (save every 10s)
- [ ] Implement event subscription system
- [ ] Add thread-safe operations (locks per section)

**Validation**: Concurrent read/writes maintain data integrity.

#### Task 2.3: Message Protocol
```python
@dataclass
class Message:
    from_agent: str
    to_agent: str  # or "broadcast"
    type: MessageType  # PLAN_UPDATE, CODE_DRAFT, WARNING, QUERY
    priority: Priority  # CRITICAL, NORMAL, LOG
    timestamp: float
    content: Dict
    metadata: Dict  # tokens_used, confidence, etc.
    
class MessageQueue:
    def __init__(self, max_tokens: int):
        self.queues = {
            Priority.CRITICAL: deque(),
            Priority.NORMAL: deque(),
            Priority.LOG: deque()
        }
        self.max_tokens = max_tokens
        
    def push(self, message: Message) -> bool:
        """Add message to appropriate priority queue"""
        pass
        
    def pop(self, priority: Priority = None) -> Message:
        """Get next message (highest priority first)"""
        pass
        
    def peek(self, count: int = 1) -> List[Message]:
        """View without removing"""
        pass
```

**Implementation**:
- [ ] Define message dataclass with validation
- [ ] Implement FIFO queues per priority level
- [ ] Add message serialization (JSON)
- [ ] Create message routing logic
- [ ] Implement broadcast vs directed messages
- [ ] Add message expiration (TTL)

**Validation**: Messages route correctly, priorities respected.

### Week 3: Pathfinding Baseline

#### Task 3.1: Solo Algorithm Implementations
- [ ] Implement BFS (breadth-first search)
- [ ] Implement DFS (depth-first search)
- [ ] Implement A* (with Manhattan distance heuristic)
- [ ] Add algorithm execution framework:
  - Step counter
  - Wall-clock timer
  - Nodes explored counter
  - Path reconstruction
- [ ] Create visualization of algorithm execution

**Validation**: All algorithms solve all benchmark mazes correctly.

#### Task 3.2: Baseline Metrics Collection
- [ ] Run each algorithm on benchmark suite (5 mazes × 5 sizes)
- [ ] Collect metrics:
  - Execution time (mean, std, min, max)
  - Path length (optimal = maze-specific)
  - Nodes explored
  - Memory usage
  - Success rate
- [ ] Create performance database (SQLite)
- [ ] Generate baseline report with visualizations

**Key Research Finding**: 
- **A*** typically fastest for large sparse mazes (40-60% faster than BFS)
- **BFS** guarantees shortest path, best for small/medium mazes
- **DFS** can be faster in maze types with long corridors but no optimality guarantee

**Validation**: Statistical significance in performance differences.

#### Task 3.3: Scoring System Implementation
```python
def calculate_hybrid_score(solution: Solution, baseline: Baseline) -> float:
    """
    Hybrid Score = (Path Quality / Path Length) × (Baseline Time / Actual Time)
    
    Path Quality = 1.0 - (dead_ends_visited / total_cells)
    
    Weights: 60% accuracy, 40% efficiency
    """
    accuracy_score = (
        0.4 * (1 - solution.path_length / baseline.optimal_length) +
        0.6 * (1 - solution.dead_ends / baseline.total_cells)
    )
    
    efficiency_score = baseline.mean_time / solution.execution_time
    
    return 0.6 * accuracy_score + 0.4 * efficiency_score
```

**Implementation**:
- [ ] Define scoring formula with configurable weights
- [ ] Implement path quality metrics
- [ ] Calculate efficiency ratio
- [ ] Create comparative scoring across algorithms
- [ ] Add penalty system for failures/timeouts

**Validation**: Scoring system ranks known-good solutions higher.

---

## Phase 2: Basic Swarm (Weeks 4-6)

### Goal
Implement minimal viable swarm with 3 agents cooperating on maze solving.

### Week 4: Leader Agent Implementation

#### Task 4.1: Leader Planning System
```python
class LeaderAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Leader",
            role="Coordinator",
            model="phi-3-mini",
            permissions=["full"],  # Full blackboard access
            temperature=0.4
        )
        
    def decompose_task(self, maze: Maze) -> List[Subtask]:
        """Break maze-solving into subtasks"""
        # Subtasks:
        # 1. Analyze maze structure (Logic Checker)
        # 2. Design algorithm approach (Code Specialist)
        # 3. Implement & optimize (Code Specialist)
        # 4. Validate solution (Logic Checker)
        # 5. Execute & report (Leader)
        pass
        
    def select_agent(self, subtask: Subtask, 
                    blackboard_state: Dict) -> Agent:
        """Dynamic agent selection based on current state"""
        pass
        
    def monitor_progress(self) -> bool:
        """Check if swarm is making progress"""
        pass
        
    def replan(self, failure: Failure) -> Plan:
        """Adapt plan on failure"""
        pass
```

**System Prompt Template**:
```
You are the Leader of a multi-agent swarm solving a maze pathfinding problem.

ROLE: Strategic coordinator responsible for task decomposition, agent 
      orchestration, and overall success.

CAPABILITIES:
- Access entire blackboard
- Select and activate specialist agents
- Replan when failures occur
- Monitor progress and enforce time limits

CURRENT SITUATION:
Maze: {maze_description}
Goal: Find shortest path from start to end
Time remaining: {time_remaining}s
Blackboard state: {blackboard_summary}

INSTRUCTIONS:
1. Analyze the problem
2. Decompose into 3-5 subtasks
3. For each subtask, select the best agent
4. Write plan to blackboard in JSON format
5. Monitor execution, replan if needed

OUTPUT FORMAT:
{
  "analysis": "brief problem analysis",
  "plan": [
    {"task": "...", "agent": "...", "priority": "..."},
    ...
  ],
  "expected_duration": "10s"
}
```

**Implementation**:
- [ ] Create leader agent class with planning logic
- [ ] Implement task decomposition algorithm
- [ ] Add agent selection based on blackboard patterns
- [ ] Create progress monitoring (10s check intervals)
- [ ] Implement replanning on failure (retry once, then reassign)
- [ ] Add few-shot examples to system prompt

**Validation**: Leader generates valid plans for benchmark mazes.

#### Task 4.2: Leader Decision-Making Flow
- [ ] Create decision tree for common scenarios
- [ ] Implement timeout detection (30s max per action)
- [ ] Add deadlock prevention (force progress or abort)
- [ ] Create fallback strategies for agent failures
- [ ] Implement consensus-checking mechanism

**Validation**: Leader adapts to simulated failures within 5s.

### Week 5: Specialist Agent Implementation

#### Task 5.1: Code Specialist Agent
```python
class CodeSpecialist(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="CodeSpecialist",
            role="Implementation",
            model="deepseek-coder",
            permissions=["artifacts", "facts", "message_queues"],
            temperature=0.1  # Deterministic code
        )
        
    def generate_algorithm(self, requirements: Dict) -> str:
        """Generate Python pathfinding code"""
        pass
        
    def optimize_code(self, code: str, constraints: Dict) -> str:
        """Optimize for speed/memory"""
        pass
```

**System Prompt Template**:
```
You are the Code Specialist in a swarm solving maze pathfinding problems.

ROLE: Generate efficient, correct Python implementations of pathfinding 
      algorithms.

TASK: {task_description}
CONSTRAINTS: {constraints}
CONTEXT: {blackboard_facts}

REQUIREMENTS:
- Python 3.10+ only
- Use standard library (no external deps)
- Include type hints
- Optimize for execution speed
- Handle edge cases (no path, cycles, etc.)

AVAILABLE ALGORITHMS: BFS, DFS, A*, Dijkstra

OUTPUT FORMAT:
```python
def solve_maze(maze: List[List[int]], start: Tuple[int, int], 
               end: Tuple[int, int]) -> List[Tuple[int, int]]:
    \"\"\"
    [Docstring explaining approach]
    Time: O(...)
    Space: O(...)
    \"\"\"
    # Implementation
    pass
```

ALSO PROVIDE:
- Brief explanation of chosen algorithm
- Expected complexity analysis
```

**Implementation**:
- [ ] Create code specialist agent class
- [ ] Implement code generation with validation
- [ ] Add syntax checking (AST parsing)
- [ ] Create code optimization prompts
- [ ] Add algorithm selection logic (based on maze properties)
- [ ] Implement few-shot examples for common patterns

**Validation**: Generated code passes unit tests, runs without errors.

#### Task 5.2: Logic Checker Agent
```python
class LogicChecker(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="LogicChecker",
            role="Validation",
            model="qwen-1.5b",
            permissions=["artifacts", "facts", "hypotheses"],
            temperature=0.0  # Strict validation
        )
        
    def validate_algorithm(self, code: str) -> ValidationResult:
        """Check for logical errors, infinite loops, etc."""
        pass
        
    def verify_solution(self, path: List, maze: Maze) -> bool:
        """Verify path is valid and optimal"""
        pass
```

**System Prompt Template**:
```
You are the Logic Checker in a swarm solving pathfinding problems.

ROLE: Validate algorithms and solutions for correctness, completeness, 
      and efficiency.

CODE TO VALIDATE: {code}
MAZE PROPERTIES: {maze_stats}

CHECK FOR:
1. Infinite loop risks (visited set, termination conditions)
2. Boundary checks (array access safety)
3. Algorithm correctness (BFS/DFS/A* properties maintained)
4. Edge case handling (no path, start=end, etc.)
5. Performance issues (redundant operations)

OUTPUT FORMAT:
{
  "valid": true/false,
  "issues": [
    {"severity": "critical/warning", "description": "...", "line": 42}
  ],
  "suggestions": ["..."],
  "confidence": 0.95
}
```

**Implementation**:
- [ ] Create logic checker agent class
- [ ] Implement static analysis checks
- [ ] Add pattern matching for common errors
- [ ] Create validation report generation
- [ ] Implement confidence scoring
- [ ] Add fix suggestions

**Validation**: Catches 90%+ of intentionally introduced bugs.

#### Task 5.3: Retrieval Specialist Agent
```python
class RetrievalSpecialist(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="RetrievalSpecialist",
            role="Knowledge",
            model="tinyllama",
            permissions=["artifacts", "event_log"],
            temperature=0.2
        )
        
    def search_solutions(self, query: Dict) -> List[Solution]:
        """Search blackboard for similar past solutions"""
        pass
        
    def find_patterns(self, maze_type: str) -> List[Pattern]:
        """Identify successful strategies for maze type"""
        pass
```

**System Prompt Template**:
```
You are the Retrieval Specialist in a swarm solving pathfinding problems.

ROLE: Search blackboard history and identify relevant past solutions 
      and patterns.

QUERY: {search_query}
CURRENT MAZE: {maze_description}

SEARCH IN:
- Event log (last 50 events)
- Artifacts (past code solutions)
- Facts (maze-algorithm performance mappings)

LOOK FOR:
- Similar maze structures (size, density, cycles)
- Successful algorithm choices
- Common pitfalls avoided
- Performance patterns

OUTPUT FORMAT:
{
  "relevant_solutions": [
    {"maze_id": "...", "algorithm": "...", "score": 0.95}
  ],
  "patterns": [
    {"pattern": "sparse mazes -> BFS faster", "confidence": 0.8}
  ]
}
```

**Implementation**:
- [ ] Create retrieval specialist agent class
- [ ] Implement similarity search (cosine similarity on maze features)
- [ ] Add pattern extraction from event log
- [ ] Create knowledge base querying
- [ ] Implement relevance scoring

**Validation**: Retrieves correct past solutions for similar mazes.

### Week 6: Swarm Coordination

#### Task 6.1: Orchestration Loop
```python
class SwarmOrchestrator:
    def __init__(self, leader, specialists, blackboard):
        self.leader = leader
        self.specialists = specialists
        self.blackboard = blackboard
        self.running = False
        
    def solve_maze(self, maze: Maze, time_limit: int) -> Solution:
        """Main orchestration loop"""
        # 1. Leader analyzes and creates plan
        plan = self.leader.decompose_task(maze)
        self.blackboard.write("current_goal", "plan", plan, "Leader")
        
        # 2. Execute subtasks sequentially
        for subtask in plan["plan"]:
            agent = self.select_specialist(subtask["agent"])
            
            # Give agent context from blackboard
            context = self.blackboard.read(
                agent.permissions, 
                requester=agent.name
            )
            
            # Agent executes
            result = agent.process_task(subtask, context)
            
            # Write result to blackboard
            self.blackboard.write(
                section=subtask["output_section"],
                key=subtask["task"],
                value=result,
                requester=agent.name
            )
            
            # Check timeout
            if time.time() - start_time > time_limit:
                return self.handle_timeout()
                
        # 3. Extract final solution
        solution = self.blackboard.read(["artifacts"], "Orchestrator")
        return self.validate_solution(solution)
```

**Implementation**:
- [ ] Create orchestrator class managing agent lifecycle
- [ ] Implement sequential task execution
- [ ] Add timeout handling (T_baseline + 30s)
- [ ] Create solution extraction and validation
- [ ] Implement error recovery (retry, reassign, abort)
- [ ] Add comprehensive logging

**Validation**: Swarm solves benchmark mazes end-to-end.

#### Task 6.2: Communication Integration
- [ ] Connect agents to message queues
- [ ] Implement message routing (directed vs broadcast)
- [ ] Add message filtering by priority
- [ ] Create communication logs for debugging
- [ ] Implement message timeout/expiration

**Validation**: Agents successfully exchange messages, no lost messages.

#### Task 6.3: Performance Measurement
- [ ] Instrument every agent action (timing, tokens)
- [ ] Collect per-round metrics
- [ ] Calculate hybrid scores
- [ ] Compare swarm vs solo baselines
- [ ] Generate performance reports

**Expected Results**:
- Swarm should match or slightly beat best solo algorithm
- Token consumption: 2000-3000 per maze
- Execution time: Within 2x baseline for first iteration

**Validation**: Metrics collected accurately, reports generated.

---

## Phase 3: Advanced Coordination (Weeks 7-9)

### Goal
Add dynamic agent selection, parallel execution, and adaptive strategies.

### Week 7: Dynamic Agent Selection

#### Task 7.1: Blackboard State Analysis
```python
class StateAnalyzer:
    def analyze_blackboard(self, blackboard: Blackboard) -> StateVector:
        """Extract features from current blackboard state"""
        features = {
            "messages_pending": len(blackboard.data["message_queues"]),
            "hypotheses_count": len(blackboard.data["hypotheses"]),
            "progress_stalled": self.check_stall(),
            "error_rate": self.calculate_error_rate(),
            "agent_confidence": self.average_confidence()
        }
        return StateVector(features)
        
    def recommend_agent(self, state: StateVector, 
                       task: Task) -> Agent:
        """Select best agent based on current state"""
        # Pattern: If logic errors high -> Logic Checker
        # Pattern: If code missing -> Code Specialist
        # Pattern: If similar problem solved -> Retrieval Specialist
        pass
```

**Implementation**:
- [ ] Create state analyzer extracting blackboard features
- [ ] Implement pattern matching for agent selection
- [ ] Add heuristic rules based on state patterns
- [ ] Create agent selection scoring
- [ ] Implement A/B testing for selection strategies

**Validation**: Dynamic selection outperforms fixed ordering.

#### Task 7.2: Multi-Strategy Generation
- [ ] Modify Code Specialist to generate 2-3 algorithm variants
- [ ] Implement parallel hypothesis testing
- [ ] Add strategy comparison logic
- [ ] Create strategy selection based on maze features
- [ ] Implement ensemble methods (combine multiple solutions)

**Validation**: Multi-strategy approach finds better solutions 20%+ cases.

### Week 8: Parallel Execution & Optimization

#### Task 8.1: Parallel Agent Execution
```python
class ParallelOrchestrator(SwarmOrchestrator):
    def solve_maze_parallel(self, maze: Maze, 
                           time_limit: int) -> Solution:
        """Execute independent subtasks in parallel"""
        
        plan = self.leader.decompose_task(maze)
        
        # Identify independent subtasks
        dependency_graph = self.build_dependency_graph(plan)
        parallel_batches = self.topological_sort(dependency_graph)
        
        for batch in parallel_batches:
            # Execute batch in parallel
            futures = []
            for subtask in batch:
                future = self.executor.submit(
                    self.execute_subtask, subtask
                )
                futures.append(future)
                
            # Wait for completion with timeout
            results = self.wait_all(futures, timeout=30)
            
            # Update blackboard
            self.update_blackboard(results)
```

**Implementation**:
- [ ] Add parallel execution using ThreadPoolExecutor
- [ ] Implement dependency graph construction
- [ ] Create topological sorting for execution order
- [ ] Add concurrent blackboard access (locks)
- [ ] Implement timeout handling for parallel tasks

**Note**: GPU models must run sequentially (VRAM constraint), CPU models can parallelize.

**Validation**: Parallel execution reduces time by 30-40% for independent tasks.

#### Task 8.2: Token Budget Optimization
- [ ] Implement per-agent token tracking
- [ ] Add dynamic budget allocation based on task complexity
- [ ] Create token-aware prompt compression
- [ ] Implement early stopping when budget depleted
- [ ] Add budget reallocation on agent failure

**Target**: Reduce token consumption by 25% without accuracy loss.

**Validation**: Token usage optimized, maintains solution quality.

#### Task 8.3: Caching & Memoization
- [ ] Cache repeated LLM queries (exact match)
- [ ] Implement semantic similarity caching
- [ ] Cache maze analysis results
- [ ] Store successful code patterns
- [ ] Add cache invalidation logic

**Validation**: Cache hit rate >40% on repeated maze structures.

### Week 9: Adaptive Strategies

#### Task 9.1: Real-Time Learning
```python
class AdaptiveController:
    def __init__(self):
        self.performance_history = []
        self.strategy_effectiveness = {}
        
    def update_strategy(self, outcome: Outcome):
        """Adjust strategies based on recent outcomes"""
        # Track which agent selections worked
        # Track which algorithms succeeded on maze types
        # Adjust weights in selection heuristics
        pass
        
    def get_recommendations(self, current_state: State) -> Dict:
        """Provide strategy recommendations"""
        # "For sparse mazes of this size, BFS has 85% success rate"
        # "Logic Checker caught errors in 3/5 recent runs"
        pass
```

**Implementation**:
- [ ] Create adaptive controller tracking outcomes
- [ ] Implement performance pattern extraction
- [ ] Add strategy effectiveness scoring
- [ ] Create recommendation system
- [ ] Implement online learning (update during run)

**Note**: This is NOT traditional ML training, but online statistics tracking.

**Validation**: Adaptive system improves scores over 20+ runs.

#### Task 9.2: Failure Recovery Strategies
- [ ] Implement graduated retry (same agent with modified prompt)
- [ ] Add agent substitution on repeated failure
- [ ] Create fallback to simpler algorithms
- [ ] Implement partial solution acceptance
- [ ] Add timeout extensions for promising paths

**Validation**: Swarm recovers from 80%+ recoverable failures.

#### Task 9.3: Meta-Planning
- [ ] Add leader self-reflection on failed plans
- [ ] Implement plan quality estimation before execution
- [ ] Create replanning triggers (not just on failure)
- [ ] Add progressive refinement (iterative improvement)
- [ ] Implement multi-plan comparison

**Validation**: Meta-planning reduces failed attempts by 30%+.

---

## Phase 4: Optimization & Intelligence (Weeks 10-12)

### Goal
Achieve state-of-the-art performance with emergent intelligent behavior.

### Week 10: Advanced Blackboard Features

#### Task 10.1: Knowledge Consolidation
```python
class KnowledgeConsolidator:
    def consolidate(self, blackboard: Blackboard):
        """Merge related facts, remove contradictions"""
        # Pattern: Multiple agents report similar findings
        # -> Consolidate into single fact with confidence
        
        # Pattern: Contradictory hypoth