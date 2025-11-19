# Ramanujan-Swarm Implementation Plan

## Executive Summary

This document provides a comprehensive, step-by-step implementation plan for the Ramanujan-Swarm system - an autonomous mathematical discovery engine using LangGraph v0.2+ parallel orchestration with genetic algorithms. The system aims to discover novel mathematical identities involving fundamental constants (π, e, φ, ζ(3)) using a swarm of 20+ Claude 3.5 Sonnet agents organized in a Map-Reduce evolutionary architecture.

**Target Performance**: 2000+ expressions/min across 20 parallel threads, 10^-1500 accuracy within < 40 generations.

---

## Table of Contents

1. [Technology Stack & Dependencies](#1-technology-stack--dependencies)
2. [Project Structure](#2-project-structure)
3. [Core Architecture Components](#3-core-architecture-components)
4. [LangGraph Graph Design](#4-langgraph-graph-design)
5. [State Management](#5-state-management)
6. [Agent Prompt Templates](#6-agent-prompt-templates)
7. [Mathematical Expression Engine](#7-mathematical-expression-engine)
8. [Validation & Scoring System](#8-validation--scoring-system)
9. [OEIS Verification](#9-oeis-verification)
10. [Output & Reporting](#10-output--reporting)
11. [Implementation Sequence](#11-implementation-sequence)
12. [Testing Strategy](#12-testing-strategy)
13. [Performance Optimization](#13-performance-optimization)

---

## 1. Technology Stack & Dependencies

### 1.1 Core Dependencies

Add to `pyproject.toml`:

```toml
[project]
name = "ramanujan-formulas"
version = "0.1.0"
description = "Autonomous Mathematical Discovery via Genetic Agents"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    # LangGraph & LangChain
    "langgraph>=0.2.45",
    "langchain>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-core>=0.3.0",

    # High-precision mathematics
    "mpmath>=1.3.0",
    "sympy>=1.13.0",

    # Web scraping & API
    "requests>=2.32.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",

    # Data handling
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",

    # Async & concurrency
    "aiohttp>=3.10.0",

    # Optional: for production checkpointing
    "aiosqlite>=0.20.0",  # for AsyncSqliteSaver
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.25.0",
    "black>=24.0.0",
    "ruff>=0.8.0",
]
```

### 1.2 Environment Setup

Create `.env` file:
```env
ANTHROPIC_API_KEY=your_api_key_here
SWARM_SIZE=20
MAX_GENERATIONS=100
GENE_POOL_SIZE=25
TARGET_CONSTANTS=pi,e,phi,apery,euler
PRECISION_DPS=1500
CHECKPOINT_DB=checkpoints.db
```

---

## 2. Project Structure

```
ramanujan-formulas/
├── main.py                          # Entry point
├── pyproject.toml
├── .env
├── .gitignore
├── README.md
├── plans/
│   └── implementation_plan.md
├── ramanujan_swarm/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── constants.py                 # Mathematical constants definitions
│   │
│   ├── graph/                       # LangGraph components
│   │   ├── __init__.py
│   │   ├── state.py                 # State definitions with reducers
│   │   ├── nodes.py                 # All graph nodes
│   │   ├── graph_builder.py        # Graph construction
│   │   └── checkpointer.py          # MemorySaver/persistence
│   │
│   ├── agents/                      # Agent logic
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Base agent class
│   │   ├── explorer.py             # Explorer agent
│   │   ├── mutator.py              # Mutator agent
│   │   ├── hybrid.py               # Hybrid agent
│   │   └── prompts.py              # All prompt templates
│   │
│   ├── math_engine/                 # Mathematical computation
│   │   ├── __init__.py
│   │   ├── expression_parser.py    # Parse LLM output to expressions
│   │   ├── evaluator.py            # High-precision evaluation
│   │   ├── validator.py            # Syntax & numeric validation
│   │   ├── deduplicator.py         # Hash-based dedup
│   │   └── elegance_scorer.py      # Elegance scoring function
│   │
│   ├── gene_pool/                   # Memory management
│   │   ├── __init__.py
│   │   ├── pool.py                 # Gene pool data structure
│   │   └── selection.py            # Selection strategies
│   │
│   ├── verification/                # External verification
│   │   ├── __init__.py
│   │   ├── oeis_client.py          # OEIS API/scraping
│   │   └── verifier.py             # Cross-reference checker
│   │
│   ├── reporting/                   # Output generation
│   │   ├── __init__.py
│   │   ├── markdown_generator.py   # FINAL_REPORT.md
│   │   ├── json_exporter.py        # JSON output
│   │   └── visualization.py        # Optional: graph viz
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logger.py               # Logging configuration
│       └── helpers.py              # Misc utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_math_engine.py
│   ├── test_graph.py
│   └── test_integration.py
│
└── outputs/
    ├── FINAL_REPORT.md             # Generated continuously
    ├── discoveries.json
    └── checkpoints.db              # LangGraph state
```

---

## 3. Core Architecture Components

### 3.1 System Flow Overview

```
[Initialization]
    ↓
[Dispatch Node] (creates Send() messages)
    ↓
[Map Phase] → 20+ Parallel Agents (Explorer/Mutator/Hybrid)
    ↓ (all return to single node)
[Validator Node] (aggregates via operator.add reducer)
    ↓
[Deduplication & Filtering]
    ↓
[Elegance Scoring]
    ↓
[Gene Pool Update]
    ↓
[Discovery Check] (Error < 1e-50?)
    ├─ Yes → [OEIS Verification] → [Save to Output]
    └─ No  → [Next Generation] → Loop back to Dispatch
```

### 3.2 Key Technical Patterns

**Map-Reduce with Send API**:
- Use `Send()` to dynamically fan out to N agent instances
- Each agent processes independently with access to gene pool
- All results aggregate via `operator.add` reducer on state list fields

**Checkpointing Strategy**:
- Use `MemorySaver` for development/testing
- Use `AsyncSqliteSaver` for production persistence
- Save state after every generation (superstep)
- Enable "time travel" debugging and recovery

**Concurrent Writes Safety**:
- Use `Annotated[list, operator.add]` for all list fields that agents update
- This ensures safe concurrent writes without collisions

---

## 4. LangGraph Graph Design

### 4.1 State Schema (`ramanujan_swarm/graph/state.py`)

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from dataclasses import dataclass

@dataclass
class Expression:
    """Individual mathematical expression candidate."""
    formula_str: str                # Original string from LLM
    parsed_expr: str                # SymPy parsed form
    target_constant: str            # pi, e, phi, etc.
    agent_type: Literal["explorer", "mutator", "hybrid"]
    generation: int
    numeric_value: str              # mpmath result as string
    error: float                    # Absolute error
    elegance_score: float
    complexity: int                 # Length of formula
    hash_syntax: str                # For structural dedup
    hash_numeric: str               # For numeric dedup
    timestamp: float

class SwarmState(TypedDict):
    """Global state shared across all nodes."""

    # Current generation counter
    generation: int

    # Gene pool: top candidates from all generations
    gene_pool: list[Expression]

    # Current generation's raw proposals (uses reducer)
    current_proposals: Annotated[list[Expression], add]

    # Filtered & scored candidates ready for gene pool
    validated_candidates: list[Expression]

    # Discoveries that passed 1e-50 threshold
    discoveries: Annotated[list[Expression], add]

    # Configuration
    swarm_size: int
    target_constants: list[str]
    precision_dps: int

    # Metrics
    total_expressions_generated: int
    expressions_per_generation: list[int]
    best_error_per_generation: list[float]
```

### 4.2 Graph Structure (`ramanujan_swarm/graph/graph_builder.py`)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from ramanujan_swarm.graph.state import SwarmState
from ramanujan_swarm.graph import nodes

def build_graph():
    """Construct the Ramanujan-Swarm LangGraph."""

    # Create graph with state schema
    graph = StateGraph(SwarmState)

    # Add nodes
    graph.add_node("initialize", nodes.initialize_node)
    graph.add_node("dispatch", nodes.dispatch_node)
    graph.add_node("agent_worker", nodes.agent_worker_node)  # Single node, many instances
    graph.add_node("validator", nodes.validator_node)
    graph.add_node("gene_pool_update", nodes.gene_pool_update_node)
    graph.add_node("discovery_check", nodes.discovery_check_node)
    graph.add_node("oeis_verify", nodes.oeis_verify_node)
    graph.add_node("save_output", nodes.save_output_node)

    # Add edges
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "dispatch")

    # Conditional edge: dispatch -> multiple agent_worker instances
    graph.add_conditional_edges(
        "dispatch",
        nodes.create_agent_tasks,  # Returns list[Send]
        ["agent_worker"]
    )

    # All agents converge to validator
    graph.add_edge("agent_worker", "validator")
    graph.add_edge("validator", "gene_pool_update")
    graph.add_edge("gene_pool_update", "discovery_check")

    # Conditional: check if we have discoveries
    graph.add_conditional_edges(
        "discovery_check",
        nodes.route_discoveries,
        {
            "verify": "oeis_verify",
            "continue": "dispatch",
            "end": END
        }
    )

    graph.add_edge("oeis_verify", "save_output")
    graph.add_edge("save_output", "dispatch")  # Continue to next generation

    # Compile with checkpointer
    checkpointer = MemorySaver()  # or AsyncSqliteSaver for production
    app = graph.compile(checkpointer=checkpointer)

    return app
```

### 4.3 Node Implementations (`ramanujan_swarm/graph/nodes.py`)

Key node implementations:

```python
from langgraph.constants import Send
from ramanujan_swarm.graph.state import SwarmState, Expression
from ramanujan_swarm.agents import ExplorerAgent, MutatorAgent, HybridAgent
from ramanujan_swarm.math_engine import Evaluator, Validator, Deduplicator, EleganceScorer
from ramanujan_swarm.gene_pool import GenePool
from ramanujan_swarm.verification import OEISVerifier

async def initialize_node(state: SwarmState) -> dict:
    """Initialize the swarm system."""
    return {
        "generation": 0,
        "gene_pool": [],
        "current_proposals": [],
        "validated_candidates": [],
        "discoveries": [],
        "total_expressions_generated": 0,
        "expressions_per_generation": [],
        "best_error_per_generation": [],
    }

def create_agent_tasks(state: SwarmState) -> list[Send]:
    """
    Dispatch function: creates Send() messages for parallel agent execution.

    This is the MAP phase - dynamically spawn N agent instances.
    """
    swarm_size = state["swarm_size"]
    agent_types = ["explorer", "mutator", "hybrid"]

    sends = []
    for i in range(swarm_size):
        # Distribute agent types
        agent_type = agent_types[i % len(agent_types)]

        # Create Send message for each agent
        # Each gets full state access
        sends.append(
            Send(
                "agent_worker",
                {
                    "agent_id": i,
                    "agent_type": agent_type,
                    "generation": state["generation"],
                    "gene_pool": state["gene_pool"],  # Read-only access
                    "target_constants": state["target_constants"],
                }
            )
        )

    return sends

async def agent_worker_node(state: dict) -> dict:
    """
    Worker node for individual agent - runs in parallel instances.

    Returns proposals via operator.add reducer on current_proposals.
    """
    agent_type = state["agent_type"]
    agent_id = state["agent_id"]
    generation = state["generation"]
    gene_pool = state["gene_pool"]

    # Instantiate appropriate agent
    if agent_type == "explorer":
        agent = ExplorerAgent(agent_id)
    elif agent_type == "mutator":
        agent = MutatorAgent(agent_id)
    else:
        agent = HybridAgent(agent_id)

    # Generate expressions (calls Claude)
    expressions = await agent.generate_expressions(
        gene_pool=gene_pool,
        generation=generation,
        target_constants=state["target_constants"]
    )

    # Return via reducer field (operator.add will concatenate)
    return {
        "current_proposals": expressions  # List[Expression]
    }

async def validator_node(state: SwarmState) -> dict:
    """
    REDUCE phase: aggregate all proposals, deduplicate, and score.

    This is the central "CPU" that processes all agent outputs.
    """
    proposals = state["current_proposals"]
    generation = state["generation"]

    # Parse and validate expressions
    validator = Validator(precision_dps=state["precision_dps"])
    evaluator = Evaluator(precision_dps=state["precision_dps"])

    valid_expressions = []
    for expr in proposals:
        # Parse with SymPy
        parsed = validator.parse_expression(expr.formula_str)
        if parsed is None:
            continue

        # Evaluate with mpmath at high precision
        numeric_result = evaluator.evaluate(parsed, expr.target_constant)
        if numeric_result is None:
            continue

        # Calculate error
        error = evaluator.compute_error(numeric_result, expr.target_constant)

        # Compute hashes for deduplication
        hash_syntax = validator.syntax_hash(parsed)
        hash_numeric = validator.numeric_hash(numeric_result)

        # Update expression
        expr.parsed_expr = str(parsed)
        expr.numeric_value = str(numeric_result)
        expr.error = error
        expr.hash_syntax = hash_syntax
        expr.hash_numeric = hash_numeric

        valid_expressions.append(expr)

    # Deduplicate
    deduplicator = Deduplicator()
    unique_expressions = deduplicator.deduplicate(valid_expressions)

    # Score by elegance
    scorer = EleganceScorer()
    for expr in unique_expressions:
        expr.elegance_score = scorer.score(expr)

    # Filter by thresholds
    # Keep error < 1e-12 for gene pool evolution
    candidates = [e for e in unique_expressions if e.error < 1e-12]

    # Sort by elegance (lower is better)
    candidates.sort(key=lambda e: e.elegance_score)

    # Update metrics
    best_error = min([e.error for e in candidates], default=float('inf'))

    return {
        "validated_candidates": candidates,
        "current_proposals": [],  # Clear for next generation
        "total_expressions_generated": state["total_expressions_generated"] + len(proposals),
        "expressions_per_generation": state["expressions_per_generation"] + [len(proposals)],
        "best_error_per_generation": state["best_error_per_generation"] + [best_error],
    }

def gene_pool_update_node(state: SwarmState) -> dict:
    """Update the gene pool with new candidates."""
    gene_pool = GenePool(max_size=25)
    gene_pool.update(
        current_pool=state["gene_pool"],
        new_candidates=state["validated_candidates"]
    )

    return {
        "gene_pool": gene_pool.get_pool(),
        "generation": state["generation"] + 1,
    }

def route_discoveries(state: SwarmState) -> str:
    """Route based on whether we have new discoveries."""
    # Check for discoveries (error < 1e-50)
    discoveries = [e for e in state["validated_candidates"] if e.error < 1e-50]

    if discoveries:
        return "verify"  # Go to OEIS verification
    elif state["generation"] >= 100:  # MAX_GENERATIONS
        return "end"
    else:
        return "continue"  # Next generation

async def discovery_check_node(state: SwarmState) -> dict:
    """Check for significant discoveries."""
    # Find expressions below the 1e-50 threshold
    discoveries = [e for e in state["validated_candidates"] if e.error < 1e-50]

    return {
        "discoveries": discoveries  # operator.add will append
    }

async def oeis_verify_node(state: SwarmState) -> dict:
    """Verify discoveries against OEIS database."""
    verifier = OEISVerifier()

    # Get recent discoveries
    recent_discoveries = state["discoveries"][-10:]  # Last 10

    for expr in recent_discoveries:
        oeis_result = await verifier.check_oeis(expr)
        expr.oeis_match = oeis_result

    return {}

async def save_output_node(state: SwarmState) -> dict:
    """Save discoveries to output files."""
    from ramanujan_swarm.reporting import MarkdownGenerator, JSONExporter

    # Generate markdown report
    md_gen = MarkdownGenerator()
    md_gen.generate_report(state)

    # Export JSON
    json_exp = JSONExporter()
    json_exp.export_discoveries(state["discoveries"])

    return {}
```

---

## 5. State Management

### 5.1 Reducer Pattern for Concurrent Writes

The key to safe parallel execution is using the `operator.add` reducer:

```python
from typing import Annotated
from operator import add

class SwarmState(TypedDict):
    # This field can be safely updated by multiple parallel agents
    current_proposals: Annotated[list[Expression], add]
    discoveries: Annotated[list[Expression], add]
```

**How it works**:
- When multiple `agent_worker` instances return `{"current_proposals": [expr1, expr2]}`, LangGraph automatically concatenates all lists
- No `INVALID_CONCURRENT_GRAPH_UPDATE` errors
- Order is preserved within each agent's output

### 5.2 Checkpointing Strategy

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Development: in-memory
checkpointer = MemorySaver()

# Production: persistent SQLite
async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = build_graph()
    app = graph.compile(checkpointer=checkpointer)

    # Run with thread_id for state isolation
    config = {"configurable": {"thread_id": "swarm-session-1"}}
    result = await app.ainvoke(initial_state, config)
```

**Benefits**:
- Resume after crashes
- Replay specific generations
- Debug specific graph executions
- Time-travel debugging

---

## 6. Agent Prompt Templates

### 6.1 Base Prompt Structure (`ramanujan_swarm/agents/prompts.py`)

```python
SYSTEM_PROMPT = """You are a mathematical discovery agent in the Ramanujan-Swarm system.

Your mission: Generate novel mathematical identities involving fundamental constants.

Target Constants:
- π (pi): 3.14159...
- e (Euler's number): 2.71828...
- φ (Golden ratio): 1.61803...
- ζ(3) (Apery's constant): 1.20205...
- γ (Euler-Mascheroni constant): 0.57721...

Output Format:
You MUST respond with valid JSON containing a list of mathematical expressions.

{
  "expressions": [
    {
      "formula": "sqrt(1 + 2*sqrt(1 + 3*sqrt(1 + 4*sqrt(1 + ...))))",
      "target_constant": "pi",
      "description": "Nested radical approximation of pi",
      "complexity_estimate": 25
    }
  ]
}

Guidelines:
1. Use Python-compatible mathematical notation
2. Functions available: sqrt, cbrt, exp, log, sin, cos, tan, factorial, etc.
3. Aim for elegance: shorter is better
4. Each expression should be computable
5. Generate 5-10 expressions per response
6. Be creative: explore nested radicals, continued fractions, series, products
"""

EXPLORER_PROMPT = """Role: EXPLORER

You are an Explorer agent focused on NOVEL territory.

Strategies:
1. **Nested Radicals**: e.g., sqrt(a + b*sqrt(c + d*sqrt(...)))
2. **Continued Fractions**: e.g., a + b/(c + d/(e + ...))
3. **Infinite Series**: e.g., sum(f(n) for n in range(N))
4. **Products**: e.g., product((n^2-1)/(n^2+1) for n in ...)
5. **Mixed Forms**: Combine multiple techniques

Inspiration from Ramanujan:
- The identity: 1/pi = (2*sqrt(2)/9801) * sum(...)
- Rogers-Ramanujan continued fractions
- Nested radical denestings

Current Generation: {generation}

Generate 8-10 novel expressions exploring uncharted mathematical territory.
"""

MUTATOR_PROMPT = """Role: MUTATOR

You are a Mutator agent focused on REFINING promising candidates.

Gene Pool (Top {gene_pool_size} candidates):
{gene_pool_formatted}

Mutation Strategies:
1. **Coefficient tweaking**: Change numerical constants slightly
2. **Structural modification**: Add/remove terms, change operators
3. **Nesting depth**: Increase/decrease recursion levels
4. **Function substitution**: Replace sqrt with exp, etc.
5. **Symmetry exploitation**: Use mathematical symmetries

Example:
Original: sqrt(1 + 2*sqrt(1 + 3*sqrt(1)))
Mutated: sqrt(1 + phi*sqrt(1 + phi^2*sqrt(1)))

Current Generation: {generation}

Generate 8-10 mutations of the best gene pool candidates.
"""

HYBRID_PROMPT = """Role: HYBRID

You combine EXPLORATION and MUTATION strategies.

Gene Pool Preview:
{gene_pool_preview}

Your Approach:
1. Generate 4-5 novel exploratory expressions
2. Generate 4-5 mutations of gene pool candidates
3. Focus on both breadth (exploration) and depth (refinement)

Current Generation: {generation}

Generate 8-10 expressions balancing novelty and refinement.
"""
```

### 6.2 Dynamic Prompt Formatting

```python
class BaseAgent:
    def format_prompt(self, gene_pool: list[Expression], generation: int) -> str:
        """Format prompt with current context."""

        # Format gene pool for display
        gene_pool_formatted = "\n".join([
            f"- {expr.formula_str} → Error: {expr.error:.2e}, Score: {expr.elegance_score:.2f}"
            for expr in gene_pool[:10]  # Show top 10
        ])

        # Get appropriate prompt template
        if isinstance(self, ExplorerAgent):
            prompt = EXPLORER_PROMPT
        elif isinstance(self, MutatorAgent):
            prompt = MUTATOR_PROMPT
        else:
            prompt = HYBRID_PROMPT

        # Fill in template
        return prompt.format(
            generation=generation,
            gene_pool_size=len(gene_pool),
            gene_pool_formatted=gene_pool_formatted,
            gene_pool_preview=gene_pool_formatted[:500]  # Truncate for hybrids
        )
```

---

## 7. Mathematical Expression Engine

### 7.1 Expression Parser (`ramanujan_swarm/math_engine/expression_parser.py`)

```python
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import sympify, symbols
import re

class ExpressionParser:
    """Parse LLM-generated mathematical expressions."""

    def __init__(self):
        self.transformations = (
            standard_transformations +
            (implicit_multiplication_application,)
        )

    def parse(self, formula_str: str):
        """Parse string to SymPy expression."""
        try:
            # Clean the input
            cleaned = self._preprocess(formula_str)

            # Parse with SymPy
            expr = parse_expr(
                cleaned,
                transformations=self.transformations,
                evaluate=False  # Keep structure
            )

            return expr

        except Exception as e:
            print(f"Parse error: {e}")
            return None

    def _preprocess(self, formula_str: str) -> str:
        """Clean and normalize formula string."""
        # Replace common patterns
        s = formula_str.strip()

        # Handle special notations
        s = s.replace("cbrt", "root")  # Cube root
        s = s.replace("^", "**")  # Exponentiation

        # Ensure proper function notation
        s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)  # 2x -> 2*x

        return s
```

### 7.2 High-Precision Evaluator (`ramanujan_swarm/math_engine/evaluator.py`)

```python
import mpmath
from sympy import lambdify, symbols
from sympy.core.expr import Expr

class Evaluator:
    """Evaluate expressions with mpmath high precision."""

    def __init__(self, precision_dps: int = 1500):
        mpmath.mp.dps = precision_dps  # Decimal places

        # Pre-define constants at high precision
        self.constants = {
            "pi": mpmath.mp.pi,
            "e": mpmath.mp.e,
            "phi": (1 + mpmath.sqrt(5)) / 2,  # Golden ratio
            "apery": mpmath.apery,  # ζ(3)
            "euler": mpmath.euler,  # γ
            "catalan": mpmath.catalan,
        }

    def evaluate(self, sympy_expr: Expr, target_constant: str) -> mpmath.mpf | None:
        """Evaluate SymPy expression to high-precision numeric value."""
        try:
            # Convert SymPy to callable function
            free_symbols = list(sympy_expr.free_symbols)

            if not free_symbols:
                # No variables, direct evaluation
                f = lambdify([], sympy_expr, "mpmath")
                result = f()
            else:
                # Has variables - try to evaluate with standard values
                # This is for continued fractions, series, etc.
                # Implementation would handle special cases
                result = self._evaluate_with_vars(sympy_expr, free_symbols)

            return mpmath.mpf(result)

        except Exception as e:
            print(f"Evaluation error: {e}")
            return None

    def compute_error(self, computed_value: mpmath.mpf, target_constant: str) -> float:
        """Compute absolute error against target constant."""
        target = self.constants[target_constant]
        error = abs(computed_value - target)
        return float(error)

    def _evaluate_with_vars(self, expr, symbols_list):
        """Handle expressions with variables (advanced)."""
        # Placeholder for complex evaluation logic
        # Would need to handle series limits, continued fraction truncation, etc.
        pass
```

### 7.3 Deduplicator (`ramanujan_swarm/math_engine/deduplicator.py`)

```python
import hashlib
from collections import defaultdict
from ramanujan_swarm.graph.state import Expression

class Deduplicator:
    """Remove duplicate expressions using dual-hash strategy."""

    def deduplicate(self, expressions: list[Expression]) -> list[Expression]:
        """
        Remove duplicates based on:
        1. Syntax hash (structural equivalence)
        2. Numeric hash (value equivalence at high precision)
        """
        seen_syntax = set()
        seen_numeric = set()
        unique = []

        for expr in expressions:
            # Check structural duplicate
            if expr.hash_syntax in seen_syntax:
                continue

            # Check numeric duplicate (only for similar errors)
            if expr.hash_numeric in seen_numeric:
                continue

            seen_syntax.add(expr.hash_syntax)
            seen_numeric.add(expr.hash_numeric)
            unique.append(expr)

        return unique

class Validator:
    """Validate and compute hashes for expressions."""

    def syntax_hash(self, sympy_expr) -> str:
        """Hash based on syntactic structure."""
        # Use canonical string representation
        canonical = str(sympy_expr)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def numeric_hash(self, mpmath_value: mpmath.mpf) -> str:
        """Hash based on numeric value (first 100 digits)."""
        # Convert to string with fixed precision
        value_str = mpmath.nstr(mpmath_value, 100, strip_zeros=False)
        return hashlib.sha256(value_str.encode()).hexdigest()[:16]
```

### 7.4 Elegance Scorer (`ramanujan_swarm/math_engine/elegance_scorer.py`)

```python
from ramanujan_swarm.graph.state import Expression

class EleganceScorer:
    """
    Score expressions by elegance.

    Formula: Score = Error × (1 + 0.03 × Length)

    Lower score = more elegant
    """

    def __init__(self, complexity_weight: float = 0.03):
        self.complexity_weight = complexity_weight

    def score(self, expr: Expression) -> float:
        """Calculate elegance score."""
        # Get complexity (formula length)
        complexity = len(expr.formula_str)

        # Apply penalty
        complexity_penalty = 1 + self.complexity_weight * complexity

        # Final score
        score = expr.error * complexity_penalty

        return score

    def compute_complexity(self, expr: Expression) -> int:
        """More sophisticated complexity metric (optional enhancement)."""
        # Could count:
        # - Number of operations
        # - Nesting depth
        # - Number of constants
        # - Function calls

        # Simple version: just string length
        return len(expr.formula_str)
```

---

## 8. Validation & Scoring System

### 8.1 Dual-Threshold Filtering

```python
class FilteringEngine:
    """Apply dual-threshold filtering strategy."""

    THRESHOLD_EVOLUTION = 1e-12  # Keep for mutation
    THRESHOLD_DISCOVERY = 1e-50  # Log as discovery

    def filter_candidates(self, expressions: list[Expression]) -> tuple[list, list]:
        """
        Returns:
            - evolution_pool: Expressions for gene pool (< 1e-12)
            - discoveries: High-quality discoveries (< 1e-50)
        """
        evolution_pool = []
        discoveries = []

        for expr in expressions:
            if expr.error < self.THRESHOLD_DISCOVERY:
                discoveries.append(expr)
                evolution_pool.append(expr)  # Also add to evolution
            elif expr.error < self.THRESHOLD_EVOLUTION:
                evolution_pool.append(expr)

        return evolution_pool, discoveries
```

### 8.2 Gene Pool Management (`ramanujan_swarm/gene_pool/pool.py`)

```python
from ramanujan_swarm.graph.state import Expression

class GenePool:
    """Manage the top-N candidates across generations."""

    def __init__(self, max_size: int = 25):
        self.max_size = max_size

    def update(self, current_pool: list[Expression],
               new_candidates: list[Expression]) -> list[Expression]:
        """
        Update gene pool with new candidates.

        Strategy:
        1. Combine current pool + new candidates
        2. Sort by elegance score
        3. Take top N
        4. Ensure diversity (optional: prevent too many similar)
        """
        # Combine
        combined = current_pool + new_candidates

        # Sort by elegance (lower is better)
        combined.sort(key=lambda e: e.elegance_score)

        # Take top N
        top_n = combined[:self.max_size]

        return top_n

    def get_pool(self) -> list[Expression]:
        """Return current gene pool."""
        return self.pool

    def get_best(self, n: int = 5) -> list[Expression]:
        """Get best N expressions."""
        return sorted(self.pool, key=lambda e: e.elegance_score)[:n]
```

---

## 9. OEIS Verification

### 9.1 OEIS Client (`ramanujan_swarm/verification/oeis_client.py`)

```python
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import Optional

class OEISClient:
    """Interface to OEIS for mathematical sequence verification."""

    BASE_URL = "https://oeis.org/search"
    RATE_LIMIT_DELAY = 1.0  # Respect OEIS rate limits

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0

    async def search(self, query: str) -> dict:
        """
        Search OEIS by sequence or formula.

        Returns JSON with matching sequences.
        """
        await self._rate_limit()

        params = {
            "fmt": "json",
            "q": query,
            "start": 0
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                else:
                    return {"results": []}

    async def search_by_decimal(self, decimal_str: str, n_digits: int = 50) -> dict:
        """
        Search OEIS by decimal expansion.

        Converts value like 3.14159... to sequence [3,1,4,1,5,9,...]
        """
        # Remove decimal point
        digits = decimal_str.replace(".", "").replace("-", "")[:n_digits]

        # Convert to comma-separated
        sequence = ",".join(digits)

        return await self.search(sequence)

    async def _rate_limit(self):
        """Ensure we respect OEIS rate limits (1 req/sec)."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.RATE_LIMIT_DELAY:
            await asyncio.sleep(self.RATE_LIMIT_DELAY - time_since_last)

        self.last_request_time = asyncio.get_event_loop().time()
```

### 9.2 Verification Logic (`ramanujan_swarm/verification/verifier.py`)

```python
from ramanujan_swarm.graph.state import Expression
from ramanujan_swarm.verification.oeis_client import OEISClient

class OEISVerifier:
    """Verify if discovered identities appear in OEIS."""

    def __init__(self):
        self.client = OEISClient()

    async def check_oeis(self, expr: Expression) -> dict:
        """
        Check if expression matches known OEIS sequences.

        Returns:
            {
                "found": bool,
                "sequences": list[str],  # OEIS IDs
                "confidence": float
            }
        """
        # Try multiple search strategies
        results = []

        # 1. Search by decimal expansion
        decimal_result = await self.client.search_by_decimal(
            expr.numeric_value,
            n_digits=50
        )
        results.append(decimal_result)

        # 2. Search by formula (if recognizable pattern)
        # This would involve extracting keywords from formula
        # e.g., "continued fraction", "nested radical", etc.

        # Aggregate results
        if decimal_result.get("results"):
            return {
                "found": True,
                "sequences": [r["number"] for r in decimal_result["results"][:5]],
                "confidence": 0.8
            }
        else:
            return {
                "found": False,
                "sequences": [],
                "confidence": 0.0
            }
```

---

## 10. Output & Reporting

### 10.1 Markdown Report Generator (`ramanujan_swarm/reporting/markdown_generator.py`)

```python
from ramanujan_swarm.graph.state import SwarmState
from datetime import datetime
import os

class MarkdownGenerator:
    """Generate continuous markdown reports."""

    OUTPUT_PATH = "outputs/FINAL_REPORT.md"

    def generate_report(self, state: SwarmState):
        """Generate/update FINAL_REPORT.md with latest results."""

        report = self._build_report(state)

        # Write to file
        os.makedirs("outputs", exist_ok=True)
        with open(self.OUTPUT_PATH, "w") as f:
            f.write(report)

    def _build_report(self, state: SwarmState) -> str:
        """Build markdown report content."""

        report = f"""# Ramanujan-Swarm Discovery Report

**Generated**: {datetime.now().isoformat()}
**Generation**: {state['generation']}
**Total Expressions Generated**: {state['total_expressions_generated']}

---

## Performance Metrics

- **Expressions per Generation**: {state['expressions_per_generation'][-1] if state['expressions_per_generation'] else 0}
- **Best Error This Generation**: {state['best_error_per_generation'][-1] if state['best_error_per_generation'] else 'N/A'}
- **Gene Pool Size**: {len(state['gene_pool'])}
- **Total Discoveries**: {len(state['discoveries'])}

---

## Top Discoveries (Error < 1e-50)

"""

        # Add discoveries
        for i, expr in enumerate(state['discoveries'][:20], 1):
            report += f"""
### Discovery {i}

**Formula**: `{expr.formula_str}`

**Target Constant**: {expr.target_constant}

**Error**: {expr.error:.6e}

**Elegance Score**: {expr.elegance_score:.4f}

**Complexity**: {expr.complexity}

**Generation Discovered**: {expr.generation}

**Agent Type**: {expr.agent_type}

**Numeric Value**:
```
{expr.numeric_value[:200]}...
```

**OEIS Verification**: {expr.oeis_match if hasattr(expr, 'oeis_match') else 'Pending'}

---
"""

        # Add gene pool
        report += "\n## Current Gene Pool (Top 25)\n\n"
        report += "| Rank | Formula | Error | Elegance | Generation |\n"
        report += "|------|---------|-------|----------|------------|\n"

        for i, expr in enumerate(state['gene_pool'][:25], 1):
            formula_truncated = expr.formula_str[:50] + "..." if len(expr.formula_str) > 50 else expr.formula_str
            report += f"| {i} | `{formula_truncated}` | {expr.error:.2e} | {expr.elegance_score:.2f} | {expr.generation} |\n"

        return report
```

### 10.2 JSON Exporter (`ramanujan_swarm/reporting/json_exporter.py`)

```python
import json
from ramanujan_swarm.graph.state import Expression
from dataclasses import asdict

class JSONExporter:
    """Export discoveries to JSON format."""

    OUTPUT_PATH = "outputs/discoveries.json"

    def export_discoveries(self, discoveries: list[Expression]):
        """Export discoveries to JSON."""

        # Convert to dict
        data = {
            "discoveries": [asdict(expr) for expr in discoveries],
            "count": len(discoveries),
            "exported_at": datetime.now().isoformat()
        }

        # Write
        os.makedirs("outputs", exist_ok=True)
        with open(self.OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
```

---

## 11. Implementation Sequence

### Phase 1: Foundation (Week 1)

**Day 1-2: Project Setup**
- [ ] Set up project structure
- [ ] Configure `pyproject.toml` with dependencies
- [ ] Set up `.env` and configuration management
- [ ] Implement logging infrastructure
- [ ] Create basic test structure

**Day 3-4: Mathematical Engine**
- [ ] Implement `ExpressionParser` with SymPy
- [ ] Implement `Evaluator` with mpmath at 1500 dps
- [ ] Implement `Validator` with hash functions
- [ ] Implement `Deduplicator`
- [ ] Implement `EleganceScorer`
- [ ] Write unit tests for math engine

**Day 5-7: State & Data Structures**
- [ ] Define `SwarmState` TypedDict with reducers
- [ ] Implement `Expression` dataclass
- [ ] Implement `GenePool` class
- [ ] Test state updates and reducers
- [ ] Implement configuration loader

### Phase 2: LangGraph Architecture (Week 2)

**Day 8-10: Graph Construction**
- [ ] Implement basic `StateGraph` structure
- [ ] Create all node functions (stubs first)
- [ ] Implement `create_agent_tasks` dispatch function
- [ ] Add conditional edges and routing
- [ ] Set up `MemorySaver` checkpointer
- [ ] Test graph compilation

**Day 11-12: Agent Integration**
- [ ] Implement `BaseAgent` class
- [ ] Set up `ChatAnthropic` with temperature 1.1
- [ ] Create prompt templates for each agent type
- [ ] Implement `ExplorerAgent`
- [ ] Implement `MutatorAgent`
- [ ] Implement `HybridAgent`
- [ ] Test agent response parsing

**Day 13-14: Node Implementation**
- [ ] Complete `initialize_node`
- [ ] Complete `agent_worker_node` with async Claude calls
- [ ] Complete `validator_node` with full pipeline
- [ ] Complete `gene_pool_update_node`
- [ ] Complete `discovery_check_node`
- [ ] Test node execution in isolation

### Phase 3: Integration & Verification (Week 3)

**Day 15-16: OEIS Integration**
- [ ] Implement `OEISClient` with rate limiting
- [ ] Implement decimal search strategy
- [ ] Implement `OEISVerifier`
- [ ] Add OEIS check to `oeis_verify_node`
- [ ] Test OEIS API calls

**Day 17-18: Output System**
- [ ] Implement `MarkdownGenerator`
- [ ] Implement `JSONExporter`
- [ ] Complete `save_output_node`
- [ ] Test report generation
- [ ] Add visualization (optional)

**Day 19-21: End-to-End Testing**
- [ ] Run complete graph with small swarm (5 agents)
- [ ] Test checkpoint/resume functionality
- [ ] Test error handling and recovery
- [ ] Debug any issues
- [ ] Optimize performance

### Phase 4: Scaling & Optimization (Week 4)

**Day 22-24: Production Readiness**
- [ ] Scale to 20+ agents
- [ ] Implement `AsyncSqliteSaver` for persistence
- [ ] Add comprehensive error handling
- [ ] Implement retry logic
- [ ] Add progress tracking and logging
- [ ] Performance profiling

**Day 25-26: Documentation & Polish**
- [ ] Write API documentation
- [ ] Create usage examples
- [ ] Add configuration options
- [ ] Improve error messages
- [ ] Code cleanup and refactoring

**Day 27-28: Validation & Tuning**
- [ ] Run extended experiments (100+ generations)
- [ ] Tune elegance scoring parameters
- [ ] Tune agent prompt templates
- [ ] Analyze discovery quality
- [ ] Final testing and validation

---

## 12. Testing Strategy

### 12.1 Unit Tests

```python
# tests/test_math_engine.py
import pytest
from ramanujan_swarm.math_engine import ExpressionParser, Evaluator

def test_parser_basic():
    parser = ExpressionParser()
    expr = parser.parse("sqrt(2)")
    assert expr is not None

def test_evaluator_pi():
    evaluator = Evaluator(precision_dps=100)
    from sympy import pi as sym_pi
    result = evaluator.evaluate(sym_pi, "pi")
    error = evaluator.compute_error(result, "pi")
    assert error < 1e-90  # Should be essentially zero

def test_elegance_scorer():
    from ramanujan_swarm.math_engine import EleganceScorer
    from ramanujan_swarm.graph.state import Expression

    scorer = EleganceScorer()
    expr = Expression(
        formula_str="sqrt(2)",
        error=1e-10,
        complexity=7,
        # ... other fields
    )
    score = scorer.score(expr)
    assert score > 0
```

### 12.2 Integration Tests

```python
# tests/test_graph.py
import pytest
from ramanujan_swarm.graph import build_graph

@pytest.mark.asyncio
async def test_graph_single_generation():
    """Test one complete generation cycle."""
    graph = build_graph()

    initial_state = {
        "generation": 0,
        "swarm_size": 3,  # Small for testing
        "target_constants": ["pi"],
        "precision_dps": 100,
        # ... other fields
    }

    config = {"configurable": {"thread_id": "test-1"}}
    result = await graph.ainvoke(initial_state, config)

    assert result["generation"] == 1
    assert len(result["current_proposals"]) > 0
```

### 12.3 End-to-End Tests

```python
# tests/test_integration.py
@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_swarm_run():
    """Test complete swarm execution for 10 generations."""
    from ramanujan_swarm import main

    # Run with test configuration
    discoveries = await main.run_swarm(
        swarm_size=5,
        max_generations=10,
        target_constants=["pi", "e"]
    )

    assert len(discoveries) > 0
```

---

## 13. Performance Optimization

### 13.1 Concurrency Optimizations

**Agent Parallelism**:
- Use `asyncio.gather()` for truly concurrent LLM calls
- Claude API supports high concurrency
- Batch token usage tracking

**Caching**:
- Cache mpmath evaluations for common sub-expressions
- Cache SymPy parsing results
- Cache OEIS query results (with expiration)

### 13.2 Memory Management

**Large Gene Pools**:
- Implement sliding window for very long runs
- Periodic checkpoint compression
- Archive old generations to disk

**Expression Storage**:
- Store only string representations until needed
- Lazy evaluation of parsed forms
- Compress numeric values for storage

### 13.3 Target Metrics

- **Throughput**: 2000+ expressions/min with 20 agents
  - Each agent generates ~8 expressions/call
  - Call frequency: ~12 calls/min/agent
  - 20 agents × 12 calls/min × 8 expr = 1920 expr/min

- **Latency**: < 5 seconds per generation cycle
  - Agent calls: ~2-3 seconds (parallel)
  - Validation: ~1 second
  - Gene pool update: < 0.5 seconds

- **Accuracy**: 10^-1500 achievable
  - mpmath supports arbitrary precision
  - Limited by algorithm convergence, not computation

---

## 14. Key Implementation Notes

### 14.1 LangGraph Patterns Used

1. **Dynamic Fan-Out with Send()**:
   ```python
   def create_agent_tasks(state):
       return [Send("agent_worker", {...}) for _ in range(N)]
   ```

2. **Reducer for Safe Concurrent Writes**:
   ```python
   current_proposals: Annotated[list[Expression], operator.add]
   ```

3. **Conditional Routing**:
   ```python
   graph.add_conditional_edges(
       "discovery_check",
       route_discoveries,
       {"verify": "oeis_verify", "continue": "dispatch", "end": END}
   )
   ```

4. **Checkpointing with MemorySaver**:
   ```python
   app = graph.compile(checkpointer=MemorySaver())
   config = {"configurable": {"thread_id": "session-1"}}
   ```

### 14.2 Mathematical Considerations

**Continued Fractions**:
- Implement truncation at reasonable depth (e.g., 100 levels)
- Use mpmath's `continued_fraction` utilities

**Nested Radicals**:
- Handle recursive evaluation carefully
- Set maximum nesting depth to prevent infinite loops

**Series & Products**:
- Use symbolic summation when possible
- Fall back to numeric approximation with high N

**Convergence Detection**:
- Monitor if error is decreasing
- Implement early stopping for non-convergent expressions

### 14.3 Agent Coordination

**Gene Pool Access**:
- All agents get read-only access to gene pool
- No agent modifies gene pool directly
- Updates happen centrally in `gene_pool_update_node`

**Diversity Maintenance**:
- Track agent types that produced best results
- Adjust agent type distribution dynamically
- Penalize very similar expressions in gene pool

### 14.4 Error Handling

**LLM Failures**:
- Retry with exponential backoff
- Fall back to simpler prompts
- Continue with remaining agents if some fail

**Parse Failures**:
- Log unparseable expressions for analysis
- Don't crash the entire generation
- Provide feedback to improve prompts

**Computation Failures**:
- Catch mpmath overflow/underflow
- Handle divide-by-zero gracefully
- Skip invalid expressions

---

## 15. Research References

Based on web research conducted, key references:

1. **LangGraph Documentation**:
   - Map-Reduce Pattern: https://langchain-ai.github.io/langgraphjs/how-tos/map-reduce/
   - Send API: Official LangGraph docs
   - MemorySaver: https://docs.langchain.com/oss/python/langgraph/add-memory

2. **mpmath Documentation**:
   - High-precision arithmetic: https://mpmath.org/
   - Mathematical constants: https://mpmath.org/doc/current/functions/constants.html

3. **SymPy Documentation**:
   - Expression parsing: https://docs.sympy.org/latest/modules/parsing.html
   - Evaluation: https://docs.sympy.org/latest/tutorials/intro-tutorial/basic_operations.html

4. **OEIS API**:
   - JSON endpoint: https://oeis.org/search?fmt=json
   - Search documentation: OEIS website

5. **Ramanujan's Work**:
   - Rogers-Ramanujan continued fraction
   - Nested radicals identities
   - Series representations of π

6. **Genetic Algorithms for Symbolic Regression**:
   - Symbolic regression literature
   - Elegance metrics in mathematical discovery

---

## 16. Configuration File Example

`ramanujan_swarm/config.py`:

```python
from pydantic_settings import BaseSettings
from typing import Literal

class Config(BaseSettings):
    """Configuration management with environment variables."""

    # API Keys
    anthropic_api_key: str

    # Swarm Configuration
    swarm_size: int = 20
    max_generations: int = 100

    # Agent Distribution
    explorer_ratio: float = 0.4  # 40% explorers
    mutator_ratio: float = 0.4   # 40% mutators
    hybrid_ratio: float = 0.2    # 20% hybrids

    # Gene Pool
    gene_pool_size: int = 25

    # Mathematical Settings
    target_constants: list[str] = ["pi", "e", "phi", "apery", "euler"]
    precision_dps: int = 1500

    # Thresholds
    threshold_evolution: float = 1e-12
    threshold_discovery: float = 1e-50

    # Elegance Scoring
    complexity_weight: float = 0.03

    # Output
    output_dir: str = "outputs"
    checkpoint_db: str = "outputs/checkpoints.db"

    # LLM Settings
    model_name: str = "claude-3-5-sonnet-20240620"
    temperature: float = 1.1
    max_tokens: int = 4096

    # Performance
    agent_timeout: int = 30  # seconds
    max_retries: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global config instance
config = Config()
```

---

## 17. Entry Point Implementation

`main.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from ramanujan_swarm.graph import build_graph
from ramanujan_swarm.config import config
from ramanujan_swarm.utils.logger import setup_logger

logger = setup_logger()

async def main():
    """Main entry point for Ramanujan-Swarm."""

    # Load environment
    load_dotenv()

    logger.info("🧬 Ramanujan-Swarm: Autonomous Mathematical Discovery")
    logger.info(f"Swarm Size: {config.swarm_size}")
    logger.info(f"Max Generations: {config.max_generations}")
    logger.info(f"Target Constants: {', '.join(config.target_constants)}")
    logger.info(f"Precision: {config.precision_dps} decimal places")

    # Build graph
    logger.info("Building LangGraph...")
    graph = build_graph()

    # Initial state
    initial_state = {
        "generation": 0,
        "gene_pool": [],
        "current_proposals": [],
        "validated_candidates": [],
        "discoveries": [],
        "swarm_size": config.swarm_size,
        "target_constants": config.target_constants,
        "precision_dps": config.precision_dps,
        "total_expressions_generated": 0,
        "expressions_per_generation": [],
        "best_error_per_generation": [],
    }

    # Run configuration
    run_config = {
        "configurable": {
            "thread_id": f"swarm-{os.getpid()}"
        }
    }

    logger.info("🚀 Starting swarm execution...")

    try:
        # Execute graph
        result = await graph.ainvoke(initial_state, run_config)

        logger.info("✅ Swarm execution completed!")
        logger.info(f"Total Expressions Generated: {result['total_expressions_generated']}")
        logger.info(f"Total Discoveries: {len(result['discoveries'])}")
        logger.info(f"Final Gene Pool Size: {len(result['gene_pool'])}")

        # Print top discoveries
        if result['discoveries']:
            logger.info("\n🎯 Top Discoveries:")
            for i, expr in enumerate(result['discoveries'][:5], 1):
                logger.info(f"  {i}. {expr.formula_str}")
                logger.info(f"     Error: {expr.error:.6e}, Score: {expr.elegance_score:.4f}")

        return result

    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Conclusion

This implementation plan provides a comprehensive, actionable blueprint for building the Ramanujan-Swarm system. The plan is based on:

1. **Extensive Web Research**: Latest LangGraph v0.2+ patterns, mpmath capabilities, OEIS APIs, and genetic algorithm strategies
2. **Proven Patterns**: Map-Reduce with Send API, reducer-based concurrent writes, checkpointing
3. **Mathematical Rigor**: High-precision evaluation, dual-threshold filtering, elegance scoring
4. **Scalability**: 20+ parallel agents, efficient deduplication, gene pool management
5. **Production Ready**: Error handling, logging, checkpointing, output generation

The implementation follows a logical sequence over 4 weeks, with clear milestones and testable components. Each module is well-defined with specific responsibilities, making the system maintainable and extensible.

**Next Steps**:
1. Begin with Phase 1: Foundation (mathematical engine)
2. Validate core algorithms with unit tests
3. Build LangGraph architecture incrementally
4. Scale progressively from 3 → 5 → 10 → 20+ agents
5. Iterate on prompts and parameters based on results

The system is designed to achieve the ambitious goal of discovering novel mathematical identities with genuine AI-driven creativity and evolutionary refinement.
