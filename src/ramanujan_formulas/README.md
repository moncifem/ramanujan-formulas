# Ramanujan-Swarm Source Code

## Module Structure

### Core Modules

- **`config.py`**: Configuration management and constants
  - Mathematical constants (π, e, φ, etc.)
  - LLM settings
  - Thresholds and parameters
  - Environment variable loading

- **`types.py`**: Type definitions
  - `Candidate`: Mathematical formula candidate
  - `ProposerInput`: Agent input state
  - `State`: Global graph state with reducers

- **`utils.py`**: Utility functions
  - Expression evaluation with mpmath
  - Error computation
  - Elegance scoring
  - Deduplication helpers

- **`verification.py`**: Novelty checking
  - OEIS database queries
  - Discovery logging (JSON + Markdown)
  - Result caching

- **`agents.py`**: Proposer agent implementation
  - Exploration mode: Generate novel expressions
  - Exploitation mode: Mutate best candidates
  - LLM prompt engineering

- **`graph.py`**: LangGraph orchestration
  - State graph definition
  - Validator node logic
  - Routing and swarm dispatch
  - MemorySaver checkpointing

- **`main.py`**: Entry point
  - Initialization and setup
  - Execution loop
  - Result reporting

## Data Flow

```
START
  ↓
Dispatch → [Proposer Agents x20] (Parallel)
  ↓
Validator Node (Sequential)
  ├─ Evaluate expressions
  ├─ Compute errors
  ├─ Update gene pool
  └─ Check for discoveries
  ↓
Route Decision
  ├─ Continue → Dispatch (next iteration)
  └─ Stop → END
```

## Key Algorithms

### Elegance Scoring

```python
score = error × (1 + 0.03 × len(expression))
```

Balances precision with simplicity.

### Exploration Rate

```python
rate = max(0.1, 0.5 - (iteration / max_iterations))
```

Decreases over time: start with exploration, converge to exploitation.

### Deduplication

Two-level filtering:
1. **Structural**: Hash of expression string
2. **Semantic**: First 20 digits of computed value

## Extension Points

### Adding New Constants

In `config.py`:

```python
CONSTANTS = {
    # ... existing ...
    "my_constant": mp.zeta(5),
}
```

### Custom Verification

Implement in `verification.py`:

```python
def check_custom_database(value: mp.mpf) -> dict:
    # Your verification logic
    pass
```

### Different LLM Models

In `.env`:

```
LLM_MODEL=claude-3-opus-20240229
```

Or programmatically in `config.py`.

## Performance Considerations

- **Parallel Agents**: Increase `SWARM_SIZE` for more exploration (higher API cost)
- **Precision**: `mp.dps = 1500` is expensive; reduce for faster evaluation
- **Caching**: Verification results are cached automatically
- **Checkpointing**: LangGraph MemorySaver enables resume from failure

## Testing

```bash
# Run basic test
python -c "from src.ramanujan_formulas.config import validate_config; validate_config()"

# Test expression evaluation
python -c "from src.ramanujan_formulas.utils import evaluate_expression; print(evaluate_expression('mp.pi'))"
```

