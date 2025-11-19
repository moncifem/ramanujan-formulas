"""State definitions for LangGraph."""

from typing import TypedDict, Annotated, List
from operator import add

# Re-export Expression from math_engine for convenience
from ramanujan_swarm.math_engine.deduplicator import Expression

__all__ = ["Expression", "SwarmState"]


class SwarmState(TypedDict):
    """Global state shared across all nodes.

    Uses Annotated with operator.add for safe concurrent writes from parallel agents.
    """

    # Current generation counter
    generation: int

    # Gene pool: top candidates from all generations
    gene_pool: List[Expression]

    # Current generation's raw proposals (uses reducer for concurrent writes)
    current_proposals: Annotated[List[Expression], add]

    # Filtered & scored candidates ready for gene pool
    validated_candidates: List[Expression]

    # Discoveries that passed discovery threshold (1e-50)
    discoveries: Annotated[List[Expression], add]

    # Configuration
    swarm_size: int
    target_constants: List[str]
    precision_dps: int
    max_generations: int

    # Metrics
    total_expressions_generated: int
    expressions_per_generation: List[int]
    best_error_per_generation: List[float]
