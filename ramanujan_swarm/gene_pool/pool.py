"""Gene pool data structure for maintaining top candidates."""

from typing import List
from ramanujan_swarm.math_engine.deduplicator import Expression


class GenePool:
    """Maintains the top N expressions across all generations."""

    def __init__(self, max_size: int = 25):
        """Initialize gene pool.

        Args:
            max_size: Maximum number of expressions to keep
        """
        self.max_size = max_size
        self.pool: List[Expression] = []

    def update(
        self, current_pool: List[Expression], new_candidates: List[Expression]
    ) -> None:
        """Update gene pool with new candidates.

        Merges current pool with new candidates and keeps top N by elegance.

        Args:
            current_pool: Existing gene pool
            new_candidates: New expressions from current generation
        """
        # Combine current pool with new candidates
        combined = current_pool + new_candidates

        # Sort by elegance score (lower is better)
        combined.sort(key=lambda e: e.elegance_score)

        # Keep top N
        self.pool = combined[: self.max_size]

    def get_pool(self) -> List[Expression]:
        """Get current gene pool.

        Returns:
            List of top expressions
        """
        return self.pool

    def get_best(self, n: int = 5) -> List[Expression]:
        """Get the N best expressions.

        Args:
            n: Number of expressions to return

        Returns:
            Top N expressions by elegance
        """
        return self.pool[:n]

    def get_random_sample(self, n: int = 3) -> List[Expression]:
        """Get random sample from gene pool for mutation.

        Args:
            n: Number of expressions to sample

        Returns:
            Random sample of expressions
        """
        import random

        if len(self.pool) == 0:
            return []

        sample_size = min(n, len(self.pool))
        return random.sample(self.pool, sample_size)

    def size(self) -> int:
        """Get current size of gene pool.

        Returns:
            Number of expressions in pool
        """
        return len(self.pool)

    def is_empty(self) -> bool:
        """Check if gene pool is empty.

        Returns:
            True if empty
        """
        return len(self.pool) == 0
