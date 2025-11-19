"""Elegance scoring function for mathematical expressions."""

from ramanujan_swarm.math_engine.deduplicator import Expression


class EleganceScorer:
    """Scores expressions by elegance = error × (1 + complexity_weight × length)."""

    def __init__(self, complexity_weight: float = 0.03):
        """Initialize scorer.

        Args:
            complexity_weight: Weight for complexity penalty (default 0.03 from Ramanujan paper)
        """
        self.complexity_weight = complexity_weight

    def score(self, expr: Expression) -> float:
        """Compute elegance score.

        Lower scores = more elegant (lower error, simpler expression).

        Args:
            expr: Expression to score

        Returns:
            Elegance score
        """
        # Calculate complexity as string length
        complexity = len(expr.parsed_expr)
        expr.complexity = complexity

        # Elegance = Error × (1 + weight × Length)
        score = expr.error * (1 + self.complexity_weight * complexity)

        return score

    def rank_by_elegance(self, expressions: list[Expression]) -> list[Expression]:
        """Rank expressions by elegance score (ascending).

        Args:
            expressions: List of expressions

        Returns:
            Sorted list (most elegant first)
        """
        return sorted(expressions, key=lambda e: e.elegance_score)
