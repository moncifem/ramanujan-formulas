"""Deduplicator for removing equivalent expressions."""

from typing import List, Set
from dataclasses import dataclass


@dataclass
class Expression:
    """Individual mathematical expression candidate."""

    formula_str: str
    parsed_expr: str
    target_constant: str
    agent_type: str
    generation: int
    numeric_value: str
    error: float
    elegance_score: float = float("inf")
    complexity: int = 0
    hash_syntax: str = ""
    hash_numeric: str = ""
    timestamp: float = 0.0


class Deduplicator:
    """Removes duplicate expressions using dual-hash strategy."""

    def deduplicate(self, expressions: List[Expression]) -> List[Expression]:
        """Remove duplicate expressions.

        Uses both structural (syntax) and numeric hashes to catch
        different forms of equivalent expressions.

        Args:
            expressions: List of expressions to deduplicate

        Returns:
            List of unique expressions
        """
        seen_syntax: Set[str] = set()
        seen_numeric: Set[str] = set()
        unique: List[Expression] = []

        for expr in expressions:
            # Check both hashes
            if (
                expr.hash_syntax not in seen_syntax
                and expr.hash_numeric not in seen_numeric
            ):
                seen_syntax.add(expr.hash_syntax)
                seen_numeric.add(expr.hash_numeric)
                unique.append(expr)

        return unique
