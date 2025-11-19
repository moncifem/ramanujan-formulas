"""Expression validator and parser."""

import hashlib
import re
from typing import Optional
from sympy import sympify, SympifyError
import mpmath as mp


class Validator:
    """Validates and parses mathematical expressions."""

    def __init__(self, precision_dps: int = 100):
        self.precision_dps = precision_dps

    def parse_expression(self, formula_str: str) -> Optional[str]:
        """Parse and validate a mathematical expression string.

        Args:
            formula_str: Raw expression string from LLM

        Returns:
            Canonical SymPy string or None if invalid
        """
        try:
            # Preprocess the string
            cleaned = self._preprocess(formula_str)

            # Try to parse with SymPy
            expr = sympify(cleaned)

            # Return canonical string representation
            return str(expr)

        except (SympifyError, ValueError, TypeError, AttributeError):
            return None

    def _preprocess(self, formula_str: str) -> str:
        """Clean up expression string before parsing.

        Args:
            formula_str: Raw expression string

        Returns:
            Cleaned string
        """
        # Remove markdown code blocks
        formula_str = re.sub(r"```.*?```", "", formula_str, flags=re.DOTALL)
        formula_str = re.sub(r"`([^`]+)`", r"\1", formula_str)

        # Strip whitespace
        formula_str = formula_str.strip()

        # Replace common Unicode characters
        replacements = {
            "π": "pi",
            "φ": "phi",
            "γ": "euler",
            "×": "*",
            "÷": "/",
            "√": "sqrt",
        }
        for old, new in replacements.items():
            formula_str = formula_str.replace(old, new)

        return formula_str

    def syntax_hash(self, expression_str: str) -> str:
        """Compute structural hash of expression for deduplication.

        Args:
            expression_str: Canonical SymPy expression string

        Returns:
            SHA-256 hash string
        """
        # Use SymPy canonical form for structural equivalence
        return hashlib.sha256(expression_str.encode()).hexdigest()[:16]

    def numeric_hash(self, numeric_value: mp.mpf, digits: int = 100) -> str:
        """Compute numeric hash based on high-precision value.

        Args:
            numeric_value: mpmath value
            digits: Number of digits to use for hashing

        Returns:
            Hash string
        """
        # Convert to string with fixed precision
        value_str = mp.nstr(numeric_value, digits, strip_zeros=False)
        return hashlib.sha256(value_str.encode()).hexdigest()[:16]

    def is_valid_syntax(self, formula_str: str) -> bool:
        """Check if expression has valid syntax.

        Args:
            formula_str: Expression string

        Returns:
            True if valid, False otherwise
        """
        return self.parse_expression(formula_str) is not None
