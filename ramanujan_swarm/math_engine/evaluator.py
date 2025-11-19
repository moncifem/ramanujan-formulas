"""High-precision expression evaluator using mpmath."""

import mpmath as mp
from sympy import sympify, lambdify
from typing import Optional
from ramanujan_swarm.constants import get_constant_value


class Evaluator:
    """Evaluates mathematical expressions at high precision."""

    def __init__(self, precision_dps: int = 100):
        self.precision_dps = precision_dps
        mp.mp.dps = precision_dps

    def evaluate(self, expression_str: str, target_constant: str) -> Optional[mp.mpf]:
        """Evaluate a symbolic expression to high-precision numeric value.

        Args:
            expression_str: SymPy expression string
            target_constant: Name of target constant for comparison

        Returns:
            High-precision mpmath value or None if evaluation fails
        """
        try:
            # Parse with SymPy
            expr = sympify(expression_str)

            # Convert to numerical function using mpmath
            func = lambdify([], expr, modules="mpmath")

            # Evaluate at high precision
            original_dps = mp.mp.dps
            mp.mp.dps = self.precision_dps

            result = func()

            # Convert to mpf
            if isinstance(result, (int, float)):
                result = mp.mpf(result)
            elif isinstance(result, complex):
                # For complex results, take real part if imaginary is negligible
                if abs(result.imag) < 1e-100:
                    result = mp.mpf(result.real)
                else:
                    mp.mp.dps = original_dps
                    return None

            mp.mp.dps = original_dps
            return result

        except Exception as e:
            # Silently fail - most LLM-generated expressions will have errors
            return None

    def compute_error(
        self, numeric_value: mp.mpf, target_constant: str
    ) -> float:
        """Compute absolute error between result and target constant.

        Args:
            numeric_value: Computed value
            target_constant: Name of target constant

        Returns:
            Absolute error as float
        """
        try:
            target_value = get_constant_value(target_constant, self.precision_dps)
            error = abs(numeric_value - target_value)
            return float(error)
        except Exception:
            return float("inf")

    def compute_relative_error(
        self, numeric_value: mp.mpf, target_constant: str
    ) -> float:
        """Compute relative error between result and target constant.

        Args:
            numeric_value: Computed value
            target_constant: Name of target constant

        Returns:
            Relative error as float
        """
        try:
            target_value = get_constant_value(target_constant, self.precision_dps)
            if target_value == 0:
                return float("inf")
            error = abs((numeric_value - target_value) / target_value)
            return float(error)
        except Exception:
            return float("inf")
