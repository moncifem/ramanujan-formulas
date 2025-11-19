"""
PSLQ (Partial Sum of Least Squares) Algorithm Implementation.
For detecting integer relations between mathematical constants.
This helps identify whether a discovered value is a linear combination of known constants.
"""

from mpmath import mp
from typing import List, Optional, Tuple
import numpy as np


class PSLQDetector:
    """
    PSLQ algorithm for finding integer relations.

    Given n real numbers x1, x2, ..., xn, finds integers a1, a2, ..., an
    such that a1*x1 + a2*x2 + ... + an*xn = 0 (or very close to 0).
    """

    def __init__(self, precision: int = 100):
        """
        Initialize PSLQ detector.

        Args:
            precision: Decimal precision for mpmath calculations
        """
        self.precision = precision
        self._prev_dps = mp.dps

    def find_relation(
        self,
        values: List[mp.mpf],
        max_coeff: int = 10000,
        tolerance: float = None
    ) -> Optional[List[int]]:
        """
        Find integer relation between given values.

        Args:
            values: List of high-precision numbers to find relation for
            max_coeff: Maximum absolute value for coefficients
            tolerance: Error tolerance (default: 10^(-precision/2))

        Returns:
            List of integers [a1, a2, ..., an] if relation found, None otherwise
        """
        if len(values) < 2:
            return None

        # Set precision
        mp.dps = self.precision

        if tolerance is None:
            tolerance = mp.mpf(10) ** (-self.precision // 2)

        try:
            # Use mpmath's built-in PSLQ if available
            if hasattr(mp, 'pslq'):
                result = mp.pslq(values, maxcoeff=max_coeff, tol=tolerance)
                if result:
                    # Verify the relation
                    error = self._compute_relation_error(values, result)
                    if error < tolerance:
                        return [int(c) for c in result]
            else:
                # Fallback to custom implementation
                result = self._pslq_custom(values, max_coeff, tolerance)
                if result:
                    return result

        finally:
            # Restore precision
            mp.dps = self._prev_dps

        return None

    def _pslq_custom(
        self,
        x: List[mp.mpf],
        max_coeff: int,
        tolerance: float
    ) -> Optional[List[int]]:
        """
        Custom PSLQ implementation as fallback.
        Based on the Ferguson-Bailey PSLQ algorithm.
        """
        n = len(x)

        # Check for trivial cases
        for i in range(n):
            if abs(x[i]) < tolerance:
                # Found x[i] ≈ 0
                result = [0] * n
                result[i] = 1
                return result

        # Initialize matrices
        A = mp.matrix(n, n-1)
        B = mp.eye(n)
        H = mp.matrix(n, n-1)

        # Compute initial H matrix (Gram-Schmidt-like)
        s = [mp.zero] * n
        s[n-1] = mp.fsum([xi**2 for xi in x])

        for i in range(n-2, -1, -1):
            s[i] = s[i+1] + x[i]**2

        for i in range(n):
            for j in range(min(i, n-1)):
                if i == j:
                    H[i, j] = mp.sqrt(s[j+1]) / mp.sqrt(s[j])
                elif i == j + 1:
                    H[i, j] = -x[j] * mp.sqrt(s[j+1]) / (mp.sqrt(s[j]) * mp.sqrt(s[j+1]))
                else:
                    H[i, j] = mp.zero

        # Main PSLQ loop
        max_iterations = 10 * n * int(mp.log10(max_coeff))

        for iteration in range(max_iterations):
            # Find pivot
            gamma = mp.sqrt(mp.mpf(4)/mp.mpf(3))
            min_val = mp.inf
            pivot = 0

            for i in range(n-1):
                val = gamma**(i+1) * abs(H[i, i])
                if val < min_val:
                    min_val = val
                    pivot = i

            # Exchange rows
            if pivot < n - 2:
                # Swap rows in H and corresponding operations
                H[pivot, :], H[pivot+1, :] = H[pivot+1, :], H[pivot, :]
                B[pivot, :], B[pivot+1, :] = B[pivot+1, :], B[pivot, :]

            # Corner reduction
            if abs(H[pivot, pivot]) < tolerance:
                # Possible relation found
                # Extract integer relation from B matrix
                for i in range(n):
                    relation = [int(round(float(B[i, j]))) for j in range(n)]
                    error = self._compute_relation_error(x, relation)
                    if error < tolerance and max(abs(c) for c in relation) <= max_coeff:
                        return relation

            # Reduction step (simplified)
            if pivot < n - 1:
                t = H[pivot+1, pivot] / H[pivot, pivot]
                H[pivot+1, :] = H[pivot+1, :] - t * H[pivot, :]
                B[pivot+1, :] = B[pivot+1, :] - t * B[pivot, :]

        return None

    def _compute_relation_error(
        self,
        values: List[mp.mpf],
        coefficients: List[int]
    ) -> mp.mpf:
        """
        Compute error of integer relation.

        Args:
            values: Original values
            coefficients: Integer coefficients

        Returns:
            Absolute error of the relation
        """
        total = mp.zero
        for val, coeff in zip(values, coefficients):
            total += mp.mpf(coeff) * val
        return abs(total)

    def check_linear_dependency(
        self,
        value: mp.mpf,
        constants: List[Tuple[str, mp.mpf]],
        max_terms: int = 5
    ) -> Optional[Tuple[List[Tuple[str, int]], float]]:
        """
        Check if a value can be expressed as a linear combination of known constants.

        Args:
            value: The value to check
            constants: List of (name, value) pairs for known constants
            max_terms: Maximum number of constants to use in combination

        Returns:
            Tuple of ([(constant_name, coefficient), ...], error) if found, None otherwise
        """
        # Try combinations of increasing size
        for num_terms in range(2, min(max_terms + 1, len(constants) + 2)):
            # Always include the value itself as first term
            test_values = [value]
            test_names = ["value"]

            # Add constants
            for name, const_val in constants[:num_terms-1]:
                test_values.append(const_val)
                test_names.append(name)

            # Find relation
            relation = self.find_relation(test_values)

            if relation and relation[0] != 0:
                # Normalize so coefficient of value is 1
                factor = relation[0]
                normalized = []

                for i in range(1, len(relation)):
                    if relation[i] != 0:
                        coeff = -relation[i] / factor
                        # Check if coefficient is close to an integer
                        int_coeff = int(round(float(coeff)))
                        if abs(coeff - int_coeff) < 1e-10:
                            normalized.append((test_names[i], int_coeff))

                if normalized:
                    # Compute error
                    reconstructed = mp.zero
                    for const_name, coeff in normalized:
                        for name, val in constants:
                            if name == const_name:
                                reconstructed += mp.mpf(coeff) * val
                                break

                    error = abs(value - reconstructed)
                    if error < mp.mpf(10) ** (-self.precision // 3):
                        return (normalized, float(error))

        return None


def analyze_discovery_with_pslq(
    expression: str,
    value: mp.mpf,
    error: float
) -> Tuple[bool, Optional[str]]:
    """
    Analyze a discovery using PSLQ to check for known relations.

    Args:
        expression: The mathematical expression
        value: The computed value
        error: The error from nearest integer

    Returns:
        (is_novel, explanation) - True if likely novel, False if reducible to known constants
    """
    from .config import CONSTANTS

    # Skip if error is too large (not a near-integer)
    if error > 1e-10:
        return (True, None)

    # Create PSLQ detector
    detector = PSLQDetector(precision=100)

    # Build list of common constants to check against
    common_constants = [
        ("pi", CONSTANTS["pi"]),
        ("e", CONSTANTS["e"]),
        ("phi", CONSTANTS["phi"]),
        ("sqrt2", CONSTANTS["sqrt2"]),
        ("sqrt3", CONSTANTS["sqrt3"]),
        ("sqrt5", CONSTANTS["sqrt5"]),
        ("ln2", CONSTANTS["ln2"]),
        ("zeta3", CONSTANTS["zeta3"]),
        ("catalan", CONSTANTS["catalan"]),
        ("euler", CONSTANTS["euler"]),
    ]

    # Check if value is a linear combination of known constants
    result = detector.check_linear_dependency(value, common_constants, max_terms=4)

    if result:
        terms, rel_error = result
        # Format the linear combination
        parts = []
        for const_name, coeff in terms:
            if coeff == 1:
                parts.append(const_name)
            elif coeff == -1:
                parts.append(f"-{const_name}")
            elif coeff > 0:
                parts.append(f"{coeff}*{const_name}")
            else:
                parts.append(f"{coeff}*{const_name}")

        combination = " + ".join(parts).replace("+ -", "- ")
        explanation = f"Linear combination: {combination} (error: {rel_error:.2e})"
        return (False, explanation)

    # Also check for simple rational multiples of pi
    pi_ratio = value / CONSTANTS["pi"]
    if abs(pi_ratio - round(float(pi_ratio))) < 1e-10:
        n = int(round(float(pi_ratio)))
        if abs(n) <= 100:
            return (False, f"Simple multiple of π: {n}π")

    # Check for powers of known constants
    for const_name, const_val in common_constants[:5]:  # Check pi, e, phi, sqrt2, sqrt3
        for power in range(2, 6):
            if abs(value - const_val**power) < 1e-10:
                return (False, f"Power of {const_name}: {const_name}^{power}")
            if abs(value - const_val**(-power)) < 1e-10:
                return (False, f"Negative power of {const_name}: {const_name}^(-{power})")

    return (True, None)


# Example usage and testing
if __name__ == "__main__":
    # Test PSLQ with known relation: e^π - π = 20
    # (This is actually false, just for testing)

    mp.dps = 50

    # Test 1: Find that φ² - φ - 1 = 0 (golden ratio property)
    phi = (1 + mp.sqrt(5)) / 2
    values = [phi**2, phi, mp.one]

    detector = PSLQDetector(precision=30)
    relation = detector.find_relation(values)

    if relation:
        print(f"Found relation for φ: {relation}")
        print(f"Verification: {relation[0]}*φ² + {relation[1]}*φ + {relation[2]} = 0")

    # Test 2: Check if a value is a known combination
    test_value = mp.pi * mp.e + mp.sqrt(2)  # π*e + √2

    constants = [
        ("π", mp.pi),
        ("e", mp.e),
        ("√2", mp.sqrt(2)),
        ("φ", phi),
    ]

    result = detector.check_linear_dependency(test_value, constants)
    if result:
        terms, error = result
        print(f"\nValue is a combination of: {terms}")
        print(f"Error: {error}")