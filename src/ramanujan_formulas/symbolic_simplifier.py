"""
Symbolic Simplification Pipeline.
Attempts to simplify discovered expressions using symbolic algebra.
"""

from mpmath import mp
import sympy as sp
from typing import Optional, Tuple, List, Any
import re


class SymbolicSimplifier:
    """
    Simplifies mathematical expressions symbolically to find cleaner forms.
    """

    def __init__(self):
        """Initialize the symbolic simplifier."""
        # Define symbolic constants
        self.sym_constants = {
            'pi': sp.pi,
            'e': sp.E,
            'phi': sp.GoldenRatio,
            'sqrt2': sp.sqrt(2),
            'sqrt3': sp.sqrt(3),
            'sqrt5': sp.sqrt(5),
            'ln2': sp.log(2),
            'catalan': sp.Catalan,
            'euler': sp.EulerGamma,
        }

    def simplify_expression(self, expr_str: str) -> Tuple[str, bool]:
        """
        Attempt to simplify a mathematical expression symbolically.

        Args:
            expr_str: String representation of expression (using mp.xxx)

        Returns:
            (simplified_expr, was_simplified) tuple
        """
        try:
            # Convert from mpmath to sympy syntax
            sympy_expr = self._convert_to_sympy(expr_str)

            if sympy_expr is None:
                return (expr_str, False)

            # Parse the expression
            expr = sp.sympify(sympy_expr, locals=self.sym_constants)

            # Apply various simplification strategies
            simplified = None
            original_complexity = self._compute_complexity(expr)

            # Try different simplification approaches
            strategies = [
                lambda e: sp.simplify(e),
                lambda e: sp.trigsimp(e),
                lambda e: sp.expand_trig(e),
                lambda e: sp.factor(e),
                lambda e: sp.cancel(e),
                lambda e: sp.radsimp(e),
                lambda e: sp.powsimp(e),
                lambda e: sp.logcombine(e, force=True),
            ]

            best_expr = expr
            best_complexity = original_complexity

            for strategy in strategies:
                try:
                    candidate = strategy(expr)
                    complexity = self._compute_complexity(candidate)

                    if complexity < best_complexity:
                        best_expr = candidate
                        best_complexity = complexity
                except:
                    continue

            # Convert back to mpmath syntax
            if best_complexity < original_complexity:
                simplified_str = self._convert_to_mpmath(str(best_expr))
                return (simplified_str, True)

            return (expr_str, False)

        except Exception as e:
            # If simplification fails, return original
            return (expr_str, False)

    def _convert_to_sympy(self, expr_str: str) -> Optional[str]:
        """
        Convert mpmath expression to sympy syntax.
        """
        try:
            result = expr_str

            # Replace mpmath functions with sympy equivalents
            replacements = [
                (r'mp\.pi\b', 'pi'),
                (r'mp\.e\b', 'e'),
                (r'mp\.phi\b', 'phi'),
                (r'mp\.sqrt\(', 'sqrt('),
                (r'mp\.exp\(', 'exp('),
                (r'mp\.log\(', 'log('),
                (r'mp\.ln\(', 'log('),
                (r'mp\.sin\(', 'sin('),
                (r'mp\.cos\(', 'cos('),
                (r'mp\.tan\(', 'tan('),
                (r'mp\.sinh\(', 'sinh('),
                (r'mp\.cosh\(', 'cosh('),
                (r'mp\.tanh\(', 'tanh('),
                (r'mp\.gamma\(', 'gamma('),
                (r'mp\.factorial\(', 'factorial('),
                (r'mp\.zeta\(', 'zeta('),
                (r'mp\.catalan\b', 'catalan'),
                (r'mp\.euler\b', 'euler'),
            ]

            for pattern, replacement in replacements:
                result = re.sub(pattern, replacement, result)

            # Handle special functions that sympy might not support
            if 'polylog' in result or 'ellipk' in result or 'agm' in result:
                # These require special handling
                return None

            return result

        except:
            return None

    def _convert_to_mpmath(self, expr_str: str) -> str:
        """
        Convert sympy expression back to mpmath syntax.
        """
        result = expr_str

        # Replace sympy notation with mpmath
        replacements = [
            (r'\bpi\b', 'mp.pi'),
            (r'\be\b', 'mp.e'),
            (r'\bE\b', 'mp.e'),
            (r'GoldenRatio', 'mp.phi'),
            (r'\bphi\b', 'mp.phi'),
            (r'sqrt\(', 'mp.sqrt('),
            (r'exp\(', 'mp.exp('),
            (r'log\(', 'mp.log('),
            (r'sin\(', 'mp.sin('),
            (r'cos\(', 'mp.cos('),
            (r'tan\(', 'mp.tan('),
            (r'sinh\(', 'mp.sinh('),
            (r'cosh\(', 'mp.cosh('),
            (r'tanh\(', 'mp.tanh('),
            (r'gamma\(', 'mp.gamma('),
            (r'factorial\(', 'mp.factorial('),
            (r'zeta\(', 'mp.zeta('),
            (r'Catalan', 'mp.catalan'),
            (r'catalan', 'mp.catalan'),
            (r'EulerGamma', 'mp.euler'),
            (r'euler', 'mp.euler'),
        ]

        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)

        return result

    def _compute_complexity(self, expr) -> int:
        """
        Compute a complexity score for an expression.
        Lower is simpler.
        """
        try:
            # Count operations and depth
            str_expr = str(expr)
            complexity = len(str_expr)

            # Penalize certain patterns
            complexity += str_expr.count('**') * 5  # Powers
            complexity += str_expr.count('gamma') * 10  # Gamma functions
            complexity += str_expr.count('zeta') * 10  # Zeta functions

            # Reward simple forms
            if str_expr.count('+') + str_expr.count('-') <= 1:
                complexity -= 10
            if str_expr.count('*') + str_expr.count('/') <= 2:
                complexity -= 5

            return complexity

        except:
            return 999999


class AlgebraicRecognizer:
    """
    Attempts to recognize algebraic numbers and express them in radical form.
    """

    def __init__(self):
        """Initialize the algebraic recognizer."""
        pass

    def recognize_algebraic(
        self,
        value: mp.mpf,
        max_degree: int = 6
    ) -> Optional[Tuple[str, List[int]]]:
        """
        Try to recognize if a value is an algebraic number.

        Args:
            value: The numerical value to check
            max_degree: Maximum polynomial degree to check

        Returns:
            (radical_form, minimal_polynomial_coeffs) if algebraic, None otherwise
        """
        # Try to find minimal polynomial using LLL/PSLQ
        for degree in range(2, max_degree + 1):
            # Build powers of value
            powers = [value**i for i in range(degree + 1)]

            # Use PSLQ to find integer relation
            from .pslq import PSLQDetector
            detector = PSLQDetector(precision=100)
            coeffs = detector.find_relation(powers)

            if coeffs and coeffs[-1] != 0:
                # Found a polynomial - try to express in radicals
                radical_form = self._express_as_radical(coeffs)
                if radical_form:
                    return (radical_form, coeffs)

        return None

    def _express_as_radical(self, coeffs: List[int]) -> Optional[str]:
        """
        Try to express polynomial roots in radical form.
        """
        # This is simplified - real implementation would use Galois theory
        degree = len(coeffs) - 1

        if degree == 2:
            # Quadratic: ax² + bx + c = 0
            a, b, c = coeffs[2], coeffs[1], coeffs[0]
            if a != 0:
                # x = (-b ± √(b²-4ac)) / 2a
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    return f"(-{b} + mp.sqrt({discriminant})) / {2*a}"

        elif degree == 3:
            # Cubic - use Cardano's formula (simplified)
            # Would need full implementation for general case
            pass

        return None


class InverseSymbolicCalculator:
    """
    Inverse Symbolic Calculator - given a numerical value,
    tries to find a symbolic expression that produces it.
    """

    def __init__(self):
        """Initialize the ISC."""
        self.simplifier = SymbolicSimplifier()
        self.recognizer = AlgebraicRecognizer()

    def find_symbolic_form(
        self,
        value: mp.mpf,
        error: float
    ) -> Optional[str]:
        """
        Try to find a symbolic form for a numerical value.

        Args:
            value: The numerical value
            error: Error from nearest integer

        Returns:
            Symbolic expression if found, None otherwise
        """
        # Strategy 1: Check if it's a simple fraction
        frac = self._check_simple_fraction(value)
        if frac:
            return frac

        # Strategy 2: Check if it's a known constant combination
        const_combo = self._check_constant_combination(value)
        if const_combo:
            return const_combo

        # Strategy 3: Check if it's algebraic
        algebraic = self.recognizer.recognize_algebraic(value)
        if algebraic:
            return algebraic[0]

        # Strategy 4: Check for logarithmic forms
        log_form = self._check_logarithmic(value)
        if log_form:
            return log_form

        return None

    def _check_simple_fraction(self, value: mp.mpf) -> Optional[str]:
        """Check if value is a simple fraction."""
        # Try to express as p/q with small p, q
        for q in range(1, 1000):
            p = int(round(float(value * q)))
            if abs(value - mp.mpf(p) / mp.mpf(q)) < 1e-10:
                if abs(p) <= 10000:
                    return f"{p}/{q}" if q != 1 else str(p)

        return None

    def _check_constant_combination(self, value: mp.mpf) -> Optional[str]:
        """Check if value is a simple combination of constants."""
        from .config import CONSTANTS

        # Check single constants
        for name, const in CONSTANTS.items():
            if abs(value - const) < 1e-10:
                return f"mp.{name}"

            # Check simple multiples
            for n in range(2, 10):
                if abs(value - n * const) < 1e-10:
                    return f"{n} * mp.{name}"
                if abs(value - const / n) < 1e-10:
                    return f"mp.{name} / {n}"

        # Check sums/differences of two constants
        const_list = list(CONSTANTS.items())
        for i, (name1, val1) in enumerate(const_list):
            for name2, val2 in const_list[i+1:]:
                if abs(value - (val1 + val2)) < 1e-10:
                    return f"mp.{name1} + mp.{name2}"
                if abs(value - abs(val1 - val2)) < 1e-10:
                    return f"abs(mp.{name1} - mp.{name2})"

        return None

    def _check_logarithmic(self, value: mp.mpf) -> Optional[str]:
        """Check if value has a logarithmic form."""
        # Check if exp(value) is recognizable
        try:
            # Only try exp if value is reasonable
            if abs(value) < 10:  # exp(10) ≈ 22026, reasonable limit
                exp_val = mp.exp(value)

                # Check if it's a simple integer
                exp_float = float(exp_val)
                if not mp.isinf(exp_val) and not mp.isnan(exp_val):
                    if abs(exp_float) < 1e10:  # Reasonable range
                        rounded = round(exp_float)
                        if abs(exp_float - rounded) < 1e-10:
                            n = int(rounded)
                            if 2 <= n <= 1000:
                                return f"mp.log({n})"
        except (OverflowError, ValueError):
            pass

        # Check if log(value) is recognizable
        if value > 0:
            log_val = mp.log(value)
            if abs(log_val - round(float(log_val))) < 1e-10:
                n = int(round(float(log_val)))
                if abs(n) <= 10:
                    return f"mp.exp({n})"

        return None


# Integration function for use in the validator
def try_simplify_expression(expr: str, value: mp.mpf, error: float) -> Tuple[str, str]:
    """
    Try to simplify an expression both symbolically and numerically.

    Args:
        expr: Original expression string
        value: Computed numerical value
        error: Error from nearest integer

    Returns:
        (simplified_expr, simplification_note)
    """
    simplifier = SymbolicSimplifier()
    isc = InverseSymbolicCalculator()

    # Try symbolic simplification
    simplified, was_simplified = simplifier.simplify_expression(expr)

    note_parts = []

    if was_simplified:
        note_parts.append("algebraically simplified")

    # Try inverse symbolic calculation
    symbolic_form = isc.find_symbolic_form(value, error)
    if symbolic_form:
        note_parts.append(f"value = {symbolic_form}")

    if note_parts:
        return (simplified, "; ".join(note_parts))

    return (expr, "no simplification found")


# Example usage
if __name__ == "__main__":
    mp.dps = 50

    # Test symbolic simplifier
    simplifier = SymbolicSimplifier()

    test_expressions = [
        "mp.exp(mp.log(mp.pi))",  # Should simplify to mp.pi
        "mp.sin(mp.pi/2)",  # Should simplify to 1
        "mp.gamma(3)",  # Should simplify to 2 (factorial)
        "(mp.phi**2 - mp.phi)",  # Should simplify to 1
    ]

    print("Testing Symbolic Simplifier:")
    for expr in test_expressions:
        simplified, was_simplified = simplifier.simplify_expression(expr)
        status = "✓" if was_simplified else "✗"
        print(f"  {status} {expr} → {simplified}")

    # Test inverse symbolic calculator
    isc = InverseSymbolicCalculator()

    print("\nTesting Inverse Symbolic Calculator:")
    test_values = [
        (mp.pi / 2, "pi/2"),
        (mp.sqrt(2), "sqrt(2)"),
        (mp.log(10), "log(10)"),
        (mp.e + mp.pi, "e + pi"),
    ]

    for value, expected in test_values:
        result = isc.find_symbolic_form(value, 1e-10)
        status = "✓" if result else "✗"
        print(f"  {status} {expected}: {result}")