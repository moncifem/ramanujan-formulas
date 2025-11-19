"""
Stability Testing Module.
Tests discovered expressions across different precisions to ensure numerical stability.
"""

from mpmath import mp
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


class StabilityTester:
    """
    Tests mathematical expressions for numerical stability across precisions.
    """

    def __init__(self):
        """Initialize the stability tester."""
        # Test precisions (in decimal digits)
        self.test_precisions = [30, 50, 100, 200, 500, 1000]
        self.original_dps = mp.dps

    def test_expression_stability(
        self,
        expr_str: str,
        target_value: Optional[mp.mpf] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Test an expression's stability across different precisions.

        Args:
            expr_str: Expression string to test
            target_value: Expected value (if known)

        Returns:
            (is_stable, analysis_dict) tuple
        """
        results = []
        errors = []

        for precision in self.test_precisions:
            mp.dps = precision

            try:
                # Evaluate expression at this precision
                value = eval(expr_str, {"mp": mp})

                if value is None or not isinstance(value, (mp.mpf, mp.mpc)):
                    continue

                # Convert to string for precision-independent comparison
                value_str = mp.nstr(value, 50)
                results.append({
                    'precision': precision,
                    'value': value,
                    'value_str': value_str
                })

                # If we have a target, compute error
                if target_value is not None:
                    mp_target = mp.mpf(str(target_value))
                    error = abs(value - mp_target)
                    errors.append(float(error))

            except Exception as e:
                results.append({
                    'precision': precision,
                    'value': None,
                    'error': str(e)
                })

        # Restore original precision
        mp.dps = self.original_dps

        # Analyze stability
        is_stable, analysis = self._analyze_stability(results, errors)

        return is_stable, analysis

    def _analyze_stability(
        self,
        results: List[Dict],
        errors: List[float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze stability test results.
        """
        analysis = {
            'tested_precisions': len(results),
            'successful_evaluations': sum(1 for r in results if r.get('value') is not None),
            'convergence': 'unknown'
        }

        # Check if we have enough successful evaluations
        successful_values = [r for r in results if r.get('value') is not None]

        if len(successful_values) < 2:
            return False, {**analysis, 'reason': 'insufficient_evaluations'}

        # Check for convergence by comparing consecutive precision values
        converging = True
        last_value_str = None
        stable_digits = []

        for result in successful_values:
            value_str = result.get('value_str', '')

            if last_value_str:
                # Count matching digits
                matching = 0
                for i in range(min(len(value_str), len(last_value_str))):
                    if value_str[i] == last_value_str[i]:
                        matching += 1
                    else:
                        break

                stable_digits.append(matching)

                # Check if we're losing precision
                if matching < 10:  # Less than 10 stable digits
                    converging = False

            last_value_str = value_str

        analysis['stable_digits'] = min(stable_digits) if stable_digits else 0
        analysis['convergence'] = 'converging' if converging else 'diverging'

        # Check error behavior if we have target
        if errors:
            analysis['error_trend'] = self._analyze_error_trend(errors)
            is_stable = analysis['error_trend'] in ['decreasing', 'stable']
        else:
            # Without target, check digit stability
            is_stable = analysis['stable_digits'] >= 20

        return is_stable, analysis

    def _analyze_error_trend(self, errors: List[float]) -> str:
        """
        Analyze the trend in errors as precision increases.
        """
        if len(errors) < 2:
            return 'unknown'

        # Check if errors are decreasing
        decreasing = all(errors[i] >= errors[i+1] * 0.9 for i in range(len(errors)-1))
        if decreasing:
            return 'decreasing'

        # Check if errors are stable
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        if std_error / (mean_error + 1e-20) < 0.1:  # Low relative variance
            return 'stable'

        # Check if errors are increasing
        increasing = all(errors[i] <= errors[i+1] * 1.1 for i in range(len(errors)-1))
        if increasing:
            return 'increasing'

        return 'erratic'


class PrecisionExplorer:
    """
    Explores how expressions behave at extreme precisions.
    """

    def __init__(self):
        """Initialize the precision explorer."""
        self.original_dps = mp.dps

    def find_precision_threshold(
        self,
        expr_str: str,
        target_error: float = 1e-100
    ) -> Optional[int]:
        """
        Find minimum precision needed to achieve target error.

        Args:
            expr_str: Expression to test
            target_error: Target error threshold

        Returns:
            Minimum precision in decimal digits, or None if not achievable
        """
        # Binary search for minimum precision
        min_prec = 10
        max_prec = 2000

        last_good_precision = None

        while min_prec <= max_prec:
            test_prec = (min_prec + max_prec) // 2
            mp.dps = test_prec

            try:
                value = eval(expr_str, {"mp": mp})

                if value is None:
                    break

                # Compute error from nearest integer
                nearest = round(float(value))
                error = abs(value - nearest)

                if error < target_error:
                    last_good_precision = test_prec
                    max_prec = test_prec - 1  # Try lower precision
                else:
                    min_prec = test_prec + 1  # Need higher precision

            except:
                min_prec = test_prec + 1

        # Restore precision
        mp.dps = self.original_dps

        return last_good_precision

    def explore_precision_behavior(
        self,
        expr_str: str
    ) -> Dict[str, Any]:
        """
        Explore how an expression behaves across precision spectrum.

        Args:
            expr_str: Expression to explore

        Returns:
            Dictionary with exploration results
        """
        results = {
            'min_viable_precision': None,
            'precision_sweet_spot': None,
            'diminishing_returns': None,
            'special_behaviors': []
        }

        # Find minimum viable precision
        for prec in [10, 20, 30, 50]:
            mp.dps = prec
            try:
                value = eval(expr_str, {"mp": mp})
                if value is not None:
                    results['min_viable_precision'] = prec
                    break
            except:
                continue

        # Find sweet spot (best accuracy/precision ratio)
        best_ratio = 0
        best_prec = None

        for prec in [30, 50, 100, 200, 500]:
            mp.dps = prec
            try:
                value = eval(expr_str, {"mp": mp})
                if value is None:
                    continue

                nearest = round(float(value))
                error = abs(value - nearest)

                if error > 0:
                    # Accuracy per precision digit
                    ratio = -mp.log10(error) / prec
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_prec = prec

            except:
                continue

        results['precision_sweet_spot'] = best_prec

        # Check for diminishing returns
        if best_prec:
            mp.dps = best_prec * 2
            try:
                value1 = eval(expr_str, {"mp": mp})
                mp.dps = best_prec * 4
                value2 = eval(expr_str, {"mp": mp})

                if value1 and value2:
                    improvement = abs(value2 - value1)
                    if improvement < mp.mpf(10) ** (-best_prec):
                        results['diminishing_returns'] = best_prec * 2

            except:
                pass

        # Check for special behaviors
        # (e.g., expressions that suddenly converge at specific precisions)
        convergence_points = []
        last_error = float('inf')

        for prec in [50, 100, 163, 200, 355, 500, 1000]:
            mp.dps = prec
            try:
                value = eval(expr_str, {"mp": mp})
                if value:
                    nearest = round(float(value))
                    error = abs(value - nearest)

                    # Sudden convergence detected
                    if last_error > 1e-10 and error < 1e-50:
                        convergence_points.append(prec)
                        results['special_behaviors'].append(
                            f"Sudden convergence at precision {prec}"
                        )

                    last_error = float(error)

            except:
                continue

        # Restore precision
        mp.dps = self.original_dps

        return results


def validate_discovery_stability(
    expr: str,
    value: mp.mpf,
    error: float
) -> Tuple[bool, str]:
    """
    Validate that a discovered expression is numerically stable.

    Args:
        expr: Expression string
        value: Computed value
        error: Error from nearest integer

    Returns:
        (is_valid, validation_message) tuple
    """
    tester = StabilityTester()
    explorer = PrecisionExplorer()

    # Test basic stability
    is_stable, analysis = tester.test_expression_stability(expr)

    if not is_stable:
        return (False, f"Unstable: {analysis.get('reason', 'convergence issues')}")

    # For high-precision discoveries, check precision requirements
    if error < 1e-50:
        min_prec = explorer.find_precision_threshold(expr, error * 10)

        if min_prec and min_prec > 500:
            return (False, f"Requires extreme precision: {min_prec} digits")

    # Explore precision behavior
    behavior = explorer.explore_precision_behavior(expr)

    # Check for concerning behaviors
    if behavior['min_viable_precision'] and behavior['min_viable_precision'] > 100:
        return (False, f"High minimum precision: {behavior['min_viable_precision']} digits")

    if not behavior['precision_sweet_spot']:
        return (False, "No stable precision sweet spot found")

    # Expression passed all stability tests
    validation_parts = []

    if analysis['stable_digits'] > 0:
        validation_parts.append(f"{analysis['stable_digits']} stable digits")

    if behavior['precision_sweet_spot']:
        validation_parts.append(f"optimal at {behavior['precision_sweet_spot']} dps")

    if behavior['special_behaviors']:
        validation_parts.append(behavior['special_behaviors'][0])

    return (True, "Stable: " + ", ".join(validation_parts))


# Example usage
if __name__ == "__main__":
    print("Testing Stability Module:\n")

    # Test expressions with different stability characteristics
    test_cases = [
        ("mp.exp(mp.pi * mp.sqrt(163))", "Should be stable (Heegner)"),
        ("mp.gamma(1/3) * mp.gamma(2/3) / mp.sqrt(mp.pi)", "Should be stable (Euler reflection)"),
        ("sum([1/mp.factorial(n) for n in range(100)])", "Should be stable (e series)"),
        ("mp.sin(mp.exp(mp.exp(mp.exp(2))))", "Might be unstable (large argument)"),
    ]

    for expr, description in test_cases:
        print(f"Testing: {description}")
        print(f"Expression: {expr[:60]}...")

        # Test stability
        tester = StabilityTester()
        is_stable, analysis = tester.test_expression_stability(expr)

        print(f"  Stable: {is_stable}")
        print(f"  Convergence: {analysis.get('convergence')}")
        print(f"  Stable digits: {analysis.get('stable_digits', 0)}")

        # Validate for discovery
        mp.dps = 100
        try:
            value = eval(expr)
            error = abs(value - round(float(value)))
            is_valid, message = validate_discovery_stability(expr, value, error)
            print(f"  Validation: {message}")
        except:
            print(f"  Validation: Failed to evaluate")

        print()