"""
Utility functions for mathematical operations and candidate evaluation.
"""

import math
from typing import Optional, Tuple
from mpmath import mp

from .config import CONSTANTS, COMPLEXITY_PENALTY, CANDIDATE_THRESHOLD


def evaluate_expression(expr: str) -> Optional[mp.mpf]:
    """
    Safely evaluate a mathematical expression using mpmath.
    
    Args:
        expr: Python expression string using mpmath (aliased as 'mp')
        
    Returns:
        mpf value if successful, None if evaluation fails
    """
    try:
        return eval(expr, {"mp": mp}, CONSTANTS)
    except Exception:
        return None


def compute_error(value: mp.mpf) -> Tuple[float, str]:
    """
    Compute the error of a value against nearest integer or known constant.

    Returns:
        Tuple of (error, target_type) where target_type is 'integer' or 'constant'
    """
    # CRITICAL: Reject trivial identities that equal zero
    # We want expressions that evaluate to interesting values (near-integers),
    # NOT expressions that equal zero (trivial identities like phi^2 - phi - 1 = 0)
    abs_value = abs(value)
    if abs_value < 1e-10:
        # This is a trivial zero identity, return infinite error to reject it
        return (float('inf'), "trivial_zero")

    # Also reject values that are exactly equal to a single known constant
    # (we want relationships between constants, not just the constants themselves)
    for const_name, const_val in CONSTANTS.items():
        if abs(value - const_val) < 1e-100:
            return (float('inf'), "trivial_constant")

    # Check distance to nearest integer
    nearest_int = mp.nint(value)
    err_int = abs(value - nearest_int)

    # CRITICAL: Reject exact integers (like Lucas numbers: phi^n + phi^(-n))
    # We want NEAR-integers with small but NON-ZERO error
    # Examples of what we want: exp(pi*sqrt(163)) which is close but not exact
    # Examples of what we DON'T want: phi^10 + phi^(-10) = 123 exactly
    if err_int < 1e-100:
        # This is an exact integer (trivial identity like Lucas numbers)
        return (float('inf'), "exact_integer")

    # For values far from integers, check known constants
    if err_int > 1e-5:
        err_const = min([abs(value - c) for c in CONSTANTS.values()])
        if err_const < err_int:
            # Also reject exact matches to constants
            if err_const < 1e-100:
                return (float('inf'), "exact_constant")
            return (float(err_const), "constant")

    return (float(err_int), "integer")


def compute_elegance_score(error: float, expression: str) -> float:
    """
    Compute elegance score: Error × (1 + complexity_penalty × length).
    
    This encourages both precision and simplicity.
    """
    complexity_factor = 1 + (COMPLEXITY_PENALTY * len(expression))
    return error * complexity_factor


def is_candidate_worthy(error: float) -> bool:
    """Check if an error is small enough to consider for the gene pool."""
    return error < CANDIDATE_THRESHOLD


def is_discovery_worthy(error: float) -> bool:
    """Check if an error is small enough to trigger verification."""
    from .config import DISCOVERY_THRESHOLD
    return error < DISCOVERY_THRESHOLD and error > 0


def format_error_magnitude(error: float) -> str:
    """Format error as 10^n for display."""
    if error == 0:
        return "10^(-∞)"
    try:
        magnitude = int(math.log10(error))
        return f"10^{magnitude}"
    except (ValueError, OverflowError):
        return str(error)


def get_value_signature(value: mp.mpf, length: int = 20) -> str:
    """
    Get a signature string from a value for deduplication.
    
    Args:
        value: The mpmath value
        length: Number of significant digits to include
        
    Returns:
        Signature string without decimal point
    """
    return str(value).replace('.', '').replace('-', '')[:length]


def expression_hash(expr: str) -> int:
    """Generate a hash for expression deduplication."""
    return hash(expr)

