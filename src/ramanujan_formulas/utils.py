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
    # CRITICAL: Reject expressions with large numbers (defeats the purpose of discovery)
    # We want to find that exp(pi*sqrt(163)) ≈ 262537412640768744,
    # NOT use 262537412640768744 as a constant in expressions!
    import re

    # Find all numbers in the expression (excluding decimals like 0.5)
    numbers = re.findall(r'\b(\d{4,})\b', expr)  # Find integers with 4+ digits

    for num_str in numbers:
        num = int(num_str)
        # Reject any large numbers (> 1000)
        # This prevents using computed results as "constants"
        # Example: Don't allow 262537412640768744 or 640320**3
        if num > 1000:
            return None  # Reject: contains large non-fundamental number

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
        # Exception: Allow exact integers if the expression is complex and involves special functions
        # This allows discovering exact identities like theta function relations
        is_complex = any(func in str(CONSTANTS.keys()) for func in ['zeta', 'gamma', 'jtheta', 'ellipk', 'qp'])
        # Actually we need to check the expression string, but we don't have it here in compute_error.
        # We only have the value.
        
        # We can return a special flag or just reject for now to stay true to "Ramanujan Constant" search.
        # But we should probably allow it if we want to find identities.
        
        # Let's just relax the "exact integer" check slightly for now, or rely on the caller to check complexity.
        # But compute_error is the gatekeeper.
        
        # For this hackathon task, let's stick to finding NEAR integers, as that's the specific goal.
        # So exact identities ARE failures for this specific "Ramanujan Machine" type search.
        # We need to find APPROXIMATIONS.
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
    Also heavily penalizes "gaming" patterns like "log(...) - 162"
    """
    complexity_factor = 1 + (COMPLEXITY_PENALTY * len(expression))

    # Detect and penalize "gaming" patterns: expressions ending with ± integer
    # Pattern: something - 162, something + 143, etc.
    import re
    gaming_pattern = r'[+\-]\s*\d{2,}(?:\*\*\d+)?[)\s]*$'
    if re.search(gaming_pattern, expression):
        # Heavy penalty for gaming - multiply score by 1000
        complexity_factor *= 1000

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

