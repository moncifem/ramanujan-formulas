"""
Filter for detecting and rejecting trivial/known mathematical identities.
Prevents rediscovering textbook results like Euler reflection formula.
"""

import re
from typing import Tuple, Optional
from mpmath import mp


def is_euler_reflection_pattern(expr: str) -> bool:
    """
    Detect if expression is a trivial consequence of Euler's reflection formula:
    Γ(x)Γ(1-x) = π/sin(πx)

    Returns True if pattern matches, meaning it should be REJECTED.
    """
    # Pattern: gamma(a/b) * gamma((b-a)/b) with sin/pi
    # Example: gamma(2/5) * gamma(3/5) / (pi/sin(2*pi/5))

    # Check for complementary gamma products
    if 'gamma' in expr.lower():
        # Look for gamma(x) * gamma(...) patterns
        gamma_pattern = r'gamma\(\s*(\d+)\s*/\s*(\d+)\s*\)'
        matches = re.findall(gamma_pattern, expr.lower())

        if len(matches) >= 2:
            # Check if they're complementary (a/b and (b-a)/b)
            fractions = [(int(a), int(b)) for a, b in matches]
            for i, (a1, b1) in enumerate(fractions):
                for a2, b2 in fractions[i+1:]:
                    if b1 == b2 and a1 + a2 == b1:
                        # Complementary pair found!
                        # Check if sin/pi are present
                        if 'sin' in expr.lower() and 'pi' in expr.lower():
                            return True

    return False


def is_gauss_multiplication_pattern(expr: str) -> bool:
    """
    Detect if expression is Gauss multiplication formula:
    ∏ Γ(k/n) = known closed form

    Returns True if trivial.
    """
    # Pattern: product of multiple gamma(k/n) with same denominator
    if expr.count('gamma') >= 3:
        gamma_pattern = r'gamma\(\s*\d+\s*/\s*(\d+)\s*\)'
        denominators = re.findall(gamma_pattern, expr.lower())

        if denominators:
            # If all denominators are same and there are 3+ terms
            if len(set(denominators)) == 1 and len(denominators) >= 3:
                return True

    return False


def is_heegner_trivial(expr: str) -> bool:
    """
    Detect if expression is trivial Heegner number identity.

    Known trivial patterns:
    - exp(π√163) and simple variations
    - exp(π√n) for n in {19, 43, 67, 163, 232, 427, 522, 652}
    """
    known_heegner = {'19', '43', '67', '163', '232', '427', '522', '652'}

    # Pattern: exp(pi * sqrt(n))
    heegner_pattern = r'exp\(.*sqrt\((\d+)\)'
    matches = re.findall(heegner_pattern, expr.lower())

    for match in matches:
        if match in known_heegner:
            return True

    return False


def is_hyperbolic_asymptotic(expr: str) -> bool:
    """
    Detect if expression is trivial hyperbolic asymptotic:
    exp(x)/sinh(x) → 2
    exp(x)/cosh(x) → 2
    cosh(x)/sinh(x) → 1

    These are trivial for large x.
    """
    # Pattern: exp(...) / sinh(...) or exp(...) / cosh(...)
    if ('exp(' in expr.lower() and 'sinh(' in expr.lower()) or \
       ('exp(' in expr.lower() and 'cosh(' in expr.lower()):

        # Check if the argument to exp and sinh/cosh are similar
        # (indicating it's the trivial asymptotic pattern)
        if 'sqrt' in expr.lower() and any(n in expr for n in ['163', '232', '43', '67']):
            return True

    return False


def is_trivial_identity(expr: str, value: mp.mpf, error: float) -> Tuple[bool, Optional[str]]:
    """
    Master filter: Check if expression is a known trivial identity.

    Returns:
        (is_trivial, reason)
    """
    # Check Euler reflection
    if is_euler_reflection_pattern(expr):
        return (True, "Euler reflection formula (textbook identity)")

    # Check Gauss multiplication
    if is_gauss_multiplication_pattern(expr):
        return (True, "Gauss multiplication formula (classical identity)")

    # Check Heegner trivial
    if is_heegner_trivial(expr):
        return (True, "Known Heegner number (classical, discovered 1952)")

    # Check hyperbolic asymptotic
    if is_hyperbolic_asymptotic(expr):
        return (True, "Trivial hyperbolic asymptotic (exp(x)/sinh(x) → 2)")

    # Check if value is exactly 1, 2, 3, 4 with tiny error
    # (likely trivial identity we missed)
    if error < 1e-40:
        nearest = int(round(float(value)))
        if 1 <= nearest <= 10 and abs(value - nearest) < 1e-40:
            return (True, f"Exact integer {nearest} (likely trivial identity)")

    return (False, None)
