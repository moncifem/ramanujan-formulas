"""Mathematical constants definitions."""

import mpmath as mp

# Set default precision
mp.mp.dps = 100


def get_constant_value(name: str, dps: int = 100) -> mp.mpf:
    """Get the high-precision value of a mathematical constant.

    Args:
        name: Name of the constant (pi, e, phi, euler, apery, catalan)
        dps: Decimal places of precision

    Returns:
        mpmath high-precision value
    """
    # Temporarily set precision
    original_dps = mp.mp.dps
    mp.mp.dps = dps

    constants_map = {
        "pi": mp.pi,
        "e": mp.e,
        "phi": mp.phi,  # Golden ratio
        "euler": mp.euler,  # Euler-Mascheroni constant γ
        "apery": mp.apery,  # Apery's constant ζ(3)
        "catalan": mp.catalan,  # Catalan's constant
    }

    result = constants_map.get(name.lower())
    if result is None:
        raise ValueError(f"Unknown constant: {name}")

    # Convert to mpf to preserve precision
    result = mp.mpf(result)

    # Restore original precision
    mp.mp.dps = original_dps

    return result


def format_constant_description(name: str) -> str:
    """Get a human-readable description of a constant."""
    descriptions = {
        "pi": "π (pi) ≈ 3.14159...",
        "e": "e (Euler's number) ≈ 2.71828...",
        "phi": "φ (golden ratio) ≈ 1.61803...",
        "euler": "γ (Euler-Mascheroni constant) ≈ 0.57721...",
        "apery": "ζ(3) (Apery's constant) ≈ 1.20205...",
        "catalan": "G (Catalan's constant) ≈ 0.91596...",
    }
    return descriptions.get(name.lower(), name)
