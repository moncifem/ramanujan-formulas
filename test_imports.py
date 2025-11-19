#!/usr/bin/env python3
"""
Quick test to verify all imports work correctly.
"""

def test_imports():
    """Test that all modules can be imported and mp.dps works."""
    print("Testing imports...")

    # Test mpmath import
    from mpmath import mp
    print(f"✓ mpmath imported, current dps: {mp.dps}")

    # Test PSLQ module
    from src.ramanujan_formulas.pslq import PSLQDetector
    detector = PSLQDetector(precision=50)
    print(f"✓ PSLQ module imported, detector created with precision: {detector.precision}")

    # Test stability tester
    from src.ramanujan_formulas.stability_tester import StabilityTester
    tester = StabilityTester()
    print(f"✓ Stability tester imported, test precisions: {tester.test_precisions[:3]}...")

    # Test symbolic simplifier
    from src.ramanujan_formulas.symbolic_simplifier import SymbolicSimplifier
    simplifier = SymbolicSimplifier()
    print("✓ Symbolic simplifier imported")

    # Test polylog explorer
    from src.ramanujan_formulas.polylog_explorer import PolylogExplorer
    explorer = PolylogExplorer()
    print("✓ Polylog explorer imported")

    # Test config
    from src.ramanujan_formulas.config import DECIMAL_PRECISION
    print(f"✓ Config imported, decimal precision: {DECIMAL_PRECISION}")

    print("\n✅ All imports successful! The system should now run without AttributeError.")
    return True

if __name__ == "__main__":
    test_imports()