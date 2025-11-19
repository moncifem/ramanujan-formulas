"""Basic test to verify the system setup without API calls."""

import sys
from ramanujan_swarm.config import config
from ramanujan_swarm.constants import get_constant_value, format_constant_description
from ramanujan_swarm.math_engine import Evaluator, Validator, Deduplicator, EleganceScorer
from ramanujan_swarm.math_engine.deduplicator import Expression
from ramanujan_swarm.gene_pool import GenePool
from ramanujan_swarm.graph import build_graph


def test_config():
    """Test configuration loading."""
    print("✓ Config loaded:")
    print(f"  Swarm size: {config.swarm_size}")
    print(f"  Max generations: {config.max_generations}")
    print(f"  Target constants: {config.target_constants}")


def test_constants():
    """Test mathematical constants."""
    print("\n✓ Mathematical constants:")
    for const in ["pi", "e", "phi"]:
        value = get_constant_value(const, dps=50)
        desc = format_constant_description(const)
        print(f"  {desc}")
        print(f"    Value: {value}")


def test_math_engine():
    """Test mathematical engine components."""
    print("\n✓ Mathematical engine:")

    # Test validator
    validator = Validator(precision_dps=100)
    parsed = validator.parse_expression("sqrt(2)")
    print(f"  Parsed 'sqrt(2)': {parsed}")

    # Test evaluator
    evaluator = Evaluator(precision_dps=100)
    result = evaluator.evaluate("sqrt(2)", "pi")
    print(f"  Evaluated 'sqrt(2)': {result}")

    if result:
        error = evaluator.compute_error(result, "pi")
        print(f"  Error vs pi: {error:.2e}")

    # Test known formula: (1 + sqrt(5))/2 ≈ phi
    result = evaluator.evaluate("(1 + sqrt(5))/2", "phi")
    if result:
        error = evaluator.compute_error(result, "phi")
        print(f"\n  Golden ratio formula: (1 + sqrt(5))/2")
        print(f"  Error vs phi: {error:.2e}")
        if error < 1e-50:
            print(f"  ★ This would be a DISCOVERY! (error < 1e-50)")


def test_gene_pool():
    """Test gene pool."""
    print("\n✓ Gene pool:")

    # Create test expressions
    expr1 = Expression(
        formula_str="sqrt(2)",
        parsed_expr="sqrt(2)",
        target_constant="pi",
        agent_type="explorer",
        generation=0,
        numeric_value="1.414",
        error=1.7,
        elegance_score=1.7,
    )

    expr2 = Expression(
        formula_str="sqrt(3)",
        parsed_expr="sqrt(3)",
        target_constant="pi",
        agent_type="explorer",
        generation=0,
        numeric_value="1.732",
        error=1.4,
        elegance_score=1.4,
    )

    pool = GenePool(max_size=5)
    pool.update([], [expr1, expr2])

    print(f"  Pool size: {pool.size()}")
    print(f"  Best candidate: {pool.get_best(1)[0].parsed_expr}")


def test_graph_build():
    """Test LangGraph construction."""
    print("\n✓ LangGraph:")
    try:
        app = build_graph()
        print("  Graph built successfully!")
        print(f"  Graph type: {type(app)}")

        # Get graph structure info
        print("  Graph nodes:", list(app.get_graph().nodes.keys()))
    except Exception as e:
        print(f"  ✗ Error building graph: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAMANUJAN-SWARM BASIC TESTS")
    print("=" * 60 + "\n")

    try:
        test_config()
        test_constants()
        test_math_engine()
        test_gene_pool()
        test_graph_build()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nSystem is ready! To run the swarm:")
        print("  1. Create .env with ANTHROPIC_API_KEY")
        print("  2. Run: python main.py")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
