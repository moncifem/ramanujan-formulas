"""Generate final report from swarm run."""

from typing import List
from ramanujan_swarm.graph.state import Expression


def generate_report(
    final_state: dict, output_path: str = "outputs/FINAL_REPORT.md"
) -> None:
    """Generate markdown report from final state.

    Args:
        final_state: Final state from LangGraph execution
        output_path: Path to output file
    """
    with open(output_path, "w") as f:
        f.write("# Ramanujan-Swarm Discovery Report\n\n")

        # Summary statistics
        f.write("## Summary\n\n")
        f.write(f"- **Total Generations**: {final_state.get('generation', 0)}\n")
        f.write(
            f"- **Total Expressions Generated**: {final_state.get('total_expressions_generated', 0)}\n"
        )
        f.write(f"- **Gene Pool Size**: {len(final_state.get('gene_pool', []))}\n")
        f.write(f"- **Discoveries**: {len(final_state.get('discoveries', []))}\n\n")

        # Discoveries
        discoveries = final_state.get("discoveries", [])
        if discoveries:
            f.write("## Discoveries (Error < 1e-50)\n\n")
            for i, expr in enumerate(discoveries, 1):
                f.write(f"### Discovery {i}\n\n")
                f.write(f"**Formula**: `{expr.parsed_expr}`\n\n")
                f.write(f"**Target**: {expr.target_constant}\n\n")
                f.write(f"**Error**: {expr.error:.2e}\n\n")
                f.write(f"**Elegance Score**: {expr.elegance_score:.2e}\n\n")
                f.write(f"**Generation**: {expr.generation}\n\n")
                f.write(f"**Agent Type**: {expr.agent_type}\n\n")
                f.write("---\n\n")

        # Top gene pool candidates
        f.write("## Top Gene Pool Candidates\n\n")
        gene_pool = final_state.get("gene_pool", [])
        if not gene_pool:
            f.write("No candidates in gene pool.\n\n")
        for i, expr in enumerate(gene_pool[:10], 1):
            f.write(f"{i}. `{expr.parsed_expr}`\n")
            f.write(f"   - Target: {expr.target_constant}\n")
            f.write(f"   - Error: {expr.error:.2e}\n")
            f.write(f"   - Elegance: {expr.elegance_score:.2e}\n\n")

        # Performance metrics
        f.write("## Performance Metrics\n\n")
        best_errors = final_state.get("best_error_per_generation", [])
        if best_errors:
            f.write("### Best Error by Generation\n\n")
            for gen, error in enumerate(best_errors):
                f.write(f"- Generation {gen}: {error:.2e}\n")

    print(f"\n✓ Report saved to {output_path}")


def print_summary(final_state: dict) -> None:
    """Print summary to console.

    Args:
        final_state: Final state from execution
    """
    print("\n" + "=" * 60)
    print("RAMANUJAN-SWARM EXECUTION COMPLETE")
    print("=" * 60)
    print(f"\nGenerations completed: {final_state.get('generation', 0)}")
    print(f"Total expressions generated: {final_state.get('total_expressions_generated', 0)}")
    print(f"Gene pool size: {len(final_state.get('gene_pool', []))}")
    print(f"Discoveries found: {len(final_state.get('discoveries', []))}")

    gene_pool = final_state.get("gene_pool", [])
    if gene_pool:
        print("\n" + "-" * 60)
        print("TOP 5 CANDIDATES:")
        print("-" * 60)
        for i, expr in enumerate(gene_pool[:5], 1):
            print(f"\n{i}. {expr.parsed_expr[:80]}")
            print(f"   Target: {expr.target_constant}")
            print(f"   Error: {expr.error:.2e}")
            print(f"   Elegance: {expr.elegance_score:.2e}")
    else:
        print("\n⚠ No expressions were generated. Check your API key and credits.")

    print("\n" + "=" * 60 + "\n")
