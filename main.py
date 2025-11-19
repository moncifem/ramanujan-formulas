"""Ramanujan-Swarm: Autonomous Mathematical Discovery via Genetic Agents.

This is the main entry point for running the parallel genetic swarm system.
"""

import asyncio
from ramanujan_swarm.graph import build_graph
from ramanujan_swarm.config import config
from ramanujan_swarm.reporting.report_generator import generate_report, print_summary


async def main():
    """Run the Ramanujan-Swarm system."""

    print("=" * 60)
    print("RAMANUJAN-SWARM: AUTONOMOUS MATHEMATICAL DISCOVERY")
    print("=" * 60)

    # Validate API keys
    try:
        config.validate_api_keys()
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("\nPlease check your .env file and ensure the correct API key is set.")
        return

    print(f"\nConfiguration:")
    print(f"  LLM Provider: {config.llm_provider.upper()}")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Swarm size: {config.swarm_size} agents")
    print(f"  Max generations: {config.max_generations}")
    print(f"  Gene pool size: {config.gene_pool_size}")
    print(f"  Target constants: {', '.join(config.target_constants)}")
    print(f"  Precision: {config.precision_dps} decimal places")
    print(f"  Evolution threshold: {config.evolution_threshold}")
    print(f"  Discovery threshold: {config.discovery_threshold}")
    print("\n" + "=" * 60 + "\n")

    # Build the LangGraph
    print("Building LangGraph with Send API for parallel agents...\n")
    app = build_graph()

    # Configuration for checkpointing and recursion limit
    graph_config = {
        "configurable": {"thread_id": "ramanujan-swarm-1"},
        "recursion_limit": 100,  # Allow up to 100 steps
    }

    # Initialize state with all required fields
    initial_state = {
        "generation": 0,
        "gene_pool": [],
        "current_proposals": [],
        "validated_candidates": [],
        "discoveries": [],
        "swarm_size": config.swarm_size,
        "target_constants": config.target_constants,
        "precision_dps": config.precision_dps,
        "max_generations": config.max_generations,
        "total_expressions_generated": 0,
        "expressions_per_generation": [],
        "best_error_per_generation": [],
    }

    print("Starting genetic swarm evolution...\n")
    print("=" * 60 + "\n")

    # Run the graph
    try:
        final_state = None
        generation_count = 0

        async for state in app.astream(initial_state, graph_config):
            # Track progress
            if "generation" in state:
                current_gen = state["generation"]
                if current_gen > generation_count:
                    generation_count = current_gen
                    print(f"\n{'='*60}")
                    print(f"GENERATION {generation_count}")
                    print(f"{'='*60}")

            # Store final state
            final_state = state

        # Get the actual final state (last node output)
        if final_state:
            # Extract the state from the last node
            if isinstance(final_state, dict) and len(final_state) == 1:
                final_state = list(final_state.values())[0]

            # Print summary
            print_summary(final_state)

            # Generate report
            generate_report(final_state)

            print("\n✓ Execution complete!")

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Saving progress...")
        if final_state:
            generate_report(final_state)

    except Exception as e:
        print(f"\n\n✗ Error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
