"""Node implementations for LangGraph."""

import json
import os
from typing import List
from langgraph.constants import Send
from ramanujan_swarm.graph.state import SwarmState, Expression
from ramanujan_swarm.agents import ExplorerAgent, MutatorAgent, HybridAgent
from ramanujan_swarm.math_engine import Evaluator, Validator, Deduplicator, EleganceScorer
from ramanujan_swarm.gene_pool import GenePool
from ramanujan_swarm.config import config

# JSON file for incremental saving
RESULTS_JSON_PATH = "outputs/formulas_progress.json"


def save_candidates_to_json(candidates: List[Expression], generation: int):
    """Save candidates to JSON file incrementally.

    Args:
        candidates: List of Expression objects to save
        generation: Current generation number
    """
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Load existing data or create new
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r") as f:
            data = json.load(f)
    else:
        data = {
            "metadata": {
                "target_constants": config.target_constants,
                "precision_dps": config.precision_dps,
                "evolution_threshold": config.evolution_threshold,
                "discovery_threshold": config.discovery_threshold,
            },
            "generations": [],
            "all_candidates": [],
            "discoveries": [],
        }

    # Convert candidates to JSON-serializable format
    generation_data = {
        "generation": generation,
        "num_candidates": len(candidates),
        "candidates": []
    }

    for expr in candidates:
        candidate_data = {
            "formula": expr.formula_str,
            "parsed": str(expr.parsed_expr),
            "target_constant": expr.target_constant,
            "numeric_value": expr.numeric_value,
            "error": float(expr.error) if expr.error != float("inf") else "inf",
            "elegance_score": float(expr.elegance_score) if hasattr(expr, "elegance_score") else None,
            "agent_type": expr.agent_type,
            "generation": expr.generation,
            "timestamp": expr.timestamp,
        }
        generation_data["candidates"].append(candidate_data)

        # Also add to all_candidates list
        data["all_candidates"].append(candidate_data)

        # Check if it's a discovery
        if expr.error < config.discovery_threshold:
            data["discoveries"].append(candidate_data)

    # Add this generation's data
    data["generations"].append(generation_data)

    # Save to file
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  💾 Saved {len(candidates)} candidates to {RESULTS_JSON_PATH}")


def initialize_node(state: dict) -> dict:
    """Initialize the swarm system.

    Args:
        state: Input state (mostly empty on first run)

    Returns:
        Initialized state
    """
    # Clear the JSON file at the start of each run
    if os.path.exists(RESULTS_JSON_PATH):
        os.remove(RESULTS_JSON_PATH)
        print(f"  🗑️  Cleared previous results from {RESULTS_JSON_PATH}")

    return {
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


def dispatch_node(state: SwarmState) -> dict:
    """Dispatch node: prepares state for fan-out.

    Args:
        state: Current swarm state

    Returns:
        Dict clearing current_proposals for new generation
    """
    # Clear proposals before dispatching agents for new generation
    return {"current_proposals": []}


def create_agent_tasks(state: SwarmState) -> List[Send]:
    """Create Send() messages for parallel agent execution.

    This function is used by add_conditional_edges to dynamically spawn N agent instances.

    Args:
        state: Current swarm state

    Returns:
        List of Send messages, one per agent
    """
    swarm_size = state["swarm_size"]
    generation = state["generation"]

    sends = []
    for i in range(swarm_size):
        # Distribute agent types based on config
        if i < swarm_size * config.explorer_fraction:
            agent_type = "explorer"
        elif i < swarm_size * (config.explorer_fraction + config.mutator_fraction):
            agent_type = "mutator"
        else:
            agent_type = "hybrid"

        # Create Send message for each agent
        sends.append(
            Send(
                "agent_worker",
                {
                    "agent_id": i,
                    "agent_type": agent_type,
                    "generation": generation,
                    "gene_pool": state["gene_pool"],
                    "target_constants": state["target_constants"],
                },
            )
        )

    return sends


async def agent_worker_node(state: dict) -> dict:
    """Worker node for individual agent - runs in parallel instances.

    Args:
        state: State for this specific agent instance

    Returns:
        Dictionary with current_proposals (will be aggregated via operator.add)
    """
    agent_type = state["agent_type"]
    agent_id = state["agent_id"]
    generation = state["generation"]
    gene_pool = state["gene_pool"]
    target_constants = state["target_constants"]

    # Instantiate appropriate agent
    if agent_type == "explorer":
        agent = ExplorerAgent(agent_id)
    elif agent_type == "mutator":
        agent = MutatorAgent(agent_id)
    else:
        agent = HybridAgent(agent_id)

    # Generate expressions (calls Claude)
    expressions = await agent.generate_expressions(
        gene_pool=gene_pool,
        generation=generation,
        target_constants=target_constants,
        num_expressions=3,  # Generate 3 expressions per agent
    )

    print(
        f"  Agent {agent_id} ({agent_type}) generated {len(expressions)} expressions"
    )

    # Return via reducer field (operator.add will concatenate)
    return {"current_proposals": expressions}


async def validator_node(state: SwarmState) -> dict:
    """REDUCE phase: aggregate all proposals, deduplicate, and score.

    This is the central "CPU" that processes all agent outputs.

    Args:
        state: Current swarm state with all agent proposals

    Returns:
        Updated state with validated candidates
    """
    proposals = state["current_proposals"]
    generation = state["generation"]

    print(f"\n→ Validator processing {len(proposals)} proposals...")

    # Initialize validators
    validator = Validator(precision_dps=state["precision_dps"])
    evaluator = Evaluator(precision_dps=state["precision_dps"])

    valid_expressions = []
    for expr in proposals:
        # Parse with SymPy
        parsed = validator.parse_expression(expr.formula_str)
        if parsed is None:
            continue

        # Evaluate with mpmath at high precision
        numeric_result = evaluator.evaluate(parsed, expr.target_constant)
        if numeric_result is None:
            continue

        # Calculate error
        error = evaluator.compute_error(numeric_result, expr.target_constant)

        # Compute hashes for deduplication
        hash_syntax = validator.syntax_hash(parsed)
        hash_numeric = validator.numeric_hash(numeric_result)

        # Update expression
        expr.parsed_expr = parsed
        expr.numeric_value = str(numeric_result)
        expr.error = error
        expr.hash_syntax = hash_syntax
        expr.hash_numeric = hash_numeric

        valid_expressions.append(expr)

    print(f"  Valid expressions: {len(valid_expressions)}")

    # Deduplicate
    deduplicator = Deduplicator()
    unique_expressions = deduplicator.deduplicate(valid_expressions)
    print(f"  After deduplication: {len(unique_expressions)}")

    # Score by elegance
    scorer = EleganceScorer()
    for expr in unique_expressions:
        expr.elegance_score = scorer.score(expr)

    # Filter by evolution threshold (1e-12)
    candidates = [
        e for e in unique_expressions if e.error < config.evolution_threshold
    ]
    print(f"  Candidates (error < {config.evolution_threshold}): {len(candidates)}")

    # Sort by elegance (lower is better)
    candidates.sort(key=lambda e: e.elegance_score)

    # Update metrics
    best_error = min([e.error for e in candidates], default=float("inf"))

    if candidates:
        print(f"  Best error this generation: {best_error:.2e}")
        print(f"  Top candidate: {candidates[0].parsed_expr[:80]}")

    # Save candidates to JSON incrementally
    if candidates:
        save_candidates_to_json(candidates, generation)

    return {
        "validated_candidates": candidates,
        "current_proposals": [],  # Clear for next generation
        "total_expressions_generated": state["total_expressions_generated"]
        + len(proposals),
        "expressions_per_generation": state["expressions_per_generation"]
        + [len(proposals)],
        "best_error_per_generation": state["best_error_per_generation"]
        + [best_error],
    }


def gene_pool_update_node(state: SwarmState) -> dict:
    """Update the gene pool with new candidates.

    Args:
        state: Current swarm state

    Returns:
        Updated state with new gene pool and incremented generation
    """
    gene_pool_manager = GenePool(max_size=config.gene_pool_size)
    gene_pool_manager.update(
        current_pool=state["gene_pool"], new_candidates=state["validated_candidates"]
    )

    new_pool = gene_pool_manager.get_pool()

    print(f"  Gene pool updated: {len(new_pool)} expressions")

    return {
        "gene_pool": new_pool,
        "generation": state["generation"] + 1,
    }


def route_discoveries(state: SwarmState) -> str:
    """Route based on whether we have new discoveries or should continue.

    Args:
        state: Current swarm state

    Returns:
        Route name: "discover", "continue", or "end"
    """
    generation = state["generation"]
    max_generations = state["max_generations"]
    total_generated = state["total_expressions_generated"]

    # Check if we've reached max generations
    if generation >= max_generations:
        print(f"\n✓ Reached max generations ({max_generations}). Ending.")
        return "end"

    # Check if no expressions are being generated (likely API key issue)
    if generation >= 3 and total_generated == 0:
        print(f"\n⚠ No expressions generated after {generation} generations.")
        print("This usually means there's an API key issue. Ending.")
        return "end"

    # Check for discoveries (error < discovery threshold)
    discoveries = [
        e
        for e in state["validated_candidates"]
        if e.error < config.discovery_threshold
    ]

    if discoveries:
        print(f"\n★ Found {len(discoveries)} discoveries! (error < {config.discovery_threshold})")
        return "discover"

    # Continue to next generation
    print(f"\n→ Generation {generation} complete. Continuing...\n")
    return "continue"


def save_discoveries_node(state: SwarmState) -> dict:
    """Save discoveries to output file.

    Args:
        state: Current swarm state

    Returns:
        Updated state with discoveries logged
    """
    discoveries = [
        e
        for e in state["validated_candidates"]
        if e.error < config.discovery_threshold
    ]

    # Append discoveries to state
    return {"discoveries": discoveries}
