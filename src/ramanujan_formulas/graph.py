"""
LangGraph orchestration module.
Defines the state graph for the Ramanujan-Swarm system.
"""

from typing import List
import math
import random

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from .types import State, ProposerInput, Candidate
from .agents import proposal_node
from .utils import (
    evaluate_expression,
    compute_error,
    compute_elegance_score,
    is_candidate_worthy,
    is_discovery_worthy,
    format_error_magnitude,
    get_value_signature,
    expression_hash,
)
from .verification import check_novelty_online, save_discovery
from .trivial_filter import is_trivial_identity
from .pslq import analyze_discovery_with_pslq
from .symbolic_simplifier import try_simplify_expression
from .stability_tester import validate_discovery_stability
from .breakthrough_injector import inject_breakthroughs, get_breakthrough_injector
from .config import (
    SWARM_SIZE,
    MAX_ITERATIONS,
    MAX_DISCOVERIES,
    GENE_POOL_SIZE,
    RESULTS_DIR,
)
import json
from datetime import datetime


def save_candidate_to_file(candidate: Candidate, iteration: int) -> None:
    """
    Save a candidate to the candidates log file.

    Args:
        candidate: The candidate to save
        iteration: Current iteration number
    """
    candidates_file = RESULTS_DIR / "candidates.jsonl"

    # Create record
    record = {
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "expression": candidate["expression"],
        "value": candidate["value_str"],
        "error": candidate["error"],
        "score": candidate["score"],
        "source": candidate["source"],
    }

    # Append to JSONL file
    with open(candidates_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class RamanujanGraph:
    """Orchestrates the genetic swarm using LangGraph."""

    def __init__(self):
        """Initialize the graph with checkpointer."""
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        graph = StateGraph(State)
        
        # Add nodes
        graph.add_node("proposer", proposal_node)
        graph.add_node("validator", self._validator_node)
        
        # Add edges
        graph.add_conditional_edges(
            START,
            self._route_to_swarm,
            ["proposer"]
        )
        graph.add_edge("proposer", "validator")
        graph.add_conditional_edges(
            "validator",
            self._route_to_swarm,
            ["proposer", END]
        )
        
        return graph
    
    def _validator_node(self, state: State) -> dict:
        """
        Validator node: evaluates proposed expressions and updates gene pool.

        Args:
            state: Current graph state

        Returns:
            Update dictionary with validated candidates and discoveries
        """
        new_candidates: List[Candidate] = []
        new_discoveries: List[dict] = []

        current_pool = state.get("best_candidates", [])
        current_iter = state.get("iteration", 1)

        seen_hashes = state.get("tested_hashes", set())
        seen_values = state.get("tested_values", set())

        proposed = state.get("proposed_expressions", [])

        print(f"\n{'='*60}")
        print(f"  ITERATION {current_iter} - Validating {len(proposed)} expressions")
        print(f"{'='*60}")

        # Show sample of proposed expressions
        if proposed:
            print(f"  📝 Sample expressions (first 5):")
            for expr in proposed[:5]:
                print(f"     • {expr[:70]}{'...' if len(expr) > 70 else ''}")

        # Track rejection reasons
        rejection_stats = {
            "duplicate_expr": 0,
            "eval_failed": 0,
            "large_numbers": 0,  # Contains numbers > 1000
            "duplicate_value": 0,
            "trivial_identity": 0,  # NEW: Euler/Gauss/known formulas
            "trivial_zero": 0,
            "exact_integer": 0,
            "trivial_constant": 0,
            "exact_constant": 0,
            "high_error": 0,
        }
        
        # Process each proposed expression
        for expr in proposed:
            # Check for duplicate expression
            h = expression_hash(expr)
            if h in seen_hashes:
                rejection_stats["duplicate_expr"] += 1
                continue
            seen_hashes.add(h)

            # Check for large numbers (>1000) before evaluation
            import re
            numbers = re.findall(r'\b(\d{4,})\b', expr)
            has_large_number = any(int(num) > 1000 for num in numbers)
            if has_large_number:
                rejection_stats["large_numbers"] += 1
                continue

            # Evaluate expression
            value = evaluate_expression(expr)
            if value is None:
                rejection_stats["eval_failed"] += 1
                continue

            # Check for duplicate value
            val_sig = get_value_signature(value)
            if val_sig in seen_values:
                rejection_stats["duplicate_value"] += 1
                continue
            seen_values.add(val_sig)

            # Compute error
            error, target_type = compute_error(value)

            # Track rejection by target type
            if target_type in rejection_stats:
                rejection_stats[target_type] += 1

            # NEW: Check if trivial identity (Euler/Gauss/Heegner/etc)
            is_trivial, trivial_reason = is_trivial_identity(expr, value, error)
            if is_trivial:
                rejection_stats["trivial_identity"] += 1
                # Log first few trivial rejections for debugging
                if rejection_stats["trivial_identity"] <= 3:
                    print(f"  ⚠️  Rejected trivial: {expr[:60]}... - {trivial_reason}")
                continue

            # Check with PSLQ for linear combinations of known constants
            if error < 1e-10:  # Only check near-integers
                is_novel_pslq, pslq_reason = analyze_discovery_with_pslq(expr, value, error)
                if not is_novel_pslq:
                    rejection_stats["trivial_identity"] += 1
                    if rejection_stats["trivial_identity"] <= 5:
                        print(f"  ⚠️  PSLQ detected known relation: {pslq_reason}")
                    continue

            # Check if worthy of gene pool
            if is_candidate_worthy(error):
                # Test stability for very small errors
                if error < 1e-30:
                    is_stable, stability_msg = validate_discovery_stability(expr, value, error)
                    if not is_stable:
                        rejection_stats["high_error"] += 1  # Use existing category
                        print(f"  ⚠️  Unstable expression: {stability_msg}")
                        continue

                # Try to simplify the expression
                simplified_expr, simplification_note = try_simplify_expression(expr, value, error)
                if simplified_expr != expr:
                    print(f"  🔧 Simplified: {simplification_note}")
                    expr = simplified_expr  # Use simplified form

                score = compute_elegance_score(error, expr)

                candidate = Candidate(
                    expression=expr,
                    value_str=str(value)[:50],
                    error=error,
                    score=score,
                    source=f"iter_{current_iter}"
                )
                new_candidates.append(candidate)

                # Save candidate to file
                save_candidate_to_file(candidate, current_iter)

                # Log interesting findings with actual values
                if error < 1e-20:
                    error_str = format_error_magnitude(error)
                    print(f"  🎯 Strong candidate ({error_str}): {expr[:60]}...")
                    print(f"      ↳ Value: {str(value)[:60]}...")
                    print(f"      ↳ Target: {target_type}")
            else:
                if error != float('inf'):
                    rejection_stats["high_error"] += 1

            # Check if worthy of verification
            if is_discovery_worthy(error):
                error_str = format_error_magnitude(error)
                print(f"\n  🚨 MAJOR HIT ({error_str})!")
                print(f"     Expression: {expr}")
                print(f"     Value: {str(value)[:80]}")

                # Verify novelty
                novelty = check_novelty_online(value)

                discovery = {
                    "expression": expr,
                    "value": str(value),
                    "error": error,
                    "iteration": current_iter,
                    "verified": novelty["novel"],
                    "oeis_status": novelty.get("oeis", "UNKNOWN")
                }

                if novelty["novel"]:
                    print(f"  ✅ VERIFIED NOVEL - Saving discovery!")
                    new_discoveries.append(discovery)
                    save_discovery(discovery)
                else:
                    print(f"  ❌ Known in database ({novelty.get('oeis')})")
        
        # Print rejection statistics
        print(f"\n  🔍 Validation Summary:")
        print(f"     • Total proposed: {len(proposed)}")
        print(f"     • Accepted: {len(new_candidates)}")
        print(f"     • Rejected: {len(proposed) - len(new_candidates)}")
        print(f"\n  📉 Rejection Breakdown:")
        for reason, count in rejection_stats.items():
            if count > 0:
                reason_display = reason.replace('_', ' ').title()
                print(f"     • {reason_display}: {count}")

        # Update gene pool
        all_candidates = current_pool + new_candidates
        all_candidates.sort(key=lambda x: x["score"])
        best_pool = all_candidates[:GENE_POOL_SIZE]

        # Report statistics
        print(f"\n  📊 Gene Pool Status:")
        print(f"     • New candidates: {len(new_candidates)}")
        print(f"     • Pool size: {len(best_pool)}/{GENE_POOL_SIZE}")

        # Analyze successful patterns and prepare for breakthrough injection
        injector = get_breakthrough_injector()
        for candidate in new_candidates:
            if candidate["error"] < 1e-20:
                injector.analyze_success_pattern(candidate["expression"], candidate["error"])

        if new_candidates:
            best = min(new_candidates, key=lambda x: x['error'])
            error_str = format_error_magnitude(best['error'])
            print(f"  🏆 Best this round: {error_str}")
            print(f"     Expression: {best['expression'][:80]}...")

        if best_pool:
            champion = min(best_pool, key=lambda x: x['error'])
            error_str = format_error_magnitude(champion['error'])
            print(f"  👑 Current champion: {error_str}")
            print(f"     Expression: {champion['expression'][:80]}...")

            # Show top 3 from gene pool
            print(f"\n  🧬 Top 3 in Gene Pool:")
            top3 = sorted(best_pool, key=lambda x: x['error'])[:3]
            for i, cand in enumerate(top3, 1):
                err_str = format_error_magnitude(cand['error'])
                print(f"     {i}. Error {err_str}: {cand['expression'][:60]}...")
        
        # Collect rejected expressions for feedback
        rejected_exprs = [expr for expr in proposed if expr not in {c["expression"] for c in new_candidates}]
        recent_failures = random.sample(rejected_exprs, min(20, len(rejected_exprs))) if rejected_exprs else []

        return {
            "best_candidates": best_pool,
            "discoveries": new_discoveries,
            "tested_hashes": seen_hashes,
            "tested_values": seen_values,
            "proposed_expressions": [],  # Clear for next iteration
            "recent_failures": recent_failures,
            "iteration": current_iter + 1
        }
    
    def _route_to_swarm(self, state: State) -> str | List[Send]:
        """
        Routing function: determines whether to continue or end.

        Args:
            state: Current graph state

        Returns:
            END if stopping conditions met, otherwise list of Send objects for swarm
        """
        iteration = state.get("iteration", 1)
        discoveries = state.get("discoveries", [])

        # Inject breakthrough expressions every few iterations
        if iteration % 3 == 0 and iteration > 1:
            state = inject_breakthroughs(state)
        
        # Check stopping conditions
        if iteration > MAX_ITERATIONS:
            print(f"\n🏁 Maximum iterations ({MAX_ITERATIONS}) reached")
            return END
        
        if len(discoveries) >= MAX_DISCOVERIES:
            print(f"\n🏁 Target discoveries ({MAX_DISCOVERIES}) achieved!")
            return END
        
        # Continue: spawn swarm of proposers
        payload = ProposerInput(
            best_candidates=state.get("best_candidates", []),
            recent_failures=state.get("recent_failures", []),
            iteration=iteration
        )
        
        return [Send("proposer", payload) for _ in range(SWARM_SIZE)]
    
    async def run(self, initial_state: State, thread_id: str = "default") -> State:
        """
        Run the graph with initial state.

        Args:
            initial_state: Starting state
            thread_id: Thread ID for checkpointing

        Returns:
            Final state after completion
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": MAX_ITERATIONS * 2 + 50  # Double iterations plus buffer for safety
        }

        final_state = None
        async for step in self.app.astream(initial_state, config=config):
            final_state = step

        return final_state
    
    def get_state(self, thread_id: str = "default") -> State | None:
        """
        Retrieve saved state for a thread.
        
        Args:
            thread_id: Thread ID to retrieve
            
        Returns:
            State if found, None otherwise
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.app.get_state(config)
            return state.values if state else None
        except Exception:
            return None


def create_graph() -> RamanujanGraph:
    """Create and return a configured RamanujanGraph instance."""
    return RamanujanGraph()


def create_initial_state() -> State:
    """Create the initial state for a new run."""
    return State(
        proposed_expressions=[],
        best_candidates=[],
        discoveries=[],
        tested_hashes=set(),
        tested_values=set(),
        recent_failures=[],
        iteration=1
    )

