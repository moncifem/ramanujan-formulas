"""
Main entry point for Ramanujan-Swarm.
Orchestrates the genetic algorithm for mathematical discovery.
"""

import asyncio
import sys
import time
from datetime import datetime
import os

# Fix Windows Unicode issues
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')

from src.ramanujan_formulas.config import (
    validate_config,
    RESULTS_DIR,
    SWARM_SIZE,
    MAX_ITERATIONS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
)
from src.ramanujan_formulas.graph import create_graph, create_initial_state
from src.ramanujan_formulas.verification import initialize_report


def print_banner():
    """Print startup banner."""
    banner = f"""
{'='*70}
    🧬 RAMANUJAN-SWARM: Mathematical Discovery System
{'='*70}

Configuration:
  • Model:            {LLM_MODEL}
  • Max Tokens:       {LLM_MAX_TOKENS:,} per request
  • Swarm Size:       {SWARM_SIZE} parallel agents
  • Max Iterations:   {MAX_ITERATIONS}
  • Precision:        1500 decimal places
  • Results:          {RESULTS_DIR.absolute()}

{'='*70}
"""
    print(banner)


def print_summary(duration: float, final_state: dict):
    """Print execution summary."""
    discoveries = final_state.get("discoveries", [])
    iterations = final_state.get("iteration", 1) - 1
    candidates = len(final_state.get("best_candidates", []))
    tested = len(final_state.get("tested_hashes", set()))
    
    summary = f"""
{'='*70}
    EXECUTION SUMMARY
{'='*70}

Duration:           {duration:.1f} seconds
Iterations:         {iterations}
Expressions Tested: {tested}
Gene Pool Size:     {candidates}
Discoveries:        {len(discoveries)}

"""
    
    if discoveries:
        summary += "Novel Discoveries:\n"
        for i, disc in enumerate(discoveries, 1):
            summary += f"  {i}. Error: 10^{int(disc['error'])} - {disc['expression'][:50]}...\n"
    
    summary += f"\n{'='*70}\n"
    
    print(summary)


async def main():
    """Main execution function."""
    try:
        # Validate configuration
        print("🔧 Validating configuration...")
        validate_config()
        
        # Print banner
        print_banner()
        
        # Initialize report
        print("📝 Initializing report files...")
        initialize_report()
        
        # Create graph
        print("🏗️  Building LangGraph architecture...")
        graph = create_graph()
        
        # Create initial state
        initial_state = create_initial_state()
        
        # Generate thread ID
        thread_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 Starting evolutionary search (Thread: {thread_id})...\n")
        
        # Run the graph
        start_time = time.time()
        final_state = await graph.run(initial_state, thread_id)
        duration = time.time() - start_time
        
        # Print summary
        print_summary(duration, final_state)
        
        print(f"✅ Complete! Results saved to {RESULTS_DIR.absolute()}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
