"""LangGraph graph construction."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ramanujan_swarm.graph.state import SwarmState
from ramanujan_swarm.graph import nodes


def build_graph():
    """Construct the Ramanujan-Swarm LangGraph.

    This creates a Map-Reduce architecture where:
    1. Dispatch node fans out to N parallel agents (MAP)
    2. Agents run in parallel and return results
    3. Validator aggregates all results (REDUCE)
    4. Gene pool is updated
    5. Loop continues for next generation

    Returns:
        Compiled LangGraph application
    """
    # Create graph with state schema
    graph = StateGraph(SwarmState)

    # Add nodes
    graph.add_node("initialize", nodes.initialize_node)
    graph.add_node("dispatch", nodes.dispatch_node)
    graph.add_node("agent_worker", nodes.agent_worker_node)
    graph.add_node("validator", nodes.validator_node)
    graph.add_node("gene_pool_update", nodes.gene_pool_update_node)
    graph.add_node("save_discoveries", nodes.save_discoveries_node)

    # Add edges
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "dispatch")

    # KEY INNOVATION: Conditional edge with Send API
    # The create_agent_tasks function returns a list[Send], which causes LangGraph to:
    # 1. Create N parallel instances of agent_worker
    # 2. Each runs independently
    # 3. All results aggregate back to the next node (validator)
    graph.add_conditional_edges(
        "dispatch",
        nodes.create_agent_tasks,  # Function that returns list[Send]
    )

    # All agents converge to validator (automatic via operator.add reducer)
    graph.add_edge("agent_worker", "validator")
    graph.add_edge("validator", "gene_pool_update")

    # Conditional routing: check discoveries and generation count
    graph.add_conditional_edges(
        "gene_pool_update",
        nodes.route_discoveries,
        {
            "discover": "save_discoveries",
            "continue": "dispatch",  # Loop back for next generation
            "end": END,
        },
    )

    # After saving discoveries, continue to next generation
    graph.add_edge("save_discoveries", "dispatch")

    # Compile with checkpointer for state persistence
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    return app
