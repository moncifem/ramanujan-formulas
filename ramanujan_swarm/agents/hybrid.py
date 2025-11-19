"""Hybrid agent combining exploration and mutation."""

from ramanujan_swarm.agents.base_agent import BaseAgent


class HybridAgent(BaseAgent):
    """Agent that combines exploration and mutation strategies."""

    def __init__(self, agent_id: int):
        super().__init__(agent_id, "hybrid")
