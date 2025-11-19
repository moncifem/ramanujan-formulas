"""Mutator agent for refining existing expressions."""

from ramanujan_swarm.agents.base_agent import BaseAgent


class MutatorAgent(BaseAgent):
    """Agent that mutates expressions from the gene pool."""

    def __init__(self, agent_id: int):
        super().__init__(agent_id, "mutator")
