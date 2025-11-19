"""Explorer agent for novel expression discovery."""

from ramanujan_swarm.agents.base_agent import BaseAgent


class ExplorerAgent(BaseAgent):
    """Agent that explores completely novel expressions."""

    def __init__(self, agent_id: int):
        super().__init__(agent_id, "explorer")
