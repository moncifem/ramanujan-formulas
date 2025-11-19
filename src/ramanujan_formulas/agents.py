"""
Agent implementations for the Ramanujan-Swarm system.
Contains proposer agents that generate and evolve mathematical formulas.
"""

import random
import math
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from .config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    CONSTANTS,
    EXPLORATION_RATE_INITIAL,
    EXPLORATION_RATE_MIN,
    MAX_ITERATIONS,
)
from .types import ProposerInput, Candidate
from .search_templates import SearchTemplates
from .polylog_explorer import generate_advanced_expressions


class ProposerAgent:
    """
    Agent responsible for proposing new mathematical expressions.
    
    Uses LLM to generate formulas through either:
    1. Exploration: Novel random expressions
    2. Exploitation: Mutations of best known candidates
    """
    
    def __init__(self):
        """Initialize the proposer agent with Claude model."""
        self.llm = ChatAnthropic(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    
    async def propose(self, state: ProposerInput) -> dict:
        """
        Propose new mathematical expressions based on current state.
        
        Args:
            state: Current proposer input containing best candidates and iteration
            
        Returns:
            Dictionary with 'proposed_expressions' key containing list of expressions
        """
        pool = state.get("best_candidates", [])
        failures = state.get("recent_failures", [])
        iter_num = state.get("iteration", 1)
        
        # Compute exploration rate (decreases over time)
        exploration_rate = max(
            EXPLORATION_RATE_MIN,
            EXPLORATION_RATE_INITIAL - (iter_num / MAX_ITERATIONS)
        )
        
        # Decide strategy: exploration vs exploitation
        is_exploration = (not pool) or (random.random() < exploration_rate)
        
        # Assign a persona to this specific call to ensure diversity
        personas = [
            "The Hyperbolic Geometer",
            "The Series Specialist", 
            "The Constant Hunter",
            "The Modular Form Expert",
            "The Continued Fraction Architect"
        ]
        current_persona = random.choice(personas)

        if is_exploration:
            prompt = self._build_exploration_prompt(failures, current_persona)
        else:
            prompt = self._build_exploitation_prompt(pool, failures, current_persona)
        
        # Add a random seed to the prompt to prevent duplicate outputs from deterministic LLM
        random_seed = random.randint(1, 100000)
        prompt_seed = f"\n\nRandom Seed: {random_seed} (Use this to randomize your choices)"
        
        # Query LLM
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt + prompt_seed)])
            expressions = self._parse_llm_response(response.content)
            return {"proposed_expressions": expressions}
        except Exception as e:
            print(f"⚠️  Proposer error: {e}")
            return {"proposed_expressions": []}
    
    def _build_exploration_prompt(self, failures: List[str], persona: str) -> str:
        """Build prompt for exploration mode (novel expressions)."""
        constants_list = ", ".join(CONSTANTS.keys())

        # Get template examples but don't force them too strictly
        mixed_examples = SearchTemplates.generate_mixed_special_functions()
        
        failures_text = ""
        if failures:
            failures_text = "**RECENT FAILURES (AVOID):**\n" + "\n".join([f"- {f}" for f in failures[:5]])

        persona_instruction = ""
        if persona == "The Hyperbolic Geometer":
            persona_instruction = "Focus on `tanh`, `sinh`, `cosh` with `sqrt` arguments and `mp.pi`. Look for symmetries."
        elif persona == "The Series Specialist":
            persona_instruction = "Focus on `mp.zeta`, `mp.polylog`, `mp.dirichlet` (altzeta), and infinite product structures."
        elif persona == "The Constant Hunter":
            persona_instruction = "Combine fundamental constants `mp.pi`, `mp.e`, `mp.phi`, `mp.euler`, `mp.glaisher` in algebraic relations."
        elif persona == "The Modular Form Expert":
            persona_instruction = "Use `mp.jtheta`, `mp.ellipk`, `mp.qp` (q-Pochhammer). Explore q-series identities."
        elif persona == "The Continued Fraction Architect":
            persona_instruction = "Create nested structures: `sqrt(a + sqrt(b...))` or ratios that look like continued fractions."

        return f"""You are **{persona}**, a mathematical discovery agent.

**Your Task**: Generate 12 UNIQUE expressions using mpmath (aliased as 'mp').

**Strategy**: {persona_instruction}

**Available Constants**: {constants_list}

{failures_text}

**GUIDELINES**:
1. **Avoid Triviality**: Do NOT output known identities like `sin(pi)`.
2. **Seek Near-Integers**: We want expressions that evaluate to values very close to integers (Ramanujan Constants).
3. **Structural Novelty**: Do not just change numbers in previous attempts. Change the *functions* and *structure*.
4. **Use Advanced Functions**: `mp.jtheta(n, z, q)`, `mp.polylog(s, z)`, `mp.zeta(s)`, `mp.ellipk(m)`.

**Output Format**: Return ONLY a Python list of strings.
["expr1", "expr2", ...]"""
    
    def _build_exploitation_prompt(self, pool: List[Candidate], failures: List[str], persona: str) -> str:
        """Build prompt for exploitation mode (mutate best candidates)."""
        # Select top candidates as parents
        parents = sorted(
            random.sample(pool, min(4, len(pool))),
            key=lambda x: x['score']
        )[:2]
        
        parents_info = []
        for p in parents:
            error_exp = int(math.log10(p['error'])) if p['error'] > 0 else -999
            parents_info.append(f"- `{p['expression']}` (Error: 10^{error_exp})")
        
        parents_text = "\n".join(parents_info)

        failures_text = ""
        if failures:
            failures_text = "**AVOID THESE FAILED MUTATIONS:**\n" + "\n".join([f"- {f}" for f in failures[:5]])
        
        return f"""You are **{persona}**, optimizing mathematical formulas.

**Champion Parents**:
{parents_text}

**Your Mission**: Mutate these parents to find *better* approximations (lower error) or *more elegant* forms.

**Strategy**:
- Apply your persona's expertise ({persona}) to the parents.
- If they use `exp`, try `jtheta`.
- If they use `sqrt`, try nested `sqrt` or `log`.
- Multiply by `mp.zeta(3)` or divide by `mp.phi` to cancel out error terms.

{failures_text}

**Output Format**: Return ONLY a Python list of 12 expression strings.
["mutant1", "mutant2", ...]"""
    
    def _parse_llm_response(self, content: str) -> List[str]:
        """
        Parse LLM response to extract list of expressions.
        """
        try:
            if "```" in content:
                if "```python" in content:
                    content = content.split("```python")[-1].split("```")[0]
                else:
                    content = content.split("```")[1]
            
            content = content.strip()
            expressions = eval(content)
            
            if isinstance(expressions, list) and all(isinstance(e, str) for e in expressions):
                return expressions
            
            print(f"⚠️  Invalid response format: {type(expressions)}")
            return []
            
        except Exception as e:
            print(f"⚠️  Error parsing LLM response: {e}")
            return []


# Global agent instance
_proposer_agent = None


def get_proposer_agent() -> ProposerAgent:
    """Get or create the global proposer agent instance."""
    global _proposer_agent
    if _proposer_agent is None:
        _proposer_agent = ProposerAgent()
    return _proposer_agent


async def proposal_node(state: ProposerInput) -> dict:
    """
    LangGraph node function for proposal generation.
    """
    agent = get_proposer_agent()
    return await agent.propose(state)
