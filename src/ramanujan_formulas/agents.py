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
        
        if is_exploration:
            prompt = self._build_exploration_prompt(failures)
        else:
            prompt = self._build_exploitation_prompt(pool, failures)
        
        # Add a random seed to the prompt to prevent duplicate outputs from deterministic LLM
        random_seed = random.randint(1, 100000)
        prompt_seed = f"\n\nRandom Seed: {random_seed} (Ignore this, just for variability)"
        
        # Query LLM
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt + prompt_seed)])
            expressions = self._parse_llm_response(response.content)
            return {"proposed_expressions": expressions}
        except Exception as e:
            print(f"⚠️  Proposer error: {e}")
            return {"proposed_expressions": []}
    
    def _build_exploration_prompt(self, failures: List[str]) -> str:
        """Build prompt for exploration mode (novel expressions)."""
        constants_list = ", ".join(CONSTANTS.keys())

        # Get template expressions to guide the agent
        path_a_examples = SearchTemplates.generate_path_a_hyperbolic()
        path_b_examples = SearchTemplates.generate_path_b_gamma_asymmetric()
        mixed_examples = SearchTemplates.generate_mixed_special_functions()

        # Add advanced special function examples
        advanced_examples = generate_advanced_expressions("mixed")

        failures_text = ""
        if failures:
            failures_text = "**RECENT FAILURES (DO NOT REPEAT THESE):**\n" + "\n".join([f"- {f}" for f in failures[:8]])

        return f"""You are a mathematical discovery agent searching for GENUINELY NEW identities.

**Your Task**: Generate 12 expressions using mpmath (aliased as 'mp').

**Available Constants**: {constants_list}

{failures_text}

**🚨 CRITICAL WARNING 🚨**
You have been generating TRIVIAL identities or variations of the same known things.
- STOP using `mp.exp(mp.pi * mp.sqrt(d)) / (mp.sinh(...) / n)` -> This just approaches `2n` trivially!
- STOP using Euler reflection `gamma(x)gamma(1-x)`.
- STOP finding just another Heegner number approximation.

**WE NEED STRUCTURAL NOVELTY.**

**✅ PATH A: Infinite Series & Products**
Ramanujan loved infinite series. Use `mp.nsum` and `mp.nprod` (but written as finite expressions or using mpmath's efficient functions if possible, OR just standard functions that represent them).
Actually, mpmath has `mp.polylog`, `mp.zeta`, `mp.ellipk`. Use them!

**✅ PATH B: Modular Forms & Theta Functions**
Use `mp.jtheta(n, z, q)` with interesting q-values (like `exp(-pi*sqrt(d))`).
Ramanujan's "mock theta functions" are still mysterious.
Try combinations like: `mp.jtheta(3, 0, mp.exp(-mp.pi*mp.sqrt(d)))`.

**✅ PATH C: Continued Fractions**
Represent continued fractions using recursive structures or specialized functions if available.
Or just expressions involving nested radicals:
`mp.sqrt(6 + 2*mp.sqrt(7 + 3*mp.sqrt(8 + ...)))` (finite truncation)

**✅ PATH D: Gamma & Zeta Asymmetries**
Use `mp.gamma(a/b)` where `a/b` has large denominator (e.g., 13, 17, 19).
Mix `mp.zeta(s)` with `mp.pi` and `mp.phi` in unexpected ratios.

**Techniques to Try** (SEARCH FOR NOVELTY):

**1. Mixed Special Function Products**:
- `mp.gamma(1/5) * mp.zeta(5) / mp.ellipk(1/3)`
- `mp.hyp2f1(1/3, 2/3, 1, 1/4) * mp.zeta(3)`

**2. Unexpected Ratios**:
- `mp.zeta(5) * mp.zeta(7) / (mp.pi**12)`
- `mp.ellipk(1/3) / mp.ellipk(1/5)`

**3. Advanced Functions**:
- `mp.polylog(3, mp.phi/3)`
- `mp.agm(1, mp.sqrt(2))`
- `mp.besselj(0, mp.pi)`

**CRITICAL RULES**:
❌ DO NOT use large numbers (>1000) - only use mathematical constants and small integers!
❌ DO NOT use computed values as constants (like 262537412640768744)
❌ DO NOT create expressions that equal ZERO (like `mp.phi**2 - mp.phi - 1`)
❌ DO NOT create expressions that equal EXACT INTEGERS
❌ DO NOT create trivial algebraic identities (like `a - a`)
❌ AVOID `exp/sinh` ratios unless you are sure they are non-trivial.
✅ ONLY use: pi, e, phi, sqrt2, sqrt3, sqrt5, zeta3, catalan, euler, and integers 1-1000
✅ DO create expressions that are CLOSE to integers but NOT exact.

**Output Format**: Return ONLY a Python list of strings, nothing else.
["expression1", "expression2", ...]"""
    
    def _build_exploitation_prompt(self, pool: List[Candidate], failures: List[str]) -> str:
        """Build prompt for exploitation mode (mutate best candidates)."""
        # Select top candidates as parents
        parents = sorted(
            random.sample(pool, min(4, len(pool))),
            key=lambda x: x['score']
        )[:2]
        
        # Format parents info
        parents_info = []
        for p in parents:
            error_exp = int(math.log10(p['error'])) if p['error'] > 0 else -999
            parents_info.append(f"- `{p['expression']}` (Error: 10^{error_exp})")
        
        parents_text = "\n".join(parents_info)

        failures_text = ""
        if failures:
            failures_text = "**AVOID THESE FAILED ATTEMPTS:**\n" + "\n".join([f"- {f}" for f in failures[:5]])
        
        return f"""You are a genetic algorithm optimizer for mathematical formulas.

**Champion Parents**:
{parents_text}

**Your Mission**: Create 12 improved mutants with HIGHER precision and ELEGANCE.

{failures_text}

**Mutation Strategies** (STRUCTURAL CHANGES):
1. **Generalize**: If parent has `sqrt(163)`, try `sqrt(d)` for other `d`. But AVOID known Heegner numbers if they failed.
2. **Function Swap**: Replace `exp` with `jtheta`, `ellipk`, or `besselj`.
3. **Argument Shift**: If `gamma(1/4)`, try `gamma(1/5)` or `gamma(2/7)`.
4. **Structure Nesting**: Wrap the parent in `mp.log(...)` or `mp.sqrt(...)`.
5. **Constants Mix**: Multiply by `mp.zeta(3)` or divide by `mp.catalan`.

**CRITICAL RULES**:
❌ DO NOT use large numbers (>1000).
❌ DO NOT use computed float values as constants.
❌ DO NOT create trivial identities.
✅ Explore the neighborhood of the parents but don't be afraid to change the functional form significantly.

**Output Format**: Return ONLY a Python list of 12 expression strings.
["mutant1", "mutant2", ...]"""
    
    def _parse_llm_response(self, content: str) -> List[str]:
        """
        Parse LLM response to extract list of expressions.
        
        Args:
            content: Raw LLM response text
            
        Returns:
            List of expression strings
        """
        try:
            # Clean up markdown code blocks if present
            if "```" in content:
                # Extract content between code fences
                if "```python" in content:
                    content = content.split("```python")[-1].split("```")[0]
                else:
                    content = content.split("```")[1]
            
            # Remove any surrounding whitespace
            content = content.strip()
            
            # Safely evaluate as Python literal
            expressions = eval(content)
            
            # Validate it's a list of strings
            if isinstance(expressions, list) and all(isinstance(e, str) for e in expressions):
                return expressions
            
            print(f"⚠️  Invalid response format: {type(expressions)}")
            return []
            
        except SyntaxError as e:
            print(f"⚠️  Syntax error parsing response: {e}")
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
    
    Args:
        state: Proposer input state
        
    Returns:
        Update dictionary with proposed expressions
    """
    agent = get_proposer_agent()
    return await agent.propose(state)
