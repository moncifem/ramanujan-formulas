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
        iter_num = state.get("iteration", 1)
        
        # Compute exploration rate (decreases over time)
        exploration_rate = max(
            EXPLORATION_RATE_MIN,
            EXPLORATION_RATE_INITIAL - (iter_num / MAX_ITERATIONS)
        )
        
        # Decide strategy: exploration vs exploitation
        is_exploration = (not pool) or (random.random() < exploration_rate)
        
        if is_exploration:
            prompt = self._build_exploration_prompt()
        else:
            prompt = self._build_exploitation_prompt(pool)
        
        # Query LLM
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            expressions = self._parse_llm_response(response.content)
            return {"proposed_expressions": expressions}
        except Exception as e:
            print(f"⚠️  Proposer error: {e}")
            return {"proposed_expressions": []}
    
    def _build_exploration_prompt(self) -> str:
        """Build prompt for exploration mode (novel expressions)."""
        constants_list = ", ".join(CONSTANTS.keys())

        return f"""You are a creative mathematical explorer discovering new identities.

**Your Task**: Generate 12 symbolic Python expressions using mpmath (aliased as 'mp').

**Available Constants**: {constants_list}

**Goal**: Find quasi-integers or quasi-constants (values very close to integers or known constants).

**CRITICAL RULES**:
❌ DO NOT use large numbers (>1000) - only use mathematical constants and small integers!
❌ DO NOT use computed values as constants (like 262537412640768744 or 640320)
❌ DO NOT create expressions that equal ZERO (like mp.phi**2 - mp.phi - 1)
❌ DO NOT create expressions that equal EXACT INTEGERS (like mp.phi**10 + mp.phi**(-10) = 123)
❌ DO NOT use Lucas number patterns: mp.phi**n + mp.phi**(-n) always equals an integer
❌ DO NOT create trivial algebraic identities (like a - a or x + y - x - y)
✅ ONLY use: pi, e, phi, sqrt2, sqrt3, sqrt5, zeta3, catalan, euler, and integers 1-1000
✅ DO create expressions that are CLOSE to integers but NOT exact (small non-zero error)
✅ Example: mp.exp(mp.pi * mp.sqrt(163)) ≈ some large integer (discovery!)

**Techniques to Try** (be CREATIVE and DIVERSE):
- Nested radicals: mp.sqrt(a + mp.sqrt(b + mp.sqrt(c)))
- Exponential towers: mp.exp(mp.pi * mp.sqrt(n)) for n in [19, 43, 67, 163, 232, 522]
- Logarithmic combinations: mp.log(mp.pi) / mp.log(mp.e)
- Power series: mp.pi**n + mp.e**m for various n, m
- Mixed products: mp.pi * mp.e * mp.phi / mp.sqrt(5)
- Trigonometric: mp.sin(mp.pi * mp.phi) or mp.cos(mp.e)
- Ratios of constants: mp.pi**2 / mp.zeta(2)
- Continued fractions with varying depths
- Products with golden ratio: mp.phi**n * mp.pi
- Exponentials of products: mp.exp(mp.pi * mp.e), mp.exp(mp.sqrt(2) * mp.sqrt(3))

**Examples of GOOD expressions** (using ONLY constants and small integers):
- "mp.exp(mp.pi * mp.sqrt(163))" (Heegner number - near-integer!)
- "mp.exp(mp.pi * mp.sqrt(67))" (another Heegner number)
- "mp.pi**4 + mp.pi**5" (power combinations)
- "mp.log(mp.phi) * mp.sqrt(5)" (ratios and products)
- "mp.exp(mp.pi * mp.e / mp.sqrt(2))" (exponentials of products)
- "mp.pi * mp.e * mp.phi" (simple products)
- "(mp.pi + mp.e)**3 - 50" (sum powers, small integer offset)
- "mp.sin(mp.pi * mp.sqrt(163)) * mp.e**20" (trig and exponentials)

**🚨 CRITICAL - DIVERSITY REQUIREMENT 🚨**:
- Each expression MUST be significantly different from previous ones
- DO NOT just change one number in a formula
- DO NOT add/subtract integers to make expressions (like "... - 162")
- Explore DIFFERENT mathematical structures, not variations of the same theme
- Mix constants in NEW ways each time

**Examples of BAD expressions to AVOID**:
- "mp.exp(mp.pi * mp.sqrt(163)) / 262537412640768744" (uses computed result as constant!)
- "mp.log(640320**3 + 744) / (mp.pi * mp.sqrt(163)) - 162" (subtracting integers)
- "mp.phi**10 + mp.phi**(-10)" (Lucas number = exact integer 123)
- "mp.phi**2 - mp.phi - 1" (equals zero exactly)
- "mp.pi" (just a constant)
- "mp.e * mp.e - mp.e**2" (trivial identity = 0)

**Output Format**: Return ONLY a Python list of strings, nothing else.
["expression1", "expression2", ...]"""
    
    def _build_exploitation_prompt(self, pool: List[Candidate]) -> str:
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
        
        return f"""You are a genetic algorithm optimizer for mathematical formulas.

**Champion Parents**:
{parents_text}

**Your Mission**: Create 12 improved mutants with HIGHER precision and ELEGANCE.

**CRITICAL RULES**:
❌ DO NOT use large numbers (>1000) - expressions will be REJECTED!
❌ DO NOT use computed values (like 262537412640768744, 640320**3, etc.)
❌ DO NOT create expressions that equal ZERO
❌ DO NOT create expressions that equal EXACT INTEGERS
❌ DO NOT use Lucas numbers: phi**n + phi**(-n) = exact integer (forbidden!)
❌ DO NOT create "expression ± integer" patterns (like parent - 162)
✅ ONLY use mathematical constants (pi, e, phi, etc.) and small integers (1-1000)
✅ DO improve expressions that are NEAR integers (small non-zero error)
✅ DO make mutations that bring near-integers EVEN CLOSER (reduce error)

**Mutation Strategies** (be CREATIVE, explore NEW directions):
1. **Try different Heegner numbers**: If parent has sqrt(163), try sqrt(19), sqrt(43), sqrt(67), sqrt(232)
2. **Change the operation**: If parent uses exp(), try log(), sin(), or power operations
3. **Hybridization**: Combine elements from BOTH parents in novel ways
4. **Different constants**: Replace pi with e, or phi with zeta(3), etc.
5. **Nested structures**: Add another layer (sqrt of sqrt, exp of exp)
6. **New denominators**: If parent divides by X, try other interesting numbers

**🚨 CRITICAL DIVERSITY RULES 🚨**:
- Don't just add tiny corrections to the same expression!
- Don't create "expression ± integer" patterns (like parent - 162)
- Change the STRUCTURE, not just the numbers
- Explore NEW mathematical territory with different operations
- If parent uses exp(), try log(), sin(), power, or sqrt() instead

**Quality Criteria**:
- Error must be SMALLER than parents (closer to integer/constant)
- Expression should be ELEGANT (avoid bloat)
- Must NOT evaluate to zero or near-zero
- Maintain mathematical beauty

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

