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

        # Get template expressions to guide the agent
        path_a_examples = SearchTemplates.generate_path_a_hyperbolic()
        path_b_examples = SearchTemplates.generate_path_b_gamma_asymmetric()
        mixed_examples = SearchTemplates.generate_mixed_special_functions()

        return f"""You are a mathematical discovery agent searching for GENUINELY NEW identities.

**Your Task**: Generate 12 expressions using mpmath (aliased as 'mp').

**Available Constants**: {constants_list}

**🚨 CRITICAL - YOU ARE REDISCOVERING TEXTBOOK IDENTITIES 🚨**

Your recent attempts found:
- Γ(x)Γ(1-x)/π = 1/sin(πx) ← Euler reflection (1729!)
- exp(π√163) ≈ integer ← Heegner (1952!)
- exp(x)/sinh(x) → 2 ← Trivial calculus!

**ALL OF THESE ARE IN TEXTBOOKS. THEY ARE NOT NEW.**

---

**✅ PATH A: Hyperbolic with NON-STANDARD discriminants**

Use these template structures (with DIFFERENT numbers each time):
{chr(10).join(f'- {ex}' for ex in path_a_examples[:2])}

**Key**: Use discriminants NOT in {{19,43,67,163,232,427,522,652}}
Try: {{13,17,21,23,29,31,33,37,41,47,53,57,61,69,71,73,77,79,83,89,...}}

---

**✅ PATH B: Gamma with ASYMMETRIC (not complementary!) combinations**

Use these template structures:
{chr(10).join(f'- {ex}' for ex in path_b_examples[:2])}

**Key**:
- DO NOT use Γ(a/p) × Γ((p-a)/p) - that's Euler reflection!
- USE Γ(a/p)^k × Γ(b/p)^m where a+b ≠ p
- Try large primes p: 17, 19, 23, 29, 31, 37, 41, 47, 53, 59, 61, 67, 71...

---

**✅ PATH C: Mixed special functions**

Examples:
{chr(10).join(f'- {ex}' for ex in mixed_examples[:2])}

**Key**: Combine Γ, ζ, ellipk, hyp2f1 in unexpected ways

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

**Techniques to Try** (SEARCH FOR NOVELTY):

**1. Mixed Special Function Products** (NOT just Heegner!):
- mp.gamma(1/5) * mp.zeta(5) / mp.ellipk(1/3)
- mp.hyp2f1(1/3, 2/3, 1, 1/4) * mp.zeta(3)
- mp.ellipe(phi/5) / mp.gamma(3/7)

**2. Unexpected Ratios**:
- mp.zeta(5) * mp.zeta(7) / (mp.pi**12)
- mp.gamma(1/7) / (mp.gamma(2/7) * mp.gamma(4/7))
- mp.ellipk(1/3) / mp.ellipk(1/5)

**3. Zeta Function Relationships** (Ramanujan's favorite):
- mp.zeta(2) * mp.zeta(3) / mp.pi**5
- mp.zeta(4) / mp.pi**4
- mp.zeta(5) * mp.phi**2
- zeta2, zeta3, zeta4, zeta5, zeta6 are all available!

**4. Gamma Function** (factorials):
- mp.gamma(1/3) * mp.gamma(2/3) / mp.pi
- mp.gamma(1/4)**4 / mp.pi**2
- mp.factorial(20) / mp.e**20

**5. Nested Radicals** (Ramanujan's specialty):
- mp.sqrt(1 + 2*mp.sqrt(1 + 3*mp.sqrt(1 + 4*mp.sqrt(1 + ...))))
- mp.sqrt(mp.phi + mp.sqrt(mp.phi + mp.sqrt(mp.phi)))

**6. Mixed Operations**:
- mp.log(mp.gamma(1/4)) * mp.sqrt(mp.pi)
- mp.sin(mp.pi/5) * mp.phi**2
- mp.log(epi) / mp.log(pie)  # epi and pie are available!

**7. Elliptic Integrals** (advanced):
- mp.ellipk(1/2) / mp.pi
- mp.ellipe(mp.phi/3)

**8. Hypergeometric Functions**:
- mp.hyp2f1(1/2, 1/2, 1, mp.phi/2)
- mp.hyp1f1(1, 2, mp.pi)

**Examples of GOOD Ramanujan-style expressions**:
- "mp.exp(mp.pi * mp.sqrt(163))" (Heegner number - his most famous!)
- "mp.gamma(1/4) / (mp.pi**(3/4))" (Gamma function relationship)
- "mp.zeta(2) * 6 / mp.pi**2" (should equal 1 - Basel problem)
- "mp.sinh(mp.pi * mp.sqrt(522)) / mp.exp(mp.pi * mp.sqrt(522))" (hyperbolic)
- "mp.ellipk(1/2)**2 / mp.pi" (elliptic integral)
- "mp.hyp2f1(1/2, 1/2, 1, 1/2)" (hypergeometric)
- "mp.sqrt(phi + mp.sqrt(phi))" (nested radical)
- "mp.log(epi) - mp.pi" (should be near 0)
- "mp.gamma(1/3) * mp.gamma(2/3) / mp.sqrt(mp.pi)" (reflection formula)

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

**Mutation Strategies** (RAMANUJAN-STYLE evolution):
1. **Try different Heegner numbers**: sqrt(19), sqrt(43), sqrt(67), sqrt(163), sqrt(232), sqrt(427), sqrt(522), sqrt(652)
2. **Add special functions**: Replace exp() with sinh(), cosh(), gamma(), zeta(), ellipk()
3. **Hybridization**: Combine both parents with * or / operations
4. **Use new constants**: zeta2, zeta4, zeta5, epi, pie, pisq
5. **Add nested structure**: sqrt(parent + sqrt(parent))
6. **Elliptic/Hypergeometric**: Wrap in mp.ellipk(), mp.hyp2f1()
7. **Gamma relationships**: Use mp.gamma(rational) with small fractions
8. **Zeta combinations**: Multiply by zeta values or divide by pi powers

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

