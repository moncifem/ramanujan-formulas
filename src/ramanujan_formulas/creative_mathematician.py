"""
Creative Mathematician Agent - Implements advanced AI techniques for mathematical discovery.
Based on research from Ramanujan Machine, FunSearch, and symbolic reasoning approaches.
"""

import random
import asyncio
from typing import List, Dict, Any, Tuple
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from mpmath import mp
import json

from .config import LLM_MODEL, LLM_MAX_TOKENS


class CreativeMathematician:
    """
    Implements advanced creativity techniques for mathematical discovery:
    1. Novelty Search - Explicitly rewards unexplored mathematical territories
    2. Conceptual Blending - Combines disparate mathematical concepts
    3. Analogical Reasoning - Transfers patterns between domains
    4. Constraint Relaxation - Temporarily violates rules to find new paths
    """
    
    def __init__(self):
        """Initialize the creative mathematician."""
        self.llm = ChatAnthropic(
            model=LLM_MODEL,
            temperature=0.95,  # High temperature for creativity
            max_tokens=LLM_MAX_TOKENS,
        )
        self.discovered_patterns = set()
        self.concept_library = self._initialize_concepts()
    
    def _initialize_concepts(self) -> Dict[str, List[str]]:
        """Initialize library of mathematical concepts for blending."""
        return {
            "number_theory": ["primes", "divisors", "congruences", "quadratic_residues"],
            "analysis": ["limits", "series", "integrals", "derivatives"],
            "algebra": ["groups", "rings", "fields", "polynomials"],
            "geometry": ["manifolds", "curvature", "symmetry", "tessellation"],
            "combinatorics": ["partitions", "permutations", "graphs", "generating_functions"],
            "special_functions": ["gamma", "zeta", "theta", "hypergeometric"],
            "constants": ["pi", "e", "phi", "euler", "catalan"],
            "techniques": ["continued_fractions", "nested_radicals", "q-series", "modular_forms"]
        }
    
    async def generate_creative_expressions(self, context: Dict[str, Any]) -> List[str]:
        """
        Generate creative mathematical expressions using multiple strategies.
        """
        strategies = [
            self._novelty_search,
            self._conceptual_blending,
            self._analogical_reasoning,
            self._constraint_relaxation,
            self._pattern_breaking
        ]
        
        all_expressions = []
        for strategy in strategies:
            expressions = await strategy(context)
            all_expressions.extend(expressions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for expr in all_expressions:
            if expr not in seen:
                seen.add(expr)
                unique.append(expr)
        
        return unique[:20]  # Return top 20 most creative
    
    async def _novelty_search(self, context: Dict[str, Any]) -> List[str]:
        """
        Generate expressions that maximize novelty rather than just accuracy.
        Based on the Novelty Search algorithm from evolutionary computation.
        """
        prompt = """You are exploring the UNEXPLORED regions of mathematical space.

**Novelty Search Directive**: Generate expressions that are MAXIMALLY DIFFERENT from anything seen before.

Known territories to AVOID:
- Heegner numbers (19, 43, 67, 163, etc.)
- Euler reflection (Γ(x)Γ(1-x))
- Lucas numbers (φⁿ + φ⁻ⁿ)

Unexplored territories to EXPLORE:
1. **Exotic Discriminants**: Use prime powers like 121, 169, 289, 361
2. **Mixed Radix Systems**: Combine different number bases
3. **Fractional Iterations**: f^(1/2)(x), f^(π)(x) 
4. **Non-standard Operations**: Tetration, superfactorials, primorials
5. **Quantum-inspired**: Use complex phases, unitary operations

Generate 5 expressions using mpmath that explore these NOVEL territories.
Focus on mathematical structures that have NEVER been combined before.

Output format: ["expr1", "expr2", ...]"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_expressions(response.content)
        except:
            return []
    
    async def _conceptual_blending(self, context: Dict[str, Any]) -> List[str]:
        """
        Blend concepts from different mathematical domains.
        Based on Conceptual Blending Theory from cognitive science.
        """
        # Randomly select two disparate domains
        domains = list(self.concept_library.keys())
        domain1, domain2 = random.sample(domains, 2)
        concepts1 = random.sample(self.concept_library[domain1], 2)
        concepts2 = random.sample(self.concept_library[domain2], 2)
        
        prompt = f"""You are a mathematical CONCEPT BLENDER.

**Blend these concepts**:
From {domain1}: {', '.join(concepts1)}
From {domain2}: {', '.join(concepts2)}

Create 5 expressions that FUSE these concepts in unprecedented ways:

Examples of conceptual blending:
- Combine "continued fractions" with "theta functions"
- Merge "partition theory" with "hyperbolic geometry"
- Fuse "prime gaps" with "elliptic integrals"

Your blends should create expressions that:
1. Use functions from BOTH domains
2. Create surprising connections
3. Might reveal hidden relationships

Output format: ["expr1", "expr2", ...]"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_expressions(response.content)
        except:
            return []
    
    async def _analogical_reasoning(self, context: Dict[str, Any]) -> List[str]:
        """
        Transfer successful patterns to new domains.
        Based on Structure Mapping Theory from AI research.
        """
        # Get best candidate if available
        best_pattern = ""
        if context.get("best_candidates"):
            best = context["best_candidates"][0]
            best_pattern = f"Pattern: {best['expression']}"
        
        prompt = f"""You are using ANALOGICAL REASONING to discover new formulas.

{best_pattern}

**Analogical Transfer Task**:
If exp(π√163) ≈ integer is like a "resonance" at 163,
then what other mathematical "resonances" exist?

Think by analogy:
1. If Heegner numbers work with exp(π√n), what works with:
   - exp(π∛n)?  (cube roots)
   - exp(e√n)?  (e instead of π)
   - sinh(π√n)? (hyperbolic sine)
   - jtheta(3, 0, exp(-π√n))? (theta functions)

2. If 163 is special for quadratic fields, what's special for:
   - Cubic fields?
   - Cyclotomic fields?
   - Elliptic curves?

Generate 5 expressions by ANALOGICAL TRANSFER.

Output format: ["expr1", "expr2", ...]"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_expressions(response.content)
        except:
            return []
    
    async def _constraint_relaxation(self, context: Dict[str, Any]) -> List[str]:
        """
        Temporarily relax mathematical constraints to explore forbidden territories.
        Based on Constraint Satisfaction Problem solving techniques.
        """
        prompt = """You are BREAKING MATHEMATICAL RULES (temporarily) to find new paths.

**Constraint Relaxation Protocol**:

Normal constraints (TEMPORARILY IGNORE):
- Functions must converge
- Arguments must be real
- Expressions must be finite

Explore the FORBIDDEN:
1. **Divergent Series** that somehow converge: sum(n!) / sum(n!!)
2. **Complex Exponents** in real contexts: π^i, e^(iπ√n)
3. **Infinite Recursion** truncated: f(f(f(...f(x)...)))
4. **Division by Zero** patterns: lim[x→0] sin(x)/x style
5. **Fractional Derivatives**: d^(1/2)/dx^(1/2)

Generate 5 expressions that VIOLATE normal rules but might reveal patterns.
Use mpmath functions creatively.

Output format: ["expr1", "expr2", ...]"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_expressions(response.content)
        except:
            return []
    
    async def _pattern_breaking(self, context: Dict[str, Any]) -> List[str]:
        """
        Explicitly break patterns that the system is stuck in.
        """
        failures = context.get("recent_failures", [])
        
        prompt = f"""You are a PATTERN BREAKER.

Recent failures show we're stuck in these patterns:
{failures[:3] if failures else "No specific patterns"}

**Pattern Breaking Directive**:

DO the OPPOSITE of what's expected:
1. If everyone uses sqrt, use cbrt (cube root)
2. If everyone uses integers, use irrationals
3. If everyone uses real numbers, use quaternions
4. If everyone uses addition, use tetration
5. If everyone uses known constants, invent new ones

Break these specific patterns:
- Stop using exp(π√n) format
- Stop using Γ(a/p) × Γ(b/p)
- Stop using simple arithmetic operations

Instead try:
- Nested tetrations: 2^^2^^2
- Primorial products: 2#, 3#, 5#
- Superfactorials: sf(n)
- Hyperfactorials: H(n)

Generate 5 PATTERN-BREAKING expressions.

Output format: ["expr1", "expr2", ...]"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_expressions(response.content)
        except:
            return []
    
    def _parse_expressions(self, content: str) -> List[str]:
        """Parse expressions from LLM response."""
        try:
            if "```" in content:
                if "```python" in content:
                    content = content.split("```python")[1].split("```")[0]
                else:
                    content = content.split("```")[1].split("```")[0]
            
            expressions = eval(content.strip())
            if isinstance(expressions, list):
                # Filter and validate
                valid = []
                for expr in expressions:
                    if isinstance(expr, str) and "mp." in expr:
                        valid.append(expr)
                return valid
        except:
            pass
        return []


class CollaborativeSwarm:
    """
    Implements collaborative strategies between agents.
    Based on Swarm Intelligence and Multi-Agent Systems research.
    """
    
    def __init__(self):
        """Initialize the collaborative swarm."""
        self.creative_math = CreativeMathematician()
        self.communication_channel = {}
    
    async def collaborative_generation(
        self,
        num_agents: int,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Generate expressions through agent collaboration.
        
        Implements:
        1. Diversity Injection - Each agent has unique perspective
        2. Cross-Pollination - Agents share successful patterns
        3. Competitive Cooperation - Agents compete but share insights
        """
        agent_roles = [
            "The Explorer",      # Seeks new territories
            "The Refiner",       # Improves existing formulas
            "The Connector",     # Links disparate concepts
            "The Challenger",    # Questions assumptions
            "The Synthesizer"    # Combines agent outputs
        ]
        
        all_expressions = []
        insights = []
        
        # Phase 1: Independent generation with roles
        tasks = []
        for i in range(min(num_agents, len(agent_roles))):
            role = agent_roles[i]
            role_context = {**context, "role": role}
            tasks.append(self._agent_with_role(role, role_context))
        
        results = await asyncio.gather(*tasks)
        
        for role_expressions, insight in results:
            all_expressions.extend(role_expressions)
            insights.append(insight)
        
        # Phase 2: Cross-pollination
        if insights:
            pollinated = await self._cross_pollinate(insights, context)
            all_expressions.extend(pollinated)
        
        # Remove duplicates
        seen = set()
        unique = []
        for expr in all_expressions:
            if expr not in seen:
                seen.add(expr)
                unique.append(expr)
        
        return unique
    
    async def _agent_with_role(
        self,
        role: str,
        context: Dict[str, Any]
    ) -> Tuple[List[str], str]:
        """Generate expressions with a specific agent role."""
        role_prompts = {
            "The Explorer": "Find completely unexplored mathematical territories",
            "The Refiner": "Take existing patterns and make them more precise",
            "The Connector": "Connect unrelated mathematical concepts",
            "The Challenger": "Challenge every assumption and try the opposite",
            "The Synthesizer": "Combine multiple approaches into unified expressions"
        }
        
        prompt = f"""You are {role}.
Your mission: {role_prompts.get(role, "Discover new mathematics")}

Generate 3 expressions following your role.
Also provide ONE KEY INSIGHT for other agents.

Output JSON:
{{
    "expressions": ["expr1", "expr2", "expr3"],
    "insight": "key insight for others"
}}"""
        
        try:
            llm = ChatAnthropic(model=LLM_MODEL, temperature=0.9)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            
            # Parse JSON
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            data = json.loads(content)
            return data.get("expressions", []), data.get("insight", "")
        except:
            return [], ""
    
    async def _cross_pollinate(
        self,
        insights: List[str],
        context: Dict[str, Any]
    ) -> List[str]:
        """Cross-pollinate insights between agents."""
        insights_text = "\n".join([f"- {ins}" for ins in insights if ins])
        
        prompt = f"""You are synthesizing insights from multiple agents:

{insights_text}

Based on these collective insights, generate 5 expressions that:
1. Combine the best ideas from all agents
2. Address weaknesses identified
3. Explore territories suggested

Output format: ["expr1", "expr2", ...]"""
        
        try:
            llm = ChatAnthropic(model=LLM_MODEL, temperature=0.8)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return self.creative_math._parse_expressions(response.content)
        except:
            return []


# Integration function for use in graph.py
async def inject_creative_expressions(state: Dict[str, Any]) -> List[str]:
    """
    Inject creative expressions into the search.
    
    This function can be called from graph.py to add creative expressions
    to the proposal pipeline.
    """
    swarm = CollaborativeSwarm()
    
    # Use collaborative generation with 5 agents
    expressions = await swarm.collaborative_generation(5, state)
    
    print(f"  🎨 Generated {len(expressions)} creative expressions")
    
    return expressions
