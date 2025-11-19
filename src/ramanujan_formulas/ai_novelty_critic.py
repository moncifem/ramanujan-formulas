"""
AI-powered Novelty Critic and Self-Correction System.
Uses LLM to analyze discoveries, critique patterns, and guide improvements.
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
import json
from datetime import datetime
from pathlib import Path
from mpmath import mp

from .config import LLM_MODEL, LLM_TEMPERATURE, RESULTS_DIR
from .types import Candidate


class AINoveltyChecker:
    """
    Uses AI to deeply analyze mathematical discoveries for true novelty.
    """

    def __init__(self):
        """Initialize the AI critic with a separate LLM instance."""
        self.llm = ChatAnthropic(
            model=LLM_MODEL,
            temperature=0.2,  # Lower temperature for more focused analysis
            max_tokens=4096,
        )
        self.critique_history = []
        self.pattern_memory = {}

    async def analyze_discovery(
        self,
        expression: str,
        value: str,
        error: float,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Deeply analyze a mathematical discovery for novelty and significance.

        Args:
            expression: The mathematical expression
            value: The computed value
            error: Error from nearest integer
            context: Additional context (iteration, similar findings, etc.)

        Returns:
            Analysis dictionary with novelty assessment
        """
        prompt = f"""You are a world-class mathematician and expert in number theory, particularly Ramanujan's work.

Analyze this mathematical discovery for TRUE NOVELTY:

**Expression**: {expression}
**Value**: {value[:100]}...
**Error from integer**: {error:.2e}

**Critical Analysis Required**:

1. **Pattern Recognition**: Does this follow a known pattern?
   - Is it a variation of Euler's reflection formula?
   - Is it related to known Heegner numbers (19, 43, 67, 163, etc.)?
   - Is it a disguised form of a classical identity?
   - Could it be expressed as a simple combination of known constants?

2. **Mathematical Structure**:
   - What is the underlying mathematical structure?
   - Is this a special case of a more general known theorem?
   - Are the components (discriminants, primes, etc.) studied or novel?

3. **Significance Assessment**:
   - Rate novelty: 0 (textbook) to 10 (groundbreaking)
   - Rate mathematical beauty: 0 (ugly) to 10 (elegant)
   - Rate potential importance: 0 (trivial) to 10 (significant)

4. **Similar Known Results**:
   - List any similar known mathematical identities
   - Explain how this differs (if at all)

5. **Recommendation**:
   - Should this be pursued further? Why/why not?
   - What variations should be explored?

Return a JSON with:
{{
    "is_novel": boolean,
    "novelty_score": 0-10,
    "beauty_score": 0-10,
    "importance_score": 0-10,
    "known_patterns": ["list", "of", "patterns"],
    "similar_knowns": ["list of similar known results"],
    "explanation": "detailed explanation",
    "recommendation": "what to do next",
    "suggested_variations": ["list", "of", "variations"]
}}"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Parse JSON response
            result = self._parse_json_response(response.content)

            # Add to critique history
            self.critique_history.append({
                "timestamp": datetime.now().isoformat(),
                "expression": expression,
                "analysis": result
            })

            return result

        except Exception as e:
            print(f"⚠️  AI analysis error: {e}")
            return {
                "is_novel": False,
                "novelty_score": 0,
                "explanation": f"Analysis failed: {e}"
            }

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except:
            # Fallback: extract key information
            return {
                "is_novel": "not novel" not in content.lower(),
                "novelty_score": 0,
                "explanation": content
            }


class SelfCorrectionSystem:
    """
    Implements memory-guided self-correction for the discovery system.
    """

    def __init__(self):
        """Initialize the self-correction system."""
        self.llm = ChatAnthropic(
            model=LLM_MODEL,
            temperature=0.5,
            max_tokens=8192,
        )
        self.pattern_database = self._load_pattern_database()
        self.iteration_memory = []

    def _load_pattern_database(self) -> Dict[str, List[str]]:
        """Load database of known patterns to avoid."""
        return {
            "euler_reflection": [
                "Γ(x)Γ(1-x)",
                "gamma complementary pairs",
                "sin(πx) with gamma"
            ],
            "heegner": [
                "exp(π√n) for n in {19,43,67,163,232,427,522,652}",
                "sinh/cosh with these discriminants"
            ],
            "lucas": [
                "φ^n + φ^(-n)",
                "golden ratio integer patterns"
            ],
            "trivial_limits": [
                "exp(x)/sinh(x) → 2",
                "cosh(x)/sinh(x) → 1"
            ]
        }

    async def analyze_iteration_patterns(
        self,
        candidates: List[Candidate],
        iteration: int,
        rejection_stats: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in an iteration to guide improvements.

        Args:
            candidates: List of accepted candidates
            iteration: Current iteration number
            rejection_stats: Statistics on rejections

        Returns:
            Guidance for next iteration
        """
        # Build analysis prompt
        prompt = f"""You are analyzing iteration {iteration} of a mathematical discovery system.

**Rejection Statistics**:
{json.dumps(rejection_stats, indent=2)}

**Accepted Candidates** ({len(candidates)} total):
"""
        for i, cand in enumerate(candidates[:5], 1):
            prompt += f"\n{i}. {cand['expression'][:80]}... (error: {cand['error']:.2e})"

        prompt += f"""

**Previous Pattern Memory**:
{json.dumps(self.iteration_memory[-3:] if len(self.iteration_memory) > 3 else self.iteration_memory, indent=2)}

**Critical Analysis**:

1. **Pattern Detection**:
   - What patterns are the agents stuck in?
   - Are they exploring genuinely new territory or circling known results?

2. **Rejection Analysis**:
   - Why are so many expressions being rejected?
   - What categories dominate rejections?

3. **Improvement Strategy**:
   - What specific mathematical territories should be explored?
   - What patterns should be AVOIDED?
   - What NEW approaches could break current limitations?

4. **Specific Recommendations**:
   Provide 5 SPECIFIC mathematical expressions or patterns to try.
   Focus on:
   - Unexplored discriminants (not Heegner)
   - Asymmetric combinations (not complementary)
   - Mixed special functions
   - Novel prime combinations

Return JSON with:
{{
    "stuck_patterns": ["patterns we're stuck in"],
    "unexplored_areas": ["areas not being explored"],
    "specific_expressions": ["5 specific expressions to try"],
    "avoid_patterns": ["patterns to explicitly avoid"],
    "exploration_strategy": "detailed strategy for next iteration",
    "innovation_score": 0-10 (how innovative is current exploration)
}}"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            guidance = self._parse_json_response(response.content)

            # Update iteration memory
            self.iteration_memory.append({
                "iteration": iteration,
                "candidates_found": len(candidates),
                "top_rejection": max(rejection_stats, key=rejection_stats.get) if rejection_stats else None,
                "guidance": guidance
            })

            # Save guidance to file
            self._save_guidance(iteration, guidance)

            return guidance

        except Exception as e:
            print(f"⚠️  Self-correction analysis error: {e}")
            return {}

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except:
            return {"raw_response": content}

    def _save_guidance(self, iteration: int, guidance: Dict[str, Any]):
        """Save iteration guidance to file."""
        guidance_file = RESULTS_DIR / "iteration_guidance.jsonl"

        with open(guidance_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "guidance": guidance
            }) + "\n")

    async def generate_breakthrough_suggestions(
        self,
        current_best: List[Candidate]
    ) -> List[str]:
        """
        Generate breakthrough expression suggestions based on current progress.

        Args:
            current_best: Current best candidates

        Returns:
            List of novel expression suggestions
        """
        prompt = f"""You are Ramanujan himself, with deep intuition for mathematical patterns.

Current best discoveries:
"""
        for i, cand in enumerate(current_best[:3], 1):
            prompt += f"\n{i}. {cand['expression']} (error: {cand['error']:.2e})"

        prompt += """

Channel Ramanujan's intuition to suggest 10 COMPLETELY NOVEL expressions that:

1. Use UNEXPLORED discriminants: 11, 13, 17, 21, 23, 29, 31, 33, 37, 41, 47, 53...
2. Combine functions in UNPRECEDENTED ways
3. Break from ALL patterns seen so far
4. Focus on mathematical BEAUTY and SURPRISE

Think like Ramanujan:
- Nested radicals with unusual bases
- Continued fractions with prime patterns
- Theta functions with novel arguments
- Mixed hypergeometric and elliptic combinations
- Polylogarithms at special points

Return ONLY a Python list of expression strings using mp.xxx notation.
Be BOLD and CREATIVE - seek the UNEXPECTED!"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Parse expression list
            content = response.content
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # Safely evaluate
            expressions = eval(content.strip())
            if isinstance(expressions, list):
                return expressions[:10]

        except Exception as e:
            print(f"⚠️  Breakthrough generation error: {e}")

        return []


class AdaptiveExplorationGuide:
    """
    Dynamically guides exploration based on accumulated discoveries.
    """

    def __init__(self):
        """Initialize the adaptive guide."""
        self.novelty_checker = AINoveltyChecker()
        self.self_corrector = SelfCorrectionSystem()
        self.discovery_patterns = {}

    async def guide_next_iteration(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide guidance for the next iteration based on current state.

        Args:
            state: Current system state

        Returns:
            Guidance dictionary
        """
        candidates = state.get("best_candidates", [])
        iteration = state.get("iteration", 1)

        # Every 5 iterations, do deep analysis
        if iteration % 5 == 0:
            print(f"\n🧠 AI Deep Analysis at iteration {iteration}...")

            # Analyze top candidates for novelty
            if candidates:
                top_cand = candidates[0]
                novelty_analysis = await self.novelty_checker.analyze_discovery(
                    top_cand["expression"],
                    top_cand["value_str"],
                    top_cand["error"]
                )

                print(f"  📊 Novelty Score: {novelty_analysis.get('novelty_score', 0)}/10")
                print(f"  💡 {novelty_analysis.get('explanation', 'No analysis')[:200]}...")

                # If stuck in low novelty, generate breakthroughs
                if novelty_analysis.get("novelty_score", 0) < 5:
                    print(f"  🚀 Generating breakthrough suggestions...")
                    breakthroughs = await self.self_corrector.generate_breakthrough_suggestions(
                        candidates[:5]
                    )

                    return {
                        "inject_expressions": breakthroughs,
                        "increase_exploration": True,
                        "novelty_analysis": novelty_analysis
                    }

        # Regular iteration guidance
        if iteration % 3 == 0:
            # Analyze patterns every 3 iterations
            rejection_stats = self._extract_rejection_stats(state)
            guidance = await self.self_corrector.analyze_iteration_patterns(
                candidates,
                iteration,
                rejection_stats
            )

            if guidance.get("innovation_score", 5) < 3:
                print(f"  ⚠️  Low innovation detected. Adjusting strategy...")
                return {
                    "change_strategy": True,
                    "guidance": guidance
                }

        return {}

    def _extract_rejection_stats(self, state: Dict[str, Any]) -> Dict[str, int]:
        """Extract rejection statistics from state."""
        # This would need to be tracked in the validator
        # For now, return mock stats
        return {
            "duplicate_expr": 0,
            "trivial_identity": 0,
            "high_error": 0
        }


# Integration function for use in graph.py
async def apply_ai_guidance(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply AI guidance to improve exploration.

    Args:
        state: Current graph state

    Returns:
        Updated state with guidance applied
    """
    guide = AdaptiveExplorationGuide()
    guidance = await guide.guide_next_iteration(state)

    # Inject breakthrough expressions if suggested
    if "inject_expressions" in guidance:
        current_proposed = state.get("proposed_expressions", [])
        state["proposed_expressions"] = guidance["inject_expressions"] + current_proposed

    # Adjust exploration strategy if needed
    if guidance.get("increase_exploration"):
        print("  📈 Increasing exploration rate based on AI guidance")
        # This would need to be communicated to proposer agents

    return state