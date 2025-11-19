"""
Breakthrough Pattern Injector.
Injects promising patterns based on successful discoveries.
"""

from typing import List, Dict, Any
import random
from mpmath import mp


class BreakthroughInjector:
    """
    Injects high-potential expressions based on successful patterns.
    """

    def __init__(self):
        """Initialize the breakthrough injector."""
        self.successful_patterns = []
        self.tested_discriminants = set()

    def analyze_success_pattern(self, expression: str, error: float):
        """
        Analyze a successful expression to extract patterns.

        Args:
            expression: Successful expression
            error: Error achieved
        """
        if error < 1e-20:
            self.successful_patterns.append({
                "expression": expression,
                "error": error,
                "pattern": self._extract_pattern(expression)
            })

    def _extract_pattern(self, expr: str) -> Dict[str, Any]:
        """Extract pattern components from expression."""
        pattern = {
            "has_tanh": "tanh" in expr,
            "has_sqrt": "sqrt" in expr,
            "has_pi": "pi" in expr,
            "has_gamma": "gamma" in expr,
            "discriminants": []
        }

        # Extract discriminants (numbers after sqrt)
        import re
        sqrt_pattern = r'sqrt\((\d+)\)'
        matches = re.findall(sqrt_pattern, expr)
        pattern["discriminants"] = [int(m) for m in matches]

        return pattern

    def generate_breakthrough_batch(self, count: int = 20) -> List[str]:
        """
        Generate a batch of breakthrough expressions based on valid Ramanujan identities.
        We use known NEAR-INTEGERS to bootstrap the gene pool.
        """
        expressions = []

        # 1. Known Ramanujan Constants (Heegner numbers)
        # These are the best starting points for mutation.
        for d in [163, 67, 43, 19, 58, 37]:
            expressions.append(f"mp.exp(mp.pi * mp.sqrt({d}))")
        
        # 2. Variations on the Heegner theme
        expressions.extend([
            "mp.exp(mp.pi * mp.sqrt(163)) / 262537412640768744",
            "mp.exp(mp.pi * mp.sqrt(58))",
            "mp.exp(mp.pi * mp.sqrt(37))",
        ])
        
        # 3. Other near-integer formulas
        expressions.extend([
            "mp.pi * mp.e * mp.phi",  # Explore constant products
            "(mp.pi**2 + mp.e**2) / 5",  # Combinations
            "mp.log(mp.log(mp.exp(mp.exp(mp.e))))",  # Nested functions
        ])
        
        # 4. Theta function explorations
        for q_exp in [1, 2, 3, 4, 5]:
            expressions.append(f"mp.jtheta(3, 0, mp.exp(-mp.pi * {q_exp}))")
        
        # 5. Simple but effective seeds
        expressions.extend([
            "mp.sqrt(2) * mp.sqrt(3) * mp.sqrt(5)",
            "mp.gamma(1/3) * mp.gamma(2/3)",
            "mp.zeta(3) * 120",
        ])

        # Fill the rest with systematic exploration
        discriminants = [11, 13, 17, 23, 29, 31, 37, 41, 47, 53, 59, 61, 71, 73, 79, 83, 89, 97]
        for d in discriminants[:count - len(expressions)]:
            expressions.append(f"mp.exp(mp.pi * mp.sqrt({d}))")

        # Remove duplicates and return
        unique_expressions = list(set(expressions))
        random.shuffle(unique_expressions)
        return unique_expressions[:count]

    def inject_into_state(
        self,
        state: Dict[str, Any],
        injection_rate: float = 0.2
    ) -> Dict[str, Any]:
        """
        Inject breakthrough expressions into the system state.

        Args:
            state: Current system state
            injection_rate: Fraction of expressions to replace with breakthroughs

        Returns:
            Updated state
        """
        current_pool = state.get("best_candidates", [])
        
        # Generate high-quality seeds (increase to 20 for better bootstrapping)
        seeds = self.generate_breakthrough_batch(20)
        
        # Create fake candidates for these seeds
        from .utils import evaluate_expression, compute_error, compute_elegance_score
        
        new_candidates = []
        for expr in seeds:
            try:
                val = evaluate_expression(expr)
                if val is not None:
                    error, _ = compute_error(val)
                    # Only add if error is valid (not inf) and small enough to be useful parent
                    if error < 1e-1 and error != float('inf'):
                        score = compute_elegance_score(error, expr)
                        new_candidates.append({
                            "expression": expr,
                            "value_str": str(val)[:50],
                            "error": error,
                            "score": score, 
                            "source": "injected_seed"
                        })
            except:
                pass
        
        state["best_candidates"] = current_pool + new_candidates
        
        print(f"  💉 Injected {len(new_candidates)} SEED expressions into Gene Pool")

        return state


# Singleton instance
_breakthrough_injector = None


def get_breakthrough_injector() -> BreakthroughInjector:
    """Get or create the breakthrough injector instance."""
    global _breakthrough_injector
    if _breakthrough_injector is None:
        _breakthrough_injector = BreakthroughInjector()
    return _breakthrough_injector


def inject_breakthroughs(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to inject breakthroughs into state.

    Args:
        state: Current system state

    Returns:
        Updated state with breakthrough expressions
    """
    injector = get_breakthrough_injector()

    # Always inject if pool is empty or small!
    pool_size = len(state.get("best_candidates", []))
    
    if pool_size < 5 or state.get("iteration", 1) % 3 == 0:
        state = injector.inject_into_state(state)

    return state
