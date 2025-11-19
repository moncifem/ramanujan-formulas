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
        for d in [163, 67, 43, 19]:
            expressions.append(f"mp.exp(mp.pi * mp.sqrt({d}))")
        
        # 2. Ramanujan's class invariants G_n (approximate integers)
        # G_n = 2^(-1/4) * e^(pi*sqrt(n)/24) * ...
        # We try the raw exponential form which often is close to integer
        for d in [58, 93, 37]:
             expressions.append(f"mp.exp(mp.pi * mp.sqrt({d})) / 24") # Often close to integer
        
        # 3. Pi approximations (Ramanujan's style)
        expressions.extend([
            "9801 / (mp.sqrt(8) * 1103)", # Approx for pi/4 * ... related to 1/pi
            "mp.exp(mp.pi * mp.sqrt(58)) / 396**4", # Related to 1/pi series
        ])

        # 4. Mathematical Constants Coincidences
        expressions.extend([
            "mp.exp(mp.pi) - mp.pi", # Close to 20
            "mp.pi**4 + mp.pi**5 - mp.e**6", # Near zero/integer?
            "163 * (mp.pi - mp.e)", # Random check
        ])
        
        # 5. Complex function values near integers
        # j-function(tau) is integer for complex multiplication CM points
        # j(i) = 1728
        expressions.extend([
            "1728", # Exact
            "mp.jtheta(3, 0, mp.exp(-mp.pi))**4", # Exact?
            "mp.exp(mp.pi * mp.sqrt(22)) - 2500000", # Example
        ])

        # Fill the rest with variations
        while len(expressions) < count:
            d = random.randint(20, 200)
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
        
        # Generate high-quality seeds
        seeds = self.generate_breakthrough_batch(10)
        
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
