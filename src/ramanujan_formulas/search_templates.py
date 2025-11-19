"""
Search templates for genuine mathematical discovery.
Based on expert recommendations for Path A (modular) and Path B (Gamma).
"""

import random
from typing import List


class SearchTemplates:
    """Templates for generating genuinely novel mathematical expressions."""

    # Path A: Non-standard discriminants (NOT the famous ones!)
    # These are less-studied and may have undiscovered properties
    NOVEL_DISCRIMINANTS = [
        13, 17, 21, 23, 29, 31, 33, 37, 41, 47,
        53, 57, 61, 69, 71, 73, 77, 79, 83, 89,
        93, 97, 101, 103, 107, 109, 113, 127, 131, 137,
        139, 149, 151, 157, 161, 167, 173, 179, 181, 191
    ]

    # Path B: Prime denominators for Gamma search
    # Large primes are less studied
    NOVEL_PRIMES = [
        17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
        103, 107, 109, 113, 127, 131, 137, 139, 149
    ]

    @staticmethod
    def generate_path_a_hyperbolic() -> List[str]:
        """
        PATH A: Modular forms and Theta functions (Structural Novelty).
        Avoids trivial exp/sinh ratios. Uses jtheta and elliptic integrals.
        """
        expressions = []

        for _ in range(4):  # Generate 4 expressions
            D = random.choice(SearchTemplates.NOVEL_DISCRIMINANTS)
            q_val = f"mp.exp(-mp.pi * mp.sqrt({D}))"
            
            templates = [
                # Ramanujan's theta function identities usually involve ratios of theta functions
                f"mp.jtheta(3, 0, {q_val}) / mp.jtheta(4, 0, {q_val})",
                f"mp.jtheta(2, 0, {q_val})**4 + mp.jtheta(4, 0, {q_val})**4",
                # q-Pochhammer symbol (Dedekind eta related)
                f"mp.qp({q_val})", 
                # Mixed theta products
                f"mp.jtheta(3, 0, {q_val}) * mp.jtheta(4, 0, {q_val})",
            ]

            expressions.append(random.choice(templates))

        return expressions

    @staticmethod
    def generate_path_b_gamma_asymmetric() -> List[str]:
        """
        PATH B: Gamma at rational arguments with ASYMMETRIC combinations.

        NOT symmetric pairs like Γ(x)Γ(1-x) - those are Euler reflection!

        Instead: Γ(a/p)^k × Γ(b/p)^m where NOT complementary.
        """
        expressions = []

        for _ in range(4):  # Generate 4 expressions
            p = random.choice(SearchTemplates.NOVEL_PRIMES)
            # Choose a, b that are NOT complementary (a + b ≠ p)
            a = random.randint(1, p-1)
            b = random.randint(1, p-1)

            # Ensure NOT complementary
            while a + b == p:
                b = random.randint(1, p-1)

            k = random.choice([1, 2, 3, -1, -2])
            m = random.choice([1, 2, 3, -1, -2])
            pi_power = random.choice([0, 1, 2, 3, -1, -2])

            if k > 0 and m > 0:
                expr = f"mp.gamma({a}/{p})**{k} * mp.gamma({b}/{p})**{m}"
            else:
                expr = f"mp.gamma({a}/{p})"
                if abs(k) > 1:
                    expr += f"**{k}"
                expr += f" * mp.gamma({b}/{p})"
                if abs(m) > 1:
                    expr += f"**{m}"

            if pi_power != 0:
                if pi_power > 0:
                    expr += f" / mp.pi**{pi_power}"
                else:
                    expr += f" * mp.pi**{abs(pi_power)}"

            expressions.append(expr)

        return expressions

    @staticmethod
    def generate_mixed_special_functions() -> List[str]:
        """
        Mixed special function combinations.
        These are less explored and might have novel properties.
        """
        expressions = []

        for _ in range(4):
            templates = [
                # Gamma × Zeta combinations
                f"mp.gamma({random.randint(1,7)}/{random.randint(8,15)}) * mp.zeta({random.choice([3,5,7])}) / mp.pi**{random.randint(1,3)}",

                # Elliptic × Gamma
                f"mp.ellipk({random.randint(1,9)}/10) / mp.gamma({random.randint(1,5)}/{random.randint(6,12)})",

                # Hypergeometric × constants
                f"mp.hyp2f1(1/{random.randint(2,5)}, 1/{random.randint(2,5)}, 1, 1/{random.randint(2,9)})",

                # Mixed products with Polylogs
                f"mp.polylog(2, 1/{random.randint(2,5)}) + mp.log({random.randint(2,5)})**2", # Dilogarithm identity area
                
                # AGM
                f"mp.agm(1, mp.sqrt({random.choice([2,3,5,7])})) / mp.pi",
            ]

            expressions.append(random.choice(templates))

        return expressions
