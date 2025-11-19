"""Prompts for generating novel curve equations."""

from typing import List
from ramanujan_swarm.math_engine.curve_types import CurveExpression


def get_curve_prompt(
    agent_type: str,
    curve_type: str,
    gene_pool: List[CurveExpression],
    num_curves: int = 3,
) -> str:
    """Get prompt for generating novel curve equations.

    Args:
        agent_type: Type of agent (explorer, mutator, hybrid)
        curve_type: Type of curve to generate (parametric, polar, implicit)
        gene_pool: Current pool of interesting curves
        num_curves: Number of curves to generate

    Returns:
        Prompt string
    """
    if agent_type == "explorer":
        return _get_explorer_prompt(curve_type, num_curves)
    elif agent_type == "mutator":
        return _get_mutator_prompt(curve_type, gene_pool, num_curves)
    else:  # hybrid
        return _get_hybrid_prompt(curve_type, gene_pool, num_curves)


def _get_explorer_prompt(curve_type: str, num_curves: int) -> str:
    """Prompt for explorer agents to generate completely novel curves."""

    if curve_type == "parametric":
        return f"""You are a mathematical artist discovering NEW curve equations that have NEVER existed before.

Your mission: Create {num_curves} COMPLETELY NOVEL parametric curve equations.

FORMAT - Return EXACTLY like this, one per line:
x(t) = [expression], y(t) = [expression]

REQUIREMENTS:
1. Create curves that look NOTHING like known curves (circles, spirals, roses, lissajous)
2. Combine functions in UNEXPECTED ways
3. Use interesting constants: pi, e, phi (golden ratio = (1+sqrt(5))/2)
4. Mix: sin, cos, tan, exp, log, sqrt, powers

CREATIVE STRATEGIES:
- Multiply trig functions by exponential decay/growth
- Use non-integer frequencies: sin(pi*t), cos(phi*t), sin(e*t)
- Combine polynomial and trigonometric: t**2 * sin(t)
- Use hyperbolic functions: sinh, cosh, tanh
- Phase shifts with irrational multiples: sin(t + pi/phi)
- Damped oscillations: exp(-t/10) * sin(5*t)
- Beating patterns: sin(t) * sin(1.1*t)

EXAMPLE NOVEL CURVES (for inspiration, create DIFFERENT ones):
x(t) = cos(t) * exp(sin(phi*t)/3), y(t) = sin(t) * cos(e*t)
x(t) = sin(t**1.5) * cos(2*t), y(t) = cos(t**1.5) * sin(3*t)
x(t) = t * cos(t) / (1 + t**2/100), y(t) = t * sin(t) / (1 + t**2/100)

Generate {num_curves} UNIQUE curves. Be CREATIVE. Surprise me.
Use t range from 0 to 2*pi (or larger for spirals)."""

    elif curve_type == "polar":
        return f"""You are a mathematical artist discovering NEW polar curve equations that have NEVER existed before.

Your mission: Create {num_curves} COMPLETELY NOVEL polar curve equations r(θ).

FORMAT - Return EXACTLY like this, one per line:
r = [expression in theta]

REQUIREMENTS:
1. Create curves that look NOTHING like standard roses, spirals, or limacons
2. Combine functions in UNEXPECTED ways
3. Use interesting constants: pi, e, phi (golden ratio)
4. Mix: sin, cos, tan, exp, log, sqrt, powers

CREATIVE STRATEGIES:
- Modulated roses: cos(n*theta) * (1 + sin(m*theta)/k)
- Spiral-rose hybrids: theta/10 + cos(5*theta)
- Exponential modulation: exp(cos(theta)) * sin(3*theta)
- Use phi: cos(phi*theta) creates aperiodic patterns
- Product of functions: sin(theta) * cos(phi*theta) * (1 + theta/20)
- Fractal-like: |cos(theta)| + |cos(3*theta)|/3 + |cos(9*theta)|/9

EXAMPLE NOVEL CURVES (create DIFFERENT ones):
r = exp(sin(theta)) * cos(phi*theta) + 1
r = sqrt(abs(sin(3*theta))) * (2 + cos(theta))
r = (1 + sin(theta)**3) * cos(e*theta)

Generate {num_curves} UNIQUE curves. Be CREATIVE and UNEXPECTED.
Use theta range from 0 to 2*pi (or larger for spirals)."""

    else:  # implicit
        return f"""You are a mathematical artist discovering NEW implicit curve equations f(x,y)=0 that have NEVER existed before.

Your mission: Create {num_curves} COMPLETELY NOVEL implicit curve equations.

FORMAT - Return EXACTLY like this, one per line:
f(x,y) = [expression] = 0

REQUIREMENTS:
1. Create curves that look NOTHING like conic sections or standard algebraic curves
2. Combine functions in UNEXPECTED ways
3. Use interesting constants: pi, e, phi
4. Mix: sin, cos, exp, log, sqrt, powers

CREATIVE STRATEGIES:
- Mix algebraic and transcendental: x**3 + y**3 - sin(x*y)
- Use products: x*y*(x**2 + y**2 - 1) - sin(x)*cos(y)
- Modular forms: (x**2 + y**2)**2 - x**3 + y*sin(x)
- Exponential decay: exp(-(x**2 + y**2)) - cos(x)*cos(y) + 0.5
- Level sets of interesting functions

EXAMPLE NOVEL CURVES (create DIFFERENT ones):
x**4 + y**4 - (x*y)**2 - sin(x + y)
(x**2 + y**2 - 1)**2 - x**3*y + exp(-x**2 - y**2)
sin(x**2 + y**2) - cos(x)*sin(y)

Generate {num_curves} UNIQUE curves. Push boundaries.
Curves should be in range x,y from -5 to 5."""


def _get_mutator_prompt(curve_type: str, gene_pool: List[CurveExpression], num_curves: int) -> str:
    """Prompt for mutator agents to evolve existing interesting curves."""

    # Get some curves from gene pool
    pool_examples = []
    for curve in gene_pool[:5]:
        if curve.curve_type == curve_type or curve_type == "any":
            pool_examples.append(curve.equation_str)

    if not pool_examples:
        # Fall back to explorer behavior
        return _get_explorer_prompt(curve_type, num_curves)

    pool_str = "\n".join(f"- {eq}" for eq in pool_examples[:3])

    if curve_type == "parametric":
        return f"""You are evolving existing interesting curves into NEW variations.

CURRENT INTERESTING CURVES (mutate these):
{pool_str}

Your mission: Create {num_curves} MUTATIONS that are BETTER than the originals.

FORMAT - Return EXACTLY like this, one per line:
x(t) = [expression], y(t) = [expression]

MUTATION STRATEGIES:
1. ADD new terms: add sin(phi*t) to an existing expression
2. MULTIPLY by modulating functions: multiply by (1 + sin(t)/3)
3. CHANGE constants: replace 2 with phi, replace 3 with e
4. COMPOSE functions: replace t with sin(t) or t**2
5. COMBINE two curves: take x from one, y from another
6. ADD damping/growth: multiply by exp(-t/20) or exp(t/50)
7. CHANGE frequencies: multiply t by phi or sqrt(2)

Create {num_curves} mutations. Each should be NOTICEABLY different but INSPIRED by the pool."""

    elif curve_type == "polar":
        return f"""You are evolving existing interesting polar curves into NEW variations.

CURRENT INTERESTING CURVES (mutate these):
{pool_str}

Your mission: Create {num_curves} MUTATIONS that are BETTER than the originals.

FORMAT - Return EXACTLY like this, one per line:
r = [expression in theta]

MUTATION STRATEGIES:
1. ADD terms: add cos(phi*theta)/2 to existing expression
2. MULTIPLY by envelope: multiply by (1 + theta/20) for spiral
3. CHANGE frequencies: multiply theta by phi or e
4. MODULATE amplitude: multiply by (1 + sin(theta)/3)
5. ADD higher harmonics: add cos(5*theta)/5
6. COMBINE curves: average two interesting curves

Create {num_curves} mutations. Each should be VISUALLY DIFFERENT but RELATED to pool."""

    else:  # implicit
        return f"""You are evolving existing interesting implicit curves into NEW variations.

CURRENT INTERESTING CURVES (mutate these):
{pool_str}

Your mission: Create {num_curves} MUTATIONS that are BETTER than the originals.

FORMAT - Return EXACTLY like this, one per line:
f(x,y) = [expression] = 0

MUTATION STRATEGIES:
1. ADD terms: add sin(x*y) or exp(-x**2 - y**2)
2. CHANGE powers: replace x**2 with x**3 or x**phi
3. ADD symmetry breaking: add small terms like 0.1*x
4. MULTIPLY parts: multiply a term by (1 + x/10)
5. COMPOSE: replace x with sin(x) in one term

Create {num_curves} mutations. Preserve what made originals interesting, add novelty."""


def _get_hybrid_prompt(curve_type: str, gene_pool: List[CurveExpression], num_curves: int) -> str:
    """Prompt for hybrid agents that combine exploration and mutation."""

    pool_examples = []
    for curve in gene_pool[:3]:
        pool_examples.append(curve.equation_str)

    pool_str = "\n".join(f"- {eq}" for eq in pool_examples) if pool_examples else "(no existing curves yet)"

    if curve_type == "parametric":
        return f"""You are a mathematical innovator creating BREAKTHROUGH curve equations.

CURRENT GENE POOL:
{pool_str}

Your mission: Create {num_curves} curves that TRANSCEND current knowledge.

FORMAT - Return EXACTLY like this, one per line:
x(t) = [expression], y(t) = [expression]

BREAKTHROUGH STRATEGIES:
1. HYBRIDIZE: Combine the BEST features of pool curves with NEW elements
2. ABSTRACT: Take a concept (like "spiraling inward while oscillating") and express it mathematically
3. GOLDEN RATIO MAGIC: Use phi extensively - it creates beautiful aperiodic patterns
4. MULTI-FREQUENCY: Combine multiple frequencies that are irrationally related
5. ENVELOPE INNOVATION: Create curves where the envelope is itself interesting

THINK ABOUT:
- What would Ramanujan create?
- What curve would represent the Fibonacci sequence visually?
- What if we crossed a spiral with a rose with a lissajous?

Create {num_curves} BREAKTHROUGH curves."""

    elif curve_type == "polar":
        return f"""You are a mathematical innovator creating BREAKTHROUGH polar curves.

CURRENT GENE POOL:
{pool_str}

Your mission: Create {num_curves} curves that TRANSCEND current knowledge.

FORMAT - Return EXACTLY like this, one per line:
r = [expression in theta]

BREAKTHROUGH STRATEGIES:
1. QUASI-PERIODIC: Use phi to create patterns that almost repeat but don't
2. FRACTAL-LIKE: Sum of terms at multiple scales
3. PHASE SPACE: r depends on theta and derivative concepts
4. GOLDEN SPIRALS: Variations on r = exp(theta/phi)

Create {num_curves} BREAKTHROUGH curves."""

    else:  # implicit
        return f"""You are a mathematical innovator creating BREAKTHROUGH implicit curves.

CURRENT GENE POOL:
{pool_str}

Your mission: Create {num_curves} curves that TRANSCEND current knowledge.

FORMAT - Return EXACTLY like this, one per line:
f(x,y) = [expression] = 0

BREAKTHROUGH STRATEGIES:
1. LEVEL SETS of novel functions
2. MIX algebraic (polynomials) with transcendental (sin, exp)
3. SYMMETRY with perturbation
4. PATTERNS inspired by physics (wave interference, etc.)

Create {num_curves} BREAKTHROUGH curves."""
