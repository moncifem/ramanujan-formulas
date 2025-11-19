"""Prompt templates for agent types."""

EXPLORER_PROMPT = """You are a mathematical explorer discovering novel expressions for fundamental constants.

Your task: Generate {num_expressions} creative mathematical expressions that approximate {target_constant}.

Target constant: {constant_description}

INNOVATION GUIDELINES:
- Explore NOVEL patterns: nested radicals, continued fractions, infinite series
- Use creative combinations of sqrt(), fractions, powers, and arithmetic
- Think like Ramanujan: elegant, surprising, beautiful identities
- Examples of interesting patterns:
  * sqrt(1 + 2*sqrt(1 + 3*sqrt(1 + 4*sqrt(...))))
  * 1/(1 + 1/(1 + 1/(1 + ...)))
  * (1 + sqrt(5))/2  # golden ratio
  * sum of series, products, nested expressions

CONSTRAINTS:
- Use only mathematical functions: sqrt, **, +, -, *, /, ()
- Keep expressions under 200 characters
- Each expression must be DIFFERENT from others
- Be creative and surprising!

Output ONLY valid Python/SymPy expressions, one per line:
sqrt(2 + sqrt(3))
(1 + sqrt(5))/2
1 + 1/(1 + 1/(2 + 1/(3 + 1/4)))
"""

MUTATOR_PROMPT = """You are a mathematical mutator improving existing expressions.

Your task: Take these best expressions from the gene pool and mutate them to find better approximations of {target_constant}.

Target constant: {constant_description}

GENE POOL (Top candidates):
{gene_pool_expressions}

MUTATION STRATEGIES:
1. Add/remove nested levels
2. Change coefficients slightly (e.g., 2 -> 3, 1 -> 2)
3. Substitute sub-expressions (e.g., sqrt(2) -> sqrt(1+1))
4. Combine two expressions (e.g., (expr1 + expr2)/2)
5. Apply algebraic transformations

CONSTRAINTS:
- Generate {num_expressions} mutated expressions
- Keep the "spirit" of the original but make meaningful changes
- Each must be DIFFERENT
- Keep under 200 characters

Output ONLY valid Python/SymPy expressions, one per line:
sqrt(3 + sqrt(5))
(2 + sqrt(7))/3
"""

HYBRID_PROMPT = """You are a hybrid agent combining exploration and mutation.

Your task: Generate {num_expressions} expressions for {target_constant} using BOTH novel exploration AND mutation of existing candidates.

Target constant: {constant_description}

GENE POOL (for mutation inspiration):
{gene_pool_expressions}

STRATEGY:
- 50% completely novel expressions (like Explorer)
- 50% mutations of gene pool candidates (like Mutator)
- Be creative and unpredictable!

CONSTRAINTS:
- Each expression must be DIFFERENT
- Keep under 200 characters
- Use only: sqrt, **, +, -, *, /, ()

Output ONLY valid Python/SymPy expressions, one per line:
"""


def format_gene_pool(gene_pool: list) -> str:
    """Format gene pool expressions for prompts.

    Args:
        gene_pool: List of Expression objects

    Returns:
        Formatted string of expressions
    """
    if not gene_pool:
        return "(Gene pool is empty - focus on pure exploration)"

    formatted = []
    for i, expr in enumerate(gene_pool[:5], 1):  # Show top 5
        formatted.append(
            f"{i}. {expr.parsed_expr} (error: {expr.error:.2e}, score: {expr.elegance_score:.2e})"
        )

    return "\n".join(formatted)


def get_prompt(
    agent_type: str,
    target_constant: str,
    constant_description: str,
    gene_pool: list,
    num_expressions: int = 5,
) -> str:
    """Get prompt for specific agent type.

    Args:
        agent_type: "explorer", "mutator", or "hybrid"
        target_constant: Name of target constant
        constant_description: Human-readable description
        gene_pool: List of top expressions
        num_expressions: Number of expressions to generate

    Returns:
        Formatted prompt string
    """
    gene_pool_str = format_gene_pool(gene_pool)

    prompts = {
        "explorer": EXPLORER_PROMPT,
        "mutator": MUTATOR_PROMPT,
        "hybrid": HYBRID_PROMPT,
    }

    template = prompts.get(agent_type, EXPLORER_PROMPT)

    return template.format(
        target_constant=target_constant,
        constant_description=constant_description,
        gene_pool_expressions=gene_pool_str,
        num_expressions=num_expressions,
    )
