"""Novel Curve Discovery: Find mathematical curves that have never existed before.

This system uses genetic algorithms and LLM creativity to discover entirely new
mathematical curves based on novelty, beauty, and mathematical richness.
"""

import asyncio
import json
import os
import random
import time
from typing import List
from ramanujan_swarm.config import config
from ramanujan_swarm.math_engine.curve_types import CurveExpression
from ramanujan_swarm.math_engine.curve_evaluator import CurveEvaluator
from ramanujan_swarm.math_engine.curve_scorer import CurveScorer
from ramanujan_swarm.agents.curve_prompts import get_curve_prompt

# Output file for discovered curves
CURVES_JSON_PATH = "outputs/novel_curves.json"


class CurveDiscoveryAgent:
    """Agent for generating novel curve equations."""

    def __init__(self, agent_id: int, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on config."""
        if config.llm_provider == "blackbox":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                api_key=config.blackbox_api_key,
                base_url="https://api.blackbox.ai/v1",
            )
        elif config.llm_provider == "claude":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                api_key=config.anthropic_api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {config.llm_provider}")

    async def generate_curves(
        self,
        gene_pool: List[CurveExpression],
        generation: int,
        num_curves: int = 3,
    ) -> List[CurveExpression]:
        """Generate novel curve equations.

        Args:
            gene_pool: Current pool of interesting curves
            generation: Current generation number
            num_curves: Number of curves to generate

        Returns:
            List of CurveExpression objects
        """
        # Randomly choose curve type
        curve_type = random.choice(["parametric", "polar", "parametric"])  # Weighted towards parametric

        # Get prompt
        prompt = get_curve_prompt(
            agent_type=self.agent_type,
            curve_type=curve_type,
            gene_pool=gene_pool,
            num_curves=num_curves,
        )

        # Call LLM
        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.content
        except Exception as e:
            print(f"  Agent {self.agent_id} LLM error: {e}")
            return []

        # Parse response into curves
        curves = self._parse_response(response_text, curve_type, generation)
        return curves

    def _parse_response(
        self, response_text: str, curve_type: str, generation: int
    ) -> List[CurveExpression]:
        """Parse LLM response into CurveExpression objects."""
        curves = []
        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip markdown and explanatory text
            if line.startswith("#") or line.startswith("```") or line.startswith("-"):
                continue
            if any(word in line.lower() for word in ["here", "example", "create", "generate", "note"]):
                continue

            try:
                curve = CurveExpression(
                    curve_type=curve_type,
                    equation_str=line,
                    agent_type=self.agent_type,
                    generation=generation,
                    timestamp=time.time(),
                )

                # Parse based on curve type
                if curve_type == "parametric" and "x(t)" in line.lower() or "=" in line and "," in line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        x_part = parts[0].split("=")[-1].strip()
                        y_part = parts[1].split("=")[-1].strip()
                        curve.x_expr = x_part
                        curve.y_expr = y_part

                elif curve_type == "polar" and "r" in line.lower():
                    r_part = line.split("=")[-1].strip()
                    curve.polar_expr = r_part

                elif curve_type == "implicit":
                    # Remove "= 0" if present
                    expr = line.replace("= 0", "").replace("=0", "")
                    if "f(x,y)" in expr.lower():
                        expr = expr.split("=")[-1].strip()
                    curve.implicit_expr = expr

                curves.append(curve)

            except Exception as e:
                continue

        return curves


async def evaluate_and_score_curve(curve: CurveExpression) -> CurveExpression:
    """Evaluate a curve and compute its properties and scores."""
    evaluator = CurveEvaluator()
    scorer = CurveScorer()

    # Render the curve
    if curve.curve_type == "parametric" and curve.x_expr and curve.y_expr:
        curve.points = evaluator.evaluate_parametric(curve.x_expr, curve.y_expr)
    elif curve.curve_type == "polar" and curve.polar_expr:
        curve.points = evaluator.evaluate_polar(curve.polar_expr)
    elif curve.curve_type == "implicit" and curve.implicit_expr:
        curve.points = evaluator.evaluate_implicit(curve.implicit_expr)

    if not curve.points:
        return curve

    # Compute properties
    curve = evaluator.compute_properties(curve)

    # Score the curve
    scorer.score(curve)

    # Find similar known curves
    curve.similar_curves = scorer.find_similar_known_curves(curve)

    return curve


def save_curves_to_json(curves: List[CurveExpression], generation: int):
    """Save discovered curves to JSON file."""
    os.makedirs("outputs", exist_ok=True)

    # Load existing or create new
    if os.path.exists(CURVES_JSON_PATH):
        with open(CURVES_JSON_PATH, "r") as f:
            data = json.load(f)
    else:
        data = {
            "metadata": {
                "purpose": "Novel Curve Discovery",
                "description": "Mathematical curves that have never existed before",
            },
            "generations": [],
            "top_curves": [],
            "all_curves": [],
        }

    # Add this generation's curves
    gen_data = {
        "generation": generation,
        "num_curves": len(curves),
        "curves": []
    }

    for curve in curves:
        curve_data = {
            "curve_type": curve.curve_type,
            "equation": curve.equation_str,
            "x_expr": curve.x_expr,
            "y_expr": curve.y_expr,
            "polar_expr": curve.polar_expr,
            "implicit_expr": curve.implicit_expr,
            "scores": {
                "total": round(float(curve.total_score), 4),
                "novelty": round(float(curve.novelty_score), 4),
                "beauty": round(float(curve.beauty_score), 4),
                "complexity": round(float(curve.complexity_score), 4),
                "richness": round(float(curve.mathematical_richness), 4),
            },
            "properties": {
                "symmetry_type": curve.symmetry_type,
                "symmetry_order": int(curve.symmetry_order),
                "is_closed": bool(curve.is_closed),
                "self_intersections": int(curve.self_intersections),
                "num_cusps": int(curve.num_cusps),
                "arc_length": round(float(curve.arc_length), 4) if curve.arc_length else 0,
            },
            "similar_known_curves": curve.similar_curves,
            "agent_type": curve.agent_type,
            "generation": int(curve.generation),
        }
        gen_data["curves"].append(curve_data)
        data["all_curves"].append(curve_data)

    data["generations"].append(gen_data)

    # Update top curves (keep best 20)
    all_scored = [(c, c["scores"]["total"]) for c in data["all_curves"]]
    all_scored.sort(key=lambda x: x[1], reverse=True)
    data["top_curves"] = [c[0] for c in all_scored[:20]]

    # Save
    with open(CURVES_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)


async def main():
    """Run the novel curve discovery system."""

    print("=" * 60)
    print("NOVEL CURVE DISCOVERY SYSTEM")
    print("Finding mathematical curves that have NEVER existed before")
    print("=" * 60)

    # Validate API keys
    try:
        config.validate_api_keys()
    except ValueError as e:
        print(f"\n Configuration Error: {e}")
        return

    print(f"\nConfiguration:")
    print(f"  LLM Provider: {config.llm_provider.upper()}")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Swarm size: {config.swarm_size} agents")
    print(f"  Max generations: {config.max_generations}")

    print("\n" + "=" * 60 + "\n")

    # Initialize agents
    agents = []
    for i in range(config.swarm_size):
        if i < config.swarm_size * config.explorer_fraction:
            agent_type = "explorer"
        elif i < config.swarm_size * (config.explorer_fraction + config.mutator_fraction):
            agent_type = "mutator"
        else:
            agent_type = "hybrid"
        agents.append(CurveDiscoveryAgent(i, agent_type))

    # Gene pool of interesting curves
    gene_pool: List[CurveExpression] = []

    # Clear previous results
    if os.path.exists(CURVES_JSON_PATH):
        os.remove(CURVES_JSON_PATH)

    # Run generations
    for generation in range(config.max_generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {generation + 1}")
        print(f"{'='*60}\n")

        # Generate curves from all agents in parallel
        all_curves = []
        tasks = [agent.generate_curves(gene_pool, generation) for agent in agents]
        results = await asyncio.gather(*tasks)

        for i, curves in enumerate(results):
            print(f"  Agent {i} ({agents[i].agent_type}) generated {len(curves)} curves")
            all_curves.extend(curves)

        print(f"\n Evaluating {len(all_curves)} curves...")

        # Evaluate and score all curves
        valid_curves = []
        for curve in all_curves:
            try:
                curve = await evaluate_and_score_curve(curve)
                if curve.points and len(curve.points) > 10 and curve.total_score > 0:
                    valid_curves.append(curve)
            except Exception as e:
                continue

        print(f"  Valid curves: {len(valid_curves)}")

        if not valid_curves:
            print("  No valid curves found, continuing...")
            continue

        # Sort by score
        valid_curves.sort(key=lambda c: c.total_score, reverse=True)

        # Show best curves
        print(f"\n Top curves this generation:")
        for i, curve in enumerate(valid_curves[:3]):
            print(f"  {i+1}. Score: {curve.total_score:.3f}")
            print(f"     Type: {curve.curve_type}")
            print(f"     Equation: {curve.equation_str[:60]}...")
            print(f"     Novelty: {curve.novelty_score:.2f}, Beauty: {curve.beauty_score:.2f}")
            print()

        # Update gene pool (keep top curves)
        gene_pool.extend(valid_curves)
        gene_pool.sort(key=lambda c: c.total_score, reverse=True)
        gene_pool = gene_pool[:config.gene_pool_size]

        # Save to JSON
        save_curves_to_json(valid_curves, generation)
        print(f" Saved {len(valid_curves)} curves to {CURVES_JSON_PATH}")

    # Final summary
    print("\n" + "=" * 60)
    print("DISCOVERY COMPLETE")
    print("=" * 60)
    print(f"\nTotal curves in gene pool: {len(gene_pool)}")
    print(f"\nTop 5 discovered curves:")
    for i, curve in enumerate(gene_pool[:5]):
        print(f"\n{i+1}. Score: {curve.total_score:.3f}")
        print(f"   {curve.equation_str}")

    print(f"\nResults saved to: {CURVES_JSON_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
