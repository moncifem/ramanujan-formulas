"""Scorer for evaluating novelty and beauty of discovered curves."""

import numpy as np
from typing import List, Tuple
from ramanujan_swarm.math_engine.curve_types import CurveExpression, KNOWN_CURVES, compute_curve_signature


class CurveScorer:
    """Scores curves based on novelty, beauty, and mathematical richness."""

    def __init__(self):
        """Initialize scorer with known curve signatures."""
        self.known_signatures = {}
        # We could precompute signatures for known curves here

    def score(self, curve: CurveExpression) -> float:
        """Compute total score for a curve.

        Args:
            curve: CurveExpression with computed properties

        Returns:
            Total score (higher is better)
        """
        if not curve.points or len(curve.points) < 10:
            return 0.0

        # Compute individual scores
        curve.novelty_score = self._compute_novelty(curve)
        curve.beauty_score = self._compute_beauty(curve)
        curve.complexity_score = self._compute_complexity(curve)
        curve.mathematical_richness = self._compute_richness(curve)

        # Weighted combination
        curve.total_score = (
            curve.novelty_score * 0.30 +
            curve.beauty_score * 0.25 +
            curve.complexity_score * 0.20 +
            curve.mathematical_richness * 0.25
        )

        return curve.total_score

    def _compute_novelty(self, curve: CurveExpression) -> float:
        """Compute novelty score based on difference from known curves.

        Returns:
            Score from 0 to 1 (1 = completely novel)
        """
        # Check equation similarity to known curves
        equation_lower = curve.equation_str.lower()

        # Penalize if it's obviously a known curve type
        known_patterns = [
            "circle", "ellipse", "spiral", "rose", "cardioid",
            "limacon", "lemniscate", "cycloid", "astroid", "deltoid",
            "lissajous", "parabola", "hyperbola"
        ]

        pattern_penalty = 0.0
        for pattern in known_patterns:
            if pattern in equation_lower:
                pattern_penalty += 0.2

        # Check if the curve uses interesting combinations
        interesting_elements = [
            "phi", "golden", "sqrt(5)",  # Golden ratio
            "pi", "e**",  # Special constants
            "sin", "cos", "tan",  # Trig functions
            "exp", "log",  # Exponential/log
            "**",  # Powers
        ]

        combination_bonus = 0.0
        element_count = sum(1 for elem in interesting_elements if elem in equation_lower)
        if element_count >= 3:
            combination_bonus = 0.2  # Novel combinations
        elif element_count >= 2:
            combination_bonus = 0.1

        # Penalize very simple expressions
        simplicity_penalty = 0.0
        if len(curve.equation_str) < 10:
            simplicity_penalty = 0.3
        elif len(curve.equation_str) < 20:
            simplicity_penalty = 0.1

        # Base novelty
        novelty = 0.7 - pattern_penalty + combination_bonus - simplicity_penalty

        # Bonus for unusual symmetry
        if curve.symmetry_type == "rotational" and curve.symmetry_order >= 5:
            novelty += 0.1
        elif curve.symmetry_type == "none" and curve.is_closed:
            novelty += 0.15  # Closed but asymmetric is interesting

        return max(0.0, min(1.0, novelty))

    def _compute_beauty(self, curve: CurveExpression) -> float:
        """Compute aesthetic beauty score.

        Returns:
            Score from 0 to 1 (1 = most beautiful)
        """
        if not curve.points:
            return 0.0

        pts = np.array(curve.points)
        beauty = 0.5  # Base score

        # Symmetry is beautiful
        if curve.symmetry_type == "rotational":
            beauty += 0.15
            if curve.symmetry_order in [3, 4, 5, 6]:  # Pleasing symmetries
                beauty += 0.1
        elif curve.symmetry_type == "reflection":
            beauty += 0.1

        # Smooth curvature is beautiful
        # Low variance in curvature = smooth
        if curve.curvature_variance < 1.0:
            beauty += 0.1
        elif curve.curvature_variance > 100:
            beauty -= 0.1  # Too spiky

        # Balanced proportions (aspect ratio close to golden ratio)
        bbox = curve.bounding_box
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        if width > 0 and height > 0:
            aspect = max(width, height) / min(width, height)
            golden = 1.618
            if abs(aspect - golden) < 0.2:
                beauty += 0.1  # Golden ratio proportions
            elif abs(aspect - 1.0) < 0.1:
                beauty += 0.05  # Square proportions

        # Closed curves are often more beautiful
        if curve.is_closed:
            beauty += 0.1

        # Penalize too many self-intersections (can be messy)
        if curve.self_intersections > 10:
            beauty -= 0.1

        # Some cusps add interest, too many are ugly
        if 1 <= curve.num_cusps <= 6:
            beauty += 0.05
        elif curve.num_cusps > 10:
            beauty -= 0.1

        return max(0.0, min(1.0, beauty))

    def _compute_complexity(self, curve: CurveExpression) -> float:
        """Compute complexity score (not too simple, not too complex).

        Returns:
            Score from 0 to 1 (1 = optimal complexity)
        """
        # Equation length as proxy for complexity
        eq_len = len(curve.equation_str)

        # Ideal length range: 15-80 characters
        if eq_len < 10:
            length_score = 0.2  # Too simple
        elif eq_len < 15:
            length_score = 0.5
        elif eq_len <= 50:
            length_score = 1.0  # Optimal
        elif eq_len <= 80:
            length_score = 0.8
        elif eq_len <= 120:
            length_score = 0.5
        else:
            length_score = 0.2  # Too complex

        # Number of mathematical operations
        ops = sum(curve.equation_str.count(op) for op in ['+', '-', '*', '/', '**', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt'])
        if ops < 2:
            ops_score = 0.3
        elif ops <= 8:
            ops_score = 1.0
        elif ops <= 15:
            ops_score = 0.7
        else:
            ops_score = 0.4

        # Geometric complexity
        geo_score = 0.5
        if curve.self_intersections > 0:
            geo_score += 0.1
        if curve.num_cusps > 0:
            geo_score += 0.1
        if not curve.is_closed:
            geo_score += 0.1  # Open curves can be more complex

        return (length_score * 0.4 + ops_score * 0.3 + geo_score * 0.3)

    def _compute_richness(self, curve: CurveExpression) -> float:
        """Compute mathematical richness score.

        Returns:
            Score from 0 to 1 (1 = most mathematically rich)
        """
        richness = 0.3  # Base score

        # Uses special constants
        equation = curve.equation_str.lower()
        if "pi" in equation:
            richness += 0.1
        if "e" in equation and ("exp" in equation or "e**" in equation):
            richness += 0.1
        if "phi" in equation or "sqrt(5)" in equation or "golden" in equation:
            richness += 0.15

        # Combines different function types
        has_trig = any(f in equation for f in ["sin", "cos", "tan"])
        has_exp = any(f in equation for f in ["exp", "log"])
        has_power = "**" in equation
        has_sqrt = "sqrt" in equation

        func_types = sum([has_trig, has_exp, has_power, has_sqrt])
        richness += func_types * 0.08

        # Interesting topological features
        if curve.is_closed and curve.enclosed_area > 0:
            richness += 0.05
        if curve.symmetry_type != "none":
            richness += 0.05
        if 1 <= curve.self_intersections <= 5:
            richness += 0.1  # Some self-intersections are interesting

        # Parametric curves that are not simple circles/ellipses
        if curve.curve_type == "parametric":
            if has_trig and has_power:
                richness += 0.1

        return max(0.0, min(1.0, richness))

    def find_similar_known_curves(self, curve: CurveExpression) -> List[str]:
        """Find known curves similar to this one.

        Args:
            curve: CurveExpression to compare

        Returns:
            List of names of similar known curves
        """
        similar = []

        # Compare based on properties
        for name, known in KNOWN_CURVES.items():
            similarity = 0

            # Same curve type
            if known.get("type") == curve.curve_type:
                similarity += 1

            # Same symmetry
            if known.get("symmetry") == curve.symmetry_type:
                similarity += 1

            if similarity >= 2:
                similar.append(name)

        return similar[:5]  # Return top 5 similar
