"""Curve data types and known curves database for novel curve discovery."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import hashlib


@dataclass
class CurveExpression:
    """Represents a mathematical curve equation."""

    # Curve definition
    curve_type: str  # "parametric", "implicit", "polar"
    equation_str: str  # Raw equation string

    # For parametric: x(t), y(t)
    x_expr: str = ""
    y_expr: str = ""

    # For implicit: f(x,y) = 0
    implicit_expr: str = ""

    # For polar: r(θ)
    polar_expr: str = ""

    # Parsed expressions (SymPy)
    parsed_x: str = ""
    parsed_y: str = ""
    parsed_implicit: str = ""
    parsed_polar: str = ""

    # Computed properties
    points: List[Tuple[float, float]] = field(default_factory=list)
    symmetry_type: str = "none"  # "rotational", "reflection", "point", "none"
    symmetry_order: int = 1
    num_loops: int = 0
    num_cusps: int = 0
    self_intersections: int = 0
    is_closed: bool = False
    bounding_box: Tuple[float, float, float, float] = (0, 0, 0, 0)  # xmin, xmax, ymin, ymax
    arc_length: float = 0.0
    enclosed_area: float = 0.0
    curvature_variance: float = 0.0
    fractal_dimension: float = 1.0

    # Scoring
    novelty_score: float = 0.0
    beauty_score: float = 0.0
    complexity_score: float = 0.0
    mathematical_richness: float = 0.0
    total_score: float = 0.0

    # Metadata
    agent_type: str = ""
    generation: int = 0
    timestamp: float = 0.0

    # Hashes for deduplication
    equation_hash: str = ""
    shape_hash: str = ""  # Based on rendered points

    # Similar known curves
    similar_curves: List[str] = field(default_factory=list)


# Database of known curves for novelty comparison
KNOWN_CURVES = {
    # Basic curves
    "circle": {
        "type": "parametric",
        "x": "cos(t)",
        "y": "sin(t)",
        "symmetry": "rotational",
        "description": "Perfect circle"
    },
    "ellipse": {
        "type": "parametric",
        "x": "a*cos(t)",
        "y": "b*sin(t)",
        "symmetry": "reflection",
        "description": "Ellipse with semi-axes a and b"
    },

    # Spirals
    "archimedean_spiral": {
        "type": "polar",
        "r": "a*theta",
        "symmetry": "none",
        "description": "Spiral of Archimedes"
    },
    "logarithmic_spiral": {
        "type": "polar",
        "r": "a*exp(b*theta)",
        "symmetry": "none",
        "description": "Logarithmic (equiangular) spiral"
    },
    "fermat_spiral": {
        "type": "polar",
        "r": "sqrt(theta)",
        "symmetry": "point",
        "description": "Fermat's spiral"
    },

    # Rose curves
    "rose_3": {
        "type": "polar",
        "r": "cos(3*theta)",
        "symmetry": "rotational",
        "description": "3-petaled rose"
    },
    "rose_4": {
        "type": "polar",
        "r": "cos(4*theta)",
        "symmetry": "rotational",
        "description": "8-petaled rose"
    },
    "rose_5": {
        "type": "polar",
        "r": "cos(5*theta)",
        "symmetry": "rotational",
        "description": "5-petaled rose"
    },

    # Limacons
    "cardioid": {
        "type": "polar",
        "r": "1 + cos(theta)",
        "symmetry": "reflection",
        "description": "Cardioid (heart shape)"
    },
    "limacon": {
        "type": "polar",
        "r": "a + b*cos(theta)",
        "symmetry": "reflection",
        "description": "Limacon of Pascal"
    },

    # Lemniscates
    "lemniscate_bernoulli": {
        "type": "polar",
        "r": "sqrt(cos(2*theta))",
        "symmetry": "point",
        "description": "Lemniscate of Bernoulli (figure-8)"
    },
    "lemniscate_gerono": {
        "type": "implicit",
        "f": "x**4 - x**2 + y**2",
        "symmetry": "point",
        "description": "Lemniscate of Gerono"
    },

    # Cycloids
    "cycloid": {
        "type": "parametric",
        "x": "t - sin(t)",
        "y": "1 - cos(t)",
        "symmetry": "none",
        "description": "Cycloid"
    },
    "epicycloid": {
        "type": "parametric",
        "x": "(R+r)*cos(t) - r*cos((R+r)*t/r)",
        "y": "(R+r)*sin(t) - r*sin((R+r)*t/r)",
        "symmetry": "rotational",
        "description": "Epicycloid"
    },
    "hypocycloid": {
        "type": "parametric",
        "x": "(R-r)*cos(t) + r*cos((R-r)*t/r)",
        "y": "(R-r)*sin(t) - r*sin((R-r)*t/r)",
        "symmetry": "rotational",
        "description": "Hypocycloid"
    },

    # Special curves
    "astroid": {
        "type": "parametric",
        "x": "cos(t)**3",
        "y": "sin(t)**3",
        "symmetry": "rotational",
        "description": "Astroid (4-cusped hypocycloid)"
    },
    "deltoid": {
        "type": "parametric",
        "x": "2*cos(t) + cos(2*t)",
        "y": "2*sin(t) - sin(2*t)",
        "symmetry": "rotational",
        "description": "Deltoid (3-cusped hypocycloid)"
    },

    # Lissajous
    "lissajous_3_2": {
        "type": "parametric",
        "x": "sin(3*t)",
        "y": "sin(2*t)",
        "symmetry": "reflection",
        "description": "Lissajous curve (3:2)"
    },
    "lissajous_5_4": {
        "type": "parametric",
        "x": "sin(5*t)",
        "y": "sin(4*t)",
        "symmetry": "reflection",
        "description": "Lissajous curve (5:4)"
    },

    # Folium and special algebraic
    "folium_descartes": {
        "type": "implicit",
        "f": "x**3 + y**3 - 3*x*y",
        "symmetry": "reflection",
        "description": "Folium of Descartes"
    },
    "witch_agnesi": {
        "type": "parametric",
        "x": "t",
        "y": "1/(1+t**2)",
        "symmetry": "reflection",
        "description": "Witch of Agnesi"
    },

    # Conic sections
    "parabola": {
        "type": "parametric",
        "x": "t",
        "y": "t**2",
        "symmetry": "reflection",
        "description": "Parabola"
    },
    "hyperbola": {
        "type": "parametric",
        "x": "cosh(t)",
        "y": "sinh(t)",
        "symmetry": "reflection",
        "description": "Hyperbola"
    },

    # Involutes and evolutes
    "involute_circle": {
        "type": "parametric",
        "x": "cos(t) + t*sin(t)",
        "y": "sin(t) - t*cos(t)",
        "symmetry": "none",
        "description": "Involute of circle"
    },

    # Cassini ovals
    "cassini_oval": {
        "type": "implicit",
        "f": "(x**2 + y**2)**2 - 2*a**2*(x**2 - y**2) - (a**4 - b**4)",
        "symmetry": "point",
        "description": "Cassini oval"
    },

    # Superellipse
    "superellipse": {
        "type": "implicit",
        "f": "abs(x/a)**n + abs(y/b)**n - 1",
        "symmetry": "point",
        "description": "Superellipse (Lamé curve)"
    },

    # Butterfly curve
    "butterfly": {
        "type": "polar",
        "r": "exp(sin(theta)) - 2*cos(4*theta) + sin((2*theta - pi)/24)**5",
        "symmetry": "reflection",
        "description": "Butterfly curve"
    }
}


def compute_curve_signature(points: List[Tuple[float, float]]) -> str:
    """Compute a signature hash for a curve based on its shape.

    This allows comparing curves for similarity regardless of scale/position.
    """
    if len(points) < 10:
        return ""

    pts = np.array(points)

    # Normalize: center and scale
    center = pts.mean(axis=0)
    pts_centered = pts - center
    scale = np.max(np.abs(pts_centered))
    if scale > 0:
        pts_normalized = pts_centered / scale
    else:
        pts_normalized = pts_centered

    # Compute shape features
    # 1. Curvature at sampled points
    curvatures = []
    for i in range(1, len(pts_normalized) - 1):
        p0, p1, p2 = pts_normalized[i-1], pts_normalized[i], pts_normalized[i+1]
        # Menger curvature
        area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        d01 = np.linalg.norm(p1 - p0)
        d12 = np.linalg.norm(p2 - p1)
        d02 = np.linalg.norm(p2 - p0)
        if d01 * d12 * d02 > 1e-10:
            k = 4 * area / (d01 * d12 * d02)
        else:
            k = 0
        curvatures.append(k)

    # 2. Quantize curvatures
    curvatures = np.array(curvatures)
    if len(curvatures) > 0:
        curvature_quantized = np.histogram(curvatures, bins=20, range=(-10, 10))[0]
    else:
        curvature_quantized = np.zeros(20)

    # 3. Create signature
    signature = curvature_quantized.tobytes()
    return hashlib.md5(signature).hexdigest()
