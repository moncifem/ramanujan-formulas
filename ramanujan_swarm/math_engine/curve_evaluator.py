"""Curve evaluator for rendering and analyzing novel curves."""

import numpy as np
from typing import List, Tuple, Optional
from sympy import sympify, symbols, pi, E, sqrt, cos, sin, tan, exp, log, lambdify
from sympy import oo, zoo, nan
from ramanujan_swarm.math_engine.curve_types import CurveExpression, compute_curve_signature


class CurveEvaluator:
    """Evaluates curve equations and computes geometric properties."""

    def __init__(self, num_points: int = 1000):
        """Initialize evaluator.

        Args:
            num_points: Number of points to sample for rendering
        """
        self.num_points = num_points

    def evaluate_parametric(
        self, x_expr: str, y_expr: str, t_range: Tuple[float, float] = (0, 2 * np.pi)
    ) -> Optional[List[Tuple[float, float]]]:
        """Evaluate a parametric curve x(t), y(t).

        Args:
            x_expr: Expression for x(t)
            y_expr: Expression for y(t)
            t_range: Range of parameter t

        Returns:
            List of (x, y) points or None if evaluation fails
        """
        try:
            t = symbols('t')
            x_sym = sympify(x_expr)
            y_sym = sympify(y_expr)

            # Create numerical functions
            x_func = lambdify(t, x_sym, modules=['numpy'])
            y_func = lambdify(t, y_sym, modules=['numpy'])

            # Generate t values
            t_vals = np.linspace(t_range[0], t_range[1], self.num_points)

            # Evaluate
            x_vals = x_func(t_vals)
            y_vals = y_func(t_vals)

            # Convert to list of tuples, filtering invalid values
            points = []
            for x, y in zip(x_vals, y_vals):
                if np.isfinite(x) and np.isfinite(y):
                    points.append((float(x), float(y)))

            return points if len(points) > 10 else None

        except Exception as e:
            return None

    def evaluate_polar(
        self, r_expr: str, theta_range: Tuple[float, float] = (0, 2 * np.pi)
    ) -> Optional[List[Tuple[float, float]]]:
        """Evaluate a polar curve r(θ).

        Args:
            r_expr: Expression for r(θ)
            theta_range: Range of angle θ

        Returns:
            List of (x, y) points or None if evaluation fails
        """
        try:
            theta = symbols('theta')
            r_sym = sympify(r_expr)

            # Create numerical function
            r_func = lambdify(theta, r_sym, modules=['numpy'])

            # Generate theta values
            theta_vals = np.linspace(theta_range[0], theta_range[1], self.num_points)

            # Evaluate r
            r_vals = r_func(theta_vals)

            # Convert to Cartesian
            points = []
            for th, r in zip(theta_vals, r_vals):
                if np.isfinite(r) and r >= 0:  # Only positive r values
                    x = r * np.cos(th)
                    y = r * np.sin(th)
                    if np.isfinite(x) and np.isfinite(y):
                        points.append((float(x), float(y)))

            return points if len(points) > 10 else None

        except Exception as e:
            return None

    def evaluate_implicit(
        self, f_expr: str, x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5), grid_size: int = 200
    ) -> Optional[List[Tuple[float, float]]]:
        """Evaluate an implicit curve f(x, y) = 0 using marching squares.

        Args:
            f_expr: Expression for f(x, y)
            x_range: Range of x values
            y_range: Range of y values
            grid_size: Resolution of grid

        Returns:
            List of (x, y) points on the curve or None if evaluation fails
        """
        try:
            x, y = symbols('x y')
            f_sym = sympify(f_expr)

            # Create numerical function
            f_func = lambdify((x, y), f_sym, modules=['numpy'])

            # Create grid
            x_vals = np.linspace(x_range[0], x_range[1], grid_size)
            y_vals = np.linspace(y_range[0], y_range[1], grid_size)
            X, Y = np.meshgrid(x_vals, y_vals)

            # Evaluate function on grid
            Z = f_func(X, Y)

            # Find zero crossings (simple approach)
            points = []
            for i in range(grid_size - 1):
                for j in range(grid_size - 1):
                    # Check if zero crossing in this cell
                    vals = [Z[i, j], Z[i+1, j], Z[i, j+1], Z[i+1, j+1]]
                    if not all(np.isfinite(vals)):
                        continue
                    if min(vals) <= 0 <= max(vals):
                        # Approximate crossing point
                        cx = (x_vals[j] + x_vals[j+1]) / 2
                        cy = (y_vals[i] + y_vals[i+1]) / 2
                        points.append((float(cx), float(cy)))

            return points if len(points) > 10 else None

        except Exception as e:
            return None

    def compute_properties(self, curve: CurveExpression) -> CurveExpression:
        """Compute geometric properties of a curve.

        Args:
            curve: CurveExpression with points already computed

        Returns:
            Updated CurveExpression with computed properties
        """
        if not curve.points or len(curve.points) < 10:
            return curve

        pts = np.array(curve.points)

        # Bounding box
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        curve.bounding_box = (float(xmin), float(xmax), float(ymin), float(ymax))

        # Check if closed
        dist_start_end = np.linalg.norm(pts[0] - pts[-1])
        curve.is_closed = dist_start_end < 0.1 * max(xmax - xmin, ymax - ymin)

        # Arc length
        diffs = np.diff(pts, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        curve.arc_length = float(np.sum(segment_lengths))

        # Enclosed area (if closed, using shoelace formula)
        if curve.is_closed:
            x = pts[:, 0]
            y = pts[:, 1]
            curve.enclosed_area = float(0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))

        # Curvature analysis
        curvatures = self._compute_curvatures(pts)
        if len(curvatures) > 0:
            curve.curvature_variance = float(np.var(curvatures))

        # Detect symmetry
        curve.symmetry_type, curve.symmetry_order = self._detect_symmetry(pts)

        # Count self-intersections
        curve.self_intersections = self._count_self_intersections(pts)

        # Count cusps (high curvature points)
        curve.num_cusps = self._count_cusps(curvatures)

        # Compute shape hash
        curve.shape_hash = compute_curve_signature(curve.points)

        return curve

    def _compute_curvatures(self, pts: np.ndarray) -> np.ndarray:
        """Compute curvature at each point using Menger curvature."""
        curvatures = []
        for i in range(1, len(pts) - 1):
            p0, p1, p2 = pts[i-1], pts[i], pts[i+1]
            # Triangle area
            area = 0.5 * abs(
                (p1[0] - p0[0]) * (p2[1] - p0[1]) -
                (p2[0] - p0[0]) * (p1[1] - p0[1])
            )
            # Side lengths
            d01 = np.linalg.norm(p1 - p0)
            d12 = np.linalg.norm(p2 - p1)
            d02 = np.linalg.norm(p2 - p0)

            if d01 * d12 * d02 > 1e-10:
                k = 4 * area / (d01 * d12 * d02)
            else:
                k = 0
            curvatures.append(k)

        return np.array(curvatures)

    def _detect_symmetry(self, pts: np.ndarray) -> Tuple[str, int]:
        """Detect symmetry type and order of a curve."""
        center = pts.mean(axis=0)
        pts_centered = pts - center

        # Check rotational symmetry for various orders
        for order in [8, 6, 5, 4, 3, 2]:
            angle = 2 * np.pi / order
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            rotated = pts_centered @ rotation.T

            # Check if rotated points match original
            if self._points_match(pts_centered, rotated):
                return "rotational", order

        # Check reflection symmetry (x-axis and y-axis)
        reflected_x = pts_centered * np.array([1, -1])
        reflected_y = pts_centered * np.array([-1, 1])

        if self._points_match(pts_centered, reflected_x) or self._points_match(pts_centered, reflected_y):
            return "reflection", 1

        # Check point symmetry (180 degree rotation)
        rotated_180 = -pts_centered
        if self._points_match(pts_centered, rotated_180):
            return "point", 2

        return "none", 1

    def _points_match(self, pts1: np.ndarray, pts2: np.ndarray, tolerance: float = 0.1) -> bool:
        """Check if two point sets represent the same shape."""
        # Simple check: for each point in pts2, find nearest in pts1
        if len(pts1) != len(pts2):
            return False

        # Sample points for efficiency
        sample_size = min(100, len(pts1))
        indices = np.linspace(0, len(pts1) - 1, sample_size, dtype=int)

        for i in indices:
            p = pts2[i]
            dists = np.linalg.norm(pts1 - p, axis=1)
            if dists.min() > tolerance:
                return False

        return True

    def _count_self_intersections(self, pts: np.ndarray) -> int:
        """Count approximate number of self-intersections."""
        # Simple grid-based approach
        if len(pts) < 20:
            return 0

        # Discretize to grid
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()

        if xmax - xmin < 1e-6 or ymax - ymin < 1e-6:
            return 0

        grid_size = 50
        grid = {}

        for i, p in enumerate(pts):
            gx = int((p[0] - xmin) / (xmax - xmin + 1e-10) * grid_size)
            gy = int((p[1] - ymin) / (ymax - ymin + 1e-10) * grid_size)
            key = (gx, gy)

            if key in grid:
                # Check if this is from a distant part of the curve
                prev_idx = grid[key]
                if abs(i - prev_idx) > len(pts) / 10:
                    grid[key] = i  # Update with new index
            else:
                grid[key] = i

        # Count cells with multiple visits from distant parts
        # This is a rough approximation
        return max(0, len([k for k in grid if isinstance(grid[k], int)]) - len(pts) // 20)

    def _count_cusps(self, curvatures: np.ndarray) -> int:
        """Count number of cusps (sharp points) in curve."""
        if len(curvatures) < 3:
            return 0

        # Find peaks in curvature
        threshold = np.percentile(np.abs(curvatures), 95) if len(curvatures) > 10 else 10
        cusps = 0

        for i in range(1, len(curvatures) - 1):
            if abs(curvatures[i]) > threshold:
                if abs(curvatures[i]) > abs(curvatures[i-1]) and abs(curvatures[i]) > abs(curvatures[i+1]):
                    cusps += 1

        return cusps
