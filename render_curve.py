"""Render a novel curve as artistic PNG image."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sympy import symbols, sympify, lambdify

def render_curve_art(
    x_expr: str,
    y_expr: str,
    output_path: str = "outputs/novel_curve.png",
    t_range: tuple = (0, 8 * np.pi),
    num_points: int = 5000,
    figsize: tuple = (12, 12),
    bg_color: str = "#0a0a0a",
    line_color: str = "plasma",
    title: str = None
):
    """Render a parametric curve as artistic PNG.

    Args:
        x_expr: Expression for x(t)
        y_expr: Expression for y(t)
        output_path: Path to save PNG
        t_range: Range of parameter t
        num_points: Number of points to sample
        figsize: Figure size in inches
        bg_color: Background color
        line_color: Colormap name or color
        title: Optional title for the plot
    """
    # Parse expressions
    t = symbols('t')
    x_sym = sympify(x_expr)
    y_sym = sympify(y_expr)

    # Create numerical functions
    x_func = lambdify(t, x_sym, modules=['numpy'])
    y_func = lambdify(t, y_sym, modules=['numpy'])

    # Generate points
    t_vals = np.linspace(t_range[0], t_range[1], num_points)
    x_vals = x_func(t_vals)
    y_vals = y_func(t_vals)

    # Filter invalid values
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]
    t_vals = t_vals[valid]

    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Create line segments for gradient coloring
    points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by parameter value (creates gradient effect)
    norm = plt.Normalize(t_vals.min(), t_vals.max())
    lc = LineCollection(segments, cmap=line_color, norm=norm,
                        linewidth=0.8, alpha=0.85)
    lc.set_array(t_vals[:-1])
    ax.add_collection(lc)

    # Auto-scale with padding
    margin = 0.1
    x_range = x_vals.max() - x_vals.min()
    y_range = y_vals.max() - y_vals.min()
    ax.set_xlim(x_vals.min() - margin * x_range, x_vals.max() + margin * x_range)
    ax.set_ylim(y_vals.min() - margin * y_range, y_vals.max() + margin * y_range)

    # Remove axes for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add title if provided
    if title:
        ax.set_title(title, color='white', fontsize=14, pad=20)

    # Equal aspect ratio
    ax.set_aspect('equal')

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=bg_color,
                bbox_inches='tight', pad_inches=0.5)
    plt.close()

    print(f"Curve rendered and saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # The novel curve to render
    x_expr = "(1 + sin(t/6)/3) * exp(-t/20) * sin(t**1.1) * cos(sqrt(7)*t) + (t**1.6) / 28"
    y_expr = "exp(-t/19) * cos(t) * sin(sqrt(2)*t) * (1 + tanh(t/8)) - t**1.2/18"

    # Render with artistic settings
    render_curve_art(
        x_expr=x_expr,
        y_expr=y_expr,
        output_path="outputs/novel_curve_art.png",
        t_range=(0, 12 * np.pi),  # Extended range for full curve
        num_points=10000,
        figsize=(14, 14),
        bg_color="#050510",  # Deep blue-black
        line_color="magma",  # Fire-like gradient
        title="Novel Curve Discovery\nNovelty: 1.0 | Beauty: 0.55 | Score: 0.737"
    )

    # Also create a version with different colormap
    render_curve_art(
        x_expr=x_expr,
        y_expr=y_expr,
        output_path="outputs/novel_curve_art_plasma.png",
        t_range=(0, 12 * np.pi),
        num_points=10000,
        figsize=(14, 14),
        bg_color="#0a0a0a",
        line_color="plasma",
        title="Novel Curve Discovery\nNovelty: 1.0 | Beauty: 0.55 | Score: 0.737"
    )

    print("\nRendered 2 versions of the curve!")
