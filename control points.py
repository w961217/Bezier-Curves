"""
bezier_function_exports.py

Write symbolic (string) Bézier functions for many control-point sets into three files:
  - <prefix>_linear.txt
  - <prefix>_quadratic.txt
  - <prefix>_cubic.txt

Requires: numpy
"""

from typing import Sequence, List, Tuple
import numpy as np
from numpy.polynomial import Polynomial
from math import comb


def _to_np(p):
    return np.asarray(p, dtype=float).reshape(-1)


def _stack_ctrl(ctrl: Sequence) -> np.ndarray:
    pts = [_to_np(p) for p in ctrl]
    dims = {p.shape[0] for p in pts}
    if len(dims) != 1:
        raise ValueError("All control points must have the same dimension.")
    return np.stack(pts, axis=0)  # shape (n_points, dim)


def _bernstein_basis_poly(n: int, i: int) -> Polynomial:
    """
    Return a Polynomial object representing B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
    (Polynomial uses coefficients in ascending order: [c0, c1, c2, ...] for c0 + c1 t + ...)
    """
    # Polynomial for t
    t_poly = Polynomial([0.0, 1.0])
    one_minus_t = Polynomial([1.0, -1.0])  # 1 - t
    basis = (t_poly**i) * (one_minus_t ** (n - i)) * comb(n, i)
    return basis


def _polynomial_to_string(poly: Polynomial, var: str = "t") -> str:
    # poly.coef is ascending: coef[0] + coef[1]*t + coef[2]*t^2 + ...
    coefs = np.round(poly.coef, 10)  # round for nicer output
    terms = []
    deg = len(coefs) - 1
    for power, c in enumerate(coefs):
        if abs(c) < 1e-12:
            continue
        # format coefficient
        # put power label
        if power == 0:
            terms.append(f"{c:.10g}")
        elif power == 1:
            terms.append(f"{c:.10g}*{var}")
        else:
            terms.append(f"{c:.10g}*{var}^{power}")
    if not terms:
        return "0"
    # join highest-power first for readability
    return " + ".join(reversed(terms))


def _vector_polynomial_to_string(
    polys: List[Polynomial], var: str = "t"
) -> Tuple[str, List[str]]:
    """
    Given list of Polynomial objects (one per coordinate), return:
      - component strings [x(t), y(t), ...]
      - tuple form string "(x(t), y(t), ...)"
    """
    comp_strs = [_polynomial_to_string(p, var) for p in polys]
    tuple_str = "(" + ", ".join(comp_strs) + ")"
    return tuple_str, comp_strs


def _bernstein_form_string(ctrl_pts: np.ndarray) -> str:
    """
    ctrl_pts shape (n_points, dim) -> produce a string of the Bernstein-form vector function
    B(t) = sum_i B_{i,n}(t) * P_i
    """
    n = ctrl_pts.shape[0] - 1
    dim = ctrl_pts.shape[1]
    parts = []
    for i in range(n + 1):
        basis = f"C({n},{i}) * t^{i} * (1-t)^{n-i}"
        # include vector coefficient with numeric values
        coords = ", ".join(f"{float(x):g}" for x in ctrl_pts[i])
        parts.append(f"{basis} * ({coords})")
    return " + ".join(parts)


def _expand_bezier_polynomial(ctrl_pts: np.ndarray) -> List[Polynomial]:
    """
    For control points of degree n (len = n+1), return list of Polynomial objects,
    one per coordinate, giving expanded polynomial in ascending order (c0 + c1 t + ... + cn t^n).
    """
    n = ctrl_pts.shape[0] - 1
    dim = ctrl_pts.shape[1]
    # prepare empty polynomials for each coordinate
    coord_polys = [Polynomial([0.0]) for _ in range(dim)]
    for i in range(n + 1):
        B_i = _bernstein_basis_poly(n, i)  # Polynomial object
        # For each coordinate, add ctrl_pts[i, coord] * B_i
        for d in range(dim):
            coord_polys[d] = coord_polys[d] + (ctrl_pts[i, d] * B_i)
    # Round tiny numerical noise
    for d in range(dim):
        coord_polys[d].coef = np.round(coord_polys[d].coef, 12)
    return coord_polys


def _format_control_points(ctrl_pts: np.ndarray) -> str:
    pts_strs = []
    for p in ctrl_pts:
        pts_strs.append("(" + ", ".join(f"{float(x):g}" for x in p) + ")")
    return "[" + ", ".join(pts_strs) + "]"


def export_functions_by_degree(
    list_of_ctrl_sets: Sequence[Sequence[Sequence[float]]],
    prefix: str = "bezier_functions",
):
    """
    Process many control-point sets and write functions to three files by degree:
      prefix_linear.txt
      prefix_quadratic.txt
      prefix_cubic.txt

    Only accepts control sets with 2, 3 or 4 control points.
    """
    linear_file = f"{prefix}_linear.txt"
    quad_file = f"{prefix}_quadratic.txt"
    cubic_file = f"{prefix}_cubic.txt"

    # Clear files
    open(linear_file, "w").close()
    open(quad_file, "w").close()
    open(cubic_file, "w").close()

    for idx, ctrl in enumerate(list_of_ctrl_sets, start=1):
        pts = _stack_ctrl(ctrl)
        n = pts.shape[0]
        if n not in (2, 3, 4):
            raise ValueError(
                f"Control set #{idx} has {n} points — only 2, 3 or 4 allowed."
            )

        dim = pts.shape[1]
        ctrl_header = f"Curve #{idx} | degree={n-1} | dim={dim}\nControl points: {_format_control_points(pts)}\n"

        # Bernstein form (string)
        bern_str = _bernstein_form_string(pts)

        # Expanded polynomial form per coordinate
        coord_polys = _expand_bezier_polynomial(pts)
        vector_tuple_str, comp_strs = _vector_polynomial_to_string(coord_polys, var="t")

        # Compose block
        block_lines = []
        block_lines.append("=== START_CURVE ===")
        block_lines.append(ctrl_header)
        block_lines.append("Bernstein form:")
        block_lines.append(bern_str)
        block_lines.append("")  # blank
        block_lines.append("Expanded polynomial form (per coordinate):")
        for dim_i, comp in enumerate(comp_strs):
            block_lines.append(f"  coord_{dim_i}(t) = {comp}")
        block_lines.append("")
        block_lines.append(f"Vector function: B(t) = {vector_tuple_str}")
        block_lines.append("=== END_CURVE ===")
        block_lines.append("\n")

        # Choose file by degree
        if n == 2:
            fname = linear_file
        elif n == 3:
            fname = quad_file
        else:
            fname = cubic_file

        with open(fname, "a") as f:
            f.write("\n".join(block_lines))

    print("Wrote functions to:")
    print(f"  Linear  -> {linear_file}")
    print(f"  Quad    -> {quad_file}")
    print(f"  Cubic   -> {cubic_file}")


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    example_sets = [
        [(160, 0), (96, 52)],
        [(96, 52), (52, 169)],
        [(52, 169), (95, 86)],
        [(95, 86), (123, 259)],
        [(123, 259), (178, 80)],
        [(178, 80), (216, 290)],
        [(216, 290), (170, 240)],
        [(170, 240), (254, 259)],
        [(254, 259), (242, 187)],
        [(242, 187), (265, 226)],
        [(265, 226), (310, 145)],
        [(310, 145), (408, 728)],
        [(822, 728), (1290, 320)],
        
        [(160, 0), (112, 15), (96, 52)],
        [(96, 52), (18, 115), (52, 169)],
        [(52, 169), (60, 107), (95, 86)],
        [(95, 86), (45, 210), (123, 259)],
        [(123, 259), (35, 123), (178, 80)],
        [(178, 80), (76, 207), (216, 290)],
        [(216, 290), (183, 266), (170, 240)],
        [(170, 240), (226, 293), (254, 259)],
        [(254, 259), (224, 234), (242, 187)],
        [(242, 187), (234, 219), (265, 226)],
        [(265, 226), (247, 145), (310, 145)],
        [(310, 145), (240, 540), (408, 728)],
        [(822, 728), (1075, 266), (1290, 320)],
        
        [(160, 0), (137, 5), (98, 32), (96, 52)],
        [(96, 52), (42, 90), (28, 154), (52, 169)],
        [(52, 169), (51, 138), (78, 96), (95, 86)],
        [(95, 86), (53, 143), (87, 262), (123, 259)],
        [(123, 259), (80, 124), (58, 107), (178, 80)],
        [(178, 80), (112, 111), (123, 267), (216, 290)],
        [(216, 290), (216, 190), (202, 283), (170, 240)],
        [(170, 240), (207, 279), (244,277), (254, 259)],
        [(254, 259), (232,243), (232,194), (242, 187)],
        [(242, 187), (233,207), (253,227), (265, 226)],
        [(265, 226), (248,195), (261,144), (310, 145)],
        [(310, 145), (397,330), (225,547), (408, 728)],
        [(822, 728), (939,657), (1006,273), (1290,20)],
    ]

    export_functions_by_degree(example_sets, prefix="bezier_output")
