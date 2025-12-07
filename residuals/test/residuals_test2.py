import numpy as np
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------
# Load reference data
# ----------------------------------------------------
ref_points = np.loadtxt(
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates_section1.txt",
    delimiter=",",
    skiprows=1
)   # shape (N, 2)

# ----------------------------------------------------
# Cubic Bézier function
# ----------------------------------------------------
def cubic_bezier(t, P0, P1, P2, P3):
    """Evaluate cubic Bézier at parameter t."""
    return ((1 - t)**3) * P0 \
         + 3 * ((1 - t)**2) * t * P1 \
         + 3 * (1 - t) * (t**2) * P2 \
         + (t**3) * P3

# ----------------------------------------------------
# Control points
# ----------------------------------------------------
P0 = np.array([160, 0])
P1 = np.array([137, 5])
P2 = np.array([98, 32])
P3 = np.array([96, 52])

# ----------------------------------------------------
# Sample Bézier curve
# ----------------------------------------------------
t_vals = np.linspace(0, 1, 200)
bezier_points = np.array([cubic_bezier(t, P0, P1, P2, P3) for t in t_vals])

# Sort curve points by x for valid interpolation
order = np.argsort(bezier_points[:, 0])
bx = bezier_points[order, 0]
by = bezier_points[order, 1]

# ----------------------------------------------------
# Select reference points in Bézier x-range
# ----------------------------------------------------
x_min, x_max = bx.min(), bx.max()
mask = (ref_points[:, 0] >= x_min) & (ref_points[:, 0] <= x_max)
section_ref = ref_points[mask]

# ----------------------------------------------------
# Interpolate Bézier y at reference x positions
# ----------------------------------------------------
bezier_y_interp = np.interp(section_ref[:, 0], bx, by)

# ----------------------------------------------------
# Compute residuals
# ----------------------------------------------------
residuals = section_ref[:, 1] - bezier_y_interp
rmse = np.sqrt(mean_squared_error(section_ref[:, 1], bezier_y_interp))

# ----------------------------------------------------
# Output
# ----------------------------------------------------
print("Residuals (y_ref - y_bezier):")
print(residuals)
print("RMSE:", rmse)
