import numpy as np
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------
# Load reference data
# ----------------------------------------------------
ref_points = np.loadtxt(
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section5.txt",
    delimiter=",",
    skiprows=1
)  # shape (N,2)

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
P0 = np.array([178, 80])
P1 = np.array([112, 111]) 
P2 = np.array([123, 267])
P3 = np.array([216, 290])

# ----------------------------------------------------
# Sample Bézier curve densely
# ----------------------------------------------------
t_vals = np.linspace(0, 1, 5000)  # dense sampling for accuracy
bezier_points = np.array([cubic_bezier(t, P0, P1, P2, P3) for t in t_vals])

# ----------------------------------------------------
# Compute residuals: minimum distance for each reference point
# ----------------------------------------------------
residuals = []
for ref in ref_points:
    # Compute distances from ref to all sampled Bézier points
    distances = np.linalg.norm(bezier_points - ref, axis=1)
    # Minimum distance
    residuals.append(np.min(distances))

residuals = np.array(residuals)
rmse = np.sqrt(np.mean(residuals**2))

# ----------------------------------------------------
# Output
# ----------------------------------------------------
print("Residuals (minimum Euclidean distance):")
print(residuals)
print("RMSE:", rmse)
