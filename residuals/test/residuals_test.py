import numpy as np
from sklearn.metrics import mean_squared_error

# --- Load reference points from TXT with header ---
# Assumes first row is 'x y'
ref_points = np.loadtxt(r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates_section1.txt", delimiter=",", skiprows=1)  # shape (N,2)

# --- Define cubic Bézier curve function ---
def cubic_bezier(t, P0, P1, P2, P3):
    return ((1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3)

# --- Bézier control points ---
P0 = np.array([160, 0])
P1 = np.array([137, 5])
P2 = np.array([98, 32])
P3 = np.array([96, 52])

# --- Evaluate Bézier curve ---
t_vals = np.linspace(0,1,200)
bezier_points = np.array([cubic_bezier(t, P0, P1, P2, P3) for t in t_vals])

# --- Determine x-range of Bézier curve ---
x_min, x_max = bezier_points[:,0].min(), bezier_points[:,0].max()

# --- Select reference points within Bézier section ---
section_mask = (ref_points[:,0] >= x_min) & (ref_points[:,0] <= x_max)
section_ref_points = ref_points[section_mask]

# --- Interpolate Bézier y-values at reference x-values ---
bezier_y_interp = np.interp(section_ref_points[:,0], bezier_points[:,0], bezier_points[:,1])

# --- Compute residuals ---
residuals = section_ref_points[:,1] - bezier_y_interp
rmse = np.sqrt(mean_squared_error(section_ref_points[:,1], bezier_y_interp))

# --- Output ---
print("Residuals (y_ref - y_bezier):", residuals)
print("RMSE:", rmse)
