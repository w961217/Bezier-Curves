import numpy as np
from sklearn.metrics import mean_squared_error
import os

# ----------------------------------------------------
# Linear Bézier function
# ----------------------------------------------------
def linear_bezier(t, P0, P1):
    """Evaluate linear Bézier (straight line) at parameter t."""
    return (1 - t) * P0 + t * P1

# ----------------------------------------------------
# Reference files and control points (linear: 2 points each)
# ----------------------------------------------------
reference_files = [
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section1.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section2.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section3.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section4.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section5.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section6.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section7.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section8.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section9.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section10.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section11.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section12.txt",
    r"C:\Users\wesle\Programming\Python\Bezier-Curves\coordinates\coordinates_section13.txt",
]

control_points_list = [
    [np.array([160, 0]), np.array([96, 52])],
    [np.array([96, 52]), np.array([52, 169])], 
    [np.array([52, 169]), np.array([95, 86])], 
    [np.array([95, 86]), np.array([123, 259])], 
    [np.array([123, 259]), np.array([178, 80])], 
    [np.array([178, 80]), np.array([216, 290])], 
    [np.array([216, 290]), np.array([170, 240])], 
    [np.array([170, 240]), np.array([254, 259])], 
    [np.array([254, 259]), np.array([242, 187])], 
    [np.array([242, 187]), np.array([265, 226])], 
    [np.array([265, 226]), np.array([310, 145])], 
    [np.array([310, 145]), np.array([408, 728])], 
    [np.array([822, 728]), np.array([1290, 320])], 
] 

# ----------------------------------------------------
# Sampling parameters
# ----------------------------------------------------
t_vals = np.linspace(0, 1, 5000)  # dense sampling

# ----------------------------------------------------
# Process each reference file with its control points
# ----------------------------------------------------
for ref_file, control_points in zip(reference_files, control_points_list):
    # Load reference data
    ref_points = np.loadtxt(ref_file, delimiter=",", skiprows=1)

    P0, P1 = control_points

    # Sample linear Bézier curve
    bezier_points = np.array([linear_bezier(t, P0, P1) for t in t_vals])

    # Compute residuals: minimum distance from each reference point
    residuals = []
    for ref in ref_points:
        distances = np.linalg.norm(bezier_points - ref, axis=1)
        residuals.append(np.min(distances))
    residuals = np.array(residuals)
    rmse = np.sqrt(np.mean(residuals**2))

    # Output
    print(f"\nResults for {os.path.basename(ref_file)}:")
    print("Residuals (minimum Euclidean distance):")
    print(residuals)
    print("RMSE:", rmse)
