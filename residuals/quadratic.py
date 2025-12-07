import numpy as np
from sklearn.metrics import mean_squared_error
import os

# ----------------------------------------------------
# Quadratic Bézier function
# ----------------------------------------------------
def quadratic_bezier(t, P0, P1, P2):
    """Evaluate quadratic Bézier at parameter t."""
    return ((1 - t)**2) * P0 + 2 * (1 - t) * t * P1 + (t**2) * P2

# ----------------------------------------------------
# Reference files and control points (quadratic: 3 points each)
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
    [np.array([160, 0]), np.array([112, 15]), np.array([96, 52])],
    [np.array([96, 52]), np.array([18, 115]), np.array([52, 169])], 
    [np.array([52, 169]), np.array([60, 107]), np.array([95, 86])], 
    [np.array([95, 86]), np.array([45, 210]), np.array([123, 259])], 
    [np.array([123, 259]), np.array([35, 123]), np.array([178, 80])], 
    [np.array([178, 80]), np.array([76, 207]), np.array([216, 290])], 
    [np.array([216, 290]), np.array([183, 266]), np.array([170, 240])], 
    [np.array([170, 240]), np.array([226, 293]), np.array([254, 259])], 
    [np.array([254, 259]), np.array([224, 234]), np.array([242, 187])], 
    [np.array([242, 187]), np.array([234, 219]), np.array([265, 226])], 
    [np.array([265, 226]), np.array([247, 145]), np.array([310, 145])], 
    [np.array([310, 145]), np.array([240, 540]), np.array([408, 728])], 
    [np.array([822, 728]), np.array([1075, 266]), np.array([1290, 320])], 
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

    P0, P1, P2 = control_points

    # Sample Bézier curve
    bezier_points = np.array([quadratic_bezier(t, P0, P1, P2) for t in t_vals])

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
