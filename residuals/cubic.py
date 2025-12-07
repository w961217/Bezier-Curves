import numpy as np
from sklearn.metrics import mean_squared_error
import os

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
# Reference files and control points
# Each reference file corresponds to one set of control points
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
    # P0, P1, P2, P3 for first reference file
    [np.array([160, 0]), np.array([137, 5]), np.array([98, 32]), np.array([96, 52])],
    # P0, P1, P2, P3 for second reference file
    [np.array([96, 52]), np.array([42, 90]), np.array([28, 154]), np.array([52, 169])], 
    [np.array([52, 169]), np.array([51, 138]), np.array([78, 96]), np.array([95, 86])], 
    [np.array([95, 86]), np.array([53, 143]), np.array([87, 262]), np.array([123, 259])], 
    [np.array([123, 259]), np.array([80, 124]), np.array([58, 107]), np.array([178, 80])], 
    [np.array([178, 80]), np.array([112, 111]), np.array([123, 267]), np.array([216, 290])], 
    [np.array([216, 290]), np.array([216, 190]), np.array([202, 283]), np.array([170, 240])], 
    [np.array([170, 240]), np.array([207, 279]), np.array([244, 277]), np.array([254, 259])], 
    [np.array([254, 259]), np.array([232, 243]), np.array([232, 194]), np.array([242, 187])], 
    [np.array([242, 187]), np.array([233, 207]), np.array([253, 227]), np.array([265, 226])], 
    [np.array([265, 226]), np.array([248, 195]), np.array([261, 144]), np.array([310, 145])], 
    [np.array([310, 145]), np.array([397, 330]), np.array([225, 547]), np.array([408, 728])], 
    [np.array([822, 728]), np.array([939, 657]), np.array([1006, 273]), np.array([1290, 320])], 
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

    P0, P1, P2, P3 = control_points

    # Sample Bézier curve
    bezier_points = np.array([cubic_bezier(t, P0, P1, P2, P3) for t in t_vals])

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
