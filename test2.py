import cv2
import numpy as np
from skimage.morphology import skeletonize
import bezier
import matplotlib.pyplot as plt

def fit_cubic_bezier(points):
    """Fit cubic Bézier curves to a sequence of points by splitting into segments of 4 points each."""
    curves = []
    n = len(points)
    if n < 4:
        return curves

    # split points into overlapping segments of 4 (step 3 for continuity)
    for i in range(0, n - 3, 3):
        segment = points[i:i+4].T  # 2x4 array: [[x1,x2,x3,x4],[y1,y2,y3,y4]]
        if segment.shape[1] == 4:  # make sure it's exactly 4 points
            curve = bezier.Curve(segment, degree=3)
            curves.append(curve)
    return curves

def export_bezier_functions(curves, output_file="bezier_functions.txt"):
    with open(output_file, "w") as f:
        for i, curve in enumerate(curves):
            nodes = curve.nodes.T.tolist()
            f.write(f"# Curve {i}\n")
            f.write(f"def curve_{i}(t):\n")
            f.write(f"    nodes = {nodes}\n")
            f.write("    x = ((1-t)**3)*nodes[0][0] + 3*((1-t)**2)*t*nodes[1][0] + "
                    "3*(1-t)*(t**2)*nodes[2][0] + (t**3)*nodes[3][0]\n")
            f.write("    y = ((1-t)**3)*nodes[0][1] + 3*((1-t)**2)*t*nodes[1][1] + "
                    "3*(1-t)*(t**2)*nodes[2][1] + (t**3)*nodes[3][1]\n")
            f.write("    return [x, y]\n\n")

def trace_image_outline(image_path, output_nodes_txt="bezier_curves.txt", output_functions_txt="bezier_functions.txt"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Binarize and skeletonize
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(binary).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_curves = []

    for contour in contours:
        if contour.shape[0] < 4:
            continue
        points = contour[:, 0, :]  # convert (N,1,2) -> (N,2)
        curves = fit_cubic_bezier(points)
        all_curves.extend(curves)

    if not all_curves:
        print("No curves detected. Check image and preprocessing.")
        return []

    # Save Bézier nodes
    with open(output_nodes_txt, "w") as f:
        for i, curve in enumerate(all_curves):
            f.write(f"Curve {i} nodes:\n{curve.nodes.T.tolist()}\n\n")

    # Export functions
    export_bezier_functions(all_curves, output_functions_txt)

    # Optional: plot
    plt.imshow(skeleton, cmap="gray")
    for curve in all_curves:
        s_vals = np.linspace(0, 1, 50)
        pts = curve.evaluate_multi(s_vals)
        plt.plot(pts[0, :], pts[1, :], 'r')
    plt.show()

    return all_curves

# Example usage
bezier_curves = trace_image_outline(r"C:\Users\wesle\Downloads\test image4.png")
