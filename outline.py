import cv2
import numpy as np
import bezier
import matplotlib.pyplot as plt


def extract_black_line_edges(image_path, dark_thresh=60):
    """Extract contours of only black (dark) lines in an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create mask for dark pixels only (lower value = darker)
    mask = cv2.inRange(gray, 0, dark_thresh)

    # Optional: clean small noise
    mask = cv2.medianBlur(mask, 3)

    # Find edges of the dark regions
    edges = cv2.Canny(mask, 50, 150)

    # Find contours around dark regions
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours, edges


def fit_bezier_segments(points, segment_length=80):
    """Fit cubic Bezier segments to the given contour points."""
    curves = []
    for i in range(0, len(points) - segment_length, segment_length):
        segment = points[i:i + segment_length]
        if len(segment) < 4:
            continue

        segment = segment.squeeze()
        if segment.ndim != 2:
            continue

        x, y = segment[:, 0], segment[:, 1]

        # Pick 4 control points evenly spaced along the contour
        indices = np.linspace(0, len(x) - 1, 4, dtype=int)
        control_points = np.asfortranarray([x[indices], y[indices]])

        curve = bezier.Curve(control_points, degree=3)
        curves.append(curve)
    return curves


def plot_black_line_bezier(image_path):
    contours, edges = extract_black_line_edges(image_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap="gray")
    plt.title("Bezier Outline (Black Lines Only)")

    for contour in contours:
        curves = fit_bezier_segments(contour)
        for curve in curves:
            s_vals = np.linspace(0.0, 1.0, 100)
            data = curve.evaluate_multi(s_vals)
            plt.plot(data[0, :], data[1, :], "r-", linewidth=1.3)
            plt.plot(curve.nodes[0, :], curve.nodes[1, :], "bo-", markersize=3, alpha=0.7)

    # plt.gca().invert_yaxis()
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    image_path = r"C:\Users\wesle\Downloads\test image.png"  # change this to your image
    plot_black_line_bezier(image_path)
