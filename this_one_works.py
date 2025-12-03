import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx

def get_ordered_skeleton_coords(image_path, dark_thresh=128, output_file="coordinates.txt"):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    height, width = img.shape

    # Threshold to binary
    _, binary = cv2.threshold(img, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    binary_bool = binary > 0

    # Skeletonize
    skeleton = skeletonize(binary_bool)

    # Get skeleton pixel coordinates
    coords = np.column_stack(np.where(skeleton))  # (y, x)

    # Build graph of connected pixels
    G = nx.Graph()
    for y, x in coords:
        G.add_node((x, y))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in G:
                    G.add_edge((x, y), neighbor)

    # Find connected components and order coordinates
    ordered_coords = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        # Find endpoints (degree 1)
        endpoints = [n for n, d in subgraph.degree() if d == 1]
        if endpoints:
            path = nx.shortest_path(subgraph, source=endpoints[0], target=endpoints[-1])
            ordered_coords.extend(path)
        else:
            ordered_coords.extend(subgraph.nodes)

    # Flip y-coordinate to match Cartesian style
    ordered_coords_flipped = [(x, height - 1 - y) for x, y in ordered_coords]

    # Save coordinates
    np.savetxt(output_file, ordered_coords_flipped, fmt="%d", delimiter=",", header="x,y", comments="")
    print(f"Saved {len(ordered_coords_flipped)} ordered pixels to {output_file}")
    return np.array(ordered_coords_flipped)

if __name__ == "__main__":
    image_path = r"C:\Users\wesle\Downloads\test image6.png"
    coords = get_ordered_skeleton_coords(image_path)
    print(coords[:10])
