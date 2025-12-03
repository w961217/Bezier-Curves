import cv2
import numpy as np
import os
from collections import deque

def get_bezier_parameters(p0, p1, p2, p3):
    """
    Returns the coefficients for the cubic Bezier equation:
    B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
    Expanded:
    B(t) = At^3 + Bt^2 + Ct + D
    """
    D = p0
    C = 3 * (p1 - p0)
    B = 3 * (p2 - 2 * p1 + p0)
    A = p3 - 3 * p2 + 3 * p1 - p0
    return A, B, C, D

def fit_bezier_to_contour(points, smoothing_factor=0.3):
    """
    Fits a sequence of cubic Bezier curves to a set of polyline points.
    Uses tangent estimation (Catmull-Rom style control point calculation).
    """
    n = len(points)
    if n < 2:
        return []

    bezier_segments = []
    
    for i in range(n - 1):
        p0 = points[i] # Current Point
        p3 = points[i+1] # Next Point
        
        # --- Calculate Tangent at P0 ---
        if i == 0:
            prev_p = points[i] # Start of line
        else:
            prev_p = points[i-1]
            
        t1 = p3 - prev_p
        
        # --- Calculate Tangent at P3 ---
        if i == n - 2:
            next_p = points[i+1] # End of line
        else:
            next_p = points[i+2]

        t2 = next_p - p0
        
        dist = np.linalg.norm(p3 - p0)
        
        # Normalize tangents
        norm1 = np.linalg.norm(t1)
        if norm1 == 0: norm1 = 1
        t1 = t1 / norm1
        
        norm2 = np.linalg.norm(t2)
        if norm2 == 0: norm2 = 1
        t2 = t2 / norm2
        
        # Calculate control points
        p1 = p0 + t1 * (dist * smoothing_factor)
        p2 = p3 - t2 * (dist * smoothing_factor)
        
        bezier_segments.append((p0, p1, p2, p3))
        
    return bezier_segments

def skeletonize_image(img_binary):
    """
    Reduces binary image features to 1-pixel width skeletons.
    """
    skel = np.zeros(img_binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    temp_img = img_binary.copy()
    
    done = False
    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        
        if cv2.countNonZero(temp_img) == 0:
            done = True
    return skel

def extract_paths_from_skeleton(skel_img):
    """
    Walks the connected components of the skeleton to retrieve ordered points.
    This replaces cv2.findContours to avoid loop/gap issues.
    """
    # Get all white pixel coordinates
    # np.where returns (row_indices, col_indices), so we zip them as (y, x)
    ys, xs = np.where(skel_img > 0)
    pixels = set(zip(xs, ys)) # Store as (x, y) for consistency
    
    paths = []
    
    while pixels:
        # Start a new stroke from an arbitrary pixel
        start_pixel = next(iter(pixels))
        pixels.remove(start_pixel)
        
        # Use a deque to build the line in both directions
        # structure: [ ... left_growth ... start ... right_growth ... ]
        stroke = deque([start_pixel])
        
        # Directions: 8-neighborhood
        neighbors_offset = [(-1, -1), (0, -1), (1, -1),
                            (-1, 0),           (1, 0),
                            (-1, 1),  (0, 1),  (1, 1)]
                            
        # 1. Grow Right/Forward
        current = start_pixel
        while True:
            found_next = None
            # Look for a neighbor in the set of remaining pixels
            for dx, dy in neighbors_offset:
                nb = (current[0] + dx, current[1] + dy)
                if nb in pixels:
                    found_next = nb
                    break
            
            if found_next:
                stroke.append(found_next)
                pixels.remove(found_next)
                current = found_next
            else:
                break # Dead end
        
        # 2. Grow Left/Backward (from the original start pixel)
        current = start_pixel
        while True:
            found_prev = None
            for dx, dy in neighbors_offset:
                nb = (current[0] + dx, current[1] + dy)
                if nb in pixels:
                    found_prev = nb
                    break
            
            if found_prev:
                stroke.appendleft(found_prev)
                pixels.remove(found_prev)
                current = found_prev
            else:
                break # Dead end
                
        # Only keep strokes that are significant
        if len(stroke) > 10:
            paths.append(list(stroke))
            
    return paths

def trace_image(image_path, output_txt_path, output_img_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    vis_img = np.ones_like(img) * 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Preprocess
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # DILATE first to close small gaps in the raw drawing
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Smooth to remove noise
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    _, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)
    
    print("Skeletonizing image...")
    skeleton = skeletonize_image(thresh)

    # 3. Extract Ordered Paths (Walking the skeleton)
    print("Tracing paths...")
    raw_paths = extract_paths_from_skeleton(skeleton)
    print(f"Found {len(raw_paths)} connected strokes.")

    with open(output_txt_path, 'w') as f:
        f.write(f"Bezier Curve Trace for {image_path}\n")
        f.write("="*50 + "\n\n")

        for cnt_idx, path_points in enumerate(raw_paths):
            
            # Convert to numpy float array
            raw_points = np.array(path_points, dtype=np.float32)

            # Simplify slightly to reduce jitter from the pixel grid
            # Epsilon 2.0 keeps it very faithful but smooths 1px zig-zags
            epsilon = 2.0 
            approx = cv2.approxPolyDP(raw_points, epsilon, False)
            points = np.array([p[0] for p in approx], dtype=np.float32)

            if len(points) < 2:
                continue

            # Fit Bezier Segments
            segments = fit_bezier_to_contour(points, smoothing_factor=0.3)
            
            f.write(f"--- Stroke {cnt_idx+1} ({len(segments)} segments) ---\n")
            
            # Vis: Red dots for control points (anchors)
            for p in points:
                cv2.circle(vis_img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            for seg_idx, (p0, p1, p2, p3) in enumerate(segments):
                Ax, Bx, Cx, Dx = get_bezier_parameters(p0[0], p1[0], p2[0], p3[0])
                Ay, By, Cy, Dy = get_bezier_parameters(p0[1], p1[1], p2[1], p3[1])
                
                f.write(f"  Segment {seg_idx+1}:\n")
                f.write(f"    P0({p0[0]:.1f},{p0[1]:.1f}) -> P3({p3[0]:.1f},{p3[1]:.1f})\n")
                f.write(f"    x(t) = {Ax:.3f}t^3 + {Bx:.3f}t^2 + {Cx:.3f}t + {Dx:.3f}\n")
                f.write(f"    y(t) = {Ay:.3f}t^3 + {By:.3f}t^2 + {Cy:.3f}t + {Dy:.3f}\n\n")
                
                # Draw Curve
                steps = 20
                t_vals = np.linspace(0, 1, steps)
                curve_points = []
                for t in t_vals:
                    xt = Ax*(t**3) + Bx*(t**2) + Cx*t + Dx
                    yt = Ay*(t**3) + By*(t**2) + Cy*t + Dy
                    curve_points.append((int(xt), int(yt)))
                
                for i in range(len(curve_points) - 1):
                    cv2.line(vis_img, curve_points[i], curve_points[i+1], (255, 0, 0), 2)

    cv2.imwrite(output_img_path, vis_img)
    print(f"Trace complete.")
    print(f"Functions saved to: {output_txt_path}")
    print(f"Visualization saved to: {output_img_path}")

if __name__ == "__main__":
    input_file = r"C:\Users\wesle\Downloads\test image3.png" 
    trace_image(input_file, "bezier_curves.txt", "bezier_trace_visualization.png") 
    