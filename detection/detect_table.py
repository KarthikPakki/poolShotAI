import cv2
import numpy as np
from ultralytics import YOLO

def detect_pockets(image_path, pocket_model_path, pocket_label="Pocket"):
    img = cv2.imread(image_path)
    model = YOLO(pocket_model_path)
    results = model(img)

    pockets = []
    for result in results:
        for box, cls in zip(result.boxes, result.boxes.cls):
            if model.names[int(cls.cpu().numpy())] == pocket_label:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                pockets.append([x_center, y_center])
    return np.array(pockets, dtype=np.float32), img

def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def refine_corners_with_pockets(corners, pockets, tolerance=100):
    """Refine corner positions using nearby pockets"""
    refined_corners = corners.copy()
    
    for i, corner in enumerate(corners):
        # Find pockets near this corner
        distances = np.linalg.norm(pockets - corner, axis=1)
        nearby_pockets = pockets[distances < tolerance]
        
        if len(nearby_pockets) > 0:
            closest_pocket = nearby_pockets[np.argmin(distances[distances < tolerance])]
            refined_corners[i] = corner * 0.9 + closest_pocket * 0.1
    
    return refined_corners

def robust_table_corners(image_path, pocket_model_path, use_pockets=True):
    img = cv2.imread(image_path)
    pockets, _ = detect_pockets(image_path, pocket_model_path)
    
    if use_pockets and pockets is not None and len(pockets) >= 4:
        print(f"DEBUG: Using {len(pockets)} pockets for corner detection")
        print(f"DEBUG: Pocket positions:\n{pockets}")
        
        hull = cv2.convexHull(pockets)
        epsilon = 0.01 * cv2.arcLength(hull, True)  # Reduced epsilon for tighter fit
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4:
            hull_pts = approx[:, 0, :]
        else:
            rect = cv2.minAreaRect(pockets)
            hull_pts = cv2.boxPoints(rect)

        hull_pts = refine_corners_with_pockets(hull_pts, pockets)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        table_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(table_cnt[:, 0, :].astype(np.float32))
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            hull_pts = approx[:, 0, :]
        else:
            rect = cv2.minAreaRect(table_cnt[:, 0, :].astype(np.float32))
            hull_pts = cv2.boxPoints(rect)

    ordered = order_corners(hull_pts)
    
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])

    return ordered

def count_pockets_on_sides(pockets, corners):
    def point_to_line_dist(point, line_start, line_end):
        return np.abs(np.cross(line_end - line_start, point - line_start)) / np.linalg.norm(line_end - line_start)
    
    side_counts = []
    for i in range(4):
        pt1 = corners[i]
        pt2 = corners[(i + 1) % 4]
        count = 0
        for pocket in pockets:
            dist = point_to_line_dist(pocket, pt1, pt2)
            if dist < 50:
                count += 1
        side_counts.append(count)
    
    return side_counts

def warp_to_birdseye(img, corners, base_length=1500, pockets=None):
    needs_rotation = False
    
    if pockets is not None and len(pockets) >= 6:
        side_counts = count_pockets_on_sides(pockets, corners)
        
        top_bottom_pockets = side_counts[0] + side_counts[2]
        left_right_pockets = side_counts[1] + side_counts[3]
        
        if left_right_pockets > top_bottom_pockets:
            needs_rotation = True
    
    if needs_rotation:
        corners = np.array([corners[3], corners[0], corners[1], corners[2]], dtype="float32")

    width_top = np.linalg.norm(corners[1] - corners[0])
    width_bottom = np.linalg.norm(corners[2] - corners[3])
    height_left = np.linalg.norm(corners[3] - corners[0])
    height_right = np.linalg.norm(corners[2] - corners[1])

    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2
    
    # Calculate the actual aspect ratio of the detected table
    actual_aspect_ratio = avg_width / avg_height
    if avg_width > avg_height:
        output_width = int(base_length * actual_aspect_ratio)
        output_height = base_length
    else:
        output_width = base_length
        output_height = int(base_length * actual_aspect_ratio)
    
    # Ensure minimum resolution
    min_dim = 800
    if output_width < min_dim or output_height < min_dim:
        scale_factor = min_dim / min(output_width, output_height)
        output_width = int(output_width * scale_factor)
        output_height = int(output_height * scale_factor)
    
    dst_pts = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(img, M, (output_width, output_height))
    return warped, M

def draw_debug(img, corners):
    debug_img = img.copy()
    
    # Draw corners
    for i, corner in enumerate(corners):
        cv2.circle(debug_img, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
        cv2.putText(debug_img, str(i), tuple(corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw lines between corners
    pts = corners.astype(int)
    cv2.polylines(debug_img, [pts], True, (255, 0, 0), 3)
    
    return debug_img

def detect_and_warp_table(image_path, pocket_model_path, base_length=1500):
    corners = robust_table_corners(image_path, pocket_model_path)
    pockets, _ = detect_pockets(image_path, pocket_model_path)
    img = cv2.imread(image_path)
    warped, H = warp_to_birdseye(img, corners, base_length, pockets)
    debug_img = draw_debug(img, corners)

    cv2.imwrite("output_warped.jpg", warped)
    cv2.imwrite("output_debug.jpg", debug_img)
    return warped, debug_img, corners, H


image_path = "C:/Users/karth/poolShotAI/testInferImage.png"
pocket_model_path = "C:/Users/karth/poolShotAI/pocket_detection_model/weights/best.pt"

warped, debug_img, corners, H = detect_and_warp_table(image_path, pocket_model_path)
print("Warped image saved:", warped.shape)
print("Corners:", corners)
print("Debug image saved as output_debug.jpg")