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
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

def robust_table_corners(image_path, pocket_model_path, use_pockets=True):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_cnt = max(contours, key=cv2.contourArea)

    pockets, _ = detect_pockets(image_path, pocket_model_path)
    if use_pockets and pockets is not None and len(pockets) >= 4:
        points = pockets
    else:
        points = table_cnt[:, 0, :].astype(np.float32)

    hull = cv2.convexHull(points)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) == 4:
        hull_pts = approx[:, 0, :]
    else:
        rect = cv2.minAreaRect(points)
        hull_pts = cv2.boxPoints(rect)

    return order_corners(hull_pts)

def draw_debug(img, corners):
    img_copy = img.copy()
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(corners):
        cv2.circle(img_copy, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(img_copy, labels[i], (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for i in range(4):
        pt1 = tuple(map(int, corners[i]))
        pt2 = tuple(map(int, corners[(i + 1) % 4]))
        cv2.line(img_copy, pt1, pt2, (255, 0, 0), 2)
    return img_copy

def warp_to_birdseye(img, corners, base_length=1000):
    width_top = np.linalg.norm(corners[1] - corners[0])
    width_bottom = np.linalg.norm(corners[2] - corners[3])
    height_left = np.linalg.norm(corners[3] - corners[0])
    height_right = np.linalg.norm(corners[2] - corners[1])

    width = max(width_top, width_bottom)
    height = max(height_left, height_right)

    output_width = base_length
    output_height = base_length // 2

    if height > width:  # portrait → rotate
        dst_pts = np.array([
            [0, 0],
            [0, output_height - 1],
            [output_width - 1, output_height - 1],
            [output_width - 1, 0]
        ], dtype="float32")
    else:
        dst_pts = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(img, M, (output_width, output_height))
    return warped, M

def detect_and_warp_table(image_path, pocket_model_path, base_length=1000):
    corners = robust_table_corners(image_path, pocket_model_path)
    img = cv2.imread(image_path)
    warped, H = warp_to_birdseye(img, corners, base_length)
    debug_img = draw_debug(img, corners)

    cv2.imwrite("output_warped.jpg", warped)
    cv2.imwrite("output_debug.jpg", debug_img)
    return warped, debug_img, corners, H


image_path = "C:/Users/karth/poolShotAI/testInferImage2.png"
pocket_model_path = "C:/Users/karth/poolShotAI/pocket_detection_model/weights/best.pt"

warped, debug_img, corners, H = detect_and_warp_table(image_path, pocket_model_path)
print("Warped image saved:", warped.shape)
print("Corners:", corners)
print("Debug image saved as output_debug.jpg")
