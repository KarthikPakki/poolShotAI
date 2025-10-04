import cv2
import numpy as np
from ultralytics import YOLO



def detect_dots(image_path, dot_model_path, dot_label="Dot"):
    img = cv2.imread(image_path)
    model = YOLO(dot_model_path)
    results = model(img)
    
    dots = []
    for result in results:
        for box, cls in zip(result.boxes, result.boxes.cls):
            cls_name = model.names[int(cls.cpu().numpy())]
            if cls_name == dot_label:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                dots.append([x_center, y_center])
                cv2.circle(img, (x_center, y_center), 5, (0, 255, 0), -1)
    
    return np.array(dots), img


'''
def detect_pockets(image_path, pocket_model_path):
    img = cv2.imread(image_path)
    model = YOLO(pocket_model_path)
    results = model(img)
    
    pockets = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            pockets.append([x_center, y_center])
    
    return np.array(pockets), img
    
'''


def order_corners(pts):
    
    ### order the corners of the board
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_corner_pockets(pockets):
    
    
    ### Find the corner pockets from the detected pockets
    center = pockets.mean(axis=0)
    distances = np.linalg.norm(pockets - center, axis=1)
    corner_indices = np.argsort(distances)[-4:]
    corners = pockets[corner_indices]
    return order_corners(corners)


def warp_to_birds_eye(img, corners, output_width=500):
    
    ### apply perspective transform to get bird's eye view
    
    
    ## 2:1 ratio of table
    output_height = output_width * 2
    
    dst_pts = np.array([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(img, M, (output_width, output_height))
    
    return warped, M


def detect_and_warp_table(image_path, pocket_model_path, output_width=500):
    
    ### main function to detect table using pockets and then warp it to bird's eye view
    pockets, img = detect_pockets(image_path, pocket_model_path)
    corners = find_corner_pockets(pockets)
    warped, homography = warp_to_birds_eye(img, corners, output_width)
    
    return {
        'warped': warped,
        'homography': homography,
        'corners': corners,
        'original': img
    }


# RUN
IMAGE = "C:/Users/karth/poolShotAI/testInferImage.jpg"
POCKET_MODEL = "C:/Users/karth/poolShotAI/pocket_detection_model/weights/best.pt"

result = detect_and_warp_table(IMAGE, POCKET_MODEL)

cv2.imwrite("output_warped.jpg", result['warped'])
print(f"Warped image saved. Shape: {result['warped'].shape}")
print(f"Found {len(result['corners'])} corners")