import cv2
import json
from ultralytics import YOLO    

def detect_balls(image_path, ball_model_path, output_image_path="output_balls_detected.jpg", output_json_path="balls_detected.json"):
    img = cv2.imread(image_path)
    model = YOLO(ball_model_path)
    results = model(img)
    annotated = img.copy()
    balls = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            cls = int(box.cls[0].cpu().numpy())
            label = model.names[cls]
            balls.append({'position': [x_center, y_center], 'label': label})

            cv2.circle(annotated, (x_center, y_center), 10, (0, 255, 0), 2)
            cv2.putText(annotated, label, (x_center - 10, y_center - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    cv2.imwrite(output_image_path, annotated)
    with open(output_json_path, "w") as json_file:
        json.dump(balls, json_file, indent=4)
        
    print(f"Detected {len(balls)} balls and saved results to {output_image_path} and {output_json_path}")

    return balls, annotated

    
# RUN
IMAGE = "C:/Users/karth/poolShotAI/detection/output_warped.jpg"
BALL_MODEL = "C:/Users/karth/poolShotAI/ball_detection_model/weights/best.pt"
balls, annotated_image = detect_balls(IMAGE, BALL_MODEL)
