from ultralytics import YOLO

data_yaml = "augmented_data/split/data.yaml"

model = YOLO("yolov8s.pt")

model.train(
    data=data_yaml,
    epochs=200,
    imgsz=1024,
    batch=8,
    project="ball_detection_model_8s",
    name="weights",
    mosaic=1.0,
    degrees=10,
    translate=0.2,
    scale=0.8,
    shear=3.0,
    perspective=0.0005,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    patience=50,
    save=True,
    save_period=10,
    cache=True,
    device=0,
    optimizer='AdamW',
    lr0=0.01,
    lrf=0.01,
    iou=0.7,
    conf=0.001
)

print("Training finished! Best model weights saved at:")
print("ball_detection_model_8s/weights/best.pt")
