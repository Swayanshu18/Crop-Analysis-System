"""
YOLOv8 Pest Detection Training Script (CPU Compatible)
"""

import os
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("       YOLOv8-Small Pest Detection Training (CPU)")
    print("=" * 70)

    # Use absolute path for YAML file
    DATA_YAML = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'pests', 'pest.yaml'))
    EPOCHS = 10
    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    DEVICE = "cpu"

    # Load YOLOv8-small pretrained weights
    model = YOLO("yolov8s.pt")

    # Train
    model.train(
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=0,
        optimizer="Adam",
        lr0=1e-3,
        workers=0,          # IMPORTANT for Windows CPU
        cos_lr=True,
        pretrained=True,
        project="runs_pest",
        name="yolov8s_pest_cpu",
        verbose=True,
        amp=False           # CPU can't use AMP
    )

    print("\nTraining finished!")
    print("Best model saved at: runs_pest/yolov8s_pest_cpu/weights/best.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
