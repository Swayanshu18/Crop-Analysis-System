from ultralytics import YOLO
import sys

try:
    print("Attempting to load model: underdogquality/yolo11s-pest-detection")
    # Trying standard YOLO class first as it handles most versions now
    model = YOLO("underdogquality/yolo11s-pest-detection")
    print("✓ Model initialization successful")
    
    # Optional: print model info
    print(f"Model task: {model.task}")
    print(f"Model names: {model.names}")

except Exception as e:
    print(f"x Failed to load model: {e}")
    sys.exit(1)
