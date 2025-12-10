from ultralytics import YOLO
import io
import os
from PIL import Image
import numpy as np
from schemas import PestAnalysisResponse, PestBox

class PestService:
    def __init__(self, model_path: str):
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded Pest YOLO from {model_path}")
        else:
            print(f"Warning: Pest model not found at {model_path}")
            self.model = None

    def analyze(self, image_bytes: bytes) -> PestAnalysisResponse:
        try:
            if not self.model:
                raise Exception("Model not loaded")

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            results = self.model.predict(img, conf=0.25)
            
            pest_boxes = []
            result = results[0]
            
            # Use boxes.xyxy (x1, y1, x2, y2) or boxes.xywh (x, y, w, h)
            # The schema expects [x, y, w, h]
            
            if result.boxes:
                for box in result.boxes:
                    xywh = box.xywh[0].cpu().numpy().tolist() # x_center, y_center, w, h
                    # Convert center to top-left for frontend if needed? 
                    # Usually bounding boxes are x,y,w,h (top-left). 
                    # YOLO xywh is center. Let's send xywh as is but label it clearly, 
                    # OR convert to top-left x,y,w,h which is standard for HTML canvas/drawing.
                    
                    # Converting to top-left x,y
                    w = xywh[2]
                    h = xywh[3]
                    x = xywh[0] - w/2
                    y = xywh[1] - h/2
                    
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = result.names[cls]
                    
                    pest_boxes.append(PestBox(
                        box=[x, y, w, h],
                        label=label,
                        confidence=conf,
                        class_id=cls
                    ))

            count = len(pest_boxes)
            if count == 0: severity = "None"
            elif count <= 3: severity = "Low"
            elif count <= 8: severity = "Moderate"
            else: severity = "High"

            return PestAnalysisResponse(
                count=count,
                severity=severity,
                boxes=pest_boxes
            )

        except Exception as e:
            print(f"Pest prediction error: {e}")
            return PestAnalysisResponse(
                count=0,
                severity="Error",
                boxes=[]
            )
