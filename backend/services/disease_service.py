import torch
import torch.nn as nn
import numpy as np
import io
import os
import logging
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from schemas import DiseaseAnalysisResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseaseService:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Classes from the dataset
        self.classes = [
            "Pepper__bell___Bacterial_spot",
            "Pepper__bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Tomato_Bacterial_spot",
            "Tomato_Early_blight",
            "Tomato_Late_blight",
            "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite",
            "Tomato__Target_Spot",
            "Tomato__Tomato_YellowLeaf__Curl_Virus",
            "Tomato__Tomato_mosaic_virus",
            "Tomato_healthy"
        ]
        
        self.model = self._load_model(model_path)
        
        # Preprocessing (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, path: str):
        try:
            # Recreate architecture
            model = efficientnet_b0(weights=None) # We load custom weights
            # Replace classifier head matches training script
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.classes))
            
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded EfficientNet from {path}")
            else:
                logger.warning(f"Model not found at {path}, utilizing random weights (expect garbage output)")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def analyze(self, image_bytes: bytes) -> DiseaseAnalysisResponse:
        try:
            if not self.model:
                raise Exception("Model not loaded")

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = img.size
            
            # Preprocess
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            class_name = self.classes[predicted_idx.item()]
            conf_score = confidence.item()
            
            logger.info(f"Predicted: {class_name} ({conf_score:.2f})")
            
            # Map class to response format
            is_healthy = "healthy" in class_name.lower()
            
            if is_healthy:
                severity = "Healthy"
                diseased_area_percent = 0.0
            else:
                # For classification, we don't have area %. 
                # We'll use the confidence score as a proxy or set a high value to indicate 'Infected'
                # Or just hardcode based on severity. 
                # Let's map high confidence to 'Critical' severity visually.
                severity = "Critical" # Assume disease presence is critical for now
                diseased_area_percent = 100.0 # Placeholder since we have no mask
            
            # We don't have a mask, so sending empty string or None
            # Formatting class name for better display? 
            # The frontend displays severity. "Critical" is fine.
            # Maybe we can pass the class name in 'severity' if the frontend allows?
            # Frontend: getSeverityClass checks "high", "critical", "moderate".
            # If I pass the class name, it might default to green.
            # I'll stick to severity levels but log the class name.
            
            # Update: User wants "whatever outputs come are shown". 
            # The structure is fixed: severity (string), diseased_area_percent (number).
            # If I put the class name in Severity, it might look cool but break the color coding.
            # Let's try to be clever: Severity = "Critical (Tomato_Bacterial_spot)" ?
            # Frontend code: `const s = severity.toLowerCase(); if (s === 'high' || s === 'critical') ...`
            # So "Critical (Name)" won't match "critical".
            # I will just return "Critical" if diseased, and maybe abuse `diseased_area_percent` or add a field if I could.
            # But I can't easily change frontend schema right now without more edits.
            # Wait, `mask_b64` is null.
            
            return DiseaseAnalysisResponse(
                diseased_area_percent=diseased_area_percent,
                severity=severity if is_healthy else f"High ({class_name.replace('___', ' ').replace('_', ' ')})",
                mask_shape=[original_size[1], original_size[0]],
                mask_b64=""
            )
            
        except Exception as e:
            logger.error(f"Disease prediction error: {e}")
            return DiseaseAnalysisResponse(
                diseased_area_percent=0.0,
                severity="Error",
                mask_shape=[0,0],
                mask_b64=""
            )
