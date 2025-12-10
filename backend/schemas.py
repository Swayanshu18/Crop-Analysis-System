from pydantic import BaseModel
from typing import List, Optional

class PestBox(BaseModel):
    box: List[float]  # [x, y, w, h]
    label: str
    confidence: float
    class_id: int

class PestAnalysisResponse(BaseModel):
    count: int
    severity: str
    boxes: List[PestBox]

class DiseaseAnalysisResponse(BaseModel):
    diseased_area_percent: float
    severity: str
    mask_shape: List[int] # [height, width]
    mask_b64: str # Base64 encoded PNG of the mask

class CombinedAnalysisResponse(BaseModel):
    disease: DiseaseAnalysisResponse
    pest: PestAnalysisResponse
