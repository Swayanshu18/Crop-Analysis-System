from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.disease_service import DiseaseService
from services.pest_service import PestService
from schemas import CombinedAnalysisResponse
import uvicorn
import os

app = FastAPI(title="Crop Analysis API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_MODEL_PATH = "efficientnet_plant_best.pt"
PEST_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8s_pest.pt")

disease_service = DiseaseService(DISEASE_MODEL_PATH)
pest_service = PestService(PEST_MODEL_PATH)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Crop Analysis Backend Running"}

@app.post("/analyze", response_model=CombinedAnalysisResponse)
async def analyze_crop(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Run analyses
    # We run them sequentially here, but could be parallelized
    disease_result = disease_service.analyze(contents)
    pest_result = pest_service.analyze(contents)
    
    return CombinedAnalysisResponse(
        disease=disease_result,
        pest=pest_result
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
