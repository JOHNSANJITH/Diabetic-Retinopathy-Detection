import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, HTTPException, UploadFile
from drd.inference import load_model, predict_image
from drd.explainability import gradcam

app=FastAPI(title="Diabetic Retinopathy CV API", version="2.0.0", description="Research/portfolio inference API; not for clinical diagnosis.")
MODEL_PATH=Path(os.getenv("MODEL_PATH","models/best.keras"))
_model=None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists(): raise HTTPException(503, f"Model not found at {MODEL_PATH}. Train a model and set MODEL_PATH.")
        _model=load_model(MODEL_PATH)
    return _model

@app.get("/health")
def health(): return {"status":"ok","model_available":MODEL_PATH.exists()}

@app.post("/predict")
async def predict(file: UploadFile=File(...), explain: bool=False):
    if file.content_type not in {"image/jpeg","image/png","image/webp"}: raise HTTPException(415,"Supported image types: JPEG, PNG, WEBP")
    suffix=Path(file.filename or "image.jpg").suffix or ".jpg"
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read()); path=Path(tmp.name)
    try:
        result=predict_image(get_model(),path)
        if explain:
            output=path.with_name(path.stem+"_gradcam.png")
            _, result["gradcam_path"], result["gradcam_layer"]=gradcam(get_model(),path,output)
        return result
    finally:
        path.unlink(missing_ok=True)
