from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse

import model_manager
from generation_engine import GenerationEngine
from schemas import (
    GenerateSAQRequest,
    GenerateSAQResponse,
    GradeSAQRequest,
    GradeSAQResponse,
)

app = FastAPI()
engine = GenerationEngine()

@app.on_event("startup")
async def startup_event():
    engine.startup()


@app.on_event("shutdown")
async def shutdown_event():
    engine.shutdown()

@app.get("/")
def root():
    return RedirectResponse(url="/generate")

@app.get("/health")
def health():
    model_status = engine.model_status()
    if model_status != "loaded":
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model": model_status, "queue_depth": engine.queue_depth()},
        )
    return {"status": "healthy", "model": model_status, "queue_depth": engine.queue_depth()}

@app.get("/models")
def list_models():
    supported_model_ids = [
        model_manager._MODELS_BY_ID[model_id]["id"]
        for model_id in model_manager._MODELS_BY_ID.keys()
    ]
    return {
        "supported": sorted(supported_model_ids),
        "loaded": engine.loaded_model_id,
        "max_loaded_on_disk": model_manager.MAX_LOADED_MODELS,
        "queue_max_size": engine.max_queue_size,
        "queue_depth": engine.queue_depth(),
    }

@app.post("/generate", response_model=GenerateSAQResponse)
def generate_text(request: GenerateSAQRequest):
    return engine.generate(request)


@app.post("/grade-saq", response_model=GradeSAQResponse)
def grade_saq(request: GradeSAQRequest):
    return engine.grade_saq(request)