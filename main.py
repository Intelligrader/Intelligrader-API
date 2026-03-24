from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from llama_cpp import Llama
import gc
import logging

import model_manager

app = FastAPI()
logger = logging.getLogger(__name__)

llm = None
loaded_model_id: str | None = None

@app.on_event("startup")
async def startup_event():
    global llm, loaded_model_id

    model_id = model_manager.get_default_model_id()
    model_path = model_manager.get_model_path(model_id)
    config = model_manager.get_model_config(model_id)

    llm = Llama(model_path=str(model_path), **config["runtime"])
    loaded_model_id = model_id
    logger.info("Default model loaded: %s", model_id)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    model: str | None = None

@app.get("/")
def root():
    return RedirectResponse(url="/generate")

@app.get("/health")
def health():
    model_status = "loaded" if llm is not None else "not loaded"
    return {"status": "healthy", "model": model_status}

@app.get("/models")
def list_models():
    supported_model_ids = [
        model_manager._MODELS_BY_ID[model_id]["id"]
        for model_id in model_manager._MODELS_BY_ID.keys()
    ]
    return {
        "supported": sorted(supported_model_ids),
        "loaded": loaded_model_id,
        "max_loaded_on_disk": model_manager.MAX_LOADED_MODELS
    }

@app.post("/generate")
def generate_text(request: GenerateRequest):
    global llm, loaded_model_id
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Resolve the requested model
    requested_model_id = request.model or loaded_model_id
    
    # Validate the requested model
    if not model_manager.is_supported(requested_model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: '{requested_model_id}'. See /models for supported options."
        )
    
    # Switch models if necessary
    if loaded_model_id != requested_model_id:
        # Unload current model and free memory
        old_llm = llm
        llm = None
        del old_llm
        gc.collect()
        
        # Load the new model
        model_path = model_manager.get_model_path(requested_model_id)
        config = model_manager.get_model_config(requested_model_id)
        llm = Llama(model_path=str(model_path), **config["runtime"])
        loaded_model_id = requested_model_id
        logger.info("Model switched to: %s", requested_model_id)
    
    try:
        output = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False
        )
        
        return {
            "prompt": request.prompt,
            "generated_text": output["choices"][0]["text"],
            "tokens_used": output["usage"]["total_tokens"],
            "model_used": loaded_model_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")