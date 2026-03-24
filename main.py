from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from llama_cpp import Llama
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

@app.get("/")
def root():
    return RedirectResponse(url="/generate")

@app.get("/health")
def health():
    model_status = "loaded" if llm is not None else "not loaded"
    return {"status": "healthy", "model": model_status}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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
            "tokens_used": output["usage"]["total_tokens"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")