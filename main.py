from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from llama_cpp import Llama
import os
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# Load a single GGUF model mounted from host storage
model_path = "./models/Intelligrader_SAQ_Grader_Qwen.gguf"
llm = None

@app.on_event("startup")
async def startup_event():
    global llm
    if not os.path.exists(model_path):
        error_message = (
            f"Required GGUF model not found at {model_path}. "
            "Ensure the deploy workflow has downloaded the file to the host-mounted models directory."
        )
        logger.error(error_message)
        raise RuntimeError(error_message)

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window
        n_threads=4,  # Number of CPU threads
        n_gpu_layers=0  # CPU only
    )
    logger.info("Model loaded successfully from %s", model_path)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/")
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