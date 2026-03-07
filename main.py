from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from llama_cpp import Llama
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.exceptions import RequestValidationError
import os

from model_manager import ensure_models, get_active_local_model_path

limiter = Limiter(key_func=get_remote_address)
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    active_models = ensure_models()
    model_path = str(get_active_local_model_path(active_models))

    if os.path.exists(model_path):
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4,  # Number of CPU threads
            n_gpu_layers=0  # CPU only
        )
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")

    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: HTTPException(status_code=429, detail="Rate limit exceeded: 10 requests per hour"))

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

@app.get("/")
@limiter.limit("10/hour")
def root(request):
    return RedirectResponse(url="/generate")

@app.get("/health")
@limiter.limit("3/hour")
def health(request):
    model_status = "loaded" if llm is not None else "not loaded"
    return {"status": "healthy", "model": model_status}

@app.post("/generate")
@limiter.limit("10/hour")
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