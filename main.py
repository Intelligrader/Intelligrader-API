from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from llama_cpp import Llama
import os

app = FastAPI()

# Load the GGUF model
model_path = "./models/SmolLM2-Rethink-360M.F32.gguf"
llm = None
load_error = None


def _looks_like_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False

@app.on_event("startup")
async def startup_event():
    global llm, load_error
    llm = None
    load_error = None

    if not os.path.exists(model_path):
        load_error = f"Model not found at {model_path}"
        print(f"Warning: {load_error}")
        return

    if _looks_like_lfs_pointer(model_path):
        load_error = (
            f"Model file at {model_path} is a Git LFS pointer, not a GGUF binary. "
            "Run git lfs pull during deployment."
        )
        print(f"Warning: {load_error}")
        return

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4,  # Number of CPU threads
            n_gpu_layers=0  # CPU only
        )
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        load_error = f"Model initialization failed: {str(e)}"
        print(f"Warning: {load_error}")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health(response: Response):
    model_status = "loaded" if llm is not None else "not loaded"
    if llm is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "model": model_status, "error": load_error}
    return {"status": "healthy", "model": model_status}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail=load_error or "Model not loaded")
    
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