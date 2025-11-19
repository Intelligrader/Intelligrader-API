from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import logging
import json
from pathlib import Path
import httpx 
import datetime

# --- Configuration ---
# Internal path must match the volume mapping in docker-compose.yml
# NOTE: This path assumes a volume named 'data_logs' is mounted to /app/model_request_data/
LOG_FILE_PATH = Path("/app/model_request_data/logs.jsonl") 
MAX_FOLLOW_UP = 2 # Limit is 2 follow-ups after the initial query (3 total requests)
# FIX APPLIED HERE: Changed port from 8000 to the model runner's actual listening port 12434
MODEL_API_URL = "http://smollm2:12434/v1/completions" 

# --- Data Models ---
class UserRequest(BaseModel):
    session_id: str
    user_query: str

class SessionControl(BaseModel):
    session_id: str

class ConversationEntry(BaseModel):
    role: str # 'user' or 'model'
    content: str
    
# --- In-Memory State Management ---
# Session ID -> List of ConversationEntry objects
conversation_history: Dict[str, List[ConversationEntry]] = {}

app = FastAPI()

# Ensure the log file directory exists for logging 
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Logging Function ---
def log_request_data(session_id: str, user_query: str, model_response: str):
    """Writes request/response data to the mounted volume."""
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": session_id,
            "user_query": user_query,
            "model_response": model_response,
        }
        with open(LOG_FILE_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logging.error(f"Failed to write to volume log: {e}")

# --- API Endpoint ---
@app.post("/chat/")
async def chat_endpoint(request: UserRequest):
    session_id = request.session_id
    user_query = request.user_query
    
    # Get current history (Q/A pairs)
    history = conversation_history.get(session_id, [])
    
    # 1. Check Follow-up Limit
    user_request_count = len([e for e in history if e.role == 'user'])
    
    if user_request_count >= MAX_FOLLOW_UP + 1: # +1 includes the initial query
        raise HTTPException(
            status_code=400,
            detail=f"Session limit of {MAX_FOLLOW_UP} follow-ups reached. Please call /new_session/ to start over."
        )

    # 2. Prepare Conversation Context (History + New Query)
    # The 'messages' structure is critical for chat models
    model_messages = [
        {"role": entry.role, "content": entry.content} 
        for entry in history
    ]
    model_messages.append({"role": "user", "content": user_query})

    # 3. Call the SmollM2 Model API
    try:
        # Use a reasonable timeout for LLM inference (60 seconds)
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                # Note: The 'model' key is often ignored by the Docker runner 
                # since the model is specified via MODEL_NAME ENV
                "model": "smollm2", 
                "messages": model_messages, 
                "max_tokens": 256 # Set a reasonable limit for small model
            }
            
            response = await client.post(MODEL_API_URL, json=payload)
            response.raise_for_status()
            
            # Extract the response text (Standard OpenAI-style API response parsing)
            model_response_data = response.json()
            # The docker model runner uses the standard OpenAI response structure
            model_response = model_response_data['choices'][0]['message']['content']
            
    except httpx.HTTPStatusError as e:
        logging.error(f"Model API returned error: {e.response.text}")
        raise HTTPException(status_code=503, detail="Model service failed to respond with a 2xx status. Check model runner logs.")
    except Exception as e:
        # This catches connection errors (e.g., wrong port, container down)
        logging.error(f"Connection error to model service: {e}")
        raise HTTPException(status_code=503, detail="Cannot connect to the model service. Is the smollm2 container running or is the internal port correct?")

    # 4. Update History
    history.append(ConversationEntry(role="user", content=user_query))
    history.append(ConversationEntry(role="model", content=model_response))
    conversation_history[session_id] = history
    
    # 5. Collect Relevant Data (Log to Volume)
    log_request_data(session_id, user_query, model_response)
    
    # Calculate follow-ups remaining
    new_request_count = len([e for e in history if e.role == 'user'])
    follow_ups_left = MAX_FOLLOW_UP - (new_request_count - 1)
    
    return {
        "session_id": session_id, 
        "response": model_response, 
        "follow_ups_left": max(0, follow_ups_left)
    }

# Endpoint to clear a session
@app.post("/new_session/")
async def new_session_endpoint(request: SessionControl):
    session_id_to_clear = request.session_id
    if session_id_to_clear in conversation_history:
        del conversation_history[session_id_to_clear]
        return {"message": f"Session {session_id_to_clear} cleared. Ready for a new conversation."}
    return {"message": "Session not found or already cleared."}
