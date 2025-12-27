from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "yo"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/add")
def add_parameters(parameter1: float, parameter2: float):
    try:
        result = parameter1 + parameter2
        return {"result": result}
    except (TypeError, ValueError) as e:
        # HTTP 400 (bad request) with error details.
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")