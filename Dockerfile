FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for llama.cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies including llama-cpp-python
# CMAKE_ARGS ensures CPU-only build
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
