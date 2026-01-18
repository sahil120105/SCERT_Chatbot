FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Explicitly set the Hugging Face cache directory
ENV HF_HOME=/app/model_cache

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY dockerRequirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r dockerRequirements.txt

# --- CRITICAL STEP: BAKE THE MODEL ---
# We run the download command NOW so it saves to the image.
# When the app starts later, it will find the files in /app/model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

COPY src/ ./src/
COPY app.py .
COPY qdrant_db ./qdrant_db

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]