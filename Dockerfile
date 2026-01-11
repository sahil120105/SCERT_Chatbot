# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# 1. Install Dependencies
# We copy requirements first to leverage Docker cache
COPY dockerRequirements.txt .
RUN pip install --no-cache-dir -r dockerRequirements.txt

# 2. Copy the Application Code
# We strictly copy only what you requested: app.py, qdrant_db, and src
COPY app.py .
COPY src/ ./src/
COPY qdrant_db/ ./qdrant_db/

# Note: We do NOT copy the 'data' folder or venv, keeping the image small.

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application
# We bind to 0.0.0.0 so the browser can access it from outside the container
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]