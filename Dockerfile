FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Gradino dependencies (from local submodule)
COPY gradino/requirements.txt /tmp/gradino_req.txt
RUN pip install --no-cache-dir -r /tmp/gradino_req.txt

# rdflib is imported by unit_converter.py but missing from requirements.txt
RUN pip install --no-cache-dir rdflib

# Copy Gradino source
COPY gradino/ /app/gradino/

# Install backend dependencies
COPY backend/requirements.txt /app/backend_requirements.txt
RUN pip install --no-cache-dir -r /app/backend_requirements.txt

# Copy backend application
COPY backend/ /app/

# Create output directory
RUN mkdir -p /app/output

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
