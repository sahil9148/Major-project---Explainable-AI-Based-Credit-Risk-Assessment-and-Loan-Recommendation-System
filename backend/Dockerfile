FROM python:3.11-slim

WORKDIR /app

# Build tools needed for compiled wheels (xgboost/shap use native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at image build time so the container is immediately
# ready to serve. For a real production system you'd separate training
# into its own pipeline/artifact store instead of baking it into the image.
RUN python train.py

EXPOSE 8000

# Shell form (not exec-array form) so ${PORT} gets expanded at container start.
# Render injects PORT env var; we listen on it (or default to 8000 for local dev).
# Let Render handle health checks — its port detection is more reliable than
# trying to probe from inside the container.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
