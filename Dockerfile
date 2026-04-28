FROM python:3.11-slim

LABEL maintainer="forecasting-system"
LABEL description="Time Series Sales Forecasting API"

# System dependencies for Prophet / CmdStan
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import cmdstanpy; cmdstanpy.install_cmdstan()" && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs training_data

# Train models at build time (optional — can also be done at runtime)
# Uncomment the next line if you want models baked into the image:
# RUN python run_training.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
