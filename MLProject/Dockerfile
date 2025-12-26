FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY python_env.yaml .

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    mlflow==2.8.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    flask==3.0.0 \
    prometheus-client==0.18.0 \
    gunicorn==21.2.0

# Copy project files
COPY . /app/

# Expose MLflow serving port
EXPOSE 5000

# Set environment variable for MLflow
ENV MLFLOW_TRACKING_URI=""

# Run MLflow model serving
CMD ["mlflow", "models", "serve", "-m", "model", "-h", "0.0.0.0", "-p", "5000"]
