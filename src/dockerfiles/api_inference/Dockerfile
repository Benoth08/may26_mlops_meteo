FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# build-essential et libgomp1 : necessaires a LightGBM. curl : pour le healthcheck.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Le contexte de build est la racine du projet (voir docker-compose)
COPY src/dockerfiles/api_inference/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# On copie l'API et le paquet src (build_features est utilise par l'API)
COPY inference_api.py .
COPY src/ ./src/

CMD ["python", "-m", "uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8058"]
