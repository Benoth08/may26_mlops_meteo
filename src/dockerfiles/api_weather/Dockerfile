FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CONTEXT .
WORKDIR /app


# build-essential et libgomp1 : necessaires a LightGBM. curl : pour le healthcheck.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances
COPY src/dockerfiles/common/requirements-ml.txt .
COPY src/dockerfiles/api_weather/requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements-ml.txt \
    -r requirements.txt

# Code métier (Entrypoint : src/api_weather/main.py
COPY src/core/ /app/core/
COPY src/api_weather/ /app/
COPY src/features/build_features.py /app/

#Entrypoint
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8057"]