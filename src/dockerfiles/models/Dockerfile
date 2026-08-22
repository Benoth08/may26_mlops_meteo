FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CONTEXT .
WORKDIR /app

# Dépendances système & LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances
COPY src/dockerfiles/common/requirements-ml.txt .
COPY src/dockerfiles/models/requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements-ml.txt \
    -r requirements.txt

# Code métier
COPY src/core/ /app/models/core/
COPY src/models/ /app/models/

# Entrypoint par défaut
CMD ["bash"]