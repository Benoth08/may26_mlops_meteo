FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CONTEXT .
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Dépendances
COPY src/dockerfiles/common/requirements-ml.txt .
COPY src/dockerfiles/data_preprocessing/requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements-ml.txt \
    -r requirements.txt

# Code métier
COPY src/core/ /app/data_preprocessing/core/
COPY src/features/ /app/data_preprocessing/
COPY src/data_preprocessing /app/data_preprocessing/
COPY data/ /data/

# Entrypoint par défaut
CMD ["bash"]