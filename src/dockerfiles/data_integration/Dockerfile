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
COPY src/dockerfiles/data_integration/requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements-ml.txt \
    -r requirements.txt

# Entrypoint
COPY src/dockerfiles/data_integration/entrypoint.py .

# Code métier
COPY src/core/ /app/core/
COPY src/data_integration/ /app/
COPY data/ /data/

CMD ["python", "entrypoint.py"]