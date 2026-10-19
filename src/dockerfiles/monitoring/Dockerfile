FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CONTEXT .
WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Dépendances
COPY src/dockerfiles/common/requirements-ml.txt .
COPY src/dockerfiles/monitoring/requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements-ml.txt \
    -r requirements.txt

# Code métier
COPY params.yml /app/monitoring/params.yml
COPY src/core/ /app/monitoring/core/
COPY src/features/ /app/monitoring/
COPY src/monitoring /app/monitoring/

# Entrypoint par défaut
CMD ["bash"]
