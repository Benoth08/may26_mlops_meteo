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
COPY src/dockerfiles/data_integration/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint
COPY src/dockerfiles/data_integration/entrypoint.py .

# Code métier
COPY src/scripts/core/ /app/scripts/core/
COPY src/scripts/data_integration/ /app/scripts/
COPY src/data/ /data/

CMD ["python", "entrypoint.py"]