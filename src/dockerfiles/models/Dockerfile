# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CONTEXT .
WORKDIR /app

# Dépendances système & LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    ca-certificates \
    curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional public CA injected at build time, for environments performing TLS
# inspection. Build with: --secret id=extra_ca,src=<public-ca.crt>
RUN --mount=type=secret,id=extra_ca \
    if [ -s /run/secrets/extra_ca ]; then \
        cp /run/secrets/extra_ca /usr/local/share/ca-certificates/extra-ca.crt; \
        update-ca-certificates; \
    fi

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# Dépendances
COPY src/dockerfiles/common/requirements-ml.txt .
COPY src/dockerfiles/models/requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements-ml.txt \
    -r requirements.txt

# Code métier
COPY src/core/ /app/models/core/
COPY src/features/build_features.py /app/models/build_features.py
COPY src/features/export_dataset.py /app/models/export_dataset.py
COPY src/models/ /app/models/
COPY params.yml /app/models/params.yml

# Entrypoint par défaut
CMD ["bash"]
