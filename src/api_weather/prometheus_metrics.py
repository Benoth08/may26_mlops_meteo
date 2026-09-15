import json
import time
from pathlib import Path

import numpy as np

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    make_asgi_app,
)


# ============================================================
# HTTP / API
# ============================================================

HTTP_REQUESTS = Counter(
    "weather_api_http_requests_total",
    "Nombre total de requetes HTTP recues par Weather API",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "weather_api_http_request_duration_seconds",
    "Duree des requetes HTTP de Weather API",
    ["method", "endpoint"],
)


# ============================================================
# MODEL SERVING
# ============================================================

MODEL_PREDICTIONS = Counter(
    "weather_model_predictions_total",
    "Nombre total de predictions effectuees par le modele",
    ["prediction"],
)

# Compteurs initialises afin que les deux series existent avant la premiere
# prediction et soient immediatement visibles dans Prometheus.
MODEL_PREDICTIONS.labels(prediction="Yes")
MODEL_PREDICTIONS.labels(prediction="No")

MODEL_INFERENCE_DURATION = Histogram(
    "weather_model_inference_duration_seconds",
    "Temps d'inference du modele",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)

MODEL_PREDICTION_PROBABILITY = Histogram(
    "weather_model_prediction_probability",
    "Distribution des probabilites de pluie produites par le modele",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
)

MODEL_ERRORS = Counter(
    "weather_model_prediction_errors_total",
    "Nombre d'erreurs pendant les predictions",
    ["error_type"],
)

MODEL_BATCH_SIZE = Histogram(
    "weather_model_batch_size",
    "Nombre d'observations traitees par batch",
    buckets=(1, 10, 50, 100, 500, 1000, 5000, 10000, 50000),
)


# ============================================================
# MODEL STATUS
# ============================================================

MODEL_LOADED = Gauge(
    "weather_model_loaded",
    "Indique si le modele est charge: 1=oui, 0=non",
)

MODEL_INFO = Info(
    "weather_model",
    "Informations sur le modele actuellement utilise",
)


# ============================================================
# OFFLINE MODEL PERFORMANCE
# ============================================================

# Ces gauges representent uniquement la derniere evaluation offline du modele.
# Elles ne mesurent en aucun cas la qualite des predictions en temps reel.
MODEL_OFFLINE_QUALITY = Gauge(
    "weather_model_offline_quality",
    "Valeur calculee lors de la derniere evaluation offline du modele",
    ["metric"],
)

MODEL_OFFLINE_METRICS_AVAILABLE = Gauge(
    "weather_model_offline_metrics_available",
    "Indique si les metriques de la derniere evaluation sont disponibles",
)

MODEL_OFFLINE_METRICS_UPDATED_TIMESTAMP = Gauge(
    "weather_model_offline_metrics_updated_timestamp_seconds",
    "Date de modification du fichier de la derniere evaluation du modele",
)

_OFFLINE_METRICS_MTIME = None

MODEL_ACCURACY = Gauge(
    "weather_model_accuracy",
    "Accuracy du modele sur le dernier jeu d'evaluation",
)

MODEL_F1 = Gauge(
    "weather_model_f1",
    "F1-score du modele sur le dernier jeu d'evaluation",
)

MODEL_PRECISION = Gauge(
    "weather_model_precision",
    "Precision classe pluie",
)

MODEL_RECALL = Gauge(
    "weather_model_recall",
    "Recall classe pluie",
)

MODEL_ROC_AUC = Gauge(
    "weather_model_roc_auc",
    "ROC-AUC du modele",
)

MODEL_PR_AUC = Gauge(
    "weather_model_pr_auc",
    "PR-AUC du modele",
)

MODEL_THRESHOLD = Gauge(
    "weather_model_prediction_threshold",
    "Seuil de classification utilise",
)


# ============================================================
# RECORD PREDICTION
# ============================================================

def record_predictions(predictions):
    """Enregistre une prediction ou un lot de labels Yes/No reussi."""
    values = np.asarray(predictions, dtype=object).reshape(-1)
    if values.size == 0:
        return

    yes_count = int(np.count_nonzero(values == "Yes"))
    no_count = int(np.count_nonzero(values == "No"))
    invalid_count = int(values.size) - yes_count - no_count
    if invalid_count:
        raise ValueError("Les predictions doivent etre 'Yes' ou 'No'.")

    if yes_count:
        MODEL_PREDICTIONS.labels(prediction="Yes").inc(yes_count)
    if no_count:
        MODEL_PREDICTIONS.labels(prediction="No").inc(no_count)

def record_prediction(probability: float, duration: float):
    prediction = "Yes" if probability >= 0.5 else "No"

    record_predictions(prediction)

    MODEL_INFERENCE_DURATION.observe(duration)

    MODEL_PREDICTION_PROBABILITY.observe(
        probability
    )


def record_batch(probabilities, duration: float):
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    MODEL_BATCH_SIZE.observe(probabilities.size)

    if probabilities.size:
        MODEL_INFERENCE_DURATION.observe(duration)

    record_predictions(
        np.where(probabilities >= 0.5, "Yes", "No")
    )

    for probability in probabilities:
        MODEL_PREDICTION_PROBABILITY.observe(
            float(probability)
        )


def record_model_error(error_type: str):
    MODEL_ERRORS.labels(
        error_type=error_type
    ).inc()


# ============================================================
# MODEL STATUS / METADATA
# ============================================================

def update_model_status(
    loaded: bool,
    metadata: dict | None = None,
):
    MODEL_LOADED.set(
        1 if loaded else 0
    )

    if loaded:
        metadata = metadata or {}

        MODEL_INFO.info(
            {
                "version": str(
                    metadata.get(
                        "model_version",
                        "unknown",
                    )
                ),
                "model_type": str(
                    metadata.get(
                        "model_type",
                        "unknown",
                    )
                ),
            }
        )


# ============================================================
# OFFLINE METRICS
# ============================================================

def load_model_evaluation_metrics(
    metrics_path="/metrics/scores.json",
):
    global _OFFLINE_METRICS_MTIME
    path = Path(metrics_path)

    if not path.exists():
        MODEL_OFFLINE_METRICS_AVAILABLE.set(0)
        _OFFLINE_METRICS_MTIME = None
        return False

    try:
        mtime = path.stat().st_mtime
        if _OFFLINE_METRICS_MTIME == mtime:
            return True

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            scores = json.load(file)

        MODEL_ACCURACY.set(
            float(scores.get("accuracy", 0))
        )

        MODEL_F1.set(
            float(scores.get("f1", 0))
        )

        MODEL_PRECISION.set(
            float(
                scores.get(
                    "precision_pluie",
                    0,
                )
            )
        )

        MODEL_RECALL.set(
            float(
                scores.get(
                    "recall_pluie",
                    0,
                )
            )
        )

        MODEL_ROC_AUC.set(
            float(scores.get("roc_auc", 0))
        )

        MODEL_PR_AUC.set(
            float(scores.get("pr_auc", 0))
        )

        MODEL_THRESHOLD.set(
            float(scores.get("threshold", 0.5))
        )

        metric_mapping = {
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision_pluie",
            "recall": "recall_pluie",
            "roc_auc": "roc_auc",
            "pr_auc": "pr_auc",
        }
        for metric, score_key in metric_mapping.items():
            MODEL_OFFLINE_QUALITY.labels(metric=metric).set(
                float(scores[score_key])
            )

        MODEL_OFFLINE_METRICS_AVAILABLE.set(1)
        MODEL_OFFLINE_METRICS_UPDATED_TIMESTAMP.set(mtime)
        _OFFLINE_METRICS_MTIME = mtime

        return True

    except Exception:
        MODEL_OFFLINE_METRICS_AVAILABLE.set(0)
        return False


# ============================================================
# FASTAPI MONITORING
# ============================================================

def setup_metrics(app):

    @app.middleware("http")
    async def prometheus_middleware(
        request,
        call_next,
    ):
        if request.url.path.startswith("/metrics"):
            # Le JSON n'est relu que si sa date de modification a change.
            load_model_evaluation_metrics()
            return await call_next(request)

        start_time = time.perf_counter()
        status_code = "500"

        try:
            response = await call_next(
                request
            )

            status_code = str(
                response.status_code
            )

            return response

        finally:
            duration = (
                time.perf_counter()
                - start_time
            )

            route = request.scope.get(
                "route"
            )

            if route is not None:
                endpoint = getattr(
                    route,
                    "path",
                    request.url.path,
                )
            else:
                endpoint = (
                    request.url.path
                )

            HTTP_REQUESTS.labels(
                method=request.method,
                endpoint=endpoint,
                status=status_code,
            ).inc()

            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duration)

    metrics_app = make_asgi_app()

    app.mount(
        "/metrics",
        metrics_app,
    )
