import time

from prometheus_client import Counter, Histogram, make_asgi_app


HTTP_REQUESTS = Counter(
    "weather_api_http_requests_total",
    "Nombre total de requetes HTTP recues par Weather API",
    ["method", "endpoint", "status"]
)


HTTP_REQUEST_DURATION = Histogram(
    "weather_api_http_request_duration_seconds",
    "Duree des requetes HTTP de Weather API",
    ["method", "endpoint"]
)


def setup_metrics(app):

    @app.middleware("http")
    async def prometheus_middleware(request, call_next):

        # Prometheus ne doit pas mesurer son propre scraping.
        if request.url.path.startswith("/metrics"):
            return await call_next(request)

        start_time = time.perf_counter()
        status_code = "500"

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response

        finally:
            duration = time.perf_counter() - start_time

            route = request.scope.get("route")

            if route is not None:
                endpoint = getattr(
                    route,
                    "path",
                    request.url.path
                )
            else:
                endpoint = request.url.path

            HTTP_REQUESTS.labels(
                method=request.method,
                endpoint=endpoint,
                status=status_code
            ).inc()

            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)

    # Endpoint Prometheus :
    # http://api-weather-X:8057/metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
