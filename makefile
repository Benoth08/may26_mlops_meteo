# =============================================================================
# Makefile — Projet Weather MLOps
# Gestion complète des dockers : build, start, stop, reset, purge
# =============================================================================
PROJECT_NAME = weather-mlops

DOCKER_COMPOSE = docker compose
# Stack principale + couche de supervision (NGINX, 6 instances API, Prometheus, Grafana)
DOCKER_COMPOSE_MON = docker compose -f docker-compose.yml -f docker-compose.monitoring.yml
MONITORING_SERVICES = grafana prometheus nginx-exporter nginx
API_REPLICAS = api-weather-2 api-weather-3 api-weather-4 api-weather-5 api-weather-6


# -----------------------------------------------------------------------------
# VERIFICATIONS ENVIRONNEMENT
# -----------------------------------------------------------------------------

check-docker:
	@command -v docker >/dev/null 2>&1 || \
		(echo "Docker n'est pas installé"; exit 1)
	@echo "Docker disponible"

check-compose:
	@docker compose version >/dev/null 2>&1 || \
		(echo "Docker Compose v2 absent. Installez docker-compose-plugin"; exit 1)
	@echo "Docker Compose v2 disponible"

check-env: check-docker check-compose


# -----------------------------------------------------------------------------
# COMMANDES PRINCIPALES
# -----------------------------------------------------------------------------

## Installation complète : build + up
install: check-env
	$(DOCKER_COMPOSE) --profile tasks build
	$(DOCKER_COMPOSE) up -d
	@echo "Stack démarrée."

## Démarrage simple
start: check-env
	$(DOCKER_COMPOSE) up -d
	@echo "Services démarrés."

## Démarrage + suivi des logs en direct (Ctrl+C n'arrête pas les services)
start-logs: check-env
	$(DOCKER_COMPOSE) up -d
	@echo "Services démarrés. Logs en direct (Ctrl+C pour quitter, les services continuent de tourner)."
	$(DOCKER_COMPOSE) logs -f

## Arrêt propre
stop:
	$(DOCKER_COMPOSE_MON) stop
	@echo "Services arrêtés."

## Arrêt + suppression des conteneurs
down:
	$(DOCKER_COMPOSE_MON) down
	@echo "Conteneurs supprimés."


## Reset total : stop + purge volumes + purge images + rebuild + restart
reset: check-env
	$(DOCKER_COMPOSE_MON) down -v --remove-orphans
	$(DOCKER_COMPOSE) --profile tasks down -v --rmi local --remove-orphans
	$(DOCKER_COMPOSE) --profile tasks build --no-cache
	$(DOCKER_COMPOSE) up -d
	@echo "Reset complet terminé."


# -----------------------------------------------------------------------------
# SUPERVISION : NGINX + 6 instances API + Prometheus + Grafana
# -----------------------------------------------------------------------------

## Build + démarrage de la stack complète avec supervision
install-monitoring: check-env
	$(DOCKER_COMPOSE_MON) --profile tasks build
	$(DOCKER_COMPOSE_MON) up -d
	@echo "Stack + supervision démarrées."
	@echo "  API (NGINX) : https://localhost/docs"
	@echo "  Prometheus  : http://localhost:9090"
	@echo "  Grafana     : http://localhost:3000"

## Démarrage de la stack complète avec supervision
start-monitoring: check-env
	$(DOCKER_COMPOSE_MON) up -d
	@echo "Stack + supervision démarrées."
	@echo "  API (NGINX) : https://localhost/docs"
	@echo "  Prometheus  : http://localhost:9090"
	@echo "  Grafana     : http://localhost:3000"

## Arrêt des seuls services de supervision (la stack principale continue)
stop-monitoring:
	$(DOCKER_COMPOSE_MON) rm -sf $(MONITORING_SERVICES) $(API_REPLICAS)
	$(DOCKER_COMPOSE) up -d api-weather
	@echo "Supervision arrêtée. API de nouveau sur http://localhost:8057"

## Etat des conteneurs (stack + supervision)
status-monitoring:
	$(DOCKER_COMPOSE_MON) ps

## Logs en direct des services de supervision
logs-monitoring:
	$(DOCKER_COMPOSE_MON) logs -f nginx prometheus grafana

## Rebuild de l'API en mode supervision (image partagée par les 6 instances)
reset-apiweather-monitoring: check-env
	$(DOCKER_COMPOSE_MON) build api-weather
	$(DOCKER_COMPOSE_MON) up -d api-weather $(API_REPLICAS) nginx
	@echo "API Weather (6 instances) reconstruite et redémarrée."


# -----------------------------------------------------------------------------
# RESET COMPLET
# -----------------------------------------------------------------------------

## Réinstalle Docker pour nettoyer tous les repertoires (layers...)
reset-docker:
	sudo systemctl stop docker
	sudo rm -rf /var/lib/docker
	sudo systemctl start docker

	

# -----------------------------------------------------------------------------
# OUTILS
# -----------------------------------------------------------------------------

## Migration manuelle d'une base existante (ancien schéma CamelCase)
migrate-db: check-env
	docker exec -i weather-postgres sh -c 'psql -v ON_ERROR_STOP=1 -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' \
		< src/sql/migrations/migrate_weather_data_raw.sql


## Purge des logs Airflow + Weather
clean-logs:
	rm -rf logs/*
	mkdir -p logs
	@echo "Logs nettoyés."

## Nettoyage Docker
clean-docker:
	@echo "Nettoyage Docker..."
	docker builder prune -af
	docker system prune -af
	@echo "Docker nettoyé."

## Nettoyage Volume
clean-volumes:
	@echo "Nettoyage Volumes..."
	docker system prune -af --volumes
	@echo "Volumes nettoyés."
	
## Rebuild uniquement Airflow
rebuild-airflow: check-env
	$(DOCKER_COMPOSE) build airflow-apiserver airflow-scheduler airflow-worker airflow-triggerer airflow-dag-processor
	@echo "Airflow reconstruit."

## Rebuild uniquement data-integration
rebuild-integration: check-env
	$(DOCKER_COMPOSE) --profile tasks build data-integration
	@echo "data-integration reconstruit."

## Rebuild uniquement data-preprocessing
rebuild-preprocessing: check-env
	$(DOCKER_COMPOSE) --profile tasks build data-preprocessing
	@echo "data-preprocessing reconstruit."

## Rebuild uniquement models
rebuild-models: check-env
	$(DOCKER_COMPOSE) --profile tasks build models
	@echo "models reconstruit."

## Rebuild image de surveillance
rebuild-monitoring: check-env
	$(DOCKER_COMPOSE) --profile tasks build monitoring
	@echo "monitoring reconstruit."

## Rebuild API Weather
reset-apiweather: check-env
	$(DOCKER_COMPOSE) build api-weather
	$(DOCKER_COMPOSE) up -d api-weather
	@echo "API Weather reconstruite et redémarré."
# -----------------------------------------------------------------------------
# HELP
# -----------------------------------------------------------------------------

help:
	@echo ""
	@echo "Commandes disponibles :"
	@echo ""
	@echo "  make install               → Verification + Build + start"
	@echo "  make start                 → Start services"
	@echo "  make start-logs            → Start services + suivi des logs en direct"
	@echo "  make stop                  → Stop services"
	@echo "  make down                  → Stop + remove containers"
	@echo "  make reset                 → Reset complet (purge + rebuild)"
	@echo ""
	@echo "  make clean-logs            → Purge logs"
	@echo "  make clean-docker          → Purge cache Docker"
	@echo "  make clean-volumes         → Purge volumes"
	@echo "  make rebuild-airflow       → Rebuild Airflow"
	@echo "  make rebuild-integration   → Rebuild data-integration"
	@echo "  make rebuild-preprocessing → Rebuild data-preprocessing"
	@echo "  make rebuild-models        → Rebuild models"
	@echo "  make reset-apiweather      → Rebuild API Weather"
	@echo ""
	@echo "  make install-monitoring    → Build + start stack avec supervision"
	@echo "  make start-monitoring      → Start stack + NGINX, Prometheus, Grafana"
	@echo "  make stop-monitoring       → Stop supervision uniquement"
	@echo "  make status-monitoring     → Etat des conteneurs"
	@echo "  make logs-monitoring       → Logs NGINX / Prometheus / Grafana"
	@echo "  make reset-apiweather-monitoring → Rebuild API (6 instances)"
	@echo ""
	@echo "  make check-env             → Vérifie Docker + Compose v2"
	@echo ""