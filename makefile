# =============================================================================
# Makefile — Projet Weather MLOps
# Gestion complète des dockers : build, start, stop, reset, purge
# =============================================================================

DOCKER_COMPOSE = docker compose
PROJECT_NAME = weather-mlops

# -----------------------------------------------------------------------------
# COMMANDES PRINCIPALES
# -----------------------------------------------------------------------------

## Installation complète : build + up
install:
	$(DOCKER_COMPOSE) build
	$(DOCKER_COMPOSE) up -d
	@echo "Stack démarrée."

## Démarrage simple
start:
	$(DOCKER_COMPOSE) up -d
	@echo "Services démarrés."

## Arrêt propre
stop:
	$(DOCKER_COMPOSE) stop
	@echo "Services arrêtés."

## Arrêt + suppression des conteneurs
down:
	$(DOCKER_COMPOSE) down
	@echo "Conteneurs supprimés."

# -----------------------------------------------------------------------------
# RESET COMPLET
# -----------------------------------------------------------------------------

## Reset total : stop + purge volumes + purge images + rebuild + restart
reset:
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -af
	$(DOCKER_COMPOSE) build --no-cache
	$(DOCKER_COMPOSE) up -d
	@echo "Reset complet terminé."

# -----------------------------------------------------------------------------
# OUTILS
# -----------------------------------------------------------------------------

## Purge des logs Airflow + Weather
clean-logs:
	rm -rf logs/*
	mkdir -p logs
	@echo "Logs nettoyés."

## Rebuild uniquement Airflow
rebuild-airflow:
	$(DOCKER_COMPOSE) build airflow-apiserver airflow-scheduler airflow-worker airflow-triggerer airflow-dag-processor
	@echo "Airflow reconstruit."

## Rebuild uniquement data-integration
rebuild-integration:
	$(DOCKER_COMPOSE) build data-integration
	@echo "data-integration reconstruit."

## Rebuild API Weather
reset-api:
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
	@echo "  make install            → Build + start"
	@echo "  make start              → Start services"
	@echo "  make stop               → Stop services"
	@echo "  make down               → Stop + remove containers"
	@echo "  make reset              → Reset complet (purge + rebuild)"
	@echo ""
	@echo "  make clean-logs         → Purge logs"
	@echo "  make rebuild-airflow    → Rebuild Airflow"
	@echo "  make rebuild-integration→ Rebuild data-integration"
	@echo "  make reset-api        	 → Rebuild API Weather"
	@echo ""

