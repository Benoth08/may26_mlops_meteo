# =============================================================================
# Makefile — Projet Weather MLOps
# Gestion complète des dockers : build, start, stop, reset, purge
# =============================================================================

DOCKER_COMPOSE = docker compose
PROJECT_NAME = weather-mlops

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
	$(DOCKER_COMPOSE) build
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
	$(DOCKER_COMPOSE) stop
	@echo "Services arrêtés."

## Arrêt + suppression des conteneurs
down:
	$(DOCKER_COMPOSE) down
	@echo "Conteneurs supprimés."

# -----------------------------------------------------------------------------
# RESET COMPLET
# -----------------------------------------------------------------------------

## Réinstalle Docker pour nettoyer tous les repertoires (layers...)
reset-docker:
	sudo systemctl stop docker
	sudo rm -rf /var/lib/docker
	sudo systemctl start docker


## Reset total : stop + purge volumes + purge images + rebuild + restart
reset: check-env
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
	$(DOCKER_COMPOSE) build data-integration
	@echo "data-integration reconstruit."

## Rebuild uniquement data-preprocessing
rebuild-preprocessing: check-env
	$(DOCKER_COMPOSE) build data-preprocessing
	@echo "data-preprocessing reconstruit."

## Rebuild uniquement models
rebuild-models: check-env
	$(DOCKER_COMPOSE) build models
	@echo "models reconstruit."

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
	@echo "  make check-env             → Vérifie Docker + Compose v2"
	@echo ""