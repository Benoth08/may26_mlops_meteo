# Architecture

## Vue d'ensemble

```
   Lots CSV simulés
         |
         v
 [weather_csv_integration] ----> PostgreSQL météo
         |                              |
         |                              v
         |                    [weather_preprocessing] ----> dataset.joblib
         |                                                        |
         |                                                        v
         |                                             [weather_models] ----> MLflow Registry
         |                                                        |                 |
         |                                                        v                 v
         |                                                  model.joblib         promotion
         |                                                        |
         v                                                        v
   [weather_drift]  <--------------------------------  API ----> NGINX ----> Prometheus / Grafana
```

Deux fichiers Compose. `docker-compose.yml` porte la chaîne de production
des modèles. `docker-compose.monitoring.yml` ajoute la mise à l'échelle de
l'API et la supervision, en surcouche :

```bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

## Conteneurs

### Chaîne de base — `docker-compose.yml`

| Service | Type | Rôle |
|---|---|---|
| `postgres-weather` | persistant | Données météo, source unique du projet |
| `postgres-airflow` | persistant | Base interne d'Airflow |
| `redis` | persistant | File de tâches Celery |
| `airflow-apiserver` | persistant | Interface web |
| `airflow-scheduler` | persistant | Planification et déclenchement |
| `airflow-worker` | persistant | Exécution des tâches |
| `airflow-dag-processor` | persistant | Analyse des DAG |
| `airflow-triggerer` | persistant | Tâches différées |
| `airflow-init` | éphémère | Migration de la base, import des variables |
| `airflow-cli` | à la demande | Console Airflow, profil `debug` |
| `api-weather` | persistant | API FastAPI |
| `data-integration` | image de tâche | Chargement des lots en base |
| `data-preprocessing` | image de tâche | Préparation du jeu de données |
| `models` | image de tâche | Entraînement, évaluation, promotion |
| `monitoring` | image de tâche | Détection de dérive |

Une **image de tâche** n'est pas un service qui tourne : le service existe
uniquement pour que `docker compose build` construise l'image. C'est Airflow
qui lance ensuite un conteneur éphémère à partir d'elle, via `DockerOperator`.

### Surcouche de supervision — `docker-compose.monitoring.yml`

| Service | Rôle |
|---|---|
| `api-weather` … `api-weather-6` | Six instances de l'API derrière le répartiteur |
| `nginx` | Terminaison TLS, répartition de charge, certificat auto-signé généré au démarrage |
| `nginx-exporter` | Expose les métriques `stub_status` à Prometheus |
| `prometheus` | Collecte toutes les 15 s, 8 cibles |
| `grafana` | Deux tableaux de bord, approvisionnés depuis `grafana/` |

Ports publiés : 80 et 443 (NGINX), 3000 (Grafana), 9090 (Prometheus),
8080 (Airflow), 5432 (PostgreSQL météo).

## Les six DAG

| DAG | Cadence | Rôle |
|---|---|---|
| `weather_rawdata_producer` | manuel | Découpe le CSV en lots dans `data/incoming`, simule l'arrivée de données |
| `weather_csv_integration` | chaque minute | Charge les lots en base, archive le fichier, se relance s'il en reste |
| `weather_preprocessing` | toutes les 20 min | Extraction, construction des variables, normalisation |
| `weather_training_trigger` | chaque heure | Décide s'il faut réentraîner, et déclenche `weather_models` |
| `weather_models` | 1er du mois à 3 h | Recherche d'hyperparamètres, entraînement, évaluation, promotion |
| `weather_drift` | lundi à 4 h | Surveillance des données et des prédictions |

Les heures sont en temps universel ; l'interface Airflow les affiche en heure
locale, d'où un décalage apparent d'une ou deux heures.

`weather_training_trigger` est le seul DAG qui ne fait que décider. Il compare
le nombre de lignes en base au dernier entraînement enregistré dans
`model_training_status`, et déclenche `weather_models` si 1 000 lignes sont
arrivées ou si 24 heures se sont écoulées. Les seuils sont dans `params.yml`.

## Chemin des données

1. Le producteur dépose des lots CSV dans `data/incoming`.
2. L'intégration les charge dans `weather_data_raw` puis archive le fichier.
   Le chargement est idempotent : relancer ne crée pas de doublon.
3. Le prétraitement lit la table, construit les variables, et produit
   `dataset.joblib` et `preprocessor.joblib`.
4. L'entraînement produit un modèle candidat.
5. L'évaluation calcule les métriques dans `metrics/scores.json` et enregistre
   le modèle dans le Registry MLflow, hébergé sur DagsHub.
6. La promotion compare le candidat au modèle en production et ne le promeut
   que s'il est meilleur. Le modèle promu devient `models/model.joblib` et
   reçoit l'alias `champion`.
7. L'API sert `models/model.joblib`.
8. La surveillance lit la base et le modèle pour détecter une dérive.

## Pipeline DVC

Cinq étapes déclarées dans `dvc.yaml`, reproductibles par `dvc repro` :

| Étape | Produit |
|---|---|
| `export_dataset` | `data/raw/weather.parquet` |
| `make_dataset` | `data/processed/dataset.joblib`, `models/preprocessor.joblib` |
| `grid_search` | `models/best_params.joblib` |
| `train_model` | le modèle candidat |
| `evaluate_model` | `data/predictions.csv`, `metrics/scores.json` |

DVC et Airflow ne font pas le même travail. DVC garantit la reproductibilité
et saute ce qui n'a pas changé. Airflow apporte le déclenchement conditionnel,
la reprise d'une étape isolée, l'isolation par conteneur et la visibilité sur
les échecs.

Le remote DVC est le dépôt DagsHub. Les artefacts volumineux n'étant pas dans
git, un clone neuf doit faire `dvc pull` pour les récupérer.

## Configuration

| Fichier | Contenu | Suivi par git |
|---|---|---|
| `.env` | Identifiants des bases, MLflow, API, chemins hôte, tags d'images | non |
| `config/variables.json` | Variables Airflow, 15 entrées | non |
| `params.yml` | Paramètres du pipeline | oui, et par DVC |
| `src/core/settings.py` | Chemins, noms de tables et de colonnes, images | oui |

Règle : les secrets dans `.env`, les réglages métier dans `params.yml`, les
constantes structurelles dans `settings.py`.

Deux fichiers d'exemple sont fournis, `.env.example` et
`config/variables.json.example`. Toute variable ajoutée dans un DAG doit être
déclarée dans l'exemple : une variable Airflow absente fait échouer le DAG au
rendu du template, avant même la première tâche.

## Modèle

LightGBM, avec pondération des classes pour compenser le déséquilibre entre
jours de pluie et jours secs — il pleut environ un jour sur cinq.

Les hyperparamètres sont réglés par recherche sur grille avec validation
croisée temporelle. Le prétraitement est intégré au pipeline scikit-learn, ce
qui évite toute fuite de données entre entraînement et inférence.

Le découpage entraînement / test est chronologique : on apprend sur le passé,
on teste sur les données les plus récentes. Un découpage aléatoire donnerait
au modèle des journées postérieures à celles du test et gonflerait le score.

Métrique de décision : le **F1 sur la classe pluie**. L'exactitude globale ne
convient pas — un modèle qui prédirait « pas de pluie » tous les jours
atteindrait près de 78 % d'exactitude sans aucune utilité.
