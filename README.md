Prévision de pluie en Australie — Projet MLOps
==============================================

Prédire s'il pleuvra le lendemain en Australie, et surtout industrialiser
cette prédiction de bout en bout : de l'arrivée d'une donnée brute
jusqu'au réentraînement automatique du modèle en production.

Classification binaire, cible `RainTomorrow`. Jeu de données Kaggle
*Weather Dataset Rattle Package* : 145 460 observations, 23 variables,
49 stations, de novembre 2007 à juin 2017.

Projet d'équipe réalisé dans le cadre de la formation MLOps
DataScientest.


Chaîne complète
---------------

```
 Lots de données (simulation d'arrivée)
            |
            v
   [ Ingestion ]  ---------->  PostgreSQL
            |                       |
            |                       v
            |              [ Préprocessing ]  --->  dataset.joblib
            |                                             |
            |                                             v
            |                              [ Entraînement ]  ---> MLflow Registry
            |                                             |            |
            |                                             v            v
            |                                      model.joblib    Promotion
            |                                             |
            v                                             v
   [ Surveillance ]  <----------------------------  API de prédiction
            |                                             |
            +-----> déclenche le réentraînement           v
                                                    NGINX (HTTPS)
                                                          |
                                                    Prometheus
                                                          |
                                                      Grafana
```


Pile technique
--------------

| Besoin | Outil |
|---|---|
| Orchestration | Apache Airflow 3.2, six DAG |
| Conteneurisation | Docker Compose |
| Stockage | PostgreSQL 16 |
| Versionnement des données | DVC |
| Suivi des expériences | MLflow sur DagsHub |
| API | FastAPI, découpée en routeurs |
| Proxy, TLS, répartition de charge | NGINX |
| Métriques | Prometheus |
| Tableaux de bord | Grafana |
| Détection de dérive | Evidently 0.7 |
| Modèle | LightGBM |
| Qualité du code | flake8, pytest, GitHub Actions |


Démarrage
---------

```bash
git clone <url> /home/ubuntu/projet_weather
cd /home/ubuntu/projet_weather

cp .env.example .env                                  # puis remplir
cp config/variables.json.example config/variables.json

make install
```

Le chemin `/home/ubuntu/projet_weather` est imposé : les DAG lancent des
conteneurs voisins, dont les montages sont interprétés par le démon
Docker de l'hôte.

Pour ajouter la couche de supervision :

```bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

| Service | Adresse |
|---|---|
| Airflow | http://localhost:8080 |
| API via NGINX | https://localhost, documentation sur `/docs` |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Le certificat est auto-signé : le navigateur affiche un avertissement, le
trafic est néanmoins chiffré.

Détail des commandes, accès et dépannage : `docs/exploitation.md`.


Les six DAG
-----------

| DAG | Cadence | Rôle |
|---|---|---|
| `weather_rawdata_producer` | manuel | Découpe le jeu de données en lots, simule une arrivée continue |
| `weather_csv_integration` | chaque minute | Détecte un fichier, le charge, archive, se relance s'il en reste |
| `weather_preprocessing` | toutes les 20 min | Extraction, prétraitement, normalisation |
| `weather_training_trigger` | chaque heure | Déclenche l'entraînement si les conditions sont réunies |
| `weather_models` | 1er du mois à 3h | Grid search, entraînement, évaluation, promotion |
| `weather_drift` | lundi à 4h | Surveillance des données et des prédictions |

Les heures sont en temps universel.


Modèle
------

LightGBM avec pondération des classes, hyperparamètres réglés par
recherche sur grille et validation croisée temporelle. Le prétraitement
est intégré au pipeline scikit-learn, ce qui exclut toute fuite de
données. Découpage chronologique : on apprend sur le passé, on teste sur
le récent.

Métrique de décision : le F1 sur la classe pluie. Avec 78 % de jours
secs, l'accuracy récompenserait un modèle qui prédit toujours « pas de
pluie ».


API
---

| Routeur | Endpoints | Accès |
|---|---|---|
| `meta` | `/`, `/health`, `/last-import` | public |
| `predict` | `/predict`, `/predict/batch`, `/predict/from-db` | authentifié |
| `weather` | `/weather`, `/weather/features`, `/weather/all-features` | authentifié |

Authentification HTTP Basic, mot de passe stocké sous forme d'empreinte
bcrypt, comparaison de l'identifiant en temps constant. `/health` reste
public pour les contrôles de santé.

L'API lit `models/model.joblib`, partagé avec les conteneurs ML : une
promotion est prise en compte à la requête suivante, sans redémarrage.


Surveillance et réentraînement
------------------------------

Quatre signaux, du plus précoce au plus fiable :

| Signal | Besoin de la vérité terrain | Délai |
|---|---|---|
| Qualité des données | non | immédiat |
| Dérive des données | non | hebdomadaire |
| Dérive des prédictions | non | hebdomadaire |
| Chute de performance (F1) | oui | au réentraînement |

En production, on ne sait pas s'il a plu avant le lendemain : le F1 ne
peut pas être calculé en temps réel. La dérive des prédictions comble ce
trou.

**Règle de décision** : la qualité prime sur la dérive. Des doublons ou
une hausse des valeurs manquantes signalent un pipeline cassé, pas un
monde qui change ; réentraîner reviendrait à apprendre la corruption.

**Deux déclencheurs** : cadence mensuelle comme filet de sécurité, et
détection de dérive.

**Promotion conditionnelle** : le nouveau modèle n'est promu que s'il bat
celui en production, les deux étant évalués sur le **même jeu de test**.
Le modèle en production est rechargé depuis le Registry et réévalué sur
les données courantes. Sans cela, on comparerait deux scores calculés sur
des jeux différents, ce qui se produit à chaque arrivée de données.

Détail de la calibration des seuils : `docs/surveillance.md`.


Reproductibilité
----------------

`dvc.yaml` décrit le pipeline en cinq étapes. Modifier un paramètre dans
`params.yml` invalide automatiquement les étapes concernées, et elles
seules.

MLflow conserve pour chaque exécution les paramètres, les métriques, les
artefacts et le modèle enregistré.

| Fichier | Contenu | Versionné |
|---|---|---|
| `.env` | Identifiants bases, MLflow, API | non |
| `config/variables.json` | Variables Airflow | non |
| `params.yml` | Paramètres métier du pipeline | oui |
| `src/core/settings.py` | Chemins, noms de tables et colonnes | oui |


Validation de bout en bout
--------------------------

Un script automatise la vérification de la chaîne complète :

```powershell
.\run_e2e_tests.ps1                        # services et contrôles rapides
.\run_e2e_tests.ps1 -FullPipeline          # + pipeline DVC complet
.\run_e2e_tests.ps1 -FullPipeline -Promote # + promotion du modèle
```

Chaque composant est rapporté en PASS, FAIL ou SKIP. Le script est écrit
en PowerShell et requiert Docker Desktop.


Intégration continue
--------------------

GitHub Actions sur chaque pull request : installation des dépendances,
analyse flake8 bloquante, puis tests pytest.


Structure du dépôt
------------------

```
├── dags/                      Les six DAG Airflow
├── data/
│   ├── incoming/              Lots en attente d'ingestion
│   ├── raw/                   Données sources
│   ├── processed/             Jeu de données préparé
│   └── archive/               Lots déjà ingérés
├── docs/                      Documentation détaillée
├── grafana/                   Tableaux de bord provisionnés
├── models/                    Modèle servi par l'API
├── nginx/                     Configuration du proxy et certificats
├── notebooks/                 Exploration
├── prometheus/                Configuration de la collecte
├── reports/                   Rapports de dérive
├── src/
│   ├── api_weather/           API FastAPI et ses routeurs
│   ├── core/                  Configuration, journalisation, paramètres
│   ├── data_integration/      Ingestion et production de lots
│   ├── data_preprocessing/    Préparation du jeu de données
│   ├── features/              Construction des variables
│   ├── models/                Entraînement, évaluation, promotion
│   ├── monitoring/            Détection de dérive
│   ├── dockerfiles/           Une image par service
│   └── sql/                   Schéma de la base
├── tests/
├── docker-compose.yml         Socle applicatif
├── docker-compose.monitoring.yml  NGINX, instances, Prometheus, Grafana
├── dvc.yaml                   Pipeline versionné
├── params.yml                 Paramètres du pipeline
├── run_e2e_tests.ps1          Validation de bout en bout
└── makefile                   Commandes courantes
```


Documentation
-------------

| Document | Contenu |
|---|---|
| `docs/architecture.md` | Conteneurs, DAG, chemin des données, configuration |
| `docs/surveillance.md` | Les quatre signaux, calibration des seuils, promotion |
| `docs/exploitation.md` | Installation, commandes, pannes courantes |
| `DVC.md` | Versionnement des données |
| `MLFLOW.md` | Suivi des expériences |

La documentation de l'API est générée automatiquement sur `/docs`.


Limites connues
---------------

- **Certificat auto-signé** : chiffre le trafic mais ne prouve pas
  l'identité du serveur, faute de nom de domaine. Let's Encrypt ne
  délivre pas de certificat pour une adresse IP.
- **Un seul compte d'API**, sans gestion de rôles.
- **Secrets dans un fichier**, suffisant en développement.
- **Métriques de qualité du modèle** issues de l'évaluation à
  l'entraînement : elles sont figées entre deux réentraînements, la
  vérité terrain n'étant pas disponible en temps réel.
- **Calibration des seuils** établie sur cinq paires d'années
  consécutives.
- **Données figées** : la simulation démontre le mécanisme, elle ne
  remplace pas un flux réel.
