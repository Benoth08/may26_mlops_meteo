Prévision météo en Australie (MLOps)
====================================

About
-----

Ce projet est un projet MLOps de bout en bout. L'objectif est de prévoir s'il pleuvra ou non le lendemain en Australie, à partir d'observations météo quotidiennes. C'est un problème de classification binaire : la cible RainTomorrow vaut Oui ou Non.

Les données proviennent du jeu de données "Weather Dataset Rattle Package" (Kaggle). Il rassemble environ dix ans d'observations quotidiennes issues de nombreuses stations météo australiennes (environ 145 000 lignes et 23 variables).

Le projet est réalisé en équipe de trois, avec une répartition des rôles : ingestion et base de données, préparation des données, et modélisation avec suivi des expériences.


Modèle et métriques
-------------------

L'algorithme d'entraînement est LightGBM, un modèle de gradient boosting choisi pour ses bonnes performances sur des données tabulaires. Il est entraîné avec une pondération des classes (class_weight équilibré) pour compenser le déséquilibre entre les jours de pluie et les jours sans pluie. Les hyperparamètres sont réglés par recherche sur grille (GridSearch) avec une validation croisée temporelle (TimeSeriesSplit), et le prétraitement est intégré au pipeline scikit-learn pour éviter toute fuite de données. Le découpage train / test est chronologique : on apprend sur le passé et on teste sur les données les plus récentes.

Plusieurs métriques sont calculées et suivies, même si le F1 reste le critère de décision :

- F1-score (classe pluie) : la métrique principale du projet.
- Accuracy : le taux global de bonnes prévisions.
- ROC-AUC : la capacité du modèle à séparer les deux classes.
- PR-AUC : le compromis précision / rappel, utile quand la classe pluie est rare.
- Précision (pluie) : la part de bonnes prévisions parmi les jours annoncés pluvieux.
- Rappel (pluie) : la part des jours de pluie réellement détectés.


App Architecture
----------------

Le projet est découpé en plusieurs conteneurs Docker. L'idée est simple : chaque conteneur a une seule responsabilité. Cela rend l'ensemble plus facile à comprendre, à redémarrer et à faire évoluer, et garantit que tout le monde travaille dans le même environnement.

Un conteneur persistant tourne en continu. Un conteneur éphémère exécute une tâche puis s'arrête.

Les conteneurs principaux du projet :

- PostgreSQL météo (persistant) : la base de données qui stocke les observations météo. C'est la source unique des données pour tout le reste du projet, elle doit rester disponible en permanence.
- Intégration des données (éphémère) : charge le fichier CSV dans la base PostgreSQL, puis s'arrête. C'est une tâche ponctuelle, pas un service.
- API (persistant) : un service FastAPI qui reste actif pour répondre aux requêtes à tout moment. Il permet de consulter les données et de demander une prévision de pluie.

L'orchestration est assurée par Airflow, qui planifie et lance automatiquement les étapes du projet (par exemple l'ingestion). Airflow est un système distribué : sa stack (base interne, file Redis, planificateur, worker, serveur web) est persistante, sauf le conteneur d'initialisation qui est éphémère (il prépare Airflow au premier démarrage puis s'arrête).

En complément, une petite interface Streamlit sert de vitrine de démonstration : elle permet de saisir des conditions météo dans un formulaire et d'afficher la prévision de pluie, sans appeler l'API à la main. C'est une couche de présentation optionnelle, pas un élément du coeur du projet.

À noter : il n'y a pas encore de conteneur Docker pour l'entraînement. Aujourd'hui, l'entraînement du modèle se lance en local (avec run_ml.py ou les quatre scripts de src), pas dans Docker. Les conteneurs d'entraînement, d'évaluation et de validation du modèle existent uniquement sous forme de lignes commentées dans le docker-compose : ils sont prévus mais pas encore construits. Les quatre scripts modulaires sont justement faits pour cela : chacun deviendra un conteneur éphémère (préparation, entraînement, évaluation), orchestré par Airflow ou via la pipeline DVC.

En résumé : l'application météo tourne sur trois conteneurs principaux (base de données persistante, ingestion éphémère, API persistante), Airflow ajoute le moteur d'orchestration, et Streamlit offre une interface de démonstration.


Repository Tree
---------------

```
may26_mlops_meteo/
├── README.md                 <- Ce fichier
├── LICENSE
├── requirements.txt          <- Dépendances Python (versions figées)
├── makefile                  <- Commandes Docker (build, start, stop, reset)
├── docker-compose.yml        <- Orchestration de tous les conteneurs
├── .env.example              <- Modèle de variables d'environnement (secrets)
├── .gitignore
├── pytest.ini                <- Configuration des tests
│
├── run_ml.py                 <- Chaîne ML complète en un seul script
├── inference_api.py          <- API d'inférence (prévision de pluie)
├── streamlit_app.py          <- Interface web de démonstration (Streamlit)
│
├── .github/
│   └── workflows/
│       └── python-app.yml    <- Intégration continue (lint + tests)
│
├── config/
│   └── variables.json.example  <- Variables Airflow (modèle)
│
├── dags/
│   └── weather_integration_dag.py  <- DAG Airflow : ingestion planifiée
│
├── data/                     <- Données (non versionnées sur Git)
│   ├── raw/                  <- CSV brut weatherAUS
│   └── processed/            <- Jeux préparés (train / test)
│
├── logs/                     <- Logs d'exécution
├── models/                   <- Modèles entraînés (non versionnés sur Git)
├── metrics/                  <- Scores du modèle (scores.json)
├── notebooks/                <- Notebooks d'exploration
├── references/               <- Dictionnaire de données, notes
├── reports/
│   └── figures/              <- Graphiques générés
│
└── src/                      <- Code source
    ├── data/
    │   └── make_dataset.py       <- Étape 1 : split chronologique train / test
    ├── features/
    │   └── build_features.py     <- Prétraitement et feature engineering
    ├── models/
    │   ├── grid_search.py        <- Étape 2 : réglage des hyperparamètres
    │   ├── train_model.py        <- Étape 3 : entraînement du modèle final
    │   ├── evaluate_model.py     <- Étape 4 : évaluation sur le test
    │   └── predict_model.py      <- Prédiction (à compléter)
    ├── visualization/
    │   └── visualize.py          <- Graphiques
    ├── scripts/
    │   ├── core/
    │   │   └── logger.py         <- Logger commun
    │   └── data_integration/
    │       └── weather_loader.py <- Chargement du CSV vers PostgreSQL
    ├── sql/
    │   └── weather/
    │       └── init_weather.sql  <- Création de la table PostgreSQL
    └── dockerfiles/              <- Un Dockerfile par service
        ├── airflow/             <- Image Airflow
        ├── api_weather/         <- Image de l'API
        └── data_integration/    <- Image du job d'ingestion
```

Note : les dossiers data, models, metrics et logs ne sont pas versionnés sur Git (voir .gitignore). Les données et les modèles seront partagés via DVC et DagsHub en Phase 2.


Étapes suivantes (Phase 2)
--------------------------

La Phase 2 ajoutera le suivi des expériences avec MLflow et le versioning des données et des modèles avec DVC. Le dépôt DagsHub, qui servira à la fois de stockage distant pour DVC et de serveur MLflow partagé entre les membres de l'équipe, est déjà connecté au projet. C'est aussi à ce moment que les conteneurs d'entraînement et d'évaluation seront construits, à partir des scripts existants.
