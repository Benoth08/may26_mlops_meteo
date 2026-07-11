MLflow — Suivi des expériences
================================

Ce document décrit le suivi des expériences avec [MLflow](https://mlflow.org), hébergé sur le serveur **DagsHub** partagé par l'équipe (même dépôt que pour DVC, cf. `DVC.md`).

Pourquoi MLflow
------------------

À chaque exécution du stage `evaluate_model` de la pipeline DVC (voir `DVC.md`), le script logue automatiquement dans MLflow :

- les **hyperparamètres** retenus par le grid search (`models/best_params.joblib`) ;
- les **métriques** (`accuracy`, `f1`, `roc_auc`, `pr_auc`, `precision_pluie`, `recall_pluie`) ;
- le **modèle entraîné** (`models/model.joblib`), sous forme d'artefact MLflow.

Ça permet de comparer les runs entre eux dans l'interface MLflow/DagsHub (onglet **Experiments**), sans avoir à ouvrir manuellement `metrics/scores.json` à chaque fois.

Configuration (une fois par machine)
----------------------------------------

```bash
./scripts/mlflow/configure.sh
```

Demande votre utilisateur et token DagsHub (le même que pour DVC — généré sur `https://dagshub.com/user/settings/tokens`), et les écrit dans `.env` à la racine du projet (**jamais commité**, déjà dans `.gitignore`) :

```
MLFLOW_TRACKING_URI=https://dagshub.com/Benoth08/may26_mlops_meteo.mlflow
MLFLOW_TRACKING_USERNAME=<votre_user>
MLFLOW_TRACKING_PASSWORD=<votre_token>
```

`scripts/dvc/repro.sh` charge automatiquement ce `.env` avant de lancer `dvc repro`. Si `.env` n'existe pas, la pipeline tourne normalement, simplement sans logguer dans MLflow (le code vérifie `MLFLOW_TRACKING_URI` avant d'activer le tracking, voir `src/models/evaluate_model.py`).

Lancer et voir les résultats
--------------------------------

```bash
./scripts/dvc/repro.sh
```

Chaque exécution du stage `evaluate_model` affiche un lien direct vers le run :

```
🏃 View run salty-rat-297 at: https://dagshub.com/Benoth08/may26_mlops_meteo.mlflow/#/experiments/0/runs/...
🧪 View experiment at: https://dagshub.com/Benoth08/may26_mlops_meteo.mlflow/#/experiments/0
```

Ou directement dans l'interface DagsHub du dépôt, onglet **Experiments**.

Model Registry
------------------

En plus du run/experiment, chaque exécution réussie de `evaluate_model` enregistre automatiquement une **nouvelle version** du modèle dans le Model Registry MLflow, sous le nom `weather-rain-model` (visible dans l'onglet **Models** sur DagsHub, ou `mlflow.sklearn.log_model(..., registered_model_name="weather-rain-model")` dans le code).

⚠️ Une version est créée à **chaque** run, y compris pour des essais/tests — pas de filtre automatique sur la qualité du modèle. Pour choisir un modèle à déployer, il faut aller dans l'UI (onglet Models → `weather-rain-model`) et promouvoir manuellement la version voulue vers un stage (`Staging`/`Production`), ou nettoyer périodiquement les versions non retenues.

Limitations connues
----------------------

- **Blocage SSL local** : sur cette machine, la connexion à `dagshub.com` était interceptée (probablement par l'antivirus Norton — voir `JOURNAL_SESSION_DVC.md` section 12 pour le diagnostic complet). Contournement appliqué : `MLFLOW_TRACKING_INSECURE_TLS=true` dans `.env` (équivalent MLflow du `ssl_verify: false` utilisé côté DVC). Si le blocage n'existe pas sur ta machine, tu peux retirer cette ligne.
- **Sérialisation du modèle** : `mlflow.sklearn.log_model` utilise par défaut le format `skops` (plus sûr que `pickle`), qui ne reconnaît pas encore nativement les objets LightGBM (`UntrustedTypesFoundException`). Le code utilise donc explicitement `serialization_format="cloudpickle"`.

Prochaines étapes possibles
--------------------------------

- Comparer plusieurs runs dans l'UI MLflow pour choisir un modèle à déployer.
- Promouvoir une version du Model Registry vers `Production`, pour préparer le service d'inférence (`inference_api.py`).
- Logger aussi les métriques intermédiaires du `grid_search` (actuellement seul `evaluate_model` logue dans MLflow).
- Filtrer l'enregistrement automatique (ex: seulement si `f1` dépasse un seuil), pour éviter d'accumuler des versions de tests.
