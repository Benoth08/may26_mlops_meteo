DVC — Versioning des données et du modèle
==========================================

Ce document décrit la mise en place de [DVC](https://dvc.org) sur le projet (Phase 2, cf. README.md), le rôle de chaque fichier, et toutes les commandes pour l'installer, reproduire la pipeline, et pousser/tirer les données depuis DagsHub.

Pourquoi DVC
------------

`data/`, `models/` et les gros artefacts ne sont **pas** versionnés dans Git (voir `.gitignore`) : trop volumineux, et Git n'est pas fait pour ça. DVC prend le relais :

- il garde en Git de petits fichiers `.dvc` / `dvc.lock` qui contiennent seulement les **hash** des données et modèles ;
- le contenu réel (CSV, `.joblib`) est stocké dans un **remote** DagsHub, séparé du dépôt Git ;
- une **pipeline** (`dvc.yaml`) définit les étapes de traitement avec leurs dépendances, ce qui permet de savoir automatiquement quoi ré-exécuter quand un fichier change (`dvc repro`).

Architecture de la pipeline
----------------------------

```
data/raw/weatherAUS.csv (suivi par dvc add)
        │
        ▼
  ┌───────────────┐
  │ make_dataset  │  src/data/make_dataset.py + src/features/build_features.py
  └───────────────┘
        │  outs: data/processed/dataset.joblib, models/preprocessor.joblib
        ├──────────────────────┐
        ▼                      ▼
  ┌───────────────┐      (utilisé aussi par train_model et evaluate_model)
  │ grid_search   │  src/models/grid_search.py
  └───────────────┘
        │  outs: models/best_params.joblib
        ▼
  ┌───────────────┐
  │ train_model   │  src/models/train_model.py
  └───────────────┘
        │  outs: models/model.joblib
        ▼
  ┌────────────────┐
  │ evaluate_model  │  src/models/evaluate_model.py
  └────────────────┘
        outs: data/predictions.csv
        metrics: metrics/scores.json (suivi directement par Git, cache: false)
```

Voir `dvc.yaml` pour la définition exacte (commande, `deps`, `outs` de chaque stage) et `dvc.lock` pour les hash figés de la dernière exécution.

Fichiers ajoutés par DVC
-------------------------

| Fichier | Suivi par | Rôle |
|---|---|---|
| `.dvc/config` | Git | remote DagsHub (URL) |
| `.dvc/config.local` | **jamais** (gitignore) | identifiants DagsHub (user/token) |
| `.dvcignore` | Git | fichiers ignorés par DVC (comme `.gitignore` pour Git) |
| `dvc.yaml` | Git | définition des 4 stages de la pipeline |
| `dvc.lock` | Git | hash des deps/outs de la dernière exécution réussie |
| `data/raw/weatherAUS.csv.dvc` | Git | pointeur (hash + taille) vers le CSV brut |
| `data/.gitignore`, `data/raw/.gitignore` | Git | générés automatiquement par DVC pour ignorer les gros fichiers |
| `metrics/scores.json` | Git (directement) | petit JSON, pratique à diff dans les PR |
| `data/raw/weatherAUS.csv`, `data/processed/*.joblib`, `models/*.joblib`, `data/predictions.csv` | **DVC uniquement** | contenu réel, stocké sur DagsHub |

Installation (une fois par machine)
------------------------------------

```bash
./scripts/dvc/setup.sh
```

Crée `.venv/` et installe `dvc` + les dépendances minimales pour exécuter la pipeline (`numpy`, `pandas`, `scikit-learn`, `lightgbm`, `joblib`). Ne pas utiliser `pip install -r requirements.txt` sur Windows : ce fichier échoue à cause de `uvloop`, qui ne supporte pas Windows (bug préexistant, indépendant de DVC — cf. section Limitations).

Configurer l'accès à DagsHub (une fois par machine)
-------------------------------------------------------

```bash
./scripts/dvc/configure-remote.sh
```

Demande votre utilisateur DagsHub et un token (généré sur `https://dagshub.com/user/settings/tokens`), et les stocke dans `.dvc/config.local` — **jamais commité** (voir `.dvc/.gitignore`). Chaque développeur exécute ce script une fois avec son propre token.

Reproduire la pipeline
------------------------

```bash
./scripts/dvc/repro.sh
```

Ce script :
1. affiche le DAG (`dvc dag`) ;
2. exécute `dvc repro` — ne relance que les stages dont une dépendance a changé (code ou données) ; si rien n'a changé, chaque stage affiche `didn't change, skipping` ;
3. affiche `dvc status` (doit être vide si tout est à jour) ;
4. affiche les métriques courantes (`dvc metrics show`).

Exemple de sortie sur une pipeline déjà à jour :

```
'data\raw\weatherAUS.csv.dvc' didn't change, skipping
Stage 'make_dataset' didn't change, skipping
Stage 'grid_search' didn't change, skipping
Stage 'train_model' didn't change, skipping
Stage 'evaluate_model' didn't change, skipping
Data and pipelines are up to date.

Path                 accuracy    f1       pr_auc    precision_pluie    recall_pluie    roc_auc
metrics\scores.json  0.81016     0.64171  0.72178   0.55435            0.76178         0.87673
```

Pousser / récupérer les données (DagsHub)
--------------------------------------------

```bash
# Envoyer les données/modèles produits localement vers DagsHub
.venv/Scripts/python -m dvc push        # Linux/macOS : .venv/bin/python

# Récupérer les données/modèles déjà versionnés (ex: après un git pull / git clone)
.venv/Scripts/python -m dvc pull
```

`dvc push`/`dvc pull` nécessitent que `./scripts/dvc/configure-remote.sh` ait été exécuté au préalable.

Autres commandes utiles
--------------------------

```bash
# Voir ce qui a changé depuis la dernière exécution, sans rien relancer
dvc status

# Comparer les métriques entre deux commits/branches
dvc metrics diff main feature/dvc-setup

# Voir le graphe de dépendances de la pipeline
dvc dag

# Re-suivre un nouveau fichier de données brut (hors pipeline)
dvc add data/raw/nouveau_fichier.csv
```

(Remplacer `dvc` par `.venv/Scripts/python -m dvc` si l'exécutable `dvc` n'est pas sur le PATH.)

Détail des 4 stages
----------------------

| Stage | Commande | Dépendances | Sorties |
|---|---|---|---|
| `make_dataset` | `python -m src.data.make_dataset` | `data/raw/weatherAUS.csv`, `src/data/make_dataset.py`, `src/features/build_features.py` | `data/processed/dataset.joblib`, `models/preprocessor.joblib` |
| `grid_search` | `python -m src.models.grid_search` | `data/processed/dataset.joblib`, `models/preprocessor.joblib`, `src/models/grid_search.py` | `models/best_params.joblib` |
| `train_model` | `python -m src.models.train_model` | `data/processed/dataset.joblib`, `models/preprocessor.joblib`, `models/best_params.joblib`, `src/models/train_model.py` | `models/model.joblib` |
| `evaluate_model` | `python -m src.models.evaluate_model` | `data/processed/dataset.joblib`, `models/model.joblib`, `src/models/evaluate_model.py` | `data/predictions.csv`, `metrics/scores.json` (metric) |

Limitations connues
----------------------

- **`requirements.txt` cassé sous Windows** : `uvloop==0.22.1` n'a pas de wheel Windows et ne supporte pas cet OS. Pré-existant, indépendant de DVC. `scripts/dvc/setup.sh` contourne le problème en installant seulement les paquets nécessaires à la pipeline ML, pas la stack API/Airflow complète.
- **Encodage console Windows** : les scripts `src/` affichent des emojis (`✅`) qui font planter `print()` sous la console Windows par défaut (cp1252). `scripts/dvc/repro.sh` fixe `PYTHONIOENCODING=utf-8` avant d'appeler `dvc repro`.
- **`python` sur le PATH** : DVC exécute les commandes des stages (`python -m ...`) via le PATH système, pas via l'interpréteur qui a lancé `dvc`. Sur une machine avec plusieurs Python installés, `scripts/dvc/repro.sh` préfixe le PATH avec `.venv/Scripts` pour être sûr d'utiliser le bon interpréteur (celui qui a `lightgbm`/`scikit-learn`).

Historique des commandes de mise en place
--------------------------------------------

Pour référence/audit — ce qui a été exécuté pour initialiser DVC sur ce projet (déjà fait, pas à rejouer) :

```bash
git worktree add ../may26_dvc-setup -b feature/dvc-setup origin/main

python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install dvc

dvc init
dvc remote add origin https://dagshub.com/Benoth08/may26_mlops_meteo.dvc
dvc remote modify origin --local auth basic

dvc add data/raw/weatherAUS.csv

dvc stage add -n make_dataset \
  -d src/data/make_dataset.py -d src/features/build_features.py -d data/raw/weatherAUS.csv \
  -o data/processed/dataset.joblib -o models/preprocessor.joblib \
  python -m src.data.make_dataset

dvc stage add -n grid_search \
  -d src/models/grid_search.py -d data/processed/dataset.joblib -d models/preprocessor.joblib \
  -o models/best_params.joblib \
  python -m src.models.grid_search

dvc stage add -n train_model \
  -d src/models/train_model.py -d data/processed/dataset.joblib -d models/preprocessor.joblib -d models/best_params.joblib \
  -o models/model.joblib \
  python -m src.models.train_model

dvc stage add -n evaluate_model \
  -d src/models/evaluate_model.py -d data/processed/dataset.joblib -d models/model.joblib \
  -o data/predictions.csv \
  -M metrics/scores.json \
  python -m src.models.evaluate_model

dvc repro
```

Prochaines étapes (Phase 2, cf. README.md)
---------------------------------------------

- Suivi des expériences avec MLflow (une branche `feat/mlflow-tracking` existe déjà côté équipe avec un premier essai dans `run_ml.py`).
- Construction des conteneurs Docker d'entraînement/évaluation à partir des scripts `src/` (actuellement commentés dans `docker-compose.yml`).
- `dvc push` vers DagsHub une fois le remote configuré par chaque développeur.
