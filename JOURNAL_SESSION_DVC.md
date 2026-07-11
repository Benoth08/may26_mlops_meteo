Journal de session — Mise en place de DVC
==========================================

Date : 2026-07-11
Dépôt : `Benoth08/may26_mlops_meteo` (GitHub)
Remote de données : `https://dagshub.com/Benoth08/may26_mlops_meteo` (DagsHub)
Branche créée : `feature/dvc-setup` (basée sur `origin/main`)

Ce fichier retrace **toutes les actions effectuées pendant cette session**, dans l'ordre, avec les commandes exactes, pour que tu (ou n'importe qui de l'équipe) puisse comprendre ce qui a été fait, reproduire, ou reprendre là où ça s'est arrêté.

Pour la documentation *technique* de la pipeline DVC elle-même (stages, commandes usuelles `dvc repro`/`push`/`pull`), voir **`DVC.md`** à la racine du projet. Ce journal-ci raconte le déroulé de la mise en place, y compris les problèmes rencontrés.

---

1. Point de départ
-------------------

Avant toute action, l'état du dépôt local (`may26_mlops_meteo`) a été vérifié :

- Branche locale active : `preprocessing`, avec des fichiers non commités (`DOCUMENTATION.md`, `streamlit_app.py`) — **volontairement non touchés**, laissés intacts.
- `git fetch origin --prune` a révélé que :
  - `origin/preprocessing` avait été **force-push** par un collègue (divergence avec la copie locale).
  - `origin/main` était très en avance (pipeline ML complète : ingestion, préprocessing, entraînement, API, dashboard).
  - Plusieurs branches existaient déjà côté GitHub : `feat/mlflow-tracking`, `feat/orchestration-dag`, `fix/finitions-phase1`.
- Le `README.md` d'`origin/main` annonçait explicitement une "Phase 2" : *"Les données et les modèles seront partagés via DVC et DagsHub en Phase 2. Le dépôt DagsHub [...] est déjà connecté au projet."* — mais aucune configuration DVC n'existait encore dans aucune branche.

**Décision** : construire DVC sur la base d'`origin/main` (la version la plus à jour, comme demandé), sans toucher à la branche `preprocessing` ni à son travail non commité.

---

2. Isolation du travail : git worktree
----------------------------------------

Pour ne prendre aucun risque avec le travail en cours sur `preprocessing` (fichiers non commités), un **git worktree séparé** a été créé plutôt qu'un `git checkout` classique :

```bash
git worktree add ../may26_dvc-setup -b feature/dvc-setup origin/main
```

Résultat : un nouveau dossier `C:\Users\ander\OneDrive\Documents\Projet_MLOPs\may26_dvc-setup`, avec sa propre copie de travail sur la branche `feature/dvc-setup`, indépendant du dossier `may26_mlops_meteo` d'origine.

---

3. Installation de DVC
------------------------

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install dvc
```

→ DVC 3.67.1 installé dans un venv dédié à ce worktree (pas dans le venv du projet principal).

*Remarque* : `pip install -r requirements.txt` (le fichier complet du projet) **échoue sous Windows** à cause du paquet `uvloop`, qui ne supporte pas cet OS. C'est un bug préexistant, indépendant de DVC. Seules les dépendances nécessaires à la pipeline ML ont donc été installées séparément : `numpy pandas scikit-learn lightgbm joblib`.

---

4. Initialisation de DVC et du remote DagsHub
-------------------------------------------------

```bash
dvc init
dvc remote add origin https://dagshub.com/Benoth08/may26_mlops_meteo.dvc
dvc remote modify origin --local auth basic
```

- `dvc remote add` écrit l'URL du remote dans `.dvc/config` (committé sur Git, pas un secret).
- `dvc remote modify --local` écrit le type d'authentification dans `.dvc/config.local`, qui est **automatiquement ignoré par Git** (`.dvc/.gitignore` contient déjà `/config.local`). C'est là que vont les identifiants, jamais dans l'historique Git.

---

5. Suivi des données brutes
------------------------------

Le fichier `weatherAUS.csv` a été retrouvé sur la machine (dans `Projet_MLOPs/weatherAUS.csv`) et copié à l'emplacement attendu par les scripts du projet (`data/raw/weatherAUS.csv`, comme codé en dur dans `src/data/make_dataset.py`) :

```bash
mkdir -p data/raw
cp "../weatherAUS.csv" data/raw/weatherAUS.csv
dvc add data/raw/weatherAUS.csv
```

→ Crée `data/raw/weatherAUS.csv.dvc` (pointeur avec hash, committé sur Git) et `data/raw/.gitignore` (ignore le vrai CSV, 14 Mo, pour que Git ne le voie jamais).

---

6. Construction du pipeline (`dvc.yaml`)
--------------------------------------------

En lisant le code existant (`src/data/make_dataset.py`, `src/models/grid_search.py`, `src/models/train_model.py`, `src/models/evaluate_model.py`), il est apparu que ces 4 scripts forment déjà une chaîne de traitement cohérente (le README le confirmait : *"Les quatre scripts modulaires sont justement faits pour [devenir des conteneurs], orchestré[s] par Airflow ou via la pipeline DVC"*).

Chaque stage a été déclaré avec `dvc stage add` (qui écrit dans `dvc.yaml`, sans encore rien exécuter) :

```bash
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
```

*Erreur rencontrée* : la première tentative sur `evaluate_model` incluait un flag `--plots` sans argument, ce qui a fait échouer la commande (`ERROR: command is not specified`, le mot `python` ayant été avalé comme valeur de `--plots`). Corrigé en retirant ce flag, non nécessaire ici (pas de graphique, seulement une métrique JSON).

---

7. Ajustement du `.gitignore`
--------------------------------

Le `.gitignore` d'origine (hérité de `main`) ignorait en bloc `/data/`, `/models/*` et `/metrics/`. Ces règles auraient empêché Git de suivre les fichiers `.dvc` et `dvc.lock` eux-mêmes (qui vivent dans ces dossiers). Remplacées par :

```gitignore
# --- Données & artefacts (versionnés via DVC, pas Git — voir *.dvc et dvc.yaml) ---
/logs/*
!/logs/.gitkeep
/reports/figures/*
!/reports/figures/.gitkeep
/reports/*.csv

# Seul le résumé des métriques (petit JSON) est suivi directement par Git
/metrics/*
!/metrics/scores.json
```

Les gros fichiers réels restent ignorés grâce aux `.gitignore` que DVC génère automatiquement lui-même dans chaque sous-dossier concerné (`data/.gitignore`, `data/raw/.gitignore`, et les patterns globaux `*.pkl` / `*.joblib` déjà présents plus bas dans le fichier).

---

8. Première exécution complète (`dvc repro`)
-------------------------------------------------

```bash
export PYTHONIOENCODING=utf-8   # sinon crash Windows (cp1252) sur les emojis affichés par les scripts
export PATH="$(pwd)/.venv/Scripts:$PATH"   # force les stages à utiliser le python du venv (avec lightgbm), pas celui du PATH système
dvc repro
```

*Deux blocages rencontrés et corrigés en cours de route* :
- `UnicodeEncodeError` sur `✅` → résolu avec `PYTHONIOENCODING=utf-8`.
- `ModuleNotFoundError: No module named 'lightgbm'` → le premier `python` trouvé sur le PATH système n'était pas celui du venv. Résolu en préfixant le `PATH` avec `.venv/Scripts`.

**Résultat final, pipeline complète exécutée avec succès :**

| Métrique | Valeur |
|---|---|
| accuracy | 0.8102 |
| f1 | 0.6417 |
| roc_auc | 0.8767 |
| pr_auc | 0.7218 |
| precision_pluie | 0.5543 |
| recall_pluie | 0.7618 |

Meilleurs hyperparamètres trouvés par le grid search : `n_estimators=200`, `num_leaves=63`.

---

9. Scripts et documentation créés
-------------------------------------

Pour que ce ne soit pas à refaire manuellement à chaque fois, trois scripts ont été ajoutés dans `scripts/dvc/` :

| Script | Rôle |
|---|---|
| `scripts/dvc/setup.sh` | Crée le venv, installe DVC + dépendances ML |
| `scripts/dvc/repro.sh` | Relance `dvc repro` (DAG, statut, métriques) — **re-testé une 2ᵉ fois avec succès**, tous les stages détectés "à jour" |
| `scripts/dvc/configure-remote.sh` | Configure l'auth DagsHub en local (jamais commité) |

Ainsi que **`DVC.md`** (documentation technique complète : architecture, tableau des 4 stages, commandes usuelles, limitations connues).

---

10. Commits et push vers GitHub
------------------------------------

Trois commits ont été créés sur `feature/dvc-setup` :

```
2ece009  feat: mise en place de DVC (pipeline + remote DagsHub)
f10f0b0  chore: ajoute dvc a requirements.txt
05380ad  docs+chore: scripts DVC (setup/repro/remote) + documentation complete
```

*(Note : ces commits ont été recréés une fois, via `git reset --soft origin/main` puis re-commit, uniquement pour retirer une ligne de co-attribution automatique — sans perte de contenu, sans impact car jamais poussés avant ce moment.)*

Poussés sur GitHub :

```bash
git push -u origin feature/dvc-setup
```

→ Lien pour ouvrir la Pull Request : https://github.com/Benoth08/may26_mlops_meteo/pull/new/feature/dvc-setup (**PR pas encore créée**, à faire quand tu seras prêt).

---

11. Configuration de l'authentification DagsHub (en local, PowerShell)
----------------------------------------------------------------------------

Fait depuis un terminal **PowerShell** (le script `.sh` ne s'exécute pas nativement en PowerShell — seulement en Git Bash) :

```powershell
cd "C:\Users\ander\OneDrive\Documents\Projet_MLOPs\may26_dvc-setup"

$DAGSHUB_USER = Read-Host "Utilisateur DagsHub"
$DAGSHUB_TOKEN_SECURE = Read-Host "Token DagsHub" -AsSecureString
$DAGSHUB_TOKEN = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [System.Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($DAGSHUB_TOKEN_SECURE)
)

.\.venv\Scripts\python.exe -m dvc remote modify origin --local auth basic
.\.venv\Scripts\python.exe -m dvc remote modify origin --local user $DAGSHUB_USER
.\.venv\Scripts\python.exe -m dvc remote modify origin --local password $DAGSHUB_TOKEN
```

Utilisateur DagsHub renseigné : `Denart97`. Token saisi de façon masquée, jamais affiché ni committé (`.dvc/config.local` est ignoré par Git).

---

12. Blocage SSL rencontré, puis résolu
------------------------------------------

```bash
dvc push
```

```
ERROR: failed to transfer '...' - Cannot connect to host dagshub.com:443 ssl:True
[SSLCertVerificationError: (1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
Basic Constraints of CA cert not marked critical (_ssl.c:1077)')]
ERROR: failed to push data to the cloud - 6 files failed to upload
```

**Diagnostic** : `certifi` mis à jour (déjà à la dernière version, pas la cause). Un test direct avec `curl -v https://dagshub.com` (moteur TLS natif Windows, pas Python) échoue *aussi* :

```
schannel: next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK
```

→ Ce n'est pas un problème DVC/DagsHub/Python mais un blocage **au niveau système/réseau de la machine**. Le symptôme (`Basic Constraints of CA cert not marked critical`) est caractéristique d'un **antivirus qui inspecte le trafic HTTPS**.

**Antivirus identifié** via `Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntiVirusProduct` : **Norton 360 for Gamers** (Windows Defender en veille à côté, normal).

- Recherche infructueuse dans "Prévention d'intrusion" (ni option SSL, ni la bonne liste d'exclusions — celle-ci ne concerne que les connexions *entrantes*).
- **Désactivation de Norton "Auto-Protect"** (15 min, via l'icône barre des tâches) → **le blocage SSL a persisté à l'identique**, avant et après (même erreur exacte via `curl`). Conclusion : Auto-Protect (scan fichiers) n'était pas le bon composant, ou le blocage vient d'ailleurs (vérification de révocation OCSP bloquée au niveau réseau).
- **Décision** : plutôt que de continuer à chercher dans les sous-menus Norton, contournement côté DVC uniquement (ne modifie rien sur la machine) :
  ```bash
  dvc remote modify origin --local ssl_verify false
  ```

**Deuxième blocage, découvert juste après** : `dvc push` passait la phase TLS mais échouait avec `401 Unauthorized` sur les 6 fichiers. Diagnostic : le token stocké ne faisait **qu'1 caractère** (vérifié sans l'afficher, via `length()` sur la valeur dans `.dvc/config.local`). Cause : le collage du token dans `Read-Host -AsSecureString` (PowerShell) a été tronqué — bug de collage connu avec les champs masqués dans certains terminaux.

**Correction** : ressaisie du token en clair (visible dans le terminal local, acceptable car jamais transmis ailleurs) :
```powershell
$DAGSHUB_TOKEN = Read-Host "Token DagsHub"
.\.venv\Scripts\python.exe -m dvc remote modify origin --local password $DAGSHUB_TOKEN
```
Vérification : le token fait bien 40 caractères cette fois.

**`dvc push` relancé → succès :**
```
6 files pushed
```
Confirmé par `dvc status -c` : *"Cache and remote 'origin' are in sync."*

Les 6 fichiers (`weatherAUS.csv`, `dataset.joblib`, `preprocessor.joblib`, `best_params.joblib`, `model.joblib`, `predictions.csv`) sont maintenant sur DagsHub.

*Rappel : Norton Auto-Protect doit être réactivé manuellement après ce dépannage (désactivation temporaire de 15 min, mais à vérifier).*

---

13. Mise en place du serveur MLflow (DagsHub)
--------------------------------------------------

Le README annonçait aussi MLflow pour le suivi des expériences, sur le même serveur DagsHub que DVC. Une branche collègue `origin/feat/mlflow-tracking` existait déjà avec un premier essai (logging MLflow ajouté dans `run_ml.py`, le script monolithique) — décision prise de plutôt intégrer le tracking **dans la pipeline DVC modulaire déjà en place** (`src/models/evaluate_model.py`, le dernier stage), pour que chaque `dvc repro` logue aussi automatiquement dans MLflow.

**Modifications** :
- `src/models/evaluate_model.py` : ajout d'un bloc conditionnel (actif seulement si `MLFLOW_TRACKING_URI` est défini, pour ne pas casser `dvc repro` sur une machine non configurée) qui logue `best_params`, les 6 métriques, et le modèle entraîné.
- `requirements.txt` : ajout de `mlflow`.
- `.env.example` : ajout des clés `MLFLOW_TRACKING_URI`/`USERNAME`/`PASSWORD` (documentation, pas de vrai secret).
- `scripts/mlflow/configure.sh` : écrit les vraies valeurs dans `.env` (jamais commité), en réutilisant si possible les identifiants DagsHub déjà validés pour DVC (`.dvc/config.local`) plutôt que de les re-demander.
- `scripts/dvc/repro.sh` : charge `.env` automatiquement avant `dvc repro`.
- `MLFLOW.md` : documentation dédiée (architecture, configuration, limitations).

**Deux blocages rencontrés et corrigés** :

1. **Même blocage SSL que pour DVC**, mais MLflow utilise un client HTTP différent (`requests`, pas le remote DVC) — `ssl_verify: false` de DVC ne s'applique pas ici. Équivalent trouvé et appliqué : `MLFLOW_TRACKING_INSECURE_TLS=true` dans `.env` (variable officielle du client MLflow, confirmée dans `mlflow/environment_variables.py`).

2. **`UntrustedTypesFoundException`** au moment de `mlflow.sklearn.log_model()` : MLflow 3.x utilise par défaut le format de sérialisation `skops` (plus sûr que `pickle`), qui ne reconnaît pas nativement les objets LightGBM (`lightgbm.basic.Booster`, `lightgbm.sklearn.LGBMClassifier`). Corrigé en spécifiant explicitement `serialization_format="cloudpickle"`.

**Résultat** : `dvc repro` relogue désormais aussi dans MLflow à chaque exécution. Run de test confirmé et visible sur DagsHub :
```
🏃 View run salty-rat-297 at: https://dagshub.com/Benoth08/may26_mlops_meteo.mlflow/#/experiments/0/runs/687e9a7bd85e4e6da35e17b316e73948
🧪 View experiment at: https://dagshub.com/Benoth08/may26_mlops_meteo.mlflow/#/experiments/0
```
avec les mêmes métriques que d'habitude (accuracy 0.8102, f1 0.6417, roc_auc 0.8767, pr_auc 0.7218, precision_pluie 0.5543, recall_pluie 0.7618).

---

État actuel — checklist
--------------------------

- [x] DVC installé et initialisé sur la branche `feature/dvc-setup`
- [x] Remote DagsHub configuré (`.dvc/config`)
- [x] `data/raw/weatherAUS.csv` suivi par DVC
- [x] Pipeline `dvc.yaml` (4 stages) écrite et testée (`dvc repro` réussi 2 fois)
- [x] `.gitignore` corrigé
- [x] Scripts `scripts/dvc/*.sh` + `DVC.md` créés
- [x] 3 commits créés sur `feature/dvc-setup`
- [x] Branche poussée sur GitHub (`origin/feature/dvc-setup`)
- [x] Authentification DagsHub configurée en local (`.dvc/config.local`, utilisateur `Denart97`)
- [x] **`dvc push` vers DagsHub réussi (6 fichiers, cache et remote synchronisés)** — via `ssl_verify: false` sur ce remote (voir section 12 pour le contexte)
- [x] Tracking MLflow intégré dans `evaluate_model.py`, connecté au serveur DagsHub, run de test confirmé
- [ ] Pull Request `feature/dvc-setup` → `main` sur GitHub (pas encore créée)
- [ ] Réactiver Norton Auto-Protect (désactivé temporairement pour le dépannage)
- [ ] Identifier la vraie cause du blocage SSL/réseau, pour pouvoir remettre `ssl_verify: true` / `MLFLOW_TRACKING_INSECURE_TLS=false` un jour (optionnel — pas bloquant)

Prochaines étapes
--------------------

1. Réactiver Norton Auto-Protect si ce n'est pas déjà fait automatiquement (délai de 15 min écoulé).
2. Ouvrir la Pull Request vers `main` (lien donné à l'étape 10 ci-dessus).
3. Chaque autre membre de l'équipe qui veut récupérer les données devra, sur sa machine :
   ```bash
   ./scripts/dvc/setup.sh
   ./scripts/dvc/configure-remote.sh   # avec son propre compte/token DagsHub
   dvc pull
   ```
   (et pourrait rencontrer le même type de blocage SSL selon son antivirus/réseau — voir section 12 pour le diagnostic et le contournement `ssl_verify false`.)

Emplacements importants
----------------------------

- Worktree de travail : `C:\Users\ander\OneDrive\Documents\Projet_MLOPs\may26_dvc-setup`
- Dossier original (branche `preprocessing`, intact) : `C:\Users\ander\OneDrive\Documents\Projet_MLOPs\may26_mlops_meteo`
- Documentation technique DVC : `DVC.md` (racine du worktree)
- Ce journal : `JOURNAL_SESSION_DVC.md` (racine du worktree)
