#!/usr/bin/env bash
# Rejoue la pipeline DVC de bout en bout (make_dataset -> grid_search ->
# train_model -> evaluate_model) et affiche le DAG, le statut et les
# metriques obtenues. A relancer apres tout changement de code/donnees pour
# verifier que la pipeline se reproduit correctement.
#
# Usage : ./scripts/dvc/repro.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    VENV_BIN=".venv/Scripts"
else
    VENV_BIN=".venv/bin"
fi

if [ ! -x "$VENV_BIN/python" ]; then
    echo "venv introuvable, lancement de setup.sh..."
    "$(dirname "$0")/setup.sh"
fi

# PYTHONIOENCODING=utf-8 : evite un crash Windows (cp1252) sur les emojis
# affiches par les scripts (✅). PATH : force les stages `python -m ...`
# lances par DVC a utiliser le venv (sinon ils prennent le premier python
# du PATH systeme, qui peut ne pas avoir lightgbm/sklearn installes).
export PYTHONIOENCODING=utf-8
export PATH="$(pwd)/$VENV_BIN:$PATH"

echo "== DAG de la pipeline =="
"$VENV_BIN/python" -m dvc dag

echo
echo "== dvc repro =="
"$VENV_BIN/python" -m dvc repro

echo
echo "== dvc status (doit etre vide si rien n'a change) =="
"$VENV_BIN/python" -m dvc status

echo
echo "== Metriques (metrics/scores.json) =="
"$VENV_BIN/python" -m dvc metrics show
