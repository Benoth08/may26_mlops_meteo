#!/usr/bin/env bash
# Cree le venv du projet et installe DVC + les dependances minimales
# necessaires pour executer la pipeline ML (numpy, pandas, scikit-learn,
# lightgbm, joblib). N'installe PAS requirements.txt en entier : ce fichier
# echoue sous Windows a cause de uvloop (non supporte hors Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    VENV_BIN=".venv/Scripts"
else
    VENV_BIN=".venv/bin"
fi

if [ ! -d .venv ]; then
    echo "== Creation du venv =="
    python -m venv .venv
fi

echo "== Installation de dvc + dependances ML =="
"$VENV_BIN/python" -m pip install --quiet --upgrade pip
"$VENV_BIN/python" -m pip install --quiet dvc numpy pandas scikit-learn lightgbm joblib

echo "== OK : $("$VENV_BIN/python" -m dvc --version) =="
