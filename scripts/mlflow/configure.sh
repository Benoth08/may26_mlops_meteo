#!/usr/bin/env bash
# Configure le tracking MLflow vers le serveur DagsHub, dans le fichier .env
# a la racine du projet (deja ignore par Git, jamais commite). A executer
# une seule fois par machine/developpeur.
#
# Meme compte/token DagsHub que pour DVC (genere depuis :
# https://dagshub.com/user/settings/tokens).
#
# Usage : ./scripts/mlflow/configure.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

ENV_FILE=".env"
touch "$ENV_FILE"

read -rp "Utilisateur DagsHub : " DAGSHUB_USER
read -rsp "Token DagsHub (saisie masquee) : " DAGSHUB_TOKEN
echo

set_env_var() {
    local key="$1" value="$2"
    if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

set_env_var "MLFLOW_TRACKING_URI" "https://dagshub.com/Benoth08/may26_mlops_meteo.mlflow"
set_env_var "MLFLOW_TRACKING_USERNAME" "$DAGSHUB_USER"
set_env_var "MLFLOW_TRACKING_PASSWORD" "$DAGSHUB_TOKEN"

echo "Tracking MLflow configure dans .env (non commite)."
