#!/usr/bin/env bash
# Configure l'authentification DagsHub pour le remote DVC "origin", en local
# uniquement (.dvc/config.local, jamais commite sur Git). A executer une
# seule fois par machine/developpeur avant le premier `dvc push`/`dvc pull`.
#
# Le token DagsHub se genere depuis : https://dagshub.com/user/settings/tokens
#
# Usage : ./scripts/dvc/configure-remote.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    VENV_BIN=".venv/Scripts"
else
    VENV_BIN=".venv/bin"
fi

read -rp "Utilisateur DagsHub : " DAGSHUB_USER
read -rsp "Token DagsHub (saisie masquee) : " DAGSHUB_TOKEN
echo

"$VENV_BIN/python" -m dvc remote modify origin --local auth basic
"$VENV_BIN/python" -m dvc remote modify origin --local user "$DAGSHUB_USER"
"$VENV_BIN/python" -m dvc remote modify origin --local password "$DAGSHUB_TOKEN"

echo "Remote 'origin' configure dans .dvc/config.local (non commite)."
