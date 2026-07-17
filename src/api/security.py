#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Authentification HTTP Basic de l'API de prédiction

    Description :
        Protège les endpoints de prédiction par authentification HTTP Basic.
        L'identifiant est comparé en temps constant, le mot de passe est
        vérifié contre son empreinte bcrypt. Les identifiants attendus sont
        lus depuis le fichier .env, jamais écrits en dur dans le code.

    Version :
        1.0.0

    Historique :
        2026-07-12  -  Création du module
===============================================================================
"""

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext


# Schema de securite HTTP Basic
security = HTTPBasic()

# Contexte passlib pour verifier le mot de passe contre son empreinte bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Identifiants attendus, lus depuis le .env
# API_AUTH_USERNAME       : l'identifiant en clair
# API_AUTH_PASSWORD_HASH  : l'empreinte bcrypt du mot de passe (jamais le mot de passe en clair)
EXPECTED_USERNAME = os.environ.get("API_AUTH_USERNAME", "")
EXPECTED_PASSWORD_HASH = os.environ.get("API_AUTH_PASSWORD_HASH", "")


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    # Comparaison de l'identifiant en temps constant (protege des attaques temporelles)
    username_ok = secrets.compare_digest(
        credentials.username.encode("utf8"),
        EXPECTED_USERNAME.encode("utf8"),
    )

    # Verification du mot de passe recu contre l'empreinte bcrypt rangee dans le .env
    password_ok = False
    if EXPECTED_PASSWORD_HASH:
        password_ok = pwd_context.verify(credentials.password, EXPECTED_PASSWORD_HASH)

    # Si l'un des deux echoue, on refuse avec une erreur 401 et l'en-tete Basic
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Identifiants valides : on renvoie l'identifiant
    return credentials.username
