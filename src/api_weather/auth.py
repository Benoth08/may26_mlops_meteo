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
import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext

# Schema de securite HTTP Basic
security = HTTPBasic()

# Contexte passlib pour verifier le mot de passe contre son empreinte bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _verify_credentials(username: str, password: str, expected: dict) -> bool:
    """
    Comparaison en temps constant (secrets.compare_digest) pour éviter une
    timing attack. Séparée de check_auth() pour rester testable sans avoir
    à construire une fausse requête FastAPI.
    """
    valid_user = secrets.compare_digest(username.encode("utf8"), expected["user"].encode("utf8"))
    
    # Verification du mot de passe recu contre l'empreinte bcrypt rangee dans le .env
    valid_pwd = pwd_context.verify(password, expected["password"])
    return valid_user and valid_pwd


def check_auth(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(security),
) -> bool:
    """
    Dépendance FastAPI utilisée par tous les endpoints protégés :
        auth: bool = Depends(check_auth)

    Lève 503 si l'API n'a pas encore fini son startup (credentials pas
    chargés), 401 si les identifiants sont invalides.
    """
    creds = request.app.state.api_credentials
    if not creds:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication unavailable")

    # Si l'un des deux echoue, on refuse avec une erreur 401 et l'en-tete Basic
    if not _verify_credentials(credentials.username, credentials.password, creds):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Basic"},
        )
        
    # Identifiants valides : on renvoie true
    return True

