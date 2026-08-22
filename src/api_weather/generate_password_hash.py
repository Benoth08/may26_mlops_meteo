#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Utilitaire de génération d'empreinte bcrypt

    Description :
        Petit utilitaire local, à lancer une seule fois, qui produit
        l'empreinte bcrypt d'un mot de passe. On colle ensuite l'empreinte
        affichée dans le .env (clé API_AUTH_PASSWORD_HASH). Le mot de passe
        en clair n'est jamais stocké ni affiché.

    Version :
        1.0.0

    Historique :
        2026-07-12  -  Création du module
===============================================================================
"""

import getpass

from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# getpass ne montre pas le mot de passe a l'ecran pendant la saisie
mot_de_passe = getpass.getpass("Mot de passe a hacher : ")

empreinte = pwd_context.hash(mot_de_passe)

print("\nEmpreinte a coller dans le .env (API_AUTH_PASSWORD_HASH) :")
print(empreinte)
