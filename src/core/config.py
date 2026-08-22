"""
===============================================================================
    WeatherAUS — Module de configuration dynamique
    ---------------------------------------------------------------------------
    Charge les variables d'environnement et construit la configuration
    nécessaire à l'API, au loader et aux jobs Airflow.
===============================================================================
"""

import os

from dataclasses import dataclass

from .settings import SETTINGS


class ConfigError(Exception):
    """Erreur de configuration (variables d'environnement manquantes)."""
    pass


@dataclass(frozen=True)
class PostgresConfig:
    user: str
    pwd: str
    db: str
    host: str
    port: int

    @property
    def sqlalchemy_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )
        
def load_postgres_config() -> PostgresConfig:
    """
    Charge la configuration Postgres depuis les variables
    d'environnement et retourne un objet PostgresConfig.
    """
    pg_defaults = SETTINGS["postgres"]
    
    user = os.getenv("POSTGRES_WTH_USER", pg_defaults["default_user"])
    pwd = os.getenv("POSTGRES_WTH_PASSWORD", pg_defaults["default_password"])
    db = os.getenv("POSTGRES_WTH_DB", pg_defaults["default_db"])
    host = os.getenv("POSTGRES_WTH_HOST", pg_defaults["default_host"])
    port = os.getenv("POSTGRES_WTH_PORT", str(pg_defaults["default_port"]))

    missing = [
        name for name, value in
        [("POSTGRES_WTH_USER", user), ("POSTGRES_WTH_PASSWORD", pwd),
         ("POSTGRES_WTH_DB", db), ("POSTGRES_WTH_HOST", host),
         ("POSTGRES_WTH_PORT", port)]
        if not value
    ]
    if missing:
        raise ConfigError(
            f"Variables d'environnement Postgres manquantes : {', '.join(missing)}"
        )

    return PostgresConfig(
        user=user,
        pwd=pwd,
        db=db,
        host=host,
        port=int(port),
    )  
