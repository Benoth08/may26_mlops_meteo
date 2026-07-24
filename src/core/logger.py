#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Fonctions liées à la gestion des log
    
    Description :
        Bibliothèque fournissant des fonctions communes pour la gestion des logs
        - 

    Version :
        1.0.0

    Historique :
        2026-06-11  -  Création du module
===============================================================================
"""

import logging
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

from .settings import SETTINGS

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "log_module": record.module,
            "log_function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str)


def get_logger(name: str = "liora_weather", console: bool = True):
    """
    Retourne un logger JSON nommé `name`, écrivant dans un fichier dédié
    `<name>.json.log`, et sur stdout si `console=True` (défaut).
    """
    log_dir = Path(str(SETTINGS["paths"]["logs"]))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    
    level = SETTINGS.get("logging",{}).get("level", "INFO")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = JsonFormatter()

    file_handler = logging.FileHandler(log_dir / f"{name}.json.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    logger.propagate = False
    
    return logger