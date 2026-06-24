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
from datetime import datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def get_logger(name: str = "liora_weather"):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = JsonFormatter()

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler("logs/weather_loader.json.log")

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger