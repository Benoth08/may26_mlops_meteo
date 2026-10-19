#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Simulation d'arrivée progressive de données météo.
    
    Description :
        - Lire un dataset source complet
        - Découper les données en lots
        - Générer des fichiers CSV dans le dossier RAW
        - Simuler un flux entrant de données
        

    Version :
        1.0.0

    Historique :
        2026-08-04  -  Création du module
===============================================================================
"""

import json
import os
import uuid
import time
import random

import pandas as pd

from datetime import datetime, timezone
from pathlib import Path

from core.settings import SETTINGS
from core.logger import get_logger


class WeatherRawDataProducer:


    def __init__(self):
        self.logger = get_logger(
            "weather_raw_data_producer"
        )
        # Identifiant unique de simulation
        self.simulation_id = (
            str(uuid.uuid4())
            .replace("-", "")[:8]
        )
        # Source dataset
        self.source_file = Path(SETTINGS["paths"]["incoming"]) / SETTINGS["models"]["rawdata"]
        
        # Répertoire RAW
        self.raw_directory = Path(SETTINGS["paths"]["data"])
        self.raw_directory.mkdir(
            parents=True,
            exist_ok=True
        )
        
        # Paramètres par défaut
        self.default_batch_size = (
            SETTINGS["incoming"]["batch_size"]
        )
        self.default_delay = (
            SETTINGS["incoming"]["delay"]
        )
        self.separator = SETTINGS["incoming"].get(
            "separator",
            ","
        )
        self.data = None

        # Prochain batch à produire, conservé entre deux runs du DAG
        self.state_file = self.raw_directory / "producer_state.json"


    # ======================================================
    # Création du fichier configuration simulation
    # ======================================================

    def create_simulation_config(
            self,
            batch_size,
            max_batches,
            start_batch,
            delay,
            generated_files
    ):

        config = {
            "simulation_id": self.simulation_id,
            "source": str(self.source_file),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "batch_size": batch_size,
            "max_batches": max_batches,
            "start_batch": start_batch,
            "delay": delay,
            "generated_batches": len(generated_files),
            "generated_files": generated_files
        }

        config_file = self.raw_directory / f"config_{self.simulation_id}.json"

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        self.logger.info(
            {
                "event": "simulation_config_created",
                "file": str(config_file)
            }
        )
        
        return config_file

    # ==========================================================
    # Chargement dataset source
    # ==========================================================
    def load_source(self):
        if not self.source_file.exists():
            raise FileNotFoundError(
                self.source_file
            )

        self.logger.info({
            "event": "loading_source",
            "file": str(self.source_file)
        })

        self.data = pd.read_csv(
            self.source_file,
            sep=self.separator,
            dtype=str,
            keep_default_na=False
        )

        self.logger.info({
            "event": "source_loaded",
            "rows": len(self.data)
        })

    # ======================================================
    # Nom fichier batch
    # ======================================================
    def generate_filename(self, batch_number):
        return (
            f"weather_batch_"
            f"{self.simulation_id}_"
            f"{batch_number:05d}.csv"
        )

    def _read_next_batch(self) -> int:
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                return int(state["next_batch"])
            except (ValueError, KeyError, json.JSONDecodeError):
                self.logger.warning({"event": "producer_state_invalid", "file": str(self.state_file)})
        return 1

    def _write_next_batch(self, next_batch: int) -> None:
        self.state_file.write_text(json.dumps({"next_batch": next_batch}), encoding="utf-8")


    # ==========================================================
    # Lancement production des batches
    # ==========================================================
    def run(
        self,
        batch_size=None,
        start_batch=None,
        max_batches=None,
        delay=None
    ):
        
        try:

            if self.data is None:
                self.load_source()

            batch_size = (
                batch_size
                or self.default_batch_size
            )

            if start_batch is None:
                start_batch = self._read_next_batch()

            delay = (
                delay
                if delay is not None
                else self.default_delay
            )
            
            self.logger.info(
                {
                    "event": "simulation_started",
                    "simulation_id": self.simulation_id,
                    "batch_size": batch_size,
                    "start_batch": start_batch,
                    "max_batches": max_batches,
                    "delay": str(delay)
                }
            )
            
            generated = 0
            generated_files = []
            current_batch = start_batch
            last_batch = None

            while True:
                start_index = (current_batch - 1)*batch_size
                end_index = start_index+batch_size
                batch = self.data.iloc[start_index:end_index]

                # fin du dataset
                if batch.empty:
                    break

                filename = self.generate_filename(
                    current_batch
                )

                output_file = self.raw_directory / filename
            
                # Écriture atomique : le sensor d'intégration ne doit jamais
                # verrouiller un CSV à moitié écrit (.tmp est ignoré par le DAG).
                tmp_file = output_file.with_name(output_file.name + ".tmp")
                batch.to_csv(tmp_file, index=False, encoding="utf-8")
                os.replace(tmp_file, output_file)

                generated_files.append(filename)
                generated += 1
                last_batch = current_batch

                self.logger.info({
                    "event": "batch_created",
                    "simulation_id": self.simulation_id,
                    "batch": current_batch,
                    "rows": len(batch),
                    "file": str(output_file)
                })

                if (
                    max_batches
                    and generated >= max_batches
                ):
                    break

                current_batch += 1

                if delay > 0:
                    # Délai fixe en minutes
                    time.sleep(delay * 60)

                elif delay == -1:
                    # Délai aléatoire entre 1 et 15 minutes
                    random_delay = random.randint(1, 15)

                    self.logger.info({
                        "event": "random_delay",
                        "delay_minutes": round(random_delay, 2)
                    })

                    time.sleep(random_delay * 60)

            if last_batch is not None:
                self._write_next_batch(last_batch + 1)

            # ==================================================
            # Création config uniquement si succès
            # ==================================================
            self.create_simulation_config(
                batch_size=batch_size,
                max_batches=max_batches,
                start_batch=start_batch,
                delay=delay,
                generated_files=generated_files
            )
            
            result = {
                "event": "producer_finished",
                "simulation_id": self.simulation_id,
                "generated_batches": generated,
                "files": generated_files
            }
            
            
            self.logger.info({
                "event": "producer_finished",
                **result
            })

            return result
        
        except Exception as e:
            self.logger.error(
                {
                    "event": "producer_failed",
                    "simulation_id": self.simulation_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            raise

   