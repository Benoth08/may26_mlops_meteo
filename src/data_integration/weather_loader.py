#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Intégrartion des données brutes
    
    Description :
        Charge les données brutes dans une base de données PGSQL
        Aucun traitement de validzation ou de préparation n'est effectué à cette étape
        

    Version :
        1.0.0

    Historique :
        2026-06-11  -  Création du module
===============================================================================
"""

import uuid
import pandas as pd
import numpy as np
import os
import shutil

from sqlalchemy import create_engine

from datetime import datetime, timezone
from pathlib import Path

from core.logger import get_logger
from core.settings import SETTINGS
from core.config import load_postgres_config, ConfigError, PostgresConfig
from core.helpers_dataframe import normalize_column


class WeatherDataLoader:

    # ============================================================
    # Valeurs considérées comme NA dans les fichiers CSV
    # ============================================================
    CSV_NA_VALUES = SETTINGS["na_values"]


    def __init__(self, cfg : PostgresConfig):
        self.logger = get_logger("weather_loader")
        self.run_id = str(uuid.uuid4())

        self.logger.info({
            "event": "init_postgres_connection",
            "host": cfg.host,
            "user": cfg.user,
            "database": cfg.db,
            "run_id": self.run_id
        })

        self.engine = create_engine(cfg.sqlalchemy_uri)

        self.logger.info({
            "event": "postgres_connected",
            "run_id": self.run_id
        })


    def archive_file(self, src_path, archive_dir=None):
        """
        Archive le fichier source après import réussi.
        """
        try:
            # Création du dossier archive si nécessaire
            archive_path = Path(
                archive_dir or SETTINGS["paths"]["archive"]
            )
            archive_path.mkdir(parents=True, exist_ok=True)

            # Nom du fichier avec timestamp
            src = Path(src_path)

            if not src.exists():
                raise FileNotFoundError(f"Source file not found: {src}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            dest_path = archive_path / f"{src.stem}_{timestamp}{src.suffix}"

            # Déplacement
            shutil.move(src, dest_path)

            self.logger.info({
                "event": "file_archived",
                "source": str(src),
                "destination": str(dest_path),
                "run_id": self.run_id
            })

            return dest_path

        except Exception as e:
            self.logger.error({
                "event": "archive_failed",
                "error": str(e),
                "run_id": self.run_id
            }, exc_info=True)
            return None



    def load_csv(self, csv_path, table_name=None, chunksize=None):
        """
        Charge un fichier de données csv dans la table postgres
        """

        if table_name is None:
            table_name = SETTINGS["postgres"]["table_raw"]
        
        if chunksize is None:
            chunksize = SETTINGS["postgres"]["csv_chunk_size"]
            
        self.logger.info({
            "event": "début_du_chargement",
            "csv_path": csv_path,
            "table_name": table_name,
            "run_id": self.run_id
        })

        # -----------------------------
        # PHASE 1. Vérifier que le fichier existe
        # -----------------------------
        if not os.path.exists(csv_path):
            error_msg = f"Fichier introuvable : {csv_path}"
            self.logger.error({
                "event": "fichier_absent",
                "error": error_msg,
                "run_id": self.run_id
            })
            return {
                "status": "failed",
                "error": error_msg,
                "rows_processed": 0
            }
            
        import_date = datetime.now(timezone.utc)
        total_rows = 0

        # -----------------------------
        # PHASE 2 : Traitement des données
        # -----------------------------
        
        #2.1 : Ouvrir la transaction manuellement
        connection = self.engine.connect()
        trans = connection.begin()

        try:
            field_importdate = SETTINGS["postgres"]["importdate_column_norm"]
            field_runid = SETTINGS["postgres"]["importrunid_column_norm"]
            field_source = SETTINGS["postgres"]["importsource_column_norm"]
            sql_batch_size = SETTINGS["postgres"]["sql_batch_size"]
            
            for i, chunk in enumerate(
                pd.read_csv(csv_path, sep=',',  dtype=str, keep_default_na=False, chunksize=chunksize)
            ):

                if chunk.empty:
                    self.logger.warning({
                        "event": "chunk_vide",
                        "batch_id": i + 1,
                        "run_id": self.run_id
                    })
                    continue

                self.logger.info({
                    "event": "batch_start",
                    "batch_id": i + 1,
                    "rows": len(chunk),
                    "run_id": self.run_id
                })

                #2.2 : import des données
      
                # Convertir les valeurs NA en vrais NaN
                chunk = chunk.replace(self.CSV_NA_VALUES, np.nan)
                
                # Ajouter les champs techniques d'import               
                chunk[field_importdate] = import_date
                chunk[field_runid] = self.run_id
                chunk[field_source] = csv_path
    
                # Normalisation des colonnes
                chunk.columns = [normalize_column(col) for col in chunk.columns]
                
                try:
                    chunk.to_sql(
                        table_name,
                        con=connection,
                        if_exists="append",
                        index=False,
                        method="multi",
                        chunksize=sql_batch_size
                    )
                except Exception as db_error:
                    self.logger.error({
                        "event": "erreur_insertion_postgres",
                        "batch_id": i + 1,
                        "failed_batch_id": i + 1,
                        "failed_row_count": len(chunk),
                        "type": type(db_error).__name__,
                        "orig": str(getattr(db_error, "orig", "")),
                        "error": str(db_error),
                        "run_id": self.run_id
                    }, exc_info=True)

                    # rollback
                    trans.rollback()
                    connection.close()
                
                    return {
                        "status": "failed",
                        "error": str(db_error),
                        "failed_batch_id": i + 1,
                        "failed_row_count": len(chunk),
                        "rows_processed": total_rows
                    }
                    
                total_rows += len(chunk)

                self.logger.info({
                    "event": "batch_success",
                    "batch_id": i + 1,
                    "rows_inserted": len(chunk),
                    "run_id": self.run_id
                })

            # 2.2 Fin du chargement
            self.logger.info({
                "event": "chargement_termine",
                "total_rows": total_rows,
                "run_id": self.run_id
            })

            # -----------------------------
            # 3. Archivage du fichier
            # -----------------------------
            archive_path = self.archive_file(csv_path)

            # Vérifier que l’archivage a réussi
            if not archive_path or not str(archive_path).strip():
                self.logger.error({
                    "event": "archive_path_invalid",
                    "message": "Le chemin d'archive est vide ou nul",
                    "source_file": str(src_path),
                    "run_id": self.run_id
                })

                #Rollback
                trans.rollback()
                connection.close()
                
                return {
                    "status": "failed",
                    "error": "Archivage du fichier impossible",
                    "rows_processed": total_rows
                }

            # -----------------------------
            # 4 : Commit SQL + succès total
            # -----------------------------
            trans.commit()
            connection.close()       
            
            self.logger.info({
                "event": "fin_du_chargement",
                "csv_path": csv_path,
                "run_id": self.run_id
            })

            return {
                "status": "success",
                "rows_imported": total_rows,
                "archive_path": archive_path
            }

        except Exception as e:
            self.logger.error({
                "event": "erreur_generale",
                "error": str(e),
                "run_id": self.run_id
            }, exc_info=True)

            #rollback
            trans.rollback()
            connection.close()

            return {"status": "failed", "error": str(e)}
