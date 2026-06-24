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
import os
import shutil

from sqlalchemy import create_engine

from datetime import datetime, timezone

from scripts.core.logger import get_logger


class WeatherDataLoader:

    def __init__(self, host, port, database, user, password):
        self.logger = get_logger()
        self.run_id = str(uuid.uuid4())

        self.logger.info({
            "event": "init_postgres_connection",
            "user": user,
            "database": database,
            "run_id": self.run_id
        })

        self.engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )

        self.logger.info({
            "event": "postgres_connected",
            "run_id": self.run_id
        })


    def archive_file(self, src_path, archive_dir="/data/archive"):
        """
        Archive le fichier source après import réussi.
        """
        try:
            # Création du dossier archive si nécessaire
            os.makedirs(archive_dir, exist_ok=True)

            # Nom du fichier avec timestamp
            base = os.path.basename(src_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_filename = f"{os.path.splitext(base)[0]}_{timestamp}{os.path.splitext(base)[1]}"
            dest_path = os.path.join(archive_dir, dest_filename)

            # Déplacement
            shutil.move(src_path, dest_path)

            self.logger.info({
                "event": "file_archived",
                "source": src_path,
                "destination": dest_path,
                "run_id": self.run_id
            })

            return dest_path

        except Exception as e:
            self.logger.error({
                "event": "archive_failed",
                "error": str(e),
                "run_id": self.run_id
            }, exc_info=True)
            return ""



    def load_csv(self, csv_path, table_name="weather_data_raw", chunksize=10000):
        """
        Charge un fichier de données csv dans la table postgres
        """

        self.logger.info({
            "event": "début_du_chargement",
            "csv_path": csv_path,
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
      
                # Ajouter la date d'import
                chunk["import_date"] = import_date
                chunk["import_run_id"] = self.run_id
                chunk["source_file"] = csv_path
    
                # Normalisation des colonnes
                chunk.columns = [
                    col.strip().lower().replace(" ", "_")
                    for col in chunk.columns
                ]
                
                try:
                    chunk.to_sql(
                        table_name,
                        con=connection,
                        if_exists="append",
                        index=False,
                        method="multi"
                    )
                except Exception as db_error:
                    self.logger.error({
                        "event": "erreur_insertion_postgres",
                        "batch_id": i + 1,
                        "failed_batch_id": i + 1,
                        "failed_row_count": len(chunk),
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
            if not archive_path:
                self.logger.error({
                    "event": "archive_path_invalid",
                    "message": "Le chemin d'archive est vide ou nul",
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
