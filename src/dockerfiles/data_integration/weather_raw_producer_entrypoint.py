#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Weather RawData Simulation
    
    Description :
        Simulation des données météo brutes via fonctions WeatherRawDataProducer
        
    Version :
        1.0.0

    Historique :
        2026-08-04  -  Création du module
===============================================================================
"""
import json
import time
import argparse

from core.logger import get_logger
from core.settings import SETTINGS

from weather_rawdata_producer import WeatherRawDataProducer

logger = get_logger("weather_raw_producer_entrypoint", console=False)


def main():
    
    parser = argparse.ArgumentParser(
        description="Simulation génération fichiers RAW météo"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Nombre de lignes par fichier CSV"
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="Nombre maximum de lots générés"
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Délai entre deux générations"
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=1,
        help="Premier batch à générer"
    )

    args = parser.parse_args()

    try:
        
        start = time.time()

        logger.info({
            "event": "raw_producer_start",
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "delay": args.delay, 
            "start_batch": args.start_batch
        })
        
        producer = WeatherRawDataProducer()
        
        producer.run(
            batch_size=args.batch_size,
            start_batch=args.start_batch,
            max_batches=args.max_batches,
            delay=args.delay
        )
        
        duration = round(
            time.time() - start,
            2
        )

        result = {
            "status": "success",
            "duration_seconds": duration,
            "batch_size": args.batch_size,
            "generated_batches": args.max_batches
        }

        logger.info(
            {
                "event": "raw_producer_finished",
                **result
            }
        )

        print(
            json.dumps(result)
        )
        
    except Exception as e:
        logger.error({"event": "raw_producer_failed", "error": str(e)}, exc_info=True)
        print(json.dumps({"status": "failed", "error": str(e)}))
        raise


if __name__ == "__main__":
    main()
    

