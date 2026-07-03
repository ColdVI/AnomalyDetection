.PHONY: up up-storage up-streaming down test minio-init \
	bronze-upload-adsb-hist bronze-rt-producer bronze-rt-consumer bronze-alfa bronze-attack \
	bronze-uav-sead \
	silver-uav-sead-local ml7-candidate \
	silver-adsb-hist silver-adsb-rt silver-alfa silver-attack silver-uav-sead silver-generic gold \
	ml-features ml6-evaluate ml6-package adsb-watch process-tars

# MinIO only — use for Silver parsing / Gold unify (saves ~2GB RAM vs full stack)
up-storage:
	docker compose up -d minio

# All services — use for realtime pipeline (Kafka + Redis + InfluxDB + MinIO)
up-streaming:
	docker compose --profile streaming up -d

up:
	docker compose --profile streaming up -d

down:
	docker compose --profile streaming down

test:
	python -m pytest

minio-init:
	python -c "from src.common.minio_io import get_minio_client, ensure_bucket, apply_realtime_retention, DEFAULT_BUCKET; import os; from dotenv import load_dotenv; load_dotenv(); client = get_minio_client(); ensure_bucket(client, os.getenv('MINIO_BRONZE_BUCKET', DEFAULT_BUCKET)); ensure_bucket(client, os.getenv('MINIO_SILVER_BUCKET', 'silver')); ensure_bucket(client, os.getenv('MINIO_GOLD_BUCKET', 'gold')); apply_realtime_retention(client); print('buckets ready + realtime retention set')"

bronze-upload-adsb-hist:
	python -m src.ingestion.upload_raw --source adsblol_historical --input $(INPUT)

bronze-rt-producer:
	python -m src.ingestion.adsblol_producer

bronze-rt-consumer:
	python -m src.ingestion.adsblol_consumer

bronze-alfa:
	python -m src.ingestion.upload_raw --source alfa --input $(INPUT)

bronze-attack:
	python -m src.ingestion.upload_raw --source uav_attack --input $(INPUT)

silver-adsb-hist:
	python -m src.silver.parse_adsblol_historical

silver-adsb-rt:
	python -m src.silver.parse_adsblol_realtime

silver-alfa:
	python -m src.silver.parse_alfa

silver-attack:
	python -m src.silver.parse_uav_attack

silver-generic:
	python -m src.silver.parse_generic --source $(SRC) --bronze-prefix $(SRC)/

bronze-uav-sead:
	python -m src.ingestion.uav_sead_downloader

silver-uav-sead:
	python -m src.silver.parse_uav_sead

silver-uav-sead-local:
	python -m src.silver.parse_uav_sead --local-bronze-dir data/objectstore/bronze/uav_sead

gold:
	python -m src.gold.unify

ml-features:
	python -m src.ml.build_features

ml6-evaluate: ml-features
	python -m scripts.run_ml6_remeasure
	python -m scripts.run_ml6_ablation
	python -m scripts.run_ml6_thresholds
	python -m scripts.run_ml6_events
	python -m scripts.run_ml6_alarm_policy

ml7-candidate: ml-features
	python -m scripts.package_ml7_candidate
	python -m scripts.run_ml6_alarm_policy --model-dir artifacts/models/uav_sead/ml7_candidate_iforest --run-name ml7_alarm_policy

ml6-package:
	python -m scripts.package_ml6_iforest
	python -m scripts.package_ml6_lstm

adsb-watch:
	python -m src.ingestion.adsb_watcher

process-tars:
	python scripts/process_tars_sequential.py
