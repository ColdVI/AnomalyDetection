.PHONY: up down test bronze-hist bronze-rt-producer bronze-rt-consumer bronze-alfa bronze-attack minio-init silver-alfa silver-attack gold

up:
	docker compose up -d

down:
	docker compose down

test:
	python -m pytest

minio-init:
	python -c "from src.common.minio_io import get_minio_client, ensure_bucket, DEFAULT_BUCKET; import os; from dotenv import load_dotenv; load_dotenv(); client = get_minio_client(); ensure_bucket(client, os.getenv('MINIO_BRONZE_BUCKET', DEFAULT_BUCKET)); ensure_bucket(client, os.getenv('MINIO_SILVER_BUCKET', 'silver')); ensure_bucket(client, os.getenv('MINIO_GOLD_BUCKET', 'gold')); print('buckets ready')"

bronze-hist:
	python -m src.ingestion.adsblol_historical_loader

bronze-rt-producer:
	python -m src.ingestion.adsblol_producer

bronze-rt-consumer:
	python -m src.ingestion.adsblol_consumer

bronze-alfa:
	python -m src.ingestion.upload_raw --source alfa --input $(INPUT)

bronze-attack:
	python -m src.ingestion.upload_raw --source uav_attack --input $(INPUT)

silver-alfa:
	python -m src.silver.parse_alfa

silver-attack:
	python -m src.silver.parse_uav_attack

gold:
	python -m src.gold.unify
