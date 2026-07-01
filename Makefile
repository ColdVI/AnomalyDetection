.PHONY: up down test bronze-hist bronze-rt-producer bronze-rt-consumer bronze-alfa bronze-attack minio-init

up:
	docker compose up -d

down:
	docker compose down

test:
	python -m pytest

minio-init:
	python -c "from src.common.io import get_minio_client, ensure_bucket, DEFAULT_BUCKET; import os; from dotenv import load_dotenv; load_dotenv(); ensure_bucket(get_minio_client(), os.getenv('MINIO_BRONZE_BUCKET', DEFAULT_BUCKET)); print('bucket ready')"

bronze-hist:
	python -m src.ingestion.adsblol_historical_loader

bronze-rt-producer:
	python -m src.ingestion.adsblol_producer

bronze-rt-consumer:
	python -m src.ingestion.adsblol_consumer

bronze-alfa:
	python -m src.ingestion.alfa_loader

bronze-attack:
	python -m src.ingestion.uav_attack_loader
