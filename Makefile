.PHONY: up down test minio-init \
	bronze-upload-adsb-hist bronze-rt-producer bronze-rt-consumer bronze-alfa bronze-attack \
	silver-adsb-hist silver-adsb-rt silver-alfa silver-attack silver-generic gold

up:
	docker compose up -d

down:
	docker compose down

test:
	python -m pytest

minio-init:
	python -c "from src.common.minio_io import get_minio_client, ensure_bucket, DEFAULT_BUCKET; import os; from dotenv import load_dotenv; load_dotenv(); client = get_minio_client(); ensure_bucket(client, os.getenv('MINIO_BRONZE_BUCKET', DEFAULT_BUCKET)); ensure_bucket(client, os.getenv('MINIO_SILVER_BUCKET', 'silver')); ensure_bucket(client, os.getenv('MINIO_GOLD_BUCKET', 'gold')); print('buckets ready')"

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

gold:
	python -m src.gold.unify
