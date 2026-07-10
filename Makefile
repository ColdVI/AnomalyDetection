.PHONY: test up-storage up-streaming down bronze-upload-adsb-hist \
	bronze-rt-producer bronze-rt-consumer silver-adsb-hist silver-adsb-rt \
	process-tars

test:
	python -m pytest tests -q

up-storage:
	docker compose up -d minio

up-streaming:
	docker compose --profile streaming up -d

down:
	docker compose --profile streaming down

bronze-upload-adsb-hist:
	python -m src.ingestion.upload_raw --source adsblol_historical --input $(INPUT)

bronze-rt-producer:
	python -m src.ingestion.adsblol_producer

bronze-rt-consumer:
	python -m src.ingestion.adsblol_consumer

silver-adsb-hist:
	python -m src.silver.parse_adsblol_historical

silver-adsb-rt:
	python -m src.silver.parse_adsblol_realtime

process-tars:
	python scripts/process_tars_sequential.py
