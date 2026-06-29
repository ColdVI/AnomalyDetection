.PHONY: up down test bronze-hist bronze-alfa bronze-attack

up:
	docker compose up -d

down:
	docker compose down

test:
	python -m pytest

bronze-hist:
	python -m src.ingestion.adsblol_historical_loader

bronze-alfa:
	python -m src.ingestion.alfa_loader

bronze-attack:
	python -m src.ingestion.uav_attack_loader
