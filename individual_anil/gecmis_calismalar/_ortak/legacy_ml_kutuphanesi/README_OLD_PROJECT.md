# UAV Anomaly Detection Data Platform

Bu repo, IHA anomali tespiti calismasinin veri platformudur. Su anda yalnizca **Bronze
fazindayiz**; Silver ve Gold implementasyonu ekip review'u tamamlanmadan baslamayacaktir.

```text
adsb.lol historical --\
adsb.lol realtime  ---+--> Bronze (raw + provenance) --> Silver --> Gold
ALFA               ---+                               [review bekliyor]
UAV Attack         --/
```

Bronze kaynak alanlarini ve degerlerini degistirmez. Yalnizca standart provenance
kolonlarini ekler; adsb.lol kayitlarinda Turkiye bbox filtresi uygular. Birim donusumleri
ve ortak sema Silver'in isidir.

## Kurulum

Python 3.10+ ve Docker gereklidir.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
docker compose up -d
```

Kafka `localhost:9092` uzerinde calisir. Kapatmak icin `docker compose down` kullanin.
GNU Make bulunan ortamlarda `make test`, `make up` ve `make down` da kullanilabilir.

## Yerel veri dizinleri

Veri repoya girmez. Indirilen dosyalari su dizinlere koyun:

```text
data/bronze/adsblol_historical/_input/   # gunluk .tar arsivleri
data/bronze/adsblol_realtime/_landing/   # ham JSONL (uygulama olusturur)
data/bronze/alfa/_input/                 # ALFA processed CSV agaci
data/bronze/uav_attack/_input/           # UAV Attack CSV/ULog agaci
```

ALFA ve UAV Attack loader'lari gercek dosya/klasor adlari gorulmeden label cikarimi
varsaymayacaktir. Kaynak kararlari [docs/decisions.md](docs/decisions.md), Bronze metadata
sozlesmesi [docs/bronze_schema.md](docs/bronze_schema.md) icindedir..
no intro
