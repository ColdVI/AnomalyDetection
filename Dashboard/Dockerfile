# ONEMLI: 3.11 -> 3.13. Kok requirements.txt'teki pin'ler (2026-07-15'te
# birlestirilirken) bu makinenin host Python'unda (3.14) dogrulanmisti;
# numpy==2.5.0 Python >=3.12 istiyor, 3.11-slim'de pip install patlar.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Dashboard/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "codes/app.py"]
