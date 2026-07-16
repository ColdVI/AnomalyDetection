# ONEMLI: 3.11 -> 3.13 (2026-07-15'teki birlestirmede pin'ler bu makinenin
# host Python'unda /3.14/ dogrulanmisti; 3.11-slim'de pip install patlardi).
#
# 2026-07-16 DUZELTME: kok requirements.txt yerine Dashboard/requirements.txt
# kullaniliyor -- kok dosya src/ pipeline + ML modelleri (torch, scikit-learn,
# scipy, pyarrow...) icin cok daha agir, Dashboard bunlarin HICBIRINI import
# etmiyor. Kok dosyayi kullanmak build'i (torch indirmesi yuzunden) saatlerce
# yavaslatiyordu -- bu, gercek bir regresyon olarak bulunup duzeltildi.
FROM python:3.13-slim

WORKDIR /app

COPY Dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Dashboard/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "codes/app.py"]
