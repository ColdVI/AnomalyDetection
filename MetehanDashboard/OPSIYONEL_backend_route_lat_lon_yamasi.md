# Opsiyonel — Yusuf'un app.py'sine küçük ek (istersen)

**Bu dosya bir talimat, otomatik uygulanmadı.** Yusuf'un `Dashboard/app.py`
dosyasına dokunmadım — kendi kodun, kendi kod tabanın.

## Ne için

Yeni dashboard'da seçili uçağın kalkış→varış rotasını haritada bir **ok/arc**
olarak çizebiliyoruz (`MapArc`). Ama bunun için origin/destination
havalimanlarının **lat/lon**'una ihtiyacım var. Yusuf'un mevcut
`/api/route/{callsign}` endpoint'i adsbdb'den bu bilgiyi zaten çekiyor
(adsbdb'nin cevabında `origin.latitude`/`origin.longitude` var — bunu
daha önceki konuşmamızda birlikte doğrulamıştık) ama şu an bu iki alanı
**forward etmiyor**, sadece isim/şehir/IATA kodunu döndürüyor.

## Ne değişiyor, ne değişmiyor

- **Sadece ekleme** — mevcut hiçbir alan silinmiyor/değişmiyor, geriye dönük
  tamamen uyumlu. Senin dashboard'un (Dash) hâlâ birebir aynı çalışır.
- Redis cache anahtarı (`iha:route:{callsign}`) aynı kalıyor — sadece
  cache'lenen JSON'a 4 yeni alan ekleniyor.

## Diff

`Dashboard/app.py` içinde `get_route()` fonksiyonunda:

```python
            if route:
                origin = route.get("origin") or {}
                dest = route.get("destination") or {}
                result = {
                    "found": True,
                    "airline": (route.get("airline") or {}).get("name"),
                    "origin_name": origin.get("name"),
                    "origin_iata": origin.get("iata_code"),
                    "origin_city": origin.get("municipality"),
+                   "origin_lat": origin.get("latitude"),
+                   "origin_lon": origin.get("longitude"),
                    "dest_name": dest.get("name"),
                    "dest_iata": dest.get("iata_code"),
                    "dest_city": dest.get("municipality"),
+                   "dest_lat": dest.get("latitude"),
+                   "dest_lon": dest.get("longitude"),
                }
```

## Uygulamazsan ne olur

Hiçbir şey bozulmaz. Yeni dashboard `route.origin_lat`/`dest_lat` gelmediğini
görünce arc'ı hiç çizmez, panelde sadece "İstanbul (IST) → New York (JFK)"
metnini gösterir (zaten böyle davranacak şekilde yazıldı — `canDrawArc`
kontrolü `null`/`undefined` durumunu güvenle ele alıyor).
