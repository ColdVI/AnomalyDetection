# AGENTS.md — Proje Bağlamı (her Codex session'ı bunu okumalı)

## Bu proje ne?
8 haftalık İHA (UAV) veri mühendisliği internship projesi. 3 stajyer, Hafta 1-4 ortak
"Bronze→Silver→Gold" veri pipeline'ı, Hafta 5-8 bireysel projeler. Bu repo'yu yazan kişinin
bireysel projesi: **UAV anomali tespiti** (Isolation Forest, Autoencoder — etiketli/etiketsiz
anomali tespiti literatürü temelinde).

## Şu an neredeyiz?
**Sadece Bronze fazındayız.** Silver ve Gold'a geçilmeyecek — üç stajyer Bronze'u birlikte
review edip onaylamadan Silver'a dokunulmayacak. Bu repo'da Silver/Gold kod istenmedikçe
yazılmamalı.

## Mimari sapma — bilerek yapıldı
Resmî ders planı OpenSky API + generic MAVLink diyordu. Bu proje yerine şunları kullanıyor:
- **adsb.lol** (OpenSky yerine — auth'suz, kredi limitsiz, ODbL lisans)
- **ALFA dataset** (ground-truth fault/anomaly etiketli — bireysel anomali projesi için gerekli)
- **UAV Attack dataset** (GPS spoofing benign/malicious etiketli — aynı sebep)

Gerekçe `docs/decisions.md`'de. Bunu sorgulama, kabul edilmiş bir karar.

## Bronze'un altın kuralı
Ham veriyi olduğu gibi indir. Unit dönüşümü, koordinat ölçekleme, kolon harmonizasyonu YAPMA
— bunlar Silver'ın işi. Bronze'da sadece: indir, provenance kolonu ekle, (adsb kaynakları için)
Türkiye bbox filtrele. Detaylı faz planı: `docs/bronze_implementasyon_plani.md` (bu dosyayı
her fazdan önce oku).

## KRİTİK — gerçek veri yoksa dur, üretme
ALFA ve UAV Attack dosya adlandırma konvansiyonları standardize değil. Eğer
`data/bronze/<source>/_input/` altında gerçek dosya yoksa:
- Sahte/hayali bir dosya adı formatı varsayıp kod yazma.
- Loader fonksiyonunu, gerçek bir örnek dosya geldiğinde kolayca ayarlanabilecek şekilde
  yaz, ama format-çıkarma mantığını (`split("_")[0]` gibi) ASCII varsayımla sabitleme.
- Bunun yerine: ilgili klasöre 1 örnek dosya koyulmasını iste, ya da mevcut dosyaları
  `view`/`ls` ile incele ve gerçek adlandırmayı kodda yorum olarak belgele.

## Ortak araçlar (tekrar yazma, kullan)
`src/common/provenance.py`, `src/common/bbox.py`, `src/common/io.py` — Faz 1'de yazıldı.
Her loader bunları import etmeli. Provenance kolonları: `_source_type`, `_ingest_ts_utc`,
`_source_file`, `_schema_version`.

## Ekip / review akışı
Kod tek kişi tarafından Codex ile yazılıyor ama üç stajyer birlikte review edecek. Bu yüzden:
- Commit mesajları açık olsun (`feat(alfa): processed CSV loader + provenance`).
- Her faz kendi PR'ı olsun, tek mega-commit yapma.
- README ve `docs/bronze_schema.md` her loader eklendiğinde güncellensin.
