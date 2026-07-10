# ADS-B — Sıfırdan Başlangıç

Durum: model kodu yok; problem ve veri doğrulaması bekleniyor.

## Ana hedef

Bir ADS-B uçuş gözleminde davranışsal anomaly olup olmadığına açık bir binary cevap
üretmek:

`anomaly = yes | no`

Modelin ayrıca karar nedeni, kullanılan gözlem aralığı ve veri-kalitesi durumunu
raporlaması gerekir. `unknown / insufficient_data`, anomaly ile aynı sınıf değildir.

## Aşama 0 — modelden önce tamamlanacaklar

1. Üç tar arşivinin tarih, aircraft, trace, örnekleme aralığı, eksik kolon ve kaynak
   türü envanteri çıkarılacak.
2. Bir uçuşun başlangıç/bitiş ve kesinti kuralları yalnız veri üzerinden gösterilecek.
3. `lat`, `lon`, barometrik/geometrik irtifa, ground speed, track ve vertical rate
   için ham zaman serileri ile haritalar üretilecek.
4. Her fizik ilişkisi için “ölçülebilir”, “yalnız geçişte ölçülebilir” veya
   “ADS-B'den ölçülemez” kararı yazılacak.
5. Binary değerlendirme birimi seçilecek: satır, sabit pencere, event veya tüm uçuş.
6. Sentetik bozulmalar modelden bağımsız bir test setinde, fiziksel anlam ve
   gözlenebilirlik kontrolünden geçirilecek.

## İlk onay kapısı

Aşama 0 veri raporu kullanıcı tarafından görülüp onaylanmadan model ailesi, eşik,
normal öğrenme yöntemi veya headline recall seçilmeyecek.

## Aktif olmayanlar

Önceki ADS-B kodu ve sonuçları
`archive/2026-07-10_rejected_adsb_attempts/` altındadır. Yeni çalışma onların devamı
değildir.
