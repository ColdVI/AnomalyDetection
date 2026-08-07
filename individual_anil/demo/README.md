# Demo

Veri kümesi indirmeden, ~2 saniyede uçtan uca hattın **gerçek modüllerle**
nasıl çalıştığını gösteren tek dosyalık bir demo.

```bash
cd individual_anil
python demo/demo_calistir.py
```

Herhangi bir dizinden de çalışır (`python individual_anil/demo/demo_calistir.py`)
— script kendi `sys.path`'ini `__file__`'a göre kurar.

## Bağımlılıklar

`numpy`, `pandas`, `scipy` yeterli olurdu, ama demo `adsb.evaluation` modülünü
(bölüm 5 — olay yakalama/yanlış alarm hesaplaması) **olduğu gibi, yeniden
yazmadan** çağırıyor ve o modül `scikit-learn`'e transitif olarak ihtiyaç
duyuyor. `scikit-learn` zaten kök `requirements.txt` ve
[../requirements.txt](../requirements.txt)'te var — ek/egzotik bir bağımlılık
değil, sadece demo'nun "hiçbir algoritmayı yeniden yazma" kuralının doğal
sonucu.

## Ne gösteriyor (6 bölüm, hepsi gerçek modül çağrısı)

1. **Veri üretimi** — fiziksel olarak tutarlı sentetik uçuşlar (3 havuz:
   yalnız-normal eğitim, normal test, olaylı test). Enjeksiyon yalnız test
   tarafına uygulanır.
2. **Feature** — [`adsb.features.build_feature_table`](../adsb/features.py)
   ile residual kanalları.
3. **Kural skorlayıcı** — [`adsb.rules.ResidualRuleScorer`](../adsb/rules.py),
   yalnız-normal eğitim havuzundan kalibre edilir.
4. **Ardışık karar** —
   [`gecmis_calismalar.anomaly_core.sequential.MultiChannelPageCUSUM`](../gecmis_calismalar/anomaly_core/sequential.py),
   eşik yalnız eğitim skorlarından, test sonuçları görülmeden dondurulur.
5. **Değerlendirme** — [`adsb.evaluation`](../adsb/evaluation.py)
   (`truth_event_table`, `event_detection_metrics`, `natural_alert_burden`)
   ile aynı dondurulmuş eşikte olay yakalama oranı + yanlış alarm/uçuş-saati.
6. **Genlik-baskınlığı denetimi** —
   [`adsb.diagnostics.magnitude_domination_check`](../adsb/diagnostics.py),
   genliğe duyarlı bir skorlayıcı (işaretlenir) ile kural skorlayıcısının
   pencere skorunu (geçer) karşılaştırır.

## Ne göstermiyor

Yazdırılan sayılar (recall, yanlış alarm oranı, korelasyonlar) **sentetik
veriye aittir ve çalışmanın bulgusu değildir** — gerçek veri kümeleri,
deneyler ve sonuçlar için [../BULGU_KOD_ESLEMESI.md](../BULGU_KOD_ESLEMESI.md)
ve [../raporlar/](../raporlar/) altındaki raporlara bakın.
