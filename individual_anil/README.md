# Bireysel Çalışma — ML ile Anomali Tespiti Fizibilitesi (Anıl Keskin)

Gerçek uçuş telemetrisinde (ADS-B ve İHA) operasyonel bir anomali dedektörünün
kurulabilirliğini araştıran bireysel fizibilite çalışması. Beş veri kümesi
(ALFA, UAV Attack, UAV-SEAD, RflyMAD, ADS-B), 12'den fazla yöntem ailesi
(kural-tabanlı skorlayıcılar, CUSUM/ardışık karar, Isolation Forest,
LSTM-AE, Dense-AE, USAD, LSTM-forecaster, ridge regresyon, conformal
kalibrasyon, TCN, denetimli/denetimsiz derin öğrenme varyantları...) ve
17'den fazla deney turu boyunca sürdürüldü.

**Sonuç: disiplinli NO-GO.** Birkaç dar senaryoda gerçek, ölçülebilir sinyal
gösterilebildi — ama dondurulmuş bir yanlış-alarm bütçesi altında,
hedeflenen yakalama oranına ulaşan hiçbir yapılandırma bulunamadı. Bu bir
başarısızlık kaydı değil, üç yapısal bulgunun (bkz. aşağı ve
[BULGU_KOD_ESLEMESI.md](BULGU_KOD_ESLEMESI.md)) sistemli biçimde ortaya
çıkarılıp koda gömüldüğü bir fizibilite kaydıdır.

## Zamanınız kadar okuyun

| Süre | Nereye bakın |
|---|---|
| 2 dk | Bu README'nin geri kalanı |
| 5 dk | [demo/](demo/) — `python demo/demo_calistir.py` |
| 15 dk | [BULGU_KOD_ESLEMESI.md](BULGU_KOD_ESLEMESI.md) — üç yapısal bulgu, hangi kodda ölçüldü |
| Derinlemesine | [KOD_HARITASI.md](KOD_HARITASI.md) — modül modül tam harita |

## Hat (pipeline)

```
   feature            model              karar            kalibrasyon         değerlendirme
(residual/kural)  (CUSUM/kural/NN)  (eşik + ardışıklık)  (yalnız normal)   (recall + yanlış-alarm
                                                                             HER ZAMAN birlikte)
     ADS-B/İHA  ─────────────►  skor  ─────────────►  alarm  ◄──────────  yalnız-normal train
     ham telemetri                                       ▲                  skorlarının
                                                           │                  yüksek yüzdeliği
                                                    (etiket YOK)
```

Değişmeyen üç kural:

1. **Etiketler eğitime girmez.** Yarı-gözetimli sözleşme — etiketler yalnız
   eşik kalibrasyonu ve değerlendirme için kullanılır, hiçbir model/kural
   etiket görerek fit edilmez.
2. **Yakalama oranı ve yanlış alarm asla ayrı raporlanmaz.** Biri diğeri
   olmadan anlamsızdır (eşik düşürülerek recall her zaman yükseltilebilir).
3. **Her model eğitiminden sonra genlik-baskınlığı denetimi zorunlu** —
   bkz. [BULGU_KOD_ESLEMESI.md](BULGU_KOD_ESLEMESI.md) bulgu 1.

## Klasör tablosu

| Klasör | İçerik |
|---|---|
| [adsb/](adsb/) | Güncel ADS-B anomali paketi — feature/rule/CUSUM/contextual-physics/diagnostics |
| [gecmis_calismalar/](gecmis_calismalar/) | Veri kümesi bazında geçmiş çalışmalar (ALFA/UAV Attack/UAV-SEAD/RflyMAD) + paylaşılan çekirdek modüller |
| [scripts/](scripts/) | Deney/rapor sürücüleri (~100 script) |
| [tests/](tests/) | ~85 test dosyası, veri/ağ/Docker gerektirmiyor |
| [configs/](configs/) | Dondurulmuş çalışma zamanı konfigürasyonları |
| [artifacts/](artifacts/) | Kayıtlı run çıktıları, manifestler, eğitilmiş modeller |
| [notebooks/](notebooks/) | Colab GPU eğitim defterleri (10 adet) |
| [reddedilen_denemeler/](reddedilen_denemeler/) | 2026-07-10'da reddedilen iki ilk ADS-B yaklaşımı (arşiv, salt-okunur) |
| [dokunti_arsivi/](dokunti_arsivi/) | Kök dizinden arşivlenmiş çalışma logları/gevşek scriptler |
| [raporlar/](raporlar/) | Tüm ADR/rapor/sunum malzemesi |
| [demo/](demo/) | Veri gerektirmeyen, gerçek modülleri çağıran uçtan uca demo |

Ölçek: bu klasör altında ~85.000 satır Python (`adsb/`, `gecmis_calismalar/`,
`scripts/`, `tests/` toplamı; `artifacts/` altındaki üretilmiş çıktılar hariç).

## Kod nasıl yazıldı

Kod büyük ölçüde bir kodlama ajanıyla ve olay-güdümlü yazıldı. Ama deney
disiplini ajana devredilmedi: her deney turu önceden yazılı bir plan/ön-kayıt
ile başladı, sonuç görüldükten sonra parametre değiştirilmedi (bir turun
eşiği/mimarisi dondurulunca bir sonraki tur yeni bir turdu), ve her yapısal
karar gerekçesiyle birlikte kayda geçirildi (bkz.
[raporlar/decisions.md](raporlar/decisions.md)). Bu çalışma çalıştırılamaz
durumda olsa bile (gerçek veri kümeleri — yüzlerce GB — repoda değil) bu
disiplin kod ve raporlardan denetlenebilir.

## Tam geçmiş

Bu klasördeki kod, `arsiv` branch'inin en güncel hâlidir. Adım adım deney
kararları, ara sonuçlar ve tam commit tarihçesi için: `git checkout arsiv`.
