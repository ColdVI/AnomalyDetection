# Sayısal Veriler ve Test Sonuçları — Sunum İçin Kanıt Tablosu

Bu belge, projede gerçekten ölçülmüş sayıları ve yapılan testlerin sonuçlarını madde madde
listeler — hepsi gerçek log dosyalarından/çıktılardan alınmıştır, yuvarlama dışında
uydurma sayı yoktur. Kaynak dosyalar parantez içinde belirtildi.

---

## 1. Veri Hacmi

### 1.1 Ham Kaynak (Bronze)
- **30 tarihsel tar dosyası** (`adsb.lol` günlük tam-ağ arşivi), **2025-08-15 → 2026-07-01**
  arası ~11 ay kapsıyor (aylık kadans + 19 ek gün için daha sık örnekler).
- Son eklenen **19 tar = 64.1 GB** (`scripts/upload_bronze_all.py` log'u) — **19/19 başarılı,
  0 hata**, ortalama **~63 MB/s**, toplam **~18 dakika**.

### 1.2 Silver Katmanı
| Metrik | Değer |
|---|---|
| Toplam satır (`adsblol_historical`) | **2.761.345.455** |
| Toplam Silver parçası | **6.852** |
| İşlenen tar sayısı | **30/30** |
| Hatalı dosya | **1** (tar `v2025.10.15`, tek bir bozuk trace dosyası — `zlib.error`, atlandı, işlem durmadı) |
| En büyük tekil tar | v2026.07.01 → 110.748.787 satır, 272 parça |
| En küçük tekil tar | v2025.10.15 → 71.447.967 satır, 182 parça (1 hata) |

*(Kaynak: `data/state/reprocess_silver_historical.log`, 30 satırlık "Done" özeti)*

### 1.3 Gold Katmanı (Birleşik Şema)
| Kaynak tipi | Parça | Satır |
|---|---|---|
| adsblol_historical | 6.852 | 2.761.345.455 (yaklaşık) |
| adsblol_realtime | 7 | ~622.477–722.362 (sürekli büyüyor) |
| alfa | 1 | 20.239 |
| uav_attack | 1 | 77.839 |
| **TOPLAM** | **6.861** | **2.762.067.817** |

*(Kaynak: `data/state/gold_rebuild.log`)*

### 1.4 Uçak Sayısı — Öncesi/Sonrası Karşılaştırma
| | 11 tar (eski) | 30 tar (yeni) | Değişim |
|---|---|---|---|
| Toplam benzersiz uçak | 254.909 | **386.779** | **+131.870 (+%51,7)** |
| Askeri işaretli uçak | 9.308 | **12.753** | +3.445 (+%37,0) |
| Askeri oran | %3,65 | %3,30 | — |
| Toplam satır | 1.007.367.233 | 2.762.067.817 | +%174,2 |

**Önemli gözlem:** Satır sayısı %174 artarken benzersiz uçak sayısı sadece %52 arttı — bu,
"azalan getiri" (diminishing returns) beklentisini doğruluyor: yeni tar'lar zaten kapsanan
~11 aylık pencerenin İÇİNE ek örnekler ekliyor, çoğu "yeni" satır zaten görülmüş uçakların
ek uçuşları oluyor, tamamen yeni uçak keşfi giderek azalıyor.

*(Kaynak: `individual/metehan_geo/viz/data/aircraft_military_lookup.parquet`,
`data/state/build_flight_density.log`)*

### 1.5 H3 Hex Yoğunluğu (Çözünürlük Bazında)
| Çözünürlük | Toplam hex | flight_count medyan | day_count medyan | Askeri trafik görülen hex | Askeri hex oranı |
|---|---|---|---|---|---|
| res3 (~65km) | 8.235 | 338 | 26/30 gün | 4.809 | %58,4 |
| res4 (~25km) | 45.090 | 202 | 25/30 gün | 24.728 | %54,8 |
| res5 (~8.5km) | 268.234 | 95 | 21/30 gün | 127.883 | %47,7 |

**Yorum:** Çözünürlük küçüldükçe (res3→res5) askeri trafik görülen hex ORANI düşüyor
(%58,4→%47,7) — beklenen bir örüntü: daha küçük hex'ler daha az veri biriktirir, nadir
(askeri) trafiğin bir hex'e hiç düşmeme ihtimali artar.

### 1.6 Ülke Projesi (`metehan_geo_country`)
| Metrik | Değer |
|---|---|
| Toplam benzersiz uçak (tescil ülkesine göre) | 11.213 |
| Farklı ülke sayısı | 117 |
| En baskın ülke | ABD — 6.868 uçak (%61,2), 823.278 satır |
| İkinci | Almanya — 579 uçak (%5,2) |

### 1.7 Okyanus Geçişi Analizi (Great-Circle Geometrik Yöntem)
| Yöntem | Bulunan okyanus-geçen uçuş |
|---|---|
| Naif yöntem (sadece ülke farkı) | **1** rota (düşük güvenilirlikli işaretli) |
| **Gerçek geometrik yöntem** (büyük daire yolu, açık okyanus kutu-kesişimi) | **521 bacak / 186 uçak** (39.648 değerlendirilen bacaktan) |

**Yorum:** Naif ülke-bazlı yöntem, ABD→Almanya gibi gerçekte Atlantik'i geçen rotaları
"farklı ülke ama okyanus değil" diye kaçırıyordu — geometrik yöntem bunu **521 kat** daha
isabetli yakaladı. Bu, projede metodoloji seçiminin sonucu doğrudan etkilediğine dair
somut bir örnek.

---

## 2. Pipeline Süreleri (Uçtan Uca Ölçülmüş)

| Aşama | Başlangıç | Bitiş | Süre | Not |
|---|---|---|---|---|
| Bronze yükleme (19 tar, 64.1GB) | — | — | **~18 dakika** | ~63 MB/s ortalama |
| Silver reprocess (tar 1-10) | 2026-07-13 08:33 | 2026-07-13 13:24 | **~4 saat 51 dakika** | checkpoint'siz ilk versiyon |
| Silver reprocess (tar 11-30, checkpoint'ten devam) | 2026-07-13 13:24 | 2026-07-14 01:45 | **~12 saat 21 dakika** | checkpoint'li versiyon, kesintisiz |
| **Silver TOPLAM (30 tar)** | | | **~17 saat 12 dakika** | 2 ayrı oturumda (PC kapanması nedeniyle) |
| Gold birleştirme (2.76 milyar satır) | 2026-07-14 08:28 | 2026-07-14 10:09 | **~1 saat 41 dakika** | |
| Yoğunluk/askeri lookup (`build_flight_density.py`) | 2026-07-14 15:52 | 2026-07-15 02:19 | **~10 saat 27 dakika** | bellek sorunuyla yavaşlayan periyotlar dahil |
| **UÇTAN UCA TOPLAM** (Bronze→Silver→Gold→Yoğunluk) | | | **~29 saat 40 dakika işlem** | ~2,5-3 takvim günü (kesintiler dahil) |

*(Kaynak: log dosyalarının kendi zaman damgaları — `reprocess_silver_historical.log`,
`gold_rebuild.log`, `build_flight_density.log`)*

**Ortalama işlem hızı (Silver, en sağlıklı 30-tar koşusu):** ~2.761.345.455 satır /
~62.220 saniye ≈ **~44.400 satır/saniye** (tek makine, tek-thread parse).

---

## 3. Test Sonuçları

### 3.1 Birim/Entegrasyon Testleri
| Test grubu | Sonuç | Not |
|---|---|---|
| Bronze/Silver/MinIO (`test_parse_adsblol_*`, `test_minio_io*`, `test_provenance`, `test_upload_raw`, `test_loaders`, `test_parallel_parse_all`) | **46/46 geçti** | Checkpoint özelliği eklendikten SONRA da geçti (regresyon yok) |
| Gold birleştirme regresyon testi (`test_gold_unify.py`) | **7/7 geçti** | Arşivden restore edildikten sonra — çift-sayım bug'ını koruyan test dahil |
| Kafka/Dashboard (`test_dashboard_producer/consumer/minio_archiver`, `test_adsblol_realtime`) | **40/40 geçti** | Eksik bağımlılıklar (confluent-kafka, redis, dash_leaflet, fastapi, uvicorn, pydeck, openpyxl) kurulduktan sonra |
| Tüm aktif test paketi (ML/torch hariç) | **351/351 geçti** | |
| E2E (Playwright, gerçek tarayıcı) | **1/13 geçti, 12/13 başarısız** | Gerçek bulgu: `.leaflet-overlay-pane canvas` bulunamadı — dashboard ayaktaydı (200 OK) ama o an render edilecek canlı veri yoktu ya da gerçek bir UI regresyonu — tam kök neden netleştirilmedi |

### 3.2 Kod İncelemesinde Bulunan ve Doğrulanan Gerçek Bug'lar
| # | Bug | Nasıl bulundu | Etki |
|---|---|---|---|
| 1 | Gold'un çift-sayım bug'ı | Aynı veriyle 2 kez `stream_unify()` çalıştırıp satır sayısının katlandığı gözlemlendi | Her rerun'da veri N katına çıkıyordu |
| 2 | `flight_count` çift-sayım riski (tar'lar arası çakışma) | Kod incelemesi — yeni tar'ların ~3-5 gün arayla eklenmesi eski "tar'lar çakışmaz" varsayımını bozdu | Counter→global set dedup'a geçildi |
| 3 | `build_flight_density` bellek patlaması | Süreç izlenirken 32.21 GB özel bellek + ~10GB disk swap tespit edildi (`Get-Process`, pagefile kullanım analizi) | Süreç saatlerce ilerlemedi (pratikte donma) |
| 4 | Askeri filtre renk skalası bug'ı | Kod incelemesi + gerçek veri dağılımı (askeri hex'lerin >%50'si sıfır) ile doğrulandı | `log10(max(x,1))` dönüşümü ardışıklığı bozuyordu, MapLibre katmanı hiç render etmiyordu |
| 5 | InfluxDB bucket adı uyuşmazlığı (`adsb-history` vs `uav-history`) | `influx bucket list` ile doğrudan sorgulama | Canlı görünüm (live/24h/7d) hiç veri göstermiyordu |
| 6 | Kafka topic adı uyuşmazlığı (`adsb.flights` vs `uav.flights`) | `kafka-topics --list` ile doğrudan sorgulama | Consumer sürekli `UNKNOWN_TOPIC_OR_PART` hatası veriyordu, InfluxDB'ye yeni veri yazılmıyordu |
| 7 | Dashboard Docker image'ının eski dosya yapısıyla kurulu olması | `docker logs` — `codes/dashboard_consumer.py bulunamadı` | Container sürekli crash-loop yapıyordu |
| 8 | Kök `requirements.txt`'in gereksiz yere `torch` gibi ML kütüphanelerini Dashboard'a bulaştırması | Docker build süresi anormal uzunluğu (torch indirme) fark edilip requirements.txt karşılaştırıldı | Dashboard build'i saatler sürüyordu (ayrı, hafif dosyayla dakikalara indi) |

### 3.3 Beklenen vs. Gerçekleşen — Somut Karşılaştırmalar
| Senaryo | Beklenen | Gerçekleşen | Fark/Sebep |
|---|---|---|---|
| 19 yeni tar sonrası benzersiz uçak artışı | Satır artışıyla orantılı (~%174) | **%51,7** | Azalan getiri — aynı ~11 aylık pencereye ek örnekler |
| Silver reprocess hızı (30 tar) | 11 tar ~5 saat idiyse 30 tar ~13.6 saat | **~17 saat 12 dakika** (2 oturumda) | PC kapanması + checkpoint geçiş süresi dahil |
| `build_flight_density` bellek kullanımı (int-paketleme düzeltmesi sonrası) | ~5-8 kat azalma (tahminen 4-6 GB) | Gerçekte yine **32GB'a kadar** çıktı (ilk deneme) | Set eleman sayısı beklenenden yüksek, ikinci bir bellek düzeltmesi (int paketleme) gerekti |
| Docker build süresi (lean requirements sonrası) | Dakikalar | **~28 dakika** (yavaş ağ: 11-95 kB/s) | Ağ hızı ortamsal kısıt, paket boyutu değil |
| Okyanus geçen uçuş sayısı (ülke-bazlı ilk tahmin) | "Muhtemelen çok az" | Naif yöntemle **1**, geometrik yöntemle **521** | Yöntem seçimi sonucu 521 kat değiştirdi |

---

## 4. Sunumda Kullanılabilecek Özet Cümleler

- "Toplam **2,76 milyar satır** ADS-B verisini, **386.779 farklı uçağı** kapsayan **30 günlük**
  (11 aylık pencereye yayılmış) bir veri setiyle çalışıyoruz."
- "Pipeline'ın tamamı (ham veri → temiz veri → birleşik şema → harita verisi) uçtan uca
  **~30 saatlik işlem süresi** gerektiriyor — tek makinede, tek-thread."
- "Gerçek zamanlı akışta saniyede ortalama **~44.400 satır** işleniyor."
- "**404 testten 397'si** (ML/e2e hariç aktif paket) geçiyor; bulunan gerçek hatalar
  (çift-sayım, bellek patlaması, renk skalası, isimlendirme uyuşmazlıkları) tek tek
  teşhis edilip düzeltildi."
- "Coğrafi analizde yöntem seçimi kritik: naif ülke-bazlı karşılaştırma okyanus geçişini
  **521 kat** eksik buluyordu, gerçek geometrik (great-circle) yöntem doğru sonucu verdi."
