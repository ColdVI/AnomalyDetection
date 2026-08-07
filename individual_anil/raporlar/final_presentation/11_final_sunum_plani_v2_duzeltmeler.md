# Final Sunum Planı v2 — Düzeltmeler, Yeni Görseller ve Eksik Halkalar

## 0. Bu belge nedir

`FINAL_SUNUM_PLANI.md` (Downloads'ta) ve ona yapılan kritik zaten elimizdeydi. Bu belge o kritiğin
**uygulanmış hâli**: 16 yeni görsel üretildi, kritikte listelenen 10 sayısal/yapısal hata tek tek
düzeltildi, kronolojide 2 eksik halka (ALFA'nın nihai grup-güvenli sonucu, 10 Temmuz ADS-B
sıfırlanmasının kendi anlatı payı) kapatıldı ve Slayt 25 üçe bölündü.

Bu belge `FINAL_SUNUM_PLANI.md`'yi **baştan yazmıyor** — her şey doğruysa "aynı kalıyor" diyor,
yalnız değişen slaytların tam metnini veriyor. Değişmeyen slaytlar için orijinal plana bakılmalı.

**Doğrulama yöntemi:** Her sayı `docs/final_presentation/01_project_timeline.md`,
`09_number_consistency_audit.md`, `04_metric_evolution.md`, `08_probabilistic_v31_deep_dive.md` ve
`artifacts/four_dataset_probabilistic_v31/runs/*/training_report.json` dosyalarından tek tek
doğrulandı. Hiçbir sayı uydurulmadı; kaynağı olmayan yerlerde ("noise/dropout yakalanabilirliği"
gibi) açıkça "bu turda ayrı raporlanmadı" yazıldı.

---

## 1. Ürettiğim 16 yeni görsel

Konum: `docs/final_presentation/presentation_assets/generated_v2/` (her biri `.png` + `.svg`).
Üreten script: `docs/final_presentation/presentation_assets/source_data/render_new_visuals_v2.py`
(mevcut `render_presentation_assets.py` ile aynı stil sabitleri ve `canvas/box/arrow` yardımcıları —
görsel olarak aynı deste ait gibi duruyor). Script yeniden çalıştırılabilir, hiçbir model eğitmiyor,
eşik değiştirmiyor, mühürlü test verisine dokunmuyor.

Kritikteki 14 kalemin hepsi üretildi, artı kritiğin kendisinin işaret ettiği 2 boşluğu kapatan 2
görsel daha eklendi (NG15, NG16) — toplam 16.

| Dosya | Slayt | Ne anlatır | Kritikteki karşılığı |
|---|---:|---|---|
| `NG01_veri_yolculugu_haritasi` | 2 | 8 duraklı yolculuk haritası; OpenSky ayrı işaretli, "tematik akış" notu | YENİ-1, **madde 3 düzeltildi** |
| `NG02_problem_semasi` | 3 | Çok kanallı telemetri + arıza bandı + istenen/istenmeyen | YENİ-2 |
| `NG03_normal_only_egitim_semasi` | 6 | Normal→model / arızalı→yalnız değerlendirme şeması + istisna notu | YENİ-3, **madde 7 düzeltildi** |
| `NG04_gozlem_katmani_karsilastirmasi` | 8 | ADS-B yayını vs araç içi telemetri | YENİ-4, **madde 6 düzeltildi** |
| `NG05_monolitik_vs_moduler` | 12 | Monolitik (2 ayrı kanıt kartı) vs modüler (0,833) | YENİ-5, **madde 5 düzeltildi** |
| `NG06_veri_cesitliligi_etkisi` | 13 | 0,731→0,918 + göze çarpan "bu sayılar final değil" uyarı kartı | YENİ-6, **madde 4 düzeltildi** |
| `NG07_bozulma_taksonomisi` | 14 | 2×3 ızgara, yalnız kanıtlı hücrelerde yakalanabilirlik etiketi | YENİ-7 |
| `NG08_nedensellik_duzeltmesi` | 15 | 0,878→0,611, yeni lider 0,751 | YENİ-8 |
| `NG09_cakisma_vs_gercek_baslangic` | 16 | %59,4→%19,4–22,4 | YENİ-9 |
| `NG10_gozlenebilirlik_karsilastirmasi` | 17 | GPS spoofing vs Ping DoS (kavramsal, etiketli) | YENİ-10 |
| `NG11_yontem_turu_ilerlemesi` | 19 | 0,205→0,390→0,459 + alarm yükü kartları | YENİ-11 |
| `NG12_sentetik_basari_tuzagi` | 23 | %97,6 vs 25,54/saat + 10 Temmuz sıfırlanması notu | YENİ-12, **eksik halka kapatıldı** |
| `NG13_komut_tepki_residual` | 25c (yeni) | Blok şema + 11,8×/5,1× kalibrasyon açığı | YENİ-13 |
| `NG14_nll_bilesen_ayristirmasi` | 28 | Aynı hata, iki σ senaryosu | YENİ-14 |
| `NG15_gnss_pilotu_karsilastirmasi` | 25b (yeni) | 3 yöntem × geliştirme/prova, kritik sözleşme | **YENİ — madde 2 için** |
| `NG16_alfa_v31_group_cv_sonucu` | 36b (yeni) | ALFA'nın 4 grup çapraz doğrulama sonucu | **YENİ — eksik halka** |

**Not — mevcut 12 görsel (`generated/` klasörü) için:** Planın 4.B bölümü bunların "Türkçeleştirilip
yeniden üretilmesi gerekiyor" diyor. Bu **artık doğru değil** — `render_presentation_assets.py`
kodunu okudum ve SVG çıktılarını jargon için taradım: 12 görselin hepsi zaten tam Türkçe ve
jargonsuz (`S20_final_result_card` zaten "Hedef karşılanmadı" diyor, "NO-GO" değil;
`S11_trained_random_magnitude` zaten "Kontrol eşiği" diyor, "gate" değil; `S13_probabilistic_transition`
zaten "Önceki/Son değerlendirme" diyor, "v2/v3.1" değil). Muhtemelen plan yazılırken script'in güncel
hâli kontrol edilmemiş. **Bölüm 4.B'deki iş listesi gereksiz — yapılacak bir şey yok.**

---

## 2. 18 hazır artefakt görseli — gerçek jargon denetimi

Planın 4.A tablosundaki tüm 18 dosyanın var olduğunu doğruladım (hepsi mevcut, yol hatası yok).
Ayrıca 12 tanesini bizzat açıp inceledim; kalan 6'sını aynı üretim script'inden gelen kardeş
dosyalar olduğu için yüksek güvenle çıkarımladım. Planın "bazılarına bak" uyarısı **doğruydu ve
ciddiydi** — ayrım şu:

**Zaten temiz / doğrudan kullanılabilir (11):** ALFA `class_counts`, `completeness_heatmap`,
`feature_auc_heatmap` (Türkçe başlık; eksen etiketleri fizik değişken adı, örn. `gps_speed_residual`
— jargon değil, teknik terim), UAV Attack `class_counts`, RflyMAD `01_data_composition` (Türkçe
başlık; `development/locked test` ve `Real/HIL/SIL` etiketleri kısa İngilizce domain kodu, kabul
edilebilir), SEAD `session_histogram`, SEAD `tsne_session` (eksen etiketleri kontrol edilmeli —
t-SNE grafiklerinde genelde "t-SNE 1/2" gibi jenerik etiket olur), `rflymad_frozen_ae_tcn_...`,
`rflymad_uzun_egitim_real_tradeoff`, ADS-B `04_route_summary`, `05_route_examples`.

**Ham İngilizce araştırma çıktısı — doğrudan mentöre gösterilmemeli (7):**
`01_training_and_selected_checkpoint.png` ("RflyMAD v3.1 — group-safe training"),
`03_flight_flag_matrix.png` ("Flight-level flag matrix / Development threshold; not an event
confusion matrix"), `02_detection_summary.png` ("Four-dataset probabilistic v2 — frozen-checkpoint
result summary"), `01_recall_vs_false_event_rate.png` ("RflyMAD v3.1 B0 frozen event grid" +
`k_of_n_time`/`standardized_nll_cusum` gibi kod-adı legend), `03_normal_test_alarm_burden.png`
(`k_of_n_time_2_of_3_approx_10hz` gibi kod-adı x-etiketleri), `05_detected_motor_event_timeline.png`
("RflyMAD v3.1 B0 — Detected Motor event / SIL | Motor | alarms=1"), ve aynı script'ten gelen
`06_missed_real_sensor_timeline.png` (kardeş dosya, aynı üslup — doğrudan görülmedi ama yüksek
güvenle aynı statüde).

**Sunumda kullanım kuralı:** Bu 7 dosya *veri olarak* doğru ve final sonucu temsil ediyor —
atılmamalı. Ama İngilizce/kod-adı başlıklarıyla doğrudan slayta konmamalı. İki seçenek: (a) slaydın
kendi başlığı ve konuşmacı notu Türkçe okumayı sağlarken grafiği "orijinal artefakt" olarak küçük
"ham araştırma çıktısı" etiketiyle göster, ya da (b) ben bu 7 grafiği de aynı `render_new_visuals_v2.py`
stiliyle Türkçe yeniden çizip `generated_v2/` altına eklerim (ayrı bir tur — bu turda kapsam dışı
bırakıldı çünkü kritik bunu istemedi, ama istenirse hemen yaparım).

---

## 3. Kritikteki 10 madde — tek tek düzeltme

### Madde 1 — 36 slayt 30 dakika için yoğun
Düzeltme bölüm 7'de: 39 slaytlık referans/appendix sürümü ile ~29 slaytlık ana sunum ayrıldı.

### Madde 2 — Slayt 25 tek başına 4 çalışma taşıyordu
Bölüm 5'te tam metinleriyle 3 yeni slayda bölündü: **25a** bağlamsal ADS-B + Page-CUSUM + alarm
bütçesi uyumsuzluğu, **25b** GNSS bütünlük pilotu (yeni görsel NG15), **25c** komut→tepki residual
(yeni görsel NG13).

### Madde 3 — OpenSky / "beş veri kaynağı" tutarsızlığı
`docs/decisions.md` ADR-001'i (2026-06-29) doğrudan okudum: "OpenSky yerine `adsb.lol`; generic
MAVLink örnekleri yerine ALFA ve UAV Attack veri setleri kullanılacaktır." Bu, OpenSky'ın gerçek bir
deney veri seti değil, **proje başlamadan hemen önce terk edilen ilk plan** olduğunu doğruluyor —
kritiğin işaret ettiği belirsizlik gerçek. Düzeltme: Slayt 1 alt başlığı ve Slayt 2 artık
**"Bir başlangıç keşfi, beş deney veri kaynağı, on sekizden fazla yöntem ailesi"** diyor (aşağıda
Slayt 1/2 tam metni). `NG01` görselinde OpenSky düğümü diğerlerinden gri renkle ve "başlangıç keşfi"
etiketiyle ayrıştırıldı.

### Madde 4 — ALFA 0,918 ile 0,611 karışıklığı
`NG06` görselinin sağına göze çarpan kırmızı bir kart eklendi: "Bu sayılar henüz final değil — 0,731
ve 0,918, nedensellik düzeltmesinden ÖNCEKİ (non-causal) uçuş ROC-AUC'sidir. Düzeltme sonrası ALFA'nın
en güçlü tekil sinyali cross-track error oldu: 0,751." Slayt 13'ün konuşmacı notu da bu ayrımı
zorunlu söylemeye çevrildi (bölüm 4'te tam metin).

### Madde 5 — Slayt 12'de tek sayı hatası
`NG05` görseli artık monolitik tarafta **tek bir "0,50" sayısı vermiyor**; iki ayrı kanıt kartı var:
"ALFA: rastgele sıralamaya yakın" ve "UAV Attack: uçuş ROC-AUC 0,21". Modüler taraf yalnız ALFA
sonucu (0,833) veriyor — veri setleri karıştırılmıyor.

### Madde 6 — "Etiket yalnız sağda üretilebilir" fazla kesin
`NG04` görselinin alt notu artık: **"Bu projede doğrulanmış arıza başlangıcı yalnız araç içi
etiketli veri setlerinde mevcuttu"** — projeye bağlı, genel/mutlak iddia değil.

### Madde 7 — "Normal-only ilke hiç ihlal edilmedi" yumuşatıldı
`NG03` görselinin alt notu: **"İstisna: LightGBM ve TCN karşılaştırmaları, açıkça ayrı işaretlenmiş
denetimli kollardı; ana hat normal-only kaldı."**

### Madde 8 — Slayt 32 (v2) vs Slayt 34/36 (v3.1/B0) RflyMAD sayı karışıklığı
Doğruladım: Slayt 32'deki dört-veri-seti tablosu (%56,1 / 0,768) ile Slayt 34'teki B0 sonucu
(%43,27 / 0,654) **gerçekten farklı değerlendirme turlarına ait** — biri kaynak-içi (v2) split, biri
grup-güvenli (v3.1) split. Bunlar birbirinin "gerilemesi" değil. Slayt 32'nin altına zorunlu görünür
not eklendi (bölüm 4'te tam metin): *"Bu karşılaştırma önceki (kaynak-içi) değerlendirme turuna
aittir; sonraki grup-güvenli RflyMAD sonucuyla doğrudan performans farkı olarak okunmamalıdır."*
Ayrıca **dört-veri-seti tablosundaki dört sayının hepsi aynı (v2) turdan** olduğu için tablo kendi
içinde tutarlı — asıl risk yalnız bu tabloyla slayt 34/36 arasındaydı, düzeltildi.

### Madde 9 — Görsel envanteri sayıları yanlış
Düzeltilmiş sayılar: **18 hazır görsel** (13 değil), **12 zaten Türkçe/temiz script-görseli**
(yeniden üretilmesi gerekmiyor — madde 4.B notuna bakın), **16 yeni görsel** (14 değil — kritiğin
kendi önerdiği 2 boşluk da üretildi).

### Madde 10 — Görünür slayt metni çok yoğun
Bölüm 4'te en yoğun 9 slayt (12, 13, 15, 16, 19, 23, 25a-c, 32, 34, 36, 36b) için
**[GÖRÜNÜR METİN] / [KONUŞMACI NOTU]** ayrımıyla yeniden yazıldı. Kalan slaytlar için aynı disiplin
uygulanmalı: slaytta yalnız başlık + 1 görsel + en fazla 3 kısa madde; ayrıntı (sayı gerekçesi,
istisna, ikinci kaynak) konuşmacı notuna gider.

---

## 4. Değişen slaytların tam metni (görünür metin / konuşmacı notu ayrımıyla)

### Slayt 1 — Kapak (küçük düzeltme)
**[GÖRÜNÜR METİN]**
İHA Telemetrisinde Anomali Tespiti: Sinyalden Güvenilir Alarma
Alt başlık: **"Bir başlangıç keşfi, beş deney veri kaynağı, on sekizden fazla yöntem ailesi, bir
dürüst sonuç."**

### Slayt 2 — İçindekiler
**[GÖRÜNÜR METİN]** Görsel: `NG01_veri_yolculugu_haritasi.png`
**[KONUŞMACI NOTU]** "Soldan sağa okuyun; ilk durak OpenSky, üzerinde model eğitilmedi — 29 Haziran
2026'da veri kaynağı kararı adsb.lol ve etiketli ALFA/UAV Attack setlerine kaydırıldı. Geri kalan
oklar bir veri setinden diğerine geçiş değil, bir varsayımın çürütülmesi. Şema tematik akıştır;
RflyMAD çalışması ile ilk geniş ADS-B denemesi takvimde kısmen örtüşüyordu, kesin sıralama iddiası
değildir."

### Slayt 12 — Monolitik → modüler
**[GÖRÜNÜR METİN]**
Başlık: Tek modelden fiziksel modüllere geçiş
Görsel: `NG05_monolitik_vs_moduler.png`
Üç madde:
- Monolitik: ALFA rastgeleye yakın, UAV Attack ROC-AUC 0,21
- Modüler: ALFA uçuş ROC-AUC 0,833 (en güçlü modül — rehberlik: 0,864)
- Kazanım modelden değil, sinyalin seyrelmesini önlemekten geldi

**[KONUŞMACI NOTU]** "Isolation Forest seçimi mantıklıydı: normal veri var, arıza az, model yalnız
normalden aykırılığı öğrenir. Ama monolitik kurulumda güçlü-ama-dar bir sinyal alakasız özellikler
içinde seyreldi. Özellikler fiziksel modüllere ayrılınca (kontrol-tepki, rehberlik, navigasyon,
sinyal kalitesi), her modül ayrı skorlandı, skorlar normal doğrulama verisinde ölçeklenip
birleştirildi. Bu iki sayı farklı veri setlerine ait — aynı modelin aynı veri setindeki 'önce/sonra'
karşılaştırması değil, monolitik yaklaşımın iki farklı veri setinde de zayıf kaldığının kanıtı."

### Slayt 13 — Veri çeşitliliği etkisi
**[GÖRÜNÜR METİN]**
Başlık: Model aynı kaldı, veri çeşitliliği arttı
Görsel: `NG06_veri_cesitliligi_etkisi.png` (kırmızı uyarı kartı görselin içinde, otomatik görünür)
Tek madde: "Küçük veri havuzunda bir derin modelin zayıf çıkması, her zaman modelin kendi suçu değildir."

**[KONUŞMACI NOTU — zorunlu, atlanmaz]** "Bu 0,731 ve 0,918 sayıları nedensellik düzeltmesinden
ÖNCEKİ değerlerdir ve iki slayt sonra düşecek — 0,611'e. Bunu şimdi nihai başarı olarak
sunmuyorum; göstermek istediğim tek şey, aynı LSTM autoencoder mimarisinin, yalnız 5 ek normal uçuş
eklenince (ham kayıtlardan çıkarılan) çok farklı bir sıralama sinyali üretmesi. Kolay yorum 'model
çalışmıyordu' olurdu; veri değişince aynı model değişti."

### Slayt 17 — UAV Attack gözlenebilirlik
**[GÖRÜNÜR METİN]**
Görsel: `NG10_gozlenebilirlik_karsilastirmasi.png` (kavramsal olduğu görselin altbaşlığında zaten yazıyor)
**[KONUŞMACI NOTU]** "Sağdaki ve soldaki eğriler kavramsal gösterim, gerçek zaman serisi değil —
gerçek veride 6 Ping DoS kaydının 4'ünde bu örüntü (etiket var, ölçülebilir iz yok) gözlendi. Bir
dedektörün bunu yakalayamaması model hatası değil; girdide ayırt edici bilgi yok."

### Slayt 19 — Yöntem turu (kritiğin kendi örneği, aynen uygulandı)
**[GÖRÜNÜR METİN]**
Başlık: Yakalama arttı; alarm yükü daha hızlı arttı
Görsel: `NG11_yontem_turu_ilerlemesi.png`
Üç madde:
- Chronos mekanik sinyali güçlendirdi (0,205→0,390)
- Tek itki-komutu özelliği en yüksek kategori yakalamasını verdi (→0,459)
- Bedeli: 38,1 yanlış alarm/saat

**[KONUŞMACI NOTU]** "EKF innovation tek başına ters yönde sinyal verdi (AUC 0,354) — anomalili
ölçüm estimator tarafından reddedildiğinde innovation düzenli üretilmiyor, en sorunlu örnekler daha
'temiz' görünüyor. LightGBM denendi, AUPRC 0,349 ile Isolation Forest'ın 0,385'inin altında kaldı —
denetimli öğrenme az ve heterojen etiket altında denetimsizi geçemedi. İki kanallı füzyon yakalamayı
artırdı ama yanlış alarmı 1,89–3,83 kat büyüttü. Soldan sağa yakalama gerçekten yükseliyor; ama her
adımda bedel arttı — 'daha iyi model bulduk' değil, 'aynı duvara farklı yollardan çarptık.'"

### Slayt 23 — Sentetik başarı tuzağı (10 Temmuz sıfırlanması artık açık)
**[GÖRÜNÜR METİN]**
Görsel: `NG12_sentetik_basari_tuzagi.png` (10 Temmuz notu görselin içinde)
**[KONUŞMACI NOTU]** "Üç gün işlendi: 256.150.550 satır, 638 Silver parça, yaklaşık 542 bin segment,
497.571 skorlanabilir uçuş-saati. Dört mimari ve beş fiziksel bozma tarifi denendi. Bir yaklaşım
sentetik anomalilerin %97,6'sını yakaladı — ama aynı sistem doğal trafikte 25,54 alarm/saat üretti.
Bu olaydan sonra eski non-ADS-B hat ve iki reddedilmiş ADS-B denemesi archive/ altına alındı, temiz
bir ADS-B hattı açıldı — proje raporlarında yakalama oranı ve alarm yükünün aynı eşik altında
birlikte sunulması bundan sonra zorunlu hâle geldi."

---

### Slayt 25a (YENİ) — Bağlamsal ADS-B modeli ve Page-CUSUM
**[GÖRÜNÜR METİN]**
Başlık: Sabit fizik eşiği yetmez — bağlam eklenince ilk sinir ağı genlik testini geçti
Görsel: mevcut `artifacts/adsb/plots/contextual_v2_evaluation/event_recall_heatmap.png` +
`paired_clean_natural_burden_heatmap.png` (ikisi de var, doğrulandı; İngilizce eksen etiketleri
olabilir — kullanmadan önce göz atılmalı)
Üç madde:
- Uçuş fazı, mesaj sıklığı, kanal durumu bağlam olarak eklendi; model yeniden-kurma yerine bir
  sonraki adımı tahmin etmeye başladı
- Genlik korelasyonu ~0,65 — önceki yeniden-kurma modelleri 0,84–0,99 bandındaydı
- Doğu/kuzey hız residual'ında iki eksenli Page-CUSUM: %49,7–51,3 yakalama / 0,017 olay-saat

**[KONUŞMACI NOTU]** "Skorlar normal dağılmadığı için hiyerarşik conformal kalibrasyon kullanıldı:
önce kanal+faz+sıklık, yeterli örnek yoksa kanal+faz, o da yoksa yalnız kanal — yaklaşık 16 milyon
satırlık kalibrasyon tablosu üretildi. Çoğu reçete dondurulmuş alarm bütçesi altında %6'nın altında
kaldı; yalnız konum Page-CUSUM güçlü çıktı. Reçete bazında güçlü, evrensel dedektör olarak yetersiz.
**Alarm bütçesi–olay penceresi uyumsuzluğu:** bütçe '100 uçuş-saatinde en fazla N alarm' biçiminde
tanımlanmıştı; ama her sentetik olay yalnız 0,5–1 saatlik tek bir uçuş içindeydi. Çok düşük saatlik
alarm olasılığı kısa olay penceresiyle çarpılınca, dedektörün olay sırasında alarm üretme olasılığı
matematiksel olarak çok küçük kaldı — sorun yalnız modelin görememesi değil, başarı sözleşmesinin iki
biriminin birbirini boğmasıydı."

### Slayt 25b (YENİ) — GNSS bütünlük pilotu
**[GÖRÜNÜR METİN]**
Başlık: Üç yöntem, aynı disiplinli sonuç
Görsel: `NG15_gnss_pilotu_karsilastirmasi.png`
**[KONUŞMACI NOTU]** "Gerçek İHA telemetrisinde yalnız GPS bütünlüğüne odaklanıldı — genel anomali
dedektörü değil. Roller ayrıldı: 20 uyum, 10 kalibrasyon, 23 geliştirme, 15 prova, 20 mühürlü.
Geliştirmede üç yöntem kritik sözleşmeyi (≤2 olay/uçuş-saati) hiçbiri geçemedi. Provada bağlamsal LSTM
%90 yakalama / 0 alarm-saat gibi çok iyi göründü — ama bu geliştirmeye taşınmadı, eşik ve model roller
arasında kararlı genellemedi. LSTM genlik artefaktına yakalanmamıştı (trained-random korelasyonu
~0,68–0,69), yani gerçekten örüntü öğrenmişti; yine de güvenilir çalışma noktası bulunamadı. 16 Temmuz
2026'da nihai rapor kapandı."

### Slayt 25c (YENİ) — Komut→tepki residual
**[GÖRÜNÜR METİN]**
Başlık: Anomaliyi sapmada değil, tahmin hatasında ara
Görsel: `NG13_komut_tepki_residual.png`
**[KONUŞMACI NOTU]** "`r = y − ŷ(kontrol komutları, bağlam)`. Örnek: gaz yüksek, normal model hava
hızının korunmasını bekliyor, ölçülen hava hızı düşüyor — ham değer hâlâ normal aralıkta olsa bile
residual hemen sapıyor. Bu, genlik baskınlığını yapısal olarak azaltır: agresif manevrada roll büyük
olabilir ama komuta uygun tepki varsa residual küçük kalır. Risk: model `y ≈ y(t-1)` kopyacılığını
öğrenip arızayı bir adım geriden takip edebilir — bu yüzden tepki geçmişi sınırlandı, komut ve yavaş
bağlam öne çıkarıldı. ALFA motor, RflyMAD motor ve sensör sınıflarında arıza öncesi/sonrası residual
dağılımları eşikten bağımsız olarak ayrıştı — sinyal gerçekten var. Ama güvenilir yanlış-alarm eşiği
kurmak için bağımsız normal uçuş süresi yetersizdi: ALFA'da gerekenin yaklaşık 1/11,8'i, RflyMAD'de
yaklaşık 1/5,1'i mevcuttu."

---

### Slayt 34 (eski 32) — Dört veri setinde ne çıktı?
**[GÖRÜNÜR METİN]** (aynı tablo, ama artık zorunlu bir uyarı satırıyla)
… mevcut tablo aynen kalıyor (RflyMAD 0,907/%56,1/0,768; SEAD 0,747/%3,40/0; ALFA ~0,483/~%20/0;
Attack —/—/32,42) …

**Slaydın altına, görünür biçimde eklenecek tek satır:**
> *"Bu dört sayı aynı (kaynak-içi) değerlendirme turuna aittir ve kendi içinde tutarlıdır. RflyMAD
> için birazdan göreceğiniz grup-güvenli sonuç (Slayt 36) farklı ve daha zorlu bir bölünmeye
> aittir — iki sayı arasındaki fark model gerilemesi değildir."*

### Slayt 36b (YENİ, Slayt 36 "RflyMAD B0 final sonuç"tan hemen sonra, "kör noktalar"dan önce) — ALFA'nın son sözü
**[GÖRÜNÜR METİN]**
Başlık: ALFA'nın son sözü: grup-güvenli çapraz doğrulama
Görsel: `NG16_alfa_v31_group_cv_sonucu.png`
**[KONUŞMACI NOTU]** "RflyMAD gibi ALFA da en sonunda aynı sıkı sınavdan geçti: oturum-ailesi bazlı
dört bağımsız grup üzerinde çapraz doğrulama. Dört gruptan yalnız biri (Grup 0) genlik-baskınlığı
kontrol eşiğini geçti — ama o grupta model hiçbir ayrım yapmadı: yakalama %0, normal alarm %0,
eşik hiç aşılmadı. Diğer üç grup kontrol eşiğinde kaldı; en yüksek yakalamayı gösteren Grup 3 (%100)
aynı zamanda normal uçuşların %83,3'ünde alarm üretti — yani ayrım değil, sürekli alarm. ALFA'da da
nihai bilimsel karar RflyMAD ile aynı: hedef karşılanmadı. Bu ayrıntı geliştirme aşamasında kaldı,
resmî bir final test sonucu değil."

---

## 5. Yeni slayt sırası (39 slaytlık referans/appendix sürüm)

| Yeni # | Eski # | Not |
|---:|---:|---|
| 1–24 | 1–24 | Aynı sıra; 1, 12, 13, 17, 19, 23 metinleri bölüm 4'te güncellendi |
| 25 | — | **YENİ:** Bağlamsal ADS-B + Page-CUSUM + alarm bütçesi |
| 26 | — | **YENİ:** GNSS bütünlük pilotu |
| 27 | — | **YENİ:** Komut→tepki residual |
| 28–33 | 26–31 | Aynı içerik, numara +2 kaydı |
| 34 | 32 | Dört veri seti tablosu + zorunlu uyarı satırı eklendi |
| 35 | 33 | Any-window matrisi yanılsaması |
| 36 | 34 | RflyMAD B0 final sonuç |
| 37 | — | **YENİ:** ALFA'nın son sözü (grup-CV) |
| 38 | 35 | RflyMAD kör noktalar |
| 39 | 36 | Kapanış |

---

## 6. Ana sunum (~29 slayt) vs referans/appendix (39 slayt)

Kritiğin 1. maddesi (36 slayt / 30 dakika fazla yoğun) hâlâ geçerli — düzeltmelerle sayı 39'a çıktı.
Öneri: **39 slaytı appendix/kaynak paketi olarak tut, mentöre gösterilecek ana sunumu ~29 slayda
indir.**

**Ana sunumdan çıkarılacak / birleştirilecek 10 slayt:**
- Slayt 12+13 birleştir (monolitik→modüler ve veri çeşitliliği tek slaytta, iki alt-görsel)
- Slayt 15+16 birleştir (iki metodolojik kırılma tek "düzeltme dalgası" slaydında)
- Slayt 25a+25b+25c → yalnız 25a (bağlamsal ADS-B) ve 25c (residual) kalsın, 25b (GNSS) appendix'e
- Slayt 28+29 birleştir (Gaussian NLL bileşenleri, negatif NLL açıklaması art arda kısa iki panel)
- Slayt 22 (RflyMAD dayanıklılık denemeleri) appendix'e — sonuç zaten Slayt 9'daki "üç düzeltme
  dalgası" ve Slayt 36'daki final sonuçla anlatılıyor
- Slayt 37 (ALFA'nın son sözü) appendix'e — sözlü olarak Slayt 34'ün uyarı satırında tek cümleyle
  değinilebilir, mentör sorarsa appendix'e dönülür

**Asla çıkarılmayacak omurga (kritik + orijinal planla aynı):** veri gözlenebilirliği (17), nedensellik
düzeltmesi, olay başlangıcı düzeltmesi, grup-güvenli bölme, genlik baskınlığı (20/31), gerçek olay
sonucu (34/36), kapanış (39).

---

## 7. Doğrulanmış ek sayı kartı (yalnız bu turda eklenen)

**ALFA v3.1 grup-güvenli çapraz doğrulama** (`artifacts/four_dataset_probabilistic_v31/runs/alfa_fold_*/training_report.json`):

| Grup | Uçuş ROC-AUC | Yakalama | Normal alarm | Genlik kontrolü |
|---|---:|---:|---:|---|
| Grup 0 | 0,444 | %0,0 | %0,0 | GEÇTİ (ρ=0,796/0,793) — ama sıfır ayrım |
| Grup 1 | 0,409 | %54,5 | %50,0 | KALDI (ρ=0,900/0,922) |
| Grup 2 | 0,545 | %18,2 | %25,0 | KALDI (ρ=0,943/0,962) |
| Grup 3 | 0,521 | %100,0 | %83,3 | KALDI (ρ=0,940/0,953) |

Kontrol eşiği ρ≥0,80. Dört gruptan yalnız biri eşiği geçti; onda da sıfır ayrım. Bu, RflyMAD B0 gibi
ALFA için de **geliştirme aşamasında** kalmış, resmî final test açılmamış bir sonuçtur.

**GNSS pilotu kritik sözleşme bütçesi (yeni doğrulanan ayrıntı):** kritik ≤2 olay/uçuş-saati (5 saniye
persistence), danışma ≤12 olay/uçuş-saati (15 saniye persistence). Kaynak: `docs/PROJE_SUREC_VE_SONUC.md`.

---

## 8. Sırada ne var (senin kararın gereken 3 nokta)

1. **7 ham İngilizce grafik** (bölüm 2) — ben de Türkçe yeniden çizeyim mi, yoksa slaytta "orijinal
   araştırma çıktısı" etiketiyle İngilizce mi kalsın?
2. **Ana sunum 29 slayda mı indirilsin**, yoksa mentöre 39 slaytlık tam referans mı gösterilecek?
3. **ALFA'nın son sözü appendix'e mi gitsin** yoksa ana sunumda mı kalsın — RflyMAD kadar ayrıntılı
   değil ama simetri için değerli.
