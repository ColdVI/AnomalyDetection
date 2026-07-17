# RESIDUAL-V1 — Kalibrasyon NO-GO Raporu Talimatı (Codex için)

Tarih: 2026-07-17 · Statü: **Yalnız talimat — implementasyon/rapor yazımı yapılmadı.**
Karar kaynağı: kullanıcı, ALFA ve RFLY'nin ikisinde de matematiksel açığın
kapatılamadığı doğrulandıktan sonra "mevcut veriyle elde edilemez olarak raporla"
seçeneğini onayladı (bkz. `docs/RESIDUAL_V1_KALIBRASYON_STOP_SONRASI_TALIMAT.md`).
Hedef bütçe yeniden müzakere edilmeyecek; test/holdout açılmayacak.

## İstenen çıktı

`artifacts/uav_gnss_integrity_v1/uav_gnss_integrity_v1_final_no_go_report.tex` ile
aynı üslup ve disiplinde bir nihai rapor: `docs/RESIDUAL_V1_KALIBRASYON_NOGO_RAPORU.md`
(veya repo konvansiyonuna uyan bir `.tex`/`.md` — mevcut RESIDUAL-V1 dokümanları `.md`
kullanıyor, o formatta kalınması tutarlı olur).

**Zorunlu bölümler:**

1. **Yönetici özeti:** RESIDUAL-V1'in ALFA-engine (R6) ve RFLY-motor/sensor (Q1, Q2)
   kollarında eşik kalibrasyonu mevcut veri ve enstrümantasyonla elde edilemedi.
   Karar: `NO-GO / not achievable with current development-normal exposure`. Bunun
   bir "genel dedektör başarısız" cümlesi OLMADIĞI açıkça yazılmalı — GNSS raporunun
   kendi diliyle aynı ayrımı yap.

2. **Bu turda NE ÇALIŞTI — gömülmesin, öne çıkar:** K5 (V-dönüşü maskesi), S-4
   (komut ablasyonu: Q1/Q2 PASS, Q3 FLAGGED), ölçekleme+S-1 (R6 tautoloji-düzeltmeli
   proxy dahil, üçü de PASS), S-3 (ALFA/engine, RFLY/motor, RFLY/sensor — **üçü de
   threshold-bağımsız ayrışma testini geçti**). Bu, GNSS pilotundan farklı ve daha
   iyi bir sonuç: orada hiçbir yöntem sinyali göstermemişti, burada sinyal KANITLANDI
   (S-3 PASS) — tıkanma nokta sinyal yokluğu değil, kalibrasyon için yetersiz normal
   uçuş-saati. Bu ayrım raporun en önemli cümlesi olmalı.

3. **Sayısal açık (iki tabloyla):**
   - Kalibrasyon maruziyet açığı: ALFA/R6 0.168846 saat (hedef 2.0, çarpan 11.845×);
     RFLY/Q1,Q2 0.786237 saat (hedef 4.0, çarpan 5.088×).
   - Veri tavanı testi: ALFA 47 uçuşluk sabit corpus, 11 toplam normal uçuş (development
     9, holdout 1, test 1) — resmi makaleyle doğrulanmış tavan. RFLY resmi kaynağında
     84 Real-No_Fault uçuş var, projede 51 kullanılıyor; kalan 33'ün tamamı ingest
     edilse bile projeksiyon 0.786+33×0.0191765≈1.419 saat — hedefin (4.0) hâlâ
     ~2.581 saat / ~135 uçuş altında. Her iki veri setinde de redistribution VE ingest
     matematiksel olarak yetersiz — bu ikisinin de denenip elendiğini göster.

4. **Neden yeniden yatırım gerektiği (GNSS raporunun "Nihai Karar ve Öneri" bölümü
   gibi):** Yeniden açılış ancak ALFA için yepyeni bir kontrollü uçuş kampanyası
   (mevcut 47 uçuşluk akademik corpus'un ötesinde) veya RFLY için resmi kaynağın
   çok üzerinde (~135+ ek uçuş) yeni normal-uçuş verisi ile anlamlıdır — bu projenin
   kapsamı dışında, ayrı bir insan kararı gerektirir.

5. **Kesinlik sınırları:** kör holdout hiç açılmadı, bilimsel açma koşulu oluşmadı.
   DO_NOT_USE işaretli ilk kalibrasyon denemesi hiçbir üretim kararına girmedi.
   Hedef bütçe (`configs/residual_v1_cusum.json`) DEĞİŞTİRİLMEDİ.

6. **Provenance:** K5/S-4/scaling/S-1/S-3/kalibrasyon run dizinlerinin tam yolları,
   `SUMMARY_FOR_CLAUDE.json`, `CALIBRATION_COVERAGE_FAILURE.md`,
   `DO_NOT_USE_THRESHOLDS.md`, ve bu turdaki RFLY kaynak-sayım sonucu (84/51/33).

## Ek işler

- Eğer bu repoda geri-döndürülemez kararlar için ayrı bir ADR pratiği varsa
  (`docs/decisions.md` veya benzeri — konvansiyonu kendi kontrol et), bu NO-GO
  kararı için de bir ADR girdisi ekle; RESIDUAL-V1'in kendi dokümanları zaten
  "ADR yalnız geri-döndürülemez kararlara" (AP-7) diyor, bu karar o kategoriye girer.
- `docs/RESIDUAL_V1_IMPLEMENTASYON_TALIMATI.md`'deki Görev 5.4 durumunu (CUSUM
  kalibrasyonu) "tamamlandı" değil "STOP — bkz. NO-GO raporu" olarak güncelle,
  ileride başka bir ajan/oturum karışıklık yaşamasın.
- Test veya holdout açma; hedef bütçeyi bu raporu yazarken bile değiştirme —
  raporun kendisi bir "olduğu gibi" belgesi, bir düzeltme fırsatı değil.

Bu belge bir talimattır; NO-GO raporunun kendisini Codex yazacak.
