# RESIDUAL-V1 — Kalibrasyon STOP Sonrası Devam Talimatı (Codex için)

Tarih: 2026-07-17 · Statü: **Yalnız talimat — implementasyon/veri değişikliği yapılmadı.**
Kaynak: `docs/RESIDUAL_V1_FAZ_E_SONUCLARI.md`'deki kalibrasyon STOP'u + bu oturumda
yapılan bağımsız split/normal-uçuş analizi. O rapor "sonraki meşru adım daha fazla
development-normal maruziyet sağlamak veya yanlış-alarm hedef sözleşmesini yeniden
onaylamak" diyor — bu belge o iki seçeneği somut sayılarla açıyor.

## Önce: "daha fazla maruziyet" matematiksel olarak ne kadar mümkün?

Split manifest'ler + Silver `events.json` üzerinden development-normal uçuş sayıları
sayıldı (development/holdout/test roldeki normal-uçuş dağılımı):

| Veri | Development normal | Holdout normal | Test normal | Toplam normal (bilinen evren) |
|---|---:|---:|---:|---:|
| ALFA | 9 | 1 | 1 | **11** |
| RFLY | 41 | 10 | 0 | **51** |

RFLY'nin 51'i, `rflymad_parsed_pool.png`'deki bilinen Real-No_Fault sayısıyla (~51)
birebir örtüşüyor — yani development ZATEN bilinen normal-uçuş evreninin büyük
kısmını (ALFA %82, RFLY %80) elinde tutuyor.

**Kritik hesap — redistribütion (test/holdout'tan development'a kaydırma) açığı
kapatamaz:**
- ALFA: 9 normal uçuş 0.168846 saat üretiyor; hedef 2.0 saat. Elde kalan TÜM normal
  uçuşları (holdout+test'teki 2 tanesi) development'a taşısan bile development
  9→11'e çıkar (**+%22**), ama gereken çarpan **11.845×**. Matematiksel olarak
  kapanmaz.
- RFLY: 41 normal uçuş 0.786237 saat üretiyor; hedef 4.0 saat. Holdout'taki 10
  normal uçuşu da development'a taşısan development 41→51'e çıkar (**+%24**), ama
  gereken çarpan **5.088×**. Aynı şekilde kapanmaz.

**Sonuç: split içi yeniden dağıtımla (test/holdout'tan "ödünç alarak") bu açık
kapatılamaz — bu yol baştan elenir, denenmemeli.** Ayrıca test/holdout'u bu amaçla
açmak zaten Faz E raporunun açıkça yasakladığı şey.

## Codex'in yapması gerekenler (sırayla)

### 1. Yalnız bir olgu sorusu: bilinen normal-uçuş evreni gerçekten 11/51 mi?

- **ALFA:** 47 uçuşluk academic corpus önceki bir oturumda ayrıca doğrulanmıştı
  (Keipour/Mousaei/Scherer, arXiv 1907.06268 — tam 47 işlenmiş uçuş; repodaki
  `Desktop/ALFA/processed/processed/` de tam 47 klasör). Bunun ötesinde ALFA normal
  uçuşu YOK — bu adımı atla, doğrudan §2'ye geç.
- **RFLY:** RflyMAD'ın resmî veri dokümantasyonına
  (`https://rfly-openha.github.io/documents/4_resources/dataset.html`) bak: kayıtlı
  Real-No_Fault kategorisinde şu an ingest edilen ~51'den FAZLA uçuş var mı? Varsa
  kaç tane, hangi alt-kümede. **Bu yalnız bir sayma/doğrulama işi — indirme/ingest
  YAPMA, önce sayıyı raporla.** Zaten aynı dataset'in tamamlanmamış bir köşesini
  tamamlamak AP-1'i ihlal etmez (yeni dataset değil), ama yine de aşağıdaki §3'teki
  büyüklük testini geçmeden ingest'e girişme.

### 2. ALFA için: matematiksel olarak veri yolu kapalı — iki seçenek insana kalır

11.85× açık, 11 normal uçuşluk sabit bir evrenle kapatılamaz. Codex burada KARAR
VERMEZ, yalnız iki seçeneği net biçimde yazıp durur:
- (a) ALFA/R6 için hedef alarm bütçesini (`configs/residual_v1_cusum.json`'daki
  `total_false_alarms_per_flight_hour`, ALFA payı) yeniden müzakere etmek — bu,
  sonucu görüp eşiği gevşetmek anlamına geleceği için AYRI bir ön-kayıt gerektirir,
  şu anki dondurulmuş sözleşmenin bir parçası değildir.
  Örnekleme veri kısıtından hedefi tersine mühendislikle sızıntı olarak sayılır.
- (b) ALFA/R6 eşik kalibrasyonunu mevcut veri/enstrümantasyonla **elde edilemez**
  olarak raporlamak — `artifacts/uav_gnss_integrity_v1/uav_gnss_integrity_v1_final_no_go_report.tex`'teki
  emsal biçimde: "sinyal var (S-3 PASS), ama kabul edilebilir alarm yüküyle
  operasyonel bir karar sınırı kurulamıyor."

**Codex'in çıktısı:** bir karar değil, bu iki seçeneği ve gerekçelerini içeren kısa
bir not (`docs/RESIDUAL_V1_ALFA_KALIBRASYON_ACIK_NOKTASI.md` gibi) — kullanıcı hangi
seçeneği onaylayacağını söylemeden hiçbiri uygulanmaz.

### 3. RFLY için: §1'in cevabına bağlı

- Eğer RflyMAD kaynağında ingest edilmemiş ek Real-No_Fault uçuş YOKSA: RFLY da
  ALFA ile aynı durumda (sabit, tükenmiş evren) — §2'deki iki seçenek RFLY/Q1/Q2
  için de aynı şekilde yazılıp insana bırakılır.
- Eğer ek uçuş VARSA: ingest etmeden önce şunu hesapla — mevcut 41 normal uçuş
  0.786237 / 41 = 0.0191765 saat/uçuş ortalaması üretiyor (development'taki
  ortalama). Gereken 4.0 saate ulaşmak için
  `(4.0 - 0.786237) / 0.0191765 = 167.588` — yani **en az 168** ek normal uçuş
  gerekir (yukarı yuvarla, kısmi uçuş olmaz) — bu, mevcut development havuzunun
  (268 uçuş) yarısından fazlası kadar yeni normal uçuş demektir. Bu sayıyı
  gerçek kaynakta bulunan ek uçuş sayısıyla karşılaştır; eğer kaynak bu kadar
  büyük değilse (muhtemel), ingest'in de açığı kapatamayacağı BAŞTAN bellidir —
  yine §2'nin iki seçeneğine düşülür. Yalnız kaynak gerçekten yeterince büyükse
  (≥168 ek uçuş), ingest planı ayrı bir onaya çıkarılır (bu da AP-2 "genişlik>derinlik"
  disipliniyle tek seferde, ölçülüp raporlanarak yapılır — sessizce büyütülmez).

## Kesin yasaklar (Faz E raporunun tekrarı, netlik için)

- Test veya sealed holdout'u bu açığı kapatmak için AÇMA.
- `configs/residual_v1_cusum.json`'daki hedefi (ya da `k`, `z_clip`, `block_s` gibi
  kalibrasyon parametrelerini) sonucu gördükten sonra sessizce değiştirip yeniden
  koşma — her değişiklik yeni bir ön-kayıt girdisi ister.
- Mevcut `20260717_113330_phaseE_cusum_calibration_seed11` altındaki
  `DO_NOT_USE_THRESHOLDS.md` işaretli eşikleri hiçbir üretim/karar kodunda kullanma.

Bu belge bir talimattır, kod veya veri değişikliği içermez.
