# RFLY-1: Olay-Aralığı Etiket Düzeltmesi + Simülasyon Kapasite Testi

Durum: ÖN-KAYIT (2026-07-09, sonuçlar görülmeden sabitlendi).
Bağımlılık: RFLY-0'ın mevcut çıktıları (`artifacts/rfly0/**`); SEAD ana zincirini
(ML-15/16) BLOKLAMAZ, paralel iş. ML-17 endgame'e girmez.

## §0 Gerekçe

RFLY-0'ın "resmi" gate koşusu (`artifacts/rfly0/rflymad/official_full`) bir
metodoloji kusuru üzerine kuruluydu: `parse_rflymad.py` her arızalı uçuşu
**baştan sona** anomali sayıyor ("whole-flight proxy"), oysa SEAD'in
event-onset-recall metriği kısa, spesifik bir aralığı yakalamayı ölçüyor. Bu iki
sayı (RFLY 0.749 vs SEAD 0.13) yan yana konamaz; RFLY'nin sayısı gerçekte
olduğundan kolay görünüyor.

Kontrol edildi: gerçek arıza başlangıç/bitiş zamanı **zaten indirilmiş `.ulg`
dosyalarının içinde** duruyor — `rfly_ctrl_lxl` uORB mesajı, `Fault injection
time` / `Test end time` alanlarıyla (bir HIL-Wind örneğinde doğrulandı: enjeksiyon
t=24s, log sonu t=69s). Yeni indirme GEREKMİYOR, yalnız `.ulg` içinden bu mesajı
okuyan bir parse adımı eksik. Ayrıca `Real-*` alt kümesinin `TestInfo_*.xlsx`
dosyaları (arıza parametresi/zamanı içerebilir) mevcut indiriciyle **hiç
indirilmemiş** — filtre kalıbı doğru görünüyor ama sonuç boş; bu ayrıca
araştırılıp düzeltilecek (§1.3).

Ayrıca kullanıcı talebiyle simülasyon verisinin (kontrollü kullanım şartıyla)
devreye alınması: Kaggle mirror'ında `SIL-Wind` (443 case) + `HIL-Wind` (443
case) = 886 case var — bunlar motor/sensörün RÜZGAR ALTINDA test edildiği
simülasyon/donanım-döngülü-simülasyon kayıtları (ör. `HIL-Wind/acce-wind/...`).
H4/D.4 dersi gereği bu veri **gerçek-uçuş normal havuzuyla asla karıştırılmaz**;
tamamen ayrı, kendi başına değerlendirilen bir iz.

## §1 Önkoşul — gerçek arıza aralığını çıkar (RFLY-0'ı da düzeltir)

1. `src/silver/parse_rflymad.py`'ye `rfly_ctrl_lxl` mesajını `.ulg`'den okuyan
   bir adım eklenir (pyulog; SEAD'in kendi olay-aralığı ayrıştırma kalıbıyla
   tutarlı). Çıktı: her arızalı case için `(fault_onset_s, fault_end_s)` —
   `label` alanı DEĞİŞMEZ, yalnız yeni bir `fault_interval` alanı eklenir.
2. **Kabul testi (sonuç görülmeden sabit):** bilinen bir örnekte (yukarıdaki
   HIL-Wind örneği) çıkarılan `(onset, end)` = `(24, 69)` ile birebir eşleşmeli
   — `tests/test_rflymad.py`'ye altın-değer testi eklenir.
3. `TestInfo_*.xlsx` dosyalarının `Real-*` alt kümesinde neden indirilmediği
   araştırılır (`rflymad_downloader.py::is_essential_file`); ya filtre düzeltilir
   ya da "gerekli değil, `.ulg` yeterli" diye belgelenir — sonuç görülmeden karar
   verilir, hangisi olursa olsun.
4. `scripts/run_rfly0_exploratory_evaluation.py` (ve varsa official runner)
   `_truth_mask`'te whole-flight yerine `fault_interval`'ı kullanacak şekilde
   güncellenir. **RFLY-0'ın resmi gate'i (`official_full`, hem rfly-only hem
   pooled) bu düzeltmeyle YENİDEN koşulur** — eski "0.749 recall" sayısı resmi
   kayıttan düşer, yerine düzeltilmiş sayı gelir. ADR-014 bir düzeltme notuyla
   güncellenir (eski sayı silinmez, "whole-flight proxy nedeniyle üstten tahmin,
   düzeltildi" diye işaretlenir).

## §2 İz A — Simülasyon-yalnız kapasite testi

Soru: aynı yöntem (aynı 8+2 modül, aynı max-füzyon, aynı karar katmanları),
GERÇEK darboğaz olmadan (bol örnek, aynı taxonomy) operasyonel hedefi geçebiliyor
mu? Cevap ne olursa olsun bilgi verir — geçerse "asıl sorun gerçek veri azlığı",
geçmezse "asıl sorun yöntemin kendisi".

1. `src/ingestion/rflymad_downloader.py --download --subsets SIL-Wind,HIL-Wind`
   (mevcut essential-only filtre; yeni kod değişikliği gerekmez, yalnız yeni
   subset adı).
2. `parse_rflymad.py` §1'deki düzeltmeyle bu case'leri de parse eder; `label`
   eşlemesi: fault tipi (ör. `acce-wind`) → `sensor_<tip>_fault_wind` (SABİT,
   gerçek RFLY etiketleriyle karışmasın diye ayrı önek).
3. Kendi train/val/test split'i (SEAD/RFLY-gerçek ile HİÇ karışmaz, ayrı
   `source="rflymad_sim"`), 5 seed, session/case bazlı.
4. `scripts/run_rfly1_simulation_capability.py` (yeni): aynı modül tanımları,
   aynı karar katmanları, aynı sabit operasyonel bütçe (kritik 2 / öneri 12
   FA-saat). **Gate S-A (zorunlu):** holdout izolasyonu, gerçek-veri hiçbir
   aşamada okunmadı/karışmadı (assert). **Gate S-B (bilgi amaçlı, "geçti/kaldı"
   değil "kapasite sinyali var/yok"):** herhangi bir satır kritik≥0.30@≤2 VEYA
   öneri≥0.50@≤12 sağlıyor mu — SABİT rapor, sonuca göre yorumlanır ama eşik
   değiştirilmez.

## §3 İz B — Şiddet taraması (mevcut donmuş modeli test et, yeniden eğitme)

Soru: elimizdeki gerçek-veri-eğitimli model (ML-14 donmuş model), arıza ne kadar
BARİZ olursa fark etmeye başlıyor? Eğitime hiç dokunmadığı için risksiz.

1. **Preflight (sonuç değil, dağılım incelemesi):** SIL-Wind/HIL-Wind
   `TestInfo.csv`'lerindeki `Fault Parameter` alanının gerçek değer aralığını
   çıkar (kaç farklı şiddet seviyesi var, nasıl dağılıyor) — bu adımdan SONRA
   şiddet-kova sınırları (ör. tertile/quartile) sonuç görülmeden sabitlenir.
2. `scripts/run_rfly1_severity_sweep.py` (yeni): ML-14'ün DONMUŞ split_00
   modelini (`artifacts/ml14/uav_sead/full_matrix/split_00/models/`) joblib ile
   yükler — **yeniden fit YOK**, statik testle kanıtlanır (`"IsolationForest"
   not in source` kalıbı, ML-13'teki gibi). Simülasyon case'lerini şiddet
   kovasına göre gruplar, her kovada donmuş modelin skor/alarm davranışını
   raporlar (recall değil — donmuş model gerçek FA bütçesine kalibre değil,
   yalnız "skor şiddetle birlikte monoton artıyor mu" ve "hangi kovadan sonra
   val-normal üstü skor üretmeye başlıyor" ölçülür).
3. Çıktı: şiddet → tespit-eğilimi tablosu/grafiği. Resmi bir Gate değil, tanı
   raporu.

## §4 Dosyalar, testler, kabul

| Dosya | İş |
|---|---|
| `src/silver/parse_rflymad.py` | `rfly_ctrl_lxl` → `fault_interval` (§1) |
| `src/ingestion/rflymad_downloader.py` | TestInfo.xlsx filtre incelemesi (§1.3) |
| `scripts/run_rfly0_exploratory_evaluation.py` | `_truth_mask` düzeltmesi (§1.4) |
| `scripts/run_rfly1_simulation_capability.py` (yeni) | §2 |
| `scripts/run_rfly1_severity_sweep.py` (yeni) | §3 |
| `tests/test_rflymad.py` | altın-değer interval testi (§1.2) |
| `tests/test_rfly1_simulation.py` (yeni) | holdout/karışma-yok assert, sim-gerçek ayrımı, determinizm |
| `tests/test_rfly1_severity_sweep.py` (yeni) | statik "yeniden eğitim yok" testi, donmuş model checksum kimliği |
| `docs/decisions.md` | ADR-014 düzeltme notu + ADR-016 (bu faz) |
| `docs/ML_YETERSIZLIKLER_KAYDI.md` | whole-flight-proxy'nin nasıl bulunup düzeltildiği kaydı |

Kabul: tam `pytest -q` yeşil (bilinen 4 MinIO hariç); RFLY-0'ın resmi sayıları
düzeltilmiş etiketle yeniden üretildi ve ADR-014 buna göre güncellendi; İz A ve
İz B SEAD'in gerçek sayılarına hiçbir şekilde karışmadı (ayrı artifact
dizinleri: `artifacts/rfly1/simulation_capability/`,
`artifacts/rfly1/severity_sweep/`); kör holdout (ne SEAD'in ne RFLY'nin) hiç
açılmadı; commit'ler co-author'suz.

## §5 Sıralama

§1 (etiket düzeltmesi) ÖNCE — hem kendi başına değerli (RFLY-0'ın mevcut yanlış
izlenimini düzeltir) hem §2/§3'ün doğru çalışması için gerekli. §2 ve §3
birbirinden bağımsız, paralel yapılabilir. SEAD'in ML-15/16 zincirini
bloklamaz.
