# RFLY-0: RflyMAD Entegrasyon Planı (paralel veri hattı)

Durum: ÖN-KAYIT (2026-07-09). SEAD zincirinden (ML-14→15→16→17) BAĞIMSIZ paralel
hat; donmuş artifact'lara ve SEAD gate'lerine dokunmaz. ML-17 adı KULLANILMAZ
(endgame'e rezerve).

## §0 Gerekçe ve kapsam kararı

- **Neden:** SEAD'in zayıf kategorileri (mechanical 41, altitude 73, global 40)
  havuz-tükenmiş (E.1). RflyMAD `Real-Motor`/`Real-Sensors`, tam bu kategorilere
  ilk kez YENİ GERÇEK veri getirir (497 gerçek uçuş; PX4 ULog — mevcut
  `parse_uav_sead.py`/`build_px4_features` hattına doğal uyum).
- **uav_attack kararı:** ana gate'lerden düşer, "siber/güvenlik karakterizasyonu"
  olarak belgeli kalır (veri/testler silinmez). ALFA "küçük sabit-kanat referansı"
  olarak kalır.
- **SIL/HIL ilk turda YOK** (kendi H4/D.4 dersimiz: SITL karışması). İlk ürün
  iddiası: "gerçek-uçuş multirotor motor/sensör/no-fault".
- Lisans: yalnız ticari olmayan kullanım (Rfly grubu) — dokümante edildi.

## §1 İndirme (uygulandı: `src/ingestion/rflymad_downloader.py`)

Kaggle mirror (`xianglile/rflymad`) zip-shard değil patlatılmış ağaç →
**case-bazlı seçici indirme**: yalnız `Log/*.ulg` + `TestInfo.csv`
(TLog/TrueData xlsx atlanır — boyut kontrolü). Disiplin uav_sead ile aynı:
skip-existing/resume, checkpoint'li listing (429 backoff), Bronze'da
`bronze/rflymad/<orijinal yol>` + case-bazlı `manifest.json` (boyut+sha256).
Sıra: SampleData → Real-NoFault → Real-Motor → Real-Sensors.

## §2 Parser ve Silver (Codex işi)

`src/silver/parse_rflymad.py`: pyulog ile `.ulg` → mevcut PX4 Silver şeması
(`parse_uav_sead` kalıbı; ortak yardımcılar yeniden kullanılır, kopyalanmaz).
Etiket kaynağı: `TestInfo.csv` (fault tipi/parametre/zamanı) + ULog içi
`rfly_ctrl_lxl` mesajı (fault id/parametre — enjeksiyon zaman aralığı buradan).
`label` eşlemesi (SABİT): Real-NoFault→`normal`, Real-Motor→`motor_fault`,
Real-Sensors→alt tipe göre `sensor_<tip>_fault`. `source_id` = case yolu.
Kabul: SampleData ile smoke (şema + kapsam raporu: kaç case parse edildi/atlandı).

## §3 Split ve kör holdout (ilk günden)

`splits.py` mevcut mantığıyla `rflymad` kaynağı eklenir: session bazlı
(TestCase kökü/uçuş günü), normal-only train, 5 seed, **final_holdout_fraction
= 0.30 ilk günden** (SEAD'deki gibi; ALFA/UAT'ın holdout'suzluğu tekrarlanmaz).
Holdout seed'i sabit; holdout hiç açılmaz.

**Amendman R1 (2026-07-09, sonuçlar görülmeden):** RflyMAD Kaggle mirror ve
essential-only Bronze/Silver sayımı `Real-No_Fault=51` normal uçuşta tavan
gösteriyor; eski 30/30 kota 61 normal istediği için uygulanabilir değil.
Bu nedenle ilk resmi RFLY-0 development kotası `n_val=12`, `n_test_normal=12`
olarak donduruldu. Türetim: düşük-kota ön koşudaki 15/15 çalışma noktası,
61→51 normal tavanına oransal indirilip aşağı yuvarlandı (`floor(15*51/61)=12`);
böylece 27 normal train kalır. Bu amendman mevcut §3 kararını silmez; yalnız
RFLY-0'ın ilk resmi koşusunda uygulanacak normal kota düzeltmesidir.

## §4 Değerlendirme (ayrı gate'ler, SEAD'e karışmaz)

Mevcut kazanan kalıp aynen: modüler IF (PX4 modülleri) + ince-modül adayları +
karar katmanları + (hazır olunca) ML-15 kalibrasyon sarmalayıcısı.
- Gate R-A: holdout izolasyonu, part-çoğalması yok, donmuş SEAD dizinleri
  bayt-bayt değişmedi.
- Gate R-B: kategori bazlı (motor/sensör) — baseline mevcut PX4 modül füzyonu;
  kural ML-9 ile aynı (≥0.05 + ≥3/5 seed).
- Gate R-C: aynı sabit operasyonel bütçeler (critical 2 / advisory 12 FA-saat).
Artifact: `artifacts/rfly0/rflymad/<run>/` checksum'lı manifest.

## §5 Sıralama

RFLY-0, SEAD ML-14/15/16 koşularını BLOKLAMAZ; Codex önce ML-14 §1→§7 +
ML-15'i bitirir, RFLY-0 parser'ına ondan sonra (veya indirme sürerken boş
kaldıysa SampleData smoke'una) başlar. Bulgular H-numarası yerine R-numarası
alır (R1, R2, ...); ADR ve yetersizlikler kaydına aynı disiplinle işlenir.
