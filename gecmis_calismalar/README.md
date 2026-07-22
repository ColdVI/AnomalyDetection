# Geçmiş Çalışmalar — ALFA / UAV-Attack / UAV-SEAD / RFLYMAD

Bu klasör, 2026-07-22'de yapılan dosyalama/toparlama çalışmasının ürünüdür. Amaç:
haftalarca süren anomali tespiti denemelerini (4 dataset + paylaşılan altyapı) tek
bir yerde, dataset × dosya-türü bazında düzenli biçimde bulunabilir kılmak. Ayrıntılı
gerekçe ve envanter için bkz. `.claude/plans/fluttering-waddling-frost.md` (plan
dosyası) — burada sadece güncel harita özetlenir.

## Klasör haritası

| Klasör | İçerik |
|---|---|
| [`_ortak/`](_ortak/) | Paylaşılan/çapraz-dataset materyal: eski ML-0..16 kütüphanesi (`legacy_ml_kutuphanesi/`), RESIDUAL-V1 birleşik ALFA+RFLYMAD raporu (`raporlar/RESIDUAL_V1.md`), hafta-3 çapraz-dataset sunum görselleri (`gorseller_sunum_hafta3/`) |
| [`ALFA/`](ALFA/) | Gerçek sabit-kanat İHA arıza verisi. `kaynak_kod_legacy/` (eski parser'lar, ölü kod), `egitim_yontemleri_ve_modeller/` (ML-8A + CUSUM baseline), `raporlar/` (görseller). **Canlı ALFA ingest kodu `residual_v1/` paketindedir** (bkz. RFLYMAD altındaki not) |
| [`UAV_ATTACK/`](UAV_ATTACK/) | Hiç canlı kodu yok — sadece `kaynak_kod/` altında git-geçmişinden restore edilmiş ölü parser + `raporlar/` (görseller, CUSUM baseline). Ham/silver/gold verisi hâlâ `data/` altında |
| [`UAV_SEAD/`](UAV_SEAD/) | Hiç canlı kodu yok — `kaynak_kod/` (ölü parser+downloader), `egitilmis_modeller/` (ML-8A..16, dense/lstm/usad SEAD modelleri, CUSUM baseline) |
| [`RFLYMAD/`](RFLYMAD/) | En büyük dataset — `legacy_rfly0_1/` (ölü kod) + `raporlar/` (12 RFLYMAD_V2_*.md doküman, Codex sohbet kaydı, TCN/convergence görselleri, çalıştırılmış notebook). **3 paralel canlı hat** (`residual_v1`, `uav_gnss`, `rfly_full`/`rfly_dl`) kendi paketleri olarak `gecmis_calismalar/` kökünde düz yerleşimde — bkz. `RFLYMAD/VERI_ARTEFAKT_KONUMU.md` |

## Önemli notlar

- **`data/` (ham/silver/gold, ~30GB) ve büyük üretilmiş `artifacts/` ağaçları
  FİZİKSEL OLARAK TAŞINMADI** — `.gitignore`'daki path-anchored kurallarla
  çakışmayı önlemek için yerinde bırakıldı. Her dataset klasöründe bir
  `VERI_ARTEFAKT_KONUMU.md` dosyası bu verinin gerçek konumuna işaret eder.
- **`scripts/` ve `tests/` kök isimleri değişmedi** — içlerinde dataset alt
  klasörleri açıldı (`scripts/RFLYMAD_rfly_full_v2/`,
  `tests/_ortak_residual_v1_ALFA_RFLYMAD/` gibi). `pytest.ini`deki
  `testpaths = tests` hâlâ geçerli, hiçbir config değişikliği gerekmedi.
- **Canlı Python paketleri** (`residual_v1/`, `uav_gnss/`, `anomaly_core/`,
  `rfly_full/`, `rfly_dl/`) bu klasörün **kökünde düz yerleşimde**
  (`gecmis_calismalar/residual_v1/` gibi) — import yolları
  `gecmis_calismalar.<paket>...` olarak güncellendi. `adsb/` paketi kullanıcı
  kararıyla **taşınmadı**, repo kökünde kaldı (yeni ADS-B çalışması onun
  üzerine kurulacak).
- Üst düzey proje raporları (`docs/PROJE_SUREC_VE_SONUC.md`, `docs/decisions.md`,
  `docs/final_rapor_ml_fizibilite_2026-07-16.md`,
  `docs/proje_basarisizlik_analiz_raporu.md/.tex`) **fiziksel olarak taşınmadı**,
  `docs/` kökünde kalmaya devam ediyor — bugünkü RFLYMAD bulgularıyla
  güncellendi, içerik burayı referans verir.
