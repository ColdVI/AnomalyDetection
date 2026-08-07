# Çalıştırma

**Tüm komutlar için çalışma dizini `individual_anil/` olmalı** — import'lar
(`adsb.*`, `gecmis_calismalar.*`) ve göreli yollar (`configs/...`,
`artifacts/...`) buna göre kurulu (bkz. [KOD_HARITASI.md](KOD_HARITASI.md)
girişi). `cd individual_anil` yapmadan çalıştırılan komutlar import hatası
verir.

## 1. Demo — veri gerekmez, önerilen giriş

```bash
cd individual_anil
python demo/demo_calistir.py
```

~2 saniyede biter, gerçek proje modüllerini (feature/kural-skorlayıcı/CUSUM/
değerlendirme/genlik-denetimi) sentetik, fiziksel olarak tutarlı uçuşlarla
çalıştırır. Detay: [demo/README.md](demo/README.md).

## 2. Testler — veri gerekmez

```bash
cd individual_anil
python -m pytest -q
```

Repo kökünden de çalışır: `python -m pytest individual_anil/tests -q`
(ikisinin de çalışması `conftest.py`'nin doğru kurulduğunu gösterir — bkz.
[conftest.py](conftest.py)).

Tüm testler veri/ağ/Docker gerektirmez. `torch` kurulu değilse derin
öğrenme modellerini (`LSTM-AE`, `Dense-AE`, `USAD`, TCN vb.) kullanan test
dosyaları **toplanma (collection) hatası** verip atlanabilir — bu bir kod
hatası değildir, `pip install -r requirements.txt` ile `torch` kurulursa
geçer.

## 3. Gerçek veriyle çalıştırma

Gerçek veri kümeleri (yüzlerce GB) bu repoda **değil**. Beş veri kümesi ve
kaynakları:

| Veri kümesi | Kaynak |
|---|---|
| ALFA | Halka açık İHA arıza veri kümesi |
| UAV Attack | Halka açık İHA saldırı/sensör-bozulma veri kümesi |
| UAV-SEAD | İTÜ AIRLab — Hugging Face |
| RflyMAD | arXiv 2311.11340 |
| ADS-B | adsb.lol (adsb.lol/readsb tar arşivleri) |

### ADS-B hattı — tipik komut sırası

`adsb/` paketinin kullandığı ham ADS-B verisi ayrıca temin edilip
`Bronze`/`Silver` katmanına (ortak takım hattı, `src/`) yerleştirilmeli.
Ondan sonra tipik sıra:

```bash
cd individual_anil

# 1. envanter -- tar arsivini tam parse etmeden hizli profil
python scripts/adsb_inventory_report.py --help

# 2. sentetik kulliyat -- test-only enjeksiyon, truth-v2 etiketli
python scripts/adsb_build_synthetic_truth_v2_corpus.py --help

# 3. kural skorlayici -- yalniz-normal train'den kalibrasyon + degerlendirme
python scripts/adsb_evaluate_rule_scorer_truth_v2.py --help

# 4. egitim / kalibrasyon (contextual-physics v1 ornegi)
python scripts/adsb_train_contextual_physics_v1.py --help
python scripts/adsb_contextual_physics_v1_calibrate.py --help

# 5. degerlendirme
python scripts/adsb_contextual_physics_v1_truth_v2_eval.py --help
python scripts/adsb_evaluate_cusum_truth_v2.py --help

# 6. yuk raporu -- dogal trafikte alarm yuku (recall'dan AYRI, HER ZAMAN birlikte okunur)
python scripts/adsb_report_s2_natural_burden.py --help
python scripts/adsb_contextual_physics_v1_cusum_burden.py --help
```

Her script kendi argümanlarını `--help` ile listeler; girdi/çıktı yolları
`configs/` ve `artifacts/adsb/...` altında.

### Diğer hatlar

| Veri kümesi / hat | Sürücüler |
|---|---|
| RESIDUAL-V1 (ALFA + RflyMAD komut→tepki) | `scripts/_ortak_residual_v1_ALFA_RFLYMAD/` |
| RflyMAD-Full v2 | `scripts/RFLYMAD_rfly_full_v2/` |
| UAV GNSS-bütünlük pilotu | `scripts/RFLYMAD_uav_gnss/` |
| Dört-veri-kümesi olasılıksal v3/v3.1 | `scripts/build_four_dataset_probabilistic_*`, `four_dataset_probabilistic_*`, `plot_four_dataset_*` |
| Eski ML-0…ML-16 hattı (ALFA/UAV Attack/UAV-SEAD/RflyMAD) | `gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/scripts/` (arşivsel, çalıştırılabilir değil) |

GPU eğitimi/sweep gerektiren turlar için `notebooks/` altındaki Colab
defterlerine bakın (girdi/çıktı sözleşmesi ilgili `scripts/*_colab_*`
betikleriyle eşleşir).

## 4. Koşmadan sonuçlara bakmak

Kod çalıştırılmasa bile sonuçlar `artifacts/` ve `raporlar/` altında
gezilebilir:

- [`raporlar/decisions.md`](raporlar/decisions.md) — ADR karar günlüğü, en
  yoğun referans noktası.
- [`raporlar/panolar/`](raporlar/panolar/) — tarayıcıda doğrudan açılabilir
  HTML panolar (`experiment_dashboard.html`,
  `adsb_contextual_physics_v1_burden_dashboard.html`).
- [`raporlar/raporlar_html_tex/`](raporlar/raporlar_html_tex/) — mentöre
  yönelik yönetici raporu ve teknik atlas (HTML/TeX/PDF).
- [`raporlar/final_presentation/`](raporlar/final_presentation/) — final
  sunumun tüm görselleri ve slayt markdown'ları (147 dosya).
- `artifacts/adsb/`, `artifacts/rfly_full/`, `artifacts/rfly_dl/`,
  `artifacts/uav_gnss_integrity_v1/`, `artifacts/four_dataset_probabilistic_v31/` —
  kayıtlı run çıktıları, manifestler, eğitilmiş model dosyaları.

Tam commit geçmişi (adım adım hangi kararın hangi sonuçtan sonra alındığı)
için: `git checkout arsiv`.
