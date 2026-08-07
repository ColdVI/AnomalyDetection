# RflyMAD-Full v2 truth audit (Aşama A)

Bu rapor `dataset_manifest.parquet` yerine v2 10 Hz parse çıktısının kendi `truth_source`/`fault_active`/`truth_crosscheck_disagreement` kolonlarından üretildi. Manifest'in truth-ile-ilgili alanları hâlâ eski 1 Hz parser'a (`rfly_full/pipeline.py`, TestInfo-önce/rfly_ctrl_lxl-fallback) dayanıyor; v2 parser (`rfly_full/v2_parser.py`) önceliği tersine çevirdi (`rfly_ctrl_lxl`-önce, TestInfo-fallback). Bu iki öncelik SIRASI farklı olduğu için manifest özetindeki eski `provisional_testinfo_truth` sayısı gerçek v2 dağılımını yansıtmıyor.

- Denetlenen uçuş: **6605**
- Parse state: `complete` (6605 completed, 0 failed)
- Truth schema: v2 (2712 selectively invalidated/reparsed)
- Manifest split_group_id join kaçırma: 0

## Gerçek v2 truth_source dağılımı

| truth_source | uçuş |
|---|---:|
| rfly_ctrl_lxl | 6069 |
| normal_no_fault | 531 |
| missing | 5 |

- Sistem-arızalı uçuş: 5188
- `rfly_ctrl_lxl` kaynaklı uçuş (crosscheck-uygun): 6069
- Crosscheck disagreement (`rfly_ctrl_lxl` vs TestInfo, >%1 örnek uyuşmazlığı): 5577 / 6069
- Eksik aktif aralık (`system_fault=True` ama hiç `fault_active` yok): 5
- **İlk örnekten itibaren aktif (t=0'dan başlıyor, şüpheli sentinel-değer sorunu adayı): 0 / 6069**

  Paket kırılımı:

  | package | uçuş |
  |---|---:|

  Bu sayaç truth-quality guard'ıdır. Truth schema v2, `SIL_Motor_*`, `HIL_Motor_*`, `SIL_Prop` ve `HIL_Prop` paketlerini canonical domain ile yorumlar; önceki sahte t=0 yoğunluğunun kök nedeni underscore paket adlarının yanlış domain/sentinel seçmesiydi. Düzeltme sonrasında burada kalan paketler ayrı ham ULog incelemesi gerektirir.

- Aktif aralık negatif/taşma ihlali: 0 (beklenen: 0)
- V2_FEATURES şema eksikliği olan uçuş: 0 (beklenen: 0)

## Eksik aktif aralık — domain/aile kırılımı

| domain | fault_family | uçuş |
|---|---|---:|
| Real | Motor | 5 |

## Near-duplicate audit (iki katmanlı heuristik)

**Tier 1 — `duration_signature`**: `(domain, fault_family, fault_subtype, süre~0.1s, satır sayısı)` eşleşmesi. Bu tier TEK BAŞINA GÜVENİLİR DEĞİL: bu denetim sırasında en büyük kümeler elle incelendi ve SIL/HIL'de standardize batch test protokollerinin (ör. birçok farklı `SIL-Sensors/<çift>` senaryosu) aynı sabit uçuş süresini paylaştığı, ama FARKLI senaryolar olduğu görüldü — yani bu tier aşırı-kümeleniyor ve yalnız bağlam için tutuluyor.

**Tier 2 — `trajectory_signature`**: Tier 1 + 5 eşit aralıklı örnekte kaba `local_x/y/z` konum parmak izi de eşleşmeli. Sızıntı riski iddiası bu tier'den okunmalı; yine de kriptografik hash değil, kaba bir içerik imzasıdır — kesin duplicate kanıtı değildir.

- Tier 1 (duration_signature) küme sayısı: 1285 (bilgi amaçlı, güvenilmez)
- Tier 2 (trajectory_signature) küme sayısı: 0
- **Tier 2'de locked_test ve development'a birlikte yayılan küme (gerçek sızıntı riski adayı): 0**

Ayrıntılar: `near_duplicate_clusters.csv`, `truth_audit_per_flight.csv`, `truth_source_by_domain_family.csv`.
