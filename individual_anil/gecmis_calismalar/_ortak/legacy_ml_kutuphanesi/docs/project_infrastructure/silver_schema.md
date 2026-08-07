# Silver Schema (src/processing/* -- REFERANS, AKTİF PIPELINE DEĞİL)

> **Güncelleme (2026-07-01, ADR-004):** Metehan'ın `docs/PIPELINE_PLAN.md`'i (ADR-003)
> mimariyi Bronze=raw/Silver=parse olarak değiştirdi. Bu dosyanın anlattığı
> `src/processing/alfa_silver.py`/`uav_attack_silver.py`/`gold.py` artık AKTİF PIPELINE
> DEĞİL — o rolü `src/silver/parse_alfa.py`/`parse_uav_attack.py` üstlendi (daha dar bir
> kolon kümesiyle). Bu dosya, ADR-004'te açıklandığı gibi, ileride Silver şeması
> zenginleştirilmek istenirse referans olarak tutuluyor. Aşağıdaki her şey hâlâ gerçek
> veriyle doğrulanmış ve doğru, ama "şu an çalışan pipeline" değil.

Bkz. `docs/decisions.md` ADR-003/ADR-004 — bu katman, ekibin Bronze review'undan önce,
Anıl'ın bireysel anomali tespiti çalışması için başlatıldı. Hem ALFA hem UAV Attack tarafı
artık gerçek veriyle doğrulandı (2026-07-01 — UAV Attack, `UAVAttackData.zip` Desktop'a
eklenip çıkarıldıktan sonra aynı oturumda doğrulandı).

## ALFA (`src/processing/alfa_silver.py`) — doğrulandı

Bronze'daki her `alfa/` objesi (bir sekansın bir topic'i) okunur, `_alfa_scenario`'ya göre
gruplanır, ve her sekans için:

- Referans (backbone) zaman ekseni: `mavctrl-rpy` (paper'ın 50 Hz referansı) varsa o, yoksa
  en çok satırlı sensor topic.
- Diğer her topic, `merge_asof(direction="nearest", tolerance=250ms)` ile backbone'a bindirilir.
  Kolonlar `<topic>__<orijinal_kolon>` olarak adlandırılır (ör. `mavros-nav_info-roll__field.commanded`).
  ROS header boilerplate'i (`header.seq/stamp/frame_id`) atılır. **Dar bir alt küme değil,
  bulunan tüm topic'ler** birleştirilir (geniş ama seyrek tasarım — feature seçimi modelleme
  aşamasına bırakılıyor).
- `failure_status-*` topic'leri sensöre karışmaz; sadece etiket kaynağıdır.

| Kolon | Anlamı |
|---|---|
| `<topic>__<col>` | Her topic'in kendi kolonları, topic adıyla önekli |
| `ts_ns` | Sekans-içi ROS zamanı (int64 ns, backbone eksen) |
| `timestamp_utc` | `ts_ns / 1e9` (Unix epoch saniye) |
| `sequence_id` | Sekans klasör adı (ör. `carbonZ_2018-07-18-15-53-31_1_engine_failure`) |
| `source_type` | Sabit `"alfa"` |
| `fault_type` | Normalize edilmiş etiket (aşağıdaki tablo) |
| `is_fault` | `failure_status-*` onset'inden itibaren `True` |
| `_alfa_failure_label` | Ham klasör-adı son-eki (traceability) |
| `_alfa_emergency_traj` | `_with_emr_traj` son-eki var mı |

### `fault_type` normalize tablosu

Gerçek 47 sekans klasörünün (2026-07-01'de tek tek listelendi) tam kapsamlı eşlemesi
(`normalize_fault_label`, `alfa_silver.py`):

| Ham son-ek (sayısal önek atılmış) | `fault_type` |
|---|---|
| `no_ground_truth` | `no_ground_truth` |
| `no_failure` | `no_failure` |
| `engine_failure`, `engine_failure_with_emr_traj` | `engine_failure` |
| `elevator_failure` | `elevator_failure` |
| `rudder_left_failure` | `rudder_left_failure` |
| `rudder_right_failure` | `rudder_right_failure` |
| `left_aileron_failure` | `left_aileron_failure` |
| `right_aileron_failure`, `right_aileron_failure_with_emr_traj` | `right_aileron_failure` |
| `both_ailerons_failure`, `left_aileron__right_aileron__failure` | `both_aileron_failure` |
| `rudder_zero__left_aileron_failure` | `rudder_aileron_failure` (paper: "Rudder & Aileron at Zero") |
| (tanınmayan yeni son-ek) | heuristik + uyarı log'u |

Gerçek veride doğrulanmış dağılım (47 sekans, `scripts/run_alfa_local.py` çıktısı,
2026-07-01): `engine_failure`=23, `no_failure`=10, `right_aileron_failure`=3,
`elevator_failure`=2, `both_aileron_failure`=2, `rudder_right_failure`=2,
`left_aileron_failure`=2, `no_ground_truth`=1, `rudder_left_failure`=1,
`rudder_aileron_failure`=1 (toplam 47). ALFA paper'ının Table 1'i ile engine/rudder/elevator/
no_failure+no_ground_truth birebir eşleşiyor; aileron alt-kategorilerinde paper'a göre ±1
fark var (paper: Left=3/Right=4/Both=1) — gerçek klasör adları bu şekilde geldiği için kod
tarafı doğru, muhtemelen paper'ın yayınlanmasından sonra dataset küçük bir revizyon görmüş.

### Bilinen veri kalitesi notu

`diagnostics-*` topic'i (ROS `diagnostic_msgs/DiagnosticArray`, serbest-form key/value alanları)
sekanslar arasında karışık tip (sayı + string) üretebiliyor; `build_alfa_silver` bunu
`_coerce_mixed_object_columns` ile kolon bazında ya sayıya çeviriyor ya da string'e sabitliyor
(hiçbir zaman kolonu sessizce silmiyor).

## UAV Attack (`src/processing/uav_attack_silver.py`) — doğrulandı

Bronze'daki `_source_file` yolundan log_id/topic çıkarılır — **bilinen 5 topic adına
(`vehicle_global_position`, `vehicle_global_position_groundtruth`, `vehicle_attitude`,
`battery_status`, `vehicle_gps_position`) ankraj edilmiş tam-eşleşme regex'iyle**, genel bir
"son `_<kelime>_<n>.csv`" sezgisiyle değil. Gerçek dosya adlarının log_id kısmı standart
değil (`log_12_2020-8-2-14-18-24_...`, `ace-benign-log_0_2033-8-19-16-27-30_...`,
`001-2021-01-27-09-08-37-708_...`) ve kendi içinde alt çizgi barındırıyor; bu yüzden eski
`parse_uav_attack.py`'nin kullandığı sezgisel regex ilk alt çizgiden bölüyordu (yanlış).

`vehicle_global_position` omurga; `vehicle_attitude` (quaternion→derece), `battery_status`,
`vehicle_gps_position` (jamming/hdop/vdop/satellites + ham `lat`/`lon`, MAVLink `x1e7`
ölçeklemesinden derece'ye çevrilmiş — `vehicle_global_position`'ınkinden FARKLI, zaten
derece cinsinden gelen kolonlarla karıştırılmasın diye `raw_gps_lat`/`raw_gps_lon`),
`vehicle_global_position_groundtruth` (sadece Simulated/SITL — simülatörün gerçek konumu,
GPS spoofing residual feature'ı için `gt_lat`/`gt_lon`/`gt_alt`) zenginleştirmesi eklenir.
Etiketler (`_attack_label`, `_attack_type`, `_attack_platform`, `_attack_collection`) zaten
Bronze'da var, yeniden çıkarılmıyor.

**Bellek notu:** Bronze, log başına ~30 uORB topic'i (bazıları çok yüksek frekanslı —
`sensor_combined`, `actuator_outputs`, `ekf2_innovations`) ayrı obje olarak tutuyor.
`read_layer` ile hepsini önce birleştirip sonra filtrelemek gerçek veride ~25M satır x
~50 kolon için 9.4 GB ayırmaya çalışıp patladı; `build_uav_attack_silver` bunun yerine her
objeyi tek tek okuyup SADECE yukarıdaki 5 bilinen topic'i tutuyor, geri kalanı hemen atıyor.

Gerçek veride doğrulanmış çıktı (`scripts/run_uav_attack_local.py`, 683.9 MB
`UAVAttackData.zip`, 767 CSV, 2026-07-01): **19 log, 79.646 satır, 34 kolon**
(1 log — `log_46_2020-8-2-19-18-26` — `vehicle_global_position` eksik olduğu için atlandı).
`_attack_label`: malicious=54.575 satır, benign=25.071 satır. `_attack_type`: ping_dos=29.200,
normal=25.071, gps_spoofing=24.269, gps_jamming=1.106. `_attack_platform` (log bazında):
PX4-H480/PLANE/QUAD-SITL/VTOL/TAIL-SITL 3'er, `live` 3, PX4-QUAD-HITL 1. 18/19 log gerçek
UTC zaman damgasına sahip (`vehicle_gps_position.time_utc_usec` üzerinden).

## Gold (`src/processing/gold.py`) — `common_uav_events`

ALFA ve UAV Attack Silver tabloları `source_type` ayracıyla UNION edilir (JOIN değil — zaman
eksenleri ve feature uzayları alakasız). Eksik kaynak nazikçe atlanır + loglanır.
