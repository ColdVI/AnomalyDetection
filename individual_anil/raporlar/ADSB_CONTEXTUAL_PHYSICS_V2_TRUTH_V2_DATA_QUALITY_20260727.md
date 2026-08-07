# ADS-B contextual_physics_v2 truth-v2 veri-kalitesi karantinasi

Tarih: 2026-07-27

## Sonuc

Faz E baslangicinda mevcut truth-v2 corpusunun alti paired parquet dosyasindaki
26.802.690 satir denetlendi. Ayni `flight_id/timestamp_utc` anahtarinda hem
`on_ground=false` hem `on_ground=true` bulunan 60 benzersiz conflict key saptandi.
Bu kayitlar 58 ucusa aitti ve her conflict alti paired dosyada tekrarlandigi icin
artifact toplam 360 satir iceriyor.

Kullanicinin post-training karantina yetkisine uygun olarak bu 58 ucusun tamami
clean ve bes injected dosyanin hepsinden dislandi. Majority vote, last-row veya
sessiz veri duzeltmesi uygulanmadi. 8.910 paired ucustan 8.852'si Faz E
metriklerinde tutuldu.

## Karantinaya alinan ucuslar

`010276_003`, `06a1db_000`, `3444c8_002`, `348401_001`, `3c6444_004`,
`4075c2_000`, `44001b_002`, `471f3d_002`, `47a0bf_003`, `49530a_004`,
`4ca292_003`, `4d2086_004`, `4d2250_005`, `5140d2_005`, `789292_004`,
`7c29d8_003`, `7c47ac_003`, `7c4a43_006`, `7c6d9e_000`, `7c7aa3_002`,
`7c7ab6_000`, `84bb06_001`, `851120_005`, `86cec5_003`, `86d1f3_002`,
`86e86e_006`, `8990a4_000`, `a08e1e_005`, `a1b7d8_000`, `a2541e_000`,
`a353c8_002`, `a353d2_001`, `a38d8f_001`, `a4dfd2_002`, `a4fe4c_000`,
`a50e7f_002`, `a5e48a_004`, `a5e949_000`, `a6bbef_001`, `a7b133_003`,
`a8943d_006`, `aa44ac_001`, `aa54fd_002`, `aa6e81_000`, `aaad49_002`,
`ab0e4c_004`, `ab86fc_001`, `ab9d9e_000`, `abae62_004`, `abd50f_004`,
`ac33a3_003`, `c00dea_001`, `c04671_002`, `c050c5_001`, `c05220_007`,
`c07aa4_000`, `c822f9_001`, `e0b0c5_007`.

## Kanit ve butunluk

- Artifact: `artifacts/adsb/runs/20260724_contextual_physics_v2_truth_v2_eval_v1/truth_v2_on_ground_quarantine.parquet`
- Artifact boyutu: 4.992 byte
- SHA-256: `4e548531d2324a3484ede807694c94302a70081e7397775e774d513529b229ef`
- Karantinaya alinan ucus-ID listesinin canonical JSON SHA-256 degeri:
  `e7b41124a9e8c44a11b249b2f91feb623a15640f6ba9e4bd72fa3a107c176989`
- Policy: `exclude_union_of_entire_flights_with_conflicting_on_ground_at_same_timestamp_from_all_paired_corpus_files`

Bu karantina egitim, epoch, threshold, Pareto izgarasi veya persistence
parametresi degisikligi degildir. Yalnizca bozuk paired ucuslari raporlayip tam
ucus seviyesinde ayiran veri-kalitesi uygulamasidir.
