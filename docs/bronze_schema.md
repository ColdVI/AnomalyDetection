# Bronze Schema

Kaynak kolonlari ve degerleri degistirilmeden korunur.

| Kolon | Anlami |
|---|---|
| `_source_type` | Kaynak kimligi |
| `_ingest_ts_utc` | Batch UTC ingestion zamani (ISO 8601) |
| `_source_file` | Orijinal dosya, arsiv uyesi veya API URI'si |
| `_schema_version` | Baslangicta `bronze_v1` |

Tablosal ciktilar Snappy Parquet'tir. Realtime adsb.lol payload'lari ayrica ham JSONL
olarak `_landing/` altinda tutulur. ALFA/UAV Attack etiket kolonlari incelenen gercek
arsiv yapilarina gore loader fazinda burada belgelenecektir.
