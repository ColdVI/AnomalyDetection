# Bronze Schema

> **ESKİ (ADR-002 dönemi).** ADR-003 (docs/PIPELINE_PLAN.md, 2026-07-01) Bronze'u
> "sadece ham dosya yükle, hiç parse/provenance yok" olarak değiştirdi — provenance
> artık Silver'da ekleniyor (bkz. `src/common/provenance.py`, varsayılan
> `schema_version="silver_v1"`). Bu dosya üç kişinin kaynaklarını da kapsayan
> `docs/schema.md`'ye (PIPELINE_PLAN'ın istediği) henüz taşınmadı/birleştirilmedi — sadece
> bu not eklendi. Aşağıdaki içerik ADR-002 dönemindeki (artık geçerli olmayan) Bronze
> provenance sözleşmesini anlatıyor, referans/tarihi kayıt olarak bırakıldı.

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

Tüm Bronze nesneleri yerel diske değil, doğrudan MinIO'ya (`bronze` bucket) yazılır;
bkz. ADR-002.
