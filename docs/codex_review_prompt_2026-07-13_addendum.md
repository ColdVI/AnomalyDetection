# Ek Not — 2026-07-13 (codex_review_prompt_2026-07-13.md'ye ek)

Kullanıcı `docs/codex_review_findings_2026-07-13.md`'yi onayladı: önerilen 9 adımlı sıralama
AYNEN uygulanacak. Bu dosya yalnız TEK bir değişiklik ekliyor — sıralamanın kendisine dokunmuyor.

## Değişiklik: holdout adayı artık 1 değil 3 dosya

Önceki rapor, Downloads'ta tek bir kapalı-tutulması-gereken aday tanımlamıştı
(`v2025.06.15-planes-readsb-prod-0-003.tar`). Bugün (2026-07-13) o dosyayla AYNI klasörde
**iki yeni tar daha belirdi**:

| Dosya | Byte | mtime | Not |
|---|---:|---|---|
| `v2024.09.01-planes-readsb-prod-0.tar` | 2,084,157,440 | 2026-07-13 15:13:26 | ad kalıbı diğer 3 orijinal günle (`-prod-0.tar`, shard eki yok) aynı |
| `v2025.02.15-planes-readsb-prod-0.tar` | 2,146,856,960 | 2026-07-13 15:13:03 | aynı, shard eki yok |
| `v2025.06.15-planes-readsb-prod-0-003.tar` | 3,093,094,400 | 2026-07-13 12:02:04 | önceki raporda zaten kayıtlı; `-003` shard eki VAR — tam gün mü parça mı belirsizliği önceki raporda zaten not edilmişti |

Bu üç dosyanın hiçbiri açılmadı, listelenmedi, hashlenmedi. Yukarıdaki tablo yalnız dosya
sistemi metadata'sıdır (boyut + mtime) — önceki rapordaki "açmadan" tanımıyla aynı disiplinde.

## İstenen

1. Önceki raporun **9 adımlı planı değişmeden** uygulanır (adım 1-8 bu üç dosyaya dokunmaz).
2. **Adım 9** ("holdout freeze manifesti") artık TEK dosya için değil, **üç dosyalık bir
   havuz** için tasarlanmalı. Üçü de aynı şekilde: raw yol/byte/SHA-256/mtime salt-okunur
   manifestte kilitlenir, içerik açılmaz.
3. Üç günün takvimde birbirinden çok uzak olması (2024-09, 2025-02, 2025-06 — 2026-02-28 fit
   gününden sırasıyla ~18, ~12, ~8 ay önce) kapsamlı bir zamansal-kaymayı test etme fırsatı —
   ama BUNU nasıl kullanacağını (hangisi önce açılır, hepsi mi tek seferde mi, farklı roller mi
   verilir) ÖNCEDEN, sonuç görülmeden Codex'in kendi bir sonraki raporunda ÖNERMESİ isteniyor;
   bu dosya bir seçim DAYATMIYOR.
4. Ad kalıbındaki `-003` tutarsızlığı (yalnız 06-15'te var) ilk mekanik kontrolde (raporun
   adım 4'ü, "başarısız şema/parser dış-geçerlilik sonucu olarak kaydedilir" ilkesiyle)
   açıklığa kavuşturulmalı — üçü de aynı mantıkla mı üretilmiş, yoksa 06-15 gerçekten bir
   shard mı, netleştirilmeden "tam gün" varsayılmasın.

Geri kalan her şey (dokunulmaz kısıtlar, rapor formatı, sayı-uydurmama kuralı) önceki
prompt dosyasıyla aynı şekilde geçerli.
