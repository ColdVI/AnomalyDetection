# Rejected ADS-B Attempts Archive

Arşiv tarihi: 2026-07-10
Karar: kullanıcı mevcut ADS-B yaklaşımını baseline olarak kabul etmedi ve sıfırdan
başlama istedi.

## Arşivlenen iki yaklaşım

1. `src/adsb/`: ADSB-0/ADSB-1 adıyla hazırlanan segmentasyon, fizik residual'ları
   ve enjeksiyon taslağı.
2. `src/adsb_behavioral/`: ayrı geliştirilen robust-rule, Isolation Forest ve
   hard-physics yaklaşımı.

Planlar, testler, çalıştırıcılar ve tüm `adsb_behavioral_stage1` artifact'leri aynı
klasörde korunmuştur.

## Neden kabul edilmedi?

- İlk V1 raporunda bariz anomaliler için sıfıra yakın recall üretildi. Bunun bir
  kısmı zero-MAD ölçekleme ve alarm-onset/aktif-durum değerlendirme hatasıydı.
- Düzeltme sonrasında sentetik kolay anomalilerde `%97.6` event recall görüldü,
  fakat doğal veride `25.54 yeni alarm/saat` oluştu. Bu, kullanılabilir bir binary
  detector değildir.
- Yüksek sentetik recall, gerçek ADS-B anomaly başarısı olarak yorumlanamaz.
- Modelden önce veri kalitesi, örnekleme aralıkları, uçuş segmenti, gözlenebilir
  anomaly tanımı ve değerlendirme birimi ortak biçimde kilitlenmemişti.
- Aynı probleme iki namespace ve iki planla başlanması sonucu anlaşılabilirliği
  bozdu.

## Yeniden kullanım kuralı

Bu klasördeki hiçbir model, eşik veya metrik yeni baseline sayılmaz. Yeni çalışma
buradan kod kopyalamadan başlayacaktır. Yalnız ham tar okuma/parsing davranışı,
ayrı testle doğrulanırsa aktif veri altyapısına yeniden uygulanabilir.
