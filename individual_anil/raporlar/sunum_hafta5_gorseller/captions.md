# Hafta 5 sunumu — görsel notları ve doğrulanmış sayılar

Bu klasördeki tüm PNG'ler gerçek repo artefaktlarından üretildi/kopyalandı;
hiçbir sayı tahmini değildir. Kaynak dosya yolları aşağıda her görselin
altında verilmiştir.

---

## 1. rflymad_frozen_ae_tcn_hedefli_karsilastirma.png

**Açıklama:** İki farklı model ailesinin gerçek dünya performans hedefine ne
kadar yakın olduğunu gösterir. Noktalar gölgeli/mavi bölgenin dışında kaldıkça
hedeften uzak demektir. Kritik alarmda dondurulmuş otomatik-kodlayıcı hedefe
görece yakın, ama danışma (advisory) alarmında iki yöntem de hedefin dışında
kalıyor.

**Kaynak:** `artifacts/rfly_full/v2/normal_temporal_ae/robustness/approved_20260722_nested_v1/candidates/R4/gate_summary.json`
(`frozen_baseline` alanı) + `artifacts/rfly_full/v2/supervised_tcn/development_5fold_20260722_v1/aggregate_metrics.csv`.
Hedef çizgileri dondurulmuş geliştirme kapısı eşikleridir (kritik: recall≥%50,
FA≤2/sa; danışma: recall≥%60, FA≤12/sa).

**⚠ Genlik-baskınlığı kontrolü:** Bu iki yöntem (dondurulmuş otomatik-kodlayıcı
ve zamansal ağ) için bu tur içinde eğitilmiş-model-vs-eğitilmemiş-model
karşılaştırması (SEAD çalışmasında bulunan "genlik-baskınlığı" artefaktının
kontrolü) **çalıştırılmadı**. Bu "geçti" değil, "ölçülmedi" demektir —
sayılar gerçek ama bu spesifik sağlamlık testi eksik.

---

## 2. rflymad_tcn_5kat_kararliligi.png

**Açıklama:** Aynı yöntemi 5 farklı veri bölümünde (fold) tekrar eğitip test
ettiğimizde sonuçların ne kadar değiştiğini gösterir. Büyük iniş-çıkışlar
(ör. gerçek-uçuş yakalama oranının bir bölümde ~0'a düşüp başka bir bölümde
~%50'ye çıkması) yöntemin hangi uçuşları gördüğüne çok bağlı olduğunu, henüz
kararlı/güvenilir olmadığını gösteriyor.

**Kaynak:** `artifacts/rfly_full/v2/supervised_tcn/development_5fold_20260722_v1/outer_fold_metrics.csv`

---

## 3. rflymad_ae_tcn_karsilastirma.png

**Açıklama:** İki farklı model mimarisini (otomatik-kodlayıcı = AE, zamansal
evrişimli ağ = TCN) aynı 5-katmanlı test düzeninde yan yana karşılaştırır.
Zamansal ağ rüzgarlı senaryoda daha az yanlış alarm üretiyor, ama bunu genel
yakalama oranından ve danışma alarmındaki alarm yükünden (yaklaşık 3,4 kat
artış) ödün vererek yapıyor.

**Kaynak:** `artifacts/rfly_full/v2/normal_temporal_ae/robustness/approved_20260722_nested_v1/candidate_comparison.csv` +
TCN `aggregate_metrics.csv` (yukarıdaki gibi).

---

## 4. rflymad_uzun_egitim_real_tradeoff.png

**Açıklama:** Modeli gerçek uçuş verisine daha uzun süre alıştırdığımızda
gerçek arızaları yakalama oranı artıyor (motor arızası %23→%29, sensör arızası
%22→%27), ama bunun bedeli genel yakalama oranının düşmesi (%59→%55) ve her
kategoride yanlış alarm sayısının artması. Yani "daha uzun eğitim" şu ana
kadar net bir kazanç değil, bir değiş-tokuş.

**Kaynak:** `artifacts/rfly_full/v2/normal_temporal_ae/robustness/approved_20260722_nested_v1/candidate_comparison_by_policy.csv`
(orta-süreli ince ayar ve uzun yakınsama denemesi adayları, critical politikası satırları).

---

## 5. rflymad_alarm_zaman_serisi_ornekleri.png

**Açıklama:** Üç örnek uçuşta modelin skorunun zaman içinde nasıl davrandığını
gösterir: gerçek bir motor arızasında alarm ancak arızanın sonlarına doğru ve
marjinal biçimde tetikleniyor; arızasız bir gerçek uçuşta yine de yanlış alarm
üretebiliyor; rüzgarlı bir simülasyonda ise sistemde hiçbir arıza yokken
tekrarlayan alarm veriyor. Görsel örnektir, kapı/kabul kararı değildir.

**Kaynak:** `artifacts/rfly_full/v2/normal_temporal_ae/robustness/approved_20260722_nested_v1/base/rotation_0/development_scores.parquet`
(script: `scripts/RFLYMAD_rfly_full_v2/render_rfly_full_v2_alarm_timeseries.py`,
başlığından iç kod adı temizlenerek yeniden render edildi).

---

## Doğrulanan sayılar (kullanıcı hafızasından — hepsi repodaki artefaktlarla BİREBİR eşleşti, fark bulunmadı)

| Sayı | Kaynak dosya | Doğrulama |
|---|---|---|
| Dondurulmuş AE, kritik: %60,43 recall / 1,28 FA-sa | `candidates/R4/gate_summary.json` → `frozen_baseline` | ✅ eşleşti |
| Dondurulmuş AE, danışma: %69,84 / 3,64 FA-sa | `candidate_comparison_by_policy.csv` (frozen_baseline, advisory) | ✅ eşleşti |
| Dondurulmuş AE, wind FA-sa: kritik 28,46 / danışma 31,54 | aynı dosyalar | ✅ eşleşti |
| Uzun yakınsama denemesi, kritik: %54,61 recall | `candidates/R4/gate_summary.json` | ✅ eşleşti |
| Uzun yakınsama denemesi, tüm-arızasız FA: 5,24/sa | aynı | ✅ eşleşti |
| Uzun yakınsama denemesi, gerçek-uçuş ortalama recall: %28,11 | aynı | ✅ eşleşti |
| Uzun yakınsama denemesi, gerçek-arızasız FA: 12,98/sa | aynı | ✅ eşleşti |
| TCN geliştirme, 5 katman: kritik %28,87 / 2,87 FA-sa | `supervised_tcn/.../aggregate_metrics.csv` | ✅ eşleşti |
| TCN geliştirme: danışma %67,86 / 12,54 FA-sa | aynı | ✅ eşleşti |
| TCN geliştirme: wind FA-sa kritik 6,41 / danışma 27,03 | aynı | ✅ eşleşti |
| TCN geliştirme: gerçek-uçuş ortalama recall %7,56 | aynı (kritik satırı, n=3) | ✅ eşleşti |
| ADS-B irtifa kuralı: 57 uçuşun 2'sinde tetiklendi, 0 doğrulanmış anomali | `artifacts/adsb/simple_anomaly_20260722/summary.json` | ✅ eşleşti |
| ADS-B rota kuralı: 95 uçuşun 13'ünde, 24 olay, 24/24 düşük-hız artefaktı | aynı dosya | ✅ eşleşti |

## Bu hafta HENÜZ tamamlanmayan / eklenemeyen kısımlar

1. **ADS-B `contextual_physics_v2` (kural+model hibrit) sonucu** — İKİ eğitim
   denemesi de bu hafta içinde başarısız oldu, ikisinde de tek bir epoch bile
   tamamlanmadı, hiçbir model veya skor üretilmedi:
   - 1. deneme (23 Temmuz): ~6 saat 36 dakika hesaplama sonrası bir kod
     hatasıyla (NumPy/PyTorch indeksleme sınırı) çöktü.
   - 2. deneme (24 Temmuz, bugün): kod hatası düzeltildi ve yeniden
     başlatıldı, ama eğitim sürerken repoya git commit'i yapıldığı için
     (koşunun kendi bütünlük kontrolü bunu bir ihlal sayıp süreci kendi
     kendine durdurdu) yine epoch 1 tamamlanmadan durduruldu.
   Tamamlandığında eklenecek: recall/FA grafiği + v1 (önceki deneme)
   karşılaştırması + genlik-baskınlığı sonucu (bu kapı zorunlu, sonuç
   `false` çıkmadan hiçbir sayı paylaşılmamalı).
2. **Uçuş-modu (flight-phase) teşhisi** — RflyMAD Real-uçuş verisinde model
   skorlarının uçuş fazına göre nasıl ayrıştığını inceleyen, bir önceki
   turun sonunda "önerilen sıradaki tek adım" olarak yazılan teşhis henüz
   BAŞLATILMADI (repoda bu isimde/amaçta hiçbir script veya çıktı yok).
   Çalıştırıldığında eklenecek: evre-bazlı dağılım görseli.

## Genlik-baskınlığı (magnitude-domination) kontrolü — durum

Bu hafta üretilen RflyMAD skorlarının (6 sağlamlık adayı + TCN 3-epoch sanity
+ TCN 5-katmanlı geliştirme turu) HİÇBİRİ için eğitilmiş-model-vs-rastgele/
eğitilmemiş-model karşılaştırması bu turda çalıştırılmadı — kodda arandı,
böyle bir kontrol bulunamadı. Bu "FLAGGED" (sorunlu bulundu) değil, "kontrol
edilmedi" demektir; ama kural gereği (SEAD'de bu atlanınca sorun olmuştu)
bu boşluk açıkça belirtiliyor, sayılar "temiz" gibi sunulmuyor. ADS-B tarafında
`contextual_physics_v1` (önceki deneme) bu kontrolden GEÇMİŞTİ (rho=0,65);
`contextual_physics_v2` bu hafta hiç bir skor üretmediği için kontrol
uygulanamadı.
