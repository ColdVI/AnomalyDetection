# Four-dataset probabilistic event evaluation v1 — ön-kayıt

Tarih: 2026-07-28

Bu ön-kayıt, tamamlanmış `four_dataset_probabilistic_gpu_v2` modellerini
yeniden eğitmeden değerlendiren yeni bir katmana aittir. V2 ön-kaydını veya
çıktılarını değiştirmez.

Makine tarafından okunabilir sözleşme:
`configs/four_dataset_probabilistic_event_eval_v1.json`.

## Birincil amaç

Uçuş etiketiyle bütün pencereleri pozitif sayan ve tek maksimum skorla bütün
uçuşu alarm kabul eden mevcut raporun yerine üç seviyeyi ayırmak:

1. ham pencere skoru;
2. süreklilik şartı taşıyan nedensel alarm eventi;
3. offline uçuş inceleme özeti.

## Dondurulmuş kalibrasyon ve event ızgarası

- Eşik adayları yalnız normal-validation skorlarının
  `[0.99, 0.995, 0.9975, 0.999, 0.9995, 0.9999]` quantile'larıdır.
- Yanlış event bütçeleri saatte `[0.1, 0.5, 1, 2, 5, 10]` eventtir.
- Minimum kesintisiz eşik-üstü süreleri `[0, 0.5, 1, 2, 5]` saniyedir.
- İki eşik-üstü bölüm arasındaki en fazla 2 saniyelik boşluk aynı eventte
  birleştirilir.
- Her persistence×bütçe hücresinde normal-validation bütçesini sağlayan en
  düşük aday eşik seçilir. Hiçbiri sağlamazsa hücre `unavailable` kalır;
  extrapolation yapılmaz.
- En az beş bağımsız normal-validation uçuşu olmayan veri seti için
  operational/generalization iddiası kurulmaz.

Tüm ızgara hücreleri raporlanır. Test sonucu görülerek tek bir hücre seçilmez.

## Truth kuralı

Whole-flight label hiçbir zaman interval truth olarak kullanılmaz.

- RflyMAD: `fault_active/condition_active` ve doğrulanmış manifest zamanları.
- ALFA: ham `failure_status` başlangıcının Gold zaman eksenine doğrulanmış
  eşlemesi.
- UAV-SEAD: veri seti tarafından sağlanan zaman-bölgesi etiketlerinin Gold
  zaman eksenine doğrulanmış eşlemesi.
- UAV Attack: mevcut Gold'da zaman aralığı yoktur; event/window-localization
  metriği üretilmez.

## Metrik ayrımı

Birincil event metrikleri event recall, normal false-event/saat, median/p90
gecikme ve süre-ağırlıklı range precision/recall'dır. Uçuş ROC-AUC değerleri
ikincildir. Max, mean, q99 ve top-1% mean birlikte gösterilir; test sonrası
seçim yapılmaz.

Point adjustment yasaktır. Bir anomalili aralıkta tek alarm noktası bulunması
bütün aralığı otomatik doğru saymaz.

## Değiştirilemezlik

Bu turda model ağırlıkları, checkpoint/epoch, scaler, özellik sırası, split,
v2 alarm quantile'ı veya v2 raporları değiştirilmez. Yeni katmanın eşik ve
event parametreleri test skoruna göre değiştirilmez. Sonuç NO-GO ise aynen
raporlanır.

## Ek A — ilk skor exportundan önce raporlama kapsamı

Tarih: 2026-07-28. Bu ek yazılırken hiçbir Colab v2 run artifacti kabul
edilmemiş, hiçbir validation/test pencere ledgerı üretilmemiş ve yeni evaluator
ile hiçbir test skoru görülmemiştir.

Kullanıcının sağladığı ikinci metodoloji incelemesi üzerine uçuş-özeti tablosu
yalnız betimleyici olmak şartıyla genişletildi: `q99.5`, `top-0.5% mean` ve
`top-2% mean`, önceden kayıtlı max/mean/q99/top-1% mean değerlerinin yanına
eklendi. Ayrıca her event-grid hücresinde alarm-time fraction, en uzun alarm
eventi ve event/saat raporlanacaktır. Bu ek seçim kuralı değildir; v2 test
sonucundan kazanan agregasyon seçmek yasaktır.

İkinci inceleme ayrıca source-safe splitin session/campaign-safe olmakla aynı
şey olmadığını gösterdi. V2 sonuçları değişmeden kalır; yeni eğitim yapılacaksa
önce ayrı bir group-safe v3 split sözleşmesi hazırlanacaktır.
