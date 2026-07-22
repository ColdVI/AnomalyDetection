# RflyMAD v2 görsel teşhis paketi

Bu klasör yalnız **development** telemetri değerlerinden üretilmiştir. Kilitli
testin yalnız manifestteki uçuş sayıları `01` grafiğinde gösterilir; test
özellikleri korelasyon, PCA, t-SNE veya k-NN hesabına sokulmamıştır.

- Development telemetri satırı: 486,877
- Uçuş-düzeyi temsil: 4,633
- PCA öncesi medyan/IQR özelliği: 52
- PCA ilk iki bileşen açıklanan varyans: %33.11
- k-NN fold-0 keşif doğruluğu: %44.93
- Ortalama 5-NN aynı arıza ailesi oranı: %77.27
- Ortalama 5-NN aynı domain oranı: %94.84
- Normal fold-0 doğrulama dağılımı: {'Real': 0, 'HIL': 40, 'SIL': 0}
- Normal fold 1–4 eğitim dağılımı: {'Real': 41, 'HIL': 160, 'SIL': 200}

## Doğru yorum

Korelasyon ilişkiyi, t-SNE yerel komşuluğu, k-NN ise mevcut temsilin yakınlık
yapısını gösterir. Bunlar operasyonel anomali başarısı değildir. Özellikle
t-SNE adalarının domain/oturum izleyip izlemediği, arıza ailesi görünümünden
birlikte değerlendirilmelidir. `08` k-NN matrisi development içi ve
grup-temelli fold-0 teşhisidir. `10` içindeki Dense AE ve TCN matrislerinin
veri kapsamları farklıdır; yan yana çizilmeleri doğrudan performans kıyası
anlamına gelmez.

Normal-only yaklaşımda yalnız development NoFault uçuşları öğrenme/eşik için
kullanılmalı; arıza etiketleri strict novelty değerlendirmesine kadar saklı
tutulmalıdır. Bilinen arızalar için etiketli TCN hattı ayrıca yürütülür.
Mevcut fold-0 normal doğrulaması yalnız HIL içerdiği için domain-bazlı eşik
kalibrasyonu yapılmadan bu split nihai eğitim sözleşmesi olarak kullanılmamalıdır.

Seçilen temsil kolonları: `local_x__median, local_y__median, local_z__median, local_vx__median, local_vy__median, local_vz__median, local_ax__median, local_ay__median, local_az__median, roll_deg__median, pitch_deg__median, yaw_deg__median, act_roll__median, act_pitch__median, act_yaw__median, act_thrust__median, output_mean__median, output_std__median, output_range__median, battery_remaining__median, vel_test_ratio__median, pos_test_ratio__median, hgt_test_ratio__median, mag_test_ratio__median, gps_eph__median, gps_epv__median, local_x__iqr, local_y__iqr, local_z__iqr, local_vx__iqr, local_vy__iqr, local_vz__iqr, local_ax__iqr, local_ay__iqr, local_az__iqr, roll_deg__iqr, pitch_deg__iqr, yaw_deg__iqr, act_roll__iqr, act_pitch__iqr, act_yaw__iqr, act_thrust__iqr, output_mean__iqr, output_std__iqr, output_range__iqr, battery_remaining__iqr, vel_test_ratio__iqr, pos_test_ratio__iqr, hgt_test_ratio__iqr, mag_test_ratio__iqr, gps_eph__iqr, gps_epv__iqr`
