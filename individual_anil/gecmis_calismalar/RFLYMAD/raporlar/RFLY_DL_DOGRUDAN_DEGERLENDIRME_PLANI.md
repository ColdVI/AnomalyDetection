# RflyMAD Doğrudan Derin Öğrenme Değerlendirmesi — Ön Kayıt

Durum: **DONDURULDU — koşum öncesi sözleşme** (2026-07-20).

Bu çalışma, RflyMAD üzerinde daha önce doğrudan genel arıza tespiti için
çalıştırılmamış üç derin öğrenme ailesini aynı veri ve karar sözleşmesi altında
karşılaştırır: **LSTM-AE, Dense-AE ve USAD**. Arşiv kodu veya modeli aktif hatta
import edilmez; çalışma `rfly_dl/` temiz namespace'inde yürütülür. Arşiv yalnız
tarihsel split ve metrik sözleşmesinin kaynağıdır.

## 1. Veri ve ayrım sözleşmesi

- Girdi: `data/gold/ml_features/rflymad/rflymad_ml_features.parquet`.
- Arıza aralığı: `data/silver/rflymad_silver.parquet` içindeki
  `fault_onset_s`, `fault_end_s`, `fault_interval_source` alanları.
- Split: `data/gold/ml_features/split_manifest.json` içindeki beş uçuş-bazlı
  RflyMAD split'i aynen kullanılır.
- Her split: 27 normal train, 12 normal validation, 12 normal test ve geçerli
  303 arızalı test uçuşu. `rfly_ctrl_lxl_no_active_fault` kayıtları testten
  çıkarılır.
- Model ve scaler yalnız normal train uçuşlarına fit edilir. Validation yalnız
  erken durdurma ve alarm kalibrasyonu içindir. Test eğitim/kalibrasyona girmez.
- Eski `final_holdout` uçuşları çalışmaya alınmaz. Ancak 2026-07-20 portföy
  incelemesinde tüm parquet üzerinde toplu şema/sayım okuması yapıldığı için bu
  küme bu yeni çalışma açısından artık “hiç okunmamış kör holdout” olarak iddia
  edilmeyecektir; yalnız **hariç tutulan tarihsel holdout** olarak raporlanır.

## 2. Girdi kanalları ve preprocessing

Etiket, klasör adı, fault id/mode, arıza zamanı veya arızayı doğrudan bildiren
kontrol mesajı modele verilmez. Sabit 35 telemetri/üretilmiş kanal kullanılır:

- hareket/irtifa: GPS hız ve ivme, düşey hız, yerel irtifa/düşey hız ve hız
  tutarlılık residual'ları;
- tutum/kontrol: roll-pitch açı ve oranları, yaw oranı, setpoint/rate hataları,
  toplam tutum hatası, roll-pitch-yaw-thrust komutları, efor ve control strain;
- kestirim/aktüatör: EKF test oranları, motor çıkış dağılımı ve konum doğruluğu;
- titreşim: üç vibration kanalı.

Her split'te medyan ve IQR yalnız train-normal satırlardan hesaplanır. Eksik
değerler 0 ile modele taşınır fakat ayrı maske ile kayıptan çıkarılır. Sonlu
ölçeklenmiş değerler, SEAD'de görülen ham-genlik baskınlığını sınırlamak için
`[-10, 10]` aralığına kırpılır. Pencere 50 örnek, stride 5 örnek ve izin verilen
en büyük zaman boşluğu 2 saniyedir. Pencere skoru yalnız pencere sonunda
gözlenebilir ve 1 Hz karar ızgarasına geçmişe doğru taşınır; gelecek bilgisi
kullanılmaz.

## 3. Modeller

- **LSTM-AE:** 32 gizli birimli encoder/decoder ve 16 boyutlu latent temsil.
- **Dense-AE:** düzleştirilmiş pencere, 6 gizli birim ve 4 latent boyut.
- **USAD:** ortak encoder + iki decoder; 6 gizli ve 4 latent boyut; standart
  iki-fazlı adversarial eğitim ve çıkarımda `alpha=beta=0.5`.

Ortak eğitim: Adam, öğrenme oranı `1e-3`, batch 64, en çok 40 epoch, patience 5,
gradient clipping 1.0. En düşük normal-validation reconstruction kaybı seçilir.
Her split kendi seed'iyle ayrı model üretir.

## 4. Skor, karar ve metrik sözleşmesi

Reconstruction skoru normal-validation ampirik CDF'i ile `[0,1]` uzayına
kalibre edilir. Her model için üç karar katmanı çalıştırılır:

1. doğrudan threshold,
2. K-of-N (`2/3` veya `3/5`, validation bütçesine göre),
3. tek yönlü CUSUM (30 s refractory, 60 s bloklu 50 saat moving-block bootstrap).

Operasyonel hedefler tarihsel Rfly cetveliyle aynıdır:

- critical: event recall ≥ 0.30 ve FA/saat ≤ 2,
- advisory: event recall ≥ 0.50 ve FA/saat ≤ 12.

Ana metrik arıza-aralığı event recall'dır; ek olarak FA/saat, ilk alarm gecikmesi,
uçuş düzeyi TP/FN/FP/TN, motor/sensör alt-grup recall'ı ve pencere AUROC/AUPRC
raporlanır. AUC operasyonel başarı yerine geçmez.

## 5. Zorunlu tanılar ve karar kuralı

Her model/split için eğitilmiş skor; rastgele başlatılmış aynı mimarinin skoru
ve ham ölçeklenmiş girdi büyüklüğüyle Spearman korelasyonuna sokulur. `rho ≥ 0.8`
ham-genlik/rastgele-init baskınlığı uyarısıdır. Bu uyarı varsa yüksek AUC tek
başına öğrenme kanıtı sayılmaz.

Beş split tamamlanmadan sonuç “tam koşum” sayılmaz. Herhangi bir model-karar-
bütçe satırı ilgili recall ve FA/saat şartını birlikte sağlamazsa operasyonel
gate başarısızdır. Sonuç görüldükten sonra aynı run içinde feature, pencere,
mimari, threshold veya bütçe post-hoc değiştirilmez.
