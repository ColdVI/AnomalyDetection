# UAV GNSS Integrity v1 — ön-kayıt

Bu çalışma genel anomaly detector veya motor-sağlığı ürünü değildir. Tek iddia,
PX4'ün mevcut EKF/GNSS telemetrisiyle RflyMAD gerçek uçuşlarındaki `ID=123456`
GNSS noise ve scale-factor arızalarının dondurulmuş gecikme ve alarm-yükü
sözleşmesi altında tespit edilebilirliğidir.

- Fit ve threshold calibration yalnız `Real-No_Fault` uçuşlarından yapılır.
- GPS klasöründe bulunmasına rağmen `ID=123455` taşıyan altı magnetometre vakası
  karantinaya alınır.
- Satır, alarm episode, event, uçuş ve scoreable-flight-hour birimleri ayrıdır.
- `not_evaluable` normal veya anomaly sınıfına çevrilmez.
- Kritik sözleşme: 5 saniye, en fazla 2 episode/uçuş-saat.
- Advisory sözleşme: 15 saniye, en fazla 12 episode/uçuş-saat.
- Yöntemler yalnız PX4-native, çok-kanallı Page CUSUM ve contextual
  location/scale LSTM'dir. Sonuç sonrası fusion veya model kataloğu genişletme yoktur.
- LSTM, trained-vs-random veya trained-vs-magnitude Spearman korelasyonu 0.80 ve
  üzerindeyse geçersiz sayılır.
- Holdout ayrı `HOLDOUT_UNSEAL.json` onayı olmadan okunamaz.
- Holdout sonucu görüldükten sonra değişiklik yeni namespace ve yeni prereg ister.

SIL/HIL `*-Wind` havuzları beş rüzgâr etkisini temsil eder; GNSS arıza
ground-truth'u veya ürün recall kanıtı değildir.

