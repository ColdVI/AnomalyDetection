# RflyMAD-Full v2 — uygulanan deney sözleşmesi

## Karar

Mevcut 1 Hz Dense Autoencoder sonucu tarihsel baseline olarak dondurulmuştur.
Model tek başına ana aday değildir. Yeni çalışma, eski çıktıyı silmeden ayrı bir
`RflyMAD-Full v2` hattında yürütülür.

Nihai operasyonel iddia ancak bütün veri parse edildikten, kilitli test setine
yalnız bir kez bakıldıktan ve en az beş grouped-CV sonucu üretildikten sonra
yapılabilir. Kısmi havuz veya tek epoch sonucu yalnız `smoke_only` sayılır.

## Uygulanan düzeltmeler

1. `dataset_manifest.parquet/csv` içinde her satır bir ULog uçuş adayıdır.
2. Exact ULog SHA-256 eşitliği aynı `canonical_case_id` ve split grubuna gider.
3. Wind, sistem arızası değil `environment_robustness` setidir.
4. `fault_family` yanında `fault_subtype` korunur.
5. Kilitli test yaklaşık yüzde 20'dir; kalan development havuzu grouped 5-fold
   ataması taşır. Aynı scenario/Real oturumu iki splite giremez. Mevcut atamalar
   `split_registry.json` içinde dondurulur; yeni veri eski grupları oynatmaz.
6. V2 parser 10 Hz ortak eksen kullanır. Merge işlemleri causal/backward yapılır.
7. Yüksek frekanslı IMU ve motor çıkışlarından mean, std, RMS, peak-to-peak ve
   first-difference RMS özetleri üretilir.
8. Aktif aralıkta önce `rfly_ctrl_lxl`, yalnız bulunamazsa `TestInfo` fallback
   kullanılır. İki kaynak uyuşmazlığı ayrıca işaretlenir.
9. Supervised TCN yalnız NoFault pencerelerini negatif, tamamı fault-active olan
   pencereleri pozitif eğitim örneği yapar. Transition, pre/post ve Wind eğitim
   loss'una girmez.
10. Girdi, train-only median/IQR ölçekli sinyaller ile eksiklik maskesinin ayrı
    kanallarından oluşur.
11. TCN iki head üretir: binary anomaly ve koşullu fault-family. Sınıf dengesi
    balanced sampler ve family ağırlıklarıyla ele alınır.
12. Binary skor validation'da temperature scaling ile kalibre edilir. Alarm
    kararı örnek sayısına değil zamana bağlıdır: son 6 saniyenin 4'ü ve 30 saniye
    refractory.

## Mevcut doğrulanmış durum

- Manifestte 5.754 indirilen/parçalanan uçuş adayı vardır.
- Exact ULog hash duplicate bulunmamıştır.
- Scenario/oturum gruplu registry 1.105/5.754 uçuşu, yani yüzde 19,20'yi
  kilitli teste ayırmıştır.
- 3.895 uçuşun eski 1 Hz etiketi geçici TestInfo kaynağına dayanmaktadır; bunlar
  v2 parser tamamlandıkça `rfly_ctrl_lxl` önceliğiyle yeniden üretilir.
- 20 uçuşta eski parse içinde aktif fault aralığı yoktur ve kalite problemi
  olarak tutulur.
- 10 Hz v2 smoke havuzu 128 uçuşa ulaşmıştır; toplam havuz henüz tamamlanmamıştır.

## Dense AE teşhisi

Wind sistem arızası recall hesabından çıkarıldığında donmuş Dense AE:

- 4.337 sistem-arızalı uçuşun 121'ini yakalamıştır.
- Event recall yüzde 2,79'dur.
- Pooled pencere AUROC 0,556; AUPRC 0,498 ve pozitif prevalans 0,483'tür.
  AUPRC'nin taban orana yakınlığı sıralama gücünün de zayıf olduğunu gösterir.
- Yalnız 72 NoFault test uçuşunda 0 FA/saat görülmüştür.
- Fault uçuşlarının pre/post normal maruziyeti de sayıldığında 5,55 FA/saat
  görülmüştür.
- Wind sağlamlık akışında 2,87 alarm/saat oluşmuştur.
- Validation-normal quantile cetvelindeki en düşük-FA kritik aday dahi 4,08
  FA/saatte yüzde 1,75 recall üretmiş; 2 FA/saat bütçesi sağlanamamıştır.
- Advisory bütçesine en yakın doğrulanmış nokta 11,72 FA/saatte yüzde 20,04
  recall üretmiştir; yüzde 50 hedefinden uzaktır.

Bu sonuç threshold sorununun ötesinde skor örtüşmesine işaret eder. Dense AE
baseline olarak saklanır; ancak ana model veya hibrit bileşen olabilmesi için
leave-one-family-out testinde hedef FA bütçesinde anlamlı recall göstermesi
gerekir.

## Supervised TCN smoke sonucu

128 uçuşluk kısmi 10 Hz havuz, bir epoch ile yalnız entegrasyon testi amacıyla
çalıştırılmıştır:

- Eğitim: 57 uçuş / 704 dengelenmiş pencere
- Validation: 25 uçuş
- Kilitli testte mevcut altküme: 29 uçuş
- Kritik politika: 2/18 event, yüzde 11,11 recall, 9,56 FA/saat; iki kapı da geçilmedi
- Advisory politika: 5/18 event, yüzde 27,78 recall, 9,56 FA/saat; recall hedefi geçilmedi

Bu değer performans iddiası değildir. Voltage ve Load henüz yoktur; v2 parse
tamamlanmamıştır; yalnız bir epoch kullanılmıştır. Çıktı `smoke_only` olarak
etiketlenmelidir. Test altkümesinde yalnız bir Real arıza uçuşu vardır; bu tek
uçuşun yakalanması Real transfer kanıtı değildir.

## Çalıştırma sırası

```powershell
.venv\Scripts\python.exe scripts\build_rfly_full_v2_manifest.py --print-summary
.venv\Scripts\python.exe scripts\parse_rfly_full_v2_10hz.py
.venv\Scripts\python.exe scripts\run_rfly_full_v2_supervised.py --epochs 12
.venv\Scripts\python.exe scripts\run_rfly_full_ae_diagnostics.py
```

Tam veri sonrasında yapılacak zorunlu deneyler:

1. Beş development fold'u için supervised TCN.
2. Motor, Sensor, Propeller, Voltage ve Load için leave-one-family-out.
3. Simulation-only → Real transfer.
4. Real-only ve simulation-pretrain + Real fine-tune karşılaştırması.
5. Hibrit karar yalnız AE held-out-family kapısını geçerse; skorlar kalibre
   edilmeden ham toplama yapılmaz.

## Fizibilite kapısı

- Kritik: en az yüzde 30 event recall ve en fazla 2 FA/saat.
- Advisory: en az yüzde 50 event recall ve en fazla 12 FA/saat.
- Metrikler flight/event/window seviyelerinde ayrı raporlanır.
- SIL/HIL başarısı Real başarı sayılmaz.
- Farklı grouped splitlerde kararlılık ve Real transfer gösterilmeden proje
  uygulanabilir ilan edilmez.
