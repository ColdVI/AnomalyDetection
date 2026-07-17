# RESIDUAL-V1 Görev 4.1 Şartlı GO Kaydı

Tarih: 2026-07-17  
Karar kaynağı: Görev 1.1–3.5 bağımsız denetim raporu  
Durum: Görev 4.1 için şartlar yerine getirildi; Faz E için K5 açık.

## Uygulanan şartlar

### K1 — bağlam geçmişi sızıntısı

ResidualChannelSpec komut ve bağlam girdilerini artık ayrı rollerle taşır.
Yalnız gerçek komutlar tri4 pencereleri ile delta_1s alır; bağlam girdileri
yalnız anlık __last değeriyle matrise girer. Yanıt/yanıt-geçmişi kilidi iki
girdi rolünü de denetler.

- Eski descriptor hash:
  b6ac3412db3f6c8229cfadd37e542c6c7b29c7b89f42bd8fb5388e6032c0f93d
- K1 sonrası descriptor hash:
  86cd49485995b4779934c6b02cf85a26bf4cf303d59c987600ed00a1080c80ca

Eski feature artefaktları kanıt zinciri olarak korunur; Görev 4.1 yalnız yeni
hash ile yeniden üretilen feature matrislerini kabul eder.

### K2 — ALFA engine/R4 beklentisinin düzeltilmesi

Sanity kanıtında engine onset anında throttle yüksek kalmıyor, sıfıra düşüyor.
Bu nedenle G1, yavaşlamanın bir bölümünü beklenen tepki olarak tahmin edebilir
ve R4 residual kayması ön-kayıttaki beklentiden küçük çıkabilir. Muhtemel sinyal
cruise + throttle=0 + sabit airspeed_cmd dağılım-dışılığıdır. Bu kayıt Görev
4.1 için başarı iddiası değildir; S-3 zayıf çıkarsa önceden kaydedilmiş açıklama
adayıdır.

### K3 — ALFA kapsam sınırı

Önceki development feature koşusunda 32 uçuşun yalnız 5'inde R1–R5 satırı
vardı; 101.913 aday satıra karşı 32.577 satır tutuldu. Development'taki 10
engine olayının yaklaşık 4'ü R4 için kullanılabilir. ALFA test bölmesinde tek
normal uçuş bulunduğundan ALFA-özel FA/uçuş-saati tahmini savunulabilir
genişlikte değildir; doğal FA bütçesinin ana dayanağı RFLY olacaktır.

**ALFA headline iddiası tek test oturumuna dayanır ve holdout'ta R1–R5 kapsamı beklenmez.**

Bu sınır nedeniyle boş coverage başarı ya da başarısızlık olarak
yorumlanmayacak; her kanal raporunda uçuş, oturum ve satır coverage'ı ayrıca
verilecektir.

### K4 — PWM trim merkezleme

Arızalı ALFA uçuşlarında aileron/elevator/rudder PWM deltaları artık yalnız ilk
arıza onset'inden önceki örneklerin medyanıyla merkezlenir. Onset öncesinde
sonlu trim örneği yoksa ingest fail-closed davranır. Normal uçuşlarda tüm uçuş
medyanı kullanılmaya devam eder.

### K5 — waypoint maskesi

R6 için waypoint değişimi çevresindeki ±2 saniye maskesi henüz yoktur. R6,
Görev 4.1 G1 regresyonunun dışında tutulur. Bu maske Faz E başlamadan önce
zorunlu olarak uygulanacaktır.

### K6 — R6 doğrudan kanal

R6 ridge ile eğitilmeyecek ve G1 metriklerine katılmayacaktır. Görev 5.1'de
öğrenmesiz biçimde doğrudan robust-z kanalına dönüştürülecektir.

## Deney disiplini

- G1 hiperparametre seçimi yalnız development içindeki oturum fold'larında
  yapılır.
- Test ve holdout feature/telemetrisi model seçimine girmez.
- Holdout açma kilidi bu aşamada kullanılmaz.
- S-3 geçmezse eşik kalibrasyonuna gidilmez; AP-1 gereği çıktı yeni veri seti
  arayışı değil hata analizi raporudur.
