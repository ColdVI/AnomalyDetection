# Master İyileştirme Planı — Uzman Teşhisi ve Sıralı Yol Haritası

Durum: ÖN-KAYIT (2026-07-07). Bu doküman, ML-0…ML-13'ün tamamı incelendikten sonra
"parametre kalabalığını" tek bir teşhise ve sıralı bir yol haritasına indirger.
Aşağıdaki Gate eşikleri sonuç görülmeden sabitlenmiştir.

---

## 0. Dürüst teşhis — 13 fazın söylediği tek şey

Fazları tek tek değil, **örüntü** olarak okuyunca resim netleşiyor:

| Faz | Kategori sinyali | Operasyonel sonuç (Gate C) |
|---|---|---|
| ML-8A (LightGBM, supervised) | baseline'ı geçemedi | KALDI |
| ML-9 (kategori residual) | +0.02 marjinal | KALDI |
| ML-10 (Chronos forecast-residual) | mechanical **0.205→0.390** | KALDI |
| ML-12 (ince modül `itki_komutu`) | mechanical **0.459** (bilinen en iyi) | KALDI |
| ML-13 (iki kanal mimarisi) | birleşik recall arttı | KALDI (FA freni) |

**Örüntü:** Kategori-düzeyi tespit sürekli iyileşti (mechanical recall 0.205 → 0.390 →
0.459, üç fazda üst üste). Ama operasyonel Gate C **hiçbir zaman geçmedi**, ve her
seferinde sebep aynı: **yanlış alarm (FA) bütçesi**, recall değil.

**Kök neden zinciri (kanıtlı):**
1. Recall zaten var — `itki_komutu` mechanical'ı 0.459 ile ayırıyor (ML-12).
2. Ama bu skor normal uçuşlarda 38 FA-saat bırakıyor (ML-12 H30).
3. Ve val→test FA kayması sistematik: bütçe 12'ye kalibre ediliyor, test'te 23.7 çıkıyor
   (~2×, ML-10/12/13 boyunca tekrarlandı).
4. Bunun altındaki gerçek neden: **heterojen normal sınıfı** — 324 gelişt. normal uçuş
   yalnız **64 oturuma** dağılıyor (H25). Normal o kadar çeşitli ki, anomaliyi yakalayacak
   kadar sıkı her eşik, "olağandışı ama normal" uçuşlarda da çalıyor.

**Sonuç (uzman görüşü):** Bu proje 6 fazdır **bir veri problemini modelleme ile
çözmeye** çalışıyor. Feature/model/mimari mühendisliğinin hepsi aynı duvara —
FA bütçesine — çarpıyor, çünkü asıl darboğaz sabit tutulan dar normal kümesidir.

---

## 1. Kaldıraç hiyerarşisi (yüksekten düşüğe)

### Kaldıraç 1 — Normal-sınıfı zenginleştirme (EN YÜKSEK, hiç denenmedi)
Kaynak `mapping.json` (1396 uçuş) canlı doğrulandı. **502 kullanılmamış gerçek Normal
uçuş, 73 hiç görülmemiş oturum** ve **133 kullanılmamış External Position uçuş, 41
görülmemiş oturum** duruyor. Bu, H25'in işaret ettiği darboğaza (oturum çeşitliliği
64) doğrudan ilk saldırı: normal manifoldunu genişletmek → val→test FA kaymasını
küçültme ihtimali → ML-10/12'nin zaten kanıtlanmış kategori kazancının nihayet Gate
C'ye taşınma ihtimali. **Zayıf kategoriler (altitude/mechanical/global_position)
havuzu %100 tükenmiş** — onlar için yeni veri YOK; kaldıraç yalnız Normal + ExtPos.

> Bu, reddedilen "normali homojenleştir" kısayolunun ZITI: modele daha çok gerçek
> çeşitlilik veriyoruz, çeşitliliği gizlemiyoruz.

### Kaldıraç 2 — Kayma-farkındalıklı FA kalibrasyonu (duvara doğrudan saldırı)
Her Gate C başarısızlığı bir FA problemi. Mevcut eşik val-normal q99/bütçe-fit ile
seçiliyor ve test'te ~2× şişiyor. Kayma **sistematik** olduğuna göre ölçülüp telafi
edilebilir: val-içi kayma tahmini + güvenlik marjı (conformal-benzeri). Bu normali
homojenleştirmek değil; çalışma noktasının dürüst kalibrasyonu.

### Kaldıraç 3 — Kendi-kendine koşullanan ince dedektörler (küresel-normal problemini es geçer)
ML-10 forecast-residual, tüm projede baseline'ı Gate B'de **kesin geçen tek fikir**di
(mechanical 0.205→0.390) ve bunu her uçuşu **kendi geçmişine** koşullayarak yapıyor —
yani küresel normal manifolduna hiç ihtiyaç duymadan. ML-12 gösterdi ki tek-feature ince
modüller daha da iyi (0.459). Bu ikisinin birleşimi (forecast-residual + ince modül) daha
tam sömürülmedi.

### Anti-kaldıraçlar (YAPMAYACAKLARIMIZ — gerekçeli)
- **Sentetik normal veri / normal etiket düzenleme:** heterojen-normal problemini kılık
  değiştirerek geri getirir; gerçek-fizik özelliğini bozar (skorları anlamsızlaştırır).
- **Oturum-koşullu model:** reddedildi (yeni oturumda referans kaybı, [[feedback-anomaly-detection-principles]]).
- **Supervised model:** ML-8A kanıtladı — az etiketli veride kaybediyor.
- **Development'ta sonsuz tuning:** disipline aykırı; her tur ön-kayıtlı Gate ister.

### Kapsam kararı — ALFA ve UAV Attack BİTTİ
ALFA (47 resmi uçuş, tam külliyat) ve UAV Attack (19 uçuş) veri-tavanlı, kalıcı olarak
sınırlı. Karakterize edildiler; daha fazla iyileştirme mümkün değil. **Tüm çaba SEAD'e**
odaklanır. Bu, parametre kalabalığının yarısını tek hamlede eler.

---

## 2. Sıralı yol haritası

```
ML-14  Veri yenileme (Normal 398→900, ExtPos 60→193) + yeniden inşa + ölçüm
         │  Gate D1 (veri kalitesi) + D2 (FA kayması düştü mü?) + D3 (Gate C geçti mi?)
         ▼
ML-15  Kayma-farkındalıklı FA kalibrasyonu  ── (D2/D3 sonucuna göre koşullu)
         ▼
ML-16  Kendi-kendine koşullanan ince dedektörler (forecast-residual × ince modül füzyonu)
         ▼
ML-17  ENDGAME: blind holdout'u BİR KEZ aç + model kartı + kapsam beyanı
```

Her ok, bir öncekinin sonucuna bağlı. Duvar veriyle yıkılırsa ML-15/16 hafifler; yıkılmazsa
ML-15 (kalibrasyon) merkeze gelir. **ML-17'de holdout yalnızca bir kez ve yalnız kullanıcı
onayıyla açılır** — bu geri alınamaz tek atıştır.

---

## 3. ML-14 ön-kaydı (sonuç görülmeden sabit)

**Aksiyon:** `uav_sead_downloader.py --normal 900 --ext-pos 193` (skip-existing; yalnız
~635 yeni uçuş çekilir, mevcutlara dokunulmaz). Altitude/Mechanical/Global mevcut kalır
(tükenmiş). Silver→Gold→features→split_manifest→scaler→CUSUM baseline **YENİ sürümlü
artifact seti** olarak üretilir; donmuş ML-9/10/12/13 dizinlerine DOKUNULMAZ (onlar
belgelenmiş tarih olarak kalır).

**Holdout koruması (kritik):** blind holdout aynı `holdout_seed` ve oturum-bazlı mantıkla
yeniden türetilir. Holdout anomali-oturum tabanlı olduğu için (anomali oturumlarının %30'u +
kardeş normalleri), yeni eklenen **normal-only oturumlar development'a** gider; holdout'un
anomali çekirdeği değişmez. Yeni normaller hiçbir holdout uçuşuna sızmamalı — assert edilir.

**Gate D1 — Veri kalitesi (zorunlu):** yeniden inşa hatasız; part-çoğalması yok (uçuş başına
tek üretim); development normal oturum sayısı **64'ten artmış** olmalı; holdout izolasyonu
hash ile korunmuş. Geçmezse dur, düzelt.

**Gate D2 — Kök nedene etki (ASIL SORU):** `existing_fusion` için val→test FA-kayma oranı
(test FA / val bütçe) eski (64-oturum) vs yeni normalde 5 seed medyanı ölçülür.
**Ön-kayıt: zenginleştirme "işe yaradı" sayılır ⇔ medyan kayma oranı göreli ≥%15 düşerse.**
Düşmezse: veri kök nedeni çözmedi — dürüstçe kaydedilir, ağırlık ML-15 kalibrasyonuna kayar.

**Gate D3 — Operasyonel ödeme (payoff):** yeni normal üzerinde ML-9 fusion + ML-12
`itki_komutu` yeniden ölçülür; **herhangi bir Gate C satırı artık geçiyor mu?**
(critical ≥0.30 @ ≤2 veya advisory ≥0.50 @ ≤12). Bu, tüm zincirin nihai testidir.
Ek olarak Position.X/Y (External Position besler) recall'u eski vs yeni raporlanır.

**Disiplin:** yeni feature tablosu ML-9/10/12/13'ün checksum'lı girdilerini geçersiz kılar
— bu bilinçli bir "temiz temel" sıfırlamasıdır; eski sonuçlar (H22-H31) belge olarak durur,
yeni veri yeni zemin olur. Bulgular H32+, karar ADR-014, kayıt SEAD-yenileme bölümü.

---

## 4. ML-15…17 taslağı (ML-14 sonucuna göre detaylanacak)

- **ML-15 — Kayma-farkındalıklı FA kalibrasyonu:** eşik seçimine val-içi kayma tahmini +
  güvenlik marjı ekle; hedef, val bütçesi 12 iken test FA'sını ≤12'de tutmak. Gate: enriched
  normal üzerinde herhangi bir kategori-aday skoru (itki_komutu/chronos_motor) operasyonel
  bütçeyi karşılıyor mu. Ön-kayıt ML-15 planında.
- **ML-16 — Kendi-kendine koşullanan ince dedektörler:** ML-10 forecast-residual'ı 2 kanaldan
  fazlasına genelle + ML-12 ince modül kalıbıyla birleştir; küresel-normal manifolduna en az
  bağımlı skor ailesi. Ön-kayıtlı Gate B/C.
  - *Aday not (2026-07-07, doğrulandı):* Google **TabFM 1.0.0** (30 Haz 2026,
    `google/tabfm-1.0.0-pytorch`, Apache-2.0, sklearn API) bu ortamda temiz kuruluyor
    (`pip --dry-run` geçti; MOMENT'in aksine). Meşru kullanım şekli SADECE
    **cross-feature residual** olur: TabFMRegressor, in-context = yalnız train-normal
    satırları ile bir kanalı (ör. actuator_output_std) diğerlerinden tahmin eder,
    residual skor olur — normal-only paradigmaya uygun, anomaly etiketi istemez.
    Supervised sınıflandırıcı olarak KULLANILMAZ (ML-8A dersi). Zorunlu ön-koşul:
    ML-10'daki gibi CPU preflight + zaman projeksiyonu (<3 saat kuralı) — ICL context
    maliyeti bilinmiyor, README'de CPU benchmark'ı yok. TimesFM (Google'ın zaman-serisi
    FM'i) ise Chronos'un doğrudan alternatifi; ancak forecast-residual hattı production
    hattı seçilirse kıyaslanır.
- **ML-17 — Endgame:** development'ta Gate C geçen EN İYİ tek konfigürasyon dondurulur; blind
  holdout **bir kez** açılır (kullanıcı onayı şart); gerçek "canlı" sayı + model kartı +
  kapsam beyanı yazılır. Geçen konfigürasyon yoksa: ulaşılabilir çalışma noktalarının dürüst
  haritası + "bu veriyle operasyonel hedef karşılanamıyor" beyanı — bu da geçerli, dürüst bir
  bilimsel çıktıdır.

---

## 5. Neden bu sıra (uzman gerekçesi, tek paragraf)

Kategori recall'u zaten çözülmüş (0.459); problem FA ve onun kök nedeni dar/heterojen
normal küme. Elimizde bu kök nedene doğrudan vuran, hiç denenmemiş ve neredeyse otomatik
bir kaldıraç var (73 kullanılmamış oturum). Onu önce çekmek hem en yüksek beklenen değer,
hem de aşağı akıştaki her şeyi (kalibrasyon, füzyon) doğru zeminde inşa etmenin tek yolu —
yoksa dar normal üzerine kurulan her kalibrasyon yeniden yapılmak zorunda kalır. Veri
duvarı yıkarsa iş biter; yıkmazsa en azından "veri değil yöntem" olduğunu KANITLAMIŞ oluruz
ve kalibrasyona geçeriz. Her iki sonuç da ilerleme.
```
