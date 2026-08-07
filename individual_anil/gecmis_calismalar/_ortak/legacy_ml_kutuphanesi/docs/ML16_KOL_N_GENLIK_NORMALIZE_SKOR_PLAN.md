# ML-16 Kol N: Genlik-Normalize Reconstruction Skoru (post-hoc, yeniden eğitim yok)

Durum: ÖN-KAYIT (2026-07-10; plan sonuç görülmeden yazılıp sabitlendi, sonra koşulacak).
Üst bağlam: `docs/decisions.md` ADR-016/017/018,
`docs/ML_YETERSIZLIKLER_KAYDI.md` madde B.5. **ÖNEMLİ isim ayrımı:** "ML-17" bu depoda
zaten farklı, spesifik bir anlama sahip — blind holdout'un kullanıcı onayıyla bir kez
açıldığı endgame fazı. Bu faz o değildir; holdout burada da AÇILMAZ. Bu doküman "Kol N"
adını taşır (Kol L/D/U'dan sonraki dördüncü, bağımsız kol — mimari değil, SKORLAMA
değişikliği).

## §0 Gerekçe ve kapsam

ADR-016/017/018 üç bağımsız mimariyi (LSTM-AE, Dense-AE, USAD) SEAD'de güncel splitlerde
eğitti; üçü de Gate B'de kaldı VE üçünün de ham recall kazancı aynı artefakttan geliyor:
`masked_mse`, `AE_FEATURES["uav_sead"]`'in 22 kanalı üzerinde HAM kare hatayı toplayıp
tek bir skaler skora indiriyor; proje-çapında kullanılan `RobustScaler` aykırı değerleri
KIRPMIYOR; bu yüzden GPS-sağlığı grubundaki birkaç seyrek kanal (`eph`,
`gps_frozen_count` vb.) bazı pencerelerde aşırı ölçekli değerler alıyor (gerçek
GPS-sahtekârlığı sıçramaları + en az bir muhtemelen yanlış-etiketli "normal" uçuş,
`2018-05-26/16_50_55`, `eph`≈25000) ve bu birkaç pencere, mimariden bağımsız olarak,
TÜM üç modelin reconstruction hatasına hakim oluyor
(`artifacts/ml_lstm_sead/uav_sead/full_matrix/magnitude_domination_diagnostic.json`:
eğitilmiş-skor vs eğitilmemiş-rastgele-başlatılmış ağ Spearman ρ=0.964, eğitilmiş-skor vs
model içermeyen ham `‖x‖²` taban-çizgisi ρ=0.965).

Bu faz, YENİDEN EĞİTİM yapmadan (maliyetli bir kırpmalı-ölçekleme + yeniden-eğitim
turundan ÖNCE, ucuz ve hızlı bir ön-deney olarak), AYNI ÜÇ ZATEN-EĞİTİLMİŞ checkpoint'i
(`artifacts/ml_lstm_sead|ml_dense_ae_sead|ml_usad_sead/uav_sead/full_matrix/split_NN/models/*.pt`)
kullanarak, ham reconstruction rezidüelini nihai skora çevirme YÖNTEMİNİ değiştirmenin
genlik-baskınlığını azaltıp azaltmadığını test eder. Modellerin ağırlıkları hiç
değişmez — yalnızca "rezidüelden skora" fonksiyonu değişir.

**Başarı kriteri açıkça iki ayrı eksen:** (i) recall/FA hedefi (critical ≥0.30 recall @
≤2 FA/saat, advisory ≥0.50 recall @ ≤12 FA/saat — donmuş, değişmez) ve (ii) — kullanıcının
özellikle talep ettiği, bu turda BİLGİLENDİRİCİ değil ZORUNLU raporlama kriteri —
genlik-baskınlığı korelasyonlarının (trained-vs-untrained-random-init Spearman,
trained-vs-ham-‖x‖²-taban-çizgisi Spearman) ~0.96 baseline'ın GERÇEKTEN altına inip
inmediği. Recall yükselip korelasyon ~0.96'da kalırsa bu "düzeltme" değil, "genlik
baskınlığının başka bir kılığı"dır ve öyle raporlanacak — **sonuç ne olursa olsun
(recall/FA hedefi geçse de geçmese de, korelasyon düşse de düşmese de) dürüstçe
raporlanacak; sonuç görüldükten sonra formül/epsilon/karar-tipi/bütçe DEĞİŞTİRİLMEYECEK.**

## §1 Değerlendirilecek skor varyantları (SABİT — iki ve yalnızca iki, üç mimarinin her
biri için — toplam 6 hücre)

Her iki varyant da, `masked_mse`'nin ürettiği TEK toplam kare-hata skaler yerine, kanal
bazlı bir ara temsil kullanır ve kanalları TOPLAMADAN (kare-toplayarak değil) ORTALAR —
böylece tek bir aşırı-genlikli kanal artık tüm pencerenin skorunu tek başına domine
edemez (kare-toplamın aksine, ortalama/percentile-ortalaması sınırlı katkı yapar).

### (a) Bağıl hata (`{arch}_relerr`)

Her kanal `c`, her zaman-adımı `t`, her pencere `i` için (yalnız maskede geçerli
`t`'lerde): `r[i,t,c] = |x[i,t,c] - x̂[i,t,c]| / (|x[i,t,c]| + ε)`. Önce zaman
ekseninde (pencere içindeki geçerli adımlar üzerinden) kanal-başı ortalama alınır,
sonra o pencerede MEVCUT (maskede en az bir geçerli adımı olan) kanallar üzerinden
ORTALAMA alınır — kanal sayısı pencereden pencereye değişebileceği için payda da
değişir (sessiz doldurma YOK; hiç geçerli kanalı olmayan bir pencere NaN kalır, ADR
disipliniyle tutarlı).

**ε = 0.1 (ölçeklenmiş birimlerde), SABİT, sonuç görülmeden seçildi.** Gerekçe:
`src/ml/data/scaling.py::fit_scaler_params` klasik `RobustScaler` kullanıyor (medyan
merkez, IQR ölçek) — bu yüzden ölçeklenmiş uzayda tipik olarak Q1≈-0.5, Q3≈+0.5 (IQR=1).
ε=0.1, bir IQR'nin ~%10'u: `x≈0` (medyan civarı) civarında paydanın sıfıra
yaklaşmasını/bölme patlamasını engellemeye yetecek kadar büyük, ama tipik orta-menzil
değerler için gerçek bağıl sapmayı boğacak kadar büyük değil. USAD'ın iki-kod-çözücülü
skoru için AYNI α=β=0.5 ağırlıklandırması (`usad_reconstruction_scores`'un paylaştığı
gibi) bağıl-hata formülüne de uygulanır: `relerr = 0.5·relerr(x, AE1(x)) +
0.5·relerr(x, AE2(AE1(x)))` — bu, "hangi rezidüel skora giriyorsa aynı ağırlıklandırma"
ilkesini korur, sonuçtan etkilenerek seçilmedi.

### (b) Kanal-bazlı yüzdelik-normalize hata (`{arch}_rankpct`)

Her kanal `c` için, o pencerenin kare-hatası (`masked_mse_per_channel`'dan — pencere
içi zaman ortalaması, yalnız geçerli adımlar) SPLİTİN KENDİ train-normal pencerelerindeki
AYNI kanalın kare-hata dağılımına karşı bir yüzdelik değerine (percentile, [0,1])
çevrilir — `src/ml/evaluation/score_fusion.py::empirical_probability` DEĞİŞTİRİLMEDEN,
kanal-bazlı çağrılarak (referans = o kanalın train-normal pencerelerindeki kare-hata
dizisi, yalnız o kanalın geçerli olduğu pencerelerden). Sonra bu percentile'lar,
pencerede MEVCUT olan (hem o pencerede geçerli hem train'de referansı olan) kanallar
üzerinden ORTALAMA alınır. Amaç: "genellikle gürültülü olan bir kanal" (örn. ham
büyüklüğü zaten yüksek olan bir GPS alanı) salt ham birimi büyük diye pencereyi domine
edemesin — kendi geçmiş dağılımına göre "bu kanal için ne kadar olağan dışı" sorusuna
dönüştürülüyor.

**Referans kümesi = split'in train-normal pencereleri (val değil).** Bilinçli seçim:
train havuzu val'den büyük (daha kararlı percentile tahmini) ve zaten modelin kendisi bu
pencerelerle eğitildi (dolayısıyla "normal bu modele göre nasıl görünür" referansı doğal
olarak buradan geliyor). Nihai skorun kalibrasyonu (aşağıda §2) HALA val-normal'e karşı
`empirical_probability` ile yapılıyor — bu iki-katmanlı kalibrasyon (önce kanal-bazlı
train-normal percentile, sonra tüm-skor val-normal percentile) kasıtlı ve iki ayrı amaca
hizmet ediyor: birincisi kanal-genlik dengesizliğini düzeltmek, ikincisi mevcut
`lstm_recon`/`dense_ae_recon`/`usad_score` ile AYNI karar-katmanı girdi sözleşmesini
korumak.

Bir kanalın train'de HİÇ geçerli penceresi yoksa (referans boş), o kanal o splitte
rankpct hesabından TAMAMEN çıkarılır (sessizce 0/nötr değer atanmaz) — bu, projenin
"sessiz doldurma yok" kuralının doğrudan uygulanması.

### Kapsam dışı (bu turda KASITLI OLARAK yapılmayacak)

- `ml14_fusion`/`itki_komutu` ile max-füzyon YOK — yalnız recon-alone, ADR-016/017/018'in
  `lstm_recon`/`dense_ae_recon`/`usad_score` satırlarıyla temiz bir 6-hücre karşılaştırma
  için.
- Yeniden eğitim YOK, hiperparametre taraması YOK, farklı pencere/stride YOK.
- Başka ad hoc formül (örn. log-hata, winsorize, z-score) denenmeyecek — yalnız bu iki
  formül.

## §2 Kalibrasyon/karar hattı — mevcut skorlarla BİREBİR AYNI protokol

Her `{arch}_relerr`/`{arch}_rankpct` ham penceresi:

1. Pencere-sonu (`t_end`) zaman damgasında hesaplanır (mevcut `build_windows` çıktısı,
   değiştirilmeden).
2. `scripts/run_ml8a_temporal_boosting.py::_align_score` (`merge_asof(...,
   direction="backward")`) ile, val∪test satır ızgarasındaki her (source_id, t_rel_s)
   çiftine "en son tamamlanmış pencere" olarak nedensel taşınır — mevcut LSTM/Dense/USAD
   skorlarıyla AYNI hizalama fonksiyonu, yeniden yazılmaz.
3. `src/ml/evaluation/score_fusion.py::empirical_probability` ile split'in val-normal
   referansına kalibre edilir (mevcut skorlarla AYNI fonksiyon/çağrı deseni).
4. `last_causal_per_bucket` ile 1 saniyelik karar kovasına indirgenir (mevcut ile AYNI).
5. `src/ml/decision/decision_layers.py` (Threshold/K-of-N/CUSUM, `_fit_policies` —
   DEĞİŞTİRİLMEDEN import) val-normal streams üzerinde iki bütçeye (critical ≤2 FA/saat,
   advisory ≤12 FA/saat) kalibre edilir.
6. `scripts/run_ml9_category_evaluation.py::_evaluate` ile test uçuşlarında recall/FA/
   kategori metrikleri hesaplanır — DEĞİŞTİRİLMEDEN import.

Hiçbir adım yeniden yazılmıyor; yalnız 1-3 arası ham-skor ÜRETİMİ yeni (kanal-bazlı
formüller), 4-6 mevcut kod. Bu, ADR-016/017/018'deki `lstm_recon` vb. ile birebir aynı
"tek skor kaynağı" muamelesidir — sadece ham skor formülü farklı.

**Skor kaynağı adları (SABİT, altı ve yalnızca altı, füzyon YOK):** `lstm_relerr`,
`lstm_rankpct`, `dense_ae_relerr`, `dense_ae_rankpct`, `usad_relerr`, `usad_rankpct`.

## §3 Zorunlu tanı: genlik-baskınlığı korelasyonları (BİLGİLENDİRİCİ DEĞİL, raporun
zorunlu parçası)

Her 6 hücre (varyant × mimari) için, `scripts/diagnose_ml_lstm_sead_magnitude_domination.py`
ile AYNI mantık, AMA ham `masked_mse` yerine bu fazın YENİ formülüyle tekrarlanır —
adil (elma-elma) karşılaştırma için üç tahminci de AYNI yeni formülle puanlanır:

1. **Eğitilmiş model** — zaten var olan, donmuş checkpoint.
2. **Eğitilmemiş (rastgele başlatılmış) aynı mimari** — `torch.manual_seed(999)` (orijinal
   teşhisle AYNI konvansiyon), YENİDEN EĞİTİLMEZ, yalnız forward-pass için kullanılır.
3. **Model içermeyen ham `‖x‖²`/`‖x‖` taban-çizgisi** — reconstruction=0 varsayılarak AYNI
   yeni formül uygulanır (`relerr`: `|x-0|/(|x|+ε)`; `rankpct`: `|x-0|²`'nin kendi
   train-normal referansına göre percentile'ı).

`rankpct` için üç tahmincinin HER BİRİ kendi train-normal pencerelerinde kendi
kanal-bazlı referansını kurar (yani "eğitilmemiş ağ" kendi ham hatasının train-normal
dağılımına göre kalibre edilir, eğitilmiş modelin referansı ödünç alınmaz) — bu, "bu
formülle puanlansaydı naif bir dedektör nasıl davranırdı" sorusuna dürüst bir cevap
vermek için kasıtlı bir tasarım kararı.

Split'in TEST pencerelerinde (orijinal teşhisle aynı kapsam) hesaplanır:

- `trained_vs_untrained_random_init_spearman`
- `trained_vs_magnitude_only_spearman`
- (bağlam için, orijinal teşhisle parite: `untrained_vs_magnitude_only_spearman`)

**Rapor kuralı (SABİT):** her hücre için bu iki ana sayı (baseline ~0.964/0.965'e karşı)
düz metin olarak raporlanır. Bir varyant recall'u yükseltip bu korelasyonları ~0.96'da
tutuyorsa, bu "düzeltilmedi, yalnızca farklı bir genlik-baskınlığı biçimi bulundu"
şeklinde AÇIKÇA yazılacak — recall artışı tek başına başarı sayılmayacak.

## §4 Uygulama notları (kod / determinizm)

- **Tek script, üç mimari yerine üç ayrı runner değil:** Kol L/D/U'nun her biri kendi
  runner dosyasını taşıyordu çünkü HER BİRİ ayrı bir eğitim döngüsü içeriyordu (mimariye
  özgü kod, kasıtlı tekrar). Bu faz YENİDEN EĞİTİM içermiyor — üç mimarinin ürettiği
  ortak arayüz (x, x̂, mask) üzerinde çalışan post-hoc bir skorlama katmanı. Bu yüzden
  `scripts/run_ml16_kol_n_magnitude_normalized_scoring.py` TEK dosya, mimariye özgü kısım
  yalnızca "checkpoint nasıl yüklenir / forward nasıl çağrılır" adaptörüne indirgenmiş
  küçük bir fonksiyon haritası. Bu, kod tekrarını azaltır ve üç mimarinin AYNI
  kod yoluyla puanlandığını garanti eder (Kol L/D/U'daki "her aile kendi dosyasını
  taşır" kuralına kasıtlı, gerekçeli bir istisna — training yok, adalet gerekiyor).
- **`masked_mse_per_channel`** (`src/ml/models/lstm_autoencoder.py`, `masked_mse`'nin
  yanına EKLENİR, DEĞİŞTİRİLMEZ): `masked_mse`'nin kare-hata payı/geçerli-sayı
  paydasının kanal ekseninde AYRIŞTIRILMIŞ hali — `numerator.sum(-1)` /
  `denominator.sum(-1).clamp(min=1.0)` `masked_mse(..., per_sample=...)` ile TAM
  ÖRTÜŞÜR (kanallar üzerinden toplandığında aynı toplamı verir — testle doğrulanır).
  `masked_mse`'nin imzası/davranışı DEĞİŞMEZ.
- **Frozen model kuralı:** checkpoint'ler `torch.load(..., weights_only=True)` ile
  yalnız `state_dict` için yüklenir; `.fit(`, `train_lstm_autoencoder`,
  `train_dense_autoencoder`, `train_usad`, `fit_modular_iforest`, `fit_scaler_params`
  YENİ script'te HİÇ ÇAĞRILMAZ (statik test bunu doğrular — `tests/test_rfly1_severity_
  sweep.py`'deki `"IsolationForest" not in source` deseninin bu faza uyarlanmış hali).
  Scaler de yeniden fit EDİLMEZ — ADR-016/017/018 koşularının split-başı donmuş
  `scaler.json`'ı DOĞRUDAN yeniden kullanılır (`fit_scaler_params` çağrısı yok).
- **Füzyon/existing_fusion/itki_komutu bu turda hiç hesaplanmaz** (§1 kapsam dışı) —
  bu yüzden `fit_modular_iforest`/`IsolationForest` bu script'te YOKTUR (Gate A'nın
  "training yok" ispatını basitleştirir: script'te zaten hiç model eğitim çağrısı yok).
- **Holdout izolasyonu:** mevcut üç script ile AYNI development-set okuma/assert deseni
  (`data & holdout` boşsa assert) — DEĞİŞTİRİLMEDEN taşınır.

## §5 Gate tanımları (SABİT)

- **Gate A (zorunlu, güvenlik):** (i) 200 uçuşluk final holdout hiçbir aşamada okunmadı
  (assert, mevcut desenle aynı); (ii) statik kod testi: yeni script'te `.fit(`,
  `train_lstm_autoencoder`, `train_dense_autoencoder`, `train_usad`,
  `fit_modular_iforest`, `fit_scaler_params`, `IsolationForest` HİÇ geçmiyor; (iii)
  `masked_mse_per_channel` kanal toplamı `masked_mse`'nin ürettiği toplamla (aynı
  girdilerde) sayısal olarak örtüşüyor (testle). Bu turda önceki fazlardaki gibi bir
  "donmuş CSV'yle birebir örtüşme" determinizm testi YOK çünkü bu YENİ bir skor —
  onun yerine yukarıdaki üç madde Gate A'yı oluşturuyor.
- **Gate B (operasyonel hedef, ml9/ml14/Kol L/D/U ile AYNI kural):** altı skor
  kaynağından (`lstm_relerr`, `lstm_rankpct`, `dense_ae_relerr`, `dense_ae_rankpct`,
  `usad_relerr`, `usad_rankpct`) HERHANGİ biri × üç karar tipinden (threshold, k_of_n,
  cusum) HERHANGİ biri × iki bütçeden birinde **critical ≥0.30 recall @ ≤2 FA/saat**
  VEYA **advisory ≥0.50 recall @ ≤12 FA/saat** sağlarsa geçer.

Sonuç görüldükten sonra formül/ε/karar-tipi/bütçe/mimari listesi DEĞİŞTİRİLMEZ. Gate B
geçse bile holdout AÇILMAZ (tek development koşusu bu karara yetmez — proje çapında ayrı
bir karar). **Sonuç ne olursa olsun dürüstçe raporlanacak.**

## §6 Dosyalar

| Dosya | İş |
|---|---|
| `src/ml/models/lstm_autoencoder.py` | `masked_mse_per_channel` EKLENİR (mevcut kod değişmez) |
| `src/ml/evaluation/magnitude_normalized_scoring.py` (yeni) | `relerr`/`rankpct` formülleri, mimari-adaptörleri, korelasyon tanısı |
| `scripts/run_ml16_kol_n_magnitude_normalized_scoring.py` (yeni) | §2/§3/§4/§5 protokolü; `--splits split_00` smoke desteği |
| `artifacts/ml16_kol_n/{lstm,dense_ae,usad}/` | `metrics.csv`, `flight_label_metrics.csv`, `category_metrics.csv`, `correlation_diagnostics.json`, `gates.json`, checksum'lı `manifest.json` |
| `tests/test_ml16_kol_n_magnitude_normalized_scoring.py` (yeni) | frozen-model statik testi, formül-doğrulama (elle hesaplanmış toy örnek), maskeli/NaN kanal davranışı, `masked_mse_per_channel` toplam-örtüşme testi |
| `docs/decisions.md` | ADR-019 eklenir |
| `docs/ML_YETERSIZLIKLER_KAYDI.md` B.5 | sonuçla güncellenir (kapandı/kısmen yardımcı oldu/hiç yardımcı olmadı — dürüstçe) |

Kabul: `--splits split_00` smoke → tam 5-split × 3-mimari koşu; tam `pytest -q` yeşil
(bilinen 4 MinIO hariç).
