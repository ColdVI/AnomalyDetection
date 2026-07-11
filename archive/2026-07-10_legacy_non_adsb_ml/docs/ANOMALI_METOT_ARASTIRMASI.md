# Anomali Tespit Metotları Araştırması (2026-07-02)

Amaç: `docs/ML1_BULGULAR_VE_HATALAR.md`'deki hata modlarına (H1–H6) literatürden metot eşlemek.
Her aday metot "hangi hatamızı çözer, maliyeti ne, ne zaman deneriz" ile birlikte verilir.

## Hata modu → metot eşleme tablosu

| Hata modu | Aday metot | Faz |
|---|---|---|
| H1 feature sulanması / satır-bazlı zayıflık | LSTM-AE (pencere bazlı), USAD, TranAD | ML-2 |
| H1 impute'un anomaliyi gizlemesi | Maske-farkındalıklı modeller (M2SC2-AD, SFAFormer, GST-Pro) | ML-2+ (araştırma) |
| H2 eşik kararsızlığı (az val uçuşu) | **EVT/POT — SPOT/DSPOT** (akış içi adaptif eşik) | ML-2 |
| H3 DoS zamanlama imzası | Inter-arrival feature'ları + matrix profile (discord) | ML-3 |
| H5 platform transferi | Araç-başına normal profil kalibrasyonu + DSPOT drift | ML-3 |
| H6 kısa senaryolar / zamansal bağlam | LSTM-AE, kısa pencere varyantı; forecasting-residual (LSTM-pred) | ML-2 |

## 1. Reconstruction tabanlı derin modeller (ML-2'nin ana ekseni)

**LSTM-Autoencoder** — planımızın çekirdeği. Pencereyi latent'e sıkıştırıp geri kurar; skor = reconstruction MSE.
Normal-only eğitimle novelty detection'a doğal uyar. UAV literatüründe hâlâ en yaygın temel:
LSTM tabanlı tahmin/reconstruction modelleri UAV arıza ve GPS spoofing tespitinde ~%99 raporluyor
(tek-datasete-özgü rakamlar; bizim leave-flight-out protokolümüz daha zorlu ve dürüst).
LSTM-AE + LOF hibritleri de pratikte kullanılıyor.

**USAD** (autoencoder + iki decoder, adversarial eğitim) — tek AE'ye göre daha keskin sınır, eğitim
hâlâ hafif. LSTM-AE zayıf kalırsa ilk yükseltme adayı.

**TranAD** (transformer, adversarial + meta-learning) — F1'i baseline'lara göre %17'ye kadar artırıp
eğitim süresini %99 düşürdüğünü raporluyor; MAML ile **az veriyle** eğitilebilmesi bizim 6-10 normal
uçuşluk rejimimize uygun. ML-2'de LSTM-AE'yi geçemezsek ML-3 adayı.

**Uyarı (literatür dersi):** Model seçimi çalışmaları, **Matrix Profile'ın** unsupervised/semi-supervised
her kurulumda tutarlı yüksek performans veren tek yöntem olduğunu; basit yöntemlerin derin modelleri
sık sık geçtiğini gösteriyor. Bizim ML-1 bulgumuz (alt_error+CUSUM = 0.878, monolitik IF = 0.5) bu
literatür deseniyle birebir uyumlu. **Derin model, residual feature'ların yerine değil üstüne kurulmalı.**

## 2. Eşik seçimi: EVT / POT ailesi (H2'nin çözümü)

Q99-persentil eşiğimiz az val verisiyle kararsız (H2). **Peaks-Over-Threshold**: yüksek bir başlangıç
eşiğini aşan değerlerin kuyruğuna Generalized Pareto Distribution oturtulur
(Pickands–Balkema–de Haan teoremi):

P(X - u > x | X > u) ≈ (1 + ξx/σ̃)^(-1/ξ)

Böylece "val'de gördüğümüz en kötü skor" yerine **kuyruk modelinden hesaplanan olasılıksal eşik**
kullanılır — az örnekle çok daha kararlı.

- **SPOT**: akışta dağılım bilgisi olmadan adaptif eşik (ilk ~n örnekle init, sonra güncelleme).
- **DSPOT**: drift'li akışlar için (hareketli ortalama üstünden göreli değerler) — platform/rejim
  kayması olan bizim veriye uygun; H5 (SEAD transferi) için de aday.
- **biSPOT/biDSPOT**: çift yönlü (alt kuyruk anomalileri de).

Uygulama: `pot`/`spot` implementasyonları mevcut (ör. cbhua/peak-over-threshold, KDD'17 Siffer et al.).
ML-2'de module eşiklerini Q99 yerine POT-GPD ile hesaplayıp karşılaştıracağız.

## 3. Eksik-veri-farkındalıklı modeller (H1-impute'un kalıcı çözümü)

Impute yerine eksikliği modele bilgi olarak veren yaklaşımlar:
- **M2SC2-AD**: çok-ölçekli missing-mask embedding.
- **SFAFormer**: örnekleme aralığını embedding'e taşıyor — interpolasyon/impute istemiyor
  (bizim ping_dos zamanlama imzası H3 ile doğrudan ilişkili).
- **GST-Pro**: eksik değerli düzensiz örneklemede graph spatiotemporal süreç.
- **Latent SDE'ler**: düzensiz örnekleme + eksikliği üretken modelle ele alıyor; ağır ama ilkesel çözüm.

Pratik ara adım (ML-2'de yapıyoruz): AE girişine **maske kanalı** eklemek (feature + is_missing bayrağı),
imputed değerlerin reconstruction hatasını maskeyle ağırlıklandırmak — "eksik değeri iyi tahmin etti"
diye ödül vermemek.

## 4. Forecasting-residual ailesi (alternatif skor üretimi)

Reconstruction yerine ileri tahmin: LSTM t+1'i tahmin eder, skor = |gerçek − tahmin|.
NAB benchmark çalışmaları forecasting tabanlı çerçevelerin nokta ve bağlamsal anomalilerde
reconstruction'a rakip olduğunu gösteriyor. Bizim `parse` katmanındaki komut→tepki fiziğiyle
(analytical redundancy) aynı matematik: ŷ_t = f(geçmiş), r_t = y_t − ŷ_t. ALFA'nın kısa
senaryolarında (H6) pencere doldurma sorunu yaşamaz — ML-2'de LSTM-AE ile birlikte küçük bir
LSTM-forecaster da koşturulabilir.

## 5. Matrix Profile / discord tabanlı (H3 + sağlam baseline)

Pencere-benzerlik uzaklığına dayalı, parametresiz, **eğitimsiz**; tek boyutlu seride "en benzersiz
alt dizi"yi (discord) bulur. Trilyon-nokta ölçeğine taşınmış versiyonları var; çok boyut için
k-boyut seçimli varyantlar mevcut. Bizim için kullanım: inter-arrival serisi (H3, DoS) ve tek-kanal
residual'larda (alt_error) discord araması — CUSUM'a paralel ikinci eğitimsiz dedektör.

## 6. Karar: ML-2'de ne yapıyoruz (öncelik sırası)

1. **LSTM-AE** (plan gereği; maske kanallı giriş + maske-ağırlıklı MSE ile H1-impute dersini uygulayarak).
2. **Dense AE** debug basamağı (pipeline doğrulama — plan gereği).
3. **POT/GPD eşik** — Q99 ile yan yana raporlanır (H2).
4. LSTM-AE yetersiz kalırsa: USAD → TranAD (az-veri MAML avantajı).
5. ML-3: matrix profile (inter-arrival/discord), missing-aware mimariler, SEAD τ-kalibrasyonu + DSPOT.

## Kaynaklar

- [Deep Learning for Time Series Anomaly Detection: A Survey](https://arxiv.org/pdf/2211.05244)
- [Unified Taxonomy for Multivariate TS-AD using Deep Learning (2026)](https://arxiv.org/pdf/2603.18941)
- [Anomaly Detection in Streams with Extreme Value Theory (SPOT/DSPOT, KDD'17)](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p1067.pdf)
- [POT/SPOT/DSPOT implementasyonu](https://github.com/cbhua/peak-over-threshold)
- [EVT ile eşikleme pratik rehber (TotalEnergies)](https://medium.com/totalenergies-digital-factory/your-journey-with-time-series-thresholding-using-extreme-value-theory-c8799049511f)
- [TranAD (VLDB'22)](http://vldb.org/pvldb/vol15/p1201-tuli.pdf) · [arXiv](https://arxiv.org/abs/2201.07284)
- [MSAD: Model Selection for TS-AD — Matrix Profile tutarlılığı](https://arxiv.org/html/2510.26643v1)
- [Matrix Profile for Multidimensional TS-AD](https://arxiv.org/abs/2409.09298)
- [Prediction-based trajectory anomaly detection under GPS spoofing (2025)](https://www.sciencedirect.com/science/article/pii/S1000936125000846)
- [DeepSpoofNet (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11935755/)
- [GNSS Spoofing/Jamming ML/DL karşılaştırması](https://arxiv.org/pdf/2501.02352)
- [UAV siber güvenlikte DL trendleri — sistematik derleme (2026)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1752124/full)
- [SFAFormer — düzensiz örnekleme farkındalıklı transformer](https://www.sciencedirect.com/science/article/abs/pii/S0020025526000253)
- [M2SC2-AD — missing-aware çok-ölçekli çerçeve](https://www.sciencedirect.com/science/article/abs/pii/S0306457326003390)
- [GST-Pro — eksik değerli graph spatiotemporal AD](https://arxiv.org/html/2401.05800v1)
- [Latent SDE'lerle seyrek/düzensiz TS-AD](https://arxiv.org/html/2606.18898v1)
- [Forecasting-tabanlı çerçeve — NAB benchmark](https://arxiv.org/pdf/2510.11141)
- [Reconstruction yöntemlerinin değerlendirmesi (AI Review 2025)](https://link.springer.com/article/10.1007/s10462-025-11401-9)
