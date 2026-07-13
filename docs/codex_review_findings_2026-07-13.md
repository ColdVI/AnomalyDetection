# ADS-B Anomali Tespiti Bağımsız İnceleme Bulguları

**Tarih:** 2026-07-13  
**Kapsam:** `docs/codex_review_prompt_2026-07-13.md` içindeki sekiz soru, mevcut kod,
testler, rapor JSON'ları, grafikler ve Silver/sentetik veri üzerinde salt-okunur doğrulamalar.  
**Bu turda yapılan değişiklik:** Yalnız bu rapor oluşturuldu; kod, veri ve mevcut artefaktlar
değiştirilmedi. Yeni `v2025.06.15` tar içeriği açılmadı, hashlenmedi veya parse edilmedi.

## Yönetici kararı

İlk iş yeni bir model veya eşik denemek değil, **interval truth ve değerlendirme sözleşmesini
düzeltmektir**. Mevcut sentetik dosya-kimliği etiketi, gerçekten temiz olan onset-öncesi
pencereleri pozitif sayıyor; ayrıca `altitude_dropout` reçetesinin gerçek bozulma bloğu ile
etiketli aralığı farklı. Bu iki sorun düzelmeden elde edilecek yeni AUC, loss ağırlığı veya
CUSUM karşılaştırması güvenilir olmaz.

Truth düzeltmesinden sonra ana teknik hat şu olmalı:

1. şeffaf residual kural skorlayıcısını değişmeden referans olarak korumak;
2. 2 m/s konum rampı için yönlü hız residual'ı üzerinde nedensel, uçuş-içi CUSUM eklemek;
3. doğal alarm yükünü üç günün tamamında bellek-güvenli, akışkan biçimde ölçmek;
4. S2'yi residual penalty'ye karıştırmadan `declared_status`, `position_quality` ve
   `altitude_availability` durum kanalları olarak kurmak;
5. NN'leri şimdilik dondurmak; yalnız ana hat istikrar kazandıktan sonra tek, ön-kayıtlı
   Dense-AE ağırlıklı-loss falsifikasyon deneyi yapmak; USAD'ı aktif plandan çıkarmak.

Mevcut kural skoru, hatalı whole-file etiketle pooled AUC `0.599760` üretmiş ve üç NN'in
`0.552371–0.572209` aralığını geçmiş olsa da bu değerlerin hiçbiri production/gate sonucu
değildir (`artifacts/adsb/models/rule_scorer_report.json:2169`,
`artifacts/adsb/models/baseline_training_report.json:5526769,11087236,16881661`).

## Kanıt ve sayı disiplini

Bu raporda “**inceleme ölçümü**” denilen değerler, 2026-07-13 tarihinde mevcut dosyalardan
salt-okunur olarak hesaplandı; kalıcı ara çıktı yazılmadı. “**Tahmin**” veya “**örnek hesap**”
etiketli değerler ölçüm değildir. Dış kaynaklar yalnız tanım ve yöntem dayanağıdır.

`artifacts/adsb/plots/` görsel denetiminde loss/ROC/AUC heatmap/confusion-score grafikleri
JSON özetleriyle çelişmedi. `injection_timelines/` onset öncesinde clean/corrupt örtüşmesini,
ground-speed ve track senaryolarında onset sonrası ayrışmayı, position rampın zayıf ayrışmasını
ve dropout'ta penalty ayrışmamasını görsel olarak doğruladı. Bu gözlem yeni sayısal metrik
olarak kullanılmadı.

### Çapraz-kesit bulgu: provenance ve run değişmezliği eksik

**Bulgu.** Belgelerdeki Silver toplamı ile dosyaların gerçek footer toplamı uyuşmuyor; mevcut
rapor ve sentetik yazma yolları da aynı adı sessizce yeniden kullanabiliyor. Bu, model
versiyonlamasından önce giderilmesi gereken veri/run provenance borcudur.

**Kanıt.** İnceleme ölçümünde 638 Parquet footer'ı ve üç parse logu şu toplamı verdi:

| Gün | Parça | Satır | Log kanıtı |
|---|---:|---:|---|
| 2026-02-28 | 237 | 88,762,032 | `adsb_parse_02_28.log:475-476` |
| 2026-03-01 | 216 | 85,991,023 | `adsb_parse_03_01.log:433-434` |
| 2026-03-16 | 185 | 81,401,954 | `adsb_parse_03_16.log:371-372` |
| **Toplam** | **638** | **256,155,009** | footer toplamı |

`adsb/README.md:21` ve `docs/decisions.md:844` ise `256,150,550` yazıyor; fark **4,459
satırdır**. Bu fark bu incelemede açıklanamadı ve sessizce “düzeltilmemelidir”. Baseline scripti
sabit `baseline_training_report.json` yoluna yazar
(`scripts/adsb_train_baseline_models.py:256-259`); dosyanın inceleme anındaki boyutu
**528,285,357 byte** idi. Bunun ana nedeni her ROC noktasının JSON'a yazılmasıdır
(`scripts/adsb_train_baseline_models.py:161-170`); rule raporu ROC'u örnekleyerek saklar
(`scripts/adsb_evaluate_rule_scorer.py:111-120`). Sentetik path guard gerçek veriye yazmayı
engelliyor, fakat aynı `name.parquet` varsa üstüne yazmayı engellemiyor
(`adsb/synthetic.py:124-135`). Ayrıca `adsb/rules.py:13-14` hâlâ “MAD=0 floor” derken çalışan
kod kanalı dışlıyor (`adsb/rules.py:56-76`); uygulanacak kural kodda doğrudur, docstring eskidir.

**Öneri.** Her koşu için fail-if-exists çalışan değişmez bir `run_id` dizini oluşturulsun.
Manifest en az girdi yolu/byte/SHA-256/footer satırı/şema hash'i, split ve dışlanan flight
kimlikleri, git commit'i ve dirty-state, config, seed, feature sırası, scaler/kalibrasyon,
sentetik manifest hash'i, metric sözleşmesi ve çıktı checksum'larını içersin. Tam ROC gerekirse
sıkıştırılmış sütunsal dosyaya ayrı yazılsın; JSON yalnız özet ve seyreltilmiş eğri taşısın.
Sentetik v1 korunmalı; v2 yeni namespace'e ve `exist_ok=False` eşdeğeri bir guard ile yazılmalıdır.

**Tahmini efor.** 0.5–1 iş günü; ölçülmüş süre değildir.

**Risk.** Bu yapılmazsa aynı dosya adı altında farklı veri, truth veya eşiklerin sonuçları
karışır; post-hoc değişiklik ve sentetik sızıntısı sonradan denetlenemez.

## 1. Pencere-etiket düzeltmesi

**Bulgu.** Enjeksiyon kodu satır etiketi üretse de pencereleyici etiketi taşımıyor; iki
değerlendirme scripti de bozuk Parquet'teki **bütün** pencereleri pozitif yapıyor. Ayrıca
dropout ve ramp için `label != null` her zaman fiziksel olarak aktif bozulma demek değildir.

**Kanıt.** `_mark`, onset'ten uçuş sonuna kadar etiketi dolduruyor
(`adsb/synthetic.py:28-35`). `build_windows` yalnız `flight_id/t_start/t_end` metadata'sı
döndürüyor (`adsb/windowing.py:32-54`). Rule evaluator `r_scores` dizisinin tamamına bir
etiketi veriyor (`scripts/adsb_evaluate_rule_scorer.py:100-110`); NN evaluator aynı şeyi
yapıyor (`scripts/adsb_train_baseline_models.py:150-176`). Dropout yalnız onset sonrasındaki
rastgele alt bloğa NaN yazar, fakat onset'ten sona kadar işaretler
(`adsb/synthetic.py:73-86`). Rampın onset satırında `dt=0`, dolayısıyla fiziksel perturbasyon
da sıfırdır (`adsb/synthetic.py:97-108`).

İnceleme ölçümü: mevcut `ground_speed_biased.parquet`, `WINDOW=12`, `STRIDE=6`,
`MAX_GAP_S=60` ve mevcut pencereleyiciyle salt-okunur tekrar oynatıldı. Toplam **705,787**
pencerenin **344,883**'ü tamamen onset öncesi, **15,446**'sı sınırı kesen ve **345,458**'i
tamamen aktifti. Whole-file proxy etiketi altında fiziksel durumu bilen state-oracle skorun AUC
referansı **0.7556748707** çıktı. Bu değer evrensel/matematiksel bir tavan değil;
`1-0.5×(344883/705787)` ile elde edilen, bütün normal pencerelerin skor bakımından
exchangeable ve aynı normal skora sahip olduğu **state-oracle AUC referansıdır**. Uçuş fazını
veya başka bir confound'u skorlayan bir sistem bu değeri aşabilir ve yine de daha iyi anomaly
detector olmayabilir. Ölçüm kalıcı rapor sayısı değil, mevcut
`data/objectstore/synthetic/adsb/ground_speed_biased.parquet` üzerinde bu incelemenin
tekrar oynatmasıdır. Forecaster yalnız son dört hedef satırını skorladığı için
(`adsb/models/lstm_forecaster.py:32-39,80-91`) aynı pencerelerin destek sınıfları
**344,883 normal / 4,201 karışık / 356,703 tam aktif** oldu; modelden bağımsız tek “whole
window” etiketi bu mimari için de yanlıştır.

Literatürde tek zorunlu pencere etiketi standardı yoktur. Range/event-aware precision-recall
olay varlığına, overlap/cardinality'ye ve isteğe bağlı konum ağırlığına ayrı bileşenler verir
([Tatbul ve diğerleri, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html));
TaPR detection/portion skorları ve olay-sonrası ambiguous aralık tanımlar
([Hwang ve diğerleri, CIKM 2019](https://dl.acm.org/doi/10.1145/3357384.3358118)). Tüm anomalik
aralığı tek isabetten sonra doğru sayan point-adjust, rastgele skorları bile olduğundan iyi
gösterebilir
([Kim ve diğerleri, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20680)).
Benchmark uygulamalarının farklı metrik ailelerini yan yana sunması da tek standart
olmadığını gösterir
([TimeSeAD, TMLR 2023](https://openreview.net/forum?id=iMmsCI0JsS)).

**Öneri.** Sentetik truth v2'de `event_id`, `event_type`, `attack_onset`,
`observable_onset`, `event_end` ve birbirinden ayrı satır-bazı `injection_active`,
`observable_changed` ve `evaluable_truth` alanları olsun. Dropout'ta yalnız gerçek NaN bloğu
`injection_active` olur; clean `alt` zaten NaN ise enjekte komut aktif olsa da gözlenebilir
değişim yoktur. Rampta sıfırdan farklı komutun Parquet/feature çözünürlüğünde gerçekten
değiştirdiği satırlar `observable_changed` olur. Eski v1 korpus ve raporlar korunmalı;
düzeltme yeni versiyon ve yeni run olarak çalışmalıdır.

Her skor için loss/score desteği \(S_w\) açıkça tanımlansın: rule/AE için bütün pencere,
forecaster için yalnız hedef satırları. `S_w` içindeki `evaluable_truth=True` satırlarda

\[
q_w = \frac{\sum_{t\in S_w} 1[\text{observable_changed}_t]}
{\sum_{t\in S_w}1[\text{evaluable_truth}_t]}
\]

hesaplansın; payda sıfırsa pencere unscoreable truth olarak ayrılsın. Ön-kayıtlı **birincil**
sentetik pencere etiketi `y_any = 1[q_w > 0]` olsun. İkincil steady-state değerlendirme yalnız
`q∈{0,1}` altkümesinde `q=1` pozitif/`q=0` negatif çalışsın; `0<q<1` pencereleri bu ikincil
metrikte yanlış negatif yapılmadan ayrıca raporlansın.
Sonuca bakarak yüzde eşiği seçilmesin. Düzensiz örneklemede satır oranı süre oranı değildir;
aktif-süre metrikleri timestamp ile ayrıca hesaplanmalıdır. Alarm zamanı nedensel olarak
`t_end` olmalıdır. Forecaster'ın ilk sekiz history satırı receptive field, son dört satırı
score desteğidir; anomalik history/normal target pencereleri `history_contaminated` geçiş
tabakası olarak raporlanmalı, doğrudan target-positive yapılmamalıdır.

Bozuk dosyanın onset-öncesi pencereleri scenario timeline'ında temiz-negatif/FP sanity check
olarak tutulmalıdır. Headline doğal alert burden ise yalnız clean/doğal exposure'ı **bir kez**
kullanmalı; bozuk dosya kopyaları bu paydaya yeniden eklenmemelidir. Sentetik AUC'nin negatif
havuzunda da clean Parquet'te birebir bulunan aynı pencereler ikinci kez ağırlıklandırılmamalıdır.
Bu dışlama “zor örneği silmek” değil, duplike temiz gözlemi tekilleştirmektir. Testlere
satır-truth → pencere desteği → etiket → metrik
entegrasyon testi ve forecaster destek testi eklenmelidir; mevcut windowing testleri yalnız
şekil/gap/NaN davranışını kapsıyor (`tests/test_adsb_windowing.py:25-54`).

**Tahmini efor.** Truth şeması, v2 üretim ve testler 1 iş günü; metrik/timeline yeniden koşusu
0.5–1 iş günü. Bunlar planlama tahminidir.

**Risk.** `q>0` çok kısa teması pozitif yapar; bu nedenle `q` tabakaları ve event düzeyi
metrik şarttır. Oran eşiğini mevcut sonuca göre seçmek post-hoc ayar olur. V1'in üstüne yazmak
provenance'i yok eder.

## 2. Stealthy ramp için nedensel CUSUM

**Bulgu.** CUSUM mevcut eşiklenmiş `rule_penalty` üzerinde değil, fiziksel işaretini koruyan
ham residual üzerinde kurulmalıdır. Mevcut `speed_residual` skaler büyüklük farkıdır; kuzeye
sabit 2 m/s konum kaymasının etkisi uçağın başına göre projekte olduğu için birikim sinyalini
zayıflatabilir.

**Kanıt.** Rule penalty, `|z|<=3` bölgesini tam sıfır yapıyor
(`adsb/rules.py:79-90`); dolayısıyla 3×MAD altındaki drift CUSUM'a verilirse bilgi zaten
kaybolmuştur. Mevcut train kalibrasyonunda `speed_residual` medyanı `0.1337676 m/s`, robust
MAD'i `1.9296692 m/s` ve 3×MAD'i yaklaşık `5.7890 m/s`'dir
(`artifacts/adsb/models/rule_scorer_report.json:22-25`). Reçete 2 m/s ramp kullanır
(`adsb/synthetic.py:89-101`).

İnceleme ölçümü: ilk Parquet row-group'unda tamamlanmış **2,095** uçuşta, parça sınırındaki
iki eksik uçuş dışlanarak mevcut north-bearing ramp tekrar oynatıldı. Skaler
`speed_residual` için uçuş-başı post-onset imzalı ortalama değişimin mutlak değerlerinin
uçuşlar-arası medyanı yalnız **0.84449 m/s**, kuzey yönlü vektör residual medyan değişimi
**-2.0 m/s** oldu. Bu bir teşhis ölçümüdür; `k/h` seçimi değildir.

Eski hattın kullanılacak **fikri**, kodu değil: full-flight MAD geçmiş skorları geleceğe göre
değiştirmiş ve causal ROC `0.878→0.611` düşmüştü
(`archive/2026-07-10_legacy_non_adsb_ml/docs/ML1_BULGULAR_VE_HATALAR.md:204-214`). Plan daha
sonra train-normal merkez/ölçek, moving-block bootstrap, reset/refractory ve prefix checksum
önermişti (`archive/2026-07-10_legacy_non_adsb_ml/docs/ML8_PLAN.md:198-208`). Bu rapor
arşivden kod taşımayı önermiyor.

**Öneri.** Önce bildirilen ground speed/track'i ve konum türevini doğu-kuzey bileşenlerine
ayıran nedensel residual üret:

\[
v^{rep}_E=gs\sin(\chi),\quad v^{rep}_N=gs\cos(\chi),\quad
e_E=v^{rep}_E-v^{pos}_E,\quad e_N=v^{rep}_N-v^{pos}_N.
\]

`v_pos`, yalnız `t` ve `t-1` konumlarından hesaplansın. Fit-normal altkümesinden sabit
medyan ve `1.4826×MAD` ile işaretli \(z_t\) üret; MAD=0 bileşeni dışla. Ön-kayıtlı iki taraflı
Page CUSUM adayı:

\[
C_t^+=\max(0,C_{t-1}^+ + \operatorname{clip}(z_t,-3,3)-k),
\]
\[
C_t^-=\max(0,C_{t-1}^- - \operatorname{clip}(z_t,-3,3)-k),\qquad
\max(C_t^+,C_t^-)>h.
\]

Bu formül tek bileşen/tek yön gösterimidir; gerçek alarm doğu/kuzey × pozitif/negatif dört
state'in birleşik maksimumıdır. Page'in özgün sürekli denetim yaklaşımı yöntem dayanağıdır
([Page, Biometrika 1954](https://doi.org/10.1093/biomet/41.1-2.100)). `k`, önceden seçilen
asgari fiziksel kaymanın normalize değeri için `k=δ*/2` olsun. Mevcut kuzey-yönlü 2 m/s
reçetede north bileşeni için \(\delta_*=2/s_{N,train}\) kullanılabilir; kerterizi bilinmeyen
2 m/s hedefte en büyük eksen bileşeni alt sınırı `2/√2 m/s`'dir veya dönüşten-bağımsız bir
vektör CUSUM ön-kayıtlanmalıdır. `h` sentetik recall'a göre değil, 2026-02-28'in uçuş-hash ile ayrılmış yalnız
normal kalibrasyon bölümünde, önceden yazılmış doğal alarm-episode/saat bütçesini sağlayan
moving-block bootstrap kuralıyla seçilsin; blok bootstrap seri bağımlılığı korumak içindir
([Künsch, Annals of Statistics 1989](https://doi.org/10.1214/aos/1176347265)). Alarm bütçesi
önceden dondurulmadan `h` nihai seçilemez. Dört state ayrı ayrı bütçelenmemeli; birleşik max
alarmı tek toplam doğal FA/saat bütçesine kalibre edilmelidir.

Uçuş başında, `on_ground=True`, out-of-order `dt<0` veya `dt>60 s` durumunda reset;
`dt=0` duplike/eşzamanlı satırlar nedensel birleştirilmeli veya state güncellemeden atlanmalıdır.
Kısa missingness'te state'i güncellemeden taşıma, uzun missingness'te reset uygulanmalı. Her
prefix aynı geçmiş skoru üretmeli; bu prefix-invariance testi zorunludur. Flight-adaptive
full-flight merkez/MAD kullanılmamalıdır. Clean korpus inceleme ölçümünde pozitif cadence
p25/medyan/p75 **1.61/3.96/11.34 s** idi; sample-bazı CUSUM yüksek-cadence uçakları kayırabilir.
Bu nedenle sabit-zaman binleri veya timestamp-aware güncelleme ön-kayıtlanmalı; en azından
natural burden cadence tabakalarında ayrıca raporlanmalıdır.

**Kabaca gecikme — örnek hesap, ölçüm değil.** Mevcut skaler MAD yalnız ölçek örneği alınırsa
`δ=2/1.929669=1.03645`, `k=0.51823` ve ideal beklenen artış yaklaşık `0.51823`/örnek olur;
dolayısıyla `E[N]≈1.9297h` örnektir. Clean korpusta pozitif ve `dt<=60 s` aralıklarının
inceleme-medyanı **3.96 s** idi. Yalnız örnek olarak `h=5` yaklaşık **9.65 örnek / 38.2 s /
0.64 dk**, `h=10` yaklaşık **19.3 örnek / 76.4 s / 1.27 dk** verir. `h=5/10` seçilmiş eşik
değildir. `E[N]≈h/(δ-k)` yaklaşımı iid/kararlı post-change mean shift, clipping'in etkin
olmaması ve robust standardizasyonun mean/σ gibi davranması varsayımlarına dayanır; yön
geometrisi, otokorelasyon, eksiklik, cadence ve resetler gerçek gecikmeyi uzatabilir.

**Tahmini efor.** Vektör residual, nedensellik testleri, kalibrasyon ve raporlama 1–2 iş günü.

**Risk.** Thresholded penalty üstünde CUSUM kurmak stealth sinyali geri döndürülemez biçimde
siler. Sentetik ramp ile `h` seçmek değerlendirmeyi eğitime dönüştürür. Reset/freshness hatası
uzun gap'leri sahte birikime çevirebilir.

## 3. Tam hacim ve yeni gün stratejisi

**Bulgu.** “Tam hacim” 638 parçayı ve bütün örtüşen pencereleri RAM'e `concat` etmek veya
NN'e yığmak olmamalıdır. Öncelikli amaç üç gerçek günde donmuş rule/CUSUM/S2'nin doğal alarm
yükünü akışkan ölçmek; yeni 2025-06-15 verisini train'e katmak değil, kör domain-shift adayı
olarak kapalı tutmaktır.

**Kanıt.** Baseline ve rule scriptleri seçili Parquet'leri bütün kolonlarla `pd.concat` eder
(`scripts/adsb_train_baseline_models.py:67-72`,
`scripts/adsb_evaluate_rule_scorer.py:54-58`). İlk 60 parça yalnız 2026-02-28 gününden,
inceleme ölçümünde **22,520,807 satır / 692,466,521 byte** idi. Mevcut rapor
**2,849,437 train** ve **705,969 validation** penceresi kaydeder
(`artifacts/adsb/models/baseline_training_report.json:2-3`). `build_windows` ham `X/M`'yi
float32 üretir (`adsb/windowing.py:35-36,44-65`), fakat scaler medyan/MAD'i float64'tür ve
transform ölçeklenmiş `X`'i float64'e yükseltir (`adsb/scaling.py:27-28,40-45`). Fonksiyon
dönüşü sonrasındaki scaled-`X` + `M` için teorik alt sınır ilk 60 parçada
**4,095,827,712 byte**; 60→638 doğrusal izdüşümü yaklaşık **43.6 GB**'dır. Preprocessing
anında ham `X` de yaşarken alt sınır **5,461,103,616 byte**, doğrusal izdüşüm yaklaşık
**58.1 GB** olur. Bunlar **tahmindir**; scaler geçicileri, DataFrame, liste, Torch tensor,
model ve optimizer belleği dahil değildir. Batched scoring
(`scripts/adsb_train_baseline_models.py:119-128`) hazırlama/eğitim yığılmasını çözmez.

Sentetik üretici her flight için tüm segment tablosunu tekrar filtreliyor
(`scripts/adsb_generate_synthetic_dataset.py:65-73`); bu tam hacimde karesel-benzeri tarama
yüküdür. Gün/parça metadata ölçümünde üç güne ait gerçek toplam, önceki bölümde verilen
**256,155,009 satırdır**.

**Öneri.** Sabit gün rolü:

- **2026-02-28:** normal fit + uçuş-hash ile ayrılmış normal kalibrasyon; sentetik v1'in
  kaynak flight kimlikleri her iki fit kümesinden kalıcı olarak dışlanır.
- **2026-03-01:** development/generalization; merkez/ölçek/`k/h` 2026-02-28 calibration'da
  ön-kayıtlı algoritmayla dondurulmuş halde sınanır. Başarısızlık görülürse aynı run içinde
  elle ayar yok; yeni hipotez, config ve versioned development run gerekir.
- **2026-03-16:** donmuş temporal rehearsal/dev-test. Bu incelemede dağılımları okunduğu için
  artık gerçek blind holdout değildir.
- **2025-06-15:** tek-atımlık kapalı domain-shift holdout adayı; train veya development'a
  alınmaz.

Akış tasarımı: değişmez input manifesti oluştur; yalnız gerekli kolonları oku; parça/aircraft
bazında segment→feature→row score→event özeti üret; tüm `X/M`'yi tutma. Her run'da bir
`(gün, source_id)` akışının tek parçada olduğu assertion ile doğrulansın; flight anahtarı
gün/source/sequence bağlamını içersin. Rule medyanı için disk destekli deterministik dış-sıralama
ve MAD için donmuş medyanla ikinci geçiş kullanılabilir. Approximate quantile seçilirse algoritma,
seed ve hata sınırı manifestte önceden dondurulmalıdır. MAD=0 kanal yine dışlanır. NN'e daha
sonra gerekirse `IterableDataset`, sınırlı shuffle buffer ve sabit flight-hash örneklemesi
kullanılmalı; “daha çok veri” bütün pencereleri belleğe almakla eşitlenmemelidir. Sentetik
üretimde flight `groupby` tek geçiş ve stream writer kullanılmalı, v1 dosyaları korunmalıdır.

**Tahmini efor.** Manifest/split guard 0.5 gün; streaming rule/S2 1–2 gün; tam koşu ve QA
0.5–1 gün; run/holdout kilidi 0.5 gün. Ölçülmüş çalışma süresi değildir.

**Risk.** Günler rastgele karıştırılırsa zamansal genelleme ölçülemez. Sentetik-korpus kaynak
flight'larını geniş fit setine geri almak sızıntıdır. Approximate median/MAD yöntemini sonuçtan
sonra değiştirmek kalibrasyonu post-hoc yapar.

## 4. S2: squawk/emergency ve konum kalite kanalları

**Bulgu.** `squawk/emergency`, bildirilmiş operasyonel durum; `NIC/NACp/SIL`, yayıncının
bildirdiği konum doğruluk/bütünlük kalitesidir. Bunlar “kesin saldırı ground-truth'u” değildir ve
residual penalty toplamına katılmamalıdır. Özellikle NIC/NACp/SIL için naif mutlak eşikler
doğal veride büyük alarm yükü üretir.

**Kanıt.** Parser `squawk`, `emergency`, `nic`, `nac_p`, `sil` ve ADS-B version alanlarını
Silver'a yazıyor (`src/silver/parse_adsblol_historical.py:115-123`). Fakat sparse `ac_dict`
alanını `last_ac` içinde ileri taşıyor ve gap/yeni leg'de sıfırlamıyor
(`src/silver/parse_adsblol_historical.py:75-82,92-123`); ardışık aynı satırlar bağımsız yeni
beyan değildir.

İnceleme ölçümü: tüm 638 parça yalnız ilgili kolonlarla akışkan tarandı; flight sınırı
`source_id` değişimi veya `gap>1800 s`, episode aktif duruma yükselen kenar ya da aktif
durumla yeni flight'a giriş olarak sayıldı. `emergency` için non-null ve literal `none`
dışındaki değerler aktifti; state her Parquet parçası başında sıfırlandı. Gün-bazı benzersiz
`source_id` sayıları **70,970 / 64,566 / 55,213** idi ve aynı `(gün, source_id)` değerinin
birden fazla parçaya yayılması ölçülmedi. Bu tanımla **542,461 flight segmentinde**:

| Durum | Satır | Episode |
|---|---:|---:|
| squawk 7500 | 502 | 8 |
| squawk 7600 | 2,360 | 16 |
| squawk 7700 | 11,399 | 16 |
| `emergency != none` | 44,771 | 157 |

Eşleşen satırlar `7500+unlawful: 318/502`, `7600+nordo: 2,220/2,360`,
`7700+general: 11,343/11,399` idi. Gap üzerinden taşınmış state, episode başlangıçlarının
sırasıyla **3/8, 1/16, 0/16** ve tüm emergency episode'larının **32/157**'sinde görüldü.
Dolayısıyla “üç ardışık satır” debounce'u üç bağımsız ADS-B güncellemesi anlamına gelmez.

Aynı taramada NIC<7 **12,655,396 satır (%4.9405)**, NACp<8 **5,068,287 satır
(%1.9786)** ve SIL<3 **22,737,205 satır (%8.8763)** kapsadı. Null sayıları NIC için
**152,479**, NACp/SIL için ayrı ayrı **3,579,446** idi. Bunlar event/FA sayısı değil,
satır sayısıdır. ADS-B version `3–7` için **19,994**, NACp `12–15` için **104** satır da
provider şeması açısından reserved/out-of-domain veri-kalite vakasıdır.

FAA dokümanında NACp≥8, NIC≥7 ve SIL=3, belirli ABD ADS-B Out/§91.227 bağlamındaki performans
referanslarıdır; dünya geneline saldırı eşiği değildir
([FAA AC 20-165B](https://www.faa.gov/documentlibrary/media/advisory_circular/ac_20-165b.pdf),
[FAA AC 90-114C](https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_90-114C.pdf)).
readsb alan sözlüğü squawk'ı dört oktal rakam, emergency'yi kategorik durum; NIC/NACp/SIL'i
kalite/bütünlük alanları olarak tanımlar
([readsb JSON dokümantasyonu](https://github.com/wiedehopf/readsb/blob/dev/README-json.md)).
FAA 7500'ü unlawful interference kodu olarak tanımlar
([FAA AIP ENR 1.13](https://www.faa.gov/air_traffic/publications/atpubs/aip_html/part2_enr_section_1.13.html)).

**Öneri.** S2 iki ana reason-code kanalı üretsin:

1. `declared_status`: her kritik squawk ve her `emergency != none` bağımsız `declared`
   reason code'udur; `7500↔unlawful`, `7600↔nordo`, `7700↔general` aynı anda ve fresh ise
   `corroborated`. Tek alan/null/stale veya eşleşmeyen fakat açıkça karşıt olmayan durum
   `not_corroborated`; ancak iki alan da fresh olup farklı kritik durumları açıkça bildiriyorsa
   `contradictory` denir. Hiçbiri alarmı bastırmaz. `lifeguard/minfuel/downed/reserved` ayrı
   beyan tipleridir. Freshness, clear/expiry ve yükselen-kenar episode semantiği parser
   düzeltmesinden sonra ön-kayıtlanmalıdır.
2. `position_quality`: reserved/out-of-domain → `schema_invalid`; null → `missing`;
   standardın kodlanmış sıfırı → `reported_unknown_or_unavailable`; airborne ve uygun ADS-B
   version/scope altında NIC<7/NACp<8/SIL<3 → yalnız
   `below_faa_reference` advisory. Temporal düşüş ancak parser gerçek alan-güncellemesi,
   update timestamp/age ve mümkünse `sil_type/sda/nac_v` bilgisini koruduktan sonra
   kullanılmalıdır.

Bu kanallar anomaly penalty'ye toplanmasın; doğal episode sıklığı ve scoreable flight-hour
başına yük ayrı raporlansın. ATC/olay ground-truth'u olmadığı için gerçek 7500/7600/7700
beyanına “false positive” denmemeli. `not_corroborated/contradictory` veri tutarlılığı ve
freshness bulgusudur, saldırı etiketi değildir.

**Tahmini efor.** Parser freshness/şema kararı 0.5–1 gün; S2 episode mantığı ve doğal-yük
raporu 1 gün. Ölçülmüş süre değildir.

**Risk.** Forward-fill'i bağımsız yayın sanmak episode sayısını şişirir. FAA değerlerini
küresel kesin-anomali eşiğine çevirmek yüksek alarm yükü ve yanlış semantik üretir. Meşru acil
durumu saldırı diye etiketlemek değerlendirme ground-truth'unu bozar.

## 5. NN hattı ve USAD kararı

**Bulgu.** Kural+CUSUM ana odakken üç çalışan NN referans olarak dondurulmalıdır. Ağırlıklı
loss fikri bilimsel olarak hâlâ ucuz bir falsifikasyon deneyi değerindedir, fakat truth ve doğal
alarm ölçümü düzelmeden yapılmamalı; 3–5× taraması yapılmamalıdır. USAD'ın tam ölçekli çözümüne
şimdi yatırım yapılmamalıdır.

**Kanıt.** Mevcut üç NN'in train-vs-untrained ve train-vs-magnitude korelasyonları eşik
`0.8` üstündedir: Dense `0.8626/0.8956`, LSTM-AE `0.8428/0.8884`, forecaster
`0.9401/0.9228`; üçü de flagged'dir
(`artifacts/adsb/models/baseline_training_report.json:34-38,5526790-5526793,11087257-11087260`).
Loss bütün sonlu kanal/hücrelerin MSE'sini eşit toplar (`adsb/windowing.py:71-75`). Kural
round-2'de pooled `0.599760`, track-frozen `0.679362` üretmiştir
(`artifacts/adsb/models/rule_scorer_report.json:896,2169`); bu karşılaştırma mevcut etiket
hatası nedeniyle yalnız yönlendirici kanıttır.

Mevcut script yalnız özet JSON'u yazar; model checkpoint/state_dict veya pencere-bazı skor/meta
saklamaz (`scripts/adsb_train_baseline_models.py:256-259`). Dolayısıyla tarihsel NN raporu
dondurulabilir, fakat corrected truth geçmiş NN skorlarına geriye dönük uygulanamaz; yeniden
karşılaştırma yeni, versioned eğitim ve skorlama gerektirir.

USAD decoder'ları sınırsız lineer çıkışlıdır (`adsb/models/usad.py:35-42`); ikinci optimizer
loss'u yeniden-yapılandırma terimini negatif işaretle içerir (`adsb/models/usad.py:105-113`).
Bu adversarial amaç yapısaldır, fakat sınırsız çıkış sayısal patlamayı kolaylaştırabilir. Özgün
USAD uygulamasının decoder sonunda sigmoid kullanması bu farkı şüpheli kılar
([resmî USAD kodu](https://raw.githubusercontent.com/manigalati/usad/master/usad.py),
[USAD makalesi](https://doi.org/10.1145/3394486.3403392)); bunun mevcut patlamanın kanıtlanmış
tek kök nedeni olduğu **ölçülmedi**. Maskelenmiş girdide AE1 çıktısının tekrar encoder'a verilmesi
de ayrıca doğrulanmalıdır.

**Öneri.** Önceki NN raporlarını değiştirmeden sakla. Truth/CUSUM/S2 hattı donduktan sonra
Dense-AE için tek, iki-kollu falsifikasyon deneyi ön-kaydet: aynı seed/split/config ile
unweighted kontrol ve fixed-weight treatment. Fit-normal MAD'i pozitif raw kanallar `1`,
fit-normal MAD'i pozitif residual kanallar `4` — `4×`, kullanıcının önerdiği 3–5× aralığının
sabit orta noktasıdır; grid/sweep yok. Train MAD'i tam sıfır olan kanal model girdisi/loss/skordan
çıkarılır ve manifestte yazılır; floor yok. Mevcut splitte bu, `altitude_source_residual`ı
dışarıda bırakıp üç residual kanalı ağırlıklandırır
(`artifacts/adsb/models/rule_scorer_report.json:31-33`).

Örnek-bazı loss formülü de dondurulsun:

\[
L=\frac{\sum_{t,c} w_c M_{t,c}(x_{t,c}-\hat{x}_{t,c})^2}
{\sum_{t,c} w_c M_{t,c}}.
\]

Kanal-bazı train/validation MSE ayrıca raporlansın. Her eğitimden sonra mevcut
`magnitude_domination_check` zorunlu olsun. Falsifikasyon ancak treatment, paired control'e
göre aynı ön-kayıtlı doğal alarm-episode/saat bütçesinde corrected event recall/gecikmesini
iyileştirir ve `magnitude_domination_flagged == false` olursa geçer; bu, iki korelasyonun da
mevcut `0.8` sınırının altında olması demektir. Synthetic AUC tek başına gate değildir. Bu
sonuç fusion izni vermez: Dense standalone karşılaştırılır; OR/max/ağırlık gibi fusion ayrı ve
sonuç görülmeden dondurulmuş deney gerektirir. Ana hatta terfi için standalone rule+CUSUM
referansına karşı aynı burden'da fayda göstermelidir. Fayda yoksa NN hattını beklemeye al;
varsa aynı sabit ayarla yalnız LSTM-AE replikasyonu değerlendir.

USAD bu fazda “unvalidated/deferred” olarak kalsın. Dense falsifikasyonu başarısızsa USAD'ı
ADSB-1 aktif aday listesinden ele. Dense başarılı olursa bile önce özgün ölçek aralığına uyumlu
bounded-output + mask davranışı için küçük, sabit smoke test; sonlu loss ve magnitude kontrolü
geçmeden tam koşu yok. Bu yeni uygulama arşivden kod kopyalamamalıdır.

**Tahmini efor.** Dense iki-kollu deney 1–2 gün; USAD yalnız koşullu smoke 0.5 gün.

**Risk.** 3/4/5× sonuçlarına bakıp en iyisini seçmek post-hoc tuning'dir. AUC yükselirken doğal
alarm yükü kötüleşebilir. USAD debug'ı ana fizik/truth hattını geciktirebilir ve “çalıştı” sonucu
operasyonel fayda göstermeyebilir.

## 6. `altitude_dropout` ve missingness

**Bulgu.** Eksikliği **fiziksel rule residual penalty'sinde** `NaN→0 katkı` olarak bırakmak
doğrudur; bu global/NN missingness politikası değildir. Ayrı availability durumları gerekir.
Normal verideki yaklaşık yüzde on eksiklik neredeyse tamamen
`on_ground` kaynaklıdır. En yüksek özgüllüklü sentetik durum “airborne baro altitude yok,
geometric altitude var”dır; “iki irtifa da yok” ise doğal veride de görülen daha zayıf bir
availability uyarısıdır.

**Kanıt.** Rule NaN katkısını sıfırlar (`adsb/rules.py:79-90`). İnceleme ölçümü, mevcut
`data/objectstore/synthetic/adsb/clean.parquet` üzerinde:

- bütün satırlarda `alt` missing oranı **%10.5621**;
- bütün satırların **%10.5105**'i `on_ground=True` ve bunların **%100**'ünde `alt` missing;
- airborne satırlarda `alt` missing oranı **%0.05766** ve bu satırların hiçbirinde
  `alt_geom_m` sonlu değildi;
- iki altitude'un da airborne missing olduğu **2,305 satır / 37 flight** vardı; detector ile
  tutarlı `gap>60 s` reset semantiğiyle **68 run**, süre medyanı **43.055 s**, p95'i
  **1,209.908 s** idi.

Bu ölçümler 8,910 uçuşlu clean korpusa aittir
(`data/objectstore/synthetic/adsb/manifest.json:2-6`), tüm günlere genellenemez. Mevcut
dropout reçetesi 8,910 flight'ın **8,235**'inde sonlu `alt` değerini gerçekten değiştirdi;
bunların **7,973**'ünde katı baro-only state görüldü. Bu, detector recall'ı değil, eşiksiz
**strict-state observability coverage = %96.82**'dir; detector recall henüz ölçülmedi. Clean
korpusta katı state sıfır episode; iki interval ucu da airborne ve `0<dt<=60 s` alınarak
hesaplanan exposure **8,049.689 saat** idi. Bağımsız Poisson varsayımı altında
“üç kuralı” üst sınırı yaklaşık **0.000373/saat** olur; bu yalnız **model-varsayımlı örnek
hesaptır**, flight kümelenmesi ve tek-gün seçimi nedeniyle operasyonel güven sınırı değildir.

**Öneri.** Residual'dan ayrı availability durumları:

- `GROUND_ALT_NOT_APPLICABLE`: `on_ground=True`; abstain, alarm yok.
- `BARO_ALT_DROPOUT`: airborne, `alt` null, `alt_geom_m` sonlu; yüksek özgüllüklü event adayı.
- `ALL_ALTITUDE_UNAVAILABLE`: airborne, ikisi de null; availability/data-quality uyarısı,
  davranış anomalisi değil.
- `MESSAGE_GAP`: gözlem sessizliği; satır-içi altitude missingness'ten ayrı interval durumu.

Episode başlangıç/bitişi timestamp ile çıkarılsın; scoreable exposure ve flight-hour ayrı
raporlansın. Persistence eşiği mevcut natural run sürelerine veya sentetik sonuca bakıp elle
seçilmemeli; ön-kayıtlı normal-kalibrasyon prosedürü ve alarm bütçesiyle belirlenmelidir.
Sentetik v2 truth, dropout'un gerçek rastgele bloğunu kullanmalıdır; onset→uçuş sonu etiketi
kullanılmamalıdır.

**Tahmini efor.** Durum üretimi, exact-truth testi ve doğal-yük raporu 0.5–1 gün.

**Risk.** `alt is null` tek başına kural yapılırsa yerdeki beklenen eksiklik alarm üretir.
İki altitude da yokken persistence tek başına kesinlik sağlamaz; clean korpusta uzun doğal
run'lar vardır. Tek gün ve sıfır gözlemden production FA garantisi çıkarılamaz.

## 7. Değerlendirme birimi ve gate'ler

**Bulgu.** Pencere-AUC S0/S1 için yararlı bir tanı metriğidir, fakat tek başına yeterli değildir.
Event-onset, aktif durum ve doğal alarm yükü **şimdi**, truth düzeltmesiyle aynı turda eklenmelidir;
S3'e ertelenmemelidir. Satır, pencere, event ve flight birimleri ayrı tutulmalıdır.

**Kanıt.** README pencereyi pragmatik varsayılan seçiyor çünkü dört model pencere üstünde
çalışıyor (`adsb/README.md:33-35`); rule ise önce satır penalty'si üretip sonra sırf NN ile
karşılaştırma için pencere ortalaması alıyor
(`scripts/adsb_evaluate_rule_scorer.py:43-51`). Mevcut JSON'larda event gecikmesi veya
alarm-episode/scoreable-hour yoktur. `confidence_threshold=0.95`, train score'un robust
z-değerine normal CDF uygulanan bir extremeness dönüşümüdür; kalibre olasılık/p-değeri değildir
(`docs/decisions.md:939-942`).

İnceleme-türevi örnek: aynı 705,787 clean pencere için `conf>=0.95` rule confusion
matrix'inde **254,687** pencereyi işaretler, yani **%36.0855**
(`artifacts/adsb/models/rule_scorer_report.json:49-56`); Dense-AE **156,204**, yani
**%22.1319** işaretler
(`artifacts/adsb/models/baseline_training_report.json:53-60`). Bunlar birbirine örtüşen
pencere oranlarıdır, FA/saat değildir ve clean korpus da yalnız seçilmiş bir güne dayanır.

**Öneri.** Yeni metric contract:

| Birim | Birincil rapor | Yorum |
|---|---|---|
| Pencere | AUROC + AUPRC; `q=0`, `0<q<1`, `q=1` tabakaları | yalnız tanı/karşılaştırma |
| Aktif aralık | zaman-ağırlıklı coverage ve precision; event-macro coverage | point-adjust yok |
| Event | event recall, ilk alarm gecikmesi, ön-kayıtlı gecikme bütçesinde recall | tek isabet tüm aralığı TP yapmaz |
| Doğal trafik | alert episode/scoreable flight-hour, clean flight'ların işaretlenen oranı | sentetik recall ile daima yan yana |
| Flight | event içeren flight recall ve flight başı alarm yükü | pencere metriğine karıştırılmaz |

Scoreability/eligibility paydası ayrıca raporlansın. Alarm episode merge/debounce/refractory
kuralı sonuç görülmeden önce dondurulsun. `conf>=0.95` mevcut raporda yalnız diagnostic isimle
korunsun; yeni operasyon eşiği normal calibration'da doğal alarm bütçesinden seçilsin. Doğal
veride olay etiketi yoksa “false alarm” yerine dürüstçe “nominal alert burden” denilsin; ancak
sentetik recall ile eşleştirilen clean referans için aynı donmuş kuralla alert burden mutlaka
verilsin. NAB gibi onset-duyarlı çerçeveler gecikme raporlamasının dayanağı olabilir
([Numenta Anomaly Benchmark](https://arxiv.org/abs/1510.03336)); tek bir literatür metriği domain
sözleşmesinin yerini almaz.

**Tahmini efor.** Metric contract, eventizer ve testler 1–2 iş günü; truth işiyle paralel
yürütülebilir.

**Risk.** Pencere bağımlılığını yok saymak güven aralıklarını aşırı dar gösterir. Point-adjust
skoru yapay yükseltir. Sentetik recall'ı doğal alarm yükü olmadan sunmak dokunulmaz kısıtı ihlal
eder.

## 8. Kör-holdout tanımı

**Bulgu.** En güçlü mevcut aday, Downloads'taki `v2025.06.15-...-003.tar` dosyasının tamamını
tek-atımlık kapalı domain-shift holdout olarak ayırmaktır. Fakat bu gün eğitimden daha eskidir;
ileri-zaman deployment drift'i değil, **geriye dönük temporal/source transfer** ölçer. Dosya adı
tek tam gün yerine bir shard'ı da gösterebilir; içerik, freeze öncesinde kontrol edilmemelidir.

**Kanıt.** Yalnız dosya metadata'sına yapılan inceleme ölçümü:
`C:\Users\PC_5812_YD26\Downloads\v2025.06.15-planes-readsb-prod-0-003.tar`,
**3,093,094,400 byte**, mtime `2026-07-13 12:02:04`. Tar açılmadı. Tarih,
2026-02-28 fit gününden **258 gün eskidir** (takvim farkı). 2026-03-16 ise bu incelemede footer
ve S2 dağılımları için okunduğundan blind sayılamaz.

**Öneri.** Kullanıcı bu dosyayı açıkça holdout seçerse, açmadan önce aşağıdaki protokol
dondurulsun:

1. Raw yol, byte, SHA-256, beklenen tarih/kaynak ve erişim günlüğü salt-okunur manifestte
   kilitlensin. SHA-256 bütün raw byte'ları okur; burada “açmamak”, tar üyelerini
   listelememek/decompress etmemek ve içerik istatistiği çıkarmamak, yalnız loglanmış hash
   geçişine izin vermek demektir.
2. Parser commit/şema, gerekli kolonlar, unit dönüşümleri, flight/window/interval truth,
   scoreability, feature ve reason-code listesi, scaler/rule kalibrasyonu, CUSUM `k/h/reset`,
   eşikler, event merge/refractory ve metric/gate'ler hashlenerek dondurulsun.
3. Holdout normal Silver dizinine yazılmasın. `parse_local_tar()` mevcut target'a append edip
   rerun'da duplicate üretebilir (`src/silver/parse_adsblol_historical.py:192-202,267-268`);
   standart `run()` ise mevcut Silver prefix'ini önce siler ve tarı belleğe indirir
   (`src/silver/parse_adsblol_historical.py:212-229`). Bu nedenle parser'a ayrı output-target
   desteği veya izole object-store/prefix gerekir. Namespace bir kez yazılır, doğrulanır,
   sonra sealed/read-only yapılır; yazım anında “salt-okunur” denmez.
4. İlk mekanik kontrolde `-003` kapsamı/timestamp/şema doğrulansın. Başarısız şema/parser
   dış-geçerlilik sonucu olarak kaydedilsin; aynı holdout'a bakarak feature/eşik/parser tamiri
   yapılıp yeniden “blind” sonuç üretilmesin.
5. Üye/satır dahil-etme ve parse-hata kuralları önceden dondurulsun. Primary sonuç bu
   sözleşmedeki bütün kapsamda raporlansın; toplam tar üyesi, başarılı/başarısız parse,
   dışlanan satır/flight ve reason code'ları zorunlu attrition tablosu olsun. İyi görünen saat
   veya aircraft seçilmesin. Development günlerinde hiç görülmemiş `source_id` altkümesi
   ön-kayıtlı secondary olabilir, primary'nin yerini alamaz.
6. Gerçek anomaly etiketi yoksa primary sonuç doğal alert/event burden ve scoreability'dir.
   Sentetik enjeksiyon yalnız donmuş reçete/truth ile secondary olarak ve aynı doğal burden'ın
   yanında raporlanabilir; sentetik asla fit/kalibrasyona girmez.
7. Bu holdout yalnız truth, rule+CUSUM/S2, doğal alarm bütçesi ve run manifesti development'ta
   donduktan; insan gate kararı kaydedildikten sonra bir kez açılır.

İleri deployment iddiası için daha sonra **2026-03-16 sonrasından** prospektif bir gün ayrıca
kilitlenmelidir; hangi gün olduğu şu an ölçülmedi/seçilmedi.

**Tahmini efor.** Freeze manifest/protokol 0.5 gün; kullanıcı gate'i sonrası tek parse/eval
0.5–1 gün. Holdout'u açma bu raporun 1–2 haftalık varsayılan planına dahil değildir.

**Risk.** Hash/freeze öncesi tar envanteri bile eşik veya şema kararını etkileyebilir. Eski günü
“gelecek drift” diye sunmak dış-geçerliliği abartır. Holdout'a özel parser düzeltmesi tek-atımlık
körlüğü tüketir.

## Dokunulmaz kısıt denetimi

**Bulgu.** Mevcut hat, sentetik train sızıntısı için runtime assertion ve sentetik path guard
taşıyor; magnitude check mevcut üç eğitimde çalışmış; çalışan rule kodu MAD=0 kanalı dışlıyor.
Eksik kalan iki temel koruma run fail-if-exists/versioning ve sentetik sonucun doğal
alert-burden ile zorunlu eşleştirilmesidir. Blind holdout henüz tanımlı değildir.

**Kanıt.** Train/synthetic flight kesişimi hata veriyor
(`scripts/adsb_train_baseline_models.py:88-100`,
`scripts/adsb_evaluate_rule_scorer.py:69-75`); path guard `synthetic` sözcüğünü zorunlu kılıyor
(`adsb/synthetic.py:124-132`); magnitude check forecaster dahil çağrılıyor
(`scripts/adsb_train_baseline_models.py:205-212,223-230,246-252`); MAD=0 `continue` ile
dışlanıyor (`adsb/rules.py:56-76`). Doğal FA/saat mevcut raporlarda yoktur. Holdout'un henüz
tanımlanmadığı `docs/decisions.md:824,883` içinde de kayıtlıdır.

**Öneri.** Dokuz kısıt run manifestinde makinece kontrol edilen gate listesi olsun: synthetic
ID/path ayrımı; fail-if-exists; result sonrası config hash değişmezliği; holdout access log;
archive import taraması; her train'de magnitude JSON; synthetic recall yanında doğal burden;
MAD=0 exclusion assertion; commit mesajında `Co-Authored-By` yokluğu. Arşivden yalnız bu raporda
belirtilen yöntem dersi alındı, kod kopyalama önerilmedi.

**Tahmini efor.** Çapraz-kesit guard/testlerin temel sürümü 0.5–1 gün.

**Risk.** İnsan hafızasına bırakılan kısıt, yeni script/run yolunda sessizce atlanır.

## Önerilen sıra — 1–2 haftalık somut plan

Aşağıdaki sürelerin tamamı **planlama tahmini**, ölçülmüş süre değildir. Holdout kapalı kalır.

| Sıra / hedef gün | Çıktı ve geçiş koşulu |
|---|---|
| **1 / Gün 1** | Run manifesti, fail-if-exists, giriş/split hashleri ve makinece kısıt gate'leri. 256,155,009 vs 256,150,550 farkı provenance notuyla kayıt altına alınır; sessiz düzeltme yok. |
| **2 / Gün 1–2** | Sentetik truth v2 (`injection_active`, `observable_changed`, `evaluable_truth` + event aralığı), mimari-destekli `q_w`, dropout exact block; v1 korunur. Unit + entegrasyon testleri geçmeden skor koşusu yok. |
| **3 / Gün 2–3** | Mevcut rule aynı donmuş kalibrasyonla corrected truth üzerinde yeniden skorlanır. NN checkpoint/skorları saklanmadığı için eski NN JSON'ları tarihsel, label-bugged referans olarak korunur ve corrected diye sunulmaz. Window AUC/AUPRC tanı; event recall/gecikme, aktif coverage ve doğal alert burden zorunlu. |
| **4 / Gün 3–5** | Doğu/kuzey hız residual'ı ve causal CUSUM; prefix/reset/missingness testleri. `k` fiziksel 2 m/s hedefinden, `h` yalnız ön-kayıtlı normal kalibrasyon + doğal alarm bütçesinden. |
| **5 / Gün 5–7** | 2026-02-28 fit/calibration akışı; 2026-03-01 development; 2026-03-16 donmuş rehearsal. Bütün süreç streaming; sentetik-kaynak flight'ları fitten dışarıda. |
| **6 / Gün 6–8** | S2 `declared_status` ve `position_quality`, parser freshness/update-age kararı; ground/baro-only/all-altitude/message-gap ayrımlı `altitude_availability`. Residual penalty'den ayrı doğal episode/burden raporu. |
| **7 / Gün 8–9** | Gate incelemesi: truth testleri, magnitude şartı, corrected event metrikleri, doğal burden, günler arası kararlılık ve provenance eksiksizse ana rule+CUSUM/S2 konfigürasyonu dondurulur. |
| **8 / Gün 9–10, koşullu** | Yalnız ana hat stabilse Dense-AE paired control + sabit `4×` treatment; train MAD=0 kanallar iki kolda da hariç, sweep/fusion yok. Magnitude flag kalkmaz veya aynı doğal burden'da treatment faydası yoksa NN beklemeye alınır. USAD yalnız bu deney başarılıysa küçük bounded-output/mask smoke; aksi halde aktif kapsamdan çıkarılır. |
| **9 / Gün 10** | Kullanıcı kararıyla 2025-06-15 raw tar için **açmadan** holdout freeze manifesti/SHA-256 hazırlanır. Bu iki haftada varsayılan olarak parse/açma yok; açılış ayrı, kayıtlı gate kararıdır. |

**İki hafta sonunda beklenen karar:** “yüksek AUC” değil; interval truth'u doğru, causal,
natural alert burden'ı ölçülmüş, günler arası tekrar oynatılabilir bir rule+CUSUM/S2 baseline.
Bu baseline yoksa NN/USAD veya blind-holdout açmak sıralama hatasıdır.
