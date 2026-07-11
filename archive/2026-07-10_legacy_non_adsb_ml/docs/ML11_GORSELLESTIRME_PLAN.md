# ML-11: Görselleştirme ve Veri Keşfi Fazı Planı

Durum: UYGULANDI (2026-07-06). Çıktılar: `scripts/make_visualizations.py`,
`notebooks/09_gorsellestirme_ve_veri_kesfi.ipynb`, `artifacts/viz/*/viz_manifest.json`,
`tests/test_ml11_viz.py`; sonuç özeti `docs/ML1_BULGULAR_VE_HATALAR.md`
"Görselleştirme sonuçları" bölümü, karar kaydı ADR-011. ML-10'dan tamamen bağımsızdır —
önce, sonra veya paralel koşulabilir; çakışan dosya yok.

## §0 Amaç ve bağlam

Mentör talebi: (a) dataset'in kendisinin görselleştirilmesi (kolonlar, satırlar,
sınıflar arası ilişkiler), (b) model eğitimlerinde kalıcı iz — epoch başına loss
eğrisi, confusion matrix, heatmap gibi standart çıktıların kaydedilmesi.

İç amaç (bunun üstüne): "zayıf kategorilerin kök nedeni model değil veri
dağılımı/kapsamı" tezini görsel kanıta dönüştürmek ve **manuel feature engineering
için ölçülü bir öncelik haritası** çıkarmak (hangi feature hangi anomali kategorisini
tek başına ne kadar ayırıyor).

Bu faz **read-only bir analiz fazıdır**: hiçbir model, eşik, split, scaler veya CUSUM
baseline artifact'ı DEĞİŞMEZ. Gate A/B/C gerekmez; onun yerine §7 kabul kriterleri
geçerlidir. İki disiplin kuralı aynen sürer:
1. **Blind holdout (131 SEAD uçuşu) hiçbir görselde/istatistikte kullanılmaz** —
   testle assert edilir.
2. Tek-feature AUC'ler KEŞİF amaçlıdır; oradan feature seçip aynı veriyle sonuç
   raporlamak overfitting olur. Seçilen adaylar ancak yeni bir Gate turuyla
   (ML-12 vb.) değerlendirilebilir.

Dışa-dönük kullanım notu: bu çıktılar mentöre sunulacaksa başlıklar faz numarasıyla
değil içerikle adlandırılmalı ("ML-11" repo-içi kısaltmadır).

## §1 Ortam gerçekleri (2026-07-06'da bu makinede doğrulandı)

- `matplotlib 3.11.0`, `scikit-learn 1.9.0`, `scipy` kurulu — §2-§4'ün tamamı için
  yeterli. **seaborn kurulu DEĞİL ve eklenmeyecek** (matplotlib yeterli, bağımlılık
  şişirmeyelim).
- `umap-learn` bu ortamda kurulaBİLİYOR (`pip install --dry-run` temiz) **ama numpy'ı
  2.5.0 → 2.4.6'ya DÜŞÜRÜYOR** (numba 0.66 pini). Bu yüzden UMAP **opsiyoneldir**:
  kurulacaksa önce tam `pytest` yeşil kanıtı, sonrasında da tekrar tam `pytest` şart.
  Varsayılan teslimat PCA + t-SNE'dir (ikisi de sklearn'de mevcut, kurulum riski sıfır).
- Güncel veri: ALFA 54 uçuş (25 engine / 16 normal / 7 aileron / 4 rudder /
  2 elevator / 1 aileron_rudder / 1 unknown), UAV Attack 19 (6 benign / 6 spoofing /
  6 ping_dos / 1 jamming), SEAD 611 (398 normal / 72 altitude / 60 ext_pos /
  41 mechanical / 40 global_pos; 480 dev + 131 holdout).

## §2 Bölüm 1 — Veri karnesi (kompozisyon görselleri)

Her dataset için:
1. **Sınıf sayım bar chart'ı** (uçuş düzeyi; her barın üstüne n yazılır — küçük-n
   disiplini D.2). SEAD için ayrıca annotasyon-kategorisi bazında event sayıları
   (development matristen, `load_uav_sead_ranges_by_category` ile).
2. **SEAD oturum histogramı**: oturum başına uçuş sayısı (`session_of` ile) —
   heterojen-normal sınırının (D.1) görsel kanıtı. Normal uçuşların kaç oturuma
   dağıldığı başlıkta yazar.
3. **Doluluk haritası**: feature × label **non-null oranı** heatmap'i. Bu,
   B.1/B.2 bulgusunun (alt_local_residual altitude'da %0, baro %7) tüm feature'lara
   genelleştirilmiş hali ve manuel feature engineering'in 1. girdisi: "bu kategoride
   bu sinyal fiziksel olarak yok" haritası.
4. **Uçuş süresi dağılımı** (histogram, label'a göre renkli).

## §3 Bölüm 2 — Embedding projeksiyonları (t-SNE / PCA / opsiyonel UMAP)

1. **Girdi**: scaler'dan geçmiş satır-düzeyi feature vektörleri; NaN'ler median-impute
   (yalnız görselleştirme için). Uçuş başına deterministik subsample (cap ~200 satır,
   `rng(seed=42)`), dataset başına hedef ≤50k nokta — t-SNE bunun üstünde saatler sürer.
2. **PCA** (2 bileşen + açıklanan-varyans grafiği) — global yapı, hızlı taban.
3. **t-SNE** (sklearn, `perplexity=30`, `init='pca'`, sabit seed). Okuma uyarısı
   figür altına yazılır: t-SNE lokal komşuluğu korur; **kümeler-arası mesafe ve küme
   büyüklüğü yorumlanmaz**.
4. **UMAP** (yalnız §1'deki opsiyonel kurulum yapılırsa; `n_neighbors=15`).
5. **Aynı projeksiyon, 4 boyama**: (a) label/annotasyon kategorisi, (b) SEAD oturumu
   (yalnız normaller — normal sınıfın oturum bazında parçalanıp parçalanmadığı),
   (c) normal-vs-anomali ikili, (d) mevcut IF-füzyon skoru (continuous renk skalası —
   modelin neyi "uzak" gördüğünün haritası).
6. Beklenen kullanım: "bazı anomali türleri homojen kümelenmiyor" gözleminin görsel
   kanıtı; hangi kategorinin normal bulutunun İÇİNE düştüğü (tespit edilemezlik) vs
   ayrı adacık oluşturduğu.

## §4 Bölüm 3 — Feature ilişki heatmap'leri

1. **Spearman korelasyon** (feature × feature), scipy hiyerarşik kümelemeyle
   sıralanmış eksenler; |ρ|>0.9 çiftleri `redundant_pairs.csv`'ye.
2. **Tek-feature ayrıştırma haritası (ANA ÇIKTI)**: her feature için
   AUC(normal-satırlar vs kategori-satırları) — feature × annotasyon-kategorisi
   heatmap + `feature_auc_matrix.csv`. Okunuşu:
   - Bir kategori HİÇBİR feature'da ayrışmıyorsa → veri/kapsam sorunu (sinyal yok).
   - Bazı feature'larda ayrışıyor ama füzyon skoru zayıfsa → füzyon/model sorunu
     (feature var, kullanılamıyor) → gerçek feature-engineering adayı.
   Rudder-tipi "seyrek uç-değer" imzaları için AUC'nin yanına per-feature
   **q99-tabanlı ayrım** (kategori q99 / normal q99 oranı) ikinci bir kolon olarak
   eklenir — mean-tabanlı metriklerin kaçırdığını yakalamak için.
3. Tüm hesaplar YALNIZ development split verisiyle (holdout assert'i §7).

## §5 Bölüm 4 — Model tanılama görselleri (mevcut artifact modellerle, eğitim YOK)

1. **Skor dağılımları**: normal vs her kategori, KDE/violin (SEAD: IF-füzyon skoru;
   ALFA: LSTM-AE skoru — `artifacts/models/<source>/` paketlerinden yüklenir).
2. **ROC + PR eğrileri**: uçuş düzeyi, 5 seed'in ortalaması + min-max bandı.
3. **Confusion matrix**: advisory çalışma noktasında uçuş-düzeyi 2×2
   (gerçek normal/anomali × alarm var/yok). Ek olarak **tür-bazlı tespit matrisi**:
   satır = gerçek anomali türü, sütun = tespit edildi/edilmedi, hücrede n —
   unsupervised ikili dedektörde çok-sınıflı klasik CM'in doğru karşılığı budur
   (model sınıf TAHMİN ETMİYOR, bunu figür altına yaz).
4. **Örnek zaman serileri**: kategori başına 1-2 uçuş; skor + eşik + gerçek anomali
   aralığı (ranges) tek eksende overlay — "alarm nerede çaldı / nerede çalmalıydı".

## §6 Bölüm 5 — Eğitim izi altyapısı (kalıcı kural)

1. LSTM-AE eğitim yoluna epoch başına train/val loss kaydı eklenir:
   `artifacts/training_logs/<source>/<model>/<run_id>/loss.csv` + otomatik PNG.
   (IF'in epoch kavramı yok — onun için zorunlu iz yok; istenirse n_estimators
   duyarlılık eğrisi opsiyonel.)
2. **Kural**: bundan sonra eğitilen her model aynı kalıpta iz bırakır (ML-10 Chronos
   zero-shot olduğu için eğitim izi yok ama residual-dağılım kaydı zaten planında var).

## §7 Uygulama düzeni ve kabul kriterleri

Dosyalar:
| Dosya | İş |
|---|---|
| `scripts/make_visualizations.py` (yeni) | `--dataset {alfa,uav_attack,uav_sead} --sections 1,2,3,4` → `artifacts/viz/<dataset>/s{1..4}_*/*.png` + `viz_manifest.json` (sha256 + üretim parametreleri) |
| `notebooks/09_gorsellestirme_ve_veri_kesfi.ipynb` (yeni) | script fonksiyonlarını çağırır; çalıştırılmış, çıktılar gömülü commit edilir |
| `tests/test_ml11_viz.py` (yeni) | (a) subsample deterministik, (b) hiçbir görsel/istatistik holdout uçuşu içermiyor, (c) manifest checksum'ları dosyalarla eşleşiyor |
| LSTM-AE eğitim modülü | §6 loss-log ekleme (davranış değişikliği yok, yalnız kayıt) |

Kabul kriterleri:
1. Üç dataset için §2 ve §4 (bölüm 1+3) üretildi; SEAD ve ALFA için §3 ve §5 tam;
   UAV Attack için §3 opsiyonel (19 uçuş — t-SNE'ye yetecek satır var ama öncelik düşük).
2. `git status`: hiçbir model/threshold/split/scaler/cusum artifact'ı değişmedi.
3. Holdout-izolasyon testi geçiyor (131 uçuş hiçbir yerde kullanılmadı).
4. `feature_auc_matrix.csv` + top-10 manuel feature-engineering adayı listesi
   `docs/ML1_BULGULAR_VE_HATALAR.md`'ye "ML-11 sonuçları" bölümü olarak işlendi.
5. Notebook 09 çalıştırılmış halde; tam `pytest` yeşil (bilinen 4 MinIO SDK hatası hariç).
6. Her figürde n değerleri görünür (küçük-n raporlama disiplini, D.2).

## §8 Çalışma sırası

1. §2 veri karnesi (saf pandas, en hızlı, anında mentöre gösterilebilir)
2. §4 feature heatmap'leri (ana analitik çıktı)
3. §3 t-SNE/PCA projeksiyonları
4. §5 model tanılama görselleri
5. §6 eğitim izi + notebook 09 + testler

ML-10 (Chronos) ile paralel koşulabilir: ML-10 `src/ml/models` + kendi script'ine,
ML-11 `scripts/make_visualizations.py` + `notebooks/09`'a dokunur — kesişim yok.
