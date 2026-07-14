# isolation_forest_contextual_v1 — paralel keşif ön-kaydı

Tarih: 2026-07-14
Durum: **keşif** — `contextual_physics_v1`'i bloklamaz, onu ikame etmez, ana adaya dahil edilmez
Kullanıcı onayı: AskUserQuestion, "Şimdi paralel başlat"

## Neden ayrı bir aday, ayrı bir mekanizma

`contextual_physics_v1` bir **zamansal-sürpriz** dedektörüdür: "bu değer, bu uçuşun kendi yakın
geçmişine göre beklenmedik mi?" sorusuna cevap verir. Isolation Forest yapısal olarak **farklı bir
soru** sorar: "bu residual vektörü, TÜM normal uçuşların residual-uzayında ne kadar izole?" — hiçbir
uçuş-içi geçmişe bakmaz, saf çok-değişkenli yoğunluk/izolasyon mantığıdır.

Bu, iki dedektörün KÖR NOKTALARININ farklı olabileceği anlamına gelir:

- IF, bağlamdan bağımsız ama kanal-KOMBİNASYONU nadir olan bir noktayı yakalayabilir (LSTM bunu
  "geçmişe göre normal" sayıp kaçırabilir).
- LSTM, tek başına nadir olmayan ama BU UÇUŞUN akışına göre ani bir sıçramayı yakalayabilir
  (IF bunu havuzda sık görülen bir kombinasyon sayıp kaçırabilir).

Bu yüzden IF, `contextual_physics_v1`'in YERİNE değil, ONUNLA birlikte değerlendirilecek ayrı bir
sinyal olarak keşfediliyor. Füzyon (varsa) ayrı, sonraki bir ön-kayıt gerektirir — bu belge füzyon
kararı VERMİYOR.

## Sözleşme (contextual_physics_v1 ile aynı disiplin)

- Eğitim SADECE `natural_clean_fit` rolündeki satırlardan; sentetik satır sayısı zorunlu sıfır
  (`StrictNaturalRobustScaler` bunu çalışma zamanında zorlar, adsb/contextual_scaling.py).
- Kanallar: `speed_residual`, `vertical_rate_residual`, `heading_residual`,
  `east_velocity_residual`, `north_velocity_residual` — `contextual_physics_v1` ile BİREBİR aynı 5
  kanal (`altitude_source_residual` aynı sebeple, MAD=0, dışlanır).
- Ölçekleme: aynı `StrictNaturalRobustScaler` (medyan/MAD, clip=5.0, MAD=0 floor'suz dışlama) —
  iki aday arasında ölçekleme kaynaklı bir fark OLMASIN diye kasıtlı paylaşım.
- Sonuç görülmeden dondurulmuş hiperparametreler: `n_estimators=200`, `max_samples="auto"`,
  `contamination="auto"` (yalnız sklearn'in iç `predict()` ofsetini etkiler, KULLANILMAZ),
  `random_state=0`. Skorlama `score_samples()` üzerinden sürekli değer olarak yapılır, ikili
  `predict()` çıktısı hiçbir yerde kullanılmaz.
- **Bilinen basitleştirme (dürüstçe beyan):** LSTM tarafının `availability mask`'ı burada YOK —
  IF, aktif 5 kanaldan herhangi biri NaN olan satırları TAMAMEN atar (complete-case). Bu, eksik
  veri deseni gerçek bir anomali sinyaliyse (örn. altitude_dropout) IF'in o satırları hiç
  göremeyeceği anlamına gelir — S2 katmanı zaten bu durumu ayrı yakalıyor, çakışma yok ama IF'in
  kapsamı LSTM'den DAHA DAR'dır, bu bir sınırlama olarak kayıtlıdır.
- Sentetik veri yalnız DEĞERLENDİRMEDE (anomaly-development rolünde, kalıcı korpus
  `data/objectstore/synthetic/adsb/`) kullanılır — fit'e asla girmez.
- Aynı alarm bütçesi/Pareto ızgarası ve kanal payı çerçevesi (ADR-037) IF'e de uygulanacak;
  IF kendi ayrı eşiğini alacak, `contextual_physics_v1`'in eşiğini paylaşmayacak.

## İlk gate

Provenance/checksum tam, sentetik eğitim sıfır, ve doğal calibration diagnostic üzerinde IF
skorunun ham-genlik taban çizgisiyle (`adsb/diagnostics.py::magnitude_only_score`) Spearman
korelasyonu ölçülüp raporlanmadan (FLAG/PASS iddiası olmadan, yalnız ölçüm) hiçbir "IF daha iyi/
kötü" karşılaştırması yapılmaz.
