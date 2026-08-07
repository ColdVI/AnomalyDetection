# ADS-B contextual_physics_v2 E?itim Incident'i ? 2026-07-23

## Y?netici ?zeti

`20260723_contextual_physics_v2_train_v1` ko?usu yakla??k 6 saat 36 dakika
hesaplama yapt?ktan sonra, epoch 1 tamamlanmadan bir boyut s?zle?mesi hatas?yla
??kt?. Ko?u bir model, `training_report.json` veya magnitude-domination sonucu
?retmedi. Dolay?s?yla bu dizindeki hi?bir k?smi ??kt? bilimsel sonu?, aday model
ya da Faz D girdisi olarak kullan?lamaz.

Bu olay k???k bir konsol hatas? de?ildir. Uzun e?itim i?inin ilerleme g?nl??? ve
epoch checkpoint'i bulunmad??? i?in saatlerce hesaplaman?n tamam? yeniden
?al??t?r?lmak zorunda kald?. Olay?n g?r?n?rl??? ancak ayr? bir izleme s?reci ve
`INCOMPLETE_DO_NOT_USE.md` i?areti sayesinde sa?land?. Gelecek raporlamalarda bu
ko?u, ba?ar?s?z/yar?m e?itimlerin neden kal?c? provenance, canl?l?k izlemesi ve
recovery checkpoint'i gerektirdi?inin ana ?rne?i olarak ele al?nmal?d?r.

## Etkilenen ve g?venilmeyecek ko?u

- Run: `artifacts/adsb/runs/20260723_contextual_physics_v2_train_v1`
- Durum: **INCOMPLETE ? DO NOT USE**
- Son hata: `ValueError: unexpected contextual forecaster input shape`
- Eksik nihai ??kt?lar: `model_state.pt`, `training_report.json`, tamamlanm??
  checksum zinciri
- De?erlendirilemeyen zorunlu kap?:
  `magnitude_domination_flagged_at_0_8`
- Sonu?: Faz D?G ba?lat?lmad?; hi?bir e?ik, b?t?e ?zgaras? veya ?n-kay?t de?eri
  bu olay nedeniyle de?i?tirilmedi.

Ba?ar?s?z dizin silinmeyecek veya yeni ko?u taraf?ndan ?zerine yaz?lmayacakt?r.
Bu dizin ve izleme kay?tlar? incident delilidir.

## Zaman ?izelgesi (Europe/Istanbul)

- 12:43:38 ? e?itim s?reci ba?lad?.
- 12:44:07 ? `run_manifest.json` yaz?ld?; ba?lang?? Git/kod s?zle?mesi kilitlendi.
- 13:02:48 ? do?al fit verisinden scaler ??kt?s? yaz?ld?.
- 13:03:21 ? ilk optimizer ad?m?ndan ?nce t?retilmi? e?itim konfig?rasyonu yaz?ld?.
- 19:17:08 ? harici monitor s?reci e?itimi son kez canl? g?zledi.
- 19:19:49 ? `INCOMPLETE_DO_NOT_USE.md` hata i?areti yaz?ld?.
- 19:22:08 ? monitor PID'nin art?k bulunmad???n? ve rapor olu?mad???n? do?rulad?.

Yakla??k 6 saat 36 dakikal?k duvar saati harcand?. Epoch 1 sat?r? ve epoch
checkpoint'i olu?mad??? i?in geri kazan?labilir bir e?itim s?n?r? yoktu.

## K?k neden

E?itim d?ng?s? NumPy tens?rlerini, `torch.randperm` ile ?retilen PyTorch indeks
tens?r?yle do?rudan indeksliyordu. Son minibatch tam olarak bir pencere
i?erdi?inde tek elemanl? PyTorch tens?r? NumPy taraf?ndan skaler indeks gibi
yorumland? ve ba?taki batch ekseni sessizce d??t?:

```text
beklenen X ?ekli : (1, history_rows, input_features)
ger?ek X ?ekli   : (history_rows, input_features)
```

Forecaster hakl? olarak ?? boyutlu giri? s?zle?mesini reddetti. Hata yaln?zca
bir part'?n pencere say?s?n?n batch boyutuna g?re kalan?n?n `1` oldu?u s?n?r
durumunda ortaya ??kt??? i?in e?itim saatlerce normal ilerleyebildi.

Bu olay:

- veri ?l?e?i veya magnitude-domination problemi de?ildir;
- model mimarisi yetersizli?i de?ildir;
- e?ik, alarm b?t?esi, ?n-kay?t veya truth-v2 sonucu de?ildir;
- veri g?nlerinin yanl?? ayr??mas? de?ildir;
- saf bir NumPy/PyTorch s?n?r?nda batch ekseni koruma hatas?d?r.

## D?zeltme ve bilimsel b?t?nl?k

?ndeks dilimi a??k?a bir boyutlu NumPy indeks dizisine ?evrildi. B?ylece tek
sat?rl? son batch `(1, history, features)` ?eklini koruyor. D?zeltme:

- perm?tasyon s?ras?n? de?i?tirmez;
- yeni random say? t?ketmez ve seed davran???n? de?i?tirmez;
- veri, ?zellik, pencere, model veya loss tan?m?n? de?i?tirmez;
- dondurulmu? e?iklere, ?zgaralara ve epoch say?s?na dokunmaz.

Ayr?ca her tamamlanan epoch sonunda model, optimizer, PyTorch RNG durumu ve
e?itim ge?mi?i `training_epoch_checkpoint.pt` dosyas?na atomik olarak yaz?l?r.
Bu dosya yaln?z incident recovery i?indir; nihai `model_state.pt` veya tamamlanm??
`training_report.json` yerine ge?emez.

## Regresyon kapsam?

Kal?c? testler ?u s?zle?meleri do?rular:

1. Tek sat?rl? remainder batch eksenini korur.
2. ?ok sat?rl? batch orijinal `randperm` s?ras?n? korur.
3. Epoch checkpoint atomik yaz?l?r; tamamlanan epoch say?s?, model, optimizer ve
   ge?mi?i ta??r; ge?ici dosya b?rakmaz.

Contextual model/windowing/persistence test ailesi de yeniden ?al??t?r?larak
d?zeltmenin kom?u s?zle?meleri bozmad??? do?rulanmal?d?r.

## Yeniden ba?latma protokol?

1. Ba?ar?s?z `v1` dizinini de?i?tirme veya yeniden kullanma.
2. D?zeltme, test ve bu incident kayd?n? tek kapsaml? commit ile sabitle.
3. Tracked Git a?ac?n?n temiz oldu?unu do?rula.
4. Yeni ve bo? bir run diziniyle `v2` ko?usunu ba?lat.
5. stdout/stderr'i run dizini d???nda kal?c? dosyalara y?nlendir.
6. Ayr? read-only monitor ile PID, kaynak kullan?m? ve artifact olu?umunu izle.
7. Her epoch sonunda recovery checkpoint'in olu?tu?unu do?rula.
8. Yaln?z `training_report.json` ?retildikten sonra
   `magnitude_domination_flagged_at_0_8` kap?s?n? de?erlendir.

Magnitude flag `true` olursa kullan?c? talimat?na g?re durulur. `false` olursa
Faz D'ye ge?ilir. Ba?ar?s?z `v1` ko?usundan herhangi bir metrik t?retilmez.

## Gelecek raporlamas? i?in ?nerilen ifade

> ?lk contextual_physics_v2 e?itim giri?imi, tek ?rnekli son minibatch'te
> NumPy/PyTorch indeksleme semanti?inin batch eksenini d???rmesi nedeniyle epoch
> 1 tamamlanmadan ??kt?. Ko?u yakla??k 6.6 saat hesaplama t?ketti fakat model veya
> de?erlendirme raporu ?retmedi. Sonu? olarak deneysel sonu? say?lmad?; e?ik ve
> ?n-kay?t dondurmas? korunarak kod d?zeltildi, regresyon testleri ve atomik epoch
> recovery checkpoint'i eklendi ve yeni kimlikli ko?u ba?lat?ld?.

## Kan?t konumlar?

- Hata i?areti:
  `artifacts/adsb/runs/20260723_contextual_physics_v2_train_v1/INCOMPLETE_DO_NOT_USE.md`
- Monitor zaman serisi:
  `artifacts/adsb/monitoring/20260723_contextual_v2_training_monitor.jsonl`
- Monitor ?zeti:
  `artifacts/adsb/monitoring/20260723_contextual_v2_training_monitor_summary.json`
- ?lk k?sa forensics notu:
  `artifacts/adsb/monitoring/20260723_contextual_v2_training_failure_report.md`
