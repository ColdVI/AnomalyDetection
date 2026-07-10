# Prompt — Coğrafi Veri Madenciliği (FAZ A): worldmonitor'dan Devşirilen Fikirler

Bunu Claude Code'a yapıştır. Bu, `BIREYSEL_PROJE_MASTER.md`'yi DEĞİŞTİRMİYOR --
ona ek, isteğe bağlı iyileştirmeler öneriyor. Rota sapması tespiti (FAZ B)
kapsam dışı, kullanıcı onu kendisi yapacak.

**Kaynak doğrulaması:** Aşağıdaki bulgular `koala73/worldmonitor` reposunun
gerçek kaynak kodu (zip olarak indirilip incelendi, GitHub sayfası üzerinden
değil) ve gpsjam.org/ilgili kaynaklar üzerinden doğrulandı -- tahmin değil.
AGPL-3.0 lisans notu geçerli: kod kopyalanmıyor, sadece mimari/yöntem
referans alınıyor.

---

## Öncelik sırası (öneri)

1. **GPSJam tarzı NIC-tabanlı jamming haritası** -- en düşük efor, en yüksek
   değer, zaten elindeki veriyle üretilebilir.
2. **Mevsim/dönem bazlı yoğunluk toggle'ı** -- worldmonitor'un `timeRange`
   fikrinin, senin ayrık (sürekli olmayan) veri yapına doğru uyarlanmış hali.
3. **Çok-çözünürlüklü H3** -- araştırma rotandaki "çözünürlük seçimi"
   sorusuna doğrudan cevap.
4. **Tek harita, çoklu katman toggle'ı** -- ayrı dosyalar yerine tek
   interaktif harita.
5. Diğerleri (hassas altyapı yakınlığı, viewport-bazlı yükleme, AIS
   literatür paraleli) -- gelecek iyileştirme notu, şimdi uygulanması
   zorunlu değil.

---

## 1) GPSJam tarzı NIC-tabanlı GPS karışma haritası

**Kaynak:** gpsjam.org -- ADS-B Exchange'in NIC (Navigation Integrity
Category) verisini H3 hex'lere göre agregat ediyor, bir hex'teki uçakların
>%10'u düşük NIC bildiriyorsa "muhtemel jamming" işaretliyor. CC-BY
lisanslı, günlük güncelleniyor.

**Neden düşük efor:** adsb.lol format referansındaki `aircraft_dict`
alanlarında `nic`, `nac_p`, `nac_v`, `sil` zaten var (459K kayıtlık
örneklemde `nic` 114.876 kez görülmüş) -- Silver şemanda muhtemelen bu
kolon zaten duruyor, sadece kullanılmıyor.

**Yeni fonksiyon (`geo.py`'ye ekle):**
```python
def compute_nic_degradation_map(df: pd.DataFrame, nic_threshold: int = 5) -> pd.DataFrame:
    """
    Her H3 hex icin, o hex'teki noktalarin yuzde kaci nic_threshold'un
    ALTINDA (dusuk navigasyon dogrulugu = muhtemel jamming/interference)
    NIC degeri bildirmis, hesaplar.

    Girdi: assign_h3_cell() sonrasi df (h3_cell kolonu dolu), 'nic' kolonu
    (Silver semanda zaten var -- yoksa once oraya eklenmesi gerekir, bu
    fonksiyonun ONCESINDE cozulmesi gereken bir bagimlilik).

    Cikti: h3_cell, total_points, low_nic_count, low_nic_ratio kolonlu
    DataFrame. compute_hex_density'nin ciktisiyla ayni sekle sahip,
    build_density_geojson'a benzer sekilde GeoJSON'a cevrilebilir.

    gpsjam.org'un esigi: hex basina >%10 dusuk NIC = "muhtemel jamming".
    Bu esigi de sabit yazma, CLI argumani yap (--nic-threshold,
    --jamming-ratio-threshold).
    """
```
Bunun çıktısını `build_density_geojson`'a benzer bir `build_jamming_geojson`
fonksiyonuyla GeoJSON'a çevirip, mevcut yoğunluk haritasının **yanına ikinci
bir toggle-edilebilir katman** olarak ekle (bkz. Madde 4).

---

## 2) Mevsim/dönem bazlı yoğunluk toggle'ı (worldmonitor'un `timeRange`'i, uyarlanmış)

**worldmonitor'da gerçekte nasıl çalışıyor (kod incelendi):** `timeRange`
değerleri (`1h, 6h, 24h, 48h, 7d, all`) **sunucudan yeniden veri çekmiyor**
-- zaten yüklenmiş veriyi zaman damgasına göre istemci tarafında filtreliyor
(`applyTimeRangeFilterDebounced`). Bu, SÜREKLİ akan veri için anlamlı bir
"kayan pencere".

**Bizim veri yapımız farklı:** Kullanıcının historical verisi sürekli değil,
**ayrık** (yaklaşık 20 gün arayla, 4 mevsim penceresinden ~11-12 gün,
`download_baseline_days.py`'deki `SEASONAL_WINDOWS`). Kayan pencere yerine
**mevsim/dönem gruplaması** doğru çerçeve:

```python
def compute_hex_density_by_period(
    df: pd.DataFrame,
    period_windows: dict[str, tuple[str, str]],
) -> dict[str, pd.DataFrame]:
    """
    period_windows: {"kis": ("2025-12-01","2026-02-28"), "ilkbahar": (...), ...}
    (download_baseline_days.py'deki SEASONAL_WINDOWS ile ayni format/etiketler
    kullanilmali -- iki yerde farkli isimlendirme olursa karisir.)

    Her donem icin df'i timestamp_utc'ye gore filtrele, compute_hex_density
    cagir. Cikti: {"kis": density_df, "ilkbahar": density_df, ...}
    """
```
`index.html`'de worldmonitor'daki gibi tıklanabilir buton grubu:
```
[Kış] [İlkbahar] [Yaz] [Sonbahar] [Tüm Yıl]
```
Her buton, önceden üretilmiş ayrı bir `.geojson` dosyasını `fetch()` ile
yükleyip mevcut katmanın verisini `source.setData(...)` ile değiştirir --
sayfa yeniden yüklenmez.

**Eğer gerçek "kayan pencere" (son 7 gün gibi) isteniyorsa:** Bunun için
sürekli veri gerekir. Yeni bir toplama mekanizması KURMA -- grup projesinin
zaten çalışan realtime pipeline'ı (`adsblol_realtime`, Gold'da, ~1 hafta
saklanıyor) buna hazır. `load_adsb_gold_data()`'da `source_type == "adsblol_rt"`
satırlarını `timestamp_utc >= now() - N gün` ile filtrelemek yeterli,
ayrı bir veri toplama scripti gerekmiyor.

---

## 3) Çok-çözünürlüklü H3 (tek sabit resolution yerine)

Mevcut `assign_h3_cell(df, resolution)` tek bir sabit resolution alıyor.
Profesyonel hex-yoğunluk haritalarında standart: zoom seviyesine göre
resolution değişir (uzaktan kaba hex, yakından ince hex). H3'ün kendi
hiyerarşik yapısı (`h3.cell_to_parent()`) bunu destekliyor.

```python
def compute_hex_density_multi_resolution(
    df: pd.DataFrame,
    resolutions: list[int] = [4, 5, 6, 7],
) -> dict[int, pd.DataFrame]:
    """
    Her resolution icin ayri assign_h3_cell + compute_hex_density calistirir.
    Cikti: {4: density_df, 5: density_df, ...}. Her biri ayri .geojson'a
    yazilir, index.html haritanin zoom seviyesine gore hangi dosyayi
    yukleyecegini secer (map.on('zoom', ...) ile resolution esiklerini
    kontrol et).
    """
```
Bu, araştırma rotandaki (`BIREYSEL_CALISMA_ROTASI.md` Adım 3) "çözünürlük
seçiminin analiz sonuçlarına etkisi" sorusuna somut, uygulanmış bir cevap
verir -- raporda "tek resolution seçmek yerine çoklu-çözünürlük
karşılaştırıldı" diye yazabilirsin.

---

## 4) Tek harita, çoklu katman toggle'ı (worldmonitor'un `layers=` deseni)

worldmonitor URL'i `layers=conflicts,bases,hotspots,...` gibi virgülle
ayrılmış, aynı haritada aç/kapa edilebilir katmanlar kullanıyor. Şu anki
planımızda `build_density_geojson` ve `build_cluster_geojson` (ve şimdi
`build_jamming_geojson`) ayrı ayrı üretiliyor -- bunları **tek `index.html`**
içinde, checkbox'larla aç/kapa edilebilir katmanlar olarak birleştir:

```
☑ Yoğunluk haritası
☑ Rota kümeleri
☐ GPS karışma (NIC-bazlı)
```
Her checkbox, ilgili MapLibre layer'ının `visibility` özelliğini
`visible`/`none` yapar (mevcut `dashboard.html`'deki
`setClusterLayersVisible` fonksiyonuyla aynı desen).

---

## 5) Gelecek iyileştirme notları (şimdi uygulanması zorunlu değil)

**Hassas altyapı yakınlığı -- ağırlıklı çok-sinyal skorlama örneği:**
worldmonitor'un "hotspot escalation" skorlaması gerçek, kod-doğrulanmış bir
örnek (`docs/hotspots.mdx`):
```
dynamic_score = haber_bileseni × 0.35 + ulke_istikrarsizligi × 0.25
              + cografi_yakinsama × 0.25 + askeri_aktivite × 0.15
final_score = statik_taban × 0.30 + dinamik_skor × 0.70
```
Bu, "birden fazla bağımsız sinyali ağırlıklı birleştirme" desenine somut
bir örnek -- ileride "anomali skoru × hassas-bölge-yakınlığı" gibi bir
çarpan eklemek istersen, bu formül yapısı doğrudan örnek alınabilir. Şu an
için sadece bir referans, uygulanması gerekmiyor (güvenilir, kamuya açık
bir tesis-konumu veri seti araştırılmadı).

**Viewport-bazlı veri yükleme:** Şu anki veri hacminde (H3 agregat edilmiş,
küçük) gerekli değil. Global ölçeğe çıkılırsa (tüm dünya + tüm mevsimler +
tüm resolution'lar birden), bir sonraki darboğaz burası olur -- not
düşülüyor, şimdi uygulanmıyor.

**Deniz taşımacılığı (AIS) paraleli:** Doğrudan uygulanacak bir kod değil,
literatür/yöntem referansı -- rota kümeleme/sapma tespiti yöntemlerinin
sadece havacılığa özgü olmadığını, AIS "dark ship" tespitinde de aynı
tekniklerin kullanıldığını rapor metninde belirtmek için.
